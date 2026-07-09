from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from ..config.settings import SCENE_CRITICAL_RISK, SCENE_WARNING_RISK


@dataclass
class SceneSummary:
    lane_quality: float
    road_occupancy: float
    occlusion_ratio: float
    visibility_score: float
    uncertainty_score: float
    dynamic_pressure: float
    risk_score: float
    driving_mode: str
    recommendation: str
    factors: dict[str, float] = field(default_factory=dict)


class SceneUnderstanding:
    def analyze(
        self,
        frame: np.ndarray,
        lane_left,
        lane_right,
        lane_deviation: float,
        detections: Iterable,
        tracks: Iterable,
        depth_frame,
        depth_observations,
        lighting_score: float,
        traffic_density: float,
        obstacle_bearing: float,
        obstacle_distance: float,
        obstacle_uncertainty: float,
    ) -> SceneSummary:
        height, width = frame.shape[:2]
        road_mask = self._build_road_mask(frame.shape, lane_left, lane_right)
        road_area = float(np.count_nonzero(road_mask)) + 1e-6

        detections = list(detections)
        tracks = list(tracks)
        depth_quality = float(getattr(depth_frame, "confidence", 0.25) or 0.25)
        depth_observation_count = len(depth_observations) if depth_observations is not None else 0
        occluded_area = 0.0
        occupied_area = 0.0

        for detection in detections:
            detection_mask = self._detection_mask(detection, frame.shape)
            occupied_area += float(np.count_nonzero(detection_mask))
            occluded_area += float(np.count_nonzero(cv2.bitwise_and(detection_mask, road_mask)))

        road_occupancy = np.clip(occupied_area / (height * width + 1e-6) * 100.0, 0, 100)
        occlusion_ratio = np.clip(occluded_area / road_area * 100.0, 0, 100)

        lane_presence = 0.0
        if lane_left is not None:
            lane_presence += 50.0
        if lane_right is not None:
            lane_presence += 50.0
        lane_deviation_penalty = min(40.0, abs(float(lane_deviation)) * 0.35)
        lane_quality = np.clip(lane_presence - lane_deviation_penalty - occlusion_ratio * 0.25, 0, 100)

        dynamic_tracks = [track for track in tracks if getattr(track, "is_dynamic", False)]
        dynamic_pressure = np.clip(len(dynamic_tracks) * 12.0 + sum(getattr(track, "speed_px", 0.0) for track in dynamic_tracks) * 0.15, 0, 100)

        visibility_score = np.clip(
            0.35 * self._normalize_lighting(lighting_score)
            + 0.35 * lane_quality
            + 0.15 * (100.0 - occlusion_ratio)
            + 0.15 * (100.0 - obstacle_uncertainty),
            0,
            100,
        )
        visibility_score = np.clip(visibility_score + depth_quality * 10.0, 0, 100)

        distance_penalty = np.clip(100.0 - float(obstacle_distance), 0, 100)
        bearing_penalty = min(25.0, abs(float(obstacle_bearing)) * 0.55)
        uncertainty_penalty = min(30.0, float(obstacle_uncertainty) * 0.55)
        traffic_penalty = min(20.0, float(traffic_density) * 1.4)
        visibility_penalty = 100.0 - visibility_score
        occupancy_penalty = min(25.0, road_occupancy * 0.22)
        dynamic_penalty = min(25.0, dynamic_pressure * 0.25)
        depth_penalty = min(15.0, (1.0 - depth_quality) * 15.0 + max(0, depth_observation_count - 1) * 0.5)

        risk_score = np.clip(
            0.28 * distance_penalty
            + 0.12 * bearing_penalty
            + 0.16 * uncertainty_penalty
            + 0.14 * traffic_penalty
            + 0.12 * visibility_penalty
            + 0.10 * occupancy_penalty
            + 0.08 * dynamic_penalty
            + 0.06 * depth_penalty,
            0,
            100,
        )

        driving_mode = "normal"
        if risk_score >= SCENE_CRITICAL_RISK or obstacle_distance <= 10:
            driving_mode = "emergency"
        elif risk_score >= SCENE_WARNING_RISK or lane_quality < 45 or visibility_score < 40:
            driving_mode = "cautious"
        elif road_occupancy > 28 or occlusion_ratio > 22:
            driving_mode = "degraded"

        recommendation = self._recommendation(driving_mode, obstacle_bearing, obstacle_distance, lane_deviation)
        factors = {
            "distance_penalty": round(float(distance_penalty), 2),
            "bearing_penalty": round(float(bearing_penalty), 2),
            "uncertainty_penalty": round(float(uncertainty_penalty), 2),
            "traffic_penalty": round(float(traffic_penalty), 2),
            "visibility_penalty": round(float(visibility_penalty), 2),
            "occupancy_penalty": round(float(occupancy_penalty), 2),
            "dynamic_penalty": round(float(dynamic_penalty), 2),
            "depth_penalty": round(float(depth_penalty), 2),
        }

        return SceneSummary(
            lane_quality=float(round(lane_quality, 2)),
            road_occupancy=float(round(road_occupancy, 2)),
            occlusion_ratio=float(round(occlusion_ratio, 2)),
            visibility_score=float(round(visibility_score, 2)),
            uncertainty_score=float(round(obstacle_uncertainty, 2)),
            dynamic_pressure=float(round(dynamic_pressure, 2)),
            risk_score=float(round(risk_score, 2)),
            driving_mode=driving_mode,
            recommendation=recommendation,
            factors=factors,
        )

    def _normalize_lighting(self, lighting_score: float) -> float:
        return float(np.clip((lighting_score / 255.0) * 100.0, 0, 100))

    def _build_road_mask(self, frame_shape: tuple[int, int, int], lane_left, lane_right) -> np.ndarray:
        height, width = frame_shape[:2]
        road_mask = np.zeros((height, width), dtype=np.uint8)
        if lane_left is not None and lane_right is not None:
            left_pts = lane_left[:, 0, :]
            right_pts = lane_right[:, 0, :]
            polygon = np.vstack((left_pts, right_pts[::-1]))
        else:
            polygon = np.array(
                [
                    (int(width * 0.12), height - 1),
                    (int(width * 0.88), height - 1),
                    (int(width * 0.62), int(height * 0.45)),
                    (int(width * 0.38), int(height * 0.45)),
                ],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
        cv2.fillPoly(road_mask, [polygon], 255)
        return road_mask

    def _detection_mask(self, detection, frame_shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if getattr(detection, "mask", None) is not None and len(detection.mask) >= 3:
            polygon = np.asarray(detection.mask, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], 255)
            return mask

        x1, y1, x2, y2 = map(int, detection.bbox)
        mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 255
        return mask

    def _recommendation(self, driving_mode: str, obstacle_bearing: float, obstacle_distance: float, lane_deviation: float) -> str:
        if driving_mode == "emergency":
            return "Immediate brake and hold steering"
        if obstacle_distance < 25:
            if obstacle_bearing < 0:
                return "Prioritize left evasive path"
            if obstacle_bearing > 0:
                return "Prioritize right evasive path"
            return "Slow down and remain centered"
        if abs(lane_deviation) > 35:
            return "Re-center lane tracking"
        return "Maintain lane and monitor surroundings"
