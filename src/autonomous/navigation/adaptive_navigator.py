"""Adaptive autonomous vehicle navigator integrating vision pipelines and fuzzy control."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from ..control.vehicle_controller import VehicleController
from ..core.neuro_fuzzy import NeuroFuzzyLogic
from ..perception.depth_estimator import DepthEstimator
from ..perception.lane_detector import LaneDetector
from ..perception.obstacle_detector import ObstacleDetector
from ..perception.scene_understanding import SceneUnderstanding
from ..tracking.object_tracker import ObjectTracker
from ..config.settings import OBSTACLE_CLASSES


@dataclass
class NavigationResult:
    frame: np.ndarray
    telemetry: dict[str, object]


class AdaptiveNavigator:
    def __init__(self):
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.depth_estimator = DepthEstimator()
        self.tracker = ObjectTracker()
        self.scene_understanding = SceneUnderstanding()
        self.decision_engine = NeuroFuzzyLogic()
        self.vehicle_controller = VehicleController()
        self._latest_telemetry: dict[str, object] = {
            **self.vehicle_controller.get_status(),
            "lane_deviation": 0.0,
            "obstacle_distance": 100.0,
            "traffic_density": 0.0,
            "lighting_score": 100.0,
            "obstacle_bearing": 0.0,
            "closest_obstacle": None,
            "obstacle_uncertainty": 0.0,
            "steering_angle": 0.0,
            "speed_control": 0.0,
            "decision_confidence": 0.0,
            "top_rules": [],
            "active_rule": "Awaiting input",
            "vehicle_status": self.vehicle_controller.get_status(),
            "detections": 0,
            "raw_detections": 0,
            "tracks": [],
            "depth_backend": "none",
            "depth_confidence": 0.0,
            "scene": {
                "lane_quality": 0.0,
                "road_occupancy": 0.0,
                "occlusion_ratio": 0.0,
                "visibility_score": 0.0,
                "uncertainty_score": 0.0,
                "dynamic_pressure": 0.0,
                "risk_score": 0.0,
                "driving_mode": "normal",
                "recommendation": "Awaiting input...",
                "factors": [],
            },
            "pipeline_stages": {
                "obstacle_detection": False,
                "lane_detection": False,
                "vehicle_detection": False,
                "depth_estimation": False,
                "traffic_analysis": False,
                "context_understanding": False,
                "anfis_rules": False,
                "neuro_fuzzy_engine": False,
            },
        }
        self._last_result: NavigationResult | None = None

    def process_frame(self, frame: np.ndarray, draw_hud: bool = True) -> NavigationResult:
        left_line, right_line, lane_deviation = self.lane_detector.detect_lanes(frame)
        lane_geometry = self.depth_estimator.estimate_lane_geometry(left_line, right_line, frame.shape)
        drivable_polygon = self._build_drivable_polygon(frame.shape, left_line, right_line)

        detections, _ = self.obstacle_detector.detect(frame)
        display_detections = [
            det for det in detections
            if det.label in OBSTACLE_CLASSES
            and (det.bbox[2] - det.bbox[0]) > 20
            and (det.bbox[3] - det.bbox[1]) > 20
        ]
        lighting_score = self.obstacle_detector.estimate_lighting(frame)
        traffic_density = self.obstacle_detector.estimate_traffic_density(display_detections, frame.shape)
        tracks = self.tracker.update(display_detections, frame.shape)
        depth_frame = self.depth_estimator.estimate_depth_frame(frame)

        scene_profile = self.depth_estimator.build_scene_profile(
            display_detections,
            frame.shape,
            frame=frame,
            lane_geometry=lane_geometry,
            depth_frame=depth_frame,
        )
        observations = scene_profile["observations"]
        closest = scene_profile["closest"]

        obstacle_distance = 100.0
        obstacle_bearing = 0.0
        closest_label = None
        obstacle_uncertainty = 0.0
        if closest is not None:
            obstacle_distance = float(closest.relative_distance)
            obstacle_bearing = float(closest.bearing)
            closest_label = closest.label
            obstacle_uncertainty = float(closest.uncertainty)

        scene = self.scene_understanding.analyze(
            frame=frame,
            lane_left=left_line,
            lane_right=right_line,
            lane_deviation=lane_deviation,
            detections=display_detections,
            tracks=tracks,
            depth_frame=depth_frame,
            depth_observations=observations,
            lighting_score=lighting_score,
            traffic_density=traffic_density,
            obstacle_bearing=obstacle_bearing,
            obstacle_distance=obstacle_distance,
            obstacle_uncertainty=obstacle_uncertainty,
        )

        decision = self.decision_engine.compute(
            lane_deviation=lane_deviation,
            obstacle_distance=obstacle_distance,
            traffic_density=traffic_density,
            lighting_score=lighting_score,
            obstacle_bearing=obstacle_bearing,
            lane_quality=scene.lane_quality,
            obstacle_uncertainty=obstacle_uncertainty,
            road_occupancy=scene.road_occupancy,
            dynamic_obstacles=sum(1 for track in tracks if track.is_dynamic),
            scene_risk=scene.risk_score,
        )

        adjusted_speed = float(decision["speed_control"]) * max(0.0, 1.0 - scene.risk_score / 140.0)
        adjusted_speed = float(np.clip(adjusted_speed, 0.0, 100.0))

        if obstacle_distance <= 10 or scene.risk_score >= 80 or scene.driving_mode == "emergency":
            self.vehicle_controller.emergency_brake()
        else:
            self.vehicle_controller.apply_control(
                steering_angle=decision["steering_angle"],
                speed=adjusted_speed,
            )

        annotated = self.lane_detector.draw_lanes(frame, left_line, right_line)
        annotated = self._draw_tracks(annotated, tracks)

        if draw_hud:
            annotated = self._draw_hud(
                annotated,
                lane_deviation=lane_deviation,
                obstacle_distance=obstacle_distance,
                traffic_density=traffic_density,
                lighting_score=lighting_score,
                obstacle_bearing=obstacle_bearing,
                closest_label=closest_label,
                obstacle_uncertainty=obstacle_uncertainty,
                active_rule=decision["active_rule"],
                scene=scene,
            )

        telemetry = self._build_telemetry(
            lane_deviation=lane_deviation,
            obstacle_distance=obstacle_distance,
            traffic_density=traffic_density,
            lighting_score=lighting_score,
            obstacle_bearing=obstacle_bearing,
            closest_label=closest_label,
            obstacle_uncertainty=obstacle_uncertainty,
            decision=decision,
            observations=observations,
            scene=scene,
            tracks=tracks,
            depth_frame=depth_frame,
            raw_detections=len(detections),
        )
        self._latest_telemetry = telemetry
        result = NavigationResult(frame=annotated, telemetry=telemetry)
        self._last_result = result
        return result

    def process_frame_obj1(self, frame: np.ndarray) -> NavigationResult:
        detections, _ = self.obstacle_detector.detect(frame)
        display_detections = [
            det for det in detections
            if det.label in OBSTACLE_CLASSES
            and (det.bbox[2] - det.bbox[0]) > 20
            and (det.bbox[3] - det.bbox[1]) > 20
        ]
        lighting_score = self.obstacle_detector.estimate_lighting(frame)
        traffic_density = self.obstacle_detector.estimate_traffic_density(display_detections, frame.shape)
        tracks = self.tracker.update(display_detections, frame.shape)
        depth_frame = self.depth_estimator.estimate_depth_frame(frame)

        obstacle_distance = 100.0
        obstacle_bearing = 0.0
        closest_label = None
        obstacle_uncertainty = 0.0
        if display_detections:
            closest_det = min(display_detections, key=lambda d: d.area)
            obstacle_distance = float(np.clip(closest_det.area / 500.0, 5.0, 100.0))
            cx = closest_det.center[0]
            obstacle_bearing = float(np.clip((cx / max(frame.shape[1], 1) - 0.5) * 60, -30, 30))
            closest_label = closest_det.label
            obstacle_uncertainty = float(1.0 - closest_det.confidence)

        scene = self.scene_understanding.analyze(
            frame=frame,
            lane_left=None,
            lane_right=None,
            lane_deviation=0.0,
            detections=display_detections,
            tracks=tracks,
            depth_frame=depth_frame,
            depth_observations=[],
            lighting_score=lighting_score,
            traffic_density=traffic_density,
            obstacle_bearing=obstacle_bearing,
            obstacle_distance=obstacle_distance,
            obstacle_uncertainty=obstacle_uncertainty,
        )

        decision = self.decision_engine.compute(
            lane_deviation=0.0,
            obstacle_distance=obstacle_distance,
            traffic_density=traffic_density,
            lighting_score=lighting_score,
            obstacle_bearing=obstacle_bearing,
            lane_quality=scene.lane_quality,
            obstacle_uncertainty=obstacle_uncertainty,
            road_occupancy=scene.road_occupancy,
            dynamic_obstacles=sum(1 for track in tracks if track.is_dynamic),
            scene_risk=scene.risk_score,
        )

        adjusted_speed = float(decision["speed_control"]) * max(0.0, 1.0 - scene.risk_score / 140.0)
        adjusted_speed = float(np.clip(adjusted_speed, 0.0, 100.0))

        if obstacle_distance <= 10 or scene.risk_score >= 80 or scene.driving_mode == "emergency":
            self.vehicle_controller.emergency_brake()
        else:
            self.vehicle_controller.apply_control(
                steering_angle=decision["steering_angle"],
                speed=adjusted_speed,
            )

        annotated = frame.copy()
        annotated = self.obstacle_detector.draw_detections(annotated, display_detections)
        annotated = self._draw_tracks(annotated, tracks)

        telemetry = self._build_telemetry(
            lane_deviation=0.0,
            obstacle_distance=obstacle_distance,
            traffic_density=traffic_density,
            lighting_score=lighting_score,
            obstacle_bearing=obstacle_bearing,
            closest_label=closest_label,
            obstacle_uncertainty=obstacle_uncertainty,
            decision=decision,
            observations=[],
            scene=scene,
            tracks=tracks,
            depth_frame=depth_frame,
            raw_detections=len(detections),
        )
        self._latest_telemetry = telemetry
        result = NavigationResult(frame=annotated, telemetry=telemetry)
        self._last_result = result
        return result

    def get_telemetry(self) -> dict[str, object]:
        return dict(self._latest_telemetry)

    def _build_telemetry(
        self,
        lane_deviation: float,
        obstacle_distance: float,
        traffic_density: float,
        lighting_score: float,
        obstacle_bearing: float,
        closest_label: str | None,
        obstacle_uncertainty: float,
        decision: dict[str, object],
        observations,
        scene,
        tracks,
        depth_frame,
        raw_detections: int,
    ) -> dict[str, object]:
        status = self.vehicle_controller.get_status()
        return {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "lane_deviation": round(float(lane_deviation), 2),
            "obstacle_distance": round(float(obstacle_distance), 2),
            "traffic_density": round(float(traffic_density), 2),
            "lighting_score": round(float(lighting_score), 2),
            "obstacle_bearing": round(float(obstacle_bearing), 2),
            "closest_obstacle": closest_label,
            "obstacle_uncertainty": round(float(obstacle_uncertainty), 2),
            "steering_angle": round(float(decision["steering_angle"]), 2),
            "speed_control": round(float(status["speed"]), 2),
            "decision_confidence": round(float(decision.get("decision_confidence", 0.0)), 2),
            "top_rules": decision.get("top_rules", []),
            "active_rule": decision["active_rule"],
            "vehicle_status": status,
            "detections": len(observations),
            "raw_detections": raw_detections,
            "tracks": [track.to_dict() for track in tracks],
            "depth_backend": depth_frame.backend,
            "depth_confidence": round(float(depth_frame.confidence), 2),
            "scene": {
                "lane_quality": scene.lane_quality,
                "road_occupancy": scene.road_occupancy,
                "occlusion_ratio": scene.occlusion_ratio,
                "visibility_score": scene.visibility_score,
                "uncertainty_score": scene.uncertainty_score,
                "dynamic_pressure": scene.dynamic_pressure,
                "risk_score": scene.risk_score,
                "driving_mode": scene.driving_mode,
                "recommendation": scene.recommendation,
                "factors": scene.factors,
            },
            "pipeline_stages": {
                "obstacle_detection": True,
                "lane_detection": True,
                "vehicle_detection": True,
                "depth_estimation": True,
                "traffic_analysis": True,
                "context_understanding": True,
                "anfis_rules": True,
                "neuro_fuzzy_engine": True,
            },
        }

    def _draw_hud(
        self,
        frame: np.ndarray,
        lane_deviation: float,
        obstacle_distance: float,
        traffic_density: float,
        lighting_score: float,
        obstacle_bearing: float,
        closest_label: str | None,
        obstacle_uncertainty: float,
        active_rule: str,
        scene,
    ) -> np.ndarray:
        return frame

    def _draw_tracks(self, frame: np.ndarray, tracks) -> np.ndarray:
        """Draw clean pill-badge labels for each tracked vehicle.

        Features:
        - Single badge per vehicle (no separate detection + tracking overlay).
        - Dark semi-transparent pill background with colour-coded border.
        - White text: ``ID {id} | {LABEL}``.
        - White corner accent marks on each bounding box corner.
        - Anti-collision: badges are stacked vertically when they overlap.
        """
        overlay = frame.copy()
        placed: list[tuple[int, int, int, int]] = []  # (bx1, by1, bx2, by2) of drawn badges

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thickness = 1
        pad_x, pad_y = 6, 4

        # Process front-to-back (largest y2 first) so foreground labels are placed first
        for track in sorted(tracks, key=lambda t: t.bbox[3], reverse=True):
            x1, y1, x2, y2 = map(int, track.bbox)
            # Green for static, amber-blue for dynamic
            color = (0, 230, 118) if not track.is_dynamic else (0, 140, 255)

            # ── Bounding box ───────────────────────────────────────────────────
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # ── Corner accent ticks ────────────────────────────────────────────
            d = max(6, min(14, (x2 - x1) // 4, (y2 - y1) // 4))
            for px, py, dx, dy in [
                (x1, y1, d, 0), (x1, y1, 0, d),
                (x2, y1, -d, 0), (x2, y1, 0, d),
                (x1, y2, d, 0), (x1, y2, 0, -d),
                (x2, y2, -d, 0), (x2, y2, 0, -d),
            ]:
                cv2.line(overlay, (px, py), (px + dx, py + dy), (255, 255, 255), 2, cv2.LINE_AA)

            # ── Badge geometry ─────────────────────────────────────────────────
            label_text = f"ID {track.track_id} | {track.label.upper()}"
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            bw = tw + pad_x * 2
            bh = th + pad_y * 2

            # Default: above the top edge of the bbox
            bx1 = x1
            by1 = y1 - bh - 5
            if by1 < 2:
                by1 = y1 + 5  # Flip below if too close to top
            bx2 = bx1 + bw
            by2 = by1 + bh

            # ── Anti-collision: shift badge vertically until no overlap ────────
            for _ in range(8):
                collision = any(
                    not (bx2 < px1 or bx1 > px2 or by2 < py1 or by1 > py2)
                    for px1, py1, px2, py2 in placed
                )
                if not collision:
                    break
                by1 -= bh + 3
                by2 = by1 + bh
                if by1 < 2:
                    # Ran out of room above — stack below instead
                    by1 = y2 + 5 + (bh + 3) * len(placed)
                    by2 = by1 + bh
                    break

            placed.append((bx1, by1, bx2, by2))

            # ── Draw pill badge ────────────────────────────────────────────────
            # Dark translucent background
            badge_bg = overlay.copy()
            cv2.rectangle(badge_bg, (bx1, by1), (bx2, by2), (12, 16, 22), -1)
            overlay = cv2.addWeighted(badge_bg, 0.88, overlay, 0.12, 0)
            # Coloured 1-px border
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
            # White label text
            cv2.putText(
                overlay, label_text,
                (bx1 + pad_x, by1 + pad_y + th),
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
            )

        return overlay

    def _build_drivable_polygon(self, frame_shape, left_line, right_line):
        height, width = frame_shape[:2]
        if left_line is not None and right_line is not None:
            left_pts = left_line[:, 0, :]
            right_pts = right_line[:, 0, :]
            points = np.vstack((left_pts, right_pts[::-1]))
        else:
            points = np.array(
                [
                    (int(width * 0.18), height - 1),
                    (int(width * 0.82), height - 1),
                    (int(width * 0.62), int(height * 0.46)),
                    (int(width * 0.38), int(height * 0.46)),
                ],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
        return points

    def _is_drivable_obstacle(self, detection, polygon) -> bool:
        if detection.label not in OBSTACLE_CLASSES:
            return False

        x1, y1, x2, y2 = detection.bbox
        sample_x = int((x1 + x2) / 2.0)
        sample_y = int(y2)
        if sample_y <= 0:
            return False

        return cv2.pointPolygonTest(polygon, (float(sample_x), float(sample_y)), False) >= 0
