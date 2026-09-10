"""Monocular depth estimation pipeline using MiDaS or classical geometric heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config.settings import DEPTH_MODEL_NAME, DEPTH_MODEL_PREFER_HUB, LANE_ROI_TOP

try:
    import torch
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None


@dataclass
class DepthObservation:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    relative_distance: float
    uncertainty: float
    bearing: float
    vertical_position: float
    size_score: float
    depth_score: float


@dataclass
class DepthFrame:
    depth_map: np.ndarray | None
    confidence: float
    backend: str


class DepthEstimator:
    def __init__(self, prefer_hub: bool = DEPTH_MODEL_PREFER_HUB, model_name: str = DEPTH_MODEL_NAME):
        self.prefer_hub = prefer_hub
        self.model_name = model_name
        self._device = None
        self._model = None
        self._transform = None
        self._backend = "geometric"
        self._orientation_sign: float | None = None
        self._backend_loaded = False

    def _load_backend(self) -> None:
        if self._backend_loaded:
            return
        self._backend_loaded = True
        if torch is None or not self.prefer_hub:
            return

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_name,
                pretrained=True,
                trust_repo=True,
            )
            self._model.to(self._device)
            self._model.eval()

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = getattr(transforms, "small_transform", None) or getattr(transforms, "default_transform", None)
            self._backend = f"midas:{self.model_name}"
        except Exception:
            self._model = None
            self._transform = None
            self._backend = "geometric"

    def estimate_lane_geometry(self, left_line, right_line, frame_shape: tuple[int, int, int]) -> dict[str, float | int | None]:
        height, width = frame_shape[:2]
        center_x = width / 2.0
        lane_center = center_x
        lane_width = None

        if left_line is not None and right_line is not None:
            left_x = float(left_line[0][0][0])
            right_x = float(right_line[0][0][0])
            lane_center = (left_x + right_x) / 2.0
            lane_width = abs(right_x - left_x)
        elif left_line is not None:
            lane_center = float(left_line[0][0][0]) + width / 4.0
        elif right_line is not None:
            lane_center = float(right_line[0][0][0]) - width / 4.0

        horizon_y = int(height * LANE_ROI_TOP)
        return {
            "center_x": center_x,
            "lane_center": lane_center,
            "lane_offset_px": lane_center - center_x,
            "lane_offset_pct": ((lane_center - center_x) / (width / 2.0)) * 100.0,
            "lane_width_px": lane_width,
            "horizon_y": horizon_y,
        }

    def estimate_depth_frame(self, frame: np.ndarray) -> DepthFrame:
        if not self._backend_loaded:
            self._load_backend()
        if self._model is None or self._transform is None or torch is None:
            return DepthFrame(depth_map=None, confidence=0.25, backend=self._backend)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(rgb).to(self._device)
            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.detach().cpu().numpy().astype(np.float32)
            confidence = self._estimate_depth_confidence(depth_map)
            return DepthFrame(depth_map=depth_map, confidence=confidence, backend=self._backend)
        except Exception:
            return DepthFrame(depth_map=None, confidence=0.25, backend="geometric")

    def _estimate_depth_confidence(self, depth_map: np.ndarray) -> float:
        normalized = self._robust_normalize(depth_map)
        top_band = normalized[: max(1, normalized.shape[0] // 4), :]
        bottom_band = normalized[max(1, normalized.shape[0] * 3 // 4) :, :]
        contrast = float(abs(np.median(bottom_band) - np.median(top_band)))
        spread = float(np.clip(np.std(normalized), 1e-3, 1.0))
        confidence = 0.5 + 0.35 * np.clip(contrast, 0.0, 1.0) + 0.15 * (1.0 - spread)
        return float(np.clip(confidence, 0.25, 0.98))

    def _robust_normalize(self, values: np.ndarray) -> np.ndarray:
        low = float(np.nanpercentile(values, 5))
        high = float(np.nanpercentile(values, 95))
        if abs(high - low) < 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _infer_orientation(self, depth_map: np.ndarray) -> float:
        if self._orientation_sign is not None:
            return self._orientation_sign

        normalized = self._robust_normalize(depth_map)
        top_band = float(np.median(normalized[: max(1, normalized.shape[0] // 4), :]))
        bottom_band = float(np.median(normalized[max(1, normalized.shape[0] * 3 // 4) :, :]))
        self._orientation_sign = 1.0 if bottom_band >= top_band else -1.0
        return self._orientation_sign

    def estimate_depth(self, detection, frame_shape: tuple[int, int, int], depth_frame: DepthFrame | None = None, lane_geometry: dict[str, float | int | None] | None = None) -> DepthObservation:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = detection.bbox
        bbox_width = max(1.0, x2 - x1)
        bbox_height = max(1.0, y2 - y1)
        bbox_area = bbox_width * bbox_height
        frame_area = float(height * width)
        bottom_y = max(y1, y2)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        depth_score = 0.5
        depth_confidence = 0.25
        if depth_frame is not None and depth_frame.depth_map is not None:
            normalized = self._robust_normalize(depth_frame.depth_map)
            orientation = self._infer_orientation(depth_frame.depth_map)
            closeness_map = normalized if orientation > 0 else 1.0 - normalized
            x1i = max(0, int(np.floor(x1)))
            y1i = max(0, int(np.floor(y1)))
            x2i = min(width, int(np.ceil(x2)))
            y2i = min(height, int(np.ceil(y2)))
            if x2i > x1i and y2i > y1i:
                depth_score = float(np.nanmedian(closeness_map[y1i:y2i, x1i:x2i]))
            else:
                depth_score = float(np.nanmedian(closeness_map))
            depth_confidence = depth_frame.confidence

        vertical_score = np.clip((1.0 - bottom_y / height) * 100.0, 0, 100)
        size_score = np.clip((1.0 - np.sqrt(bbox_area / (frame_area + 1e-6))) * 100.0, 0, 100)
        lane_score = vertical_score
        if lane_geometry and lane_geometry.get("lane_width_px"):
            lane_width = float(lane_geometry["lane_width_px"] or 0.0)
            lane_ratio = np.clip(lane_width / width, 0, 1)
            lane_score = np.clip((1.0 - lane_ratio) * 100.0, 0, 100)

        relative_distance = float(
            0.42 * (100.0 - 100.0 * depth_score)
            + 0.28 * vertical_score
            + 0.20 * size_score
            + 0.10 * lane_score
        )
        relative_distance = float(np.clip(relative_distance, 0, 100))
        uncertainty = float(
            np.clip(
                (1.0 - detection.confidence) * 35.0
                + (1.0 - depth_confidence) * 30.0
                + (1.0 - min(1.0, bbox_area / frame_area)) * 20.0,
                0,
                100,
            )
        )

        lane_center = float(lane_geometry["lane_center"]) if lane_geometry else width / 2.0
        bearing = float(np.clip(((center_x - lane_center) / (width / 2.0)) * 45.0, -45.0, 45.0))

        return DepthObservation(
            label=detection.label,
            confidence=detection.confidence,
            bbox=(x1, y1, x2, y2),
            relative_distance=relative_distance,
            uncertainty=uncertainty,
            bearing=bearing,
            vertical_position=float(center_y / height),
            size_score=float(size_score),
            depth_score=float(depth_score),
        )

    def build_scene_profile(
        self,
        detections,
        frame_shape: tuple[int, int, int],
        frame: np.ndarray | None = None,
        lane_geometry: dict[str, float | int | None] | None = None,
        depth_frame: DepthFrame | None = None,
    ) -> dict[str, object]:
        if depth_frame is None:
            base_frame = frame if frame is not None else np.zeros(frame_shape, dtype=np.uint8)
            depth_frame = self.estimate_depth_frame(base_frame)
        observations = [self.estimate_depth(det, frame_shape, depth_frame, lane_geometry) for det in detections]
        closest = min(observations, key=lambda item: item.relative_distance, default=None)
        return {
            "observations": observations,
            "closest": closest,
            "depth_frame": depth_frame,
        }

    def draw_depth_overlay(self, frame: np.ndarray, observations: list[DepthObservation]) -> np.ndarray:
        annotated = frame.copy()
        for observation in observations:
            x1, y1, x2, y2 = map(int, observation.bbox)
            label = (
                f"{observation.label} d={observation.relative_distance:.0f}"
                f" u={observation.uncertainty:.0f}"
            )
            cv2.putText(
                annotated,
                label,
                (x1, min(frame.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated
