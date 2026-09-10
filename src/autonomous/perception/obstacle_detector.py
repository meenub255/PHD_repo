"""YOLOv8-World based open-vocabulary obstacle detection module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config.settings import (
    DETECTION_PROMPTS,
    DETECTION_INTERVAL,
    MODEL_CONFIDENCE,
    MODEL_IOU,
    MODEL_PATH,
    OBSTACLE_CLASSES,
    ROAD_CLASSES,
    SIGN_CLASSES,
)

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency fallback
    YOLO = None


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CANDIDATES = (
    Path(MODEL_PATH),
    ROOT_DIR / MODEL_PATH,
    ROOT_DIR / "models" / "yolov8n-seg.pt",
)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int
    mask: np.ndarray | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ObstacleDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = MODEL_CONFIDENCE,
        iou: float = MODEL_IOU,
        prompts: list[str] | None = None,
        detection_interval: int = DETECTION_INTERVAL,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.prompts = prompts or list(DETECTION_PROMPTS)
        self.detection_interval = max(1, int(detection_interval))
        self._frame_counter = 0
        self._class_whitelist: list[int] | None = None
        self.model = self._load_model()
        self._last_detections: list[Detection] = []

    def _resolve_model_path(self, model_path: str | Path | None) -> Path | None:
        candidates = []
        if model_path is not None:
            candidates.append(Path(model_path))
        candidates.extend(DEFAULT_MODEL_CANDIDATES)

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None

    def _load_model(self):
        if YOLO is None or self.model_path is None:
            return None

        model = YOLO(str(self.model_path))
        if self.prompts and "world" in self.model_path.name.lower() and hasattr(model, "set_classes"):
            try:
                model.set_classes(self.prompts)
            except Exception:
                pass
        try:
            names = getattr(model, "names", {})
            whitelist = [class_id for class_id, name in names.items() if name in self.prompts]
            self._class_whitelist = sorted(set(int(class_id) for class_id in whitelist)) or None
        except Exception:
            self._class_whitelist = None
        return model

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        self._frame_counter += 1
        if self.model is None:
            return [], frame.copy()

        should_infer = self._frame_counter % self.detection_interval == 1 or not self._last_detections
        if should_infer:
            detections = self._infer(frame)
            self._last_detections = detections
        else:
            detections = self._last_detections

        annotated = self.draw_detections(frame, detections)
        return detections, annotated

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        try:
            result = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                verbose=False,
                retina_masks=True,
                classes=self._class_whitelist,
            )[0]
        except Exception:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        detections: list[Detection] = []

        if boxes is None:
            return detections

        mask_polygons = getattr(masks, "xy", None) if masks is not None else None

        for index, box in enumerate(boxes):
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = names.get(class_id, str(class_id))

            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            if width < 10 or height < 10:
                continue

            mask_xy = None
            if mask_polygons is not None and index < len(mask_polygons):
                mask_xy = np.asarray(mask_polygons[index], dtype=np.float32)
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    class_id=class_id,
                    mask=mask_xy,
                )
            )

        return detections

    def estimate_lighting(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.clip(np.mean(gray), 0, 255))

    def estimate_traffic_density(self, detections: list[Detection], frame_shape: tuple[int, int, int]) -> float:
        frame_area = float(frame_shape[0] * frame_shape[1]) + 1e-6
        obstacle_count = sum(1 for det in detections if det.label in OBSTACLE_CLASSES)
        road_count = sum(1 for det in detections if det.label in ROAD_CLASSES)
        sign_count = sum(1 for det in detections if det.label in SIGN_CLASSES)
        box_occupancy = sum(det.area for det in detections) / frame_area
        density = obstacle_count * 1.5 + road_count * 1.0 + sign_count * 0.6 + box_occupancy * 18.0
        return float(np.clip(density, 0, 10))

    def nearest_front_obstacle(self, observations: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = [item for item in observations if item.get("label") in OBSTACLE_CLASSES]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.get("relative_distance", 100.0))

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        annotated = frame.copy()
        overlay = annotated.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            color = (0, 255, 0)
            if detection.label in OBSTACLE_CLASSES:
                color = (0, 0, 255)
            elif detection.label in ROAD_CLASSES:
                color = (0, 165, 255)
            elif detection.label in SIGN_CLASSES:
                color = (255, 200, 0)

            if detection.mask is not None and len(detection.mask) >= 3:
                polygon = np.asarray(detection.mask, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [polygon], color)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        return cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
