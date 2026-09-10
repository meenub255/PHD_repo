"""Multi-object tracking using Hungarian algorithm assignment over bounding box IoU."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config.settings import TRACK_IOU_THRESHOLD, TRACK_MAX_AGE, TRACK_MOTION_THRESHOLD


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter_area + 1e-6
    return float(inter_area / union)


@dataclass
class TrackState:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    age: int = 1
    hits: int = 1
    misses: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    is_dynamic: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return _bbox_center(self.bbox)

    @property
    def stability(self) -> float:
        return self.hits / max(1, self.age)

    @property
    def speed_px(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "bbox": tuple(round(float(value), 2) for value in self.bbox),
            "center": tuple(round(float(value), 2) for value in self.center),
            "velocity": tuple(round(float(value), 2) for value in self.velocity),
            "speed_px": round(float(self.speed_px), 2),
            "age": self.age,
            "hits": self.hits,
            "misses": self.misses,
            "stability": round(float(self.stability), 2),
            "is_dynamic": self.is_dynamic,
            "history": [(round(float(x), 2), round(float(y), 2)) for x, y in self.history[-8:]],
        }


class ObjectTracker:
    def __init__(
        self,
        max_age: int = TRACK_MAX_AGE,
        motion_threshold: float = TRACK_MOTION_THRESHOLD,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
    ):
        self.max_age = max_age
        self.motion_threshold = motion_threshold
        self.iou_threshold = iou_threshold
        self._tracks: list[TrackState] = []
        self._next_id = 1

    def update(self, detections: Iterable, frame_shape: tuple[int, int, int]) -> list[TrackState]:
        detections = list(detections)
        if not self._tracks and not detections:
            return []

        for track in self._tracks:
            track.age += 1
            track.misses += 1

        if not self._tracks:
            self._tracks = [self._create_track(det) for det in detections]
            return self._tracks.copy()

        if not detections:
            self._tracks = [track for track in self._tracks if track.misses <= self.max_age]
            return self._tracks.copy()

        cost_matrix = np.ones((len(self._tracks), len(detections)), dtype=np.float32) * 10.0
        frame_height, frame_width = frame_shape[:2]

        for track_index, track in enumerate(self._tracks):
            track_center = np.asarray(track.center, dtype=np.float32)
            for det_index, detection in enumerate(detections):
                det_center = np.asarray(detection.center, dtype=np.float32)
                iou_score = _iou(track.bbox, detection.bbox)
                if iou_score < self.iou_threshold:
                    continue
                distance = float(np.linalg.norm((track_center - det_center) / np.asarray([frame_width, frame_height], dtype=np.float32)))
                label_penalty = 0.0 if track.label == detection.label else 0.20
                cost_matrix[track_index, det_index] = 1.0 - iou_score + distance + label_penalty

        assignments = []
        if cost_matrix.size:
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            assignments = [
                (int(track_index), int(detection_index))
                for track_index, detection_index in zip(track_indices, detection_indices)
                if cost_matrix[track_index, detection_index] < 1.6
            ]

        matched_tracks = set()
        matched_detections = set()
        for track_index, detection_index in assignments:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            self._update_track(track, detection)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.is_dynamic = track.speed_px > self.motion_threshold

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._tracks.append(self._create_track(detection))

        self._tracks = [track for track in self._tracks if track.misses <= self.max_age]
        return sorted(self._tracks, key=lambda item: item.track_id)

    def _create_track(self, detection) -> TrackState:
        center = detection.center
        track = TrackState(
            track_id=self._next_id,
            label=detection.label,
            bbox=detection.bbox,
            confidence=detection.confidence,
            history=[center],
        )
        self._next_id += 1
        return track

    def _update_track(self, track: TrackState, detection) -> None:
        previous_center = np.asarray(track.center, dtype=np.float32)
        current_center = np.asarray(detection.center, dtype=np.float32)
        velocity = tuple((current_center - previous_center).tolist())
        track.velocity = tuple(0.7 * old + 0.3 * new for old, new in zip(track.velocity, velocity))
        track.label = detection.label
        track.confidence = detection.confidence
        track.bbox = detection.bbox
        track.hits += 1
        track.misses = 0
        track.history.append(detection.center)
        track.is_dynamic = track.speed_px > self.motion_threshold
