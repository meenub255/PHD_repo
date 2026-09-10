"""TwinLiteNet deep learning lane detection."""
from collections import deque
import logging
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..config.settings import LANE_SMOOTHING

logger = logging.getLogger(__name__)


class LaneDetector:
    """Lane detector using TwinLiteNet ONNX model."""

    def __init__(self, model_path: str | Path | None = None):
        self._left_history = deque(maxlen=8)
        self._right_history = deque(maxlen=8)
        self._left_stable = None
        self._right_stable = None
        self._left_miss_count = 0
        self._right_miss_count = 0
        self._smoothing_alpha = max(0.15, min(float(LANE_SMOOTHING), 0.35))
        self._last_lane_mask = None
        self._last_da_mask = None
        self._last_labels = None
        self._ego_left_idx = None
        self._ego_right_idx = None

        # Resolve TwinLiteNet ONNX model path
        if model_path is None:
            root_dir = Path(__file__).resolve().parents[3]
            candidates = [
                root_dir / "TwinLiteNet-onnxruntime" / "models" / "best.onnx",
                root_dir / "models" / "best.onnx",
                Path("TwinLiteNet-onnxruntime/models/best.onnx"),
            ]
            for c in candidates:
                if c.exists():
                    model_path = c
                    break

        self.session = None
        self._input_name = None
        if ort is not None and model_path is not None and Path(model_path).exists():
            try:
                self.session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self.session.get_inputs()[0].name
                logger.info("TwinLiteNet ONNX model loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load TwinLiteNet ONNX model: %s", e)
                self.session = None

    def detect_lanes(self, frame: np.ndarray):
        """Detect lanes using TwinLiteNet ONNX model."""
        if self.session is not None:
            return self._detect_lanes_twinlitenet(frame)

        # Return empty result if model not loaded
        return None, None, 0.0

    def _detect_lanes_twinlitenet(self, frame: np.ndarray):
        """Run TwinLiteNet ONNX inference. Isolates only the vehicle's ego driving lane
        using connected-component analysis so adjacent highway lanes are excluded."""
        h_orig, w_orig = frame.shape[:2]
        cx = w_orig // 2

        # ── Preprocess ────────────────────────────────────────────────────────
        img = cv2.resize(frame, (640, 360)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        outputs = self.session.run(None, {self._input_name: img})

        # Store drivable-area mask
        da_mask_360 = np.argmax(outputs[0][0], axis=0).astype(np.uint8) * 255
        self._last_da_mask = cv2.resize(da_mask_360, (w_orig, h_orig))

        # ── Full lane segmentation mask ────────────────────────────────────────
        lane_mask_360 = np.argmax(outputs[1][0], axis=0).astype(np.uint8) * 255
        full_lane_mask = cv2.resize(lane_mask_360, (w_orig, h_orig))

        # ── Find ego lane components via connected components ─────────────────
        roi_top = int(h_orig * 0.40)

        num_labels, labels_full, stats, centroids = cv2.connectedComponentsWithStats(full_lane_mask)

        ego_left_idx = None
        ego_right_idx = None
        best_left_x = -1
        best_right_x = w_orig * 2

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 300:
                continue
            y = stats[i, cv2.CC_STAT_TOP]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            if y + ch < roi_top:
                continue
            if ch < h_orig * 0.15:
                continue
            if ch < cw * 0.5:
                continue
            cent_x = centroids[i][0]
            if cent_x < cx and cent_x > w_orig * 0.10:
                right_edge = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                if right_edge > best_left_x:
                    best_left_x = right_edge
                    ego_left_idx = i
            elif cent_x >= cx and cent_x < w_orig * 0.90:
                left_edge = stats[i, cv2.CC_STAT_LEFT]
                if left_edge < best_right_x:
                    best_right_x = left_edge
                    ego_right_idx = i

        # Build ego-only pixel mask
        ego_mask = np.zeros_like(full_lane_mask)
        if ego_left_idx is not None:
            ego_mask[labels_full == ego_left_idx] = 255
        if ego_right_idx is not None:
            ego_mask[labels_full == ego_right_idx] = 255

        self._last_lane_mask = ego_mask
        self._last_labels = labels_full
        self._ego_left_idx = ego_left_idx
        self._ego_right_idx = ego_right_idx

        # ── Fit lines on ego components ────────────────────────────────────────
        left_curve = None
        right_curve = None
        left_bottom_x = None
        right_bottom_x = None
        fit_roi_top = int(h_orig * 0.58)
        roi_bottom = h_orig
        min_lane_separation = int(w_orig * 0.08)

        if ego_left_idx is not None:
            pts = np.column_stack(np.where(labels_full == ego_left_idx))[:, ::-1]
            if len(pts) >= 50:
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                slope = float(vy / (vx + 1e-6))
                intercept = float(y0 - slope * x0)
                if abs(slope) > 0.5 and abs(slope) < 10.0:
                    x1 = int(np.clip((roi_bottom - intercept) / slope, 0, cx - min_lane_separation))
                    x2 = int(np.clip((fit_roi_top - intercept) / slope, 0, cx - min_lane_separation))
                    left_bottom_x = x1
                    left_orig = np.array([[[x1, roi_bottom], [x2, fit_roi_top]]], dtype=np.int32)
                    left_curve, _ = self._stabilize_lines(frame.shape, left_orig, None)

        if ego_right_idx is not None:
            pts = np.column_stack(np.where(labels_full == ego_right_idx))[:, ::-1]
            if len(pts) >= 50:
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                slope = float(vy / (vx + 1e-6))
                intercept = float(y0 - slope * x0)
                if abs(slope) > 0.5 and abs(slope) < 10.0:
                    x1 = int(np.clip((roi_bottom - intercept) / slope, cx + min_lane_separation, w_orig))
                    x2 = int(np.clip((fit_roi_top - intercept) / slope, cx + min_lane_separation, w_orig))
                    right_bottom_x = x1
                    right_orig = np.array([[[x1, roi_bottom], [x2, fit_roi_top]]], dtype=np.int32)
                    _, right_curve = self._stabilize_lines(frame.shape, None, right_orig)

        # ── Reject if lines are too close (overlapping) ─────────────────────────
        if left_curve is not None and right_curve is not None:
            lb = left_curve.reshape(-1, 2)[0, 0]
            rb = right_curve.reshape(-1, 2)[0, 0]
            if abs(lb - rb) < min_lane_separation:
                left_curve = None
                right_curve = None

        # ── Deviation ─────────────────────────────────────────────────────────
        lane_center = cx
        if left_bottom_x is not None and right_bottom_x is not None:
            lane_center = (left_bottom_x + right_bottom_x) // 2
        elif left_bottom_x is not None:
            lane_center = left_bottom_x + (w_orig // 4)
        elif right_bottom_x is not None:
            lane_center = right_bottom_x - (w_orig // 4)

        deviation_px = lane_center - cx
        deviation = float(np.clip((deviation_px / (w_orig / 2.0)) * 100.0, -100.0, 100.0))
        return left_curve, right_curve, deviation

    def _stabilize_lines(self, frame_shape, left_orig, right_orig):
        left_stable = self._stabilize_side(
            frame_shape, left_orig, self._left_history,
            self._left_stable, "left"
        )
        right_stable = self._stabilize_side(
            frame_shape, right_orig, self._right_history,
            self._right_stable, "right"
        )

        self._left_stable = left_stable
        self._right_stable = right_stable

        return left_stable, right_stable

    def _stabilize_side(self, frame_shape, current_line, history, last_stable, side):
        w = frame_shape[1]
        cx = w // 2

        if current_line is not None:
            coords = current_line[0]
            x1, y1 = coords[0]
            x2, y2 = coords[1]

            if side == "left":
                if x1 >= cx or x2 >= cx:
                    current_line = None
            else:
                if x1 <= cx or x2 <= cx:
                    current_line = None

        if current_line is not None:
            if side == "left":
                self._left_miss_count = 0
            else:
                self._right_miss_count = 0

            history.append(current_line[0].copy())

            avg_coords = np.mean(history, axis=0).astype(np.int32)

            if last_stable is not None:
                alpha = self._smoothing_alpha
                smoothed = (
                    alpha * avg_coords.astype(np.float64)
                    + (1.0 - alpha) * last_stable[0].astype(np.float64)
                ).astype(np.int32)
                return np.array([smoothed], dtype=np.int32)
            else:
                return np.array([avg_coords], dtype=np.int32)
        else:
            miss_count = self._left_miss_count if side == "left" else self._right_miss_count
            miss_count += 1
            if side == "left":
                self._left_miss_count = miss_count
            else:
                self._right_miss_count = miss_count

            if last_stable is not None and miss_count <= 3:
                return last_stable
            else:
                if side == "left":
                    self._left_history.clear()
                    self._left_stable = None
                else:
                    self._right_history.clear()
                    self._right_stable = None
                return None

    def draw_lanes(self, frame: np.ndarray, left_curve=None, right_curve=None) -> np.ndarray:
        """Draw segmented lane pixels only — no fill, no boundary lines."""
        annotated = frame.copy()

        if self._last_lane_mask is not None and np.any(self._last_lane_mask):
            green_overlay = annotated.copy()
            green_overlay[self._last_lane_mask > 0] = (0, 255, 0)
            annotated = cv2.addWeighted(annotated, 0.22, green_overlay, 0.78, 0)

        return annotated
