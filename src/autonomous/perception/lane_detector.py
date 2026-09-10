"""TwinLiteNet deep learning and classical computer-vision lane detection."""
from collections import deque
import logging
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from ..config.settings import (
    GAMMA_NIGHT,
    LANE_ROI_BOTTOM,
    LANE_ROI_LEFT,
    LANE_ROI_RIGHT,
    LANE_ROI_TOP,
    LANE_SMOOTHING,
    NIGHT_BRIGHTNESS_THRESHOLD,
)

logger = logging.getLogger(__name__)


class LaneDetector:
    """Lane detector using TwinLiteNet ONNX model with classical Hough fallback."""

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
                logger.info("TwinLiteNet ONNX model loaded successfully from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load TwinLiteNet ONNX model: %s", e)
                self.session = None

    def detect_lanes(self, frame: np.ndarray):
        """Detect lanes using TwinLiteNet ONNX model, falling back to classical pipeline."""
        if self.session is not None:
            try:
                return self._detect_lanes_twinlitenet(frame)
            except Exception as e:
                logger.warning("TwinLiteNet inference error: %s, falling back to classical", e)

        return self._detect_lanes_classical(frame)

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
        # Only consider the lower part of the frame (actual road)
        roi_top = int(h_orig * 0.40)
        roi_bottom = h_orig

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
            # Must reach the lower road region
            if y + ch < roi_top:
                continue
            # Component must have reasonable vertical span (at least 15% of frame height)
            if ch < h_orig * 0.15:
                continue
            # Component must be taller than wide (lane-like shape, not a blob)
            if ch < cw * 0.5:
                continue
            cent_x = centroids[i][0]
            # Ego left: component to the left of center, not too far left
            if cent_x < cx and cent_x > w_orig * 0.10:
                right_edge = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                if right_edge > best_left_x:
                    best_left_x = right_edge
                    ego_left_idx = i
            # Ego right: component to the right of center, not too far right
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

        self._last_lane_mask = ego_mask  # Only ego lane pixels stored
        self._last_labels = labels_full
        self._ego_left_idx = ego_left_idx
        self._ego_right_idx = ego_right_idx

        # ── Fit lines on ego components ────────────────────────────────────────
        left_curve = None
        right_curve = None
        left_bottom_x = None
        right_bottom_x = None
        fit_roi_top = int(h_orig * 0.58)
        min_lane_separation = int(w_orig * 0.08)  # 8% of width minimum gap

        if ego_left_idx is not None:
            pts = np.column_stack(np.where(labels_full == ego_left_idx))[:, ::-1]  # (x, y)
            if len(pts) >= 50:
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                slope = float(vy / (vx + 1e-6))
                intercept = float(y0 - slope * x0)
                # Lane lines should be roughly vertical (slope magnitude > 0.5)
                # and not near-horizontal (which would be noise)
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
                # Lane lines should be roughly vertical (slope magnitude > 0.5)
                # and not near-horizontal (which would be noise)
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

    def _detect_lanes_classical(self, frame: np.ndarray):
        """Classical Canny + Hough transform fallback."""
        lane_image = np.copy(frame)
        canny_image = self.canny(lane_image)
        cropped_image = self.region_of_interest(canny_image)

        h, w = cropped_image.shape[:2]
        lines = cv2.HoughLinesP(
            cropped_image, 1, np.pi / 180, 50,
            np.array([]), minLineLength=40, maxLineGap=150
        )

        left_line, right_line = self._average_lines(lines, w, h)

        left_curve, right_curve = None, None
        left_orig, right_orig = None, None

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            left_orig = np.array([[[x1, y1], [x2, y2]]], dtype=np.int32)
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            right_orig = np.array([[[x1, y1], [x2, y2]]], dtype=np.int32)

        left_curve, right_curve = self._stabilize_lines(
            frame.shape, left_orig, right_orig
        )

        averaged_lines = [
            np.array([left_curve[0][0][0], left_curve[0][0][1],
                       left_curve[0][1][0], left_curve[0][1][1]])
            if left_curve is not None else None,
            np.array([right_curve[0][0][0], right_curve[0][0][1],
                       right_curve[0][1][0], right_curve[0][1][1]])
            if right_curve is not None else None,
        ]

        deviation = self.compute_deviation(averaged_lines, frame.shape[1])
        return left_curve, right_curve, deviation

    def _average_lines(self, lines, w, h):
        if lines is None:
            return None, None

        left_slopes = []
        left_intercepts = []
        right_slopes = []
        right_intercepts = []

        cx = w // 2
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            intercept = y1 - slope * x1
            mid_x = (x1 + x2) / 2
            if mid_x < cx:
                left_slopes.append(slope)
                left_intercepts.append(intercept)
            else:
                right_slopes.append(slope)
                right_intercepts.append(intercept)

        left_line = None
        right_line = None

        if left_slopes:
            avg_slope = np.mean(left_slopes)
            avg_intercept = np.mean(left_intercepts)
            left_line = self._make_line(h, avg_slope, avg_intercept, w)
            if left_line is not None:
                x1 = left_line[0]
                if x1 >= w * 0.6:
                    left_line = None

        if right_slopes:
            avg_slope = np.mean(right_slopes)
            avg_intercept = np.mean(right_intercepts)
            right_line = self._make_line(h, avg_slope, avg_intercept, w)
            if right_line is not None:
                x1 = right_line[0]
                if x1 <= w * 0.4:
                    right_line = None

        if left_line is not None:
            left_line[0] = min(left_line[0], cx)
            left_line[2] = min(left_line[2], cx)

        if right_line is not None:
            right_line[0] = max(right_line[0], cx)
            right_line[2] = max(right_line[2], cx)

        if left_line is not None and right_line is not None:
            left_bottom, left_top = left_line[0], left_line[2]
            right_bottom, right_top = right_line[0], right_line[2]
            if left_bottom >= right_bottom or left_top >= right_top:
                left_line = None
                right_line = None

        return left_line, right_line

    def _make_line(self, h, slope, intercept, w, side=None):
        if abs(slope) < 0.001:
            slope = 0.001 if slope >= 0 else -0.001
        y1 = h
        y2 = int(h * LANE_ROI_TOP)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        return [x1, y1, x2, y2]

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

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < NIGHT_BRIGHTNESS_THRESHOLD:
            inv_gamma = 1.0 / GAMMA_NIGHT
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blur, 50, 150)

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        polygons = np.array([
            [
                (int(width * LANE_ROI_LEFT), int(height * LANE_ROI_BOTTOM)),
                (int(width * LANE_ROI_RIGHT), int(height * LANE_ROI_BOTTOM)),
                (int(width * 0.55), int(height * LANE_ROI_TOP)),
                (int(width * 0.45), int(height * LANE_ROI_TOP)),
            ]
        ], dtype=np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        return cv2.bitwise_and(image, mask)

    def compute_deviation(self, averaged_lines, width):
        center_x = width // 2
        lane_center = center_x

        if averaged_lines is not None:
            left = averaged_lines[0] if len(averaged_lines) > 0 else None
            right = averaged_lines[1] if len(averaged_lines) > 1 else None

            if left is not None and right is not None:
                l_x = left[0]
                r_x = right[0]
                lane_center = (l_x + r_x) // 2
            elif left is not None:
                l_x = left[0]
                lane_center = l_x + width // 4
            elif right is not None:
                r_x = right[0]
                lane_center = r_x - width // 4

        deviation_px = lane_center - center_x
        return np.clip((deviation_px / (width / 2)) * 100, -100, 100)

    def draw_lanes(self, frame: np.ndarray, left_curve=None, right_curve=None) -> np.ndarray:
        """Draw ONLY the vehicle's ego driving lane (no adjacent highway lanes).

        Pipeline:
          1. Soft green drivable carpet between ego left/right boundary lines.
          2. Colour ONLY the ego lane segmentation pixels green (stored in
             ``_last_lane_mask`` which now contains only ego-component pixels).
          3. Crisp green boundary lines with a fine highlight edge.
        """
        annotated = frame.copy()
        w = frame.shape[1]
        min_separation = w * 0.05  # minimum 5% of width between left and right

        # Validate curves are not near-horizontal (false detections)
        def _is_valid_lane_curve(curve):
            if curve is None:
                return False
            try:
                pts = curve.reshape(-1, 2)
                if len(pts) < 2:
                    return False
                # Check vertical span - lane lines should span a significant portion of the frame
                y_span = abs(pts[-1, 1] - pts[0, 1])
                if y_span < frame.shape[0] * 0.25:
                    return False
                # Check that line is not too horizontal
                x_span = abs(pts[-1, 0] - pts[0, 0])
                if x_span > y_span * 2:
                    return False
                return True
            except Exception:
                return False

        left_valid = _is_valid_lane_curve(left_curve)
        right_valid = _is_valid_lane_curve(right_curve)

        # If only one side is valid, check if it's too far to one side (likely noise)
        if left_valid and not right_valid:
            try:
                pts = left_curve.reshape(-1, 2)
                if np.mean(pts[:, 0]) > w * 0.65:
                    left_valid = False
            except Exception:
                pass
        if right_valid and not left_valid:
            try:
                pts = right_curve.reshape(-1, 2)
                if np.mean(pts[:, 0]) < w * 0.35:
                    right_valid = False
            except Exception:
                pass

        # Check if left and right curves overlap
        curves_overlap = False
        if left_valid and right_valid:
            try:
                left_pts = left_curve.reshape(-1, 2)
                right_pts = right_curve.reshape(-1, 2)
                left_bottom_x = left_pts[0, 0]
                right_bottom_x = right_pts[0, 0]
                if abs(left_bottom_x - right_bottom_x) < min_separation:
                    curves_overlap = True
            except Exception:
                pass

        # ── 1. Translucent drivable-area carpet ────────────────────────────────
        if left_valid and right_valid and not curves_overlap:
            try:
                left_pts = left_curve.reshape(-1, 2)
                right_pts = right_curve.reshape(-1, 2)
                poly_pts = np.vstack([left_pts, right_pts[::-1]])
                poly_overlay = annotated.copy()
                cv2.fillPoly(poly_overlay, [poly_pts], (0, 200, 70))
                annotated = cv2.addWeighted(annotated, 0.82, poly_overlay, 0.18, 0)
            except Exception:
                pass

        # ── 2. Ego lane segmentation pixels (already isolated) ─────────────────
        if self._last_lane_mask is not None and np.any(self._last_lane_mask):
            green_overlay = annotated.copy()
            green_overlay[self._last_lane_mask > 0] = (0, 255, 0)
            annotated = cv2.addWeighted(annotated, 0.22, green_overlay, 0.78, 0)

        return annotated
