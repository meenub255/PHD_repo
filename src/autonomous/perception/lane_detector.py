import cv2
import numpy as np
from collections import deque
from ..config.settings import (
    GAMMA_NIGHT,
    LANE_ROI_BOTTOM, LANE_ROI_TOP, LANE_ROI_LEFT, LANE_ROI_RIGHT,
    LANE_SMOOTHING,
    NIGHT_BRIGHTNESS_THRESHOLD,
)


class LaneDetector:
    def __init__(self):
        self._left_history = deque(maxlen=8)
        self._right_history = deque(maxlen=8)
        self._left_stable = None
        self._right_stable = None
        self._left_miss_count = 0
        self._right_miss_count = 0
        self._smoothing_alpha = max(0.15, min(float(LANE_SMOOTHING), 0.35))

    def detect_lanes(self, frame):
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
        w = frame_shape[1]
        cx = w // 2
        if left_stable is not None:
            left_stable[0][0][0] = min(int(left_stable[0][0][0]), cx)
            left_stable[0][1][0] = min(int(left_stable[0][1][0]), cx)
        if right_stable is not None:
            right_stable[0][0][0] = max(int(right_stable[0][0][0]), cx)
            right_stable[0][1][0] = max(int(right_stable[0][1][0]), cx)
        if left_stable is not None and right_stable is not None:
            lb, lt = int(left_stable[0][0][0]), int(left_stable[0][1][0])
            rb, rt = int(right_stable[0][0][0]), int(right_stable[0][1][0])
            if lb >= rb or lt >= rt:
                left_stable = None
                right_stable = None
                self._left_stable = None
                self._right_stable = None
                self._left_history.clear()
                self._right_history.clear()
        self._left_stable = left_stable
        self._right_stable = right_stable
        return left_stable, right_stable

    def _stabilize_side(self, frame_shape, current_line, history, previous_line, side):
        miss_attr = "_left_miss_count" if side == "left" else "_right_miss_count"
        miss_count = getattr(self, miss_attr)
        w = frame_shape[1]

        if current_line is not None:
            if previous_line is not None:
                prev_pts = np.asarray(previous_line[0], dtype=np.float32)
                curr_pts = np.asarray(current_line[0], dtype=np.float32)
                max_jump = w * 0.1
                if (abs(curr_pts[0][0] - prev_pts[0][0]) > max_jump and
                    abs(curr_pts[1][0] - prev_pts[1][0]) > max_jump):
                    current_line = previous_line
                else:
                    history.append(curr_pts)
                    setattr(self, miss_attr, 0)
            else:
                history.append(np.asarray(current_line[0], dtype=np.float32))
                setattr(self, miss_attr, 0)
        else:
            miss_count += 1
            setattr(self, miss_attr, miss_count)
            if miss_count > 4:
                history.clear()
                return None

        if not history:
            return current_line if current_line is not None else None

        stacked = np.stack(list(history), axis=0)
        ref = np.median(stacked, axis=0)

        if previous_line is not None:
            prev = np.asarray(previous_line[0], dtype=np.float32)
            smoothed = (1.0 - self._smoothing_alpha) * prev + self._smoothing_alpha * ref
        else:
            smoothed = ref

        h = frame_shape[0]
        y1 = h
        y2 = int(h * LANE_ROI_TOP)
        x1 = int(np.clip(round(smoothed[0][0]), 0, w))
        x2 = int(np.clip(round(smoothed[1][0]), 0, w))
        return np.array([[[x1, y1], [x2, y2]]], dtype=np.int32)

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < NIGHT_BRIGHTNESS_THRESHOLD:
            gray = self.enhance_low_light(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blur, 50, 150)

    def enhance_low_light(self, gray):
        adjusted = cv2.convertScaleAbs(gray, alpha=1.3, beta=18)
        gamma = max(float(GAMMA_NIGHT), 1.0)
        inv_gamma = 1.0 / gamma
        lookup = np.array([(index / 255.0) ** inv_gamma * 255 for index in range(256)], dtype=np.uint8)
        adjusted = cv2.LUT(adjusted, lookup)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(adjusted)

    def region_of_interest(self, image):
        height = image.shape[0]
        width = image.shape[1]
        polygons = np.array([
            [(int(width * LANE_ROI_LEFT), height),
             (int(width * LANE_ROI_RIGHT), height),
             (int(width * LANE_ROI_RIGHT), int(height * LANE_ROI_TOP)),
             (int(width * LANE_ROI_LEFT), int(height * LANE_ROI_TOP))]
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

    def draw_lanes(self, frame, left_curve, right_curve):
        overlay = frame.copy()

        if left_curve is not None:
            cv2.polylines(overlay, [left_curve], isClosed=False, color=(0, 255, 0), thickness=3)
        if right_curve is not None:
            cv2.polylines(overlay, [right_curve], isClosed=False, color=(0, 255, 0), thickness=3)

        return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
