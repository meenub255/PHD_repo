import sys
import os
import cv2
import numpy as np
import pickle
from collections import deque

SECOND_LANE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'second lane'))
if SECOND_LANE_DIR not in sys.path:
    sys.path.insert(0, SECOND_LANE_DIR)

from birdseye import BirdsEye
from lanefilter import LaneFilter
from curves import Curves


class AdvancedLaneDetector:
    def __init__(self):
        cal_path = os.path.join(SECOND_LANE_DIR, 'calibration_data.p')
        with open(cal_path, 'rb') as f:
            cal = pickle.load(f)

        cam_matrix = cal['camera_matrix']
        dist_coef = cal['distortion_coefficient']

        h, w = 720, 1280
        src_points = [
            (int(w * 0.05), int(h * 0.95)),
            (int(w * 0.45), int(h * 0.62)),
            (int(w * 0.55), int(h * 0.62)),
            (int(w * 0.95), int(h * 0.95))
        ]
        dst_points = [
            (int(w * 0.05), h),
            (int(w * 0.05), 0),
            (int(w * 0.95), 0),
            (int(w * 0.95), h)
        ]

        self.birdseye = BirdsEye(src_points, dst_points, cam_matrix, dist_coef)

        lane_filter_params = {
            'sat_thresh': 100,
            'light_thresh': 70,
            'light_thresh_agr': 150,
            'grad_thresh': (0.7, 1.3),
            'mag_thresh': 50,
            'x_thresh': 20,
        }
        self.lane_filter = LaneFilter(lane_filter_params)

        self.curves = Curves(
            number_of_windows=9,
            margin=80,
            minimum_pixels=30,
            ym_per_pix=30.0 / 720,
            xm_per_pix=3.7 / 700
        )

        self._left_history = deque(maxlen=8)
        self._right_history = deque(maxlen=8)
        self._left_stable = None
        self._right_stable = None
        self._left_miss_count = 0
        self._right_miss_count = 0
        self._smoothing_alpha = 0.25

    def detect_lanes(self, frame):
        h, w = frame.shape[:2]

        warped = self.birdseye.sky_view(frame)

        filtered = self.lane_filter.apply(warped)

        binary = (filtered > 0).astype(np.uint8)

        result = self.curves.fit(binary)

        left_curve = None
        right_curve = None

        if result.get('pixel_left_best_fit_curve') is not None and result.get('left_pixels_x') is not None and len(result['left_pixels_x']) > 0:
            kl = result['pixel_left_best_fit_curve']
            ys = np.linspace(0, h - 1, h)
            lxs = kl[0] * (ys ** 2) + kl[1] * ys + kl[2]
            lxs = np.clip(lxs, 0, w).astype(np.int32)
            ys = ys.astype(np.int32)

            src_pts = np.float32([[lxs[i], ys[i]] for i in range(0, h, 10)]).reshape(-1, 1, 2)
            inv_warp = self.birdseye.inv_warp_matrix
            dst_pts = cv2.perspectiveTransform(src_pts, inv_warp)
            dst_pts = dst_pts.reshape(-1, 2)

            valid = (dst_pts[:, 0] >= 0) & (dst_pts[:, 0] <= w) & \
                    (dst_pts[:, 1] >= 0) & (dst_pts[:, 1] <= h)
            dst_pts = dst_pts[valid]

            if len(dst_pts) >= 2:
                order = np.argsort(dst_pts[:, 1])[::-1]
                dst_pts = dst_pts[order]
                left_curve = dst_pts.astype(np.int32).reshape(-1, 1, 2)

        if result.get('pixel_right_best_fit_curve') is not None and result.get('right_pixels_x') is not None and len(result['right_pixels_x']) > 0:
            kr = result['pixel_right_best_fit_curve']
            ys = np.linspace(0, h - 1, h)
            rxs = kr[0] * (ys ** 2) + kr[1] * ys + kr[2]
            rxs = np.clip(rxs, 0, w).astype(np.int32)
            ys = ys.astype(np.int32)

            src_pts = np.float32([[rxs[i], ys[i]] for i in range(0, h, 10)]).reshape(-1, 1, 2)
            inv_warp = self.birdseye.inv_warp_matrix
            dst_pts = cv2.perspectiveTransform(src_pts, inv_warp)
            dst_pts = dst_pts.reshape(-1, 2)

            valid = (dst_pts[:, 0] >= 0) & (dst_pts[:, 0] <= w) & \
                    (dst_pts[:, 1] >= 0) & (dst_pts[:, 1] <= h)
            dst_pts = dst_pts[valid]

            if len(dst_pts) >= 2:
                order = np.argsort(dst_pts[:, 1])[::-1]
                dst_pts = dst_pts[order]
                right_curve = dst_pts.astype(np.int32).reshape(-1, 1, 2)

        left_curve, right_curve = self._stabilize_curves(
            frame.shape, left_curve, right_curve
        )

        averaged_lines = [
            np.array([left_curve[0][0][0], left_curve[0][0][1],
                       left_curve[0][-1][0], left_curve[0][-1][1]])
            if left_curve is not None else None,
            np.array([right_curve[0][0][0], right_curve[0][0][1],
                       right_curve[0][-1][0], right_curve[0][-1][1]])
            if right_curve is not None else None,
        ]

        deviation = self._compute_deviation(averaged_lines, w)
        return left_curve, right_curve, deviation

    def _stabilize_curves(self, frame_shape, left_curve, right_curve):
        h, w = frame_shape[:2]

        if left_curve is not None:
            self._left_history.append(left_curve[:, 0, 0].copy())
            self._left_miss_count = 0
        else:
            self._left_miss_count += 1
            if self._left_miss_count > 4:
                self._left_history.clear()

        if right_curve is not None:
            self._right_history.append(right_curve[:, 0, 0].copy())
            self._right_miss_count = 0
        else:
            self._right_miss_count += 1
            if self._right_miss_count > 4:
                self._right_history.clear()

        left_stable = None
        right_stable = None

        if len(self._left_history) > 0:
            stacked = np.stack(list(self._left_history), axis=0)
            median_x = np.median(stacked, axis=0)
            if self._left_stable is not None and len(self._left_stable) == len(median_x):
                smoothed_x = (1.0 - self._smoothing_alpha) * self._left_stable + self._smoothing_alpha * median_x
            else:
                smoothed_x = median_x
            self._left_stable = smoothed_x
            ys = np.linspace(h - 1, 0, len(smoothed_x)).astype(np.int32)
            pts = np.column_stack((np.clip(smoothed_x, 0, w).astype(np.int32), ys))
            left_stable = pts.reshape(-1, 1, 2)

        if len(self._right_history) > 0:
            stacked = np.stack(list(self._right_history), axis=0)
            median_x = np.median(stacked, axis=0)
            if self._right_stable is not None and len(self._right_stable) == len(median_x):
                smoothed_x = (1.0 - self._smoothing_alpha) * self._right_stable + self._smoothing_alpha * median_x
            else:
                smoothed_x = median_x
            self._right_stable = smoothed_x
            ys = np.linspace(h - 1, 0, len(smoothed_x)).astype(np.int32)
            pts = np.column_stack((np.clip(smoothed_x, 0, w).astype(np.int32), ys))
            right_stable = pts.reshape(-1, 1, 2)

        return left_stable, right_stable

    def _compute_deviation(self, averaged_lines, width):
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
            cv2.polylines(overlay, [left_curve], isClosed=False, color=(0, 255, 0), thickness=5)
        if right_curve is not None:
            cv2.polylines(overlay, [right_curve], isClosed=False, color=(0, 255, 0), thickness=5)

        return overlay
