"""Interval Type-2 Neuro-Fuzzy Logic System for autonomous vehicle navigation decision making."""
from __future__ import annotations

import numpy as np

from ..config.settings import (
    FUZZY_DENSITY_MFS,
    FUZZY_LIGHT_MFS,
    FUZZY_LANE_MFS,
    FUZZY_OBST_MFS,
    SPEED_CONSEQUENTS,
    STEERING_CONSEQUENTS,
)


class IntervalType2MF:
    def __init__(self, mean, std_lower, std_upper):
        self.mean = mean
        self.std_lower = std_lower
        self.std_upper = std_upper

    def get_membership(self, x):
        upper_mu = np.exp(-0.5 * ((x - self.mean) / self.std_upper) ** 2)
        lower_mu = np.exp(-0.5 * ((x - self.mean) / self.std_lower) ** 2)
        return lower_mu, upper_mu


class NeuroFuzzyLogic:
    def __init__(self):
        self._init_membership_functions()
        self._init_rules()

    def _init_membership_functions(self):
        self.lane_mfs = {k: IntervalType2MF(*v) for k, v in FUZZY_LANE_MFS.items()}
        self.obst_mfs = {k: IntervalType2MF(*v) for k, v in FUZZY_OBST_MFS.items()}
        self.density_mfs = {k: IntervalType2MF(*v) for k, v in FUZZY_DENSITY_MFS.items()}
        self.light_mfs = {k: IntervalType2MF(*v) for k, v in FUZZY_LIGHT_MFS.items()}
        self.lane_quality_mfs = {
            "poor": IntervalType2MF(25, 10, 18),
            "fair": IntervalType2MF(50, 12, 18),
            "good": IntervalType2MF(75, 10, 16),
            "excellent": IntervalType2MF(92, 6, 10),
        }
        self.uncertainty_mfs = {
            "low": IntervalType2MF(10, 5, 10),
            "medium": IntervalType2MF(40, 10, 18),
            "high": IntervalType2MF(75, 10, 18),
        }
        self.occupancy_mfs = {
            "sparse": IntervalType2MF(10, 5, 12),
            "moderate": IntervalType2MF(35, 8, 14),
            "dense": IntervalType2MF(70, 10, 18),
        }
        self.motion_mfs = {
            "static": IntervalType2MF(0, 2, 5),
            "moving": IntervalType2MF(8, 3, 7),
            "fast": IntervalType2MF(18, 5, 10),
        }
        self.risk_mfs = {
            "low": IntervalType2MF(20, 8, 15),
            "medium": IntervalType2MF(50, 10, 18),
            "high": IntervalType2MF(80, 8, 14),
        }
        self.bearing_mfs = {
            "left": IntervalType2MF(-25, 8, 15),
            "center": IntervalType2MF(0, 8, 12),
            "right": IntervalType2MF(25, 8, 15),
        }
        self.steering_consequents = STEERING_CONSEQUENTS
        self.speed_consequents = SPEED_CONSEQUENTS

    def _init_rules(self):
        self.rules = [
            ("far_left", None, None, None, None, None, None, None, "hard_right", "medium", "Recover from far-left drift"),
            ("left", None, None, None, None, None, None, None, "right", "medium", "Correct left drift"),
            ("center", None, None, None, None, None, None, None, "straight", "fast", "Hold center lane"),
            ("right", None, None, None, None, None, None, None, "left", "medium", "Correct right drift"),
            ("far_right", None, None, None, None, None, None, None, "hard_left", "medium", "Recover from far-right drift"),
            (None, "critical", None, None, None, None, None, None, None, "stop", "Critical obstacle"),
            (None, "close", None, None, None, None, None, None, None, "slow", "Obstacle ahead, slow down"),
            ("center", "close", None, None, None, None, None, None, "left", "slow", "Centered obstacle, evasive left"),
            ("center", "close", None, None, None, None, None, "moving", "left", "slow", "Moving obstacle in lane"),
            (None, "medium", None, None, None, None, None, None, None, "medium", "Maintain cruising speed"),
            (None, "far", None, None, None, None, None, None, None, "fast", "Path clear"),
            (None, None, "high", None, None, None, None, None, None, "slow", "Dense traffic"),
            (None, None, None, "night", None, None, None, None, None, "medium", "Night driving"),
            (None, None, None, None, "poor", "high", "dense", None, None, "slow", "Poor visibility and dense road use"),
            (None, None, None, None, "good", "low", "sparse", "static", None, "fast", "Stable and clear scene"),
            (None, None, None, None, "fair", "medium", "moderate", "moving", None, "medium", "Moderate scene complexity"),
            (None, "close", None, None, "poor", "high", "dense", "fast", "hard_left", "stop", "High-risk evasive action"),
        ]

    def compute(
        self,
        lane_deviation,
        obstacle_distance,
        traffic_density=0,
        lighting_score=150,
        obstacle_bearing=0,
        lane_quality=100,
        obstacle_uncertainty=0,
        road_occupancy=0,
        dynamic_obstacles=0,
        scene_risk=0,
    ):
        memberships = {
            "lane": {k: mf.get_membership(lane_deviation) for k, mf in self.lane_mfs.items()},
            "obstacle": {k: mf.get_membership(obstacle_distance) for k, mf in self.obst_mfs.items()},
            "density": {k: mf.get_membership(traffic_density) for k, mf in self.density_mfs.items()},
            "light": {k: mf.get_membership(lighting_score) for k, mf in self.light_mfs.items()},
            "quality": {k: mf.get_membership(lane_quality) for k, mf in self.lane_quality_mfs.items()},
            "uncertainty": {k: mf.get_membership(obstacle_uncertainty) for k, mf in self.uncertainty_mfs.items()},
            "occupancy": {k: mf.get_membership(road_occupancy) for k, mf in self.occupancy_mfs.items()},
            "motion": {k: mf.get_membership(dynamic_obstacles) for k, mf in self.motion_mfs.items()},
            "risk": {k: mf.get_membership(scene_risk) for k, mf in self.risk_mfs.items()},
            "bearing": {k: mf.get_membership(obstacle_bearing) for k, mf in self.bearing_mfs.items()},
        }

        steering_num = 0.0
        steering_den = 0.0
        speed_num = 0.0
        speed_den = 0.0
        max_firing = -1.0
        active_rule_desc = "Analyzing..."
        rule_scores: list[tuple[float, str]] = []

        for rule in self.rules:
            firing, desc, steering_label, speed_label = self._evaluate_rule(rule, memberships)
            rule_scores.append((firing, desc))
            if firing > max_firing:
                max_firing = firing
                active_rule_desc = desc
            if steering_label:
                y = self.steering_consequents[steering_label]
                steering_num += y * firing
                steering_den += firing
            if speed_label:
                y = self.speed_consequents[speed_label]
                speed_num += y * firing
                speed_den += firing

        final_steering = steering_num / (steering_den + 1e-6)
        final_speed = speed_num / (speed_den + 1e-6)

        risk_penalty = np.clip((scene_risk * 0.45 + obstacle_uncertainty * 0.25 + road_occupancy * 0.15) / 100.0, 0, 0.85)
        visibility_penalty = np.clip((100.0 - lane_quality) / 180.0, 0, 0.45)
        speed_scale = float(np.clip(1.0 - risk_penalty - visibility_penalty, 0.05, 1.0))
        final_speed *= speed_scale

        if obstacle_distance <= 8 or scene_risk >= 85:
            final_speed = 0.0

        confidence = max_firing if max_firing >= 0 else 0.0
        top_rules = [desc for firing, desc in sorted(rule_scores, reverse=True)[:3] if firing > 0]

        return {
            "steering_angle": float(final_steering),
            "speed_control": float(np.clip(final_speed, 0, 100)),
            "active_rule": active_rule_desc,
            "decision_confidence": float(np.clip(confidence, 0, 1)),
            "top_rules": top_rules,
            "risk_score": float(np.clip(scene_risk, 0, 100)),
        }

    def _evaluate_rule(self, rule, memberships):
        values = list(rule)
        desc = values[-1]
        steering_label = values[-3]
        speed_label = values[-2]
        labels = values[:-3]
        groups = [
            ("lane", labels[0] if len(labels) > 0 else None),
            ("obstacle", labels[1] if len(labels) > 1 else None),
            ("density", labels[2] if len(labels) > 2 else None),
            ("light", labels[3] if len(labels) > 3 else None),
            ("quality", labels[4] if len(labels) > 4 else None),
            ("uncertainty", labels[5] if len(labels) > 5 else None),
            ("occupancy", labels[6] if len(labels) > 6 else None),
            ("motion", labels[7] if len(labels) > 7 else None),
            ("bearing", labels[8] if len(labels) > 8 else None),
        ]

        low_values = []
        up_values = []
        for group_name, label in groups:
            if label is None:
                continue
            low, up = memberships[group_name][label]
            low_values.append(low)
            up_values.append(up)

        if not low_values:
            return 0.0, desc, steering_label, speed_label

        firing_low = min(low_values)
        firing_up = min(up_values)
        firing = float((firing_low + firing_up) / 2.0)
        return firing, desc, steering_label, speed_label

