"""
Centralized configuration for the autonomous system.
"""

DETECTION_PROMPTS = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "stop sign", "traffic light", "speed limit sign", "yield sign",
    "crosswalk", "construction zone", "pothole", "wet road"
]

OBSTACLE_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
SIGN_CLASSES = {"stop sign", "traffic light", "speed limit sign", "yield sign", "crosswalk"}
ROAD_CLASSES = {"construction zone", "pothole", "wet road"}

MODEL_PATH = "yolov8s-world.pt"
MODEL_CONFIDENCE = 0.30
MODEL_IOU = 0.45
TRACK_PERSIST = True
DETECTION_INTERVAL = 3
TRACK_MAX_AGE = 12
TRACK_IOU_THRESHOLD = 0.25
TRACK_MOTION_THRESHOLD = 8
DEPTH_MODEL_NAME = "MiDaS_small"
DEPTH_MODEL_PREFER_HUB = True
SCENE_WARNING_RISK = 55
SCENE_CRITICAL_RISK = 80

LANE_ROI_BOTTOM = 0.95
LANE_ROI_TOP = 0.58
LANE_ROI_LEFT = 0.25
LANE_ROI_RIGHT = 0.75
LANE_SMOOTHING = 0.2

FUZZY_LANE_MFS = {
    "far_left": (-100, 15, 25),
    "left": (-50, 15, 25),
    "center": (0, 10, 20),
    "right": (50, 15, 25),
    "far_right": (100, 15, 25),
}

FUZZY_OBST_MFS = {
    "critical": (0, 10, 20),
    "close": (30, 10, 20),
    "medium": (60, 15, 25),
    "far": (100, 20, 30),
}

FUZZY_DENSITY_MFS = {
    "low": (0, 2, 4),
    "medium": (5, 2, 3),
    "high": (10, 3, 5),
}

FUZZY_LIGHT_MFS = {
    "night": (50, 30, 40),
    "day": (180, 50, 70),
}

STEERING_CONSEQUENTS = {
    "hard_right": 45,
    "right": 25,
    "straight": 0,
    "left": -25,
    "hard_left": -45,
}

SPEED_CONSEQUENTS = {
    "stop": 0,
    "slow": 30,
    "medium": 60,
    "fast": 100,
}

FUZZY_RULES = [
    ("far_left", None, None, None, "hard_right", "medium", "Far Left -> Correct Hard Right"),
    ("left", None, None, None, "right", "medium", "Left Drift -> Steer Right"),
    ("center", None, None, None, "straight", "fast", "Lane Center -> Hold Straight"),
    ("right", None, None, None, "left", "medium", "Right Drift -> Steer Left"),
    ("far_right", None, None, None, "hard_left", "medium", "Far Right -> Correct Hard Left"),
    (None, "critical", None, None, None, "stop", "CRITICAL OBSTACLE -> EMERGENCY BRAKE"),
    (None, "close", None, None, None, "slow", "Obstacle Ahead -> Reduce Speed"),
    ("center", "close", None, None, "left", "slow", "Centered Obstacle Close -> Evasive Steer Left"),
    (None, "medium", None, None, None, "medium", "Traffic Ahead -> Cruising Speed"),
    (None, "far", None, None, None, "fast", "Path Clear -> Full Speed"),
    (None, None, "high", None, None, "slow", "Dense Traffic -> Cautious Speed"),
    (None, None, None, "night", None, "medium", "Night Time -> Reduced Max Speed"),
    (None, "medium", "high", "night", None, "slow", "Night + Dense Traffic -> Very Cautious"),
]

EVASIVE_DIST_THRESHOLD = 40
EVASIVE_BEARING_THRESHOLD = 15
EVASIVE_GAIN = 0.35

RISK_CRITICAL = 20
RISK_WARNING = 50

MAX_FRAME_DIM = 960
NIGHT_BRIGHTNESS_THRESHOLD = 110
GAMMA_NIGHT = 1.5

TRACK_MAXLEN = 30
TRACK_MOTION_THRESHOLD = 15

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
SERVER_DEBUG = False
SERVER_THREADED = True

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp", "webp"}
