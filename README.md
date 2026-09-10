# Perception Intelligence for Autonomous Vehicles using Monocular Vision and Deep Learning

An autonomous driving perception and navigation platform integrating monocular vision, deep learning models, and Interval Type-2 Neuro-Fuzzy Logic for real-time risk assessment and vehicle control.

---

## Overview

The platform addresses perception and adaptive decision-making under uncertain environmental conditions:

1. **Objective 1 — Obstacle Detection & Adaptive Navigation**:
   - Open-vocabulary obstacle detection using **YOLOv8-World**
   - Multi-object tracking with Hungarian algorithm association
   - Relative depth estimation using monocular vision
   - Interval Type-2 Neuro-Fuzzy inference for collision avoidance steering and velocity commands

2. **Objective 2 — Lane Monitoring & Vehicle Perception**:
   - Sliding-window polynomial curve fitting for lane detection
   - Perspective transform (Bird's-Eye View) and lane curvature analysis
   - Vehicle offset calculation and trajectory prediction

3. **Objective 3 — Unified Perception Intelligence Framework**:
   - Comprehensive situational awareness combining lane analysis, obstacle spatial tracking, and depth cues
   - Dynamic danger assessment and automated collision warning / emergency braking

4. **Complete Workflow**:
   - End-to-end synchronized execution displaying live telemetry, fuzzy decision variables, and overlaid video streams

---

## Directory Structure

```text
├── app.py                          # Application entry point
├── pyproject.toml                  # Build and package metadata
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── src/
│   └── autonomous/
│       ├── __init__.py             # Package exports
│       ├── api/                    # Web application layer
│       │   ├── app.py              # Flask server and route handlers
│       │   ├── templates/          # Jinja2 HTML templates
│       │   │   ├── base.html
│       │   │   ├── overview.html
│       │   │   ├── objective1.html
│       │   │   ├── objective2.html
│       │   │   ├── objective3.html
│       │   │   └── complete_workflow.html
│       │   └── static/             # Static web assets
│       │       └── css/
│       │           └── main.css
│       ├── config/                 # Settings & parameters
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── control/                # Actuator & speed control
│       │   ├── __init__.py
│       │   └── vehicle_controller.py
│       ├── core/                   # Interval Type-2 Neuro-Fuzzy logic
│       │   ├── __init__.py
│       │   └── neuro_fuzzy.py
│       ├── navigation/             # Adaptive navigation planner
│       │   ├── __init__.py
│       │   └── adaptive_navigator.py
│       ├── perception/             # Vision pipelines
│       │   ├── __init__.py
│       │   ├── depth_estimator.py
│       │   ├── lane_detector.py
│       │   ├── obstacle_detector.py
│       │   ├── advanced_lane_detector.py
│       │   └── scene_understanding.py
│       └── tracking/               # Object tracking
│           ├── __init__.py
│           └── object_tracker.py
│
├── models/                         # Model weights directory (.gitignored)
├── uploads/                        # Uploaded test media (.gitignored)
├── logs/                           # System and debug logs (.gitignored)
└── tests/                          # Automated test suite
```

---

## Installation

### 1. Clone & Setup Environment

```bash
git clone <repo-url>
cd "fuzzy system"

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Linux/macOS
# or
.\venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Web Interface

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Available Routes

| Route | Description |
|---|---|
| `/` | System overview and pipeline selection portal |
| `/objective1` | Obstacle detection & adaptive navigation |
| `/objective2` | Lane monitoring & road curvature perception |
| `/objective3` | Unified perception intelligence framework |
| `/complete-workflow` | End-to-end integrated video pipeline & telemetry |
| `/api/telemetry` | Real-time JSON telemetry endpoint |
| `/video_feed` | Multipart JPEG video streaming route |

---

## License

Academic / Research use.
