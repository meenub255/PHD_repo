from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for

from ..config.settings import ALLOWED_EXTENSIONS, SERVER_HOST, SERVER_PORT, UPLOAD_FOLDER, MAX_FRAME_DIM
from ..navigation.adaptive_navigator import AdaptiveNavigator


ROOT_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = ROOT_DIR / UPLOAD_FOLDER
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _default_source() -> str | int:
    video_files = sorted(
        [path for path in UPLOAD_DIR.iterdir() if path.suffix.lower().lstrip(".") in {"mp4", "avi", "mov", "mkv"}],
        key=lambda path: path.stat().st_mtime,
    )
    if video_files:
        return str(video_files[-1])
    return 0


def _is_image_source(source: str | int) -> bool:
    if not isinstance(source, str):
        return False
    return Path(source).suffix.lower().lstrip(".") in {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _open_capture(source: str | int) -> cv2.VideoCapture:
    return cv2.VideoCapture(source)


def _placeholder_frame(message: str) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, message, (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _source_frame_iterator(source: str | int) -> Iterator[np.ndarray]:
    if _is_image_source(source):
        image = cv2.imread(str(source))
        if image is None:
            image = _placeholder_frame("Unable to load uploaded image")
        while True:
            yield image.copy()
        return

    capture = _open_capture(source)
    if not capture.isOpened():
        while True:
            yield _placeholder_frame("No video source available")
        return

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        yield frame
    capture.release()


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_FRAME_DIM:
        return frame
    scale = MAX_FRAME_DIM / max_dim
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def generate_frames() -> Iterator[bytes]:
    for frame in _source_frame_iterator(active_source):
        try:
            frame = _resize_frame(frame)
            result = navigator.process_frame(frame)
            encoded = cv2.imencode(".jpg", result.frame)
            if not encoded[0]:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded[1].tobytes() + b"\r\n"
            )
        except Exception:
            continue


navigator = AdaptiveNavigator()
active_source: str | int = _default_source()


def create_app() -> Flask:
    app = Flask(__name__)

    overview_template = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Perception Intelligence for Autonomous Vehicles using Monocular Vision and Deep Learning</title>
      <style>
        :root {
          font-size: 12px;
          color-scheme: light;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          --orange: #ff8c00;
          --orange-light: #ffa500;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          min-height: 100vh;
          color: #1a1a1a;
          background: #ffffff;
          line-height: 1.7;
        }
        .page {
          max-width: 1400px;
          margin: 0 auto;
          padding: 24px 20px 40px;
        }
        .back-link {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          color: var(--orange);
          text-decoration: none;
          font-size: 0.75rem;
          font-weight: 600;
          margin-bottom: 20px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          border: 1.5px solid var(--orange);
          padding: 5px 12px;
          border-radius: 8px;
          background: transparent;
          transition: background 0.2s, color 0.2s;
        }
        .back-link:hover {
          background: var(--orange);
          color: #fff;
        }
        h1 {
          font-size: clamp(1.1rem, 2.2vw, 1.6rem);
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: var(--orange);
          margin-bottom: 6px;
        }
        .subtitle {
          color: #000;
          font-size: 1.2rem;
          font-weight: 700;
          margin-bottom: 24px;
          max-width: 720px;
        }
        .objectives-table {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }
        .objective-cell {
          border: 2px solid #ddd;
          border-radius: 14px;
          padding: 20px 18px;
          background: #fff;
          transition: border-color 0.3s, box-shadow 0.3s;
        }
        .objective-cell:hover {
          border-color: var(--orange);
          box-shadow: 0 4px 24px rgba(255, 140, 0, 0.15);
        }
        .objective-cell h2 {
          font-size: 0.9rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--orange);
          margin-bottom: 10px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .objective-cell h2 .num {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
          border-radius: 8px;
          background: var(--orange);
          color: #fff;
          font-size: 0.72rem;
          font-weight: 700;
          flex: 0 0 auto;
        }
        .objective-cell p {
          color: #1a1a1a;
          font-size: 0.85rem;
          line-height: 1.7;
        }
        .objective-cell {
          cursor: pointer;
        }
        .steps-toggle {
          display: inline-block;
          margin-top: 10px;
          font-size: 0.72rem;
          font-weight: 600;
          color: var(--orange);
          letter-spacing: 0.04em;
          text-transform: uppercase;
          border: 1px solid var(--orange);
          padding: 5px 12px;
          border-radius: 6px;
          background: transparent;
          transition: background 0.2s, color 0.2s;
          cursor: pointer;
        }
        .steps-toggle:hover {
          background: var(--orange);
          color: #fff;
        }
        .steps-tree {
          display: none;
          margin-top: 12px;
          padding: 12px 14px;
          border-radius: 10px;
          background: #2a2a2a;
          border: 1px solid #444;
          font-size: 0.82rem;
          line-height: 1.7;
          color: #fff;
        }
        .steps-tree.open {
          display: block;
        }
        .tree-item {
          padding: 2px 0;
        }
        .tree-line {
          color: #888;
          margin-right: 4px;
        }
        @media (max-width: 1200px) {
          .objectives-table { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 900px) {
          .objectives-table { grid-template-columns: 1fr; }
        }
        @media (max-width: 640px) {
          .page { padding: 24px 16px 40px; }
        }
        .workflow-cell:hover {
          border-color: var(--orange) !important;
          box-shadow: 0 4px 20px rgba(255, 140, 0, 0.12);
        }

      </style>
    </head>
    <body>
      <div class="page">
        
        <h1>Perception Intelligence for Autonomous Vehicles using Monocular Vision and Deep Learning</h1>
        <p class="subtitle">Project Overview</p>

        <div class="objectives-table">
          <div class="objective-cell" onclick="window.location.href='{{ url_for('objective1') }}'">
            <h2><span class="num">1</span> Objective 1</h2>
            <p>To develop an intelligent autonomous vehicle navigation framework for obstacle detection and environmental understanding using monocular vision and deep learning techniques. The proposed framework focuses on detecting static and dynamic obstacles, improving path awareness, and supporting adaptive navigation under complex traffic environments. Further, intelligent perception mechanisms are incorporated to improve navigation reliability and real-time vehicle response during uncertain driving conditions.</p>
            <div class="steps-toggle" style="pointer-events:none;">Open Pipeline &rarr;</div>
          </div>
          <div class="objective-cell" onclick="window.location.href='{{ url_for('objective2') }}'">
            <h2><span class="num">2</span> Objective 2</h2>
            <p>To design a vision-based lane and vehicle perception framework using YOLOv8-seg and neuro-fuzzy reasoning for accurate lane monitoring and intelligent driving assistance. The framework focuses on improving lane boundary detection, surrounding vehicle perception, and adaptive decision-making under challenging scenarios such as low illumination, occlusion, faded lane markings, and dense traffic conditions. In addition, neuro-fuzzy inference is incorporated to enable context-aware and human-like navigation support.</p>
            <div class="steps-toggle" style="pointer-events:none;">Open Pipeline &rarr;</div>
          </div>
          <div class="objective-cell" onclick="window.location.href='{{ url_for('objective3') }}'">
            <h2><span class="num">3</span> Objective 3</h2>
            <p>To develop a unified perception intelligence framework that integrates monocular depth estimation, geometric computation, contextual understanding, and neuro-fuzzy decision systems for real-time autonomous driving applications. The proposed framework focuses on improving explainable decision-making, uncertainty handling, and perception reliability using cost-effective monocular vision systems while supporting adaptive and safe autonomous navigation in dynamic road environments.</p>
            <div class="steps-toggle" style="pointer-events:none;">Open Pipeline &rarr;</div>
          </div>
          <div class="objective-cell workflow-cell" onclick="window.location.href='{{ url_for('complete_workflow') }}'">
            <h2><span class="num">&#8635;</span> Complete Workflow</h2>
            <p>End-to-end unified perception intelligence pipeline combining obstacle detection, lane monitoring, depth estimation, geometric computation, contextual understanding, and neuro-fuzzy decision making for autonomous vehicle navigation. This integrates all three objectives into a single real-time autonomous driving system.</p>
            <div class="steps-toggle">Open Workflow &rarr;</div>
          </div>
        </div>


      </div>
      <script>
        function toggleSteps(id, cell) {
          var el = document.getElementById(id);
          var btn = cell.querySelector('.steps-toggle');
          if (el.classList.contains('open')) {
            el.classList.remove('open');
            btn.textContent = 'View Steps';
          } else {
            el.classList.add('open');
            btn.textContent = 'Hide Steps';
          }
        }
      </script>
    </body>
    </html>
    """

    @app.route("/overview")
    def overview():
        return render_template_string(overview_template)

    @app.route("/")
    def index():
        return render_template_string(overview_template)

    @app.route("/upload", methods=["POST"])
    def upload():
        global active_source
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return redirect(url_for("overview"))
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(url_for("overview"))

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/telemetry")
    def telemetry():
        return jsonify(navigator.get_telemetry())

    obj1_template = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Objective 1 - Obstacle Detection & Adaptive Navigation</title>
      <style>
        :root {
          font-size: 12px;
          color-scheme: light;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          --orange: #ff8c00;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { min-height: 100vh; color: #1a1a1a; background: #fff; line-height: 1.6; }
        .page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }
        .back-link {
          display: inline-flex; align-items: center; gap: 6px;
          color: var(--orange); text-decoration: none; font-size: 0.88rem; font-weight: 600;
          margin-bottom: 24px; letter-spacing: 0.04em; text-transform: uppercase;
          border: 2px solid var(--orange); padding: 8px 16px; border-radius: 10px;
          transition: background 0.2s, color 0.2s;
        }
        .back-link:hover { background: var(--orange); color: #fff; }
        h1 { font-size: 1.6rem; color: var(--orange); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
        .subtitle { color: #555; font-size: 1rem; margin-bottom: 28px; }
        .layout { display: grid; grid-template-columns: 1fr 340px; gap: 24px; align-items: stretch; }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s; height: calc(100% - 60px);
        }
        .video-panel img { width: 100%; height: 100%; display: block; object-fit: cover; }
        .video-badge {
          position: absolute; left: 14px; top: 14px;
          padding: 7px 14px; border-radius: 999px;
          background: rgba(255,140,0,0.85); color: #fff;
          font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        }
        .upload-row { display: flex; gap: 10px; margin-top: 14px; align-items: center; flex-wrap: wrap; }
        .upload-row input[type="file"] { display: none; }
        .upload-row button {
          padding: 10px 18px; border-radius: 10px; border: 0; cursor: pointer;
          background: var(--orange); color: #fff; font-weight: 700; font-size: 0.88rem;
          transition: opacity 0.2s;
        }
        .upload-row button:hover { opacity: 0.85; }
        .upload-hint { color: #888; font-size: 0.85rem; }
        .pipeline-panel {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 14px;
          align-items: stretch;
        }
        .pipeline-panel .step-arrow {
          display: none;
        }
        .step-card {
          display: flex;
          flex-direction: column;
        }
        .step-card .step-detail {
          margin-top: auto;
        }
        .step-card:last-child {
          grid-column: 1 / -1;
        }
        .step-card {
          border: 2px solid #ddd; border-radius: 14px; padding: 18px 20px;
          background: #fff; transition: border-color 0.3s, box-shadow 0.3s;
        }
        .step-card.active {
          border-color: var(--orange);
          box-shadow: 0 2px 16px rgba(255,140,0,0.12);
        }
        .step-card h3 {
          font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em;
          color: var(--orange); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
        }
        .step-card h3 .step-num {
          width: 26px; height: 26px; border-radius: 8px; background: var(--orange);
          color: #fff; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.75rem; font-weight: 700; flex: 0 0 auto;
        }
        .step-card .step-status {
          font-size: 0.88rem; font-weight: 600; color: #1a1a1a; margin-bottom: 4px;
        }
        .step-card .step-detail {
          font-size: 0.8rem; color: #666; line-height: 1.5;
        }
        .step-arrow {
          display: flex; justify-content: center; padding: 6px 0; color: var(--orange); font-size: 1rem; font-weight: 700;
        }
        @keyframes pulse-border {
          0% { border-color: #ddd; }
          50% { border-color: var(--orange); box-shadow: 0 0 12px rgba(255,140,0,0.15); }
          100% { border-color: #ddd; }
        }
        .step-card.processing { animation: pulse-border 1s ease-in-out infinite; }
        @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
      </style>
    </head>
    <body>
      <div class="page">
        <a class="back-link" href="{{ url_for('overview') }}">&larr; Back to Overview</a>
        <h1>Objective 1</h1>
        <p class="subtitle">Obstacle Detection &bull; Environmental Understanding &bull; Adaptive Navigation</p>

        <div class="layout">
          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('obj1_video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('obj1_upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*">
                <button type="button" id="load-btn">Load Video</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video to start processing</span>
            </div>
          </div>

          <div class="pipeline-panel">
            <div class="step-card" id="step1">
              <h3><span class="step-num">1</span> Obstacle Detection</h3>
              <div class="step-status" id="s1-status">0 detections</div>
            </div>
            <div class="step-card" id="step2">
              <h3><span class="step-num">2</span> Environmental Understanding</h3>
              <div class="step-status" id="s2-status">Waiting...</div>
            </div>
            <div class="step-card" id="step4">
              <h3><span class="step-num">3</span> Neuro-Fuzzy Decision</h3>
              <div class="step-status" id="s4-status">Waiting...</div>
            </div>
            <div class="step-card" id="step5">
              <h3><span class="step-num">4</span> Autonomous Ground Vehicle Navigation</h3>
              <div class="step-status" id="s5-status">Waiting...</div>
            </div>
            <div class="step-card" id="step3">
              <h3><span class="step-num">5</span> Adaptive Navigation</h3>
              <div class="step-status" id="s3-status">Waiting...</div>
            </div>
          </div>
        </div>
      </div>
      <script>
        var loadBtn = document.getElementById("load-btn");
        var videoInput = document.getElementById("video-input");
        var hint = document.getElementById("upload-hint");
        loadBtn.addEventListener("click", function() { videoInput.click(); });
        videoInput.addEventListener("change", function() {
          if (!videoInput.files || !videoInput.files.length) return;
          hint.textContent = "Uploading " + videoInput.files[0].name + "...";
          document.getElementById("upload-form").submit();
        });

        function fmt(v, d) { var n = Number(v); return Number.isFinite(n) ? n.toFixed(d || 1) : "0.0"; }

        function refreshObj1() {
          fetch("{{ url_for('obj1_telemetry') }}").then(function(r){ return r.json(); }).then(function(p) {
            var detections = p.raw_detections || 0;
            var closest = p.closest_obstacle || "None";
            var risk = p.scene ? p.scene.risk_score : 0;
            var mode = p.scene ? p.scene.driving_mode : "normal";
            var steer = p.steering_angle || 0;
            var speed = p.speed_control || 0;
            var brake = (p.vehicle_status && p.vehicle_status.brake_active) ? "ON" : "OFF";
            var dev = p.lane_deviation || 0;

            document.getElementById("s1-status").textContent = detections + " detections | Nearest: " + closest;

            document.getElementById("s2-status").textContent = "Risk " + fmt(risk) + " | Lighting " + fmt(p.lighting_score, 0) + " | Visibility " + fmt(p.scene ? p.scene.visibility_score : 0, 0);

            var topRules = (p.top_rules || []).map(function(r) { return r.rule || r; }).slice(0, 2).join(", ");
            document.getElementById("s4-status").textContent = (p.active_rule || "---") + " | Confidence " + fmt(p.decision_confidence, 2);

            document.getElementById("s5-status").textContent = mode.charAt(0).toUpperCase() + mode.slice(1) + " | " + (p.scene ? p.scene.recommendation : "---");

            document.getElementById("s3-status").textContent = "Steering " + fmt(steer) + " degrees | Speed " + fmt(speed, 0) + " percent | Brake " + brake;

            var cards = ["step1", "step2", "step3", "step4", "step5"];
            cards.forEach(function(c) { document.getElementById(c).classList.remove("active", "processing"); });
            if (detections > 0) { document.getElementById("step1").classList.add("active", "processing"); }
            if (risk > 0) { document.getElementById("step2").classList.add("active", "processing"); }
            if (p.decision_confidence > 0) { document.getElementById("step4").classList.add("active", "processing"); }
            if (mode !== "normal") { document.getElementById("step5").classList.add("active", "processing"); }
            if (steer !== 0 || speed > 0) { document.getElementById("step3").classList.add("active", "processing"); }
          }).catch(function(){});
        }
        refreshObj1();
        setInterval(refreshObj1, 1000);
      </script>
    </body>
    </html>
    """

    @app.route("/objective1")
    def objective1():
        return render_template_string(obj1_template)

    @app.route("/objective1/upload", methods=["POST"])
    def obj1_upload():
        global active_source
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return redirect(url_for("objective1"))
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(url_for("objective1"))

    def obj1_generate_frames() -> Iterator[bytes]:
        for frame in _source_frame_iterator(active_source):
            try:
                frame = _resize_frame(frame)
                result = navigator.process_frame_obj1(frame)
                encoded = cv2.imencode(".jpg", result.frame)
                if not encoded[0]:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded[1].tobytes() + b"\r\n"
                )
            except Exception:
                continue

    @app.route("/objective1/video_feed")
    def obj1_video_feed():
        return Response(obj1_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/objective1/telemetry")
    def obj1_telemetry():
        return jsonify(navigator.get_telemetry())

    obj2_template = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Objective 2 - Lane Monitoring & Vehicle Perception</title>
      <style>
        :root {
          font-size: 12px;
          color-scheme: light;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          --orange: #ff8c00;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { min-height: 100vh; color: #1a1a1a; background: #fff; line-height: 1.6; }
        .page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }
        .back-link {
          display: inline-flex; align-items: center; gap: 6px;
          color: var(--orange); text-decoration: none; font-size: 0.88rem; font-weight: 600;
          margin-bottom: 24px; letter-spacing: 0.04em; text-transform: uppercase;
          border: 2px solid var(--orange); padding: 8px 16px; border-radius: 10px;
          transition: background 0.2s, color 0.2s;
        }
        .back-link:hover { background: var(--orange); color: #fff; }
        h1 { font-size: 1.6rem; color: var(--orange); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
        .subtitle { color: #555; font-size: 1rem; margin-bottom: 28px; }
        .layout { display: grid; grid-template-columns: 1fr 340px; gap: 24px; align-items: stretch; }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s; height: calc(100% - 60px);
        }
        .video-panel img { width: 100%; height: 100%; display: block; object-fit: cover; }
        .video-badge {
          position: absolute; left: 14px; top: 14px;
          padding: 7px 14px; border-radius: 999px;
          background: rgba(255,140,0,0.85); color: #fff;
          font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        }
        .upload-row { display: flex; gap: 10px; margin-top: 14px; align-items: center; flex-wrap: wrap; }
        .upload-row input[type="file"] { display: none; }
        .upload-row button {
          padding: 10px 18px; border-radius: 10px; border: 0; cursor: pointer;
          background: var(--orange); color: #fff; font-weight: 700; font-size: 0.88rem;
        }
        .upload-row button:hover { opacity: 0.85; }
        .upload-hint { color: #888; font-size: 0.85rem; }
        .pipeline-panel {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 14px;
          align-items: stretch;
        }
        .step-card {
          border: 2px solid #ddd; border-radius: 14px; padding: 18px 20px;
          background: #fff; transition: border-color 0.3s, box-shadow 0.3s;
          display: flex; flex-direction: column; justify-content: center;
        }
        .step-card.active {
          border-color: var(--orange);
          box-shadow: 0 2px 16px rgba(255,140,0,0.12);
        }
        .step-card h3 {
          font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
          color: var(--orange); margin-bottom: 6px; display: flex; align-items: center; gap: 8px;
        }
        .step-card h3 .step-num {
          width: 26px; height: 26px; border-radius: 8px; background: var(--orange);
          color: #fff; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.75rem; font-weight: 700; flex: 0 0 auto;
        }
        .step-card .step-status {
          font-size: 0.88rem; font-weight: 600; color: #1a1a1a;
        }
        .pipeline-panel .step-arrow { display: none; }
        @keyframes pulse-border {
          0% { border-color: #ddd; }
          50% { border-color: var(--orange); box-shadow: 0 0 12px rgba(255,140,0,0.15); }
          100% { border-color: #ddd; }
        }
        .step-card.processing { animation: pulse-border 1s ease-in-out infinite; }
        @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
      </style>
    </head>
    <body>
      <div class="page">
        <a class="back-link" href="{{ url_for('overview') }}">&larr; Back to Overview</a>
        <h1>Objective 2</h1>
        <p class="subtitle">Lane Monitoring &bull; Vehicle Perception &bull; Driving Assistance &bull; Challenging Road Conditions</p>

        <div class="layout">
          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('obj2_video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('obj2_upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*">
                <button type="button" id="load-btn">Load Video</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video to start processing</span>
            </div>
          </div>

          <div class="pipeline-panel">
            <div class="step-card" id="step1">
              <h3><span class="step-num">1</span> Lane Monitoring</h3>
              <div class="step-status" id="s1-status">Waiting...</div>
            </div>
            <div class="step-card" id="step2">
              <h3><span class="step-num">2</span> Vehicle Perception</h3>
              <div class="step-status" id="s2-status">Waiting...</div>
            </div>
            <div class="step-card" id="step3">
              <h3><span class="step-num">3</span> YOLOv8-Seg</h3>
              <div class="step-status" id="s3-status">Waiting...</div>
            </div>
            <div class="step-card" id="step4">
              <h3><span class="step-num">4</span> ANFIS</h3>
              <div class="step-status" id="s4-status">Waiting...</div>
            </div>
            <div class="step-card" id="step5">
              <h3><span class="step-num">5</span> Driving Assistance</h3>
              <div class="step-status" id="s5-status">Waiting...</div>
            </div>
            <div class="step-card" id="step6">
              <h3><span class="step-num">6</span> Challenging Road Conditions</h3>
              <div class="step-status" id="s6-status">Waiting...</div>
            </div>
          </div>
        </div>
      </div>
      <script>
        var loadBtn = document.getElementById("load-btn");
        var videoInput = document.getElementById("video-input");
        var hint = document.getElementById("upload-hint");
        loadBtn.addEventListener("click", function() { videoInput.click(); });
        videoInput.addEventListener("change", function() {
          if (!videoInput.files || !videoInput.files.length) return;
          hint.textContent = "Uploading " + videoInput.files[0].name + "...";
          document.getElementById("upload-form").submit();
        });

        function fmt(v, d) { var n = Number(v); return Number.isFinite(n) ? n.toFixed(d || 1) : "0.0"; }

        function refreshObj2() {
          fetch("{{ url_for('obj2_telemetry') }}").then(function(r){ return r.json(); }).then(function(p) {
            var scene = p.scene || {};
            var leftFound = p.lane_deviation !== undefined && p.lane_deviation !== null;
            var detections = p.raw_detections || 0;
            var tracked = p.detections || 0;
            var mode = scene.driving_mode || "normal";
            var risk = scene.risk_score || 0;
            var rule = p.active_rule || "---";

            document.getElementById("s1-status").textContent = "Deviation " + fmt(p.lane_deviation) + " percent | Lane Quality " + fmt(scene.lane_quality, 0);

            document.getElementById("s2-status").textContent = "Lane detected: " + (p.lane_deviation !== undefined ? "Yes" : "No");

            document.getElementById("s3-status").textContent = "Model loaded | Processing frames";

            document.getElementById("s4-status").textContent = "Rule: " + rule;

            document.getElementById("s5-status").textContent = "Steering " + fmt(p.steering_angle) + " degrees | Speed " + fmt(p.speed_control, 0) + " percent";

            var lighting = p.lighting_score || 0;
            var vis = scene.visibility_score || 0;
            document.getElementById("s6-status").textContent = "Lighting " + fmt(lighting, 0) + " | Visibility " + fmt(vis, 0) + " | Risk " + fmt(risk);

            var cards = ["step1", "step2", "step3", "step4", "step5", "step6"];
            cards.forEach(function(c) { document.getElementById(c).classList.remove("active", "processing"); });
            if (leftFound) { document.getElementById("step1").classList.add("active", "processing"); }
            document.getElementById("step2").classList.add("active", "processing");
            document.getElementById("step3").classList.add("active", "processing");
            if (rule !== "---") { document.getElementById("step4").classList.add("active", "processing"); }
            if (p.steering_angle !== 0 || p.speed_control > 0) { document.getElementById("step5").classList.add("active", "processing"); }
            if (lighting < 50 || vis < 50 || risk > 5) { document.getElementById("step6").classList.add("active", "processing"); }
          }).catch(function(){});
        }
        refreshObj2();
        setInterval(refreshObj2, 1000);
      </script>
    </body>
    </html>
    """

    @app.route("/objective2")
    def objective2():
        return render_template_string(obj2_template)

    @app.route("/objective2/upload", methods=["POST"])
    def obj2_upload():
        global active_source
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return redirect(url_for("objective2"))
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(url_for("objective2"))

    @app.route("/objective2/video_feed")
    def obj2_video_feed():
        def obj2_generate_frames() -> Iterator[bytes]:
            for frame in _source_frame_iterator(active_source):
                try:
                    frame = _resize_frame(frame)
                    result = navigator.process_frame(frame, draw_hud=False)
                    encoded = cv2.imencode(".jpg", result.frame)
                    if not encoded[0]:
                        continue
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + encoded[1].tobytes() + b"\r\n"
                    )
                except Exception:
                    continue
        return Response(obj2_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/objective2/telemetry")
    def obj2_telemetry():
        return jsonify(navigator.get_telemetry())

    obj3_template = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Objective 3 - Unified Perception Intelligence Framework</title>
      <style>
        :root {
          font-size: 12px;
          color-scheme: light;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          --orange: #ff8c00;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { min-height: 100vh; color: #1a1a1a; background: #fff; line-height: 1.6; }
        .page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }
        .back-link {
          display: inline-flex; align-items: center; gap: 6px;
          color: var(--orange); text-decoration: none; font-size: 0.88rem; font-weight: 600;
          margin-bottom: 24px; letter-spacing: 0.04em; text-transform: uppercase;
          border: 2px solid var(--orange); padding: 8px 16px; border-radius: 10px;
          transition: background 0.2s, color 0.2s;
        }
        .back-link:hover { background: var(--orange); color: #fff; }
        h1 { font-size: 1.6rem; color: var(--orange); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
        .subtitle { color: #555; font-size: 1rem; margin-bottom: 28px; }
        .layout { display: grid; grid-template-columns: 1fr 340px; gap: 24px; align-items: stretch; }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s; height: calc(100% - 60px);
        }
        .video-panel img { width: 100%; height: 100%; display: block; object-fit: cover; }
        .video-badge {
          position: absolute; left: 14px; top: 14px;
          padding: 7px 14px; border-radius: 999px;
          background: rgba(255,140,0,0.85); color: #fff;
          font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        }
        .upload-row { display: flex; gap: 10px; margin-top: 14px; align-items: center; flex-wrap: wrap; }
        .upload-row input[type="file"] { display: none; }
        .upload-row button {
          padding: 10px 18px; border-radius: 10px; border: 0; cursor: pointer;
          background: var(--orange); color: #fff; font-weight: 700; font-size: 0.88rem;
        }
        .upload-row button:hover { opacity: 0.85; }
        .upload-hint { color: #888; font-size: 0.85rem; }
        .pipeline-panel {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 14px;
          align-items: stretch;
        }
        .step-card {
          border: 2px solid #ddd; border-radius: 14px; padding: 18px 20px;
          background: #fff; transition: border-color 0.3s, box-shadow 0.3s;
          display: flex; flex-direction: column; justify-content: center;
        }
        .step-card.active {
          border-color: var(--orange);
          box-shadow: 0 2px 16px rgba(255,140,0,0.12);
        }
        .step-card h3 {
          font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
          color: var(--orange); margin-bottom: 6px; display: flex; align-items: center; gap: 8px;
        }
        .step-card h3 .step-num {
          width: 26px; height: 26px; border-radius: 8px; background: var(--orange);
          color: #fff; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.75rem; font-weight: 700; flex: 0 0 auto;
        }
        .step-card .step-status {
          font-size: 0.88rem; font-weight: 600; color: #1a1a1a;
        }
        .pipeline-panel .step-arrow { display: none; }
        @keyframes pulse-border {
          0% { border-color: #ddd; }
          50% { border-color: var(--orange); box-shadow: 0 0 12px rgba(255,140,0,0.15); }
          100% { border-color: #ddd; }
        }
        .step-card.processing { animation: pulse-border 1s ease-in-out infinite; }
        @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
      </style>
    </head>
    <body>
      <div class="page">
        <a class="back-link" href="{{ url_for('overview') }}">&larr; Back to Overview</a>
        <h1>Objective 3</h1>
        <p class="subtitle">Unified Perception Intelligence Framework &bull; Depth Estimation &bull; Geometric Computation &bull; Context-Aware Perception &bull; Spatial Understanding &bull; Uncertainty Handling &bull; Neuro-Fuzzy Decision Systems</p>

        <div class="layout">
          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('obj3_video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('obj3_upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*">
                <button type="button" id="load-btn">Load Video</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video to start processing</span>
            </div>
          </div>

          <div class="pipeline-panel">
            <div class="step-card" id="step1">
              <h3><span class="step-num">1</span> Monocular Depth Estimation</h3>
              <div class="step-status" id="s1-status">Waiting...</div>
            </div>
            <div class="step-card" id="step2">
              <h3><span class="step-num">2</span> Geometric Computation</h3>
              <div class="step-status" id="s2-status">Waiting...</div>
            </div>
            <div class="step-card" id="step3">
              <h3><span class="step-num">3</span> Context-Aware Perception</h3>
              <div class="step-status" id="s3-status">Waiting...</div>
            </div>
            <div class="step-card" id="step4">
              <h3><span class="step-num">4</span> Spatial Understanding</h3>
              <div class="step-status" id="s4-status">Waiting...</div>
            </div>
            <div class="step-card" id="step5">
              <h3><span class="step-num">5</span> Uncertainty Handling</h3>
              <div class="step-status" id="s5-status">Waiting...</div>
            </div>
            <div class="step-card" id="step6">
              <h3><span class="step-num">6</span> Neuro-Fuzzy Decision Systems</h3>
              <div class="step-status" id="s6-status">Waiting...</div>
            </div>
          </div>
        </div>
      </div>
      <script>
        var loadBtn = document.getElementById("load-btn");
        var videoInput = document.getElementById("video-input");
        var hint = document.getElementById("upload-hint");
        loadBtn.addEventListener("click", function() { videoInput.click(); });
        videoInput.addEventListener("change", function() {
          if (!videoInput.files || !videoInput.files.length) return;
          hint.textContent = "Uploading " + videoInput.files[0].name + "...";
          document.getElementById("upload-form").submit();
        });

        function fmt(v, d) { var n = Number(v); return Number.isFinite(n) ? n.toFixed(d || 1) : "0.0"; }

        function refreshObj3() {
          fetch("{{ url_for('obj3_telemetry') }}").then(function(r){ return r.json(); }).then(function(p) {
            var scene = p.scene || {};
            var depth_conf = p.depth_confidence || 0;
            var decision_conf = p.decision_confidence || 0;
            var active_rule = p.active_rule || "---";

            document.getElementById("s1-status").textContent = "Depth confidence " + fmt(depth_conf, 2) + " | " + (p.depth_backend || "none");

            document.getElementById("s2-status").textContent = "Obstacle distance " + fmt(p.obstacle_distance, 1) + "m | Bearing " + fmt(p.obstacle_bearing, 1) + "°";

            document.getElementById("s3-status").textContent = "Lighting " + fmt(p.lighting_score, 0) + " | Traffic density " + fmt(p.traffic_density, 1);

            document.getElementById("s4-status").textContent = "Detections " + (p.raw_detections || 0) + " | Tracks " + (p.detections || 0);

            document.getElementById("s5-status").textContent = "Obstacle uncertainty " + fmt(p.obstacle_uncertainty, 2) + " | Scene risk " + fmt(scene.risk_score, 1);

            document.getElementById("s6-status").textContent = active_rule + " | Confidence " + fmt(decision_conf, 2);

            var cards = ["step1", "step2", "step3", "step4", "step5", "step6"];
            cards.forEach(function(c) { document.getElementById(c).classList.remove("active", "processing"); });
            if (p.raw_detections > 0 || depth_conf > 0) { document.getElementById("step1").classList.add("active", "processing"); }
            if (p.obstacle_distance > 0) { document.getElementById("step2").classList.add("active", "processing"); }
            if (p.lighting_score > 0) { document.getElementById("step3").classList.add("active", "processing"); }
            if (p.raw_detections > 0) { document.getElementById("step4").classList.add("active", "processing"); }
            if (p.obstacle_uncertainty > 0 || scene.risk_score > 0) { document.getElementById("step5").classList.add("active", "processing"); }
            if (decision_conf > 0) { document.getElementById("step6").classList.add("active", "processing"); }
          }).catch(function(){});
        }
        refreshObj3();
        setInterval(refreshObj3, 1000);
      </script>
    </body>
    </html>
    """

    @app.route("/objective3")
    def objective3():
        return render_template_string(obj3_template)

    @app.route("/objective3/upload", methods=["POST"])
    def obj3_upload():
        global active_source
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return redirect(url_for("objective3"))
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(url_for("objective3"))

    @app.route("/objective3/video_feed")
    def obj3_video_feed():
        def obj3_generate_frames() -> Iterator[bytes]:
            for frame in _source_frame_iterator(active_source):
                try:
                    frame = _resize_frame(frame)
                    result = navigator.process_frame(frame, draw_hud=False)
                    encoded = cv2.imencode(".jpg", result.frame)
                    if not encoded[0]:
                        continue
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + encoded[1].tobytes() + b"\r\n"
                    )
                except Exception:
                    continue
        return Response(obj3_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/objective3/telemetry")
    def obj3_telemetry():
        return jsonify(navigator.get_telemetry())

    complete_workflow_template = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Complete Workflow - Perception Intelligence for Autonomous Vehicles</title>
      <style>
        :root {
          font-size: 12px;
          color-scheme: light;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          --orange: #ff8c00;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          min-height: 100vh;
          color: #1a1a1a;
          background: #ffffff;
          line-height: 1.6;
        }
        .page {
          max-width: 1400px;
          margin: 0 auto;
          padding: 24px 20px 40px;
        }
        .layout {
          display: grid;
          grid-template-columns: 1fr 340px;
          gap: 24px;
          align-items: start;
        }
        @media (max-width: 800px) {
          .layout { grid-template-columns: 1fr; }
        }
        .back-link {
          display: inline-flex; align-items: center; gap: 4px;
          color: var(--orange); text-decoration: none; font-size: 0.75rem; font-weight: 600;
          margin-bottom: 20px; letter-spacing: 0.04em; text-transform: uppercase;
          border: 1.5px solid var(--orange); padding: 5px 12px; border-radius: 8px;
          background: transparent; transition: background 0.2s, color 0.2s;
        }
        .back-link:hover { background: var(--orange); color: #fff; }
        h1 {
          font-size: clamp(1rem, 2vw, 1.4rem);
          letter-spacing: 0.04em; text-transform: uppercase;
          color: var(--orange); margin-bottom: 4px;
        }
        .subtitle {
          color: #555; font-size: 0.85rem; margin-bottom: 24px; max-width: 720px;
        }
        .section-title {
          font-size: 0.8rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--orange);
          margin: 0 0 10px;
          padding-bottom: 6px;
          border-bottom: 2px solid #eee;
        }
        .decision-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 10px;
          margin-bottom: 14px;
        }
        .decision-chip {
          padding: 10px;
          border-radius: 10px;
          border: 2px solid #ddd;
          background: #fff;
          text-align: center;
          transition: border-color 0.3s;
        }
        .decision-chip:hover {
          border-color: var(--orange);
        }
        .decision-chip .chip-label {
          font-size: 0.65rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: #888;
          margin-bottom: 4px;
        }
        .decision-chip .chip-value {
          font-size: 0.95rem;
          font-weight: 700;
          color: var(--orange);
        }
        .status-info {
          color: #555;
          font-size: 0.95rem;
          line-height: 1.6;
          margin-bottom: 4px;
        }
        .status-info strong {
          color: #1a1a1a;
        }
        .safe-banner {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 10px;
          background: linear-gradient(135deg, #fff8f0, #fff);
          border: 2px solid var(--orange);
          margin-bottom: 12px;
        }
        .safe-dot {
          width: 10px; height: 10px;
          border-radius: 50%;
          background: var(--orange);
          box-shadow: 0 0 10px rgba(255,140,0,0.5);
        }
        .safe-banner p {
          margin: 0;
          font-weight: 600;
          font-size: 0.85rem;
          color: #1a1a1a;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 8px;
          margin-top: 10px;
        }
        .metric {
          padding: 10px;
          border-radius: 10px;
          background: #f9f6f2;
          border: 1px solid #eee;
        }
        .metric span {
          display: block;
          font-size: 0.62rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: #888;
          margin-bottom: 4px;
        }
        .metric strong {
          font-size: 0.82rem;
          color: #1a1a1a;
        }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s;
        }
        .video-panel img { width: 100%; display: block; }
        .video-badge {
          position: absolute; left: 14px; top: 14px;
          padding: 6px 12px; border-radius: 999px;
          background: rgba(255,140,0,0.85); color: #fff;
          font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        }
        .upload-row { display: flex; gap: 8px; margin-top: 10px; align-items: center; flex-wrap: wrap; }
        .upload-row input[type="file"] { display: none; }
        .upload-row button {
          padding: 8px 14px; border-radius: 8px; border: 0; cursor: pointer;
          background: var(--orange); color: #fff; font-weight: 700; font-size: 0.78rem;
          transition: opacity 0.2s;
        }
        .upload-row button:hover { opacity: 0.85; }
        .upload-hint { color: #888; font-size: 0.72rem; }
      </style>
    </head>
    <body>
      <div class="page">
        <a class="back-link" href="{{ url_for('overview') }}">&larr; Back to Overview</a>

        <h1>Complete Workflow</h1>

        <div class="layout">
          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*">
                <button type="button" id="load-btn">Load Video</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video to start processing</span>
            </div>
          </div>

          <div>
            <h2 class="section-title">Final Decision</h2>
            <div class="decision-grid">
              <div class="decision-chip">
                <div class="chip-label">Steering</div>
                <div class="chip-value" id="steering-value">0.0&deg;</div>
              </div>
              <div class="decision-chip">
                <div class="chip-label">Brake</div>
                <div class="chip-value" id="brake-value">OFF</div>
              </div>
              <div class="decision-chip">
                <div class="chip-label">Throttle</div>
                <div class="chip-value" id="throttle-value">0.0%</div>
              </div>
            </div>
            <p class="status-info">Active rule: <strong id="active-rule">Analyzing...</strong></p>
            <p class="status-info">Decision confidence: <strong id="decision-confidence">0.00</strong></p>

            <h2 class="section-title" style="margin-top: 20px;">Autonomous Vehicle Movement</h2>
            <div class="safe-banner">
              <div class="safe-dot"></div>
              <p id="safe-status">Safe Navigation Status</p>
            </div>
            <p class="status-info" id="scene-recommendation">Waiting for telemetry...</p>
            <div class="metric-grid">
              <div class="metric"><span>Lane Deviation</span><strong id="lane-deviation">0.0%</strong></div>
              <div class="metric"><span>Obstacle Distance</span><strong id="obstacle-distance">0.0</strong></div>
              <div class="metric"><span>Scene Risk</span><strong id="scene-risk">0.0</strong></div>
              <div class="metric"><span>Driving Mode</span><strong id="driving-mode">normal</strong></div>
            </div>
          </div>
        </div>
      </div>

      <script>
        function formatNumber(value, digits) {
          digits = digits || 1;
          var number = Number(value);
          return Number.isFinite(number) ? number.toFixed(digits) : "0.0";
        }
        function getBrakeState(vehicleStatus, scene) {
          if ((scene && scene.driving_mode === "emergency") || (vehicleStatus && vehicleStatus.brake_active)) return "ON";
          return "OFF";
        }
        function getStatusLabel(scene) {
          if (!scene) return "Safe Navigation Status";
          if (scene.driving_mode === "emergency") return "Emergency Stop";
          if (scene.driving_mode === "cautious") return "Cautious Navigation";
          if (scene.driving_mode === "degraded") return "Degraded Visibility";
          return "Safe Navigation Status";
        }
        function renderTelemetry(p) {
          var scene = p.scene || {};
          var status = p.vehicle_status || {};
          document.getElementById("steering-value").innerHTML = formatNumber(p.steering_angle, 1) + "&deg;";
          document.getElementById("brake-value").textContent = getBrakeState(status, scene);
          document.getElementById("throttle-value").textContent = formatNumber(p.speed_control, 1) + "%";
          document.getElementById("active-rule").textContent = p.active_rule || "Analyzing...";
          document.getElementById("decision-confidence").textContent = formatNumber(p.decision_confidence, 2);
          document.getElementById("safe-status").textContent = getStatusLabel(scene);
          document.getElementById("scene-recommendation").textContent = scene.recommendation || "Waiting for telemetry...";
          document.getElementById("lane-deviation").textContent = formatNumber(p.lane_deviation, 1) + "%";
          document.getElementById("obstacle-distance").textContent = formatNumber(p.obstacle_distance, 1);
          document.getElementById("scene-risk").textContent = formatNumber(scene.risk_score, 1);
          document.getElementById("driving-mode").textContent = scene.driving_mode || "normal";

        }
        var videoInput = document.getElementById("video-input");
        var uploadHint = document.getElementById("upload-hint");
        document.getElementById("load-btn").addEventListener("click", function() { videoInput.click(); });
        videoInput.addEventListener("change", function() {
          if (!videoInput.files || !videoInput.files.length) return;
          uploadHint.textContent = "Uploading " + videoInput.files[0].name + "...";
          document.getElementById("upload-form").submit();
        });
        function refreshTelemetry() {
          fetch("{{ url_for('telemetry') }}").then(function(r){ return r.json(); }).then(function(payload) {
            renderTelemetry(payload);
          }).catch(function(e) {});
        }
        refreshTelemetry();
        setInterval(refreshTelemetry, 1000);
      </script>
    </body>
    </html>
    """

    @app.route("/complete-workflow")
    def complete_workflow():
        return render_template_string(complete_workflow_template)

    return app


app = create_app()


def main() -> None:
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
