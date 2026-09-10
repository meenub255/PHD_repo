"""
Flask web application for the Autonomous Vehicle Perception System.

Provides routes for the project overview, individual objective pipelines
(obstacle detection, lane monitoring, unified perception), and a complete
end-to-end workflow page.  Each objective page streams processed video
frames and exposes a JSON telemetry endpoint consumed by the front-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from ..config.settings import (
    ALLOWED_EXTENSIONS,
    SERVER_HOST,
    SERVER_PORT,
    UPLOAD_FOLDER,
    MAX_FRAME_DIM,
)
from ..navigation.adaptive_navigator import AdaptiveNavigator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = ROOT_DIR / UPLOAD_FOLDER
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_source() -> str | int:
    """Return the most recently uploaded video, or webcam ``0``."""
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
    """Check whether *filename* has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _open_capture(source: str | int) -> cv2.VideoCapture:
    return cv2.VideoCapture(source)


def _placeholder_frame(message: str) -> np.ndarray:
    """Generate a black 720p frame with a centred text message."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, message, (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _source_frame_iterator(source: str | int) -> Iterator[np.ndarray]:
    """Yield frames from a video file, image, or camera, looping video sources."""
    if _is_image_source(source):
        image = cv2.imread(str(source))
        if image is None:
            image = _placeholder_frame("Unable to load uploaded image")
        while True:
            if str(source) != str(active_source):
                return
            yield image.copy()
        return

    while True:
        capture = _open_capture(source)
        if not capture.isOpened():
            while True:
                if str(source) != str(active_source):
                    return
                yield _placeholder_frame("No video source available")
            return

        while capture.isOpened():
            if str(source) != str(active_source):
                capture.release()
                return
            success, frame = capture.read()
            if not success:
                break
            yield frame
        capture.release()


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    """Down-scale *frame* so its largest dimension ≤ ``MAX_FRAME_DIM``."""
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_FRAME_DIM:
        return frame
    scale = MAX_FRAME_DIM / max_dim
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

navigator = AdaptiveNavigator()
active_source: str | int = _default_source()


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

def generate_frames() -> Iterator[bytes]:
    """Full pipeline frame generator (complete workflow)."""
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


def _obj1_generate_frames() -> Iterator[bytes]:
    """Objective-1 frame generator (obstacle detection only)."""
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


def _obj_generate_frames_no_hud() -> Iterator[bytes]:
    """Frame generator for objectives 2 & 3 (no HUD overlay)."""
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Create and configure the Flask application."""
    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    # ----- Overview / Home --------------------------------------------------

    @app.route("/")
    def index():
        return render_template("overview.html")

    @app.route("/overview")
    def overview():
        return render_template("overview.html")

    # ----- Demo video (range-request aware) ---------------------------------

    @app.route("/demo_video")
    def demo_video():
        demo_path = UPLOAD_DIR / "demo_final.mp4"
        if not demo_path.exists():
            return "No demo video", 404
        file_size = demo_path.stat().st_size
        range_header = request.headers.get("Range")
        if range_header:
            byte_start = 0
            byte_end = file_size - 1
            match = __import__("re").match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                byte_start = int(match.group(1))
                if match.group(2):
                    byte_end = int(match.group(2))
            length = byte_end - byte_start + 1

            def generate():
                with open(demo_path, "rb") as f:
                    f.seek(byte_start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            response = Response(generate(), status=206, mimetype="video/mp4")
            response.headers["Content-Range"] = f"bytes {byte_start}-{byte_end}/{file_size}"
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Content-Length"] = length
            return response

        def generate_full():
            with open(demo_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    yield chunk

        response = Response(generate_full(), status=200, mimetype="video/mp4")
        response.headers["Content-Length"] = file_size
        response.headers["Accept-Ranges"] = "bytes"
        return response

    # ----- Global upload / feeds / telemetry --------------------------------

    @app.route("/upload", methods=["POST"])
    @app.route("/complete-workflow/upload", methods=["POST"])
    def upload():
        global active_source
        uploaded = request.files.get("file")
        target = request.referrer or url_for("complete_workflow")
        if not uploaded or uploaded.filename == "":
            return redirect(target)
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(target)

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/telemetry")
    def telemetry():
        return jsonify(navigator.get_telemetry())

    # ----- Objective 1 ------------------------------------------------------

    @app.route("/objective1")
    def objective1():
        return render_template("objective1.html")

    @app.route("/objective1/upload", methods=["POST"])
    def obj1_upload():
        global active_source
        uploaded = request.files.get("file")
        target = request.referrer or url_for("objective1")
        if not uploaded or uploaded.filename == "":
            return redirect(target)
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(target)

    @app.route("/objective1/video_feed")
    def obj1_video_feed():
        return Response(_obj1_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/objective1/telemetry")
    def obj1_telemetry():
        return jsonify(navigator.get_telemetry())

    # ----- Objective 2 ------------------------------------------------------

    @app.route("/objective2")
    def objective2():
        return render_template("objective2.html")

    @app.route("/objective2/upload", methods=["POST"])
    def obj2_upload():
        global active_source
        uploaded = request.files.get("file")
        target = request.referrer or url_for("objective2")
        if not uploaded or uploaded.filename == "":
            return redirect(target)
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(target)

    @app.route("/objective2/video_feed")
    def obj2_video_feed():
        return Response(_obj_generate_frames_no_hud(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/objective2/telemetry")
    def obj2_telemetry():
        return jsonify(navigator.get_telemetry())

    # ----- Objective 3 ------------------------------------------------------

    @app.route("/objective3")
    def objective3():
        return render_template("objective3.html")

    @app.route("/objective3/upload", methods=["POST"])
    def obj3_upload():
        global active_source
        uploaded = request.files.get("file")
        target = request.referrer or url_for("objective3")
        if not uploaded or uploaded.filename == "":
            return redirect(target)
        if not allowed_file(uploaded.filename):
            return jsonify({"error": "unsupported file type"}), 400
        destination = UPLOAD_DIR / Path(uploaded.filename).name
        uploaded.save(destination)
        active_source = str(destination)
        return redirect(target)

    @app.route("/objective3/video_feed")
    def obj3_video_feed():
        return Response(_obj_generate_frames_no_hud(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/objective3/telemetry")
    def obj3_telemetry():
        return jsonify(navigator.get_telemetry())

    # ----- Complete Workflow ------------------------------------------------

    @app.route("/complete-workflow")
    def complete_workflow():
        return render_template("complete_workflow.html")

    return app


# ---------------------------------------------------------------------------
# Module-level app instance & entry point
# ---------------------------------------------------------------------------

app = create_app()


def main() -> None:
    """Start the development server."""
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
