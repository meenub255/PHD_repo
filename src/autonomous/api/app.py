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

    overview_template = r"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Perception Intelligence for Autonomous Vehicles using Monocular Vision and Deep Learning</title>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <style>
        :root {
          font-size: 12px;
          color-scheme: light;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          --orange: #ff8c00;
          --orange-glow: rgba(255,140,0,0.4);
          --dark: #0a0a0a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          min-height: 100vh;
          color: #1a1a1a;
          background: #ffffff;
          line-height: 1.7;
          overflow-x: hidden;
        }

        /* ======================== */
        /* BOOT-UP SEQUENCE OVERLAY */
        /* ======================== */
        #boot-overlay {
          position: fixed;
          inset: 0;
          z-index: 9999;
          background: #000;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          transition: opacity 0.4s ease;
        }
        #boot-overlay.done {
          opacity: 0;
          pointer-events: none;
        }
        #boot-reticle {
          width: 120px;
          height: 120px;
          border: 2px solid var(--orange);
          border-radius: 50%;
          position: relative;
          margin-bottom: 30px;
        }
        #boot-reticle::before {
          content: '';
          position: absolute;
          inset: -10px;
          border: 1px dashed rgba(255,140,0,0.5);
          border-radius: 50%;
          animation: reticleSpin 2s linear infinite;
        }
        #boot-reticle::after {
          content: '';
          position: absolute;
          top: 50%; left: 50%;
          width: 0; height: 0;
          background: var(--orange);
          border-radius: 50%;
          transform: translate(-50%,-50%);
          animation: reticlePulse 0.8s ease-in-out infinite alternate;
        }
        #boot-reticle .crosshair {
          position: absolute;
          background: var(--orange);
        }
        #boot-reticle .crosshair.h {
          width: 100%; height: 1px; top: 50%; left: 0;
        }
        #boot-reticle .crosshair.v {
          width: 1px; height: 100%; top: 0; left: 50%;
        }
        @keyframes reticleSpin { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes reticlePulse { from{width:4px;height:4px;opacity:1} to{width:16px;height:16px;opacity:0.6} }

        #boot-terminal {
          font-family: 'Courier New', monospace;
          font-size: 11px;
          color: var(--orange);
          text-align: left;
          width: 380px;
          max-height: 140px;
          overflow: hidden;
          background: rgba(255,140,0,0.03);
          border: 1px solid rgba(255,140,0,0.15);
          border-radius: 8px;
          padding: 12px 16px;
          line-height: 1.8;
        }
        #boot-terminal .line {
          opacity: 0;
          transform: translateX(-10px);
          animation: termLine 0.15s ease forwards;
        }
        #boot-terminal .line.ok { color: #0f0; }
        #boot-terminal .line.warn { color: #ff0; }
        @keyframes termLine { to{opacity:1;transform:translateX(0)} }

        #boot-progress {
          width: 380px;
          height: 3px;
          background: rgba(255,140,0,0.15);
          border-radius: 3px;
          margin-top: 16px;
          overflow: hidden;
        }
        #boot-progress-bar {
          height: 100%;
          width: 0%;
          background: var(--orange);
          border-radius: 3px;
          transition: width 0.1s linear;
        }
        #boot-pct {
          font-family: 'Courier New', monospace;
          font-size: 11px;
          color: var(--orange);
          margin-top: 8px;
        }

        @keyframes reticleExpand {
          0% { transform: scale(0.2); opacity: 1; }
          60% { transform: scale(1.5); opacity: 0.8; }
          100% { transform: scale(3); opacity: 0; }
        }
        #boot-reticle.explode {
          animation: reticleExpand 0.6s ease forwards;
        }

        /* ======================== */
        /* 3D HERO SECTION          */
        /* ======================== */
        .hero {
          position: relative;
          width: 100%;
          height: 420px;
          background: linear-gradient(135deg, #0a0a0a 0%, #1a1008 50%, #0a0a0a 100%);
          border-radius: 18px;
          overflow: hidden;
          margin-bottom: 32px;
          cursor: grab;
        }
        .hero:active { cursor: grabbing; }
        .hero canvas { display: block; width: 100% !important; height: 100% !important; }
        .hero-overlay {
          position: absolute;
          bottom: 0; left: 0; right: 0;
          padding: 30px 28px 24px;
          background: linear-gradient(transparent, rgba(0,0,0,0.85));
          pointer-events: none;
        }
        .hero-overlay h1 {
          font-size: clamp(1.2rem, 2.5vw, 1.8rem);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          color: var(--orange);
          margin-bottom: 6px;
        }
        .hero-overlay .subtitle {
          color: rgba(255,255,255,0.7);
          font-size: 0.95rem;
          font-weight: 600;
        }
        .hero-badge {
          position: absolute;
          top: 16px; left: 16px;
          padding: 6px 14px;
          border-radius: 999px;
          background: rgba(255,140,0,0.15);
          border: 1px solid rgba(255,140,0,0.3);
          color: var(--orange);
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          backdrop-filter: blur(8px);
        }
        .hero-hint {
          position: absolute;
          top: 16px; right: 16px;
          color: rgba(255,255,255,0.4);
          font-size: 0.68rem;
          letter-spacing: 0.04em;
        }

        /* HUD Overlay */
        .hud-overlay {
          position: absolute;
          inset: 0;
          pointer-events: none;
          z-index: 3;
        }
        .hud-corner {
          position: absolute;
          padding: 14px 16px;
        }
        .hud-tl { top: 0; left: 0; }
        .hud-tr { top: 0; right: 0; text-align: right; }
        .hud-label {
          font-family: 'Courier New', monospace;
          font-size: 0.62rem;
          color: rgba(255,140,0,0.6);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 4px;
          line-height: 1.4;
        }
        .hud-label span {
          color: rgba(255,140,0,0.9);
          font-weight: 700;
        }
        .hud-ok { color: #00ff88 !important; }
        #hud-wave {
          display: block;
          margin-top: 4px;
          margin-left: auto;
        }

        /* Depth Tooltip */
        .depth-tooltip {
          position: absolute;
          display: none;
          padding: 6px 12px;
          background: rgba(0,0,0,0.85);
          border: 1px solid rgba(255,140,0,0.5);
          border-radius: 6px;
          z-index: 20;
          pointer-events: none;
          backdrop-filter: blur(4px);
          font-family: 'Courier New', monospace;
          font-size: 0.65rem;
          color: #fff;
          white-space: nowrap;
          transform: translate(-50%, -120%);
        }
        .depth-tooltip .depth-title {
          display: block;
          color: var(--orange);
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          margin-bottom: 2px;
        }
        .depth-tooltip .depth-value {
          color: rgba(255,255,255,0.7);
        }
        .depth-tooltip.visible { display: block; }

        .section-label {
          font-size: 0.75rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--orange);
          margin-bottom: 14px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .section-label::after {
          content: '';
          flex: 1;
          height: 1px;
          background: linear-gradient(90deg, var(--orange), transparent);
        }

        /* ======================== */
        /* MAIN CONTENT             */
        /* ======================== */
        .page {
          max-width: 1400px;
          margin: 0 auto;
          padding: 24px 20px 40px;
          position: relative;
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
        .back-link:hover { background: var(--orange); color: #fff; }

        /* ======================== */
        /* OBJECTIVE CARDS          */
        /* ======================== */
        .objectives-table {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
          opacity: 0;
          transform: translateY(20px);
          animation: staggerUp 0.6s ease 1.4s forwards;
        }
        .objective-cell {
          border: 2px solid #ddd;
          border-radius: 14px;
          padding: 20px 18px;
          background: #fff;
          cursor: pointer;
          position: relative;
          overflow: hidden;
          transition: border-color 0.3s, box-shadow 0.3s, transform 0.3s;
        }
        .objective-cell:hover {
          border-color: var(--orange);
          box-shadow: 0 4px 24px rgba(255,140,0,0.15);
          transform: translateY(-3px);
        }

        /* Bounding box draw-in corners */
        .objective-cell::before,
        .objective-cell::after {
          content: '';
          position: absolute;
          width: 0; height: 0;
          border: 2px solid var(--orange);
          transition: width 0.35s ease, height 0.35s ease;
          pointer-events: none;
        }
        .objective-cell::before {
          top: 0; left: 0;
          border-right: none; border-bottom: none;
          border-radius: 14px 0 0 0;
        }
        .objective-cell::after {
          bottom: 0; right: 0;
          border-left: none; border-top: none;
          border-radius: 0 0 14px 0;
        }
        .objective-cell:hover::before,
        .objective-cell:hover::after {
          width: 50%; height: 50%;
        }

        /* Reticle ring on number */
        .num-ring {
          position: relative;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }
        .num-ring::before {
          content: '';
          position: absolute;
          width: 38px; height: 38px;
          border: 1.5px dashed rgba(255,140,0,0.35);
          border-radius: 50%;
          opacity: 0;
          transition: opacity 0.3s;
        }
        .objective-cell:hover .num-ring::before {
          opacity: 1;
          animation: rotateReticle 3s linear infinite;
        }
        @keyframes rotateReticle { from{transform:rotate(0)} to{transform:rotate(360deg)} }

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
          width: 26px; height: 26px;
          border-radius: 8px;
          background: var(--orange);
          color: #fff;
          font-size: 0.72rem;
          font-weight: 700;
          flex: 0 0 auto;
          position: relative;
          z-index: 1;
        }
        .objective-cell p {
          color: #1a1a1a;
          font-size: 0.82rem;
          line-height: 1.7;
        }

        /* Laser scan button */
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
          cursor: pointer;
          position: relative;
          overflow: hidden;
          transition: background 0.2s, color 0.2s;
        }
        .steps-toggle::after {
          content: '';
          position: absolute;
          top: -100%; left: 0;
          width: 100%; height: 100%;
          background: linear-gradient(180deg, transparent, rgba(255,140,0,0.18), transparent);
          pointer-events: none;
        }
        .objective-cell:hover .steps-toggle::after {
          animation: laserScan 1.2s ease-in-out infinite;
        }
        @keyframes laserScan { 0%{top:-100%} 100%{top:200%} }
        .steps-toggle:hover { background: var(--orange); color: #fff; }

        /* Workflow gradient border */
        .workflow-cell {
          background: linear-gradient(#fff, #fff) padding-box,
                      linear-gradient(0deg, var(--orange), transparent, var(--orange)) border-box;
          border: 2px solid transparent;
        }
        @keyframes gradientMarch { from{--angle:0deg} to{--angle:360deg} }

        @keyframes fadeDown { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes staggerUp { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }

        @media (max-width: 1200px) { .objectives-table { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 900px) { .objectives-table { grid-template-columns: 1fr; } }
        @media (max-width: 640px) { .hero { height: 280px; } }
      </style>
    </head>
    <body>

      <!-- ============================== -->
      <!-- BOOT-UP OVERLAY                -->
      <!-- ============================== -->
      <div id="boot-overlay">
        <div id="boot-reticle">
          <div class="crosshair h"></div>
          <div class="crosshair v"></div>
        </div>
        <div id="boot-terminal"></div>
        <div id="boot-progress"><div id="boot-progress-bar"></div></div>
        <div id="boot-pct">0%</div>
      </div>

      <!-- ============================== -->
      <!-- 3D POINT-CLOUD HERO            -->
      <!-- ============================== -->
      <div class="page">
        <div class="hero" id="hero-3d">
          <div class="hero-badge">LiDAR SIMULATION</div>
          <div class="hero-hint">Click &amp; Drag to Rotate</div>
          <div class="hud-overlay">
            <div class="hud-corner hud-tl">
              <div class="hud-label">FPS <span id="hud-fps">60</span></div>
              <div class="hud-label">LATENCY <span id="hud-latency">12ms</span></div>
            </div>
            <div class="hud-corner hud-tr">
              <div class="hud-label">SYSTEM <span class="hud-ok">OPTIMAL</span></div>
              <canvas id="hud-wave" width="80" height="24"></canvas>
            </div>
          </div>
          <div class="depth-tooltip" id="depth-tooltip">
            <span class="depth-title"></span>
            <span class="depth-value"></span>
          </div>
          <div class="hero-overlay">
            <h1>Perception Intelligence for Autonomous Vehicles</h1>
            <p class="subtitle">Monocular Vision &bull; Deep Learning &bull; Neuro-Fuzzy Decision Systems</p>
          </div>
        </div>

        <!-- ============================== -->
        <!-- OBJECTIVE CARDS                -->
        <!-- ============================== -->
        <div class="objectives-table">
          <div class="objective-cell" onclick="window.location.href='{{ url_for('objective1') }}'">
            <h2><span class="num-ring"><span class="num">1</span></span> Objective 1</h2>
            <p>To develop an intelligent autonomous vehicle navigation framework for obstacle detection and environmental understanding using monocular vision and deep learning techniques. The proposed framework focuses on detecting static and dynamic obstacles, improving path awareness, and supporting adaptive navigation under complex traffic environments.</p>
            <div class="steps-toggle" style="pointer-events:none;">Open Pipeline &rarr;</div>
          </div>
          <div class="objective-cell" onclick="window.location.href='{{ url_for('objective2') }}'">
            <h2><span class="num-ring"><span class="num">2</span></span> Objective 2</h2>
            <p>To design a vision-based lane and vehicle perception framework using YOLOv8-seg and neuro-fuzzy reasoning for accurate lane monitoring and intelligent driving assistance. The framework focuses on improving lane boundary detection, surrounding vehicle perception, and adaptive decision-making under challenging scenarios.</p>
            <div class="steps-toggle" style="pointer-events:none;">Open Pipeline &rarr;</div>
          </div>
          <div class="objective-cell" onclick="window.location.href='{{ url_for('objective3') }}'">
            <h2><span class="num-ring"><span class="num">3</span></span> Objective 3</h2>
            <p>To develop a unified perception intelligence framework that integrates monocular depth estimation, geometric computation, contextual understanding, and neuro-fuzzy decision systems for real-time autonomous driving applications.</p>
            <div class="steps-toggle" style="pointer-events:none;">Open Pipeline &rarr;</div>
          </div>
          <div class="objective-cell workflow-cell" onclick="window.location.href='{{ url_for('complete_workflow') }}'">
            <h2><span class="num-ring"><span class="num">&#8635;</span></span> Complete Workflow</h2>
            <p>End-to-end unified perception intelligence pipeline combining obstacle detection, lane monitoring, depth estimation, geometric computation, contextual understanding, and neuro-fuzzy decision making for autonomous vehicle navigation.</p>
            <div class="steps-toggle">Open Workflow &rarr;</div>
          </div>
        </div>

      </div>

      <!-- ============================== -->
      <!-- BOOT SEQUENCE SCRIPT           -->
      <!-- ============================== -->
      <script>
      (function(){
        var bootLines = [
          {text:'> INITIALIZING PERCEPTION ENGINE...', cls:''},
          {text:'  LOADING YOLOV8-SEG MODEL...', cls:''},
          {text:'  YOLOV8-SEG: ONLINE', cls:'ok'},
          {text:'  MONOCULAR DEPTH ESTIMATION: ONLINE', cls:'ok'},
          {text:'  LANE DETECTION MODULE: ONLINE', cls:'ok'},
          {text:'  GEOMETRIC COMPUTATION: ONLINE', cls:'ok'},
          {text:'  NEURO-FUZZY INFERENCE: ACTIVE', cls:'ok'},
          {text:'  VEHICLE TRACKER: ONLINE', cls:'ok'},
          {text:'  SCENE UNDERSTANDING: ONLINE', cls:'ok'},
          {text:'  ADAPTIVE NAVIGATION: READY', cls:'ok'},
          {text:'> ALL SYSTEMS NOMINAL — 100%', cls:'warn'}
        ];
        var terminal = document.getElementById('boot-terminal');
        var progressBar = document.getElementById('boot-progress-bar');
        var pctText = document.getElementById('boot-pct');
        var overlay = document.getElementById('boot-overlay');
        var reticle = document.getElementById('boot-reticle');
        var i = 0;
        function addLine() {
          if (i >= bootLines.length) {
            setTimeout(function() {
              reticle.classList.add('explode');
              setTimeout(function() {
                overlay.classList.add('done');
                setTimeout(function(){ overlay.remove(); }, 600);
              }, 500);
            }, 300);
            return;
          }
          var div = document.createElement('div');
          div.className = 'line ' + bootLines[i].cls;
          div.textContent = bootLines[i].text;
          terminal.appendChild(div);
          terminal.scrollTop = terminal.scrollHeight;
          var pct = Math.round(((i + 1) / bootLines.length) * 100);
          progressBar.style.width = pct + '%';
          pctText.textContent = pct + '%';
          i++;
          setTimeout(addLine, 120 + Math.random() * 80);
        }
        setTimeout(addLine, 400);
      })();
      </script>

      <!-- ============================== -->
      <!-- THREE.JS 3D POINT CLOUD        -->
      <!-- ============================== -->
      <script>
      (function(){
        var container = document.getElementById('hero-3d');
        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 4, 12);
        camera.lookAt(0, 0, 0);

        var renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x0a0a0a, 1);
        container.insertBefore(renderer.domElement, container.firstChild);

        var ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        scene.add(ambientLight);
        var pointLight = new THREE.PointLight(0xff8c00, 1.5, 50);
        pointLight.position.set(0, 6, 0);
        scene.add(pointLight);

        // ===== CAR =====
        var carGroup = new THREE.Group();
        var bodyGeo = new THREE.BoxGeometry(2.4, 0.7, 1.2);
        var bodyMat = new THREE.MeshPhongMaterial({color:0x222222, emissive:0x111111, shininess:80});
        var bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.y = 0.55;
        carGroup.add(bodyMesh);

        var cabinGeo = new THREE.BoxGeometry(1.4, 0.55, 1.0);
        var cabinMat = new THREE.MeshPhongMaterial({color:0x111111, emissive:0x050505, shininess:120, transparent:true, opacity:0.85});
        var cabin = new THREE.Mesh(cabinGeo, cabinMat);
        cabin.position.set(-0.1, 1.1, 0);
        carGroup.add(cabin);

        var wheelGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.15, 16);
        var wheelMat = new THREE.MeshPhongMaterial({color:0x333333});
        [[-0.7,0.22,0.65],[-0.7,0.22,-0.65],[0.7,0.22,0.65],[0.7,0.22,-0.65]].forEach(function(p) {
          var w = new THREE.Mesh(wheelGeo, wheelMat);
          w.position.set(p[0], p[1], p[2]);
          w.rotation.x = Math.PI / 2;
          carGroup.add(w);
        });

        var lidarGeo = new THREE.CylinderGeometry(0.15, 0.18, 0.12, 16);
        var lidarMat = new THREE.MeshPhongMaterial({color:0xff8c00, emissive:0xff8c00, emissiveIntensity:0.5});
        var lidarMesh = new THREE.Mesh(lidarGeo, lidarMat);
        lidarMesh.position.set(0, 1.4, 0);
        carGroup.add(lidarMesh);
        scene.add(carGroup);

        // ===== GROUND GRID =====
        var gridHelper = new THREE.GridHelper(80, 80, 0x332200, 0x1a1100);
        scene.add(gridHelper);

        // ===== ROAD LANES =====
        var laneGroup = new THREE.Group();
        function createLane(x, z, len, dir) {
          var geo = new THREE.PlaneGeometry(0.12, len);
          var mat = new THREE.MeshBasicMaterial({color:0xff8c00, transparent:true, opacity:0.6, side:THREE.DoubleSide});
          var mesh = new THREE.Mesh(geo, mat);
          mesh.rotation.x = -Math.PI / 2;
          mesh.position.set(x, 0.02, z);
          if (dir === 'h') { mesh.rotation.z = Math.PI / 2; }
          mesh.userData.baseZ = z;
          return mesh;
        }
        var laneLines = [];
        [-2.0, -0.65, 0.65, 2.0].forEach(function(x) {
          for (var z = -40; z < 40; z += 4) {
            var lane = createLane(x, z, 2.5, 'v');
            laneGroup.add(lane);
            laneLines.push(lane);
          }
        });
        // Center dashed line
        for (var z = -40; z < 40; z += 5) {
          var cl = createLane(0, z, 3, 'v');
          cl.material.color.setHex(0xffffff);
          cl.material.opacity = 0.4;
          laneGroup.add(cl);
          laneLines.push(cl);
        }
        scene.add(laneGroup);

        // ===== POINT CLOUD =====
        var particleCount = 1500;
        var pGeo = new THREE.BufferGeometry();
        var positions = new Float32Array(particleCount * 3);
        var colors = new Float32Array(particleCount * 3);
        var particleClusterIds = new Int32Array(particleCount);

        var clusterDefs = [
          {cx:-5, cy:1.5, cz:-6, r:3, label:'Car', conf:'0.94', dist:'14.2m', cls:0},
          {cx:4, cy:1.5, cz:-10, r:3, label:'Car', conf:'0.91', dist:'22.8m', cls:1},
          {cx:-3, cy:0.8, cz:-5, r:1.5, label:'Pedestrian', conf:'0.87', dist:'11.6m', cls:2},
          {cx:6, cy:1.2, cz:-4, r:2, label:'Obstacle', conf:'0.96', dist:'8.3m', cls:3},
          {cx:-6, cy:2, cz:-12, r:2.5, label:'Truck', conf:'0.82', dist:'28.1m', cls:4},
          {cx:2, cy:0.6, cz:-7, r:1.5, label:'Pedestrian', conf:'0.89', dist:'16.4m', cls:5}
        ];

        for (var i = 0; i < particleCount; i++) {
          var placed = false;
          for (var c = 0; c < clusterDefs.length; c++) {
            if (Math.random() < 0.4) {
              var cd = clusterDefs[c];
              positions[i*3] = cd.cx + (Math.random()-0.5) * cd.r;
              positions[i*3+1] = cd.cy + (Math.random()-0.5) * cd.r * 0.6;
              positions[i*3+2] = cd.cz + (Math.random()-0.5) * cd.r;
              particleClusterIds[i] = c;
              colors[i*3] = 1; colors[i*3+1] = 0.55; colors[i*3+2] = 0;
              placed = true;
              break;
            }
          }
          if (!placed) {
            positions[i*3] = (Math.random()-0.5) * 50;
            positions[i*3+1] = Math.random() * 10;
            positions[i*3+2] = (Math.random()-0.5) * 50;
            particleClusterIds[i] = -1;
            var g = 0.4 + Math.random()*0.4;
            colors[i*3] = g; colors[i*3+1] = g; colors[i*3+2] = g;
          }
        }
        pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        pGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        var pMat = new THREE.PointsMaterial({size:0.08, vertexColors:true, transparent:true, opacity:0.8});
        var particles = new THREE.Points(pGeo, pMat);
        scene.add(particles);

        // ===== BOUNDING BOXES ON CLUSTERS =====
        var clusterBoxes = [];
        clusterDefs.forEach(function(cd) {
          var s = cd.r * 1.4;
          var bg = new THREE.BoxGeometry(s, s * 0.8, s);
          var edges = new THREE.EdgesGeometry(bg);
          var lm = new THREE.LineBasicMaterial({color:0xff8c00, transparent:true, opacity:0.0});
          var line = new THREE.LineSegments(edges, lm);
          line.position.set(cd.cx, cd.cy, cd.cz);
          line.userData = {def: cd, baseOpacity: 0.0, flash: 0};
          scene.add(line);
          clusterBoxes.push(line);
        });

        // ===== LIDAR WAVE RINGS =====
        var rings = [];
        function spawnRing() {
          var ringGeo = new THREE.RingGeometry(0.1, 0.25, 64);
          var ringMat = new THREE.MeshBasicMaterial({color:0xff8c00, transparent:true, opacity:0.6, side:THREE.DoubleSide});
          var ring = new THREE.Mesh(ringGeo, ringMat);
          ring.rotation.x = -Math.PI / 2;
          ring.position.set(0, 1.45, 0);
          ring.userData = {age:0};
          scene.add(ring);
          rings.push(ring);
        }
        var ringTimer = 0;

        // ===== MOUSE / CAMERA CONTROL =====
        var isDragging = false;
        var prevMouse = {x:0, y:0};
        var rotY = 0, rotX = 0.3;
        var targetRotY = 0, targetRotX = 0.3;
        var cameraTarget = new THREE.Vector3(0, 0.5, 0);
        var cameraTargetGoal = new THREE.Vector3(0, 0.5, 0);
        var cameraRadius = 12;
        var targetRadius = 12;

        container.addEventListener('mousedown', function(e) {
          isDragging = true;
          prevMouse = {x:e.clientX, y:e.clientY};
        });
        window.addEventListener('mouseup', function() { isDragging = false; });
        window.addEventListener('mousemove', function(e) {
          if (!isDragging) return;
          var dx = e.clientX - prevMouse.x;
          var dy = e.clientY - prevMouse.y;
          targetRotY += dx * 0.005;
          targetRotX += dy * 0.003;
          targetRotX = Math.max(-0.5, Math.min(1.0, targetRotX));
          prevMouse = {x:e.clientX, y:e.clientY};
        });

        // ===== HOVER / RAYCASTER =====
        var raycaster = new THREE.Raycaster();
        var mouse = new THREE.Vector2(-999, -999);
        var depthTooltip = document.getElementById('depth-tooltip');
        container.addEventListener('mousemove', function(e) {
          var rect = container.getBoundingClientRect();
          mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
          depthTooltip.style.left = (e.clientX - rect.left) + 'px';
          depthTooltip.style.top = (e.clientY - rect.top) + 'px';
        });
        container.addEventListener('mouseleave', function() {
          mouse.set(-999, -999);
          depthTooltip.classList.remove('visible');
        });

        // ===== CARD CLICK -> CAMERA FOCUS =====
        window.focusCameraOn = function(target, radius, height) {
          cameraTargetGoal.copy(target);
          targetRadius = radius;
          targetRotX = height;
        };

        // ===== HUD WAVE CANVAS =====
        var hudWave = document.getElementById('hud-wave');
        var wCtx = hudWave.getContext('2d');
        var waveData = new Float32Array(80);
        function drawHudWave() {
          wCtx.clearRect(0, 0, 80, 24);
          wCtx.strokeStyle = 'rgba(255,140,0,0.5)';
          wCtx.lineWidth = 1;
          wCtx.beginPath();
          for (var x = 0; x < 80; x++) {
            waveData[x] = waveData[x+1] || 0;
          }
          var t2 = Date.now() * 0.003;
          for (var x = 0; x < 80; x++) {
            waveData[x] = Math.sin(t2 + x * 0.15) * 8 + Math.sin(t2 * 2.3 + x * 0.08) * 4;
            var y = 12 + waveData[x];
            if (x === 0) wCtx.moveTo(x, y); else wCtx.lineTo(x, y);
          }
          wCtx.stroke();
        }

        // ===== FPS COUNTER =====
        var frameCount = 0;
        var lastFpsTime = performance.now();
        var fpsDisplay = document.getElementById('hud-fps');
        var latencyDisplay = document.getElementById('hud-latency');

        // ===== ANIMATION LOOP =====
        var clock = new THREE.Clock();
        function animate() {
          requestAnimationFrame(animate);
          var t = clock.getElapsedTime();
          var dt = clock.getDelta();

          // FPS
          frameCount++;
          var now = performance.now();
          if (now - lastFpsTime > 500) {
            var fps = Math.round(frameCount / ((now - lastFpsTime) / 1000));
            fpsDisplay.textContent = fps;
            latencyDisplay.textContent = Math.round(1000 / Math.max(fps, 1)) + 'ms';
            frameCount = 0;
            lastFpsTime = now;
          }

          // Camera
          rotY += (targetRotY - rotY) * 0.06;
          rotX += (targetRotX - rotX) * 0.06;
          cameraRadius += (targetRadius - cameraRadius) * 0.06;
          cameraTarget.lerp(cameraTargetGoal, 0.06);
          camera.position.x = Math.sin(rotY) * cameraRadius;
          camera.position.z = Math.cos(rotY) * cameraRadius;
          camera.position.y = 2 + rotX * 8;
          camera.lookAt(cameraTarget);

          // Road lanes scroll backward (driving illusion)
          laneLines.forEach(function(lane) {
            lane.position.z += 0.08;
            if (lane.position.z > 40) lane.position.z -= 80;
          });

          // LiDAR rings
          ringTimer++;
          if (ringTimer % 40 === 0) spawnRing();
          for (var r = rings.length - 1; r >= 0; r--) {
            var ring = rings[r];
            ring.userData.age += 0.02;
            var s = 1 + ring.userData.age * 10;
            ring.scale.set(s, s, 1);
            ring.material.opacity = 0.6 * Math.max(0, 1 - ring.userData.age);
            if (ring.userData.age > 1) {
              scene.remove(ring);
              ring.geometry.dispose();
              ring.material.dispose();
              rings.splice(r, 1);
            }
          }

          // Particle shimmer
          var posArr = particles.geometry.attributes.position.array;
          for (var i = 0; i < particleCount; i++) {
            posArr[i*3+1] += Math.sin(t * 2 + i) * 0.001;
          }
          particles.geometry.attributes.position.needsUpdate = true;

          // Pulse light
          pointLight.intensity = 1.2 + Math.sin(t * 3) * 0.3;

          // Raycaster for bounding box hover
          raycaster.setFromCamera(mouse, camera);
          var hits = raycaster.intersectObjects(clusterBoxes, false);
          var hoveredBox = hits.length > 0 ? hits[0].object : null;

          clusterBoxes.forEach(function(bl) {
            var def = bl.userData.def;
            var isHovered = (bl === hoveredBox);

            // Flash on LiDAR ring pass
            var ringDist = Math.abs(bl.position.z);
            var ringFlash = 0;
            rings.forEach(function(ring) {
              var dist = Math.abs(ring.position.z - bl.position.z);
              if (dist < 3 && ring.userData.age < 0.3) ringFlash = 0.8;
            });
            bl.userData.flash = Math.max(ringFlash, bl.userData.flash * 0.95);

            var targetOp = isHovered ? 0.9 : (bl.userData.flash > 0.1 ? 0.7 + bl.userData.flash * 0.3 : 0.15);
            bl.material.opacity += (targetOp - bl.material.opacity) * 0.1;

            if (isHovered) {
              depthTooltip.querySelector('.depth-title').textContent = def.label + ' [' + Math.round(parseFloat(def.conf)*100) + '%]';
              depthTooltip.querySelector('.depth-value').textContent = 'Depth: ' + def.dist;
              depthTooltip.classList.add('visible');
            }
          });

          if (!hoveredBox) depthTooltip.classList.remove('visible');

          // HUD wave
          drawHudWave();

          renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', function() {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        });
      })();
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
        .layout { display: grid; grid-template-columns: 280px 1fr; gap: 40px; align-items: start; }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s;
        }
        .video-panel img { width: 100%; height: auto; display: block; object-fit: contain; max-height: 70vh; }
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
          gap: 8px;
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
          border: 2px solid #ddd; border-radius: 10px; padding: 8px 10px;
          background: #fff; transition: border-color 0.3s, box-shadow 0.3s;
        }
        .step-card.active {
          border-color: var(--orange);
          box-shadow: 0 2px 12px rgba(255,140,0,0.12);
        }
        .step-card h3 {
          font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
          color: var(--orange); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
        }
        .step-card h3 .step-num {
          width: 18px; height: 18px; border-radius: 5px; background: var(--orange);
          color: #fff; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.55rem; font-weight: 700; flex: 0 0 auto;
        }
        .step-card .step-status {
          font-size: 0.7rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0;
        }
        .step-card .step-detail {
          font-size: 0.62rem; color: #666; line-height: 1.3;
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

          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('obj1_video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('obj1_upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*,image/*">
                <button type="button" id="load-btn">Load Media</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video or image to start processing</span>
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
        .layout { display: grid; grid-template-columns: 280px 1fr; gap: 40px; align-items: start; }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s;
        }
        .video-panel img { width: 100%; height: auto; display: block; object-fit: contain; max-height: 70vh; }
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
          gap: 8px;
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
          grid-column: auto;
        }
        .step-card {
          border: 2px solid #ddd; border-radius: 10px; padding: 8px 10px;
          background: #fff; transition: border-color 0.3s, box-shadow 0.3s;
        }
        .step-card.active {
          border-color: var(--orange);
          box-shadow: 0 2px 12px rgba(255,140,0,0.12);
        }
        .step-card h3 {
          font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
          color: var(--orange); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
        }
        .step-card h3 .step-num {
          width: 18px; height: 18px; border-radius: 5px; background: var(--orange);
          color: #fff; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.55rem; font-weight: 700; flex: 0 0 auto;
        }
        .step-card .step-status {
          font-size: 0.7rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0;
        }
        .step-card .step-detail {
          font-size: 0.62rem; color: #666; line-height: 1.3;
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
        <h1>Objective 2</h1>
        <p class="subtitle">Lane Monitoring &bull; Vehicle Perception &bull; Driving Assistance &bull; Challenging Road Conditions</p>

        <div class="layout">
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

          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('obj2_video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('obj2_upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*,image/*">
                <button type="button" id="load-btn">Load Media</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video or image to start processing</span>
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
        .layout { display: grid; grid-template-columns: 280px 1fr; gap: 40px; align-items: start; }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s;
        }
        .video-panel img { width: 100%; height: auto; display: block; object-fit: contain; max-height: 70vh; }
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
          gap: 8px;
          align-items: stretch;
        }
        .step-card {
          display: flex; flex-direction: column;
        }
        .step-card {
          border: 2px solid #ddd; border-radius: 10px; padding: 8px 10px;
          background: #fff; transition: border-color 0.3s, box-shadow 0.3s;
        }
        .step-card.active {
          border-color: var(--orange);
          box-shadow: 0 2px 12px rgba(255,140,0,0.12);
        }
        .step-card h3 {
          font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
          color: var(--orange); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
        }
        .step-card h3 .step-num {
          width: 18px; height: 18px; border-radius: 5px; background: var(--orange);
          color: #fff; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.55rem; font-weight: 700; flex: 0 0 auto;
        }
        .step-card .step-status {
          font-size: 0.7rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0;
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

          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('obj3_video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('obj3_upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*,image/*">
                <button type="button" id="load-btn">Load Media</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video or image to start processing</span>
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
          grid-template-columns: 280px 1fr;
          gap: 20px;
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
          padding: 8px 10px;
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
          font-size: 0.62rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: #888;
          margin-bottom: 4px;
        }
        .decision-chip .chip-value {
          font-size: 0.7rem;
          font-weight: 700;
          color: var(--orange);
        }
        .status-info {
          color: #555;
          font-size: 0.7rem;
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
          padding: 8px 10px;
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
          font-size: 0.7rem;
          color: #1a1a1a;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 8px;
          margin-top: 10px;
        }
        .metric {
          padding: 8px 10px;
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
          font-size: 0.7rem;
          color: #1a1a1a;
        }
        .video-panel {
          border: 2px solid #ddd; border-radius: 16px; overflow: hidden; background: #000;
          position: relative; transition: border-color 0.3s;
        }
        .video-panel img { width: 100%; height: auto; display: block; object-fit: contain; max-height: 70vh; }
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

          <div>
            <div class="video-panel">
              <div class="video-badge">Live Processing</div>
              <img src="{{ url_for('video_feed') }}" alt="video feed">
            </div>
            <div class="upload-row">
              <form id="upload-form" action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data">
                <input id="video-input" type="file" name="file" accept="video/*,image/*">
                <button type="button" id="load-btn">Load Media</button>
              </form>
              <span class="upload-hint" id="upload-hint">Upload a video or image to start processing</span>
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
