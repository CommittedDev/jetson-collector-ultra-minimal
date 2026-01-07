#!/usr/bin/env python3
"""
Camera alignment tool - web-based live feed from both cameras side by side.
Uses highest resolution detected for each camera (same as capture mode).

Usage:
    python3 align_cameras.py [--port 8080]

Then open in browser: http://<jetson-ip>:8080
"""

import os
import sys
import argparse
import subprocess
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np

# Flask for web streaming
try:
    from flask import Flask, Response, render_template_string
except ImportError:
    print("Flask not installed. Installing...")
    os.system("pip3 install flask")
    from flask import Flask, Response, render_template_string


app = Flask(__name__)

# Global state
cameras = []
resolutions = []
max_resolutions = []
device_paths = []
running = True
snapshot_count = 0

# Frame buffer for background capture
latest_frames = [None, None]
frame_lock = threading.Lock()
capture_fps = 0.0


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Camera Alignment</title>
    <style>
        body {
            background: #1a1a1a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .info {
            color: #888;
            margin-bottom: 20px;
        }
        .stream-container {
            display: inline-block;
            background: #000;
            border: 2px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .controls {
            margin-top: 20px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #45a049;
        }
        .status {
            margin-top: 15px;
            color: #888;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Camera Alignment Tool</h1>
    <div class="info">
        Camera 1: {{ max_res1 }} | Camera 2: {{ max_res2 }}
        <br><small>Capturing at full resolution (same FOV as production)</small>
    </div>
    <div class="stream-container">
        <img src="/stream" alt="Camera Feed">
    </div>
    <div class="controls">
        <button onclick="snapshot()">Save Snapshot</button>
    </div>
    <div class="status" id="status"></div>

    <script>
        function snapshot() {
            document.getElementById('status').innerText = 'Capturing...';
            fetch('/snapshot')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerText =
                        data.success ? 'Saved: ' + data.filename : 'Error: ' + data.error;
                });
        }
    </script>
</body>
</html>
"""


def discover_cameras() -> List[str]:
    """Find all real USB cameras."""
    found = []

    for i in range(20):
        device = f"/dev/video{i}"
        if not os.path.exists(device):
            continue

        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', device, '--all'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if 'Video Capture' not in result.stdout:
                continue

            device_id = int(device.replace('/dev/video', ''))
            cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    found.append(device)
                    print(f"Found camera: {device}")
            else:
                cap.release()

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    return found


def detect_max_resolution(device_path: str) -> Tuple[int, int]:
    """Auto-detect maximum supported resolution."""
    test_resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 2K
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (640, 480),    # VGA
    ]

    device_id = int(device_path.replace('/dev/video', ''))
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap.release()
        return (1920, 1080)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    best = (640, 480)
    for w, h in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == w and actual_h == h:
            best = (w, h)
            break

    cap.release()
    return best


def capture_thread_func():
    """Background thread that captures frames continuously."""
    global cameras, latest_frames, frame_lock, running, capture_fps

    frame_count = 0
    fps_time = time.time()

    while running:
        frames = []
        for cap in cameras:
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            else:
                frames.append(None)

        with frame_lock:
            for i, frame in enumerate(frames):
                if i < len(latest_frames):
                    latest_frames[i] = frame

        # Calculate capture FPS
        frame_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            capture_fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now


def generate_frames():
    """Generator for MJPEG stream - reads from buffer updated by capture thread."""
    global latest_frames, frame_lock, running, resolutions, capture_fps

    # Fixed display size (each camera panel)
    display_w = 960
    display_h = 540

    while running:
        with frame_lock:
            frames_copy = [f.copy() if f is not None else None for f in latest_frames]

        display_frames = []
        for i, frame in enumerate(frames_copy):
            if frame is not None:
                frame_resized = cv2.resize(frame, (display_w, display_h))

                # Add camera label with actual resolution
                if i < len(resolutions):
                    label = f"Camera {i+1} ({resolutions[i][0]}x{resolutions[i][1]})"
                else:
                    label = f"Camera {i+1}"
                cv2.putText(
                    frame_resized, label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

                # Add center crosshair
                cx, cy = display_w // 2, display_h // 2
                cv2.line(frame_resized, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
                cv2.line(frame_resized, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

                display_frames.append(frame_resized)
            else:
                display_frames.append(np.zeros((display_h, display_w, 3), dtype=np.uint8))

        # Combine side by side
        if len(display_frames) >= 2:
            combined = cv2.hconcat(display_frames[:2])
            cv2.line(combined, (display_w, 0), (display_w, display_h), (255, 255, 255), 2)

            # Add FPS counter
            cv2.putText(
                combined, f"Capture: {capture_fps:.1f} FPS",
                (combined.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            # Encode as JPEG (lower quality for faster streaming)
            _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Stream at higher rate than capture - shows latest frame smoothly
        time.sleep(0.05)  # 20 FPS streaming rate


@app.route('/')
def index():
    global max_resolutions
    return render_template_string(
        HTML_TEMPLATE,
        max_res1=f"{max_resolutions[0][0]}x{max_resolutions[0][1]}" if len(max_resolutions) > 0 else "N/A",
        max_res2=f"{max_resolutions[1][0]}x{max_resolutions[1][1]}" if len(max_resolutions) > 1 else "N/A"
    )


@app.route('/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/snapshot')
def snapshot():
    """Save snapshot from current frames (already at full resolution)."""
    global latest_frames, frame_lock, snapshot_count

    try:
        with frame_lock:
            frames = [f.copy() if f is not None else None for f in latest_frames]

        valid_frames = [f for f in frames if f is not None]

        if len(valid_frames) >= 2:
            # Resize second to match first if different
            if valid_frames[0].shape != valid_frames[1].shape:
                valid_frames[1] = cv2.resize(
                    valid_frames[1],
                    (valid_frames[0].shape[1], valid_frames[0].shape[0])
                )

            combined = cv2.hconcat(valid_frames[:2])
            filename = f"alignment_snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
            snapshot_count += 1
            return {"success": True, "filename": filename}
        else:
            return {"success": False, "error": "Could not get frames from both cameras"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def init_cameras():
    """Initialize cameras at full resolution (same FOV as capture mode)."""
    global cameras, resolutions, max_resolutions, device_paths, latest_frames

    print("=" * 60)
    print("Camera Alignment Tool (Web)")
    print("=" * 60)
    print("Discovering cameras...")

    device_paths = discover_cameras()

    if len(device_paths) < 2:
        print(f"ERROR: Found {len(device_paths)} cameras, need at least 2")
        sys.exit(1)

    print(f"\nDetecting max resolutions...")

    # First detect max resolutions
    for i, device in enumerate(device_paths[:2]):
        max_res = detect_max_resolution(device)
        max_resolutions.append(max_res)
        print(f"Camera {i+1}: {device} max resolution {max_res[0]}x{max_res[1]}")

    print(f"\nInitializing cameras at FULL resolution (for correct FOV)...")
    print("Note: Capture rate limited by USB bandwidth, but FOV matches production")

    # Initialize at full resolution to match capture FOV
    for i, device in enumerate(device_paths[:2]):
        full_res = max_resolutions[i]
        resolutions.append(full_res)

        device_id = int(device.replace('/dev/video', ''))
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"ERROR: Failed to open {device}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, full_res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, full_res[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # Warmup
        print(f"Camera {i+1}: Warming up...")
        for _ in range(30):
            cap.read()

        cameras.append(cap)
        print(f"Camera {i+1}: Ready at {full_res[0]}x{full_res[1]}")

    # Initialize frame buffer
    latest_frames = [None] * len(cameras)

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Camera Alignment Tool')
    parser.add_argument('--port', type=int, default=8080, help='Web server port')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    init_cameras()

    # Start background capture thread
    capture_thread = threading.Thread(target=capture_thread_func, daemon=True)
    capture_thread.start()
    print("Background capture thread started")

    print(f"Starting web server on http://0.0.0.0:{args.port}")
    print(f"Open in browser: http://<jetson-ip>:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        global running
        running = False
        capture_thread.join(timeout=2)
        for cap in cameras:
            cap.release()
        print("\nCameras released. Done.")


if __name__ == '__main__':
    main()
