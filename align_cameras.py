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
running = True
snapshot_count = 0


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
        Camera 1: {{ res1 }} | Camera 2: {{ res2 }}
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


def generate_frames():
    """Generator for MJPEG stream."""
    global cameras, resolutions, running

    # Calculate display size
    max_display_width = 1920
    display_height = 540

    scale = min(
        (max_display_width / 2) / max(r[0] for r in resolutions),
        display_height / max(r[1] for r in resolutions)
    )

    display_w = int(resolutions[0][0] * scale)
    display_h = int(resolutions[0][1] * scale)

    while running:
        frames = []

        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_resized = cv2.resize(frame, (display_w, display_h))

                # Add camera label
                label = f"Camera {i+1} ({resolutions[i][0]}x{resolutions[i][1]})"
                cv2.putText(
                    frame_resized, label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

                # Add center crosshair
                h, w = frame_resized.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.line(frame_resized, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
                cv2.line(frame_resized, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

                frames.append(frame_resized)
            else:
                frames.append(np.zeros((display_h, display_w, 3), dtype=np.uint8))

        # Combine side by side
        combined = cv2.hconcat(frames)
        cv2.line(combined, (display_w, 0), (display_w, display_h), (255, 255, 255), 2)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    global resolutions
    return render_template_string(
        HTML_TEMPLATE,
        res1=f"{resolutions[0][0]}x{resolutions[0][1]}" if len(resolutions) > 0 else "N/A",
        res2=f"{resolutions[1][0]}x{resolutions[1][1]}" if len(resolutions) > 1 else "N/A"
    )


@app.route('/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/snapshot')
def snapshot():
    global cameras, snapshot_count

    try:
        full_frames = []
        for cap in cameras:
            ret, frame = cap.read()
            if ret:
                full_frames.append(frame)

        if len(full_frames) == 2:
            if full_frames[0].shape != full_frames[1].shape:
                full_frames[1] = cv2.resize(
                    full_frames[1],
                    (full_frames[0].shape[1], full_frames[0].shape[0])
                )

            combined = cv2.hconcat(full_frames)
            filename = f"alignment_snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
            snapshot_count += 1
            return {"success": True, "filename": filename}
        else:
            return {"success": False, "error": "Could not capture from both cameras"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def init_cameras():
    """Initialize cameras."""
    global cameras, resolutions

    print("=" * 60)
    print("Camera Alignment Tool (Web)")
    print("=" * 60)
    print("Discovering cameras...")

    devices = discover_cameras()

    if len(devices) < 2:
        print(f"ERROR: Found {len(devices)} cameras, need at least 2")
        sys.exit(1)

    print(f"\nInitializing {len(devices)} cameras at max resolution...")

    for i, device in enumerate(devices[:2]):
        resolution = detect_max_resolution(device)
        resolutions.append(resolution)
        print(f"Camera {i+1}: {device} @ {resolution[0]}x{resolution[1]}")

        device_id = int(device.replace('/dev/video', ''))
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"ERROR: Failed to open {device}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # Warmup
        print(f"Camera {i+1}: Warming up...")
        for _ in range(30):
            cap.read()

        cameras.append(cap)

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Camera Alignment Tool')
    parser.add_argument('--port', type=int, default=8080, help='Web server port')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    init_cameras()

    print(f"Starting web server on http://0.0.0.0:{args.port}")
    print(f"Open in browser: http://<jetson-ip>:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        global running
        running = False
        for cap in cameras:
            cap.release()
        print("\nCameras released. Done.")


if __name__ == '__main__':
    main()
