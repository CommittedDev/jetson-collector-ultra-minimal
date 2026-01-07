#!/usr/bin/env python3
"""
Camera alignment tool - shows live feed from both cameras side by side.
Uses highest resolution detected for each camera (same as capture mode).

Usage:
    python3 align_cameras.py

Controls:
    q / ESC  - Quit
    s        - Save snapshot to ./alignment_snapshot.jpg
"""

import os
import sys
import subprocess
from typing import List, Tuple

import cv2
import numpy as np


def discover_cameras() -> List[str]:
    """Find all real USB cameras."""
    cameras = []

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
            cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    cameras.append(device)
                    print(f"Found camera: {device}")
            else:
                cap.release()

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    return cameras


def detect_max_resolution(device_path: str) -> Tuple[int, int]:
    """Auto-detect maximum supported resolution."""
    resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 2K
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (640, 480),    # VGA
    ]

    device_id = int(device_path.replace('/dev/video', ''))
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        cap.release()
        return (1920, 1080)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    best = (640, 480)
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == w and actual_h == h:
            best = (w, h)
            break

    cap.release()
    return best


def main():
    print("=" * 60)
    print("Camera Alignment Tool")
    print("=" * 60)
    print("Discovering cameras...")

    devices = discover_cameras()

    if len(devices) < 2:
        print(f"ERROR: Found {len(devices)} cameras, need at least 2")
        sys.exit(1)

    print(f"\nInitializing {len(devices)} cameras at max resolution...")

    caps = []
    resolutions = []

    for i, device in enumerate(devices[:2]):  # Use first 2 cameras
        resolution = detect_max_resolution(device)
        resolutions.append(resolution)
        print(f"Camera {i+1}: {device} @ {resolution[0]}x{resolution[1]}")

        device_id = int(device.replace('/dev/video', ''))
        cap = cv2.VideoCapture(device_id)

        if not cap.isOpened():
            print(f"ERROR: Failed to open {device}")
            sys.exit(1)

        # Configure for max resolution (same as collector)
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

        caps.append(cap)

    # Calculate display size (fit side-by-side in reasonable window)
    max_display_width = 1920  # Total width for both cameras
    display_height = 540      # Height per camera

    # Scale factor to fit both cameras side by side
    scale = min(
        (max_display_width / 2) / max(r[0] for r in resolutions),
        display_height / max(r[1] for r in resolutions)
    )

    display_w = int(resolutions[0][0] * scale)
    display_h = int(resolutions[0][1] * scale)

    print(f"\nDisplay size per camera: {display_w}x{display_h}")
    print("=" * 60)
    print("Controls:")
    print("  q / ESC  - Quit")
    print("  s        - Save snapshot")
    print("=" * 60)

    cv2.namedWindow('Camera Alignment', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Alignment', display_w * 2, display_h)

    snapshot_count = 0

    try:
        while True:
            frames = []

            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Resize for display
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
                    # Black frame if capture failed
                    frames.append(np.zeros((display_h, display_w, 3), dtype=np.uint8))

            # Combine side by side
            combined = cv2.hconcat(frames)

            # Add dividing line
            cv2.line(combined, (display_w, 0), (display_w, display_h), (255, 255, 255), 2)

            cv2.imshow('Camera Alignment', combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                # Save full resolution snapshot
                full_frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if ret:
                        full_frames.append(frame)

                if len(full_frames) == 2:
                    # Resize second to match first if different
                    if full_frames[0].shape != full_frames[1].shape:
                        full_frames[1] = cv2.resize(
                            full_frames[1],
                            (full_frames[0].shape[1], full_frames[0].shape[0])
                        )

                    snapshot = cv2.hconcat(full_frames)
                    filename = f"alignment_snapshot_{snapshot_count}.jpg"
                    cv2.imwrite(filename, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    snapshot_count += 1
                    print(f"Saved: {filename}")

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("\nCameras released. Done.")


if __name__ == '__main__':
    main()
