#!/usr/bin/env python3
"""
Debug camera feed with real-time sharpness display.
Press 'q' to quit, 's' to save a snapshot, 'f' to toggle autofocus.
"""

import cv2
import numpy as np
import sys


def calculate_sharpness(image):
    """
    Calculate Laplacian variance (sharpness score).

    Normalized by resolution so scores are comparable across
    different image sizes. Reference: VGA (640x480 = 307200 pixels).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize by resolution (reference: VGA 640x480)
    pixels = image.shape[0] * image.shape[1]
    reference_pixels = 640 * 480
    scale_factor = pixels / reference_pixels

    return variance * scale_factor


def list_cameras():
    """List available cameras."""
    print("Scanning for cameras...")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, w, h))
                print(f"  [{len(available)}] /dev/video{i}: {w}x{h}")
            cap.release()
    return available


def show_camera_preview(cameras):
    """Show preview of each camera for selection."""
    print("\nShowing preview of each camera. Press any key to move to next, or 'q' to quit.")

    for idx, (cam_id, w, h) in enumerate(cameras):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            continue

        # Warmup
        for _ in range(10):
            cap.read()

        ret, frame = cap.read()
        if ret:
            # Resize for display
            display = frame.copy()
            dh, dw = display.shape[:2]
            if dw > 800:
                scale = 800 / dw
                display = cv2.resize(display, (int(dw * scale), int(dh * scale)))

            # Add label
            label = f"Camera {idx + 1}: /dev/video{cam_id} ({w}x{h}) - Press SPACE to select, any other key for next"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Camera Preview - SPACE to select, other key for next', display)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            cap.release()

            if key == ord(' '):
                return cam_id
            elif key == ord('q'):
                return None
        else:
            cap.release()

    return None


def main():
    # List cameras
    cameras = list_cameras()

    if not cameras:
        print("No cameras found!")
        sys.exit(1)

    # If camera index provided as argument, use it
    if len(sys.argv) > 1:
        camera_idx = int(sys.argv[1])
    else:
        # Show preview for selection
        camera_idx = show_camera_preview(cameras)
        if camera_idx is None:
            print("No camera selected.")
            sys.exit(0)

    print(f"\nOpening camera {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"Failed to open camera {camera_idx}")
        sys.exit(1)

    # Set to MJPEG for high resolution support
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Try highest resolution first (4K down to VGA)
    resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 2K
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (640, 480),    # VGA
    ]

    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            print(f"Set resolution: {w}x{h}")
            break

    # Enable autofocus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save snapshot")
    print("  f - Toggle autofocus")
    print("  + - Increase manual focus")
    print("  - - Decrease manual focus")
    print()

    autofocus = True
    manual_focus = 0
    snapshot_count = 0

    # Warmup
    print("Warming up (30 frames)...")
    for _ in range(30):
        cap.read()

    print("Streaming... Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Calculate sharpness
        sharpness = calculate_sharpness(frame)

        # Draw info on frame
        display = frame.copy()

        # Sharpness indicator - adjusted thresholds based on real-world testing
        color = (0, 255, 0) if sharpness >= 50 else (0, 255, 255) if sharpness >= 30 else (0, 0, 255)
        cv2.putText(display, f"Sharpness: {sharpness:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Quality assessment
        if sharpness >= 50:
            quality = "GOOD"
        elif sharpness >= 30:
            quality = "OK"
        else:
            quality = "BLURRY"
        cv2.putText(display, f"Quality: {quality}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Focus mode
        focus_text = "Autofocus: ON" if autofocus else f"Manual focus: {manual_focus}"
        cv2.putText(display, focus_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Resolution
        res_text = f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
        cv2.putText(display, res_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Resize for display if too large
        h, w = display.shape[:2]
        if w > 1280:
            scale = 1280 / w
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        cv2.imshow('Camera Debug - Press Q to quit', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{snapshot_count:03d}_sharpness_{sharpness:.0f}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            snapshot_count += 1
        elif key == ord('f'):
            autofocus = not autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            print(f"Autofocus: {'ON' if autofocus else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            if not autofocus:
                manual_focus = min(255, manual_focus + 10)
                cap.set(cv2.CAP_PROP_FOCUS, manual_focus)
                print(f"Manual focus: {manual_focus}")
        elif key == ord('-'):
            if not autofocus:
                manual_focus = max(0, manual_focus - 10)
                cap.set(cv2.CAP_PROP_FOCUS, manual_focus)
                print(f"Manual focus: {manual_focus}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()
