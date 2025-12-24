"""
Auto-discovery dual camera with burst capture and sharpness selection.
Cameras stay warm (persistent VideoCapture) for consistent image quality.
"""

import os
import subprocess
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values = sharper image.

    Args:
        image: BGR image array

    Returns:
        Laplacian variance (sharpness score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def discover_cameras() -> List[str]:
    """
    Find all real USB cameras, regardless of which USB port.
    Filters out metadata/control devices that USB cameras expose.

    Returns:
        List of valid camera device paths (e.g., ['/dev/video0', '/dev/video2'])
    """
    cameras = []

    for i in range(20):  # Scan video0 to video19
        device = f"/dev/video{i}"
        if not os.path.exists(device):
            continue

        try:
            # Use v4l2-ctl to check if it's a real capture device
            result = subprocess.run(
                ['v4l2-ctl', '-d', device, '--all'],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Real cameras have "Video Capture" in capabilities
            if 'Video Capture' not in result.stdout:
                continue

            # Additional check: try to open and read a frame
            device_id = int(device.replace('/dev/video', ''))
            cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    cameras.append(device)
                    logging.info(f"[Discovery] Found camera: {device}")
            else:
                cap.release()

        except subprocess.TimeoutExpired:
            logging.warning(f"[Discovery] Timeout checking {device}")
        except Exception as e:
            logging.warning(f"[Discovery] Error checking {device}: {e}")

    return cameras


def detect_max_resolution(device_path: str) -> Tuple[int, int]:
    """
    Auto-detect maximum supported resolution for a camera.
    Tries common resolutions from highest to lowest.

    Args:
        device_path: Device path like /dev/video0

    Returns:
        (width, height) tuple of best supported resolution
    """
    # Common resolutions to try (highest first)
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
        return (1920, 1080)  # Default fallback

    # Set MJPEG for better high-res support on USB cameras
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    best = (640, 480)
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == w and actual_h == h:
            best = (w, h)
            break  # Found highest supported

    cap.release()
    return best


class CameraInfo:
    """Info about a discovered camera."""
    def __init__(self, device_path: str, resolution: Tuple[int, int]):
        self.device_path = device_path
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None


class MultiCamera:
    """
    Manages multiple USB cameras with persistent connections.
    Supports burst capture with sharpness-based frame selection.
    """

    def __init__(
        self,
        min_cameras: int = 2,
        burst_frames: int = 5,
        sharpness_threshold: float = 100.0,
        warmup_frames: int = 30
    ):
        """
        Initialize cameras with auto-discovery.

        Args:
            min_cameras: Minimum cameras required (fails if fewer found)
            burst_frames: Number of frames to capture for sharpness selection
            sharpness_threshold: Minimum acceptable sharpness (warning if below)
            warmup_frames: Frames to discard during warmup
        """
        self.min_cameras = min_cameras
        self.burst_frames = burst_frames
        self.sharpness_threshold = sharpness_threshold
        self.warmup_frames = warmup_frames

        self._cameras: List[CameraInfo] = []

        self._discover_and_init()

    def _discover_and_init(self):
        """Discover cameras and initialize them."""
        logging.info("[Camera] Starting auto-discovery...")

        device_paths = discover_cameras()

        if len(device_paths) < self.min_cameras:
            raise RuntimeError(
                f"Found {len(device_paths)} cameras, but {self.min_cameras} required. "
                f"Devices found: {device_paths}"
            )

        logging.info(f"[Camera] Found {len(device_paths)} cameras: {device_paths}")

        # Initialize each camera
        for idx, device_path in enumerate(device_paths):
            resolution = detect_max_resolution(device_path)
            info = CameraInfo(device_path, resolution)

            # Open camera
            device_id = int(device_path.replace('/dev/video', ''))
            cap = cv2.VideoCapture(device_id)

            if not cap.isOpened():
                logging.error(f"[Camera {idx+1}] Failed to open {device_path}")
                continue

            # Configure camera
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Enable auto settings
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto mode
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)

            # Warmup - let auto-exposure/focus settle
            logging.info(f"[Camera {idx+1}] Warming up {device_path} ({self.warmup_frames} frames)...")
            for _ in range(self.warmup_frames):
                cap.read()

            info.cap = cap
            self._cameras.append(info)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"[Camera {idx+1}] Ready: {device_path} @ {actual_w}x{actual_h}")

        if len(self._cameras) < self.min_cameras:
            raise RuntimeError(
                f"Only {len(self._cameras)} cameras initialized successfully, "
                f"but {self.min_cameras} required."
            )

    def _capture_burst(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """Capture burst of frames from a camera."""
        frames = []
        for _ in range(self.burst_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        return frames

    def _select_sharpest(self, frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        """
        Select sharpest frame from burst.

        Returns:
            (best_frame, sharpness) or (None, 0.0) if no frames
        """
        if not frames:
            return None, 0.0

        best_frame = None
        best_sharpness = 0.0

        for frame in frames:
            sharpness = calculate_sharpness(frame)
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame

        # Log warning if below threshold
        if best_sharpness < self.sharpness_threshold:
            logging.warning(
                f"Best sharpness {best_sharpness:.1f} below threshold {self.sharpness_threshold}"
            )

        return best_frame, best_sharpness

    def capture_sharpest(self, camera_index: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Capture burst and return sharpest frame.

        Args:
            camera_index: 0-based camera index

        Returns:
            (image, sharpness) tuple
        """
        if camera_index < 0 or camera_index >= len(self._cameras):
            logging.error(f"Invalid camera index: {camera_index}")
            return None, 0.0

        info = self._cameras[camera_index]
        if info.cap is None or not info.cap.isOpened():
            logging.error(f"Camera {camera_index} not available")
            return None, 0.0

        frames = self._capture_burst(info.cap)
        return self._select_sharpest(frames)

    def get_camera_count(self) -> int:
        """Get number of available cameras."""
        return len(self._cameras)

    def get_camera_info(self, camera_index: int) -> Optional[dict]:
        """Get info about a specific camera."""
        if camera_index < 0 or camera_index >= len(self._cameras):
            return None

        info = self._cameras[camera_index]
        return {
            'device': info.device_path,
            'resolution': info.resolution,
            'ready': info.cap is not None and info.cap.isOpened()
        }

    def get_all_camera_info(self) -> List[dict]:
        """Get info about all cameras."""
        return [self.get_camera_info(i) for i in range(len(self._cameras))]

    def release(self):
        """Release all camera resources."""
        for idx, info in enumerate(self._cameras):
            if info.cap is not None:
                info.cap.release()
                info.cap = None
                logging.info(f"[Camera {idx+1}] Released {info.device_path}")
        self._cameras.clear()

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
