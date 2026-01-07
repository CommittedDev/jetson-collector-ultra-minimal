#!/usr/bin/env python3
"""
Ultra-minimal image collector for Jetson.
No OCR, no state machine, no wizard - just dual camera capture with sharpness selection.

Cameras are auto-discovered at startup. Just plug them into any USB port.
"""

import os
import sys
import time
import signal
import socket
import logging
from datetime import datetime

from camera import MultiCamera


def sd_notify(message: str):
    """Send notification to systemd (if running under systemd)."""
    notify_socket = os.environ.get('NOTIFY_SOCKET')
    if not notify_socket:
        return False

    try:
        if notify_socket.startswith('@'):
            notify_socket = '\0' + notify_socket[1:]

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(notify_socket)
        sock.sendall(message.encode())
        sock.close()
        return True
    except Exception:
        return False


def sd_notify_ready():
    """Tell systemd the service is ready."""
    sd_notify('READY=1')


def sd_notify_watchdog():
    """Send watchdog keep-alive to systemd."""
    sd_notify('WATCHDOG=1')


from storage import Storage, SyncManager
from heartbeat import HeartbeatManager


def get_config():
    """Load configuration from environment variables."""
    return {
        'capture_interval': int(os.environ.get('CAPTURE_INTERVAL_SECONDS', '10')),
        'batch_size': int(os.environ.get('BATCH_SIZE', '20')),
        'device_id': os.environ.get('DEVICE_ID', '') or socket.gethostname(),
        'server_url': os.environ.get('SERVER_URL', ''),
        'api_key': os.environ.get('API_KEY', ''),
        'burst_frames': int(os.environ.get('BURST_FRAMES', '5')),
        'sharpness_threshold': float(os.environ.get('SHARPNESS_THRESHOLD', '250.0')),
        'storage_path': os.environ.get('STORAGE_PATH', './data/captures'),
        'max_storage_gb': float(os.environ.get('MAX_STORAGE_GB', '10.0')),
        'sync_interval': int(os.environ.get('SYNC_IDLE_INTERVAL_SECONDS', '60')),
        'catchup_delay': int(os.environ.get('SYNC_CATCHUP_DELAY_SECONDS', '2')),
        'min_cameras': int(os.environ.get('MIN_CAMERAS', '2')),
        # Cronitor heartbeat (optional)
        'cronitor_api_key': os.environ.get('CRONITOR_API_KEY', ''),
        'cronitor_monitor_key': os.environ.get('CRONITOR_MONITOR_KEY', ''),
        'heartbeat_interval': int(os.environ.get('HEARTBEAT_INTERVAL_SECONDS', '60')),
        # Retention policy
        'retention_days': int(os.environ.get('RETENTION_DAYS', '3')),
    }


def setup_logging():
    """Configure logging to stdout for systemd/Docker."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_config(config: dict):
    """Validate required configuration."""
    errors = []

    if not config['server_url']:
        errors.append("SERVER_URL environment variable is required")
    if not config['api_key']:
        errors.append("API_KEY environment variable is required")

    if errors:
        for error in errors:
            logging.error(error)
        sys.exit(1)


class UltraMinimalCollector:
    """Ultra-minimal dual camera collector with sharpness selection."""

    def __init__(self, config: dict):
        self.config = config
        self.camera = None
        self.storage = None
        self.sync_manager = None
        self.heartbeat_manager = None
        self._running = False
        self._capture_count = 0

    def start(self):
        """Initialize cameras and storage."""
        logging.info("=" * 60)
        logging.info("Ultra-Minimal Collector Starting")
        logging.info("=" * 60)
        logging.info(f"Device ID: {self.config['device_id']}")
        logging.info(f"Capture interval: {self.config['capture_interval']}s")
        logging.info(f"Burst frames: {self.config['burst_frames']}")
        logging.info(f"Sharpness threshold: {self.config['sharpness_threshold']}")
        logging.info(f"Batch size: {self.config['batch_size']}")
        logging.info(f"Sync idle interval: {self.config['sync_interval']}s")
        logging.info(f"Sync catchup delay: {self.config['catchup_delay']}s")
        logging.info(f"Storage path: {self.config['storage_path']}")
        logging.info("=" * 60)

        # Initialize cameras (auto-discovery + warmup happens here)
        logging.info("Initializing cameras...")
        self.camera = MultiCamera(
            min_cameras=self.config['min_cameras'],
            burst_frames=self.config['burst_frames'],
            sharpness_threshold=self.config['sharpness_threshold'],
            warmup_frames=30
        )

        # Log camera info
        for i, info in enumerate(self.camera.get_all_camera_info()):
            logging.info(f"Camera {i+1}: {info['device']} @ {info['resolution'][0]}x{info['resolution'][1]}")

        # Initialize storage
        logging.info("Initializing storage...")
        self.storage = Storage(
            base_path=self.config['storage_path'],
            device_id=self.config['device_id'],
            max_size_gb=self.config['max_storage_gb']
        )

        # Initialize sync manager (background thread)
        logging.info("Initializing sync manager...")
        self.sync_manager = SyncManager(
            storage=self.storage,
            server_url=self.config['server_url'],
            api_key=self.config['api_key'],
            batch_size=self.config['batch_size'],
            sync_interval=self.config['sync_interval'],
            catchup_delay=self.config['catchup_delay'],
            retention_days=self.config['retention_days']
        )
        self.sync_manager.start()

        # Initialize heartbeat manager (optional - only if Cronitor config present)
        if self.config['cronitor_api_key'] and self.config['cronitor_monitor_key']:
            logging.info("Initializing heartbeat manager...")
            self.heartbeat_manager = HeartbeatManager(
                cronitor_api_key=self.config['cronitor_api_key'],
                monitor_key=self.config['cronitor_monitor_key'],
                device_id=self.config['device_id'],
                server_url=self.config['server_url'],
                storage=self.storage,
                sync_manager=self.sync_manager,
                interval=self.config['heartbeat_interval']
            )
            self.heartbeat_manager.start()
        else:
            logging.info("Heartbeat disabled (CRONITOR_API_KEY/CRONITOR_MONITOR_KEY not set)")

        self._running = True
        logging.info("=" * 60)
        logging.info("Collector started successfully!")
        logging.info("=" * 60)

        # Notify systemd we're ready
        sd_notify_ready()

    def stop(self):
        """Cleanup resources."""
        logging.info("Stopping collector...")
        self._running = False

        if self.heartbeat_manager:
            self.heartbeat_manager.stop()
        if self.sync_manager:
            self.sync_manager.stop()
        if self.camera:
            self.camera.release()

        logging.info("Collector stopped")

    def run(self):
        """Main capture loop."""
        logging.info(f"Entering capture loop (interval: {self.config['capture_interval']}s)")

        while self._running:
            try:
                self._capture_cycle()
            except Exception as e:
                logging.error(f"Capture error: {e}")

            # Notify systemd watchdog we're still alive
            sd_notify_watchdog()

            # Sleep until next capture
            for _ in range(self.config['capture_interval']):
                if not self._running:
                    break
                time.sleep(1)

    def _capture_cycle(self):
        """Execute one capture cycle for all cameras."""
        timestamp = datetime.now()

        images = {}
        sharpness = {}

        # Capture from all cameras
        camera_count = self.camera.get_camera_count()
        for cam_idx in range(camera_count):
            image, sharp = self.camera.capture_sharpest(cam_idx)
            if image is not None:
                images[cam_idx] = image
                sharpness[cam_idx] = sharp

            # Update heartbeat with camera status
            if self.heartbeat_manager:
                self.heartbeat_manager.update_camera_status(
                    camera_idx=cam_idx,
                    ok=(image is not None),
                    sharpness=sharp
                )

        # Log capture results
        status_parts = []
        for cam_idx in range(camera_count):
            if cam_idx in images:
                status_parts.append(f"cam{cam_idx+1}={sharpness[cam_idx]:.0f}")
            else:
                status_parts.append(f"cam{cam_idx+1}=FAIL")

        status_str = ", ".join(status_parts)
        self._capture_count += 1

        # Save to local storage
        if images:
            capture_id = self.storage.save(
                images=images,
                timestamp=timestamp,
                sharpness=sharpness
            )
            if capture_id:
                pending = self.storage.get_pending_count()
                logging.info(f"Capture #{self._capture_count}: {status_str} -> {capture_id} (pending: {pending})")
            else:
                logging.warning(f"Capture #{self._capture_count}: {status_str} -> SAVE FAILED")
        else:
            logging.warning(f"Capture #{self._capture_count}: {status_str} -> NO IMAGES")


def main():
    """Entry point."""
    setup_logging()

    logging.info("Loading configuration...")
    config = get_config()
    validate_config(config)

    collector = UltraMinimalCollector(config)

    # Signal handlers for graceful shutdown
    def shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logging.info(f"Received {sig_name}")
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        collector.start()
        collector.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        collector.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
