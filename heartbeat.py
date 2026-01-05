"""
Cronitor heartbeat monitoring for ultra-minimal collector.
Sends periodic health status to Cronitor for external monitoring.
"""

import json
import time
import socket
import logging
import threading
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urlencode

import requests


class HeartbeatManager:
    """
    Manages periodic heartbeat pings to Cronitor.

    Tracks:
    - Camera health (updated by collector after each capture)
    - Storage stats (pending, total, synced)
    - Sync health (consecutive failures)
    - Server reachability (GET /health)
    """

    CRONITOR_BASE_URL = "https://cronitor.link/p"
    FAIL_THRESHOLD = 3  # consecutive sync failures before reporting 'fail'

    def __init__(
        self,
        cronitor_api_key: str,
        monitor_key: str,
        device_id: str,
        server_url: str,
        storage,  # Storage instance
        sync_manager,  # SyncManager instance
        interval: int = 60,
        timeout: int = 10
    ):
        self.cronitor_api_key = cronitor_api_key
        self.monitor_key = monitor_key
        self.device_id = device_id
        self.server_url = server_url.rstrip('/')
        self.storage = storage
        self.sync_manager = sync_manager
        self.interval = interval
        self.timeout = timeout

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Camera status tracking (updated by collector)
        self._camera_status: Dict[int, Dict] = {}

        logging.info(f"[Heartbeat] Initialized (interval: {interval}s, monitor: {monitor_key})")

    def update_camera_status(self, camera_idx: int, ok: bool, sharpness: float):
        """
        Update camera status after a capture.
        Called by collector after each capture cycle.
        """
        with self._lock:
            self._camera_status[camera_idx] = {
                'ok': ok,
                'sharpness': sharpness,
                'last_update': datetime.now().isoformat()
            }

    def _check_server_health(self) -> Dict:
        """
        Check server health via GET /health endpoint.

        Returns:
            {
                'reachable': bool,
                's3_connected': bool,
                'latency_ms': int,
                'version': str
            }
        """
        result = {
            'reachable': False,
            's3_connected': False,
            'latency_ms': 0,
            'version': 'unknown'
        }

        try:
            start = time.time()
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout
            )
            latency_ms = int((time.time() - start) * 1000)

            if response.status_code == 200:
                result['reachable'] = True
                result['latency_ms'] = latency_ms

                try:
                    data = response.json()
                    result['s3_connected'] = data.get('s3_connected', False)
                    result['version'] = data.get('version', 'unknown')
                except (json.JSONDecodeError, KeyError):
                    pass

        except requests.exceptions.Timeout:
            logging.warning("[Heartbeat] Server health check timed out")
        except requests.exceptions.ConnectionError:
            logging.warning("[Heartbeat] Server unreachable")
        except Exception as e:
            logging.warning(f"[Heartbeat] Server health check error: {e}")

        return result

    def _collect_health(self) -> Dict:
        """
        Collect all health metrics.

        Returns:
            Complete health status dict
        """
        # Camera status
        with self._lock:
            cameras = {}
            for idx, status in self._camera_status.items():
                cameras[f"camera_{idx + 1}"] = {
                    'ok': status['ok'],
                    'sharpness': round(status['sharpness'], 1)
                }

        # Storage stats
        storage_stats = self.storage.get_stats()

        # Sync status
        sync_status = self.sync_manager.get_status()

        # Server health
        server_health = self._check_server_health()

        return {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'cameras': cameras,
            'storage': {
                'pending': storage_stats.get('pending', 0),
                'total': storage_stats.get('total_captures', 0),
                'synced': storage_stats.get('synced_captures', 0)
            },
            'sync': {
                'ok': sync_status.get('consecutive_failures', 0) < self.FAIL_THRESHOLD,
                'failures': sync_status.get('consecutive_failures', 0),
                'last_sync': sync_status.get('last_sync')
            },
            'server': server_health
        }

    def _determine_state(self, health: Dict) -> str:
        """
        Determine Cronitor state based on health metrics.

        Returns:
            'ok' or 'fail'
        """
        # FAIL if sync has >= FAIL_THRESHOLD consecutive failures
        if health['sync']['failures'] >= self.FAIL_THRESHOLD:
            logging.warning(f"[Heartbeat] State=fail: {health['sync']['failures']} sync failures")
            return 'fail'

        # FAIL if server unreachable
        if not health['server']['reachable']:
            logging.warning("[Heartbeat] State=fail: server unreachable")
            return 'fail'

        # FAIL if S3 disconnected
        if not health['server']['s3_connected']:
            logging.warning("[Heartbeat] State=fail: S3 disconnected")
            return 'fail'

        # OK (camera issues are warnings, not failures)
        return 'ok'

    def _send_ping(self, state: str, health: Dict) -> bool:
        """
        Send heartbeat ping to Cronitor.

        Args:
            state: 'ok' or 'fail'
            health: Health dict to include in message

        Returns:
            True if successful
        """
        # Build Cronitor URL with params
        params = {
            'state': state,
            'host': socket.gethostname(),
            'message': json.dumps(health, separators=(',', ':'))  # Compact JSON
        }

        # Add metrics
        metrics = []
        metrics.append(f"count:pending:{health['storage']['pending']}")
        metrics.append(f"count:synced:{health['storage']['synced']}")

        # Count camera failures
        camera_failures = sum(1 for cam in health['cameras'].values() if not cam['ok'])
        if camera_failures > 0:
            metrics.append(f"error_count:camera_failures:{camera_failures}")

        if metrics:
            params['metric'] = ','.join(metrics)

        url = f"{self.CRONITOR_BASE_URL}/{self.cronitor_api_key}/{self.monitor_key}"

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                logging.info(f"[Heartbeat] Ping sent: state={state}, pending={health['storage']['pending']}")
                return True
            else:
                logging.warning(f"[Heartbeat] Ping failed: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logging.warning("[Heartbeat] Ping timed out")
        except requests.exceptions.ConnectionError:
            logging.warning("[Heartbeat] Cronitor unreachable")
        except Exception as e:
            logging.error(f"[Heartbeat] Ping error: {e}")

        return False

    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        # Initial delay to let system stabilize
        time.sleep(10)

        while self._running:
            try:
                health = self._collect_health()
                state = self._determine_state(health)
                self._send_ping(state, health)

            except Exception as e:
                logging.error(f"[Heartbeat] Loop error: {e}")

            # Sleep in small increments for graceful shutdown
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)

    def start(self):
        """Start background heartbeat thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logging.info(f"[Heartbeat] Started (interval: {self.interval}s)")

    def stop(self):
        """Stop heartbeat thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logging.info("[Heartbeat] Stopped")

    def force_ping(self) -> bool:
        """Force an immediate heartbeat ping (for testing)."""
        health = self._collect_health()
        state = self._determine_state(health)
        return self._send_ping(state, health)
