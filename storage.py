"""
Local storage and sync manager for ultra-minimal collector.
Persistent queue survives reboots, batch sync with offline resilience.
"""

import os
import json
import time
import uuid
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import requests


@dataclass
class CaptureMetadata:
    """Metadata for a capture."""
    id: str
    timestamp: str
    device_id: str
    trigger: str  # Always 'interval' for ultra-minimal
    images: Dict[str, str]  # {'camera_1': 'filename.jpg', ...}
    sharpness: Dict[str, float]  # {'camera_1': 123.4, ...}
    synced: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Storage:
    """
    Local storage for captured images.

    Directory structure:
        base_path/
        ├── images/
        │   ├── camera_1_20251224_143052.jpg
        │   ├── camera_2_20251224_143052.jpg
        │   └── ...
        ├── metadata/
        │   └── {capture_id}.json
        ├── queue.json       # Persistent queue (survives reboots)
        └── stats.json       # Collection statistics
    """

    def __init__(
        self,
        base_path: str = '/data/captures',
        device_id: str = 'collector',
        max_size_gb: float = 10.0,
        image_quality: int = 85
    ):
        self.base_path = Path(base_path)
        self.device_id = device_id
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.image_quality = image_quality

        # Create directories
        self.images_dir = self.base_path / 'images'
        self.metadata_dir = self.base_path / 'metadata'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Queue and stats files
        self._queue_file = self.base_path / 'queue.json'
        self._stats_file = self.base_path / 'stats.json'

        # Load persistent queue
        self._queue: List[str] = self._load_queue()
        self._stats = self._load_stats()

        self._lock = threading.Lock()

        logging.info(f"[Storage] Initialized at {self.base_path}")
        logging.info(f"[Storage] Pending captures: {len(self._queue)}")

    def _load_queue(self) -> List[str]:
        """Load queue from disk."""
        if self._queue_file.exists():
            try:
                with open(self._queue_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"[Storage] Failed to load queue: {e}")
        return []

    def _save_queue(self):
        """Persist queue to disk."""
        try:
            with open(self._queue_file, 'w') as f:
                json.dump(self._queue, f)
        except Exception as e:
            logging.error(f"[Storage] Failed to save queue: {e}")

    def _load_stats(self) -> Dict:
        """Load statistics."""
        if self._stats_file.exists():
            try:
                with open(self._stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'total_captures': 0, 'synced_captures': 0, 'last_capture': None}

    def _save_stats(self):
        """Persist statistics."""
        try:
            with open(self._stats_file, 'w') as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            logging.error(f"[Storage] Failed to save stats: {e}")

    def save(
        self,
        images: Dict[int, Optional[np.ndarray]],
        timestamp: datetime,
        sharpness: Dict[int, float]
    ) -> Optional[str]:
        """
        Save captured images with metadata.

        Args:
            images: Dict mapping camera_index to image (can be None)
            timestamp: Capture timestamp
            sharpness: Dict mapping camera_index to sharpness score

        Returns:
            Capture ID or None if save failed
        """
        with self._lock:
            capture_id = uuid.uuid4().hex[:12]
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            iso_timestamp = timestamp.isoformat()

            saved_images = {}
            saved_sharpness = {}

            # Save each camera image
            for cam_idx, image in images.items():
                if image is None:
                    continue

                camera_name = f"camera_{cam_idx + 1}"
                filename = f"{camera_name}_{timestamp_str}.jpg"
                filepath = self.images_dir / filename

                try:
                    cv2.imwrite(
                        str(filepath),
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
                    )
                    saved_images[camera_name] = filename
                    saved_sharpness[camera_name] = sharpness.get(cam_idx, 0.0)
                except Exception as e:
                    logging.error(f"[Storage] Failed to save {filename}: {e}")

            if not saved_images:
                return None

            # Create metadata
            metadata = CaptureMetadata(
                id=capture_id,
                timestamp=iso_timestamp,
                device_id=self.device_id,
                trigger='interval',
                images=saved_images,
                sharpness=saved_sharpness
            )

            # Save metadata
            metadata_path = self.metadata_dir / f"{capture_id}.json"
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
            except Exception as e:
                logging.error(f"[Storage] Failed to save metadata: {e}")
                return None

            # Add to queue
            self._queue.append(str(metadata_path))
            self._save_queue()

            # Update stats
            self._stats['total_captures'] += 1
            self._stats['last_capture'] = iso_timestamp
            self._save_stats()

            return capture_id

    def get_pending_captures(self, limit: int = 20) -> List[Dict]:
        """Get pending captures for sync."""
        captures = []
        for metadata_path in self._queue[:limit]:
            try:
                with open(metadata_path, 'r') as f:
                    captures.append(json.load(f))
            except Exception as e:
                logging.warning(f"[Storage] Failed to load {metadata_path}: {e}")
        return captures

    def get_pending_count(self) -> int:
        """Get number of pending captures."""
        return len(self._queue)

    def mark_synced(self, capture_ids: List[str]):
        """Mark captures as synced and remove from queue."""
        with self._lock:
            paths_to_remove = []

            for metadata_path in self._queue[:]:
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)

                    if meta['id'] in capture_ids:
                        meta['synced'] = True
                        meta['synced_at'] = datetime.now().isoformat()
                        with open(metadata_path, 'w') as f:
                            json.dump(meta, f, indent=2)
                        paths_to_remove.append(metadata_path)
                        self._stats['synced_captures'] += 1
                except Exception as e:
                    logging.warning(f"[Storage] Error marking synced: {e}")

            for path in paths_to_remove:
                if path in self._queue:
                    self._queue.remove(path)

            self._save_queue()
            self._save_stats()

            if paths_to_remove:
                logging.info(f"[Storage] Marked {len(paths_to_remove)} captures as synced")

    def get_image_path(self, filename: str) -> Path:
        """Get full path to an image file."""
        return self.images_dir / filename

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            **self._stats,
            'pending': len(self._queue),
            'storage_path': str(self.base_path)
        }

    def cleanup_old_synced(self, retention_days: int = 3) -> int:
        """
        Delete synced captures older than retention_days.

        Args:
            retention_days: Days to keep synced captures (0 = keep forever)

        Returns:
            Number of captures deleted
        """
        if retention_days <= 0:
            return 0

        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0

        with self._lock:
            # Scan all metadata files (not just queue - queue only has unsynced)
            for filename in list(self.metadata_dir.iterdir()):
                if not filename.suffix == '.json':
                    continue

                try:
                    with open(filename, 'r') as f:
                        meta = json.load(f)

                    # Only delete synced captures
                    if not meta.get('synced', False):
                        continue

                    # Check sync timestamp (or capture timestamp as fallback)
                    sync_time_str = meta.get('synced_at') or meta.get('timestamp')
                    if not sync_time_str:
                        continue

                    # Parse timestamp
                    try:
                        sync_time = datetime.fromisoformat(sync_time_str.replace('Z', '+00:00'))
                        # Remove timezone info for comparison
                        if sync_time.tzinfo:
                            sync_time = sync_time.replace(tzinfo=None)
                    except ValueError:
                        continue

                    # Skip if not old enough
                    if sync_time > cutoff:
                        continue

                    # Delete image files
                    for cam_key, img_filename in meta.get('images', {}).items():
                        img_path = self.images_dir / img_filename
                        if img_path.exists():
                            try:
                                img_path.unlink()
                            except Exception as e:
                                logging.warning(f"[Storage] Failed to delete {img_path}: {e}")

                    # Delete metadata file
                    try:
                        filename.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logging.warning(f"[Storage] Failed to delete {filename}: {e}")

                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"[Storage] Error reading {filename}: {e}")
                except Exception as e:
                    logging.warning(f"[Storage] Cleanup error for {filename}: {e}")

        if deleted_count > 0:
            logging.info(f"[Storage] Cleaned up {deleted_count} old synced captures (>{retention_days} days)")

        return deleted_count


class SyncManager:
    """
    Background sync manager for batch uploads.

    - Runs in background thread
    - Batches captures according to batch_size
    - Uses server batch API endpoint
    - Handles offline scenarios with retry
    """

    def __init__(
        self,
        storage: Storage,
        server_url: str,
        api_key: str,
        batch_size: int = 20,
        sync_interval: int = 60,
        catchup_delay: int = 2,
        timeout: int = 60,
        max_retries: int = 3,
        retention_days: int = 3
    ):
        self.storage = storage
        # Ensure URL ends with batch endpoint
        base_url = server_url.rstrip('/')
        if not base_url.endswith('/api/v1/captures/batch'):
            self.server_url = base_url + '/api/v1/captures/batch'
        else:
            self.server_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.sync_interval = sync_interval  # Idle interval when queue empty
        self.catchup_delay = catchup_delay  # Delay between batches when catching up
        self.timeout = timeout
        self.max_retries = max_retries
        self.retention_days = retention_days

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_sync: Optional[datetime] = None
        self._consecutive_failures = 0
        self._cleanup_counter = 0  # Run cleanup every N sync cycles

    def start(self):
        """Start background sync thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logging.info(f"[Sync] Started (interval: {self.sync_interval}s, batch: {self.batch_size})")
        logging.info(f"[Sync] Server: {self.server_url}")

    def stop(self):
        """Stop sync thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logging.info("[Sync] Stopped")

    def _sync_loop(self):
        """Background sync loop."""
        # Initial delay to let captures accumulate
        time.sleep(10)

        while self._running:
            try:
                pending = self.storage.get_pending_count()

                # Sync if we have captures
                if pending > 0:
                    uploaded, _ = self._sync_batch()

                    # Run cleanup periodically after successful syncs
                    if uploaded > 0 and self.retention_days > 0:
                        self._cleanup_counter += 1
                        if self._cleanup_counter >= 10:  # Every 10 sync cycles
                            self.storage.cleanup_old_synced(self.retention_days)
                            self._cleanup_counter = 0

            except Exception as e:
                logging.error(f"[Sync] Loop error: {e}")
                self._consecutive_failures += 1

            # Determine sleep time based on queue state
            remaining = self.storage.get_pending_count()

            if self._consecutive_failures > 0:
                # Exponential backoff on failures
                sleep_time = min(self.sync_interval * (2 ** self._consecutive_failures), 600)
                logging.info(f"[Sync] Backoff: sleeping {sleep_time}s")
            elif remaining > 0:
                # Queue has items - use short catchup delay
                sleep_time = self.catchup_delay
            else:
                # Queue empty - use idle interval
                sleep_time = self.sync_interval

            # Sleep in small increments to allow stopping
            for _ in range(int(sleep_time)):
                if not self._running:
                    break
                time.sleep(1)

    def _sync_batch(self) -> Tuple[int, int]:
        """
        Sync a batch of captures to server.

        Returns:
            (synced_count, failed_count)
        """
        captures = self.storage.get_pending_captures(self.batch_size)
        if not captures:
            return 0, 0

        logging.info(f"[Sync] Uploading {len(captures)} captures...")

        for attempt in range(self.max_retries):
            files_to_close = []
            try:
                # Build metadata array
                metadata_array = []
                files_list = []

                for capture in captures:
                    # Server expects: id, timestamp, trigger, device_id, telemetry, gps
                    meta = {
                        'id': capture['id'],
                        'timestamp': capture['timestamp'],
                        'trigger': capture['trigger'],
                        'device_id': capture['device_id'],
                        'telemetry': {
                            'sharpness': capture.get('sharpness', {})
                        },
                        'gps': None
                    }
                    metadata_array.append(meta)

                    # Add image files
                    for cam_key, filename in capture.get('images', {}).items():
                        filepath = self.storage.get_image_path(filename)
                        if filepath.exists():
                            # Server expects: {capture_id}_{type}.{ext}
                            # Send as camera_1, camera_2, etc.
                            server_filename = f"{capture['id']}_{cam_key}.jpg"

                            f = open(filepath, 'rb')
                            files_to_close.append(f)
                            files_list.append(
                                ('files', (server_filename, f, 'image/jpeg'))
                            )

                # Make request
                headers = {'X-API-Key': self.api_key}
                data = {'captures': json.dumps(metadata_array)}

                response = requests.post(
                    self.server_url,
                    headers=headers,
                    data=data,
                    files=files_list,
                    timeout=self.timeout
                )

                if response.status_code in (200, 201):
                    result = response.json()
                    synced_ids = [
                        r['capture_id']
                        for r in result.get('results', [])
                        if r.get('success')
                    ]

                    if synced_ids:
                        self.storage.mark_synced(synced_ids)

                    self._consecutive_failures = 0
                    self._last_sync = datetime.now()

                    uploaded = result.get('uploaded', len(synced_ids))
                    failed = result.get('failed', 0)
                    logging.info(f"[Sync] Complete: uploaded={uploaded}, failed={failed}")
                    return uploaded, failed
                else:
                    logging.warning(f"[Sync] Server error: {response.status_code} - {response.text[:200]}")

            except requests.exceptions.Timeout:
                logging.warning(f"[Sync] Timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                logging.warning(f"[Sync] Connection error (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logging.error(f"[Sync] Unexpected error: {e}")
            finally:
                for f in files_to_close:
                    try:
                        f.close()
                    except Exception:
                        pass

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff between retries

        self._consecutive_failures += 1
        return 0, len(captures)

    def force_sync(self) -> Tuple[int, int]:
        """Force an immediate sync (for testing/manual trigger)."""
        return self._sync_batch()

    def get_status(self) -> Dict:
        """Get sync status."""
        return {
            'running': self._running,
            'last_sync': self._last_sync.isoformat() if self._last_sync else None,
            'consecutive_failures': self._consecutive_failures,
            'server_url': self.server_url
        }
