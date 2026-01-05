#!/usr/bin/env python3
"""
Drain queue script - uploads all pending captures while service is stopped.

Usage:
    python3 drain_queue.py [options]

Options:
    --path PATH       Base path for captures (default: ./data/captures)
    --batch-size N    Uploads per batch (default: 20)
    --dry-run         Show what would be uploaded without uploading
    --env PATH        Path to .env file (default: .env)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import requests


def load_env(env_path: str):
    """Load environment variables from .env file."""
    if not os.path.exists(env_path):
        return {}

    env = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                env[key.strip()] = value.strip()
    return env


def load_queue(base_path: Path) -> list:
    """Load the queue from queue.json."""
    queue_file = base_path / 'queue.json'
    if not queue_file.exists():
        return []

    with open(queue_file, 'r') as f:
        return json.load(f)


def save_queue(base_path: Path, queue: list):
    """Save the queue to queue.json."""
    queue_file = base_path / 'queue.json'
    with open(queue_file, 'w') as f:
        json.dump(queue, f)


def load_metadata(metadata_path: str) -> dict:
    """Load metadata from a JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def mark_synced(metadata_path: str):
    """Mark a capture as synced."""
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        meta['synced'] = True
        meta['synced_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"  Warning: Failed to mark synced: {e}")


def upload_batch(captures: list, base_path: Path, server_url: str, api_key: str, timeout: int = 120) -> list:
    """
    Upload a batch of captures to the server.

    Returns:
        List of successfully uploaded capture IDs
    """
    if not captures:
        return []

    images_dir = base_path / 'images'

    # Build metadata array and files list
    metadata_array = []
    files_list = []
    files_to_close = []

    for capture in captures:
        meta = {
            'id': capture['id'],
            'timestamp': capture['timestamp'],
            'trigger': capture.get('trigger', 'interval'),
            'device_id': capture.get('device_id', 'unknown'),
            'telemetry': {
                'sharpness': capture.get('sharpness', {})
            },
            'gps': None
        }
        metadata_array.append(meta)

        # Add image files
        for cam_key, filename in capture.get('images', {}).items():
            filepath = images_dir / filename
            if filepath.exists():
                server_filename = f"{capture['id']}_{cam_key}.jpg"
                f = open(filepath, 'rb')
                files_to_close.append(f)
                files_list.append(('files', (server_filename, f, 'image/jpeg')))

    try:
        # Ensure URL ends with batch endpoint
        if not server_url.endswith('/api/v1/captures/batch'):
            url = server_url.rstrip('/') + '/api/v1/captures/batch'
        else:
            url = server_url

        headers = {'X-API-Key': api_key}
        data = {'captures': json.dumps(metadata_array)}

        response = requests.post(url, headers=headers, data=data, files=files_list, timeout=timeout)

        if response.status_code in (200, 201):
            result = response.json()
            synced_ids = [
                r['capture_id']
                for r in result.get('results', [])
                if r.get('success')
            ]
            return synced_ids
        else:
            print(f"  Server error: {response.status_code} - {response.text[:200]}")
            return []

    except requests.exceptions.Timeout:
        print(f"  Timeout uploading batch")
        return []
    except requests.exceptions.ConnectionError:
        print(f"  Connection error")
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []
    finally:
        for f in files_to_close:
            try:
                f.close()
            except:
                pass


def drain_queue(base_path: Path, server_url: str, api_key: str, batch_size: int = 20, dry_run: bool = False):
    """Drain the entire queue by uploading all pending captures."""

    print(f"\n{'='*60}")
    print("Queue Drain Script")
    print(f"{'='*60}")
    print(f"Base path: {base_path}")
    print(f"Server: {server_url}")
    print(f"Batch size: {batch_size}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    queue = load_queue(base_path)
    total = len(queue)

    if total == 0:
        print("Queue is empty. Nothing to upload.")
        return

    print(f"Found {total} pending captures in queue.\n")

    if dry_run:
        print("DRY RUN - No uploads will be performed.")
        print(f"Would upload {total} captures in {(total + batch_size - 1) // batch_size} batches.")
        return

    uploaded = 0
    failed = 0
    batch_num = 0
    total_batches = (total + batch_size - 1) // batch_size

    start_time = time.time()

    while queue:
        batch_num += 1
        batch_paths = queue[:batch_size]

        # Load metadata for this batch
        captures = []
        valid_paths = []
        for path in batch_paths:
            meta = load_metadata(path)
            if meta:
                captures.append(meta)
                valid_paths.append(path)
            else:
                # Remove invalid entry from queue
                queue.remove(path)
                print(f"  Skipped invalid: {path}")

        if not captures:
            continue

        print(f"Batch {batch_num}/{total_batches}: Uploading {len(captures)} captures...", end=' ', flush=True)

        synced_ids = upload_batch(captures, base_path, server_url, api_key)

        if synced_ids:
            print(f"OK ({len(synced_ids)} uploaded)")
            uploaded += len(synced_ids)

            # Mark synced and remove from queue
            for path, meta in zip(valid_paths, captures):
                if meta['id'] in synced_ids:
                    mark_synced(path)
                    if path in queue:
                        queue.remove(path)

            # Save queue after each batch (resume-friendly)
            save_queue(base_path, queue)
        else:
            print("FAILED")
            failed += len(captures)
            # Remove failed items from queue to avoid infinite loop
            for path in valid_paths:
                if path in queue:
                    queue.remove(path)
            save_queue(base_path, queue)

        # Small delay between batches
        if queue:
            time.sleep(0.5)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("Drain Complete")
    print(f"{'='*60}")
    print(f"Total uploaded: {uploaded}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Queue remaining: {len(load_queue(base_path))}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Drain upload queue')
    parser.add_argument('--path', default='./data/captures', help='Base path for captures')
    parser.add_argument('--batch-size', type=int, default=20, help='Uploads per batch')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded')
    parser.add_argument('--env', default='.env', help='Path to .env file')
    args = parser.parse_args()

    # Load environment
    env = load_env(args.env)

    server_url = env.get('SERVER_URL', os.environ.get('SERVER_URL', ''))
    api_key = env.get('API_KEY', os.environ.get('API_KEY', ''))

    if not server_url:
        print("Error: SERVER_URL not set in .env or environment")
        sys.exit(1)
    if not api_key:
        print("Error: API_KEY not set in .env or environment")
        sys.exit(1)

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Error: Path not found: {base_path}")
        sys.exit(1)

    drain_queue(base_path, server_url, api_key, args.batch_size, args.dry_run)


if __name__ == '__main__':
    main()
