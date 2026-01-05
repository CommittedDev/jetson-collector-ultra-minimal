#!/usr/bin/env python3
"""
Cleanup script for corrupted metadata files in the capture queue.
Run this on the Jetson to fix storage errors.

Usage:
    python3 cleanup_queue.py [path_to_captures]

Default path: /opt/collector-ultra-minimal/data/captures
"""

import os
import sys
import json
from pathlib import Path


def cleanup_queue(base_path: str):
    """Clean up corrupted metadata files and rebuild the queue."""
    base = Path(base_path)
    metadata_dir = base / 'metadata'
    queue_file = base / 'queue.json'

    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        sys.exit(1)

    print(f"Scanning {metadata_dir}...")

    valid_entries = []
    corrupted_count = 0
    empty_count = 0

    for filename in sorted(os.listdir(metadata_dir)):
        if not filename.endswith('.json'):
            continue

        filepath = metadata_dir / filename

        # Check if file is empty
        if filepath.stat().st_size == 0:
            print(f"  Removing empty file: {filename}")
            filepath.unlink()
            empty_count += 1
            continue

        # Try to parse JSON
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Validate required fields
            if 'id' not in data or 'timestamp' not in data:
                print(f"  Removing invalid (missing fields): {filename}")
                filepath.unlink()
                corrupted_count += 1
                continue

            # Check if already synced
            if data.get('synced', False):
                # Don't add synced items to queue
                continue

            # Valid unsynced entry
            valid_entries.append(str(filepath))

        except json.JSONDecodeError as e:
            print(f"  Removing corrupted JSON: {filename} ({e})")
            filepath.unlink()
            corrupted_count += 1
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            corrupted_count += 1

    # Backup old queue
    if queue_file.exists():
        backup = queue_file.with_suffix('.json.bak')
        queue_file.rename(backup)
        print(f"\nBacked up old queue to {backup}")

    # Write new queue
    with open(queue_file, 'w') as f:
        json.dump(valid_entries, f, indent=2)

    print(f"\n=== Cleanup Complete ===")
    print(f"Empty files removed: {empty_count}")
    print(f"Corrupted files removed: {corrupted_count}")
    print(f"Valid pending entries: {len(valid_entries)}")
    print(f"Queue saved to: {queue_file}")


def main():
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = '/opt/collector-ultra-minimal/data/captures'

    print(f"Cleanup queue script")
    print(f"Base path: {base_path}")
    print()

    cleanup_queue(base_path)


if __name__ == '__main__':
    main()
