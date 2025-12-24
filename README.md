# Jetson Collector - Ultra Minimal

Stripped-down image collector for Jetson Orin Nano. Captures from multiple cameras with sharpness selection, saves locally, and batch syncs to server.

## Features

- **Plug & Play Cameras**: Auto-discovers USB cameras on any port
- **Sharp Images**: Burst capture (5 frames), picks sharpest via Laplacian variance
- **Offline Resilient**: Persistent queue survives reboots
- **Batch Sync**: Upload in batches, exponential backoff on failures
- **Auto-Start**: systemd service for boot startup
- **Minimal Footprint**: ~80MB memory (no OCR, no wizard)

## Quick Start

```bash
# 1. Run setup (one-time)
./setup.sh

# 2. Edit configuration
nano .env
# Set SERVER_URL and API_KEY

# 3. Test manually
./run.sh

# 4. Install for auto-boot
sudo cp -r . /opt/collector-ultra-minimal
sudo cp collector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable collector.service
sudo systemctl start collector.service

# 5. Check logs
journalctl -u collector -f
```

## Configuration

All settings via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_URL` | (required) | Capture server URL |
| `API_KEY` | (required) | API authentication key |
| `DEVICE_ID` | hostname | Device identifier |
| `CAPTURE_INTERVAL_SECONDS` | 10 | Seconds between captures |
| `BATCH_SIZE` | 20 | Images per sync batch |
| `BURST_FRAMES` | 5 | Frames per burst for sharpness |
| `SHARPNESS_THRESHOLD` | 100.0 | Minimum Laplacian variance |
| `MIN_CAMERAS` | 2 | Minimum cameras required |
| `MAX_STORAGE_GB` | 10.0 | Local storage limit |
| `SYNC_INTERVAL_SECONDS` | 60 | Sync check interval |

**No camera device configuration needed** - cameras are auto-discovered!

## Image Naming

Local files: `camera_1_20251224_143052.jpg`, `camera_2_20251224_143052.jpg`

Server upload: `{capture_id}_camera_1.jpg`, `{capture_id}_camera_2.jpg`

S3 storage: `{device_id}/{date}/{capture_id}/camera_1_20251224_143052.jpg`

## Architecture

```
Jetson Boot
    │
    ▼
systemd starts collector.service
    │
    ▼
collector.py
    ├── Auto-discover cameras (any USB port)
    ├── Warmup cameras (30 frames)
    │
    ▼
Main Loop (every N seconds)
    ├── Burst capture (5 frames per camera)
    ├── Select sharpest (Laplacian variance)
    ├── Save locally with metadata
    │
    ▼
Background Sync Thread
    ├── Batch uploads (every 20 captures or 60s)
    ├── Exponential backoff on failure
    └── Mark synced, remove from queue
```

## Files

```
jetson-collector-ultra-minimal/
├── collector.py      # Main entry point
├── camera.py         # Auto-discovery + burst capture
├── storage.py        # Local queue + batch sync
├── requirements.txt  # Python deps (numpy, requests)
├── setup.sh          # One-time setup
├── run.sh            # Manual run script
├── collector.service # systemd unit
├── .env.example      # Config template
├── Dockerfile        # (optional) Docker build
└── docker-compose.yml# (optional) Docker compose
```

## Comparison with jetson-collector-minimal

| Feature | Minimal | Ultra-Minimal |
|---------|---------|---------------|
| Memory | ~300MB | ~80MB |
| OCR/Weight | Yes | No |
| State Machine | Yes | No |
| GPS | Yes | No |
| Wizard | Yes | No |
| Config | YAML | Env vars |
| Camera Config | Manual device | Auto-discover |
| Image Quality | Single frame | Burst + sharpest |

## Troubleshooting

### No cameras found

```bash
# Check connected cameras
ls /dev/video*
v4l2-ctl --list-devices

# Common issue: cameras expose multiple /dev/video* devices
# The collector filters for real capture devices automatically
```

### Images are blurry

- Increase `BURST_FRAMES` (default: 5)
- Decrease `CAPTURE_INTERVAL_SECONDS` to let autofocus settle
- Check camera lens for smudges

### Sync failing

```bash
# Check logs
journalctl -u collector -f

# Test server connectivity
curl -H "X-API-Key: YOUR_KEY" https://your-server.com/health
```

### Service won't start

```bash
# Check service status
sudo systemctl status collector.service

# Check for missing .env
ls -la /opt/collector-ultra-minimal/.env

# Check venv exists
ls -la /opt/collector-ultra-minimal/venv/
```

## Hardware Requirements

- NVIDIA Jetson Orin Nano (or similar)
- JetPack 6.x (L4T r36.x)
- 2+ USB cameras
- Network connectivity (Ethernet or WiFi)
- 16GB+ storage for captures

## License

Part of the ContainerVision / GoldBond project.
