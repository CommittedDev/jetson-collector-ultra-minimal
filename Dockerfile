# Jetson Data Collector - Ultra Minimal
# For NVIDIA Jetson Orin Nano with JetPack 6.x
# OPTIONAL: For future Docker deployment if needed

FROM nvcr.io/nvidia/l4t-base:r36.2.0

LABEL maintainer="ContainerVision"
LABEL description="Ultra-minimal image collector for Jetson"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-opencv \
    v4l-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY collector.py camera.py storage.py ./

# Create data directory
RUN mkdir -p /data/captures

# Default environment (override in docker-compose or .env)
ENV CAPTURE_INTERVAL_SECONDS=10
ENV BATCH_SIZE=20
ENV BURST_FRAMES=5
ENV SHARPNESS_THRESHOLD=100.0
ENV STORAGE_PATH=/data/captures
ENV MAX_STORAGE_GB=10.0
ENV SYNC_INTERVAL_SECONDS=60
ENV MIN_CAMERAS=2

# Run collector
CMD ["python3", "collector.py"]
