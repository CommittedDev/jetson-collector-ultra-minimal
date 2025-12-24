#!/bin/bash
# Run the collector (for testing/manual execution)
# For auto-boot, use the systemd service instead

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Run ./setup.sh first, then edit .env with your configuration."
    exit 1
fi

# Check for venv
if [ ! -d venv ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Run ./setup.sh first."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Validate required vars
if [ -z "$SERVER_URL" ]; then
    echo "ERROR: SERVER_URL not set in .env"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY not set in .env"
    exit 1
fi

# Activate venv and run
source venv/bin/activate
exec python3 collector.py
