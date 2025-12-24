#!/bin/bash
# One-time setup for native deployment on Jetson
# Run this script once after cloning the repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Ultra-Minimal Collector Setup"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running as root for system packages
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Note: Some steps require sudo. You may be prompted for password.${NC}"
fi

# Install system dependencies
echo ""
echo -e "${GREEN}[1/5] Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-opencv v4l-utils

# Create virtual environment
echo ""
echo -e "${GREEN}[2/5] Creating Python virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "Created venv/"
fi

# Activate and install dependencies
echo ""
echo -e "${GREEN}[3/5] Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory
echo ""
echo -e "${GREEN}[4/5] Creating data directory...${NC}"
if sudo mkdir -p /data/captures; then
    sudo chown "$USER:$USER" /data/captures 2>/dev/null || true
    echo "Created /data/captures"
else
    # Fallback to local directory
    mkdir -p ./data/captures
    echo "Created ./data/captures (fallback)"
fi

# Copy .env template
echo ""
echo -e "${GREEN}[5/5] Setting up configuration...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env from template${NC}"
    echo -e "${RED}IMPORTANT: Edit .env with your SERVER_URL and API_KEY${NC}"
else
    echo ".env already exists, skipping..."
fi

echo ""
echo "========================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your SERVER_URL and API_KEY:"
echo "     nano .env"
echo ""
echo "  2. Test the collector:"
echo "     ./run.sh"
echo ""
echo "  3. Install as systemd service for auto-boot:"
echo "     sudo cp -r . /opt/collector-ultra-minimal"
echo "     sudo cp collector.service /etc/systemd/system/"
echo "     sudo systemctl daemon-reload"
echo "     sudo systemctl enable collector.service"
echo "     sudo systemctl start collector.service"
echo ""
echo "  4. Check logs:"
echo "     journalctl -u collector -f"
echo ""
