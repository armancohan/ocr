#!/bin/bash
# olmOCR 2 Setup Script for Cloud GPU Instances
# Tested on: Ubuntu 22.04/24.04 with NVIDIA GPUs (RTX 4090, L40S, A100, H100)

set -e

echo "============================================"
echo "  olmOCR 2 PDF-to-Markdown Setup Script"
echo "============================================"

# Check if running as root for apt commands
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    echo "You may need to install CUDA drivers first."
fi

echo ""
echo "[1/5] Installing system dependencies..."

# Fix for outdated/broken repository sources (common on older cloud images)
# Remove problematic backports if they exist
if [ -f /etc/apt/sources.list ]; then
    $SUDO sed -i '/bullseye-backports/d' /etc/apt/sources.list 2>/dev/null || true
fi
if [ -d /etc/apt/sources.list.d ]; then
    $SUDO find /etc/apt/sources.list.d -name "*.list" -exec sed -i '/bullseye-backports/d' {} \; 2>/dev/null || true
fi

# Update package lists (continue even if some repos fail)
$SUDO apt-get update --allow-releaseinfo-change 2>/dev/null || $SUDO apt-get update || true

# Install core dependencies
$SUDO apt-get install -y \
    poppler-utils \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools \
    git \
    || $SUDO apt-get install -y \
    poppler-utils \
    git

# Try to install MS fonts (optional, may require EULA acceptance)
SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y ttf-mscorefonts-installer 2>/dev/null || \
    echo "Note: MS fonts skipped (optional)"

echo ""
echo "[2/5] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "[4/5] Installing olmOCR with GPU support..."
pip install 'olmocr[gpu]' --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[5/5] Installing optional dependencies for faster inference..."
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl || echo "FlashInfer installation skipped (optional optimization)"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To convert PDFs to Markdown:"
echo "  python pdf_to_md.py --input /path/to/pdfs --output /path/to/output"
echo ""
echo "Example:"
echo "  python pdf_to_md.py --input ./pdfs --output ./markdown_output"
echo ""
