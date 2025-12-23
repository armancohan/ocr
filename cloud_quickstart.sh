#!/bin/bash
# One-command setup and run for cloud GPU instances
# Supports: AWS, GCP, Azure, Lambda Labs, RunPod, Vast.ai, etc.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/YOUR_REPO/main/cloud_quickstart.sh | bash -s -- /path/to/pdfs /path/to/output
#
# Or after cloning:
#   ./cloud_quickstart.sh /path/to/pdfs /path/to/output

set -e

INPUT_DIR="${1:-./pdfs}"
OUTPUT_DIR="${2:-./markdown_output}"

echo "============================================"
echo "  olmOCR 2 Cloud Quick Start"
echo "============================================"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS=$(uname -s)
fi

echo "Detected OS: $OS"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No NVIDIA GPU detected"
    echo "This script requires an NVIDIA GPU"
    exit 1
fi

echo ""

# Install system dependencies
echo "[1/4] Installing system dependencies..."
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        poppler-utils \
        fonts-crosextra-caladea \
        fonts-crosextra-carlito \
        gsfonts \
        python3.11 \
        python3.11-venv \
        python3-pip \
        2>/dev/null || true
    # Try msttcorefonts separately (may require EULA acceptance)
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ttf-mscorefonts-installer 2>/dev/null || true
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "fedora" ]; then
    sudo yum install -y poppler-utils python3.11 2>/dev/null || true
else
    echo "Unsupported OS: $OS"
    echo "Please install dependencies manually or use Docker"
    exit 1
fi

# Setup Python environment
echo "[2/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q

# Install olmOCR
echo "[3/4] Installing olmOCR (this may take a few minutes)..."
pip install 'olmocr[gpu]' --extra-index-url https://download.pytorch.org/whl/cu128 -q

# Optional: FlashInfer for faster inference
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl -q 2>/dev/null || true

# Run conversion
echo "[4/4] Converting PDFs..."
mkdir -p "$OUTPUT_DIR"

if [ -d "$INPUT_DIR" ]; then
    PDF_COUNT=$(find "$INPUT_DIR" -name "*.pdf" -o -name "*.PDF" 2>/dev/null | wc -l)
    echo "Found $PDF_COUNT PDF file(s) in $INPUT_DIR"

    if [ "$PDF_COUNT" -gt 0 ]; then
        python -m olmocr.pipeline "$OUTPUT_DIR/.workspace" \
            --markdown \
            --pdfs "$INPUT_DIR"/*.pdf \
            --gpu-memory-utilization 0.9

        # Move results
        mv "$OUTPUT_DIR/.workspace/markdown"/* "$OUTPUT_DIR/" 2>/dev/null || true
        rm -rf "$OUTPUT_DIR/.workspace"

        echo ""
        echo "============================================"
        echo "  Conversion Complete!"
        echo "============================================"
        echo "Output: $OUTPUT_DIR"
        ls -la "$OUTPUT_DIR"/*.md 2>/dev/null | head -20
    else
        echo "No PDF files found in $INPUT_DIR"
    fi
else
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi
