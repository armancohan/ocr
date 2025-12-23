#!/bin/bash
# Quick Docker-based PDF to Markdown conversion
# No installation required - just Docker with NVIDIA Container Toolkit

set -e

INPUT_DIR="${1:-./pdfs}"
OUTPUT_DIR="${2:-./markdown_output}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    echo ""
    echo "Usage: ./run_docker.sh [input_dir] [output_dir]"
    echo "  input_dir:  Directory containing PDF files (default: ./pdfs)"
    echo "  output_dir: Directory for output markdown files (default: ./markdown_output)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get absolute paths
INPUT_ABS=$(cd "$INPUT_DIR" && pwd)
OUTPUT_ABS=$(cd "$OUTPUT_DIR" && pwd)

# Count PDFs
PDF_COUNT=$(find "$INPUT_ABS" -maxdepth 1 -name "*.pdf" -o -name "*.PDF" 2>/dev/null | wc -l)

echo "============================================"
echo "  olmOCR 2 Docker PDF-to-Markdown"
echo "============================================"
echo "Input:  $INPUT_ABS ($PDF_COUNT PDF files)"
echo "Output: $OUTPUT_ABS"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Verify GPU access in Docker
echo "Checking GPU access..."
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "Error: Cannot access GPU from Docker"
    echo ""
    echo "Please ensure:"
    echo "  1. NVIDIA drivers are installed: nvidia-smi"
    echo "  2. NVIDIA Container Toolkit is installed:"
    echo "     sudo apt-get install -y nvidia-container-toolkit"
    echo "     sudo systemctl restart docker"
    echo ""
    exit 1
fi

echo "GPU access confirmed:"
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Pull latest image
echo "Pulling olmOCR Docker image (this may take a while on first run)..."
docker pull alleninstituteforai/olmocr:latest-with-model

# Run conversion with interactive terminal for live output
echo ""
echo "Starting conversion..."
echo "Note: Model loading may take 1-2 minutes on first run"
echo ""

docker run --rm -it --gpus all \
    --shm-size=16g \
    -v "$INPUT_ABS:/input:ro" \
    -v "$OUTPUT_ABS:/output" \
    alleninstituteforai/olmocr:latest-with-model \
    bash -c '
        set -e
        echo "Container started, checking GPU..."
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
        echo ""
        echo "Starting olmOCR pipeline..."
        python -m olmocr.pipeline /output/.workspace \
            --markdown \
            --pdfs /input/*.pdf \
            --gpu-memory-utilization 0.9

        echo ""
        echo "Moving output files..."
        if [ -d "/output/.workspace/markdown" ]; then
            mv /output/.workspace/markdown/* /output/ 2>/dev/null || true
        fi
        rm -rf /output/.workspace
        echo "Done!"
    '

echo ""
echo "============================================"
echo "  Conversion Complete!"
echo "============================================"
echo "Output files: $OUTPUT_ABS"
ls -la "$OUTPUT_ABS"/*.md 2>/dev/null || echo "No markdown files generated"
