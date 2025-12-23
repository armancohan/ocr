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

echo "============================================"
echo "  olmOCR 2 Docker PDF-to-Markdown"
echo "============================================"
echo "Input:  $INPUT_ABS"
echo "Output: $OUTPUT_ABS"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check for NVIDIA Docker
if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "Warning: NVIDIA Container Toolkit may not be installed"
    echo "Install it with: sudo apt-get install -y nvidia-container-toolkit"
fi

# Pull latest image
echo "Pulling olmOCR Docker image..."
docker pull alleninstituteforai/olmocr:latest-with-model

# Run conversion
echo ""
echo "Starting conversion..."
docker run --rm --gpus all \
    -v "$INPUT_ABS:/input:ro" \
    -v "$OUTPUT_ABS:/output" \
    alleninstituteforai/olmocr:latest-with-model \
    -c "python -m olmocr.pipeline /output/.workspace --markdown --pdfs /input/*.pdf && \
        mv /output/.workspace/markdown/* /output/ 2>/dev/null || true && \
        rm -rf /output/.workspace"

echo ""
echo "============================================"
echo "  Conversion Complete!"
echo "============================================"
echo "Output files: $OUTPUT_ABS"
ls -la "$OUTPUT_ABS"/*.md 2>/dev/null || echo "No markdown files generated"
