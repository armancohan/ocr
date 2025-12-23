#!/bin/bash
# Quick Docker-based PDF to Markdown conversion
# No installation required - just Docker with NVIDIA Container Toolkit

set -e
shopt -s nullglob globstar 2>/dev/null || true

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

# Count PDFs (recursive)
PDF_COUNT=$(find "$INPUT_ABS" -type f -iname "*.pdf" 2>/dev/null | wc -l)

echo "============================================"
echo "  olmOCR 2 Docker PDF-to-Markdown"
echo "============================================"
echo "Input:  $INPUT_ABS ($PDF_COUNT PDF files)"
echo "Output: $OUTPUT_ABS"
echo ""

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "Error: No PDF files found in $INPUT_ABS"
    echo "Contents:"
    ls -la "$INPUT_ABS" | head -10
    exit 1
fi

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

# Run conversion - using exact syntax from olmOCR documentation
echo ""
echo "Starting conversion..."
echo "Note: Model loading may take 1-2 minutes on first run"
echo ""

# Build list of PDF files, skipping already converted ones
PDF_LISTFILE=$(mktemp)
trap "rm -f $PDF_LISTFILE" EXIT

TOTAL_PDFS=0
SKIPPED=0

find "$INPUT_ABS" -type f -iname "*.pdf" | while read -r pdf; do
    relpath="${pdf#$INPUT_ABS/}"
    basename="${relpath%.pdf}"
    basename="${basename%.PDF}"

    # Check if markdown already exists (try multiple possible output names)
    md_file="$OUTPUT_ABS/${basename}.md"
    md_file_flat="$OUTPUT_ABS/$(basename "$basename").md"

    if [ -f "$md_file" ] || [ -f "$md_file_flat" ]; then
        echo "SKIP" >> "${PDF_LISTFILE}.skip"
    else
        echo "/input/$relpath" >> "$PDF_LISTFILE"
    fi
done

PDF_COUNT_FOUND=$(wc -l < "$PDF_LISTFILE" 2>/dev/null || echo 0)
SKIPPED_COUNT=$(wc -l < "${PDF_LISTFILE}.skip" 2>/dev/null || echo 0)
rm -f "${PDF_LISTFILE}.skip"

echo "PDF files found: $((PDF_COUNT_FOUND + SKIPPED_COUNT))"
echo "Already converted (skipped): $SKIPPED_COUNT"
echo "To process: $PDF_COUNT_FOUND"
echo ""

if [ "$PDF_COUNT_FOUND" -eq 0 ]; then
    echo "All files already converted! Nothing to do."
    exit 0
fi

echo "Files to process:"
head -5 "$PDF_LISTFILE"
echo ""

echo "Running olmOCR pipeline..."
echo "(Model loading takes 1-2 minutes, then you'll see progress)"
echo ""

# Create workspace directory (olmOCR writes markdown to workspace/markdown/)
WORKSPACE="$OUTPUT_ABS/.olmocr_workspace"
mkdir -p "$WORKSPACE"

docker run --rm -t --gpus all \
    --shm-size=16g \
    -e PYTHONUNBUFFERED=1 \
    -v "$INPUT_ABS:/input:ro" \
    -v "$WORKSPACE:/workspace" \
    -v "$OUTPUT_ABS:/output" \
    -v "$PDF_LISTFILE:/pdf_list.txt:ro" \
    alleninstituteforai/olmocr:latest-with-model \
    -c "python -m olmocr.pipeline /workspace --markdown --pdfs /pdf_list.txt && \
        cp /workspace/markdown/*.md /output/ 2>/dev/null || true"

# Also try to copy if docker command didn't do it
if [ -d "$WORKSPACE/markdown" ]; then
    cp "$WORKSPACE/markdown"/*.md "$OUTPUT_ABS/" 2>/dev/null || true
fi

# Count results
MD_COUNT=$(find "$OUTPUT_ABS" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo "  Conversion Complete!"
echo "============================================"
echo "Markdown files created: $MD_COUNT"
echo "Output directory: $OUTPUT_ABS"
ls -la "$OUTPUT_ABS"/*.md 2>/dev/null | head -10 || echo "No markdown files generated"

# Cleanup workspace (optional - comment out to keep for debugging)
# rm -rf "$WORKSPACE"
