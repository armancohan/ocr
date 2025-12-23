# PDF to Markdown Converter

A reliable and efficient tool for batch converting PDFs to Markdown format using [olmOCR 2](https://github.com/allenai/olmocr) by Allen AI.

## Features

- **GPU Accelerated**: Uses FP8 quantization for ~3,400 tokens/sec throughput
- **Batch Processing**: Handle thousands of PDFs with progress tracking and resume capability
- **Multiple Deployment Options**: Local GPU, Docker, or cloud API servers
- **Multi-GPU Support**: Scale across multiple GPUs with tensor parallelism
- **Robust**: Automatic retries, error handling, and detailed logging
- **Cost Effective**: Process ~10,000 pages for under $2 (local GPU)

## Requirements

### Hardware
- NVIDIA GPU with 12GB+ VRAM
- Recommended: RTX 4090, L40S, A100, H100
- 30GB free disk space

### Software
- Ubuntu 22.04/24.04 (or Docker)
- Python 3.11+
- CUDA 12.8+

## Quick Start

### Option 1: Docker (Easiest - No Installation Required)

```bash
# Pull and run with Docker
./run_docker.sh ./data ./output
```

Requirements: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Option 2: Cloud GPU Instance (One Command)

```bash
# Automatically installs everything and runs conversion
./cloud_quickstart.sh ./data ./output
```

Works on AWS, GCP, Azure, Lambda Labs, RunPod, Vast.ai, etc.

### Option 3: Full Installation

```bash
# Install dependencies and create virtual environment
./setup.sh

# Activate environment
source venv/bin/activate

# Convert PDFs
python pdf_to_md.py --input ./data --output ./output
```

## Usage

### Basic Conversion

```bash
# Convert all PDFs in a directory
python pdf_to_md.py --input ./data --output ./output

# Convert a single PDF
python pdf_to_md.py --input document.pdf --output ./output
```

### Batch Processing with Progress Tracking

For large numbers of PDFs, use the batch converter which includes:
- Progress bar and ETA
- Resume capability (skips already converted files)
- Detailed logging

```bash
python batch_convert.py --input ./data --output ./output

# Customize batch size
python batch_convert.py --input ./data --output ./output --batch-size 100

# Start fresh (don't resume)
python batch_convert.py --input ./data --output ./output --no-resume
```

### Multi-GPU Setup

```bash
# Use 2 GPUs with tensor parallelism
python pdf_to_md.py --input ./data --output ./output --tensor-parallel-size 2
```

### Using Cloud API (No GPU Required)

If you don't have a GPU, use hosted API providers:

```bash
# DeepInfra ($0.09/$0.19 per million tokens)
python pdf_to_md.py --input ./data --output ./output \
  --server https://api.deepinfra.com/v1/openai

# Parasail ($0.10/$0.20 per million tokens)
python pdf_to_md.py --input ./data --output ./output \
  --server https://api.parasail.io/v1
```

### Docker Usage

```bash
# Basic usage
./run_docker.sh ./data ./output

# Control batch size (smaller = more frequent output)
PAGES_PER_GROUP=30 ./run_docker.sh ./data ./output
```

### External vLLM Server (Recommended for Large Jobs)

Running vLLM as a separate server avoids the 2+ minute model loading time on each run. This is ideal for:
- Processing multiple batches of PDFs
- Debugging (separate server and pipeline logs)
- Restarting the pipeline without reloading the model

**Step 1: Start vLLM server (once)**

```bash
docker run -d --name olmocr-vllm --gpus all \
    --shm-size=16g \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model allenai/olmOCR-2-7B-1025-FP8 \
    --max-model-len 16384 \
    --served-model-name allenai/olmOCR-2-7B-1025-FP8

# Wait for server to be ready (check logs)
docker logs -f olmocr-vllm
```

**Step 2: Run OCR pipeline (instant startup)**

```bash
VLLM_SERVER=http://host.docker.internal:8000/v1 ./run_docker.sh ./data ./output
```

**Verify server is running:**

```bash
curl http://localhost:8000/v1/models
```

**Stop server when done:**

```bash
docker stop olmocr-vllm && docker rm olmocr-vllm
```

## CLI Options

### pdf_to_md.py

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | Required | Input PDF file or directory |
| `--output`, `-o` | Required | Output directory for Markdown files |
| `--model`, `-m` | `allenai/olmOCR-2-7B-1025-FP8` | Model to use |
| `--workers`, `-w` | 8 | Number of concurrent workers |
| `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory for KV-cache |
| `--max-retries` | 3 | Maximum retries per page |
| `--pages-per-group` | 10 | Pages per work batch |
| `--server` | None | External vLLM server URL |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--docker` | False | Use Docker instead of local installation |

### batch_convert.py

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | Required | Input directory containing PDFs |
| `--output`, `-o` | Required | Output directory for Markdown files |
| `--batch-size`, `-b` | 50 | Number of PDFs per batch |
| `--resume` | True | Resume from last progress |
| `--no-resume` | False | Start fresh, don't resume |

### run_docker.sh Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAGES_PER_GROUP` | 50 | Pages per work item (smaller = more frequent output) |
| `VLLM_SERVER` | None | External vLLM server URL (e.g., `http://host.docker.internal:8000/v1`) |

## Configuration

Copy `config.example.json` to `config.json` and customize:

```json
{
    "model": "allenai/olmOCR-2-7B-1025-FP8",
    "workers": 8,
    "gpu_memory_utilization": 0.9,
    "max_page_retries": 3,
    "pages_per_group": 10,
    "tensor_parallel_size": 1
}
```

### Troubleshooting Memory Issues

If you encounter OOM (Out of Memory) errors:

1. Reduce GPU memory utilization:
   ```bash
   python pdf_to_md.py --input ./data --output ./output --gpu-memory-utilization 0.7
   ```

2. Reduce workers:
   ```bash
   python pdf_to_md.py --input ./data --output ./output --workers 4
   ```

3. Use smaller batches:
   ```bash
   python batch_convert.py --input ./data --output ./output --batch-size 20
   ```

## Output Structure

```
./output/
├── document1.md      # Converted markdown files
├── document2.md
├── ...
└── .progress.json    # Progress tracking (batch_convert.py)
```

## Performance

Based on olmOCR 2 benchmarks:

| Metric | Value |
|--------|-------|
| Accuracy (olmOCR-Bench) | 82.4% |
| Throughput | ~3,400 tokens/sec (FP8) |
| Cost | <$2 per 10,000 pages |
| Speed | 170x faster than competitors |

## Supported Content

olmOCR 2 handles:
- Standard text documents
- Tables and structured data
- Mathematical equations
- Handwritten content
- Multi-column layouts
- Mixed text and images

## Cloud Deployment

### AWS EC2

```bash
# Launch g5.xlarge or p4d.24xlarge instance with Deep Learning AMI
git clone <your-repo-url>
cd pdf-to-md
./cloud_quickstart.sh ./data ./output
```

### Google Cloud

```bash
# Launch N1 with T4/V100/A100 GPU
git clone <your-repo-url>
cd pdf-to-md
./cloud_quickstart.sh ./data ./output
```

### Lambda Labs / RunPod / Vast.ai

```bash
# These typically come with CUDA pre-installed
git clone <your-repo-url>
cd pdf-to-md
./cloud_quickstart.sh ./data ./output
```

## License

This tool is provided as-is. olmOCR is licensed under [Apache 2.0](https://github.com/allenai/olmocr/blob/main/LICENSE) by Allen AI.

## Acknowledgments

- [olmOCR](https://github.com/allenai/olmocr) by Allen AI
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference
