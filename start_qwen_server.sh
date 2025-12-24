#!/bin/bash
# Start vLLM server with Qwen3-VL-30B-A3B-Instruct-FP8
# This server runs separately and handles OCR requests

set -e

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

echo "============================================"
echo "  Qwen3-VL vLLM Server"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo ""

# Check for --docker flag
if [ "$1" = "--docker" ] || [ "$1" = "-d" ]; then
    echo "Starting with Docker..."
    echo ""

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q '^qwen-vllm$'; then
        echo "Container 'qwen-vllm' already exists."
        echo "To restart: docker stop qwen-vllm && docker rm qwen-vllm"
        echo "Then run this script again."
        exit 1
    fi

    docker run -d \
        --name qwen-vllm \
        --gpus all \
        --ipc=host \
        -p ${PORT}:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        vllm/vllm-openai:latest \
        "$MODEL" \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --trust-remote-code \
        --limit-mm-per-prompt '{"video": 0}'

    echo ""
    echo "Server starting in background..."
    echo "Container name: qwen-vllm"
    echo ""
    echo "Check logs:  docker logs -f qwen-vllm"
    echo "Stop server: docker stop qwen-vllm && docker rm qwen-vllm"
    echo ""
    echo "Wait for 'Uvicorn running on http://0.0.0.0:8000' in logs"
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:${PORT}/v1/models"

else
    echo "Starting locally with vllm serve..."
    echo "(Use --docker or -d flag for Docker deployment)"
    echo ""

    # Check vllm is installed
    if ! command -v vllm &> /dev/null; then
        echo "Error: vllm not found. Install with:"
        echo "  pip install vllm>=0.11.0"
        exit 1
    fi

    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --trust-remote-code \
        --limit-mm-per-prompt '{"video": 0}'
fi
