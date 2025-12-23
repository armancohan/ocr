#!/usr/bin/env python3
"""
PDF to Markdown Converter using olmOCR 2

A reliable and efficient tool for batch converting PDFs to Markdown format
using Allen AI's olmOCR 2 model with GPU acceleration.

Usage:
    python pdf_to_md.py --input /path/to/pdfs --output /path/to/output
    python pdf_to_md.py --input ./pdfs --output ./markdown --workers 4
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_gpu_available() -> bool:
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            for i, gpu in enumerate(gpus):
                logger.info(f"GPU {i}: {gpu}")
            return True
    except FileNotFoundError:
        pass
    return False


def get_pdf_files(input_path: Path) -> list[Path]:
    """Get all PDF files from input path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            return [input_path]
        else:
            raise ValueError(f"Input file is not a PDF: {input_path}")

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob('**/*.pdf'))
        pdf_files.extend(sorted(input_path.glob('**/*.PDF')))
        return list(set(pdf_files))  # Remove duplicates

    raise ValueError(f"Input path does not exist: {input_path}")


def run_olmocr_pipeline(
    pdf_files: list[Path],
    output_dir: Path,
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    workers: int = 8,
    gpu_memory_utilization: float = 0.9,
    max_page_retries: int = 3,
    pages_per_group: int = 10,
    server: Optional[str] = None,
    tensor_parallel_size: int = 1,
) -> bool:
    """
    Run olmOCR pipeline for batch PDF processing.

    Args:
        pdf_files: List of PDF file paths
        output_dir: Output directory for markdown files
        model: Model identifier
        workers: Number of concurrent workers
        gpu_memory_utilization: Fraction of GPU memory for KV-cache
        max_page_retries: Maximum retries per page
        pages_per_group: Pages per work batch
        server: External vLLM server URL (optional)
        tensor_parallel_size: Number of GPUs for tensor parallelism

    Returns:
        True if successful, False otherwise
    """
    # Create a temporary workspace
    workspace = output_dir / ".olmocr_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, "-m", "olmocr.pipeline",
        str(workspace),
        "--markdown",
        "--model", model,
        "--workers", str(workers),
        "--max_page_retries", str(max_page_retries),
        "--pages_per_group", str(pages_per_group),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--pdfs",
    ]

    # Add PDF files
    cmd.extend([str(f) for f in pdf_files])

    # Add server if specified (for remote inference)
    if server:
        cmd.extend(["--server", server])

    # Add tensor parallel size for multi-GPU
    if tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    logger.info(f"Processing {len(pdf_files)} PDF(s)...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model}")

    try:
        # Run the pipeline
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )

        # Move markdown files to output directory
        markdown_dir = workspace / "markdown"
        if markdown_dir.exists():
            for md_file in markdown_dir.glob("*.md"):
                dest = output_dir / md_file.name
                shutil.move(str(md_file), str(dest))
                logger.info(f"Created: {dest}")

        # Cleanup workspace
        shutil.rmtree(workspace, ignore_errors=True)

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return False


def run_with_docker(
    pdf_files: list[Path],
    output_dir: Path,
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
) -> bool:
    """
    Run olmOCR using Docker (alternative if local installation fails).

    Args:
        pdf_files: List of PDF file paths
        output_dir: Output directory for markdown files
        model: Model identifier

    Returns:
        True if successful, False otherwise
    """
    # Create temporary directory for input files
    with tempfile.TemporaryDirectory() as temp_input:
        temp_input_path = Path(temp_input)

        # Copy PDF files to temp directory
        for pdf in pdf_files:
            shutil.copy(pdf, temp_input_path / pdf.name)

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker", "run", "--gpus", "all",
            "-v", f"{temp_input_path}:/input:ro",
            "-v", f"{output_dir}:/output",
            "alleninstituteforai/olmocr:latest-with-model",
            "-c",
            f"python -m olmocr.pipeline /output/.workspace --markdown --model {model} --pdfs /input/*.pdf && mv /output/.workspace/markdown/* /output/ && rm -rf /output/.workspace"
        ]

        logger.info("Running with Docker...")

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker execution failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown using olmOCR 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all PDFs in a directory
  python pdf_to_md.py --input ./pdfs --output ./markdown

  # Convert a single PDF
  python pdf_to_md.py --input document.pdf --output ./output

  # Use multiple GPUs
  python pdf_to_md.py --input ./pdfs --output ./output --tensor-parallel-size 2

  # Use external API server (DeepInfra, Parasail, etc.)
  python pdf_to_md.py --input ./pdfs --output ./output --server https://api.deepinfra.com/v1/openai

  # Use Docker instead of local installation
  python pdf_to_md.py --input ./pdfs --output ./output --docker
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for Markdown files"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="allenai/olmOCR-2-7B-1025-FP8",
        help="Model to use (default: allenai/olmOCR-2-7B-1025-FP8)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for KV-cache (default: 0.9)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per page (default: 3)"
    )
    parser.add_argument(
        "--pages-per-group",
        type=int,
        default=10,
        help="Pages per work batch (default: 10)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="External vLLM server URL (e.g., https://api.deepinfra.com/v1/openai)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker instead of local installation"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Get PDF files
    try:
        pdf_files = get_pdf_files(args.input)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not pdf_files:
        logger.error(f"No PDF files found in: {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF file(s)")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Check GPU availability (unless using external server)
    if not args.server and not args.docker:
        if not check_gpu_available():
            logger.warning("No NVIDIA GPU detected. Consider using --server or --docker")

    start_time = time.time()

    # Run conversion
    if args.docker:
        success = run_with_docker(pdf_files, args.output, args.model)
    else:
        success = run_olmocr_pipeline(
            pdf_files=pdf_files,
            output_dir=args.output,
            model=args.model,
            workers=args.workers,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_page_retries=args.max_retries,
            pages_per_group=args.pages_per_group,
            server=args.server,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    elapsed_time = time.time() - start_time

    if success:
        # Count output files
        output_files = list(args.output.glob("*.md"))
        logger.info(f"Conversion complete! {len(output_files)} markdown file(s) created")
        logger.info(f"Total time: {elapsed_time:.1f} seconds")
        logger.info(f"Output directory: {args.output}")
    else:
        logger.error("Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
