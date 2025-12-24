#!/usr/bin/env python3
"""
PDF to Markdown using Qwen3-VL via vLLM Server

Converts PDFs to markdown using the Qwen3-VL vision-language model
running on a vLLM server.

Requirements:
    pip install openai pdf2image pillow tqdm

System dependencies:
    apt-get install poppler-utils

Usage:
    # Start vLLM server first (in another terminal)
    ./start_qwen_server.sh --docker

    # Then run OCR
    python qwen_ocr.py --input ./pdfs --output ./markdown
    python qwen_ocr.py --input document.pdf --output ./output
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFPageCountError
except ImportError:
    print("Error: pdf2image package required. Install with: pip install pdf2image")
    print("Also install poppler: apt-get install poppler-utils")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: pillow package required. Install with: pip install pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    server_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    max_tokens: int = 4096
    temperature: float = 0.0
    dpi: int = 200  # DPI for PDF rendering
    image_format: str = "PNG"
    max_image_size: int = 2048  # Max dimension for images
    workers: int = 4  # Concurrent pages to process
    timeout: int = 300  # Request timeout in seconds


# Default OCR prompt for document extraction
DEFAULT_PROMPT = """Extract all text from this document image and convert it to well-formatted Markdown.

Instructions:
- Preserve the document structure (headings, paragraphs, lists, tables)
- Use appropriate Markdown formatting (# for headings, - for lists, | for tables)
- Maintain the reading order
- For tables, use Markdown table syntax
- For equations, use LaTeX syntax wrapped in $ or $$
- Preserve any emphasized text (bold, italic)
- Do not add any commentary, just output the extracted content
- If there are images or figures, describe them briefly in [brackets]

Output the Markdown directly:"""


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def resize_image_if_needed(image: Image.Image, max_size: int) -> Image.Image:
    """Resize image if larger than max_size while maintaining aspect ratio."""
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF pages to PIL Images."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
    except PDFPageCountError as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {e}")
        return []


def ocr_image(
    client: OpenAI,
    image: Image.Image,
    config: OCRConfig,
    prompt: str = DEFAULT_PROMPT,
    page_num: Optional[int] = None
) -> str:
    """Send image to vLLM server for OCR."""
    # Resize if needed
    image = resize_image_if_needed(image, config.max_image_size)

    # Convert to base64
    base64_image = encode_image_to_base64(image, config.image_format)

    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{config.image_format.lower()};base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        page_info = f" (page {page_num})" if page_num else ""
        logger.error(f"OCR failed{page_info}: {e}")
        return f"[OCR Error{page_info}: {str(e)}]"


def process_pdf(
    pdf_path: Path,
    output_path: Path,
    client: OpenAI,
    config: OCRConfig,
    prompt: str = DEFAULT_PROMPT
) -> bool:
    """Process a single PDF file and save as markdown."""
    logger.info(f"Processing: {pdf_path.name}")

    # Convert PDF to images
    images = pdf_to_images(pdf_path, config.dpi)
    if not images:
        logger.error(f"No pages extracted from {pdf_path}")
        return False

    logger.info(f"  Pages: {len(images)}")

    # Process each page
    markdown_parts = []

    if config.workers > 1 and len(images) > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(
                    ocr_image, client, img, config, prompt, i + 1
                ): i for i, img in enumerate(images)
            }

            results = [None] * len(images)
            for future in tqdm(as_completed(futures), total=len(images), desc="  OCR"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"  Page {idx + 1} failed: {e}")
                    results[idx] = f"[OCR Error on page {idx + 1}]"

            markdown_parts = results
    else:
        # Sequential processing
        for i, image in enumerate(tqdm(images, desc="  OCR")):
            result = ocr_image(client, image, config, prompt, i + 1)
            markdown_parts.append(result)

    # Combine pages with page separators
    if len(markdown_parts) > 1:
        markdown_content = "\n\n---\n\n".join(
            f"<!-- Page {i + 1} -->\n\n{content}"
            for i, content in enumerate(markdown_parts)
        )
    else:
        markdown_content = markdown_parts[0] if markdown_parts else ""

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_content, encoding='utf-8')
    logger.info(f"  Saved: {output_path}")

    return True


def get_pdf_files(input_path: Path) -> list[Path]:
    """Get all PDF files from input path."""
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            return [input_path]
        else:
            raise ValueError(f"Not a PDF file: {input_path}")

    if input_path.is_dir():
        return sorted(input_path.glob("**/*.pdf"))

    raise ValueError(f"Path does not exist: {input_path}")


def check_server(client: OpenAI, model: str) -> bool:
    """Check if vLLM server is running and model is loaded."""
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if model in available or any(model in m for m in available):
            return True
        logger.warning(f"Model {model} not found. Available: {available}")
        return len(available) > 0
    except Exception as e:
        logger.error(f"Cannot connect to server: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown using Qwen3-VL via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server first
  ./start_qwen_server.sh --docker

  # Convert PDFs
  python qwen_ocr.py --input ./pdfs --output ./markdown
  python qwen_ocr.py --input doc.pdf --output ./output --dpi 300

  # Use custom server
  python qwen_ocr.py --input ./pdfs --output ./out --server http://gpu-server:8000/v1
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input PDF file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for markdown files"
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        help="Model name (default: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF rendering (default: 200)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per page (default: 4096)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Concurrent pages to process (default: 4)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have markdown output"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom OCR prompt (default: built-in document extraction prompt)"
    )

    args = parser.parse_args()

    # Create config
    config = OCRConfig(
        server_url=args.server,
        model=args.model,
        max_tokens=args.max_tokens,
        dpi=args.dpi,
        workers=args.workers
    )

    # Initialize OpenAI client (for vLLM)
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't need a real key
        base_url=config.server_url,
        timeout=config.timeout
    )

    # Check server connection
    logger.info(f"Connecting to vLLM server at {config.server_url}...")
    if not check_server(client, config.model):
        logger.error("Server not ready. Start with: ./start_qwen_server.sh --docker")
        sys.exit(1)
    logger.info("Server connected!")

    # Get PDF files
    try:
        pdf_files = get_pdf_files(args.input)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not pdf_files:
        logger.error(f"No PDF files found in {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF file(s)")

    # Filter already processed
    if args.skip_existing:
        original_count = len(pdf_files)
        pdf_files = [
            p for p in pdf_files
            if not (args.output / f"{p.stem}.md").exists()
        ]
        skipped = original_count - len(pdf_files)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already converted file(s)")

    if not pdf_files:
        logger.info("All files already processed!")
        sys.exit(0)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process each PDF
    prompt = args.prompt or DEFAULT_PROMPT
    success_count = 0
    fail_count = 0

    start_time = time.time()

    for pdf_path in pdf_files:
        output_path = args.output / f"{pdf_path.stem}.md"
        try:
            if process_pdf(pdf_path, output_path, client, config, prompt):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            fail_count += 1

    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Completed in {elapsed:.1f} seconds")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
