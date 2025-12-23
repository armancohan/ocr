#!/usr/bin/env python3
"""
Batch PDF to Markdown Converter with Progress Tracking

For processing large numbers of PDFs with:
- Progress bar
- Resume capability (skips already converted files)
- Batch processing to manage memory
- Detailed logging

Usage:
    python batch_convert.py --input ./pdfs --output ./markdown
    python batch_convert.py --input ./pdfs --output ./markdown --batch-size 50 --resume
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_convert.log')
    ]
)
logger = logging.getLogger(__name__)


class BatchConverter:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        model: str = "allenai/olmOCR-2-7B-1025-FP8",
        batch_size: int = 50,
        workers: int = 8,
        gpu_memory_utilization: float = 0.9,
        resume: bool = True,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.batch_size = batch_size
        self.workers = workers
        self.gpu_memory_utilization = gpu_memory_utilization
        self.resume = resume

        # Track progress
        self.progress_file = output_dir / ".progress.json"
        self.workspace = output_dir / ".olmocr_workspace"

    def get_all_pdfs(self) -> list[Path]:
        """Get all PDF files from input directory."""
        pdfs = list(self.input_dir.glob("**/*.pdf"))
        pdfs.extend(self.input_dir.glob("**/*.PDF"))
        return sorted(set(pdfs))

    def get_completed_pdfs(self) -> set[str]:
        """Get set of already converted PDF names."""
        if not self.resume or not self.progress_file.exists():
            return set()

        try:
            with open(self.progress_file) as f:
                progress = json.load(f)
                return set(progress.get("completed", []))
        except (json.JSONDecodeError, IOError):
            return set()

    def save_progress(self, completed: list[str], failed: list[str]):
        """Save conversion progress."""
        progress = {
            "completed": completed,
            "failed": failed,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def convert_batch(self, pdf_files: list[Path]) -> tuple[list[str], list[str]]:
        """Convert a batch of PDFs."""
        self.workspace.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "olmocr.pipeline",
            str(self.workspace),
            "--markdown",
            "--model", self.model,
            "--workers", str(self.workers),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--pdfs",
        ]
        cmd.extend([str(f) for f in pdf_files])

        try:
            subprocess.run(cmd, check=True)

            # Move results
            markdown_dir = self.workspace / "markdown"
            converted = []
            if markdown_dir.exists():
                for md_file in markdown_dir.glob("*.md"):
                    dest = self.output_dir / md_file.name
                    shutil.move(str(md_file), str(dest))
                    # Track by PDF name (without .md extension)
                    pdf_name = md_file.stem + ".pdf"
                    converted.append(pdf_name)

            # Cleanup workspace
            shutil.rmtree(self.workspace, ignore_errors=True)

            return converted, []

        except subprocess.CalledProcessError as e:
            logger.error(f"Batch conversion failed: {e}")
            # Return all as failed
            return [], [f.name for f in pdf_files]

    def run(self):
        """Run the batch conversion."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all PDFs
        all_pdfs = self.get_all_pdfs()
        logger.info(f"Found {len(all_pdfs)} PDF file(s)")

        # Filter already completed
        completed = list(self.get_completed_pdfs())
        failed = []

        pending = [p for p in all_pdfs if p.name not in completed]

        if self.resume and len(completed) > 0:
            logger.info(f"Resuming: {len(completed)} already converted, {len(pending)} remaining")

        if not pending:
            logger.info("All files already converted!")
            return

        # Process in batches
        total_batches = (len(pending) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for i in range(0, len(pending), self.batch_size):
            batch = pending[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

            batch_completed, batch_failed = self.convert_batch(batch)
            completed.extend(batch_completed)
            failed.extend(batch_failed)

            # Save progress after each batch
            self.save_progress(completed, failed)

            # Progress update
            elapsed = time.time() - start_time
            done = len(completed)
            remaining = len(all_pdfs) - done
            if done > 0:
                rate = elapsed / done
                eta = rate * remaining
                logger.info(f"Progress: {done}/{len(all_pdfs)} ({done*100//len(all_pdfs)}%) - ETA: {eta/60:.1f} min")

        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"Conversion complete!")
        logger.info(f"  Total time: {elapsed/60:.1f} minutes")
        logger.info(f"  Converted: {len(completed)}")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Output: {self.output_dir}")

        if failed:
            logger.warning(f"Failed files: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert PDFs to Markdown with progress tracking"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input directory containing PDFs")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory for Markdown files")
    parser.add_argument("--model", "-m", type=str,
                        default="allenai/olmOCR-2-7B-1025-FP8",
                        help="Model to use")
    parser.add_argument("--batch-size", "-b", type=int, default=50,
                        help="Number of PDFs per batch (default: 50)")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Number of concurrent workers")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory fraction for KV-cache")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from last progress (default: True)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume")

    args = parser.parse_args()

    if not args.input.is_dir():
        logger.error(f"Input must be a directory: {args.input}")
        sys.exit(1)

    converter = BatchConverter(
        input_dir=args.input,
        output_dir=args.output,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        gpu_memory_utilization=args.gpu_memory_utilization,
        resume=not args.no_resume,
    )

    converter.run()


if __name__ == "__main__":
    main()
