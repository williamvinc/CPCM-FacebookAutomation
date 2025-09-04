#!/usr/bin/env python3
"""
main.py

Usage:
  python main.py                 # default: run dump_sheet.py
  python main.py --task dump     # run dump only
  python main.py --task classify # run classification only
  python main.py --task both     # run dump then classification

Env:
  SHEET_ID, gid, GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_RPM, WRITE_BACK, etc.
"""

import sys
import argparse
from pathlib import Path
import importlib.util
import logging


def run_dump_sheet(script_path: Path) -> None:
    """Load and execute dump_sheet.py as-is."""
    spec = importlib.util.spec_from_file_location("dump_sheet", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def run_classification() -> None:
    """Import and run the classification pipeline."""
    try:
        import classification_posts as cp
    except ImportError as e:
        raise RuntimeError(f"Failed to import classification_posts: {e}")
    cp.run()


def main():
    parser = argparse.ArgumentParser(description="Runner for dump/classify tasks.")
    parser.add_argument(
        "--task",
        choices=["dump", "classify", "both"],
        default="dump",
        help="Task to run: 'dump', 'classify', or 'both' (dump then classify).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO", format="%(asctime)s | %(levelname)s | %(message)s"
    )

    script = Path(__file__).with_name("dump_sheet.py")

    if args.task in ("dump", "both"):
        if not script.exists():
            print(f"[ERROR] File not found: {script}")
            sys.exit(1)
        try:
            logging.info("Running dump_sheet.py ...")
            run_dump_sheet(script)
            logging.info("dump_sheet.py finished.")
        except Exception as e:
            print(f"[ERROR] Error while running {script.name}: {e}")
            sys.exit(1)

    if args.task in ("classify", "both"):
        try:
            logging.info("Running classification_posts pipeline ...")
            run_classification()
            logging.info("classification_posts finished.")
        except Exception as e:
            print(f"[ERROR] Error while running classification: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
