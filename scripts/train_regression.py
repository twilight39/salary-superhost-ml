#!/usr/bin/env python3
"""CLI script to train the full regression pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_portfolio.regression.pipeline import run_pipeline
from ml_portfolio.shared.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regression salary prediction pipeline")
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Path to raw salary-survey.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    results = run_pipeline(
        raw_path=args.raw,
        models_dir=args.models_dir,
        random_state=args.seed,
    )
    print("\nPipeline complete. Artifacts:")
    for name, path in results["models"].items():
        print(f"  {name:20s} -> {path}")


if __name__ == "__main__":
    main()
