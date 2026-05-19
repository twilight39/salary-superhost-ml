"""Shared utilities used by both regression and classification pipelines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

RANDOM_STATE = 42


def get_project_root() -> Path:
    """Return the repository root directory."""
    # src/ml_portfolio/shared/utils.py -> 3 levels up -> repo root
    return Path(__file__).resolve().parents[3]


def get_default_paths(track: str) -> dict[str, Path]:
    """Return canonical data / model / figure paths for a given track."""
    root = get_project_root()
    return {
        "raw_dir": root / "data" / "raw",
        "cleaned_dir": root / "data" / "cleaned" / track,
        "models_dir": root / "models" / track,
        "figures_dir": root / "figures" / track if track == "classification" else root / "figures",
    }


class Timer:
    """Simple context manager for timing pipeline stages."""

    def __init__(self, label: str = "Stage") -> None:
        self.label = label
        self.start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start  # type: ignore[operator]
        print(f"[{self.label}] finished in {self.elapsed:.2f}s")


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
