"""Configuration helpers for reproducible project scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from mv_audit.utils.io_utils import read_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    return read_yaml(path)


def set_random_seed(seed: int) -> None:
    """Set random seeds for common Python and numeric libraries when available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
