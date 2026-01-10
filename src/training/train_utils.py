"""
Training Utility Functions

Contains helper functions for training setup:
    - setup_seed: Random seed initialization
    - verify_environment: Environment verification
    - setup_file_logging: File logging setup
    - find_latest_checkpoint: Auto-detect latest checkpoint

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint file.

    Sorts by modification time and returns the most recent checkpoint_*.pt file.

    Args:
        checkpoint_dir: Checkpoint directory

    Returns:
        Path to latest checkpoint, or None if not found
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def setup_seed(seed: int) -> None:
    """
    Setup random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def verify_environment() -> None:
    """Verify environment requirements."""
    import sys

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if py_version != "3.12":
        logger.warning(f"Python {py_version} detected, recommended: 3.12")

    if not torch.__version__.startswith("2."):
        logger.warning(f"PyTorch {torch.__version__} detected, recommended: 2.8.x")

    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        logger.info(f"CUDA {cuda_version} available")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, training will be slow")

    try:
        from flash_attn import flash_attn_func  # noqa: F401
        logger.info("Flash Attention available")
    except ImportError:
        logger.warning("Flash Attention not available")


def setup_file_logging(log_dir: Path, experiment_name: str = "train") -> None:
    """
    Setup file logging.

    Args:
        log_dir: Log directory
        experiment_name: Experiment name for log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_file}")


__all__ = [
    "find_latest_checkpoint",
    "setup_seed",
    "verify_environment",
    "setup_file_logging",
]
