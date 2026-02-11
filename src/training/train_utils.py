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
import re
import shutil

import torch

logger = logging.getLogger(__name__)
_CHECKPOINT_EPOCH_STEP_RE = re.compile(r"^checkpoint_epoch(\d+)_step(\d+)\.pt$")


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

    def _sort_key(path: Path) -> tuple:
        mtime = path.stat().st_mtime
        match = _CHECKPOINT_EPOCH_STEP_RE.match(path.name)
        if match is None:
            return (mtime, -1, 0, path.name)

        epoch = int(match.group(1))
        step = int(match.group(2))
        # Tie-breaker for same step: prefer smaller epoch to avoid
        # inflated epoch names from legacy misnaming.
        return (mtime, step, -epoch, path.name)

    latest = max(checkpoints, key=_sort_key)
    return latest


def epoch_for_checkpoint_step(step: int, steps_per_epoch: int) -> int:
    """Convert optimizer step to 1-based checkpoint epoch index."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if step <= 0:
        return 1
    return max(1, ((step - 1) // steps_per_epoch) + 1)


def repair_checkpoint_filenames(
    checkpoint_dir: Path,
    steps_per_epoch: int,
    dry_run: bool = False,
) -> int:
    """Repair periodic checkpoint filenames to canonical epoch-step form.

    Canonical rule:
        checkpoint_epoch{epoch_for_checkpoint_step(step)}_step{step}.pt

    Only files matching ``checkpoint_epoch{N}_step{S}.pt`` are considered.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        steps_per_epoch: Configured epoch step budget (len(dataloader)).
        dry_run: If True, only log proposed renames.

    Returns:
        Number of files renamed (or that would be renamed in dry-run).
    """
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if not checkpoint_dir.exists():
        return 0

    renamed = 0
    for ckpt in checkpoint_dir.glob("checkpoint_epoch*_step*.pt"):
        match = _CHECKPOINT_EPOCH_STEP_RE.match(ckpt.name)
        if match is None:
            continue

        epoch_in_name = int(match.group(1))
        step_in_name = int(match.group(2))
        expected_epoch = epoch_for_checkpoint_step(step_in_name, steps_per_epoch)
        if epoch_in_name == expected_epoch:
            continue

        target = ckpt.with_name(
            f"checkpoint_epoch{expected_epoch}_step{step_in_name}.pt"
        )
        if target.exists():
            logger.warning(
                "Skip checkpoint rename because target exists: %s -> %s",
                ckpt.name,
                target.name,
            )
            continue

        if dry_run:
            logger.info(
                "Checkpoint rename (dry-run): %s -> %s",
                ckpt.name,
                target.name,
            )
        else:
            try:
                ckpt.rename(target)
                logger.info("Checkpoint renamed: %s -> %s", ckpt.name, target.name)
            except OSError as e:
                # Some Windows handles disallow delete/rename even when read is allowed.
                shutil.copy2(ckpt, target)
                logger.warning(
                    "Checkpoint rename blocked (%s). Created canonical copy instead: %s -> %s",
                    e,
                    ckpt.name,
                    target.name,
                )
        renamed += 1

    return renamed


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
    "epoch_for_checkpoint_step",
    "repair_checkpoint_filenames",
    "setup_seed",
    "verify_environment",
    "setup_file_logging",
]
