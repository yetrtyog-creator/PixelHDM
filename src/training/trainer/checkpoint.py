"""
Checkpoint Manager

Contains CheckpointManager class for saving and loading checkpoints.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import logging
import os
import tempfile

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler
    from torch.utils.data import DataLoader
    from ...config.model_config import PixelHDMConfig, TrainingConfig
    from ..optimization import ZClip, EMA

from .metrics import TrainerState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and loading."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Optional["PixelHDMConfig"] = None,
        training_config: Optional["TrainingConfig"] = None,
        ema: Optional["EMA"] = None,
        zclip: Optional["ZClip"] = None,
        scaler: Optional["GradScaler"] = None,
        dataloader: Optional["DataLoader"] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.training_config = training_config
        self.ema = ema
        self.zclip = zclip
        self.scaler = scaler
        self.dataloader = dataloader
        self._lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def set_lr_scheduler(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler
    ) -> None:
        """Set the LR scheduler for saving/loading."""
        self._lr_scheduler = scheduler

    def save(
        self,
        path: str | Path,
        state: TrainerState,
        checkpoint_name: Optional[str] = None,
        cleanup: bool = True,
    ) -> None:
        """Save checkpoint.

        Args:
            path: Checkpoint directory
            state: Trainer state
            checkpoint_name: Checkpoint name (without .pt)
            cleanup: Whether to cleanup old checkpoints
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = self._build_checkpoint(state)
        checkpoint_path = self._get_checkpoint_path(path, checkpoint_name, state.step)

        # Atomic save: write to temp file first, then rename
        # This prevents corruption if save is interrupted
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".pt.tmp", dir=str(path)
        )
        try:
            os.close(temp_fd)
            torch.save(checkpoint, temp_path)
            # Atomic rename (on POSIX) or replace (on Windows)
            os.replace(temp_path, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        if cleanup:
            self._cleanup_old_checkpoints(path)

    def _build_checkpoint(self, state: TrainerState) -> Dict[str, Any]:
        """Build checkpoint dictionary."""
        checkpoint: Dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state": {
                "step": state.step,
                "epoch": state.epoch,
                "batch_idx": state.batch_idx,
                "best_loss": state.best_loss,
                "total_samples": state.total_samples,
            },
        }

        if self.config is not None:
            checkpoint["config"] = self.config.to_dict()
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()
        if self.zclip is not None:
            checkpoint["zclip"] = self.zclip.state_dict()
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()
        if self._lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self._lr_scheduler.state_dict()

        self._save_sampler_state(checkpoint)
        return checkpoint

    def _save_sampler_state(self, checkpoint: Dict[str, Any]) -> None:
        """Save sampler state if available."""
        if self.dataloader is None:
            return

        sampler = getattr(self.dataloader, 'batch_sampler', None)
        if sampler is None:
            sampler = getattr(self.dataloader, 'sampler', None)

        if sampler is not None and hasattr(sampler, 'state_dict'):
            checkpoint["sampler"] = sampler.state_dict()
            logger.debug("Sampler state saved to checkpoint")

    def _get_checkpoint_path(
        self, path: Path, name: Optional[str], step: int
    ) -> Path:
        """Get checkpoint file path."""
        if name is not None:
            return path / f"{name}.pt"
        return path / f"checkpoint_step_{step}.pt"

    def _cleanup_old_checkpoints(
        self,
        save_dir: Path,
        pattern: str = "checkpoint_*.pt",
    ) -> None:
        """Delete old checkpoints, keeping only max_keep."""
        max_keep = 1
        if self.training_config is not None:
            max_keep = getattr(self.training_config, 'max_checkpoints', 1)

        if max_keep <= 0:
            return

        if not save_dir.exists():
            return

        checkpoints: List[Path] = sorted(
            save_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for ckpt in checkpoints[max_keep:]:
            logger.info(f"Deleting old checkpoint (max={max_keep}): {ckpt.name}")
            try:
                ckpt.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete checkpoint {ckpt}: {e}")

    def load(
        self,
        path: str | Path,
        state: TrainerState,
        load_optimizer: bool = True,
        load_ema: bool = True,
        load_scheduler: bool = True,
    ) -> None:
        """Load checkpoint.

        Args:
            path: Checkpoint path
            state: Trainer state to update
            load_optimizer: Whether to load optimizer state
            load_ema: Whether to load EMA state
            load_scheduler: Whether to load scheduler state
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self._load_model(checkpoint)
        self._load_optimizer(checkpoint, load_optimizer)
        self._load_ema(checkpoint, load_ema)
        self._load_zclip(checkpoint)
        self._load_scaler(checkpoint)
        self._load_scheduler(checkpoint, load_scheduler)
        self._load_sampler(checkpoint)
        self._load_state(checkpoint, state)

        logger.info(f"Checkpoint loaded: {path} (step {state.step})")

    def _load_model(self, checkpoint: Dict[str, Any]) -> None:
        """Load model state."""
        missing, unexpected = self.model.load_state_dict(
            checkpoint["model"], strict=False
        )
        if missing:
            logger.warning(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    def _load_optimizer(self, checkpoint: Dict[str, Any], load: bool) -> None:
        """Load optimizer state."""
        if load and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _load_ema(self, checkpoint: Dict[str, Any], load: bool) -> None:
        """Load EMA state."""
        if load and "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])

    def _load_zclip(self, checkpoint: Dict[str, Any]) -> None:
        """Load ZClip state."""
        if "zclip" in checkpoint and self.zclip is not None:
            self.zclip.load_state_dict(checkpoint["zclip"])

    def _load_scaler(self, checkpoint: Dict[str, Any]) -> None:
        """Load GradScaler state."""
        if "scaler" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _load_scheduler(self, checkpoint: Dict[str, Any], load: bool) -> None:
        """Load LR scheduler state."""
        if load and "lr_scheduler" in checkpoint and self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _load_sampler(self, checkpoint: Dict[str, Any]) -> None:
        """Load sampler state."""
        if "sampler" not in checkpoint or self.dataloader is None:
            return

        sampler = getattr(self.dataloader, 'batch_sampler', None)
        if sampler is None:
            sampler = getattr(self.dataloader, 'sampler', None)

        if sampler is not None and hasattr(sampler, 'load_state_dict'):
            sampler.load_state_dict(checkpoint["sampler"])
            logger.info("Sampler state restored from checkpoint")

    def _load_state(self, checkpoint: Dict[str, Any], state: TrainerState) -> None:
        """Load trainer state."""
        saved_state = checkpoint.get("state", {})
        state.step = saved_state.get("step", 0)
        state.epoch = saved_state.get("epoch", 0)
        state.batch_idx = saved_state.get("batch_idx", 0)
        state.best_loss = saved_state.get("best_loss", float("inf"))
        state.total_samples = saved_state.get("total_samples", 0)


__all__ = ["CheckpointManager"]
