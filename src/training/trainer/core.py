"""
PixelHDM Trainer Core

Main Trainer class that integrates all training components.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional
import gc
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig, TrainingConfig

from .metrics import TrainerState, TrainMetrics
from .scheduler_factory import create_lr_scheduler, apply_warmup_lr
from .step import StepExecutor
from .checkpoint import CheckpointManager
from .loop import TrainingLoop
from .init_helpers import init_optimizer, init_training_components, init_amp, get_training_params

logger = logging.getLogger(__name__)


class Trainer:
    """PixelHDM Trainer integrating all training components."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional["PixelHDMConfig"] = None,
        training_config: Optional["TrainingConfig"] = None,
        dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        text_encoder: Optional[nn.Module] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.training_config = training_config
        self.dataloader = dataloader
        self._init_all(optimizer, text_encoder)

    def _init_all(self, optimizer, text_encoder) -> None:
        """Initialize all components."""
        self.max_grad_norm, self.gradient_accumulation_steps = get_training_params(self.training_config)
        self.optimizer = init_optimizer(self.model, self.training_config, self.device, optimizer)
        (self.flow_matching, self.combined_loss, self.zclip,
         self.ema, self.cpu_checkpoint) = init_training_components(
            self.config, self.training_config, self.model, self.device
        )
        self._init_text_encoder(text_encoder)
        self.use_amp, self.amp_dtype, self.scaler = init_amp(self.training_config)
        self.state = TrainerState()
        self._lr_scheduler = None
        self._scheduler_skip_sync = False  # Flag to skip step sync on reset
        self._init_executors()

    def _init_text_encoder(self, text_encoder: Optional[nn.Module]) -> None:
        """Initialize text encoder."""
        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def _init_executors(self) -> None:
        """Initialize step executor and checkpoint manager."""
        self._step_executor = StepExecutor(
            model=self.model, optimizer=self.optimizer,
            flow_matching=self.flow_matching, combined_loss=self.combined_loss,
            zclip=self.zclip, device=self.device, use_amp=self.use_amp,
            amp_dtype=self.amp_dtype, scaler=self.scaler, ema=self.ema,
            cpu_checkpoint=self.cpu_checkpoint, text_encoder=self.text_encoder,
            config=self.config, gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
        )
        self._checkpoint_manager = CheckpointManager(
            model=self.model, optimizer=self.optimizer, device=self.device,
            config=self.config, training_config=self.training_config,
            ema=self.ema, zclip=self.zclip, scaler=self.scaler, dataloader=self.dataloader,
        )

    @property
    def lr_scheduler(self):
        """Lazy create LR scheduler.

        When scheduler is newly created (e.g., after reset_scheduler=True),
        it must be synchronized to the current training step to ensure
        correct learning rate calculation.

        IMPORTANT: state.step is batch steps, but scheduler expects optimizer steps.
        optimizer_steps = batch_steps / gradient_accumulation_steps
        """
        if self._lr_scheduler is None:
            self._lr_scheduler = create_lr_scheduler(
                self.optimizer, self.training_config, self.dataloader, self.gradient_accumulation_steps
            )
            self._checkpoint_manager.set_lr_scheduler(self._lr_scheduler)

            # Sync scheduler to current step ONLY if NOT reset
            # CRITICAL: Convert batch steps to optimizer steps!
            # state.step = batch steps (increments every batch)
            # scheduler expects optimizer steps (increments every gradient_accumulation_steps batches)
            # When reset_scheduler=True, skip sync to start fresh from cycle 0
            if self.state.step > 0 and not self._scheduler_skip_sync:
                optimizer_steps = self.state.step // self.gradient_accumulation_steps
                self._sync_scheduler_to_step(optimizer_steps)
            # Reset flag after use
            self._scheduler_skip_sync = False

        return self._lr_scheduler

    def _sync_scheduler_to_step(self, target_step: int) -> None:
        """Synchronize scheduler state to target step.

        This is needed when reset_scheduler=True during resume, so the new
        scheduler computes LR based on the actual training progress.

        Args:
            target_step: The optimizer step (NOT batch step!) to sync to
        """
        if self._lr_scheduler is None:
            return

        # Use step(epoch) to set scheduler state directly if supported
        if hasattr(self._lr_scheduler, '_compute_state_from_epoch'):
            # SteppedCosineRestartScheduler supports direct state computation
            self._lr_scheduler.step(target_step)
            logger.info(
                f"Scheduler synced to step {target_step}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            # For other schedulers, step() incrementally
            for _ in range(target_step):
                self._lr_scheduler.step()
            logger.info(
                f"Scheduler advanced to step {target_step}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
            )

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _warmup_lr(self, step: int) -> None:
        """Apply warmup learning rate."""
        if self.training_config:
            apply_warmup_lr(self.optimizer, step, self.training_config.warmup_steps, self.training_config.learning_rate)

    def set_dino_encoder(self, encoder: nn.Module) -> None:
        """Set DINOv3 encoder for REPA Loss."""
        self.combined_loss.set_dino_encoder(encoder)
        self._step_executor.set_dino_encoder(encoder)
        self._dino_encoder = encoder

    def set_text_encoder(self, encoder: nn.Module) -> None:
        """Set text encoder."""
        self.text_encoder = encoder.to(self.device)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self._step_executor.text_encoder = self.text_encoder

    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainMetrics:
        """Execute single training step."""
        warmup_steps = self.training_config.warmup_steps if self.training_config else 0
        metrics, batch_size = self._step_executor.execute(
            batch, self.state.step, lr_scheduler=self.lr_scheduler,
            warmup_fn=self._warmup_lr, warmup_steps=warmup_steps,
        )
        self.state.step += 1
        self.state.total_samples += batch_size
        return metrics

    def _halve_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Halve batch size."""
        return {k: v[:v.size(0)//2] if isinstance(v, torch.Tensor) and v.size(0) > 1 else v for k, v in batch.items()}

    def safe_train_step(self, batch: Dict[str, torch.Tensor], retry_on_oom: bool = True, max_retries: int = 3) -> Optional[TrainMetrics]:
        """Train step with OOM recovery.

        Automatically halves batch size on OOM. Returns None if batch cannot be
        reduced further (size <= 1) or max retries exceeded.
        """
        retries, current_batch = 0, batch
        while retries <= max_retries:
            try:
                return self.train_step(current_batch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and retry_on_oom:
                    retries += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    if retries <= max_retries:
                        # Check if batch can be halved (size > 1)
                        current_size = current_batch["images"].size(0)
                        if current_size <= 1:
                            logger.warning(f"OOM: Cannot reduce batch size further (size={current_size})")
                            return None
                        current_batch = self._halve_batch(current_batch)
                        new_size = current_batch["images"].size(0)
                        logger.warning(f"OOM: Reduced batch size {current_size} -> {new_size} (retry {retries}/{max_retries})")
                        if new_size == 0:
                            return None
                    elif retries > max_retries:
                        logger.warning(f"OOM: Max retries ({max_retries}) exceeded")
                        return None
                else:
                    raise
        return None

    def train(
        self, num_steps: Optional[int] = None, num_epochs: Optional[int] = None,
        log_interval: int = 100, save_interval: int = 1000,
        save_every_epochs: int = 0, log_every_epochs: int = 0,
        gc_interval: int = 100, save_path: Optional[str | Path] = None,
        callback: Optional[Callable[[int, TrainMetrics], None]] = None, use_progress_bar: bool = True,
    ) -> None:
        """Run training loop."""
        if self.dataloader is None:
            raise RuntimeError("dataloader required for training")
        loop = TrainingLoop(self.dataloader, self.training_config, self.state)
        loop.run(
            train_step_fn=self.train_step, save_checkpoint_fn=self.save_checkpoint,
            num_steps=num_steps, num_epochs=num_epochs, log_interval=log_interval,
            save_interval=save_interval, save_every_epochs=save_every_epochs,
            log_every_epochs=log_every_epochs, gc_interval=gc_interval,
            save_path=save_path, callback=callback, use_progress_bar=use_progress_bar,
        )

    def save_checkpoint(self, path: str | Path, checkpoint_name: Optional[str] = None, cleanup: bool = True) -> None:
        """Save checkpoint."""
        self._checkpoint_manager.set_lr_scheduler(self._lr_scheduler)
        self._checkpoint_manager.save(path, self.state, checkpoint_name, cleanup)

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True, load_ema: bool = True, load_scheduler: bool = True) -> None:
        """Load checkpoint."""
        if load_scheduler:
            _ = self.lr_scheduler
            self._checkpoint_manager.set_lr_scheduler(self._lr_scheduler)
        self._checkpoint_manager.load(path, self.state, load_optimizer, load_ema, load_scheduler)

    def _cleanup_old_checkpoints(self, save_dir: Path, max_keep: int, pattern: str = "checkpoint_*.pt") -> None:
        """Cleanup old checkpoints (backward compatibility)."""
        self._checkpoint_manager._cleanup_old_checkpoints(save_dir, pattern)


def create_trainer(
    model: nn.Module, config: Optional["PixelHDMConfig"] = None,
    training_config: Optional["TrainingConfig"] = None, dataloader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None, text_encoder: Optional[nn.Module] = None,
) -> Trainer:
    """Create Trainer instance."""
    return Trainer(model=model, config=config, training_config=training_config,
                   dataloader=dataloader, device=device, text_encoder=text_encoder)


__all__ = ["Trainer", "create_trainer"]
