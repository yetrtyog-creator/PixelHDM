"""
Training Loop

Contains TrainingLoop class for managing the training process.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Tuple
import gc
import logging

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from ...config.model_config import TrainingConfig

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from .metrics import TrainerState, TrainMetrics

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Manages the main training loop."""

    def __init__(
        self,
        dataloader: DataLoader,
        training_config: Optional["TrainingConfig"],
        state: TrainerState,
    ) -> None:
        self.dataloader = dataloader
        self.training_config = training_config
        self.state = state
        self._validate_dataloader()

    def _validate_dataloader(self) -> None:
        """Validate dataloader is not empty."""
        try:
            dataloader_len = len(self.dataloader)
            if dataloader_len == 0:
                raise ValueError(
                    "Dataloader is empty (length=0). "
                    "Check your dataset path and configuration."
                )
        except TypeError:
            # DataLoader with IterableDataset doesn't support len()
            logger.warning(
                "Cannot determine dataloader length. "
                "Ensure your dataset is not empty."
            )

    def run(
        self,
        train_step_fn: Callable,
        save_checkpoint_fn: Callable,
        num_steps: Optional[int] = None,
        num_epochs: Optional[int] = None,
        log_interval: int = 100,
        save_interval: int = 1000,
        save_every_epochs: int = 0,
        log_every_epochs: int = 0,
        gc_interval: int = 100,
        save_path: Optional[str | Path] = None,
        callback: Optional[Callable[[int, TrainMetrics], None]] = None,
        use_progress_bar: bool = True,
    ) -> None:
        """Run training loop."""
        total_steps = self._calculate_total_steps(num_steps, num_epochs)
        save_path, save_every_epochs, log_every_epochs = self._get_epoch_settings(
            save_path, save_every_epochs, log_every_epochs
        )

        self._log_training_config(
            total_steps, save_interval, save_every_epochs, log_interval, log_every_epochs
        )

        pbar = self._create_progress_bar(total_steps, use_progress_bar)
        data_iter = iter(self.dataloader)

        try:
            self._loop(
                data_iter, total_steps, pbar, train_step_fn, save_checkpoint_fn,
                log_interval, save_interval, save_every_epochs, log_every_epochs,
                gc_interval, save_path, callback
            )
        finally:
            if pbar is not None:
                pbar.close()

        if save_path is not None:
            save_checkpoint_fn(save_path, checkpoint_name="checkpoint_completed")
        logger.info(f"Training completed: {self.state.step} steps")

    def _calculate_total_steps(
        self, num_steps: Optional[int], num_epochs: Optional[int]
    ) -> int:
        """Calculate total training steps."""
        if num_steps is not None:
            return num_steps
        if num_epochs is not None:
            return num_epochs * len(self.dataloader)

        if self.training_config:
            mode = getattr(self.training_config, 'training_mode', 'epochs')
            if mode == "epochs":
                return self.training_config.num_epochs * len(self.dataloader)
            return self.training_config.max_steps
        return 10000

    def _get_epoch_settings(
        self,
        save_path: Optional[str | Path],
        save_every_epochs: int,
        log_every_epochs: int,
    ) -> Tuple[Optional[str | Path], int, int]:
        """Get epoch-based settings from config."""
        if self.training_config:
            if save_every_epochs == 0:
                save_every_epochs = getattr(self.training_config, 'save_every_epochs', 0)
            if log_every_epochs == 0:
                log_every_epochs = getattr(self.training_config, 'log_every_epochs', 0)
            if save_path is None and save_every_epochs > 0:
                save_path = getattr(self.training_config, 'checkpoint_dir', None)
        return save_path, save_every_epochs, log_every_epochs

    def _log_training_config(
        self,
        total_steps: int,
        save_interval: int,
        save_every_epochs: int,
        log_interval: int,
        log_every_epochs: int,
    ) -> None:
        """Log training configuration."""
        logger.info(f"Starting training for {total_steps} steps")
        if save_interval > 0:
            logger.info(f"  - Save every {save_interval} steps")
        if save_every_epochs > 0:
            logger.info(f"  - Save every {save_every_epochs} epochs")
        if log_interval > 0:
            logger.info(f"  - Log every {log_interval} steps")
        if log_every_epochs > 0:
            logger.info(f"  - Log every {log_every_epochs} epochs")

    def _create_progress_bar(self, total_steps: int, use_progress_bar: bool) -> Optional[Any]:
        """Create progress bar if available."""
        if use_progress_bar and TQDM_AVAILABLE:
            return tqdm(
                total=total_steps, initial=self.state.step, desc="Training", unit="step"
            )
        return None

    def _loop(
        self,
        data_iter: Iterator[Dict[str, Any]],
        total_steps: int,
        pbar: Optional[Any],
        train_step_fn: Callable[[Dict[str, Any]], TrainMetrics],
        save_checkpoint_fn: Callable[..., None],
        log_interval: int,
        save_interval: int,
        save_every_epochs: int,
        log_every_epochs: int,
        gc_interval: int,
        save_path: Optional[str | Path],
        callback: Optional[Callable[[int, TrainMetrics], None]],
    ) -> None:
        """Main training loop."""
        while self.state.step < total_steps:
            batch, data_iter = self._get_batch(
                data_iter, save_checkpoint_fn, save_path, save_every_epochs, log_every_epochs
            )
            metrics = train_step_fn(batch)

            self._handle_gc(gc_interval)
            self._update_progress_bar(pbar, metrics)
            self._handle_logging(log_interval, total_steps, metrics)

            if callback is not None:
                callback(self.state.step, metrics)

            self._handle_step_checkpoint(save_checkpoint_fn, save_path, save_interval)

    def _get_batch(
        self,
        data_iter: Iterator[Dict[str, Any]],
        save_checkpoint_fn: Callable[..., None],
        save_path: Optional[str | Path],
        save_every_epochs: int,
        log_every_epochs: int,
    ) -> Tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Get next batch, handling epoch boundaries."""
        try:
            batch = next(data_iter)
            self.state.batch_idx += 1
            return batch, data_iter
        except StopIteration:
            self.state.epoch += 1
            self.state.batch_idx = 0

            if log_every_epochs > 0 and self.state.epoch % log_every_epochs == 0:
                logger.info(f"Epoch {self.state.epoch} completed")
            if save_path and save_every_epochs > 0:
                if self.state.epoch % save_every_epochs == 0:
                    save_checkpoint_fn(
                        save_path, checkpoint_name=f"checkpoint_epoch_{self.state.epoch}"
                    )

            data_iter = iter(self.dataloader)
            batch = next(data_iter)
            self.state.batch_idx += 1
            return batch, data_iter

    def _handle_gc(self, gc_interval: int):
        """Handle garbage collection."""
        if gc_interval > 0 and self.state.step % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _update_progress_bar(self, pbar, metrics: TrainMetrics):
        """Update progress bar."""
        if pbar is not None:
            pbar.set_postfix({
                "loss": f"{metrics.loss:.4f}",
                "v": f"{metrics.loss_vloss:.3f}",
                "f": f"{metrics.loss_freq:.3f}",
                "r": f"{metrics.loss_repa:.3f}",
                "lr": f"{metrics.learning_rate:.1e}",
            })
            pbar.update(1)

    def _handle_logging(self, log_interval: int, total_steps: int, metrics: TrainMetrics):
        """Handle step-based logging."""
        if log_interval > 0 and self.state.step % log_interval == 0:
            logger.info(
                f"Step {self.state.step}/{total_steps} | "
                f"Loss: {metrics.loss:.4f} "
                f"(v:{metrics.loss_vloss:.3f} f:{metrics.loss_freq:.3f} "
                f"r:{metrics.loss_repa:.3f}) | "
                f"LR: {metrics.learning_rate:.2e}"
            )

    def _handle_step_checkpoint(
        self, save_checkpoint_fn: Callable, save_path: Optional[str | Path], save_interval: int
    ):
        """Handle step-based checkpoint saving."""
        if save_path and save_interval > 0 and self.state.step % save_interval == 0:
            save_checkpoint_fn(save_path)


__all__ = ["TrainingLoop"]
