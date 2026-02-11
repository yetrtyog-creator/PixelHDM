"""
Training Loop

Contains TrainingLoop class for managing the training process.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Tuple

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

from .metrics import TrainMetrics, TrainerState

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
        self._dataloader_len = self._detect_dataloader_len()
        self._validate_dataloader()

    def _detect_dataloader_len(self) -> Optional[int]:
        """Safely get dataloader length when available."""
        try:
            return len(self.dataloader)
        except TypeError:
            return None

    def _validate_dataloader(self) -> None:
        """Validate dataloader is not empty."""
        if self._dataloader_len is None:
            logger.warning(
                "Cannot determine dataloader length. "
                "Ensure your dataset is not empty."
            )
            return

        if self._dataloader_len == 0:
            raise ValueError(
                "Dataloader is empty (length=0). "
                "Check your dataset path and configuration."
            )

    def _resolve_accumulation_settings(self) -> Tuple[int, bool]:
        """Resolve accumulation behavior from training config."""
        grad_accum = 1
        drop_last = True
        if self.training_config is not None:
            grad_accum = int(getattr(self.training_config, "gradient_accumulation_steps", 1))
            drop_last = bool(getattr(self.training_config, "drop_last_accumulation", True))
        return max(1, grad_accum), drop_last

    def _resolve_training_mode(
        self,
        num_steps: Optional[int],
        num_epochs: Optional[int],
    ) -> str:
        """Resolve effective runtime training mode."""
        if num_epochs is not None:
            return "epochs"
        if num_steps is not None:
            return "steps"
        if self.training_config is None:
            return "steps"
        return getattr(self.training_config, "training_mode", "epochs")

    def _validate_unknown_len_policy(
        self,
        training_mode: str,
        save_every_epochs: int,
        log_every_epochs: int,
    ) -> None:
        """Enforce fail-fast policy when len(dataloader) is unavailable."""
        if self._dataloader_len is not None:
            return
        if training_mode == "epochs":
            raise ValueError("Epoch mode requires len(dataloader)")
        if save_every_epochs > 0 or log_every_epochs > 0:
            raise ValueError("len(dataloader) unavailable: epoch-based save/log cannot be enabled")
        logger.warning("len(dataloader) unavailable: epoch semantics disabled (steps mode only)")

    def _realign_epoch_from_step_on_resume(self) -> None:
        """Realign epoch counter from step budget on resume."""
        if self._dataloader_len is None or self._dataloader_len <= 0:
            return
        expected_epoch = self.state.step // self._dataloader_len
        if self.state.epoch != expected_epoch:
            logger.warning(
                "Realigning epoch from %d to %d based on step=%d and len(dataloader)=%d",
                self.state.epoch,
                expected_epoch,
                self.state.step,
                self._dataloader_len,
            )
            self.state.epoch = expected_epoch

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
        save_path, save_every_epochs, log_every_epochs = self._get_epoch_settings(
            save_path, save_every_epochs, log_every_epochs
        )
        training_mode = self._resolve_training_mode(num_steps, num_epochs)
        self._validate_unknown_len_policy(training_mode, save_every_epochs, log_every_epochs)
        total_steps = self._calculate_total_steps(num_steps, num_epochs)
        self._realign_epoch_from_step_on_resume()

        self._log_training_config(
            total_steps, save_interval, save_every_epochs, log_interval, log_every_epochs
        )

        pbar = self._create_progress_bar(total_steps, use_progress_bar)
        data_iter = iter(self.dataloader)

        try:
            self._loop(
                data_iter,
                total_steps,
                pbar,
                train_step_fn,
                save_checkpoint_fn,
                log_interval,
                save_interval,
                save_every_epochs,
                log_every_epochs,
                gc_interval,
                save_path,
                callback,
                *self._resolve_accumulation_settings(),
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
            if self._dataloader_len is None:
                raise ValueError("Epoch mode requires len(dataloader)")
            return num_epochs * self._dataloader_len

        if self.training_config:
            mode = getattr(self.training_config, "training_mode", "epochs")
            if mode == "epochs":
                if self._dataloader_len is None:
                    raise ValueError("Epoch mode requires len(dataloader)")
                return self.training_config.num_epochs * self._dataloader_len
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
                save_every_epochs = getattr(self.training_config, "save_every_epochs", 0)
            if log_every_epochs == 0:
                log_every_epochs = getattr(self.training_config, "log_every_epochs", 0)
            if save_path is None:
                save_path = getattr(self.training_config, "checkpoint_dir", None)

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
        train_step_fn: Callable[[Dict[str, Any] | list[Dict[str, Any]]], TrainMetrics],
        save_checkpoint_fn: Callable[..., None],
        log_interval: int,
        save_interval: int,
        save_every_epochs: int,
        log_every_epochs: int,
        gc_interval: int,
        save_path: Optional[str | Path],
        callback: Optional[Callable[[int, TrainMetrics], None]],
        gradient_accumulation_steps: int,
        drop_last_accumulation: bool,
    ) -> None:
        """Main training loop."""
        while self.state.step < total_steps:
            micro_batches, data_iter = self._collect_micro_batches(
                data_iter,
                gradient_accumulation_steps=gradient_accumulation_steps,
                drop_last_accumulation=drop_last_accumulation,
            )
            metrics = train_step_fn(micro_batches)

            epoch_boundary, epoch_advanced = self._sync_epoch_with_step()
            if epoch_advanced and log_every_epochs > 0 and self.state.epoch % log_every_epochs == 0:
                logger.info(f"Epoch {self.state.epoch} completed")

            self._handle_gc(gc_interval)
            self._update_progress_bar(pbar, metrics)
            self._handle_logging(log_interval, total_steps, metrics)

            if callback is not None:
                callback(self.state.step, metrics)

            self._handle_periodic_checkpoint(
                save_checkpoint_fn=save_checkpoint_fn,
                save_path=save_path,
                save_interval=save_interval,
                save_every_epochs=save_every_epochs,
                epoch_boundary=epoch_boundary,
            )

    def _collect_micro_batches(
        self,
        data_iter: Iterator[Dict[str, Any]],
        gradient_accumulation_steps: int,
        drop_last_accumulation: bool,
    ) -> Tuple[Dict[str, Any] | list[Dict[str, Any]], Iterator[Dict[str, Any]]]:
        """Collect one training payload (single batch or micro-batch window)."""
        if gradient_accumulation_steps <= 1:
            batch, data_iter = self._get_batch(data_iter)
            return batch, data_iter

        if (
            drop_last_accumulation
            and self._dataloader_len is not None
            and self.state.batch_idx > 0
            and (self.state.batch_idx + gradient_accumulation_steps - 1) > self._dataloader_len
        ):
            data_iter = iter(self.dataloader)
            self.state.batch_idx = 0

        micro_batches: list[Dict[str, Any]] = []
        for _ in range(gradient_accumulation_steps):
            batch, data_iter = self._get_batch(data_iter)
            micro_batches.append(batch)
        return micro_batches, data_iter

    def _get_batch(
        self,
        data_iter: Iterator[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Get next batch, handling dataloader reset."""
        try:
            batch = next(data_iter)
            self.state.batch_idx += 1
            return batch, data_iter
        except StopIteration:
            data_iter = iter(self.dataloader)
            self.state.batch_idx = 0
            batch = next(data_iter)
            self.state.batch_idx += 1
            return batch, data_iter

    def _sync_epoch_with_step(self) -> Tuple[bool, bool]:
        """Synchronize epoch with step-based boundaries.

        Returns:
            (epoch_boundary, epoch_advanced)
        """
        if self._dataloader_len is None or self._dataloader_len <= 0:
            return False, False

        expected_epoch = self.state.step // self._dataloader_len
        epoch_advanced = expected_epoch > self.state.epoch
        if epoch_advanced:
            self.state.epoch = expected_epoch

        epoch_boundary = self.state.step > 0 and self.state.step % self._dataloader_len == 0
        return epoch_boundary, epoch_advanced

    def _current_epoch_for_checkpoint(self) -> int:
        """Get current epoch index (1-based) for checkpoint naming."""
        if self._dataloader_len is not None and self._dataloader_len > 0 and self.state.step > 0:
            return ((self.state.step - 1) // self._dataloader_len) + 1

        if self.state.epoch > 0:
            return self.state.epoch
        return 1

    def _handle_periodic_checkpoint(
        self,
        save_checkpoint_fn: Callable[..., None],
        save_path: Optional[str | Path],
        save_interval: int,
        save_every_epochs: int,
        epoch_boundary: bool,
    ) -> None:
        """Handle periodic checkpoint saving with trigger dedupe."""
        if not save_path:
            return

        step_trigger = save_interval > 0 and self.state.step % save_interval == 0
        epoch_trigger = False
        if epoch_boundary and save_every_epochs > 0:
            epoch_trigger = self.state.epoch > 0 and self.state.epoch % save_every_epochs == 0

        if not (step_trigger or epoch_trigger):
            return

        checkpoint_name = f"checkpoint_epoch{self._current_epoch_for_checkpoint()}_step{self.state.step}"
        save_checkpoint_fn(save_path, checkpoint_name=checkpoint_name)

    def _handle_gc(self, gc_interval: int):
        """Handle garbage collection."""
        if gc_interval > 0 and self.state.step % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _update_progress_bar(self, pbar, metrics: TrainMetrics):
        """Update progress bar."""
        if pbar is not None:
            pbar.set_postfix(
                {
                    "loss": f"{metrics.loss:.4f}",
                    "v": f"{metrics.loss_vloss:.3f}",
                    "f": f"{metrics.loss_freq:.3f}",
                    "r": f"{metrics.loss_repa:.3f}",
                    "lr": f"{metrics.learning_rate:.1e}",
                }
            )
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


__all__ = ["TrainingLoop"]
