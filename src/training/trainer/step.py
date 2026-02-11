"""
Training Step Executor

Contains StepExecutor class for executing single training steps.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler

    from ...config.model_config import PixelHDMConfig
    from ..flow_matching import PixelHDMFlowMatching
    from ..losses import CombinedLoss
    from ..optimization import CPUMemoryCheckpoint, EMA, ZClip

from .metrics import TrainMetrics
from .optimizer_step import OptimizerStepMixin


class StepExecutor(OptimizerStepMixin):
    """Executes single training steps."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        flow_matching: "PixelHDMFlowMatching",
        combined_loss: "CombinedLoss",
        zclip: "ZClip",
        device: torch.device,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        scaler: Optional["GradScaler"] = None,
        ema: Optional["EMA"] = None,
        cpu_checkpoint: Optional["CPUMemoryCheckpoint"] = None,
        text_encoder: Optional[nn.Module] = None,
        config: Optional["PixelHDMConfig"] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.flow_matching = flow_matching
        self.combined_loss = combined_loss
        self.zclip = zclip
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.ema = ema
        self.cpu_checkpoint = cpu_checkpoint
        self.text_encoder = text_encoder
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._dino_encoder: Optional[nn.Module] = None
        self._null_text_embed: Optional[torch.Tensor] = None
        self._null_text_mask: Optional[torch.Tensor] = None
        self._loss_accum = 0.0
        self._loss_accum_steps = 0
        self._loss_dict_accum: Dict[str, float] = {}
        self._total_batch_size = 0
        self._step_start = 0.0

    def set_dino_encoder(self, encoder: nn.Module) -> None:
        """Set DINOv3 encoder for REPA loss."""
        self._dino_encoder = encoder

    def execute(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        warmup_fn: Optional[Callable[[int], None]] = None,
        warmup_steps: int = 0,
    ) -> Tuple[TrainMetrics, int]:
        """Execute single training step (standalone, no external micro-batches)."""
        self.reset_accumulators()
        self.forward_backward(batch, step=step, denom=1)
        return self.optimizer_step_and_metrics(
            step=step,
            lr_scheduler=lr_scheduler,
            warmup_fn=warmup_fn,
            warmup_steps=warmup_steps,
        )

    def forward_backward(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        denom: int,
    ) -> int:
        """Run one micro-batch forward/backward and accumulate losses/metrics."""
        if self._loss_accum_steps == 0:
            self._step_start = time.time()

        self.model.train()

        images, text_embeddings, text_mask = self._prepare_batch(batch)
        batch_size = images.shape[0]
        self._total_batch_size += batch_size

        t, z_t, x_clean, noise = self._prepare_training_data(images)
        compute_repa = self._should_compute_repa(step)

        loss_dict, _ = self._forward(
            z_t, t, x_clean, noise, text_embeddings, text_mask, step, compute_repa
        )
        self._backward_scaled(loss_dict["total"], denom=max(1, int(denom)))

        self._loss_accum += float(loss_dict["total"].item())
        self._loss_accum_steps += 1
        self._accumulate_loss_dict(loss_dict)
        return batch_size

    def optimizer_step_and_metrics(
        self,
        step: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        warmup_fn: Optional[Callable[[int], None]] = None,
        warmup_steps: int = 0,
    ) -> Tuple[TrainMetrics, int]:
        """Run optimizer update once for the current accumulation window."""
        if self._loss_accum_steps <= 0:
            raise RuntimeError("optimizer_step_and_metrics() called without accumulated micro-batches")

        avg_loss = self._loss_accum / max(self._loss_accum_steps, 1)
        grad_norm = self._optimizer_step(
            step, avg_loss, lr_scheduler, warmup_fn, warmup_steps
        )
        self._update_ema(step)

        metrics = self._create_metrics_from_accum(grad_norm)
        total_batch_size = self._total_batch_size
        self._reset_accumulators()
        return metrics, total_batch_size

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare batch data.

        Returns:
            Tuple of (images, text_embeddings, text_mask)
        """
        images = batch["images"].to(self.device)
        text_embeddings = batch.get("text_embeddings")
        text_mask = batch.get("text_mask")

        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(self.device)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)
        elif self.text_encoder is not None:
            text_embeddings, text_mask = self._encode_captions(batch)
            if text_embeddings is not None:
                text_embeddings = text_embeddings.to(self.device)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)

        text_embeddings, text_mask = self._apply_cfg_dropout(text_embeddings, text_mask)
        return images, text_embeddings, text_mask

    def _encode_captions(
        self, batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode captions using text encoder.

        Returns:
            Tuple of (text_embeddings, text_mask)
        """
        captions = batch.get("captions")
        if captions is None:
            return None, None

        text_encoder_frozen = True
        if self.config is not None:
            text_encoder_frozen = getattr(self.config, "text_encoder_frozen", True)

        def _do_encode() -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            result = self.text_encoder(texts=captions, return_pooled=False)
            if isinstance(result, tuple):
                if len(result) >= 2:
                    return result[0], result[1]
                return result, None
            if isinstance(result, dict):
                embeddings = result.get("hidden_states", result.get("last_hidden_state"))
                mask = result.get("attention_mask")
                return embeddings, mask
            return result, None

        if text_encoder_frozen:
            with torch.no_grad():
                return _do_encode()
        return _do_encode()

    def _get_null_text_embed(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get null embedding and mask for cfg dropout."""
        if self.text_encoder is None:
            raise ValueError("cfg_dropout requires a text_encoder to build null text.")

        text_encoder_frozen = True
        if self.config is not None:
            text_encoder_frozen = getattr(self.config, "text_encoder_frozen", True)

        if text_encoder_frozen and self._null_text_embed is not None:
            null_embed = self._null_text_embed
            null_mask = self._null_text_mask
        else:
            null_embed, null_mask = self._encode_captions({"captions": [""]})
            if null_embed is None:
                raise ValueError("Text encoder returned no embeddings for null text.")
            if text_encoder_frozen:
                self._null_text_embed = null_embed.detach()
                self._null_text_mask = null_mask.detach() if null_mask is not None else None

        null_embed = null_embed.to(device=device, dtype=dtype)
        if null_embed.shape[0] == 1 and batch_size > 1:
            null_embed = null_embed.expand(batch_size, -1, -1)
        elif null_embed.shape[0] != batch_size:
            raise ValueError(
                f"Null text embedding batch size {null_embed.shape[0]} does not match target batch size {batch_size}."
            )

        if null_mask is not None:
            null_mask = null_mask.to(device=device, dtype=torch.bool)
            if null_mask.shape[0] == 1 and batch_size > 1:
                null_mask = null_mask.expand(batch_size, -1)
            elif null_mask.shape[0] != batch_size:
                raise ValueError(
                    f"Null text mask batch size {null_mask.shape[0]} does not match target batch size {batch_size}."
                )

        return null_embed, null_mask

    def _apply_cfg_dropout(
        self,
        text_embeddings: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply classifier-free guidance dropout."""
        cfg_dropout = 0.0
        if self.config is not None:
            cfg_dropout = float(getattr(self.config, "cfg_dropout", 0.0))

        if cfg_dropout <= 0.0:
            return text_embeddings, text_mask
        if text_embeddings is None:
            raise ValueError("cfg_dropout > 0 requires text_embeddings in batch.")
        if self.text_encoder is None:
            raise ValueError("cfg_dropout > 0 requires a text_encoder.")

        batch_size, seq_len, _ = text_embeddings.shape
        if text_mask is None:
            text_mask = torch.ones(
                (batch_size, seq_len), device=text_embeddings.device, dtype=torch.bool
            )
        else:
            text_mask = text_mask.to(device=text_embeddings.device, dtype=torch.bool)

        null_embed, null_mask = self._get_null_text_embed(
            batch_size=batch_size,
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        if null_embed.shape[1] != seq_len:
            raise ValueError(
                f"Null text length {null_embed.shape[1]} does not match text length {seq_len}."
            )

        if null_mask is None:
            null_mask = torch.ones(
                (batch_size, seq_len), device=text_embeddings.device, dtype=torch.bool
            )
        else:
            null_mask = null_mask.to(device=text_embeddings.device, dtype=torch.bool)

        drop_mask = torch.rand(batch_size, device=text_embeddings.device) < cfg_dropout
        if not drop_mask.any():
            return text_embeddings, text_mask

        drop_mask_embed = drop_mask.view(-1, 1, 1)
        drop_mask_mask = drop_mask.view(-1, 1)
        text_embeddings = torch.where(drop_mask_embed, null_embed, text_embeddings)
        text_mask = torch.where(drop_mask_mask, null_mask, text_mask)
        return text_embeddings, text_mask

    def _prepare_training_data(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare flow matching training data."""
        t, z_t, x_clean, noise = self.flow_matching.prepare_training(images)

        if z_t.shape[1] == 3:
            z_t = z_t.permute(0, 2, 3, 1)
            x_clean = x_clean.permute(0, 2, 3, 1)
            noise = noise.permute(0, 2, 3, 1)

        return t, z_t, x_clean, noise

    def _should_compute_repa(self, step: int) -> bool:
        """Check if REPA loss should be computed."""
        if self._dino_encoder is None:
            return False
        if self.config is None:
            return True
        return step < getattr(self.config, "repa_early_stop", 250000)

    def _should_compute_gamma_l2(self) -> bool:
        """Check if gamma L2 penalty should be computed."""
        if self.config is None:
            return False
        return getattr(self.config, "pixel_gamma_l2_lambda", 0.0) > 0

    def _forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        step: int,
        compute_repa: bool,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Execute forward pass and loss computation."""
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        compute_gamma_l2 = self._should_compute_gamma_l2()

        with autocast(device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            if compute_gamma_l2:
                result = self.model(
                    z_t,
                    t,
                    text_embed=text_embeddings,
                    text_mask=text_mask,
                    return_features=compute_repa,
                    return_aux=True,
                )
                v_pred, h_t, gamma_l2 = result
            else:
                result = self.model(
                    z_t,
                    t,
                    text_embed=text_embeddings,
                    text_mask=text_mask,
                    return_features=compute_repa,
                )
                if isinstance(result, tuple):
                    v_pred, h_t = result
                else:
                    v_pred = result
                    h_t = None
                gamma_l2 = None

            loss_dict = self.combined_loss(
                v_pred=v_pred,
                x_clean=x_clean,
                noise=noise,
                h_t=h_t,
                step=step,
                gamma_l2=gamma_l2,
            )

        return loss_dict, v_pred

    def _backward(self, loss: torch.Tensor) -> None:
        """Execute backward pass."""
        scaled_loss = loss / self.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _backward_scaled(self, loss: torch.Tensor, denom: int) -> None:
        """Backward with explicit denominator (micro-batch count)."""
        scaled_loss = loss / max(1, int(denom))
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _accumulate_loss_dict(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        for key, value in loss_dict.items():
            self._loss_dict_accum[key] = self._loss_dict_accum.get(key, 0.0) + float(value.item())

    def _create_metrics_from_accum(self, grad_norm: float) -> TrainMetrics:
        """Create averaged metrics from the accumulation window."""
        n = max(self._loss_accum_steps, 1)
        step_time = max(time.time() - self._step_start, 1e-8)
        metrics = TrainMetrics(
            loss=self._loss_dict_accum.get("total", 0.0) / n,
            loss_vloss=self._loss_dict_accum.get("vloss", 0.0) / n,
            loss_freq=self._loss_dict_accum.get("freq_loss", 0.0) / n,
            loss_repa=self._loss_dict_accum.get("repa_loss", 0.0) / n,
            loss_gamma_l2=self._loss_dict_accum.get("gamma_l2", 0.0) / n,
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            samples_per_sec=self._total_batch_size / step_time,
            step_time=step_time,
        )
        return metrics

    def _create_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        grad_norm: float,
        batch_size: int,
        step_start: float,
    ) -> TrainMetrics:
        """Create training metrics."""
        step_time = max(time.time() - step_start, 1e-8)

        return TrainMetrics(
            loss=loss_dict["total"].item(),
            loss_vloss=loss_dict["vloss"].item(),
            loss_freq=loss_dict["freq_loss"].item(),
            loss_repa=loss_dict.get("repa_loss", torch.tensor(0.0)).item(),
            loss_gamma_l2=loss_dict.get("gamma_l2", torch.tensor(0.0)).item(),
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            samples_per_sec=batch_size / step_time,
            step_time=step_time,
        )

    def _reset_accumulators(self) -> None:
        self._loss_accum = 0.0
        self._loss_accum_steps = 0
        self._loss_dict_accum = {}
        self._total_batch_size = 0
        self._step_start = 0.0

    def reset_accumulators(self) -> None:
        """Public helper used by trainer/oom recovery tests."""
        self._reset_accumulators()


__all__ = ["StepExecutor"]
