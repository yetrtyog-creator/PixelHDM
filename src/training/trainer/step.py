"""
Training Step Executor

Contains StepExecutor class for executing single training steps.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.amp import autocast

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler
    from ...config.model_config import PixelHDMConfig
    from ..flow_matching import PixelHDMFlowMatching
    from ..losses import CombinedLoss
    from ..optimization import ZClip, EMA, CPUMemoryCheckpoint

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
        """Execute single training step."""
        step_start = time.time()
        self.model.train()

        images, text_embeddings, text_mask, pooled_text_embed = self._prepare_batch(batch)
        batch_size = images.shape[0]

        t, z_t, x_clean, noise = self._prepare_training_data(images)
        compute_repa = self._should_compute_repa(step)

        loss_dict, x_pred = self._forward(
            z_t, t, x_clean, noise, text_embeddings, text_mask, pooled_text_embed, step, compute_repa
        )

        self._backward(loss_dict["total"])

        grad_norm = self._optimizer_step(
            step, loss_dict["total"].item(), lr_scheduler, warmup_fn, warmup_steps
        )

        self._update_ema(step)

        return self._create_metrics(
            loss_dict, grad_norm, batch_size, step_start
        ), batch_size

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare batch data.

        Returns:
            Tuple of (images, text_embeddings, text_mask, pooled_text_embed)
        """
        images = batch["images"].to(self.device)
        text_embeddings = batch.get("text_embeddings")
        text_mask = batch.get("text_mask")
        pooled_text_embed = batch.get("pooled_text_embed")

        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(self.device)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)
            if pooled_text_embed is not None:
                pooled_text_embed = pooled_text_embed.to(self.device)
        elif self.text_encoder is not None:
            text_embeddings, text_mask, pooled_text_embed = self._encode_captions(batch)
            # Move encoded tensors to training device
            if text_embeddings is not None:
                text_embeddings = text_embeddings.to(self.device)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)
            if pooled_text_embed is not None:
                pooled_text_embed = pooled_text_embed.to(self.device)

        return images, text_embeddings, text_mask, pooled_text_embed

    def _encode_captions(
        self, batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode captions using text encoder.

        Returns:
            Tuple of (text_embeddings, text_mask, pooled_text_embed)

        Design Note:
            使用 torch.no_grad() 包裝是正確的設計選擇：
            1. 文本編碼器已凍結 (requires_grad=False)，梯度不會傳回
            2. 即使移除 no_grad，梯度也無法流回凍結的編碼器參數
            3. no_grad() 提供性能優化：不記錄計算圖，節省記憶體
            4. 梯度通過 TextProjector (如果有) 傳播，不需要流回文本編碼器
        """
        captions = batch.get("captions")
        if captions is None:
            return None, None, None

        # 文本編碼器已凍結，使用 no_grad 節省記憶體（見上方 Design Note）
        with torch.no_grad():
            result = self.text_encoder(texts=captions, return_pooled=True)
            if isinstance(result, tuple):
                if len(result) == 3:
                    # (hidden_states, attention_mask, pooled_output)
                    return result
                elif len(result) == 2:
                    # Legacy format: (hidden_states, attention_mask)
                    return result[0], result[1], None
            elif isinstance(result, dict):
                embeddings = result.get("hidden_states", result.get("last_hidden_state"))
                mask = result.get("attention_mask")
                pooled = result.get("pooled_output")
                return embeddings, mask, pooled
            return result, None, None

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
        return step < getattr(self.config, 'repa_early_stop', 250000)

    def _should_compute_gamma_l2(self) -> bool:
        """Check if gamma L2 penalty should be computed."""
        if self.config is None:
            return False
        return getattr(self.config, 'pixel_gamma_l2_lambda', 0.0) > 0

    def _forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        pooled_text_embed: Optional[torch.Tensor],
        step: int,
        compute_repa: bool,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Execute forward pass.

        V-Prediction: 模型直接輸出 velocity v = x - ε
        """
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        compute_gamma_l2 = self._should_compute_gamma_l2()

        with autocast(device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            # Use return_aux when we need gamma_l2
            if compute_gamma_l2:
                result = self.model(
                    z_t, t,
                    text_embed=text_embeddings,
                    text_mask=text_mask,
                    pooled_text_embed=pooled_text_embed,
                    return_features=compute_repa,
                    return_aux=True,
                )
                # return_aux returns (v_pred, repa_features_or_None, gamma_l2)
                v_pred, h_t, gamma_l2 = result
            else:
                result = self.model(
                    z_t, t,
                    text_embed=text_embeddings,
                    text_mask=text_mask,
                    pooled_text_embed=pooled_text_embed,
                    return_features=compute_repa,
                )
                if isinstance(result, tuple):
                    v_pred, h_t = result
                else:
                    v_pred = result
                    h_t = None
                gamma_l2 = None

            # V-Prediction: combined_loss 直接接收 velocity
            loss_dict = self.combined_loss(
                v_pred=v_pred, x_clean=x_clean,
                noise=noise, h_t=h_t, step=step,
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

    def _create_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        grad_norm: float,
        batch_size: int,
        step_start: float,
    ) -> TrainMetrics:
        """Create training metrics."""
        step_time = time.time() - step_start

        return TrainMetrics(
            loss=loss_dict["total"].item(),
            loss_vloss=loss_dict["vloss"].item(),
            loss_freq=loss_dict["freq_loss"].item(),
            loss_repa=loss_dict["repa_loss"].item(),
            loss_gamma_l2=loss_dict.get("gamma_l2", torch.tensor(0.0)).item(),
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            samples_per_sec=batch_size / step_time,
            step_time=step_time,
        )


__all__ = ["StepExecutor"]
