"""
PixelHDM-RPEA-DinoV3 優化器工廠

支援:
    - AdamW (PyTorch 標準)
    - AdamW 8bit (bitsandbytes)
    - 參數分組 (不同學習率)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
import logging

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...config.model_config import TrainingConfig

logger = logging.getLogger(__name__)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_modules: Optional[Tuple[type, ...]] = None,
    no_decay_names: Optional[Tuple[str, ...]] = None,
    lr_scale_patterns: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    創建參數分組，支援:
    - 不同模塊類型不同 weight decay
    - 不同參數名稱不同學習率

    Args:
        model: 模型
        weight_decay: 預設 weight decay
        no_decay_modules: 不應用 weight decay 的模塊類型
        no_decay_names: 不應用 weight decay 的參數名稱關鍵字
        lr_scale_patterns: 學習率縮放模式 {"pattern": scale}

    Returns:
        參數分組列表
    """
    if no_decay_modules is None:
        no_decay_modules = (nn.LayerNorm, nn.Embedding)

    if no_decay_names is None:
        no_decay_names = ("bias", "norm", "embeddings")

    lr_scale_patterns = lr_scale_patterns or {}

    # 收集參數
    decay_params: Dict[float, List[torch.nn.Parameter]] = {}
    no_decay_params: Dict[float, List[torch.nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 計算學習率縮放
        lr_scale = 1.0
        for pattern, scale in lr_scale_patterns.items():
            if pattern in name:
                lr_scale = scale
                break

        # 判斷是否應用 weight decay
        apply_decay = True

        # 檢查模塊類型
        for module_name, module in model.named_modules():
            if module_name and name.startswith(module_name):
                if isinstance(module, no_decay_modules):
                    apply_decay = False
                    break

        # 檢查參數名稱
        if any(nd in name.lower() for nd in no_decay_names):
            apply_decay = False

        # 分組
        if apply_decay:
            if lr_scale not in decay_params:
                decay_params[lr_scale] = []
            decay_params[lr_scale].append(param)
        else:
            if lr_scale not in no_decay_params:
                no_decay_params[lr_scale] = []
            no_decay_params[lr_scale].append(param)

    # 構建參數組
    param_groups = []

    for lr_scale, params in decay_params.items():
        param_groups.append({
            "params": params,
            "weight_decay": weight_decay,
            "lr_scale": lr_scale,
        })

    for lr_scale, params in no_decay_params.items():
        param_groups.append({
            "params": params,
            "weight_decay": 0.0,
            "lr_scale": lr_scale,
        })

    # 統計
    total_params = sum(p.numel() for group in param_groups for p in group["params"])
    logger.info(f"Created {len(param_groups)} parameter groups, total {total_params:,} parameters")

    return param_groups


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    use_8bit: bool = False,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    創建優化器

    Args:
        model: 模型
        optimizer_type: 優化器類型 ("adamw", "adam", "sgd")
        lr: 學習率
        weight_decay: 權重衰減
        betas: Adam betas
        eps: Adam epsilon
        use_8bit: 是否使用 8bit 優化器 (需要 bitsandbytes)
        **kwargs: 額外參數

    Returns:
        優化器實例
    """
    # 獲取參數分組
    param_groups = get_parameter_groups(
        model,
        weight_decay=weight_decay,
        lr_scale_patterns=kwargs.get("lr_scale_patterns"),
    )

    # 設置基礎學習率
    for group in param_groups:
        lr_scale = group.pop("lr_scale", 1.0)
        group["lr"] = lr * lr_scale

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw":
        if use_8bit:
            # bitsandbytes 8-bit optimizer requires CUDA
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using standard AdamW instead of 8bit")
            else:
                try:
                    import bitsandbytes as bnb
                    logger.info("Using AdamW 8bit (bitsandbytes)")
                    return bnb.optim.AdamW8bit(
                        param_groups,
                        lr=lr,
                        betas=betas,
                        eps=eps,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, using standard AdamW")

        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    elif optimizer_type == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
        )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_optimizer_from_config(
    model: nn.Module,
    config: "TrainingConfig",
) -> torch.optim.Optimizer:
    """從配置創建優化器"""
    return create_optimizer(
        model=model,
        optimizer_type=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


__all__ = [
    "get_parameter_groups",
    "create_optimizer",
    "create_optimizer_from_config",
]
