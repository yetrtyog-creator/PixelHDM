"""
Dropout 工具函數

包含:
    - configure_selective_dropout: 配置選擇性 Dropout
    - get_dropout_stats: 獲取 Dropout 統計信息

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


def _parse_layer_index(name: str) -> Optional[int]:
    """
    從模組名稱解析層索引

    Args:
        name: 模組名稱 (如 "layers.0.attention")

    Returns:
        層索引，或 None 如果無法解析
    """
    try:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                return int(parts[i + 1])
    except (ValueError, IndexError):
        pass
    return None


def _should_skip_layer(name: str, layer_indices: Optional[list[int]]) -> bool:
    """
    檢查是否應該跳過該層

    Args:
        name: 模組名稱
        layer_indices: 目標層索引列表

    Returns:
        True 如果應該跳過
    """
    if layer_indices is None:
        return False

    layer_idx = _parse_layer_index(name)
    return layer_idx is not None and layer_idx not in layer_indices


def _configure_attention_dropout(
    module: nn.Module,
    dropout_rate: float,
    disable: bool,
) -> None:
    """配置注意力模組的 dropout"""
    if disable:
        module.dropout_p = 0.0
        if hasattr(module, "dropout") and isinstance(module.dropout, nn.Dropout):
            module.dropout.p = 0.0
    elif dropout_rate >= 0:
        module.dropout_p = dropout_rate
        if hasattr(module, "dropout") and isinstance(module.dropout, nn.Dropout):
            module.dropout.p = dropout_rate


def configure_selective_dropout(
    model: nn.Module,
    dropout_rate: float = 0.0,
    layer_indices: Optional[list[int]] = None,
    disable_attention_dropout: bool = False,
    disable_ffn_dropout: bool = False,
) -> None:
    """
    配置選擇性 Dropout

    允許對模型的不同部分應用不同的 dropout 策略：
    - 特定層啟用/禁用 dropout
    - 注意力和 FFN 分別控制

    Args:
        model: 要配置的模型
        dropout_rate: Dropout 率
        layer_indices: 要配置的層索引 (None 表示所有層)
        disable_attention_dropout: 是否禁用注意力 dropout
        disable_ffn_dropout: 是否禁用 FFN dropout
    """
    # 延遲導入避免循環依賴
    from .gated_attention import GatedMultiHeadAttention

    for name, module in model.named_modules():
        if _should_skip_layer(name, layer_indices):
            continue

        # 配置 GatedMultiHeadAttention
        if isinstance(module, GatedMultiHeadAttention):
            _configure_attention_dropout(module, dropout_rate, disable_attention_dropout)

        # 配置 FFN Dropout
        if "ffn" in name.lower() or "feedforward" in name.lower():
            if hasattr(module, "dropout") and isinstance(module.dropout, nn.Dropout):
                if disable_ffn_dropout:
                    module.dropout.p = 0.0
                elif dropout_rate >= 0:
                    module.dropout.p = dropout_rate

        # 配置通用 Dropout 模塊
        if isinstance(module, nn.Dropout):
            _configure_generic_dropout(
                module, name, dropout_rate,
                disable_attention_dropout, disable_ffn_dropout
            )


def _configure_generic_dropout(
    module: nn.Dropout,
    name: str,
    dropout_rate: float,
    disable_attention_dropout: bool,
    disable_ffn_dropout: bool,
) -> None:
    """配置通用 Dropout 模塊"""
    parent_name = name.rsplit(".", 1)[0] if "." in name else ""

    is_attention = "attention" in parent_name.lower() or "attn" in parent_name.lower()
    is_ffn = "ffn" in parent_name.lower() or "feedforward" in parent_name.lower()

    if disable_attention_dropout and is_attention:
        module.p = 0.0
        return
    if disable_ffn_dropout and is_ffn:
        module.p = 0.0
        return

    if dropout_rate >= 0:
        if not (is_attention and disable_attention_dropout):
            if not (is_ffn and disable_ffn_dropout):
                module.p = dropout_rate


def get_dropout_stats(model: nn.Module) -> dict:
    """
    獲取模型的 dropout 統計信息

    Args:
        model: 模型

    Returns:
        統計字典，包含:
        - total_dropout_modules: Dropout 模塊總數
        - attention_dropout_modules: 注意力 dropout 模塊數
        - ffn_dropout_modules: FFN dropout 模塊數
        - dropout_rates: 各模塊的 dropout 率分布
    """
    # 延遲導入避免循環依賴
    from .gated_attention import GatedMultiHeadAttention

    stats = {
        "total_dropout_modules": 0,
        "attention_dropout_modules": 0,
        "ffn_dropout_modules": 0,
        "dropout_rates": {},
    }

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            stats["total_dropout_modules"] += 1
            stats["dropout_rates"][name] = module.p

            if "attention" in name.lower() or "attn" in name.lower():
                stats["attention_dropout_modules"] += 1
            elif "ffn" in name.lower() or "feedforward" in name.lower():
                stats["ffn_dropout_modules"] += 1

        if isinstance(module, GatedMultiHeadAttention):
            stats["dropout_rates"][f"{name}.attention_dropout"] = module.dropout_p

    return stats


__all__ = [
    "configure_selective_dropout",
    "get_dropout_stats",
]
