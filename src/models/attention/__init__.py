"""
PixelHDM-RPEA-DinoV3 注意力模組

包含:
    - GatedMultiHeadAttention: 門控多頭注意力 (GQA + QK Norm + Gating)
    - TokenCompaction: Token 壓縮-注意力-擴展管線
    - gating: 門控機制子模組
    - projections: QKV 投影子模組
    - dropout_utils: Dropout 工具函數
"""

from .gated_attention import GatedMultiHeadAttention, repeat_kv
from .token_compaction import TokenCompaction, TokenCompactionNoResidual
from .factory import (
    create_attention,
    create_attention_from_config,
    create_token_compaction,
    create_token_compaction_from_config,
)
from .dropout_utils import configure_selective_dropout, get_dropout_stats

# Submodule exports for advanced usage
from .gating import GateProjection, IdentityGate, create_gate
from .projections import QKVProjection, OutputProjection


__all__ = [
    # Gated Attention
    "GatedMultiHeadAttention",
    "repeat_kv",
    "create_attention",
    "create_attention_from_config",
    "configure_selective_dropout",
    "get_dropout_stats",
    # Token Compaction
    "TokenCompaction",
    "TokenCompactionNoResidual",
    "create_token_compaction",
    "create_token_compaction_from_config",
    # Gating submodule
    "GateProjection",
    "IdentityGate",
    "create_gate",
    # Projections submodule
    "QKVProjection",
    "OutputProjection",
]
