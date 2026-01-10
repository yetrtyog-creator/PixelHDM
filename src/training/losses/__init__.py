"""
PixelHDM-RPEA-DinoV3 損失函數模組

包含:
    - VLoss: PixelHDM 風格的 V-Loss
    - FrequencyLoss: DeCo 風格的頻率感知損失
    - REPALoss: DINOv3 特徵對齊損失
    - CombinedLoss: Triple Loss 整合
"""

from .vloss import (
    VLoss,
    VLossWithVelocity,
    create_vloss,
    create_vloss_from_config,
    create_vloss_with_velocity,
)

from .freq_loss import (
    FreqLossConfig,
    FrequencyLoss,
    BlockDCT2D,
    rgb_to_ycbcr,
    create_freq_loss,
    create_freq_loss_from_config,
)

from .repa_loss import (
    REPALoss,
    REPALossWithProjector,
    create_repa_loss,
    create_repa_loss_from_config,
)

from .combined_loss import (
    CombinedLoss,
    CombinedLossSimple,
    create_combined_loss,
    create_combined_loss_from_config,
    create_combined_loss_simple,
)


__all__ = [
    # V-Loss
    "VLoss",
    "VLossWithVelocity",
    "create_vloss",
    "create_vloss_from_config",
    "create_vloss_with_velocity",
    # Frequency Loss
    "FreqLossConfig",
    "FrequencyLoss",
    "BlockDCT2D",
    "rgb_to_ycbcr",
    "create_freq_loss",
    "create_freq_loss_from_config",
    # REPA Loss
    "REPALoss",
    "REPALossWithProjector",
    "create_repa_loss",
    "create_repa_loss_from_config",
    # Combined Loss
    "CombinedLoss",
    "CombinedLossSimple",
    "create_combined_loss",
    "create_combined_loss_from_config",
    "create_combined_loss_simple",
]
