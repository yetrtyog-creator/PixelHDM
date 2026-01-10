"""
PixelHDM-RPEA-DinoV3 訓練優化工具模組

包含:
    - ZClip: Z-Score 自適應梯度剪裁
    - EMA: 指數移動平均
    - CPUMemoryCheckpoint: CPU 記憶體快照檢查點
    - create_optimizer: 優化器工廠函數
    - CosineWarmupScheduler: 餘弦退火學習率調度器
"""

from .gradient_clip import ZClip, ZClipStats, clip_grad_norm_with_zclip
from .ema import EMA, EMAModelWrapper
from .checkpoint import CPUMemoryCheckpoint, CheckpointStats
from .optimizers import (
    create_optimizer,
    create_optimizer_from_config,
    get_parameter_groups,
)
from .scheduler import (
    CosineWarmupScheduler,
    LinearWarmupScheduler,
    ConstantWarmupScheduler,
    CosineAnnealingWarmRestartsWithDecay,
    SteppedCosineRestartScheduler,
    create_scheduler,
    create_scheduler_from_config,
)  # Now imports from scheduler/ submodule

__all__ = [
    # 優化器
    "create_optimizer",
    "create_optimizer_from_config",
    "get_parameter_groups",
    # 學習率調度器
    "CosineWarmupScheduler",
    "LinearWarmupScheduler",
    "ConstantWarmupScheduler",
    "CosineAnnealingWarmRestartsWithDecay",
    "SteppedCosineRestartScheduler",
    "create_scheduler",
    "create_scheduler_from_config",
    # 梯度剪裁
    "ZClip",
    "ZClipStats",
    "clip_grad_norm_with_zclip",
    # EMA
    "EMA",
    "EMAModelWrapper",
    # 檢查點
    "CPUMemoryCheckpoint",
    "CheckpointStats",
]
