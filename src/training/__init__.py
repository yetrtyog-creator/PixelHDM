"""
PixelHDM-RPEA-DinoV3 訓練系統

包含:
    - losses: 損失函數 (V-Loss, Freq Loss, REPA Loss, Combined Loss)
    - optimization: 優化器 (EMA, Checkpoint, Gradient Clip)
    - flow_matching: PixelHDM Flow Matching
    - trainer: 訓練器
    - dataset: 數據集
    - bucket: 分桶系統 (多分辨率訓練、RAM 優化)
"""

from . import losses
from . import optimization
from . import bucket

from .losses import (
    # V-Loss
    VLoss,
    VLossWithVelocity,
    create_vloss,
    create_vloss_from_config,
    # Frequency Loss
    FrequencyLoss,
    create_freq_loss,
    create_freq_loss_from_config,
    # REPA Loss
    REPALoss,
    create_repa_loss,
    create_repa_loss_from_config,
    # Combined Loss
    CombinedLoss,
    CombinedLossSimple,
    create_combined_loss,
    create_combined_loss_from_config,
)

from .flow_matching import (
    PixelHDMFlowMatching,
    PixelHDMSampler,
    create_flow_matching,
    create_flow_matching_from_config,
    create_sampler,
)

from .optimization import (
    # 優化器
    create_optimizer,
    create_optimizer_from_config,
    get_parameter_groups,
    # 梯度剪裁
    ZClip,
    ZClipStats,
    clip_grad_norm_with_zclip,
    # EMA
    EMA,
    EMAModelWrapper,
    # 檢查點
    CPUMemoryCheckpoint,
    CheckpointStats,
)

from .trainer import (
    Trainer,
    TrainerState,
    TrainMetrics,
    create_trainer,
    CheckpointManager,
    StepExecutor,
    MetricsLogger,
    create_lr_scheduler,
    apply_warmup_lr,
)

from .dataset import (
    ImageTextDataset,
    collate_fn,
    create_dataloader,
    create_dataloader_from_config,
    # Bucket-based
    BucketImageTextDataset,
    bucket_collate_fn,
    create_bucket_dataloader,
    create_dataloader_from_config_v2,
)

from .bucket import (
    BucketConfig,
    AspectRatioBucket,
    BucketSampler,
    SequentialBucketSampler,
    BufferedShuffleBucketSampler,
    AdaptiveBufferManager,
    scan_images_for_buckets,
    create_bucket_manager,
    create_bucket_manager_from_config,
)


__all__ = [
    # 子模組
    "losses",
    "optimization",
    "bucket",
    # V-Loss
    "VLoss",
    "VLossWithVelocity",
    "create_vloss",
    "create_vloss_from_config",
    # Frequency Loss
    "FrequencyLoss",
    "create_freq_loss",
    "create_freq_loss_from_config",
    # REPA Loss
    "REPALoss",
    "create_repa_loss",
    "create_repa_loss_from_config",
    # Combined Loss
    "CombinedLoss",
    "CombinedLossSimple",
    "create_combined_loss",
    "create_combined_loss_from_config",
    # Flow Matching
    "PixelHDMFlowMatching",
    "PixelHDMSampler",
    "create_flow_matching",
    "create_flow_matching_from_config",
    "create_sampler",
    # 優化器
    "create_optimizer",
    "create_optimizer_from_config",
    "get_parameter_groups",
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
    # Trainer
    "Trainer",
    "TrainerState",
    "TrainMetrics",
    "create_trainer",
    "CheckpointManager",
    "StepExecutor",
    "MetricsLogger",
    "create_lr_scheduler",
    "apply_warmup_lr",
    # Dataset
    "ImageTextDataset",
    "collate_fn",
    "create_dataloader",
    "create_dataloader_from_config",
    # Bucket-based Dataset
    "BucketImageTextDataset",
    "bucket_collate_fn",
    "create_bucket_dataloader",
    "create_dataloader_from_config_v2",
    # Bucket System
    "BucketConfig",
    "AspectRatioBucket",
    "BucketSampler",
    "SequentialBucketSampler",
    "BufferedShuffleBucketSampler",
    "AdaptiveBufferManager",
    "scan_images_for_buckets",
    "create_bucket_manager",
    "create_bucket_manager_from_config",
]
