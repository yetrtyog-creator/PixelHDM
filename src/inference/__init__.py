"""
PixelHDM-RPEA-DinoV3 推理系統

包含:
    - sampler: 採樣器 (Euler, Heun, DPM++)
    - pipeline: 推理管線 (T2I, I2I)
    - cfg: Classifier-Free Guidance
"""

# Import from new modular sampler package
from .sampler import (
    # Enums and Config
    SamplerMethod,
    SamplerConfig,
    # Base class
    BaseSampler,
    # Concrete samplers
    EulerSampler,
    HeunSampler,
    DPMPPSampler,
    # Unified interface
    UnifiedSampler,
    # Factory functions
    create_sampler,
    create_sampler_from_config,
)

# Import from new modular pipeline package
from .pipeline import (
    # Constants
    MAX_TOKENS,
    DEFAULT_PATCH_SIZE,
    # Validation
    validate_resolution,
    compute_max_resolution,
    # 配置
    GenerationConfig,
    PipelineOutput,
    # 管線
    PixelHDMPipeline,
    PixelHDMPipelineForImg2Img,
    # Mock 編碼器 (測試用)
    MockTextEncoder,
    # 工廠函數
    create_pipeline,
    create_pipeline_from_pretrained,
    create_pipeline_from_config,
)

# Import from new modular cfg package
from .cfg import (
    # 枚舉和配置
    CFGScheduleType,
    CFGConfig,
    # 調度器
    CFGScheduler,
    # CFG 方法
    BaseCFG,
    StandardCFG,
    RescaledCFG,
    CFGWithInterval,
    PerplexityCFG,
    # 包裝器
    CFGWrapper,
    # 便捷函數
    apply_cfg,
    compute_guidance_scale_schedule,
    # 工廠函數
    create_cfg,
    create_cfg_scheduler,
)


__all__ = [
    # === Sampler ===
    # 枚舉
    "SamplerMethod",
    "SamplerConfig",
    # 基類
    "BaseSampler",
    # 具體採樣器
    "EulerSampler",
    "HeunSampler",
    "DPMPPSampler",
    # 統一接口
    "UnifiedSampler",
    # 工廠函數
    "create_sampler",
    "create_sampler_from_config",

    # === Pipeline ===
    # Constants
    "MAX_TOKENS",
    "DEFAULT_PATCH_SIZE",
    # Validation
    "validate_resolution",
    "compute_max_resolution",
    # 配置
    "GenerationConfig",
    "PipelineOutput",
    # 管線
    "PixelHDMPipeline",
    "PixelHDMPipelineForImg2Img",
    # Mock 編碼器 (測試用)
    "MockTextEncoder",
    # 工廠函數
    "create_pipeline",
    "create_pipeline_from_pretrained",
    "create_pipeline_from_config",

    # === CFG ===
    # 枚舉和配置
    "CFGScheduleType",
    "CFGConfig",
    # 調度器
    "CFGScheduler",
    # CFG 方法
    "BaseCFG",
    "StandardCFG",
    "RescaledCFG",
    "CFGWithInterval",
    "PerplexityCFG",
    # 包裝器
    "CFGWrapper",
    # 便捷函數
    "apply_cfg",
    "compute_guidance_scale_schedule",
    # 工廠函數
    "create_cfg",
    "create_cfg_scheduler",
]
