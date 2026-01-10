"""
PixelHDM-RPEA-DinoV3 Classifier-Free Guidance

CFG tools and enhancements:
    - Standard CFG
    - CFG Rescale (prevents oversaturation)
    - Dynamic CFG (time-based)
    - Multi-condition CFG

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from .base import (
    CFGScheduleType,
    CFGConfig,
    CFGScheduler,
    BaseCFG,
)

from .standard import (
    StandardCFG,
    PerplexityCFG,
)

from .rescaled import (
    RescaledCFG,
)

from .interval import (
    CFGWithInterval,
)

from .wrapper import (
    CFGWrapper,
)

from .utils import (
    apply_cfg,
    compute_guidance_scale_schedule,
    create_cfg,
    create_cfg_scheduler,
)


__all__ = [
    # Enums and Config
    "CFGScheduleType",
    "CFGConfig",
    # Scheduler
    "CFGScheduler",
    # Base class
    "BaseCFG",
    # CFG methods
    "StandardCFG",
    "RescaledCFG",
    "CFGWithInterval",
    "PerplexityCFG",
    # Wrapper
    "CFGWrapper",
    # Utility functions
    "apply_cfg",
    "compute_guidance_scale_schedule",
    # Factory functions
    "create_cfg",
    "create_cfg_scheduler",
]
