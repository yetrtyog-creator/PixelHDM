"""
PixelHDM-RPEA-DinoV3 Sampler Package

PixelHDM Flow Matching samplers for image generation from noise.

Core Design:
    - Time direction: t=0 noise, t=1 clean image (PixelHDM style)
    - Sampling direction: t from 0 -> 1
    - ODE solvers: Euler, Heun, DPM++ methods
    - CFG support: Classifier-Free Guidance

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from .base import (
    SamplerMethod,
    SamplerConfig,
    BaseSampler,
)
from .euler import EulerSampler
from .heun import HeunSampler
from .dpm import DPMPPSampler
from .unified import (
    UnifiedSampler,
    create_sampler,
    create_sampler_from_config,
)
from .timesteps import (
    get_timesteps,
    get_lambda,
)
from .full_heun import FullHeunCFGMixin

__all__ = [
    # Enums and Config
    "SamplerMethod",
    "SamplerConfig",
    # Base class
    "BaseSampler",
    # Concrete samplers
    "EulerSampler",
    "HeunSampler",
    "DPMPPSampler",
    # Unified interface
    "UnifiedSampler",
    # Factory functions
    "create_sampler",
    "create_sampler_from_config",
    # Utilities
    "get_timesteps",
    "get_lambda",
    # Mixins
    "FullHeunCFGMixin",
]
