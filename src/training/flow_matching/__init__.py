"""
PixelHDM Flow Matching Package (V-Prediction)

Provides Flow Matching training and sampling based on PixelHDM paper.

Time Convention:
    - t=0: noise
    - t=1: clean image

V-Prediction:
    - 模型直接輸出 velocity v = x - noise
    - 不需要噪聲縮放 (使用標準單位方差噪聲)

Modules:
    - training: PixelHDMFlowMatching
    - pixelhdm_sampler: PixelHDMSampler
    - time_sampling: Logit-Normal time distribution
    - noise: Interpolation

Author: PixelHDM-RPEA-DinoV3 Project
"""

from .training import PixelHDMFlowMatching
from .pixelhdm_sampler import PixelHDMSampler
from .core import (
    create_flow_matching,
    create_flow_matching_from_config,
    create_sampler,
)
from .noise import interpolate
from .time_sampling import (
    sample_logit_normal,
    get_sampling_timesteps,
    TimeSampler,
)

__all__ = [
    # Core classes
    "PixelHDMFlowMatching",
    "PixelHDMSampler",
    # Factory functions
    "create_flow_matching",
    "create_flow_matching_from_config",
    "create_sampler",
    # Noise functions
    "interpolate",
    # Time sampling
    "sample_logit_normal",
    "get_sampling_timesteps",
    "TimeSampler",
]
