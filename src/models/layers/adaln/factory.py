"""
Factory Functions for AdaLN Modules
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .token_adaln import TokenAdaLN
from .pixelwise_adaln import PixelwiseAdaLN

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


def create_token_adaln(
    hidden_dim: int = 1024,
    condition_dim: Optional[int] = None,
    num_params: int = 6,
) -> TokenAdaLN:
    """Create Token-Independent AdaLN."""
    return TokenAdaLN(
        hidden_dim=hidden_dim,
        condition_dim=condition_dim,
        num_params=num_params,
    )


def create_pixelwise_adaln(
    hidden_dim: int = 1024,
    pixel_dim: int = 16,
    patch_size: int = 16,
    num_params: int = 6,
) -> PixelwiseAdaLN:
    """Create Pixel-wise AdaLN."""
    return PixelwiseAdaLN(
        config=None,
        hidden_dim=hidden_dim,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
        num_params=num_params,
    )


def create_pixelwise_adaln_from_config(
    config: "PixelHDMConfig",
    num_params: int = 6,
) -> PixelwiseAdaLN:
    """Create Pixel-wise AdaLN from config."""
    return PixelwiseAdaLN(config=config, num_params=num_params)


__all__ = [
    "create_token_adaln",
    "create_pixelwise_adaln",
    "create_pixelwise_adaln_from_config",
]
