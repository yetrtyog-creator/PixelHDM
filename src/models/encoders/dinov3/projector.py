"""
DINOv3 Feature Projector

Projects DiT intermediate features to DINOv3 feature space for REPA Loss.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class DINOv3FeatureProjector(nn.Module):
    """
    DINOv3 Feature Projector

    Projects DiT hidden features to DINOv3 feature space for REPA Loss.

    Args:
        config: PixelHDMConfig
        input_dim: DiT hidden dimension
        output_dim: DINOv3 feature dimension

    Shape:
        - Input: (B, L, D_dit)
        - Output: (B, L, D_dino)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        input_dim: int = 1024,
        output_dim: int = 768,
    ) -> None:
        super().__init__()

        if config is not None:
            input_dim = config.hidden_dim
            output_dim = config.repa_hidden_size

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Two-layer MLP projection
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: DiT intermediate features (B, L, D_dit)

        Returns:
            Projected features (B, L, D_dino)
        """
        return self.projector(x)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


# Alias for backward compatibility
DINOFeatureProjector = DINOv3FeatureProjector


__all__ = [
    "DINOv3FeatureProjector",
    "DINOFeatureProjector",
]
