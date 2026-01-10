"""
RoPE (Rotary Position Embedding) Module

Supports:
    - RoPE1D: Standard 1D RoPE for text sequences
    - RoPE2D: 2D RoPE for image patches (p=16)
    - MRoPE: Multi-axis RoPE for joint text-image (16-24-24 split)

Public API:
    - RoPE1D, RoPE2D, MRoPE: Core classes
    - RotaryPositionEmbedding2D: Alias for RoPE2D (backward compatibility)
    - precompute_freqs_cis, apply_rotary_emb: Utility functions
    - create_rope_2d, create_mrope, create_rope_from_config: Factory functions
    - create_image_positions, create_image_positions_batched: Position utilities

Author: PixelHDM-RPEA-DinoV3 Project
"""

from .utils import (
    precompute_freqs_cis,
    apply_rotary_emb,
    create_image_positions,
    create_image_positions_batched,
    create_position_ids,
    create_position_ids_batched,
    create_image_only_position_ids,
    create_image_only_position_ids_batched,
)

from .rope_1d import RoPE1D
from .rope_2d import RoPE2D, RotaryPositionEmbedding2D
from .mrope import MRoPE
from .factory import create_rope_2d, create_mrope, create_rope_from_config


__all__ = [
    # Core classes
    "RoPE1D",
    "RoPE2D",
    "MRoPE",
    "RotaryPositionEmbedding2D",
    # Utility functions
    "precompute_freqs_cis",
    "apply_rotary_emb",
    # Factory functions
    "create_rope_2d",
    "create_mrope",
    "create_rope_from_config",
    # Position utilities (legacy)
    "create_image_positions",
    "create_image_positions_batched",
    # Position IDs (Lumina2 style)
    "create_position_ids",
    "create_position_ids_batched",
    "create_image_only_position_ids",
    "create_image_only_position_ids_batched",
]
