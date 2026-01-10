"""
PixelHDM-RPEA-DinoV3 Embedding Modules

Public API (backward compatible):
    - PixelEmbedding: 1×1 Patchify (per-pixel linear embedding)
    - PatchEmbedding: 16×16 Patchify (Image -> Patch Tokens)
    - TimeEmbedding: Timestep -> Time embedding
    - PixelUnpatchify: Patch Tokens -> Pixel Features
    - PixelPatchify: Pixel Features -> Image
    - LearnedPositionalEmbedding: Learnable position embeddings
    - Factory functions: create_*
"""

from .patch import (
    PixelEmbedding,
    PatchEmbedding,
    PixelUnpatchify,
    PixelPatchify,
    create_pixel_embedding,
    create_pixel_embedding_from_config,
    create_patch_embedding,
    create_patch_embedding_from_config,
    create_pixel_unpatchify,
    create_pixel_unpatchify_from_config,
    create_pixel_patchify,
    create_pixel_patchify_from_config,
)
from .time import (
    TimeEmbedding,
    create_time_embedding,
    create_time_embedding_from_config,
)
from .positional import (
    LearnedPositionalEmbedding,
)

__all__ = [
    # Classes
    "PixelEmbedding",
    "PatchEmbedding",
    "TimeEmbedding",
    "PixelUnpatchify",
    "PixelPatchify",
    "LearnedPositionalEmbedding",
    # Factory functions
    "create_pixel_embedding",
    "create_pixel_embedding_from_config",
    "create_patch_embedding",
    "create_patch_embedding_from_config",
    "create_time_embedding",
    "create_time_embedding_from_config",
    "create_pixel_unpatchify",
    "create_pixel_unpatchify_from_config",
    "create_pixel_patchify",
    "create_pixel_patchify_from_config",
]
