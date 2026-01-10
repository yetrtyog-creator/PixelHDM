"""
PixelHDM-RPEA-DinoV3 核心層模組

包含:
    - RMSNorm, AdaptiveRMSNorm, QKNorm: 正規化層
    - SwiGLU, AdaptiveSwiGLU: 門控 FFN
    - RoPE1D, RoPE2D, MRoPE: 旋轉位置編碼
    - TokenAdaLN, PixelwiseAdaLN: 自適應層正規化
"""

from .normalization import (
    RMSNorm,
    AdaptiveRMSNorm,
    QKNorm,
    create_norm,
)

from .feedforward import (
    SwiGLU,
    AdaptiveSwiGLU,
    create_ffn,
    create_ffn_from_config,
)

from .rope import (
    RoPE1D,
    RoPE2D,
    MRoPE,
    RotaryPositionEmbedding2D,
    precompute_freqs_cis,
    apply_rotary_emb,
    create_rope_2d,
    create_mrope,
    create_rope_from_config,
    create_image_positions,
    create_image_positions_batched,
)

from .adaln import (
    TokenAdaLN,
    PixelwiseAdaLN,
    AdaLNZero,
    create_token_adaln,
    create_pixelwise_adaln,
    create_pixelwise_adaln_from_config,
)

from .embedding import (
    PatchEmbedding,
    TimeEmbedding,
    PixelUnpatchify,
    PixelPatchify,
    LearnedPositionalEmbedding,
    create_patch_embedding,
    create_patch_embedding_from_config,
    create_time_embedding,
    create_time_embedding_from_config,
    create_pixel_unpatchify,
    create_pixel_unpatchify_from_config,
    create_pixel_patchify,
    create_pixel_patchify_from_config,
)


__all__ = [
    # Normalization
    "RMSNorm",
    "AdaptiveRMSNorm",
    "QKNorm",
    "create_norm",
    # Feedforward
    "SwiGLU",
    "AdaptiveSwiGLU",
    "create_ffn",
    "create_ffn_from_config",
    # RoPE
    "RoPE1D",
    "RoPE2D",
    "MRoPE",
    "RotaryPositionEmbedding2D",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "create_rope_2d",
    "create_mrope",
    "create_rope_from_config",
    "create_image_positions",
    "create_image_positions_batched",
    # AdaLN
    "TokenAdaLN",
    "PixelwiseAdaLN",
    "AdaLNZero",
    "create_token_adaln",
    "create_pixelwise_adaln",
    "create_pixelwise_adaln_from_config",
    # Embedding
    "PatchEmbedding",
    "TimeEmbedding",
    "PixelUnpatchify",
    "PixelPatchify",
    "LearnedPositionalEmbedding",
    "create_patch_embedding",
    "create_patch_embedding_from_config",
    "create_time_embedding",
    "create_time_embedding_from_config",
    "create_pixel_unpatchify",
    "create_pixel_unpatchify_from_config",
    "create_pixel_patchify",
    "create_pixel_patchify_from_config",
]
