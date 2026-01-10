"""
PixelHDM-RPEA-DinoV3 Inference Pipeline

Modular pipeline system for text-to-image generation.

Modules:
    - validation: Input validation (resolution, parameters)
    - preprocessing: Text encoding, latent preparation
    - postprocessing: Output format conversion
    - generation: Core sampling logic
    - core: Main pipeline class
    - img2img: Image-to-image pipeline
    - mock_encoder: Testing utilities
    - factory: Pipeline creation utilities

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

# === Validation ===
from .validation import (
    # Constants (for backward compatibility)
    DEFAULT_PATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    # Classes
    InputValidator,
    # Functions
    validate_resolution,
    compute_max_resolution,
)

# Backward compatibility aliases
MAX_TOKENS = DEFAULT_MAX_TOKENS

# === Preprocessing ===
from .preprocessing import (
    GenerationInputs,
    Preprocessor,
)

# === Postprocessing ===
from .postprocessing import (
    PipelineOutput,
    Postprocessor,
)

# === Generation ===
from .generation import (
    GenerationConfig,
    Generator,
)

# === Core Pipeline ===
from .core import PixelHDMPipeline

# === Image-to-Image ===
from .img2img import PixelHDMPipelineForImg2Img

# === Mock Encoder ===
from .mock_encoder import MockTextEncoder

# === Factory Functions ===
from .factory import (
    create_pipeline,
    create_pipeline_from_pretrained,
    create_pipeline_from_config,
)


__all__ = [
    # Constants
    "MAX_TOKENS",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_MAX_TOKENS",
    # Validation
    "InputValidator",
    "validate_resolution",
    "compute_max_resolution",
    # Preprocessing
    "GenerationInputs",
    "Preprocessor",
    # Postprocessing
    "PipelineOutput",
    "Postprocessor",
    # Generation
    "GenerationConfig",
    "Generator",
    # Pipelines
    "PixelHDMPipeline",
    "PixelHDMPipelineForImg2Img",
    # Mock Encoder
    "MockTextEncoder",
    # Factory Functions
    "create_pipeline",
    "create_pipeline_from_pretrained",
    "create_pipeline_from_config",
]
