"""
Tests for PixelHDM Inference Pipeline.

This module tests:
- Pipeline initialization and configuration
- Resolution validation and token limits
- Sampling with various configurations
- Pipeline utility methods (torch.compile, precision, mock encoder)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typing import Optional, Tuple
from unittest.mock import Mock, MagicMock, patch

from src.inference.pipeline import (
    PixelHDMPipeline,
    MockTextEncoder,
    validate_resolution,
    compute_max_resolution,
    MAX_TOKENS,
    DEFAULT_PATCH_SIZE,
    PipelineOutput,
    GenerationConfig,
)
from src.config import PixelHDMConfig


# ============================================================================
# Fixtures
# ============================================================================

class DummyPixelHDMModel(nn.Module):
    """
    Minimal dummy model mimicking PixelHDM for pipeline testing.

    Accepts:
        x: (B, H, W, 3) or (B, 3, H, W) input
        t: (B,) timesteps
        text_embed: (B, L, D) text embeddings

    Returns:
        Prediction of same shape as input x
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = PixelHDMConfig.for_testing()
        # Simple identity-like operation
        self.linear = nn.Linear(3, 3, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass returning prediction of same shape."""
        # Handle both (B, H, W, 3) and (B, 3, H, W) formats
        if x.dim() == 4 and x.shape[-1] == 3:
            # (B, H, W, 3) format
            out = self.linear(x)
        elif x.dim() == 4 and x.shape[1] == 3:
            # (B, 3, H, W) format - permute, process, permute back
            x_permuted = x.permute(0, 2, 3, 1)  # (B, H, W, 3)
            out_permuted = self.linear(x_permuted)
            out = out_permuted.permute(0, 3, 1, 2)  # (B, 3, H, W)
        else:
            out = x
        return out


@pytest.fixture
def dummy_model():
    """Create a dummy PixelHDM model for testing."""
    return DummyPixelHDMModel()


@pytest.fixture
def pipeline_with_mock(dummy_model):
    """Create a pipeline with mock text encoder."""
    pipeline = PixelHDMPipeline(
        model=dummy_model,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    pipeline.use_mock_text_encoder(hidden_size=256, max_length=64)
    return pipeline


@pytest.fixture
def config_for_testing():
    """Create minimal config for testing."""
    return PixelHDMConfig.for_testing()


# ============================================================================
# Test: Resolution Validation
# ============================================================================

class TestValidateResolution:
    """Tests for validate_resolution function."""

    def test_validate_resolution_valid_512(self):
        """Test validation passes for 512x512 with default patch_size=16."""
        # 512x512, patch_size=16 -> 32x32 = 1024 tokens (exactly at limit)
        validate_resolution(512, 512, patch_size=16, max_tokens=1024)
        # Should not raise any exception

    def test_validate_resolution_valid_256(self):
        """Test validation passes for 256x256."""
        # 256x256, patch_size=16 -> 16x16 = 256 tokens
        validate_resolution(256, 256, patch_size=16, max_tokens=1024)
        # Should not raise any exception

    def test_validate_resolution_valid_rectangular(self):
        """Test validation passes for rectangular resolutions."""
        # 512x256, patch_size=16 -> 32x16 = 512 tokens
        validate_resolution(512, 256, patch_size=16, max_tokens=1024)
        # Should not raise any exception

    def test_validate_resolution_not_divisible_height_raises(self):
        """Test that non-divisible height raises ValueError."""
        with pytest.raises(ValueError, match="patch_size"):
            validate_resolution(500, 512, patch_size=16, max_tokens=1024)

    def test_validate_resolution_not_divisible_width_raises(self):
        """Test that non-divisible width raises ValueError."""
        with pytest.raises(ValueError, match="patch_size"):
            validate_resolution(512, 500, patch_size=16, max_tokens=1024)

    def test_validate_resolution_exceeds_tokens_raises(self):
        """Test that exceeding token limit raises ValueError."""
        # 1024x1024, patch_size=16 -> 64x64 = 4096 tokens (exceeds 1024)
        with pytest.raises(ValueError, match="超過"):
            validate_resolution(1024, 1024, patch_size=16, max_tokens=1024)

    def test_validate_resolution_custom_patch_size(self):
        """Test validation with custom patch_size."""
        # 512x512, patch_size=32 -> 16x16 = 256 tokens
        validate_resolution(512, 512, patch_size=32, max_tokens=1024)
        # Should not raise any exception


class TestComputeMaxResolution:
    """Tests for compute_max_resolution function."""

    def test_compute_max_resolution_default(self):
        """Test default max resolution calculation."""
        # max_tokens=1024, patch_size=16 -> sqrt(1024)*16 = 32*16 = 512
        max_res = compute_max_resolution(patch_size=16, max_tokens=1024)
        assert max_res == 512

    def test_compute_max_resolution_larger_tokens(self):
        """Test max resolution with larger token limit."""
        # max_tokens=4096, patch_size=16 -> sqrt(4096)*16 = 64*16 = 1024
        max_res = compute_max_resolution(patch_size=16, max_tokens=4096)
        assert max_res == 1024

    def test_compute_max_resolution_different_patch_size(self):
        """Test max resolution with different patch size."""
        # max_tokens=1024, patch_size=8 -> sqrt(1024)*8 = 32*8 = 256
        max_res = compute_max_resolution(patch_size=8, max_tokens=1024)
        assert max_res == 256

    def test_compute_max_resolution_consistency(self):
        """Test that max resolution passes validation."""
        max_res = compute_max_resolution(patch_size=16, max_tokens=1024)
        # This should not raise
        validate_resolution(max_res, max_res, patch_size=16, max_tokens=1024)


# ============================================================================
# Test: Pipeline Initialization
# ============================================================================

class TestPipelineInit:
    """Tests for PixelHDMPipeline initialization."""

    def test_pipeline_init_basic(self, dummy_model):
        """Test basic pipeline initialization."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert pipeline.model is dummy_model
        assert pipeline.device == torch.device("cpu")
        assert pipeline.dtype == torch.float32

    def test_pipeline_init_with_mock_encoder(self, dummy_model):
        """Test pipeline initialization with mock text encoder."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        pipeline.use_mock_text_encoder(hidden_size=256, max_length=64)

        assert pipeline._text_encoder is not None
        assert isinstance(pipeline._text_encoder, MockTextEncoder)
        assert pipeline._text_encoder.hidden_size == 256
        assert pipeline._text_encoder.max_length == 64

    def test_pipeline_device_placement(self, dummy_model):
        """Test pipeline device placement."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
        )

        assert pipeline.device == torch.device("cpu")

        # Test to() method
        pipeline = pipeline.to(torch.device("cpu"))
        assert pipeline.device == torch.device("cpu")

    def test_pipeline_config_from_model(self, dummy_model):
        """Test that pipeline gets config from model."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
        )

        # Config should be retrieved from model
        assert pipeline.config is not None


# ============================================================================
# Test: Pipeline Sampling
# ============================================================================

class TestPipelineSampling:
    """Tests for PixelHDMPipeline sampling/generation."""

    def test_pipeline_prepare_latents(self, pipeline_with_mock):
        """Test latent preparation."""
        latents = pipeline_with_mock.prepare_latents(
            batch_size=2,
            height=256,
            width=256,
        )

        assert latents.shape == (2, 256, 256, 3)
        assert latents.device == torch.device("cpu")

    def test_pipeline_prepare_latents_with_seed(self, pipeline_with_mock):
        """Test that latents are deterministic with seed."""
        generator1 = torch.Generator().manual_seed(42)
        generator2 = torch.Generator().manual_seed(42)

        latents1 = pipeline_with_mock.prepare_latents(
            batch_size=1,
            height=256,
            width=256,
            generator=generator1,
        )
        latents2 = pipeline_with_mock.prepare_latents(
            batch_size=1,
            height=256,
            width=256,
            generator=generator2,
        )

        assert torch.allclose(latents1, latents2)

    def test_pipeline_prepare_latents_resolution_validation(self, pipeline_with_mock):
        """Test that prepare_latents validates resolution."""
        # Should raise for non-divisible resolution
        with pytest.raises(ValueError):
            pipeline_with_mock.prepare_latents(
                batch_size=1,
                height=255,  # Not divisible by 16
                width=256,
            )

    def test_pipeline_postprocess_pil(self, pipeline_with_mock):
        """Test postprocessing to PIL format."""
        # Simulate generated images in [-1, 1]
        images = torch.randn(2, 256, 256, 3).clamp(-1, 1)

        pil_images = pipeline_with_mock.postprocess(images, output_type="pil")

        assert len(pil_images) == 2
        # Check first image dimensions
        assert pil_images[0].size == (256, 256)

    def test_pipeline_postprocess_tensor(self, pipeline_with_mock):
        """Test postprocessing to tensor format."""
        images = torch.randn(2, 256, 256, 3)

        tensor_out = pipeline_with_mock.postprocess(images, output_type="tensor")

        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.shape == (2, 256, 256, 3)

    def test_pipeline_postprocess_numpy(self, pipeline_with_mock):
        """Test postprocessing to numpy format."""
        import numpy as np

        images = torch.randn(2, 256, 256, 3)

        numpy_out = pipeline_with_mock.postprocess(images, output_type="numpy")

        assert isinstance(numpy_out, np.ndarray)
        assert numpy_out.shape == (2, 256, 256, 3)

    def test_pipeline_get_nfe_stats(self, pipeline_with_mock):
        """Test NFE statistics retrieval."""
        stats = pipeline_with_mock.get_nfe_stats(
            num_steps=50,
            use_cfg=True,
            sampler_method="heun",
        )

        assert "nfe" in stats
        assert "nfe_per_step" in stats
        assert "sampler_method" in stats
        assert "use_cfg" in stats
        assert "num_steps" in stats

        assert stats["nfe"] > 0
        assert stats["num_steps"] == 50
        assert stats["sampler_method"] == "heun"
        assert stats["use_cfg"] == True

    def test_pipeline_get_nfe_stats_without_cfg(self, pipeline_with_mock):
        """Test NFE statistics without CFG."""
        stats_with_cfg = pipeline_with_mock.get_nfe_stats(
            num_steps=50, use_cfg=True, sampler_method="euler"
        )
        stats_no_cfg = pipeline_with_mock.get_nfe_stats(
            num_steps=50, use_cfg=False, sampler_method="euler"
        )

        # With CFG should have more NFE
        assert stats_with_cfg["nfe"] >= stats_no_cfg["nfe"]


# ============================================================================
# Test: Pipeline Methods
# ============================================================================

class TestPipelineMethods:
    """Tests for PixelHDMPipeline utility methods."""

    def test_enable_torch_compile(self, pipeline_with_mock):
        """Test torch.compile enabling."""
        # This should not raise, even if torch.compile is not available
        result = pipeline_with_mock.enable_torch_compile(
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True,
        )

        # Should return self for chaining
        assert result is pipeline_with_mock

    def test_set_text_encoder_precision(self, pipeline_with_mock):
        """Test text encoder precision setting."""
        result = pipeline_with_mock.set_text_encoder_precision(
            dtype=torch.float16,
        )

        # Should return self for chaining
        assert result is pipeline_with_mock

        # Mock encoder should be updated
        assert pipeline_with_mock._text_encoder.dtype == torch.float16

    def test_use_mock_text_encoder_clears_cache(self, pipeline_with_mock):
        """Test that using mock encoder clears null text cache."""
        # Pre-populate cache
        pipeline_with_mock._null_text_embed = torch.randn(1, 32, 256)
        pipeline_with_mock._null_text_mask = torch.ones(1, 32, dtype=torch.bool)

        # Use mock encoder
        pipeline_with_mock.use_mock_text_encoder(hidden_size=128)

        # Cache should be cleared
        assert pipeline_with_mock._null_text_embed is None
        assert pipeline_with_mock._null_text_mask is None

    def test_get_sampler(self, pipeline_with_mock):
        """Test sampler retrieval."""
        sampler = pipeline_with_mock.get_sampler(method="euler", num_steps=30)

        assert sampler is not None
        assert sampler.num_steps == 30

    def test_get_sampler_caching(self, pipeline_with_mock):
        """Test that sampler is cached for same method."""
        sampler1 = pipeline_with_mock.get_sampler(method="heun", num_steps=50)
        sampler2 = pipeline_with_mock.get_sampler(method="heun", num_steps=50)

        # Should be the same instance
        assert sampler1 is sampler2

    def test_pipeline_to_device(self, pipeline_with_mock):
        """Test moving pipeline to device."""
        # Move to CPU (already there, but should work)
        pipeline = pipeline_with_mock.to(torch.device("cpu"))

        assert pipeline.device == torch.device("cpu")
        assert pipeline is pipeline_with_mock  # Should return self


# ============================================================================
# Test: MockTextEncoder
# ============================================================================

class TestMockTextEncoder:
    """Tests for MockTextEncoder."""

    def test_mock_encoder_init(self):
        """Test mock encoder initialization."""
        encoder = MockTextEncoder(
            hidden_size=512,
            max_length=128,
        )

        assert encoder.hidden_size == 512
        assert encoder.max_length == 128

    def test_mock_encoder_forward(self):
        """Test mock encoder forward pass."""
        encoder = MockTextEncoder(hidden_size=256, max_length=64)

        texts = ["Hello world", "Test prompt"]
        embeddings, mask, pooled = encoder(texts)

        assert embeddings.shape[0] == 2  # Batch size
        assert embeddings.shape[2] == 256  # Hidden size
        assert mask.shape[0] == 2  # Batch size
        assert mask.dtype == torch.bool
        assert pooled.shape == (2, 256)  # Pooled output

    def test_mock_encoder_deterministic(self):
        """Test that mock encoder is deterministic."""
        encoder = MockTextEncoder(hidden_size=256)

        texts = ["Test"]
        emb1, _, _ = encoder(texts)
        emb2, _, _ = encoder(texts)

        # Should produce same output due to fixed seed
        assert torch.allclose(emb1, emb2)

    def test_mock_encoder_attention_mask(self):
        """Test attention mask generation."""
        encoder = MockTextEncoder(hidden_size=256, max_length=64)

        # Different length texts
        texts = ["Short", "This is a much longer text for testing"]
        embeddings, mask, pooled = encoder(texts)

        # Mask should indicate valid positions
        assert mask.any()  # At least some True values

    def test_mock_encoder_to_device(self):
        """Test moving mock encoder to device."""
        encoder = MockTextEncoder(hidden_size=256)

        encoder = encoder.to(torch.device("cpu"))
        assert encoder.device == torch.device("cpu")

        encoder = encoder.to(dtype=torch.float16)
        assert encoder.dtype == torch.float16


# ============================================================================
# Test: GenerationConfig
# ============================================================================

class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_generation_config_defaults(self):
        """Test default generation config values."""
        config = GenerationConfig()

        assert config.height == 512
        assert config.width == 512
        assert config.num_steps == 50
        assert config.sampler_method == "heun"
        assert config.guidance_scale == 7.5
        assert config.batch_size == 1
        assert config.output_type == "pil"

    def test_generation_config_custom(self):
        """Test custom generation config."""
        config = GenerationConfig(
            height=256,
            width=256,
            num_steps=30,
            guidance_scale=5.0,
            seed=42,
        )

        assert config.height == 256
        assert config.num_steps == 30
        assert config.guidance_scale == 5.0
        assert config.seed == 42


# ============================================================================
# Test: PipelineOutput
# ============================================================================

class TestPipelineOutput:
    """Tests for PipelineOutput dataclass."""

    def test_pipeline_output_creation(self):
        """Test creating pipeline output."""
        images = [Mock()]
        latents = torch.randn(1, 64, 64, 3)

        output = PipelineOutput(
            images=images,
            latents=latents,
            metadata={"prompt": "test"},
        )

        assert output.images == images
        assert output.latents is latents
        assert output.metadata["prompt"] == "test"

    def test_pipeline_output_optional_fields(self):
        """Test optional fields default to None."""
        output = PipelineOutput(images=[])

        assert output.latents is None
        assert output.intermediates is None
        assert output.metadata == {}


# ============================================================================
# Test: Text Encoder Property
# ============================================================================

class TestTextEncoderProperty:
    """Tests for text_encoder property."""

    def test_text_encoder_from_pipeline(self, dummy_model):
        """Test text_encoder returns _text_encoder when set."""
        mock_encoder = MockTextEncoder(hidden_size=256)
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            text_encoder=mock_encoder,
            device=torch.device("cpu"),
        )

        assert pipeline.text_encoder is mock_encoder

    def test_text_encoder_from_model(self, dummy_model):
        """Test text_encoder returns model's encoder when available."""
        # Add text_encoder to model
        dummy_model.text_encoder = MockTextEncoder(hidden_size=256)
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
        )

        assert pipeline.text_encoder is dummy_model.text_encoder

    def test_text_encoder_none_when_unavailable(self, dummy_model):
        """Test text_encoder returns None when no encoder available."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
        )

        # No _text_encoder set and model doesn't have text_encoder
        assert pipeline.text_encoder is None


# ============================================================================
# Test: Encode Prompt
# ============================================================================

class TestEncodePrompt:
    """Tests for encode_prompt method."""

    def test_encode_prompt_single_string(self, pipeline_with_mock):
        """Test encoding a single string prompt."""
        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = pipeline_with_mock.encode_prompt(
            prompt="test prompt"
        )

        assert text_embed.shape[0] == 1  # Batch size
        assert text_mask.shape[0] == 1
        assert pooled is not None
        assert null_embed is not None
        assert null_mask is not None

    def test_encode_prompt_list(self, pipeline_with_mock):
        """Test encoding a list of prompts."""
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = pipeline_with_mock.encode_prompt(
            prompt=prompts
        )

        assert text_embed.shape[0] == 3  # Batch size
        assert text_mask.shape[0] == 3
        assert pooled.shape[0] == 3

    def test_encode_prompt_with_negative_prompt(self, pipeline_with_mock):
        """Test encoding with negative prompt."""
        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = pipeline_with_mock.encode_prompt(
            prompt="test prompt",
            negative_prompt="negative prompt",
        )

        assert text_embed is not None
        assert null_embed is not None
        assert null_embed.shape[0] == 1

    def test_encode_prompt_with_num_images_per_prompt(self, pipeline_with_mock):
        """Test encoding with num_images_per_prompt > 1."""
        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = pipeline_with_mock.encode_prompt(
            prompt="test",
            num_images_per_prompt=3,
        )

        # Embeddings should be repeated
        assert text_embed.shape[0] == 3
        assert pooled.shape[0] == 3
        assert null_embed.shape[0] == 3

    def test_encode_prompt_no_encoder_raises(self, dummy_model):
        """Test encode_prompt raises when no encoder available."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="文本編碼器"):
            pipeline.encode_prompt("test")


# ============================================================================
# Test: Get Null Text Embed
# ============================================================================

class TestGetNullTextEmbed:
    """Tests for _get_null_text_embed method."""

    def test_get_null_text_embed_caches(self, pipeline_with_mock):
        """Test that null text embed is cached."""
        # First call - should generate
        embed1, mask1, pooled1 = pipeline_with_mock._get_null_text_embed(batch_size=2)

        # Second call - should use cache
        embed2, mask2, pooled2 = pipeline_with_mock._get_null_text_embed(batch_size=2)

        # Should be different tensors but from same source
        assert embed1.shape == embed2.shape
        assert mask1.shape == mask2.shape

    def test_get_null_text_embed_expands_batch(self, pipeline_with_mock):
        """Test that null embed expands to batch size."""
        embed, mask, pooled = pipeline_with_mock._get_null_text_embed(batch_size=4)

        assert embed.shape[0] == 4
        assert mask.shape[0] == 4
        assert pooled.shape[0] == 4

    def test_get_null_text_embed_no_encoder_raises(self, dummy_model):
        """Test _get_null_text_embed raises when no encoder."""
        pipeline = PixelHDMPipeline(
            model=dummy_model,
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="無法獲取"):
            pipeline._get_null_text_embed(batch_size=1)


# ============================================================================
# Test: Full Pipeline Call
# ============================================================================

class TestPipelineCall:
    """Tests for pipeline __call__ method."""

    def test_pipeline_call_basic(self, pipeline_with_mock):
        """Test basic pipeline call."""
        output = pipeline_with_mock(
            prompt="test prompt",
            height=256,
            width=256,
            num_steps=2,  # Minimal steps for fast testing
            guidance_scale=1.0,
        )

        assert isinstance(output, PipelineOutput)
        assert len(output.images) == 1
        assert output.metadata["prompt"] == ["test prompt"]
        assert output.metadata["height"] == 256
        assert output.metadata["width"] == 256

    def test_pipeline_call_with_seed(self, pipeline_with_mock):
        """Test pipeline call with seed for reproducibility."""
        output1 = pipeline_with_mock(
            prompt="test",
            height=256,
            width=256,
            num_steps=2,
            seed=42,
        )
        output2 = pipeline_with_mock(
            prompt="test",
            height=256,
            width=256,
            num_steps=2,
            seed=42,
        )

        # Latents should be the same with same seed
        assert torch.allclose(output1.latents, output2.latents)

    def test_pipeline_call_multiple_prompts(self, pipeline_with_mock):
        """Test pipeline call with multiple prompts."""
        output = pipeline_with_mock(
            prompt=["prompt 1", "prompt 2"],
            height=256,
            width=256,
            num_steps=2,
        )

        assert len(output.images) == 2

    def test_pipeline_call_tensor_output(self, pipeline_with_mock):
        """Test pipeline call with tensor output."""
        output = pipeline_with_mock(
            prompt="test",
            height=256,
            width=256,
            num_steps=2,
            output_type="tensor",
        )

        assert isinstance(output.images, torch.Tensor)
        assert output.images.shape == (1, 256, 256, 3)

    def test_pipeline_call_numpy_output(self, pipeline_with_mock):
        """Test pipeline call with numpy output."""
        import numpy as np

        output = pipeline_with_mock(
            prompt="test",
            height=256,
            width=256,
            num_steps=2,
            output_type="numpy",
        )

        assert isinstance(output.images, np.ndarray)
        assert output.images.shape == (1, 256, 256, 3)

    def test_pipeline_call_with_intermediates(self, pipeline_with_mock):
        """Test pipeline call with intermediate results."""
        output = pipeline_with_mock(
            prompt="test",
            height=256,
            width=256,
            num_steps=4,
            return_intermediates=True,
        )

        assert output.intermediates is not None
        assert len(output.intermediates) > 0

    def test_pipeline_call_different_sampler(self, pipeline_with_mock):
        """Test pipeline call with different sampler methods."""
        for method in ["euler", "heun"]:
            output = pipeline_with_mock(
                prompt="test",
                height=256,
                width=256,
                num_steps=2,
                sampler_method=method,
            )

            assert output.metadata["sampler_method"] == method

    def test_pipeline_call_num_images_per_prompt(self, pipeline_with_mock):
        """Test pipeline call with multiple images per prompt."""
        output = pipeline_with_mock(
            prompt="test",
            height=256,
            width=256,
            num_steps=2,
            num_images_per_prompt=2,
        )

        assert len(output.images) == 2


# ============================================================================
# Test: Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Tests for pipeline utility methods."""

    def test_enable_model_cpu_offload(self, pipeline_with_mock):
        """Test enable_model_cpu_offload returns self."""
        result = pipeline_with_mock.enable_model_cpu_offload()

        assert result is pipeline_with_mock

    def test_enable_attention_slicing(self, pipeline_with_mock):
        """Test enable_attention_slicing returns self."""
        result = pipeline_with_mock.enable_attention_slicing()

        assert result is pipeline_with_mock

        # With slice_size
        result = pipeline_with_mock.enable_attention_slicing(slice_size=4)
        assert result is pipeline_with_mock


# ============================================================================
# Test: PixelHDMPipelineForImg2Img
# ============================================================================

class TestPixelHDMPipelineForImg2Img:
    """Tests for PixelHDMPipelineForImg2Img."""

    @pytest.fixture
    def img2img_pipeline(self, dummy_model):
        """Create img2img pipeline with mock encoder."""
        from src.inference.pipeline import PixelHDMPipelineForImg2Img

        pipeline = PixelHDMPipelineForImg2Img(
            model=dummy_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        pipeline.use_mock_text_encoder(hidden_size=256)
        return pipeline

    def test_img2img_pipeline_init(self, img2img_pipeline):
        """Test img2img pipeline initialization."""
        assert img2img_pipeline is not None

    def test_img2img_preprocess_image(self, img2img_pipeline):
        """Test image preprocessing."""
        from PIL import Image

        # Create test image
        pil_image = Image.new("RGB", (260, 260), color="red")

        tensor = img2img_pipeline._preprocess_image(pil_image)

        # Should be resized to multiple of patch_size (16)
        assert tensor.shape[0] == 1
        assert tensor.shape[1] % 16 == 0
        assert tensor.shape[2] % 16 == 0
        assert tensor.shape[3] == 3

    def test_img2img_preprocess_rgba_image(self, img2img_pipeline):
        """Test RGBA image is converted to RGB."""
        from PIL import Image

        # Create RGBA image
        pil_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))

        tensor = img2img_pipeline._preprocess_image(pil_image)

        assert tensor.shape[3] == 3  # RGB, not RGBA

    def test_img2img_call(self, img2img_pipeline):
        """Test img2img pipeline call."""
        from PIL import Image

        # Create test image
        pil_image = Image.new("RGB", (256, 256), color="blue")

        output = img2img_pipeline(
            prompt="test",
            image=pil_image,
            strength=0.8,
            num_steps=4,
        )

        assert isinstance(output, PipelineOutput)
        assert len(output.images) == 1
        assert output.metadata["strength"] == 0.8

    def test_img2img_call_with_list_images(self, img2img_pipeline):
        """Test img2img pipeline call with list of images."""
        from PIL import Image

        images = [
            Image.new("RGB", (256, 256), color="red"),
            Image.new("RGB", (256, 256), color="green"),
        ]

        output = img2img_pipeline(
            prompt=["prompt1", "prompt2"],
            image=images,
            strength=0.5,
            num_steps=4,
        )

        assert len(output.images) == 2


# ============================================================================
# Test: Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Tests for pipeline factory functions."""

    def test_create_pipeline(self, dummy_model):
        """Test create_pipeline function."""
        from src.inference.pipeline import create_pipeline

        pipeline = create_pipeline(
            model=dummy_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert isinstance(pipeline, PixelHDMPipeline)
        assert pipeline.model is dummy_model

    def test_create_pipeline_with_text_encoder(self, dummy_model):
        """Test create_pipeline with text encoder."""
        from src.inference.pipeline import create_pipeline

        mock_encoder = MockTextEncoder(hidden_size=256)

        pipeline = create_pipeline(
            model=dummy_model,
            text_encoder=mock_encoder,
            device=torch.device("cpu"),
        )

        assert pipeline._text_encoder is mock_encoder


# ============================================================================
# Test: Constants
# ============================================================================

class TestPipelineConstants:
    """Tests for pipeline constants."""

    def test_max_tokens_constant(self):
        """Test MAX_TOKENS constant value."""
        assert MAX_TOKENS == 1024

    def test_default_patch_size_constant(self):
        """Test DEFAULT_PATCH_SIZE constant value."""
        assert DEFAULT_PATCH_SIZE == 16

    def test_constants_are_consistent(self):
        """Test that constants produce valid max resolution."""
        # With default constants, max square resolution should be 512
        max_res = compute_max_resolution(DEFAULT_PATCH_SIZE, MAX_TOKENS)
        assert max_res == 512

        # And validation should pass
        validate_resolution(max_res, max_res, DEFAULT_PATCH_SIZE, MAX_TOKENS)
