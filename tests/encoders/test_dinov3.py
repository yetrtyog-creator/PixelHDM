"""
DINOv3 Encoder Tests

Tests for DINOv3Encoder and DINOv3FeatureProjector with mocked model loading.

Test Categories:
    - Initialization (6 tests): Config parsing, model name validation
    - Feature Projector (5 tests): Shape transformation, weight init
    - Factory Functions (4 tests): Creation methods
    - Forward Pass with Mock (5 tests): Input handling, output format

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from src.models.encoders.dinov3 import (
    DINOv3Encoder,
    DINOv3FeatureProjector,
    DINOFeatureProjector,
    create_dinov3_encoder,
    create_dinov3_encoder_from_config,
    create_feature_projector,
    create_feature_projector_from_config,
)
from src.config import PixelHDMConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def testing_config() -> PixelHDMConfig:
    """Create minimal config for testing."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def mock_dino_output():
    """Create mock DINOv3 model output."""
    class MockOutput:
        def __init__(self, batch_size: int, num_patches: int, embed_dim: int):
            # Include CLS token (first token)
            self.last_hidden_state = torch.randn(batch_size, num_patches + 1, embed_dim)
    return MockOutput


# ============================================================================
# DINOv3Encoder Initialization Tests
# ============================================================================

class TestDINOv3EncoderInit:
    """Tests for DINOv3Encoder initialization."""

    def test_default_init(self):
        """Test default initialization without loading model."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        assert encoder.model_name == "dinov3-vitb16"
        assert encoder.embed_dim == 768
        assert encoder.patch_size == 16
        assert encoder._loaded is False
        assert encoder.model is None

    def test_init_with_config(self, testing_config):
        """Test initialization from config."""
        encoder = DINOv3Encoder(config=testing_config, pretrained=False)

        # Config uses "dinov3-vit-b" which should parse to "dinov3-vitb16"
        assert encoder.embed_dim == 768
        assert encoder.patch_size == 16

    def test_model_configs_all_patch_size_16(self):
        """Test all model configs have patch_size=16."""
        for model_name, config in DINOv3Encoder.MODEL_CONFIGS.items():
            assert config["patch_size"] == 16, f"{model_name} should have patch_size=16"

    def test_parse_model_name_variants(self):
        """Test model name parsing for various formats."""
        encoder = DINOv3Encoder(pretrained=False)

        # Base variants
        assert encoder._parse_model_name("dinov3-vitb16") == "dinov3-vitb16"
        assert encoder._parse_model_name("dinov3-vit-b") == "dinov3-vitb16"
        assert encoder._parse_model_name("dinov3_vitb16") == "dinov3-vitb16"

        # Large variants
        assert encoder._parse_model_name("dinov3-vitl16") == "dinov3-vitl16"
        assert encoder._parse_model_name("large") == "dinov3-vitl16"
        assert encoder._parse_model_name("vit-l") == "dinov3-vitl16"

        # Small variants
        assert encoder._parse_model_name("small") == "dinov3-vits16"
        assert encoder._parse_model_name("vit-s") == "dinov3-vits16"

        # 7B variant
        assert encoder._parse_model_name("7b") == "dinov3-vit7b16"
        assert encoder._parse_model_name("giant") == "dinov3-vit7b16"

    def test_unknown_model_name_defaults_to_base(self):
        """Test unknown model name defaults to base variant."""
        # Unknown names are parsed and default to base variant
        encoder = DINOv3Encoder(model_name="unknown-model", pretrained=False)
        assert encoder.model_name == "dinov3-vitb16"
        assert encoder.embed_dim == 768

    def test_embed_dims_correct(self):
        """Test embed_dim is correct for each model variant."""
        expected_dims = {
            "dinov3-vit7b16": 4096,
            "dinov3-vitl16": 1024,
            "dinov3-vitb16": 768,
            "dinov3-vits16": 384,
        }

        for model_name, expected_dim in expected_dims.items():
            encoder = DINOv3Encoder(model_name=model_name, pretrained=False)
            assert encoder.embed_dim == expected_dim, f"{model_name} should have embed_dim={expected_dim}"


# ============================================================================
# DINOv3Encoder Forward Pass Tests (Mocked)
# ============================================================================

class TestDINOv3EncoderForward:
    """Tests for DINOv3Encoder forward pass with mocked model."""

    def test_forward_with_mock_model(self, mock_dino_output):
        """Test forward pass with mocked model."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        # Create mock model
        B, H, W = 2, 256, 256
        num_patches = (H // 16) * (W // 16)  # 256
        embed_dim = 768

        mock_model = MagicMock()
        mock_model.return_value = mock_dino_output(B, num_patches, embed_dim)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))

        # Inject mock model
        encoder.model = mock_model
        encoder._loaded = True

        # Forward pass
        x = torch.randn(B, H, W, 3)
        output = encoder(x)

        # Check output shape (CLS token excluded)
        assert output.shape == (B, num_patches, embed_dim)

    def test_forward_bhwc_format(self, mock_dino_output):
        """Test forward with (B, H, W, C) input format."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        B, H, W = 2, 256, 256
        num_patches = (H // 16) * (W // 16)
        embed_dim = 768

        mock_model = MagicMock()
        mock_model.return_value = mock_dino_output(B, num_patches, embed_dim)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))

        encoder.model = mock_model
        encoder._loaded = True

        # (B, H, W, C) format
        x = torch.randn(B, H, W, 3)
        output = encoder(x)

        assert output.shape == (B, num_patches, embed_dim)

    def test_forward_bchw_format(self, mock_dino_output):
        """Test forward with (B, C, H, W) input format."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        B, H, W = 2, 256, 256
        num_patches = (H // 16) * (W // 16)
        embed_dim = 768

        mock_model = MagicMock()
        mock_model.return_value = mock_dino_output(B, num_patches, embed_dim)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))

        encoder.model = mock_model
        encoder._loaded = True

        # (B, C, H, W) format
        x = torch.randn(B, 3, H, W)
        output = encoder(x)

        assert output.shape == (B, num_patches, embed_dim)

    def test_forward_return_dict(self, mock_dino_output):
        """Test forward with return_dict=True."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        B, H, W = 2, 256, 256
        num_patches = (H // 16) * (W // 16)
        embed_dim = 768

        mock_model = MagicMock()
        mock_model.return_value = mock_dino_output(B, num_patches, embed_dim)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))

        encoder.model = mock_model
        encoder._loaded = True

        x = torch.randn(B, H, W, 3)
        output = encoder(x, return_dict=True)

        assert isinstance(output, dict)
        assert "patch_tokens" in output
        assert "cls_token" in output
        assert output["patch_tokens"].shape == (B, num_patches, embed_dim)
        assert output["cls_token"].shape == (B, embed_dim)

    def test_forward_invalid_input_raises(self):
        """Test forward with invalid input raises error."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)
        encoder._loaded = True
        encoder.model = MagicMock()

        # 3D input should fail
        x = torch.randn(2, 256, 256)

        with pytest.raises(ValueError, match="4D"):
            encoder(x)


# ============================================================================
# DINOv3FeatureProjector Tests
# ============================================================================

class TestDINOv3FeatureProjector:
    """Tests for DINOv3FeatureProjector."""

    def test_projector_init_default(self):
        """Test default initialization."""
        proj = DINOv3FeatureProjector(input_dim=1024, output_dim=768)

        assert proj.input_dim == 1024
        assert proj.output_dim == 768

    def test_projector_init_from_config(self, testing_config):
        """Test initialization from config."""
        proj = DINOv3FeatureProjector(config=testing_config)

        assert proj.input_dim == testing_config.hidden_dim
        assert proj.output_dim == testing_config.repa_hidden_size

    def test_projector_forward_shape(self):
        """Test forward pass produces correct shape."""
        proj = DINOv3FeatureProjector(input_dim=1024, output_dim=768)

        B, L = 2, 256
        x = torch.randn(B, L, 1024)
        output = proj(x)

        assert output.shape == (B, L, 768)

    def test_projector_forward_no_nan(self):
        """Test forward pass produces no NaN values."""
        proj = DINOv3FeatureProjector(input_dim=1024, output_dim=768)

        x = torch.randn(2, 256, 1024)
        output = proj(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_projector_gradient_flow(self):
        """Test gradients flow through projector."""
        proj = DINOv3FeatureProjector(input_dim=1024, output_dim=768)

        x = torch.randn(2, 256, 1024, requires_grad=True)
        output = proj(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_alias_exists(self):
        """Test DINOFeatureProjector alias exists."""
        assert DINOFeatureProjector is DINOv3FeatureProjector


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_dinov3_encoder(self):
        """Test create_dinov3_encoder factory."""
        encoder = create_dinov3_encoder(
            model_name="dinov3-vitb16",
            pretrained=False,
        )

        assert isinstance(encoder, DINOv3Encoder)
        assert encoder.model_name == "dinov3-vitb16"
        assert encoder.embed_dim == 768

    def test_create_dinov3_encoder_from_config(self, testing_config):
        """Test create_dinov3_encoder_from_config factory."""
        encoder = create_dinov3_encoder_from_config(testing_config)

        assert isinstance(encoder, DINOv3Encoder)
        assert encoder.embed_dim == 768

    def test_create_feature_projector(self):
        """Test create_feature_projector factory."""
        proj = create_feature_projector(input_dim=1024, output_dim=768)

        assert isinstance(proj, DINOv3FeatureProjector)
        assert proj.input_dim == 1024
        assert proj.output_dim == 768

    def test_create_feature_projector_from_config(self, testing_config):
        """Test create_feature_projector_from_config factory."""
        proj = create_feature_projector_from_config(testing_config)

        assert isinstance(proj, DINOv3FeatureProjector)
        assert proj.input_dim == testing_config.hidden_dim
        assert proj.output_dim == testing_config.repa_hidden_size


# ============================================================================
# Extra Methods Tests
# ============================================================================

class TestExtraMethods:
    """Tests for extra methods."""

    def test_get_output_dim(self):
        """Test get_output_dim method."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)
        assert encoder.get_output_dim() == 768

        encoder_large = DINOv3Encoder(model_name="dinov3-vitl16", pretrained=False)
        assert encoder_large.get_output_dim() == 1024

    def test_extra_repr_encoder(self):
        """Test extra_repr for encoder."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)
        repr_str = encoder.extra_repr()

        assert "dinov3-vitb16" in repr_str
        assert "768" in repr_str
        assert "16" in repr_str

    def test_extra_repr_projector(self):
        """Test extra_repr for projector."""
        proj = DINOv3FeatureProjector(input_dim=1024, output_dim=768)
        repr_str = proj.extra_repr()

        assert "1024" in repr_str
        assert "768" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
