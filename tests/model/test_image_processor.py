"""
Image Processor Unit Tests.

Validates ImageProcessorBlock and ImageProcessorStack:
- Shape correctness
- Time conditioning (TokenAdaLN)
- No RoPE injection
- Backward compatibility (layers=0)
- Integration with PixelHDM

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-02-04
"""

import pytest
import torch
import torch.nn as nn

from src.models.blocks.image import ImageProcessorBlock, ImageProcessorStack
from src.models.layers.adaln import TokenAdaLN
from src.models.pixelhdm.core import PixelHDM
from src.config import PixelHDMConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def testing_config():
    """Create minimal config for testing."""
    config = PixelHDMConfig.for_testing()
    object.__setattr__(config, "text_processor_layers", 0)
    object.__setattr__(config, "image_processor_layers", 0)
    return config


@pytest.fixture
def config_with_image_processor():
    """Create config with ImageProcessor enabled."""
    config = PixelHDMConfig.for_testing()
    # Enable ImageProcessor with 1 layer
    object.__setattr__(config, "text_processor_layers", 0)
    object.__setattr__(config, "image_processor_layers", 1)
    return config


@pytest.fixture
def sample_image_tokens(testing_config):
    """Create sample image tokens (B, L, D)."""
    batch_size = 2
    seq_len = 256  # 16x16 patches
    hidden_dim = testing_config.hidden_dim
    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def sample_t_embed(testing_config):
    """Create sample timestep embedding (B, D)."""
    batch_size = 2
    hidden_dim = testing_config.hidden_dim
    return torch.randn(batch_size, hidden_dim)


# =============================================================================
# Test Class: ImageProcessorBlock
# =============================================================================


class TestImageProcessorBlock:
    """Test ImageProcessorBlock."""

    def test_forward_shape(self, testing_config, sample_image_tokens, sample_t_embed):
        """Test output shape matches input shape."""
        block = ImageProcessorBlock(config=testing_config)
        output = block(sample_image_tokens, t_embed=sample_t_embed)

        assert output.shape == sample_image_tokens.shape

    def test_forward_with_mask(self, testing_config, sample_image_tokens, sample_t_embed):
        """Test forward with attention mask."""
        block = ImageProcessorBlock(config=testing_config)
        B, L, D = sample_image_tokens.shape
        mask = torch.ones(B, L, dtype=torch.bool)

        output = block(sample_image_tokens, t_embed=sample_t_embed, image_mask=mask)
        assert output.shape == sample_image_tokens.shape

    def test_no_rope_fn_parameter(self, testing_config):
        """Test that forward does not accept rope_fn parameter."""
        block = ImageProcessorBlock(config=testing_config)

        # ImageProcessorBlock.forward signature should NOT include rope_fn
        import inspect
        sig = inspect.signature(block.forward)
        param_names = list(sig.parameters.keys())

        assert "rope_fn" not in param_names
        assert "position_ids" not in param_names

    def test_has_adaln(self, testing_config):
        """Test that block has TokenAdaLN module."""
        block = ImageProcessorBlock(config=testing_config)

        assert hasattr(block, "adaln")
        assert isinstance(block.adaln, TokenAdaLN)

    def test_has_t_embed_parameter(self, testing_config):
        """Test that forward accepts t_embed parameter."""
        block = ImageProcessorBlock(config=testing_config)

        import inspect
        sig = inspect.signature(block.forward)
        param_names = list(sig.parameters.keys())

        assert "t_embed" in param_names

    def test_timestep_conditioning_changes_output(
        self,
        testing_config,
        sample_image_tokens,
        sample_t_embed,
    ):
        """Test that different t_embed produces different output."""
        torch.manual_seed(0)
        block = ImageProcessorBlock(config=testing_config)
        block.eval()

        t_embed_1 = sample_t_embed
        t_embed_2 = sample_t_embed + 0.5

        with torch.no_grad():
            out_1 = block(sample_image_tokens, t_embed=t_embed_1)
            out_2 = block(sample_image_tokens, t_embed=t_embed_2)

        assert not torch.allclose(out_1, out_2)

    def test_residual_scale(self, testing_config):
        """Test residual scale is computed correctly."""
        # When config is passed, num_layers is read from config.image_processor_layers
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "image_processor_layers", 4)
        block = ImageProcessorBlock(config=config)

        # For k=2, scale = 1/sqrt(2 * num_layers)
        import math
        expected_scale = 1.0 / math.sqrt(2 * 4)
        assert abs(block.residual_scale - expected_scale) < 1e-6


# =============================================================================
# Test Class: ImageProcessorStack
# =============================================================================


class TestImageProcessorStack:
    """Test ImageProcessorStack."""

    def test_forward_shape(self, config_with_image_processor, sample_image_tokens, sample_t_embed):
        """Test output shape matches input shape."""
        stack = ImageProcessorStack(config=config_with_image_processor)
        output = stack(sample_image_tokens, t_embed=sample_t_embed)

        assert output.shape == sample_image_tokens.shape

    def test_num_layers_from_config(self, config_with_image_processor):
        """Test that num_layers is read from config."""
        stack = ImageProcessorStack(config=config_with_image_processor)
        assert stack.num_layers == 1
        assert len(stack.blocks) == 1

    def test_multiple_layers(self, testing_config):
        """Test stack with multiple layers."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "image_processor_layers", 2)
        stack = ImageProcessorStack(config=config)

        assert stack.num_layers == 2
        assert len(stack.blocks) == 2

        # Create sample with correct hidden_dim
        sample = torch.randn(2, 64, config.hidden_dim)
        t_embed = torch.randn(2, config.hidden_dim)
        output = stack(sample, t_embed=t_embed)
        assert output.shape == sample.shape


# =============================================================================
# Test Class: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with image_processor_layers=0."""

    def test_no_image_processor_when_layers_zero(self, testing_config):
        """Test PixelHDM has no image_processor when layers=0."""
        model = PixelHDM(config=testing_config)

        assert model.image_processor is None
        assert model.image_processor_layers == 0

    def test_forward_works_without_image_processor(self, testing_config):
        """Test PixelHDM forward works without ImageProcessor."""
        model = PixelHDM(config=testing_config)
        model.eval()

        # Create sample inputs
        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.no_grad():
            output = model(x_t, t)

        assert output.shape == x_t.shape

    def test_forward_identical_outputs_when_layers_zero(self, testing_config):
        """Test that layers=0 produces same behavior as before."""
        model = PixelHDM(config=testing_config)
        model.eval()

        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)
        text_embed = torch.randn(2, 32, testing_config.hidden_dim)
        text_mask = torch.ones(2, 32)

        torch.manual_seed(42)
        with torch.no_grad():
            output1 = model(x_t, t, text_embed, text_mask)

        # Reset and run again - should be deterministic
        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model(x_t, t, text_embed, text_mask)

        assert torch.allclose(output1, output2)


# =============================================================================
# Test Class: PixelHDM Integration
# =============================================================================


class TestPixelHDMIntegration:
    """Test ImageProcessor integration with PixelHDM."""

    def test_model_with_image_processor(self, config_with_image_processor):
        """Test PixelHDM with ImageProcessor enabled."""
        model = PixelHDM(config=config_with_image_processor)

        assert model.image_processor is not None
        assert model.image_processor_layers == 1

    def test_forward_with_image_processor(self, config_with_image_processor):
        """Test PixelHDM forward with ImageProcessor."""
        model = PixelHDM(config=config_with_image_processor)
        model.eval()

        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.no_grad():
            output = model(x_t, t)

        assert output.shape == x_t.shape

    def test_forward_with_text_and_image_processor(self, config_with_image_processor):
        """Test PixelHDM forward with text and ImageProcessor."""
        model = PixelHDM(config=config_with_image_processor)
        model.eval()

        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)
        text_embed = torch.randn(2, 32, config_with_image_processor.hidden_dim)
        text_mask = torch.ones(2, 32)

        with torch.no_grad():
            output = model(x_t, t, text_embed, text_mask)

        assert output.shape == x_t.shape

    def test_parameter_count_includes_image_processor(self, config_with_image_processor):
        """Test parameter count includes ImageProcessor."""
        model = PixelHDM(config=config_with_image_processor)
        params = model.count_parameters()

        assert "image_processor" in params
        assert params["image_processor"] > 0

    def test_parameter_count_zero_when_no_processor(self, testing_config):
        """Test image_processor parameter count is 0 when disabled."""
        model = PixelHDM(config=testing_config)
        params = model.count_parameters()

        assert "image_processor" in params
        assert params["image_processor"] == 0

    def test_image_processor_receives_t_embed(self, config_with_image_processor):
        """Test PixelHDM passes t_embed to ImageProcessor."""
        model = PixelHDM(config=config_with_image_processor)
        model.eval()

        captured = {}
        original_forward = model.image_processor.forward

        def spy_forward(x, t_embed=None, image_mask=None):
            captured["t_embed"] = t_embed
            return original_forward(x, t_embed=t_embed, image_mask=image_mask)

        model.image_processor.forward = spy_forward

        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.no_grad():
            _ = model(x_t, t)

        assert "t_embed" in captured
        assert captured["t_embed"] is not None
        assert captured["t_embed"].shape == (2, config_with_image_processor.hidden_dim)

    def test_image_processor_timestep_toggle(self):
        """Test image_processor_use_timestep disables timestep conditioning."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "text_processor_layers", 0)
        object.__setattr__(config, "image_processor_layers", 1)
        object.__setattr__(config, "image_processor_use_timestep", False)

        model = PixelHDM(config=config)
        model.eval()

        captured = {}
        original_forward = model.image_processor.forward

        def spy_forward(x, t_embed=None, image_mask=None):
            captured["t_embed"] = t_embed
            return original_forward(x, t_embed=t_embed, image_mask=image_mask)

        model.image_processor.forward = spy_forward

        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.no_grad():
            _ = model(x_t, t)

        assert "t_embed" in captured
        assert captured["t_embed"] is not None
        assert torch.allclose(captured["t_embed"], torch.zeros_like(captured["t_embed"]))


# =============================================================================
# Test Class: Config Validation
# =============================================================================


class TestConfigValidation:
    """Test ImageProcessor config validation."""

    def test_negative_layers_raises(self):
        """Test that negative layers raises ValueError."""
        config = PixelHDMConfig.for_testing()

        # Manually set invalid value (bypassing __post_init__)
        object.__setattr__(config, "image_processor_layers", -1)

        from src.config.validators import _validate_image_processor
        with pytest.raises(ValueError, match="must be >= 0"):
            _validate_image_processor(config)

    def test_zero_layers_valid(self, testing_config):
        """Test that zero layers is valid."""
        from src.config.validators import _validate_image_processor
        _validate_image_processor(testing_config)  # Should not raise

    def test_positive_layers_valid(self, config_with_image_processor):
        """Test that positive layers is valid."""
        from src.config.validators import _validate_image_processor
        _validate_image_processor(config_with_image_processor)  # Should not raise

    def test_mlp_ratio_default(self, testing_config):
        """Test that mlp_ratio defaults to main mlp_ratio."""
        assert testing_config.image_processor_mlp_ratio == testing_config.mlp_ratio


# =============================================================================
# Test Class: RoPE Single Injection
# =============================================================================


class TestRoPESingleInjection:
    """Test that RoPE is only injected in joint blocks, not ImageProcessor."""

    def test_image_processor_attention_no_rope(self, testing_config):
        """Test ImageProcessor attention is called without rope_fn."""
        block = ImageProcessorBlock(config=testing_config)
        x = torch.randn(2, 64, testing_config.hidden_dim)
        t_embed = torch.randn(2, testing_config.hidden_dim)

        # Patch the attention forward to verify rope_fn is None
        call_args = []
        original_forward = block.attention.forward

        def mock_forward(*args, **kwargs):
            call_args.append(kwargs)
            return original_forward(*args, **kwargs)

        block.attention.forward = mock_forward

        with torch.no_grad():
            block(x, t_embed=t_embed)

        # Check rope_fn was None
        assert len(call_args) == 1
        assert call_args[0].get("rope_fn") is None
        assert call_args[0].get("position_ids") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
