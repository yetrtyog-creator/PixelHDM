"""
Text Processor Unit Tests.

Validates TextProcessorBlock and TextProcessorStack:
- Shape correctness
- No RoPE injection
- Backward compatibility (layers=0)
- Integration with PixelHDM
"""

import pytest
import torch

from src.models.blocks.text import TextProcessorBlock, TextProcessorStack
from src.models.pixelhdm.core import PixelHDM
from src.config import PixelHDMConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def testing_config():
    """Create minimal config for testing."""
    config = PixelHDMConfig.for_testing()
    object.__setattr__(config, "image_processor_layers", 0)
    object.__setattr__(config, "text_processor_layers", 1)
    return config


@pytest.fixture
def config_with_text_processor():
    """Create config with TextProcessor enabled."""
    config = PixelHDMConfig.for_testing()
    object.__setattr__(config, "image_processor_layers", 0)
    object.__setattr__(config, "text_processor_layers", 1)
    return config


@pytest.fixture
def sample_text_tokens(testing_config):
    """Create sample text tokens (B, T, D)."""
    batch_size = 2
    seq_len = 32
    hidden_dim = testing_config.hidden_dim
    return torch.randn(batch_size, seq_len, hidden_dim)


# =============================================================================
# Test Class: TextProcessorBlock
# =============================================================================


class TestTextProcessorBlock:
    """Test TextProcessorBlock."""

    def test_forward_shape(self, testing_config, sample_text_tokens):
        """Test output shape matches input shape."""
        block = TextProcessorBlock(config=testing_config)
        output = block(sample_text_tokens)

        assert output.shape == sample_text_tokens.shape

    def test_forward_with_mask(self, testing_config, sample_text_tokens):
        """Test forward with attention mask."""
        block = TextProcessorBlock(config=testing_config)
        B, T, _ = sample_text_tokens.shape
        mask = torch.ones(B, T, dtype=torch.bool)

        output = block(sample_text_tokens, text_mask=mask)
        assert output.shape == sample_text_tokens.shape

    def test_no_rope_fn_parameter(self, testing_config):
        """Test that forward does not accept rope_fn parameter."""
        block = TextProcessorBlock(config=testing_config)

        import inspect
        sig = inspect.signature(block.forward)
        param_names = list(sig.parameters.keys())

        assert "rope_fn" not in param_names
        assert "position_ids" not in param_names

    def test_no_adaln(self, testing_config):
        """Test that block does not have AdaLN module."""
        block = TextProcessorBlock(config=testing_config)

        assert not hasattr(block, "adaln")

    def test_no_t_embed_parameter(self, testing_config):
        """Test that forward does not accept t_embed parameter."""
        block = TextProcessorBlock(config=testing_config)

        import inspect
        sig = inspect.signature(block.forward)
        param_names = list(sig.parameters.keys())

        assert "t_embed" not in param_names

    def test_residual_scale(self):
        """Test residual scale is computed correctly."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "text_processor_layers", 4)
        block = TextProcessorBlock(config=config)

        import math
        expected_scale = 1.0 / math.sqrt(2 * 4)
        assert abs(block.residual_scale - expected_scale) < 1e-6


# =============================================================================
# Test Class: TextProcessorStack
# =============================================================================


class TestTextProcessorStack:
    """Test TextProcessorStack."""

    def test_forward_shape(self, config_with_text_processor, sample_text_tokens):
        """Test output shape matches input shape."""
        stack = TextProcessorStack(config=config_with_text_processor)
        output = stack(sample_text_tokens)

        assert output.shape == sample_text_tokens.shape

    def test_num_layers_from_config(self, config_with_text_processor):
        """Test that num_layers is read from config."""
        stack = TextProcessorStack(config=config_with_text_processor)
        assert stack.num_layers == 1
        assert len(stack.blocks) == 1

    def test_multiple_layers(self, testing_config):
        """Test stack with multiple layers."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "text_processor_layers", 2)
        stack = TextProcessorStack(config=config)

        assert stack.num_layers == 2
        assert len(stack.blocks) == 2

        sample = torch.randn(2, 64, config.hidden_dim)
        output = stack(sample)
        assert output.shape == sample.shape


# =============================================================================
# Test Class: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with text_processor_layers=0."""

    def test_no_text_processor_when_layers_zero(self):
        """Test PixelHDM has no text_processor when layers=0."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "text_processor_layers", 0)
        object.__setattr__(config, "image_processor_layers", 0)
        model = PixelHDM(config=config)

        assert model.text_processor is None
        assert model.text_processor_layers == 0


# =============================================================================
# Test Class: PixelHDM Integration
# =============================================================================


class TestPixelHDMIntegration:
    """Test TextProcessor integration with PixelHDM."""

    def test_model_with_text_processor(self, config_with_text_processor):
        """Test PixelHDM with TextProcessor enabled."""
        model = PixelHDM(config=config_with_text_processor)

        assert model.text_processor is not None
        assert model.text_processor_layers == 1

    def test_forward_with_text_processor(self, config_with_text_processor):
        """Test PixelHDM forward with TextProcessor."""
        model = PixelHDM(config=config_with_text_processor)
        model.eval()

        x_t = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)
        text_embed = torch.randn(2, 32, config_with_text_processor.hidden_dim)
        text_mask = torch.ones(2, 32)

        with torch.no_grad():
            output = model(x_t, t, text_embed, text_mask)

        assert output.shape == x_t.shape

    def test_parameter_count_includes_text_processor(self, config_with_text_processor):
        """Test parameter count includes TextProcessor."""
        model = PixelHDM(config=config_with_text_processor)
        params = model.count_parameters()

        assert "text_processor" in params
        assert params["text_processor"] > 0


# =============================================================================
# Test Class: Config Validation
# =============================================================================


class TestConfigValidation:
    """Test TextProcessor config validation."""

    def test_negative_layers_raises(self):
        """Test that negative layers raises ValueError."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "text_processor_layers", -1)

        from src.config.validators import _validate_text_processor
        with pytest.raises(ValueError, match="must be >= 0"):
            _validate_text_processor(config)

    def test_zero_layers_valid(self):
        """Test that zero layers is valid."""
        config = PixelHDMConfig.for_testing()
        object.__setattr__(config, "text_processor_layers", 0)
        from src.config.validators import _validate_text_processor
        _validate_text_processor(config)

    def test_positive_layers_valid(self, config_with_text_processor):
        """Test that positive layers is valid."""
        from src.config.validators import _validate_text_processor
        _validate_text_processor(config_with_text_processor)

    def test_mlp_ratio_default(self, testing_config):
        """Test that mlp_ratio defaults to main mlp_ratio."""
        assert testing_config.text_processor_mlp_ratio == testing_config.mlp_ratio


# =============================================================================
# Test Class: RoPE Single Injection
# =============================================================================


class TestRoPESingleInjection:
    """Test that RoPE is only injected in joint blocks, not TextProcessor."""

    def test_text_processor_attention_no_rope(self, testing_config):
        """Test TextProcessor attention is called without rope_fn."""
        block = TextProcessorBlock(config=testing_config)
        x = torch.randn(2, 32, testing_config.hidden_dim)
        mask = torch.ones(2, 32, dtype=torch.bool)

        call_args = []
        original_forward = block.attention.forward

        def mock_forward(*args, **kwargs):
            call_args.append(kwargs)
            return original_forward(*args, **kwargs)

        block.attention.forward = mock_forward

        with torch.no_grad():
            block(x, text_mask=mask)

        assert len(call_args) == 1
        assert call_args[0].get("rope_fn") is None
        assert call_args[0].get("position_ids") is None
        assert call_args[0].get("attention_mask") is mask


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
