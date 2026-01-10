"""
Pipeline Input Validation Unit Tests.

Validates resolution validation and input parameter checking:
- Resolution divisibility checks
- Token limit validation
- Prompt normalization
- Guidance scale validation

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
from typing import List

from src.inference.pipeline.validation import (
    InputValidator,
    validate_resolution,
    compute_max_resolution,
    DEFAULT_PATCH_SIZE,
    DEFAULT_MAX_TOKENS,
)
from src.config import PixelHDMConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def validator() -> InputValidator:
    """Create validator with default settings."""
    return InputValidator()


@pytest.fixture
def validator_with_config() -> InputValidator:
    """Create validator with PixelHDMConfig."""
    config = PixelHDMConfig.for_testing()
    return InputValidator(config=config)


# =============================================================================
# Test Class: Resolution Divisibility
# =============================================================================


class TestResolutionDivisibility:
    """Test resolution divisibility validation."""

    def test_valid_square_resolution(self, validator: InputValidator):
        """Test valid square resolution passes."""
        # 512x512 is divisible by 16
        validator.validate_resolution(512, 512)
        # 256x256 is divisible by 16
        validator.validate_resolution(256, 256)

    def test_valid_rectangular_resolution(self):
        """Test valid rectangular resolution passes."""
        # Use high max_tokens to allow large resolutions
        validator = InputValidator(patch_size=16, max_tokens=10000)
        validator.validate_resolution(256, 512)
        validator.validate_resolution(768, 384)

    def test_invalid_height_raises(self, validator: InputValidator):
        """Test non-divisible height raises ValueError."""
        with pytest.raises(ValueError, match="高度"):
            validator.validate_resolution(517, 512)  # 517 not divisible by 16

    def test_invalid_width_raises(self, validator: InputValidator):
        """Test non-divisible width raises ValueError."""
        with pytest.raises(ValueError, match="寬度"):
            validator.validate_resolution(512, 513)  # 513 not divisible by 16

    @pytest.mark.parametrize("height,width", [
        (256, 256),
        (512, 512),
        (256, 512),
    ])
    def test_valid_resolutions_parametrized(
        self, validator: InputValidator, height: int, width: int
    ):
        """Test various valid resolutions within default token limit."""
        validator.validate_resolution(height, width)

    @pytest.mark.parametrize("height,width", [
        (768, 768),
        (1024, 1024),
        (384, 768),
    ])
    def test_large_resolutions_high_token_limit(self, height: int, width: int):
        """Test large resolutions with high token limit."""
        validator = InputValidator(patch_size=16, max_tokens=10000)
        validator.validate_resolution(height, width)


# =============================================================================
# Test Class: Token Limit Validation
# =============================================================================


class TestTokenLimitValidation:
    """Test token limit validation."""

    def test_within_token_limit(self, validator: InputValidator):
        """Test resolution within token limit passes."""
        # 512x512 with patch_size=16 = 32*32 = 1024 tokens
        validator.validate_resolution(512, 512)

    def test_exceeds_token_limit_raises(self):
        """Test resolution exceeding token limit raises ValueError."""
        # Create validator with low max_tokens
        validator = InputValidator(patch_size=16, max_tokens=100)

        # 512x512 = 1024 tokens > 100
        with pytest.raises(ValueError, match="Token 總數"):
            validator.validate_resolution(512, 512)

    def test_token_limit_boundary(self):
        """Test at exact token limit boundary."""
        # 160x160 with patch_size=16 = 10*10 = 100 tokens
        validator = InputValidator(patch_size=16, max_tokens=100)
        validator.validate_resolution(160, 160)  # Should pass

    def test_token_count_calculation(self):
        """Test token count is calculated correctly."""
        # 256x256 with patch_size=16 = 16*16 = 256 tokens
        validator = InputValidator(patch_size=16, max_tokens=256)
        validator.validate_resolution(256, 256)


# =============================================================================
# Test Class: Prompt Validation
# =============================================================================


class TestPromptValidation:
    """Test prompt normalization."""

    def test_single_string_prompt(self, validator: InputValidator):
        """Test single string is converted to list."""
        result = validator.validate_prompt("a cat")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "a cat"

    def test_list_of_prompts(self, validator: InputValidator):
        """Test list of prompts is preserved."""
        prompts = ["a cat", "a dog", "a bird"]
        result = validator.validate_prompt(prompts)
        assert result == prompts

    def test_empty_string_prompt(self, validator: InputValidator):
        """Test empty string prompt."""
        result = validator.validate_prompt("")
        assert result == [""]

    def test_single_item_list(self, validator: InputValidator):
        """Test single item list."""
        result = validator.validate_prompt(["a cat"])
        assert result == ["a cat"]


# =============================================================================
# Test Class: Guidance Scale Validation
# =============================================================================


class TestGuidanceScaleValidation:
    """Test guidance scale validation."""

    def test_positive_guidance_scale(self, validator: InputValidator):
        """Test positive guidance scale passes."""
        result = validator.validate_guidance_scale(7.5)
        assert result == 7.5

    def test_zero_guidance_scale(self, validator: InputValidator):
        """Test zero guidance scale."""
        result = validator.validate_guidance_scale(0.0)
        assert result == 0.0

    def test_negative_guidance_scale_warning(self, validator: InputValidator):
        """Test negative guidance scale returns absolute value."""
        result = validator.validate_guidance_scale(-7.5)
        assert result == 7.5


# =============================================================================
# Test Class: Max Resolution Computation
# =============================================================================


class TestMaxResolutionComputation:
    """Test max resolution computation."""

    def test_compute_max_resolution_default(self, validator: InputValidator):
        """Test max resolution with default settings."""
        # sqrt(1024) = 32 patches, 32*16 = 512
        max_res = validator.compute_max_resolution()
        assert max_res == 512

    def test_compute_max_resolution_custom(self):
        """Test max resolution with custom settings."""
        validator = InputValidator(patch_size=16, max_tokens=4096)
        # sqrt(4096) = 64 patches, 64*16 = 1024
        max_res = validator.compute_max_resolution()
        assert max_res == 1024


# =============================================================================
# Test Class: Standalone Functions
# =============================================================================


class TestStandaloneFunctions:
    """Test standalone validation functions."""

    def test_validate_resolution_function(self):
        """Test standalone validate_resolution function."""
        validate_resolution(512, 512)

        with pytest.raises(ValueError):
            validate_resolution(517, 512)

    def test_compute_max_resolution_function(self):
        """Test standalone compute_max_resolution function."""
        max_res = compute_max_resolution()
        assert max_res == 512

    def test_default_constants(self):
        """Test default constants are set correctly."""
        assert DEFAULT_PATCH_SIZE == 16
        assert DEFAULT_MAX_TOKENS == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
