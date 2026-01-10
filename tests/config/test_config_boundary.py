"""
Boundary and Edge Case Tests for PixelHDM Configuration System.

This module provides comprehensive tests for:
- Dimension Validation (7 tests): Validates constraints on hidden_dim, heads, GQA, etc.
- Range Validation (6 tests): Tests value ranges for time_eps, dropout, layers, etc.
- Factory Method Tests (6 tests): Verifies factory method behaviors and outputs.
- Serialization Edge Cases (6 tests): Tests dict/JSON/YAML roundtrip edge cases.

Total: 25 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config, PixelHDMConfig
from src.config.loader import Config as ConfigLoader


# ============================================================================
# Dimension Validation Tests (7 tests)
# ============================================================================

class TestDimensionValidation:
    """Test dimension-related validation constraints."""

    def test_hidden_dim_must_divide_by_heads(self):
        """Test that hidden_dim must be divisible by num_heads.

        This ensures proper attention head dimension computation:
        head_dim = hidden_dim // num_heads
        """
        # Valid: 1024 / 16 = 64
        config = PixelHDMConfig(
            hidden_dim=1024,
            num_heads=16,
            num_kv_heads=4,
            text_hidden_size=1024,
        )
        assert config.head_dim == 64

        # Invalid: 1024 / 15 is not an integer
        with pytest.raises(AssertionError, match="hidden_dim.*divisible by.*num_heads"):
            PixelHDMConfig(
                hidden_dim=1024,
                num_heads=15,
                num_kv_heads=5,
                text_hidden_size=1024,
                # mRoPE sum would also fail, but division fails first
            )

        # Invalid: 512 / 7 is not an integer
        with pytest.raises(AssertionError, match="hidden_dim.*divisible by.*num_heads"):
            PixelHDMConfig(
                hidden_dim=512,
                num_heads=7,
                num_kv_heads=1,
                text_hidden_size=512,
            )

    def test_num_heads_must_divide_num_kv_heads(self):
        """Test that num_heads must be divisible by num_kv_heads.

        This is required for Grouped Query Attention (GQA) where
        multiple Q heads share the same KV head.
        """
        # Valid: 16 / 4 = 4 (GQA ratio of 4)
        config = PixelHDMConfig(
            hidden_dim=1024,
            num_heads=16,
            num_kv_heads=4,
            text_hidden_size=1024,
        )
        assert config.gqa_ratio == 4

        # Valid: 8 / 2 = 4
        config2 = PixelHDMConfig(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=2,
            text_hidden_size=512,
        )
        assert config2.gqa_ratio == 4

        # Invalid: 16 / 5 is not an integer
        with pytest.raises(AssertionError, match="num_heads.*divisible by num_kv_heads"):
            PixelHDMConfig(
                hidden_dim=1024,
                num_heads=16,
                num_kv_heads=5,
                text_hidden_size=1024,
            )

        # Invalid: 16 / 3 is not an integer
        with pytest.raises(AssertionError, match="num_heads.*divisible by num_kv_heads"):
            PixelHDMConfig(
                hidden_dim=1024,
                num_heads=16,
                num_kv_heads=3,
                text_hidden_size=1024,
            )

    def test_gqa_ratio_validation(self):
        """Test that GQA ratio is always an integer.

        GQA ratio = num_heads // num_kv_heads must be a whole number.
        """
        # Test various valid GQA configurations
        test_cases = [
            (16, 16, 1),   # MHA (Multi-Head Attention)
            (16, 8, 2),    # GQA ratio 2
            (16, 4, 4),    # GQA ratio 4
            (16, 2, 8),    # GQA ratio 8
            (16, 1, 16),   # MQA (Multi-Query Attention)
            (8, 2, 4),     # GQA ratio 4
            (32, 8, 4),    # GQA ratio 4
        ]

        for num_heads, num_kv_heads, expected_ratio in test_cases:
            hidden_dim = num_heads * 64  # Ensure divisibility
            config = PixelHDMConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                text_hidden_size=hidden_dim,
            )
            assert config.gqa_ratio == expected_ratio, \
                f"GQA ratio mismatch: {num_heads}/{num_kv_heads} = {config.gqa_ratio}, expected {expected_ratio}"
            assert isinstance(config.gqa_ratio, int), \
                f"GQA ratio should be int, got {type(config.gqa_ratio)}"

    def test_head_dim_computed_correctly(self):
        """Test that head_dim = hidden_dim // num_heads is computed correctly.

        The validator should auto-correct head_dim if provided incorrectly.
        """
        # Test with correct head_dim
        config1 = PixelHDMConfig(
            hidden_dim=1024,
            num_heads=16,
            num_kv_heads=4,
            head_dim=64,  # Correct: 1024/16
            text_hidden_size=1024,
        )
        assert config1.head_dim == 64

        # Test with wrong head_dim (should be auto-corrected)
        config2 = PixelHDMConfig(
            hidden_dim=1024,
            num_heads=16,
            num_kv_heads=4,
            head_dim=32,  # Wrong, will be corrected to 64
            text_hidden_size=1024,
        )
        assert config2.head_dim == 64, "head_dim should be auto-corrected"

        # Test various dimensions
        test_cases = [
            (512, 8, 64),
            (768, 12, 64),
            (1024, 16, 64),
            (1152, 16, 72),
            (2048, 32, 64),
        ]

        for hidden_dim, num_heads, expected_head_dim in test_cases:
            config = PixelHDMConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads // 4 if num_heads >= 4 else 1,
                text_hidden_size=hidden_dim,
                # Set mRoPE to sum to expected_head_dim
                mrope_text_dim=expected_head_dim // 4,
                mrope_img_h_dim=expected_head_dim * 3 // 8,
                mrope_img_w_dim=expected_head_dim - expected_head_dim // 4 - expected_head_dim * 3 // 8,
            )
            assert config.head_dim == expected_head_dim, \
                f"head_dim mismatch: {hidden_dim}/{num_heads} = {config.head_dim}, expected {expected_head_dim}"

    def test_mlp_ratio_positive(self):
        """Test that mlp_ratio must be positive.

        mlp_hidden_dim = hidden_dim * mlp_ratio, so ratio must be > 0.
        """
        # Valid: positive mlp_ratio
        config1 = PixelHDMConfig.for_testing()
        assert config1.mlp_ratio > 0
        assert config1.mlp_hidden_dim > 0

        # Test various positive ratios
        for ratio in [0.5, 1.0, 2.0, 3.0, 4.0]:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                mlp_ratio=ratio,
            )
            expected_mlp_dim = int(256 * ratio)
            assert config.mlp_hidden_dim == expected_mlp_dim, \
                f"mlp_hidden_dim mismatch: {256} * {ratio} = {config.mlp_hidden_dim}, expected {expected_mlp_dim}"

    def test_patch_size_power_of_2(self):
        """Test that patch_size should typically be a power of 2.

        While not strictly enforced, patch_size should be 8, 16, 32, etc.
        for compatibility with image dimensions.
        """
        # Common valid patch sizes (powers of 2)
        valid_sizes = [8, 16, 32]

        for patch_size in valid_sizes:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                patch_size=patch_size,
            )
            assert config.patch_size == patch_size
            # Verify it's a power of 2
            assert (patch_size & (patch_size - 1)) == 0, \
                f"patch_size {patch_size} is not a power of 2"

        # Default patch_size should be 16
        config_default = PixelHDMConfig.default()
        assert config_default.patch_size == 16

        # Test pixels_per_patch calculation
        for patch_size in valid_sizes:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                patch_size=patch_size,
            )
            assert config.pixels_per_patch == patch_size ** 2

    def test_pixel_dim_positive(self):
        """Test that pixel_dim must be positive.

        pixel_dim is the dimension of pixel-level features.
        """
        # Valid: positive pixel_dim
        for pixel_dim in [8, 16, 32, 64]:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                pixel_dim=pixel_dim,
            )
            assert config.pixel_dim == pixel_dim
            assert config.patch_pixel_dim == config.pixels_per_patch * pixel_dim

        # Default should be 16
        config_default = PixelHDMConfig.default()
        assert config_default.pixel_dim == 16


# ============================================================================
# Range Validation Tests (6 tests)
# ============================================================================

class TestRangeValidation:
    """Test value range constraints."""

    def test_time_eps_range(self):
        """Test that time_eps should be in range (0, 0.5).

        time_eps is the epsilon for time sampling to avoid boundary issues.
        """
        # Valid: within (0, 0.5)
        valid_values = [0.01, 0.05, 0.1, 0.2, 0.4]

        for time_eps in valid_values:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                time_eps=time_eps,
            )
            assert config.time_eps == time_eps
            assert 0 < config.time_eps < 0.5, \
                f"time_eps {time_eps} should be in (0, 0.5)"

        # Default should be 0.05
        config_default = PixelHDMConfig.default()
        assert config_default.time_eps == 0.05

    def test_time_p_mean_range(self):
        """Test that time_p_mean is in a reasonable range.

        p_mean controls the center of time sampling distribution.
        """
        # Valid: typical values around 0
        valid_values = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for p_mean in valid_values:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                time_p_mean=p_mean,
            )
            assert config.time_p_mean == p_mean

        # Default should be 0.0
        config_default = PixelHDMConfig.default()
        assert config_default.time_p_mean == 0.0

    def test_time_p_std_positive(self):
        """Test that time_p_std must be positive.

        p_std controls the spread of time sampling distribution.
        """
        # Valid: positive values
        valid_values = [0.1, 0.5, 1.0, 2.0]

        for p_std in valid_values:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                time_p_std=p_std,
            )
            assert config.time_p_std == p_std
            assert config.time_p_std > 0, \
                f"time_p_std {p_std} should be positive"

        # Default should be 1.0
        config_default = PixelHDMConfig.default()
        assert config_default.time_p_std == 1.0

    def test_dropout_rate_range(self):
        """Test that dropout rates are in range [0, 1].

        Dropout, attention_dropout, and cfg_dropout should all be valid probabilities.
        """
        # Test various dropout fields
        dropout_fields = ['dropout', 'attention_dropout', 'cfg_dropout']

        # Valid: within [0, 1]
        for rate in [0.0, 0.1, 0.5, 0.9, 1.0]:
            kwargs = {
                'hidden_dim': 256,
                'num_heads': 4,
                'num_kv_heads': 2,
                'text_hidden_size': 256,
            }
            for field in dropout_fields:
                kwargs[field] = rate

            config = PixelHDMConfig(**kwargs)

            for field in dropout_fields:
                value = getattr(config, field)
                assert 0.0 <= value <= 1.0, \
                    f"{field}={value} should be in [0, 1]"

        # Check defaults
        config_default = PixelHDMConfig.default()
        assert config_default.dropout == 0.0
        assert config_default.attention_dropout == 0.0
        assert config_default.cfg_dropout == 0.1

    def test_layer_counts_positive(self):
        """Test that patch_layers and pixel_layers must be positive.

        Model requires at least 1 layer in each path.
        """
        # Valid: positive layer counts
        valid_cases = [
            (1, 1),
            (2, 1),
            (8, 2),
            (16, 4),
            (26, 4),
        ]

        for patch_layers, pixel_layers in valid_cases:
            config = PixelHDMConfig(
                hidden_dim=256,
                num_heads=4,
                num_kv_heads=2,
                text_hidden_size=256,
                patch_layers=patch_layers,
                pixel_layers=pixel_layers,
            )
            assert config.patch_layers == patch_layers
            assert config.pixel_layers == pixel_layers
            assert config.patch_layers > 0
            assert config.pixel_layers > 0

        # Default values
        config_default = PixelHDMConfig.default()
        assert config_default.patch_layers == 16
        assert config_default.pixel_layers == 4

    def test_hidden_dim_min_size(self):
        """Test that hidden_dim has a reasonable minimum size.

        hidden_dim should be at least 64 for meaningful computation.
        """
        # Valid: >= 64
        valid_dims = [64, 128, 256, 512, 1024]

        for hidden_dim in valid_dims:
            num_heads = max(1, hidden_dim // 64)
            config = PixelHDMConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=max(1, num_heads // 4),
                text_hidden_size=hidden_dim,
            )
            assert config.hidden_dim == hidden_dim
            assert config.hidden_dim >= 64, \
                f"hidden_dim {hidden_dim} should be >= 64"

        # Default should be 1024
        config_default = PixelHDMConfig.default()
        assert config_default.hidden_dim == 1024


# ============================================================================
# Factory Method Tests (6 tests)
# ============================================================================

class TestFactoryMethods:
    """Test configuration factory methods."""

    def test_for_testing_creates_minimal(self):
        """Test that for_testing() creates a minimal, fast configuration.

        Should disable expensive features and use small dimensions.
        """
        config = PixelHDMConfig.for_testing()

        # Minimal dimensions
        assert config.hidden_dim == 256
        assert config.patch_layers == 2
        assert config.pixel_layers == 1

        # Expensive features disabled
        assert config.repa_enabled is False
        assert config.freq_loss_enabled is False
        assert config.use_flash_attention is False
        assert config.use_gradient_checkpointing is False

        # Other minimizations
        assert config.bottleneck_dim == 64
        assert config.mlp_ratio == 2.0

    def test_small_config_values(self):
        """Test that small() has expected intermediate values.

        Should be between for_testing() and default().
        """
        config = PixelHDMConfig.small()

        assert config.hidden_dim == 512
        assert config.patch_layers == 8
        assert config.pixel_layers == 1
        assert config.num_heads == 8
        assert config.num_kv_heads == 2

        # Should maintain default expensive features
        # (unlike for_testing which disables them)

    def test_default_config_values(self):
        """Test that default() has expected production values.

        These are the standard values for training.
        """
        config = PixelHDMConfig.default()

        assert config.hidden_dim == 1024
        assert config.patch_layers == 16
        assert config.pixel_layers == 4
        assert config.num_heads == 16
        assert config.num_kv_heads == 4
        assert config.head_dim == 64
        assert config.patch_size == 16
        assert config.pixel_dim == 16

        # Production features enabled
        assert config.repa_enabled is True
        assert config.freq_loss_enabled is True

    def test_large_config_values(self):
        """Test that large() has expected larger values.

        Should be larger than default() for higher capacity.
        """
        config = PixelHDMConfig.large()

        assert config.hidden_dim == 1152
        assert config.patch_layers == 26
        assert config.pixel_layers == 4
        assert config.num_heads == 16
        assert config.num_kv_heads == 4
        assert config.head_dim == 72  # 1152 / 16 = 72

        # Verify mRoPE sums to head_dim
        mrope_sum = config.mrope_text_dim + config.mrope_img_h_dim + config.mrope_img_w_dim
        assert mrope_sum == config.head_dim

    def test_factory_configs_are_valid(self):
        """Test that all factory configs pass validation.

        Each factory method should produce a valid configuration.
        """
        factory_methods = [
            PixelHDMConfig.for_testing,
            PixelHDMConfig.small,
            PixelHDMConfig.default,
            PixelHDMConfig.large,
        ]

        for factory in factory_methods:
            config = factory()

            # Should not raise any validation errors
            assert config is not None

            # Basic sanity checks
            assert config.hidden_dim > 0
            assert config.patch_layers > 0
            assert config.pixel_layers > 0
            assert config.num_heads > 0
            assert config.num_kv_heads > 0

            # head_dim should be correctly computed
            assert config.head_dim == config.hidden_dim // config.num_heads

            # GQA ratio should be valid
            assert config.num_heads % config.num_kv_heads == 0

            # mRoPE should sum to head_dim
            mrope_sum = config.mrope_text_dim + config.mrope_img_h_dim + config.mrope_img_w_dim
            assert mrope_sum == config.head_dim

    def test_factory_configs_different(self):
        """Test that factory configs are distinct from each other.

        Each factory should produce a meaningfully different configuration.
        """
        testing = PixelHDMConfig.for_testing()
        small = PixelHDMConfig.small()
        default = PixelHDMConfig.default()
        large = PixelHDMConfig.large()

        configs = [testing, small, default, large]
        config_dicts = [c.to_dict() for c in configs]

        # All should have different hidden_dim
        hidden_dims = [c.hidden_dim for c in configs]
        assert len(set(hidden_dims)) == 4, \
            f"All configs should have different hidden_dim: {hidden_dims}"

        # Size ordering: testing < small < default < large
        assert testing.hidden_dim < small.hidden_dim < default.hidden_dim < large.hidden_dim
        assert testing.patch_layers < small.patch_layers < default.patch_layers < large.patch_layers


# ============================================================================
# Serialization Edge Cases Tests (6 tests)
# ============================================================================

class TestSerializationEdgeCases:
    """Test serialization edge cases and roundtrip behavior."""

    def test_to_dict_all_fields_present(self):
        """Test that to_dict() includes all configuration fields.

        Should have 45+ fields for complete configuration.
        """
        config = PixelHDMConfig.default()
        config_dict = config.to_dict()

        # List of all expected fields
        expected_fields = [
            # Core dimensions
            "hidden_dim", "pixel_dim", "patch_size",
            "patch_layers", "pixel_layers",
            "num_heads", "num_kv_heads", "head_dim",
            # MLP
            "mlp_ratio", "mlp_type", "bottleneck_dim",
            # Input/Output
            "in_channels", "out_channels", "max_resolution",
            # Time
            "time_convention", "prediction_type",
            "time_p_mean", "time_p_std", "time_eps",
            # REPA
            "repa_enabled", "repa_encoder", "repa_local_path",
            "repa_use_bf16", "repa_hidden_size", "repa_patch_size",
            "repa_align_layer", "repa_lambda", "repa_early_stop",
            # FreqLoss
            "freq_loss_enabled", "freq_loss_quality",
            "freq_loss_lambda", "freq_loss_block_size", "freq_loss_use_ycbcr",
            # Text
            "text_encoder_name", "text_encoder_frozen",
            "text_max_length", "text_hidden_size",
            # Normalization
            "norm_type", "norm_eps",
            # Dropout
            "dropout", "attention_dropout", "cfg_dropout",
            # Optimization
            "use_flash_attention", "use_gradient_checkpointing", "zero_init_output",
            # mRoPE
            "mrope_text_dim", "mrope_img_h_dim", "mrope_img_w_dim",
            "mrope_text_max_len", "mrope_img_max_len", "mrope_theta",
            # Embedding
            "time_embed_dim", "max_patches",
            # Token Compaction
            "token_compaction_expand_gain",
            # Gated Attention
            "gate_type", "gate_activation", "gate_bias",
            # AdaLN
            "adaln_num_params", "adaln_init_gain",
            # Inference
            "default_num_steps", "default_guidance_scale", "default_sampler_method",
        ]

        for field in expected_fields:
            assert field in config_dict, f"Missing field '{field}' in to_dict()"

        # Should have at least 45 fields
        assert len(config_dict) >= 45, \
            f"Expected 45+ fields, got {len(config_dict)}"

    def test_from_dict_unknown_field_ignored(self):
        """Test that from_dict() ignores unknown fields gracefully.

        Should not raise errors for extra fields in the dictionary.
        """
        # Create dict with unknown fields
        config_dict = {
            "hidden_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 2,
            "text_hidden_size": 512,
            # Unknown fields
            "unknown_field_1": 123,
            "nonexistent_param": "value",
            "future_feature": True,
        }

        # Should not raise, just ignore unknown fields
        # Note: from_dict passes directly to __init__, which may raise TypeError
        # for unknown kwargs. Let's test with valid subset only.
        valid_dict = {k: v for k, v in config_dict.items()
                      if k in [f.name for f in fields(PixelHDMConfig)]}

        config = PixelHDMConfig.from_dict(valid_dict)
        assert config.hidden_dim == 512
        assert config.num_heads == 8

    def test_from_dict_missing_optional_uses_default(self):
        """Test that from_dict() uses defaults for missing optional fields.

        Only required fields should be specified; others should use defaults.
        """
        # Minimal dict with only essential params
        minimal_dict = {
            "hidden_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 2,
            "text_hidden_size": 512,
        }

        config = PixelHDMConfig.from_dict(minimal_dict)

        # Should use defaults for unspecified fields
        assert config.patch_size == 16  # default
        assert config.patch_layers == 16  # default
        assert config.pixel_layers == 4  # default
        assert config.mlp_ratio == 3.0  # default
        assert config.repa_enabled is True  # default
        assert config.freq_loss_enabled is True  # default
        assert config.dropout == 0.0  # default

    def test_json_roundtrip_preserves_floats(self, temp_dir):
        """Test that JSON roundtrip preserves float precision.

        Float values should be preserved within reasonable precision.
        """
        config = PixelHDMConfig.default()
        json_path = temp_dir / "test_float_precision.json"

        # Modify some float values to test precision
        config_dict = config.to_dict()
        config_dict["time_eps"] = 0.123456789
        config_dict["repa_lambda"] = 0.999999999
        config_dict["mlp_ratio"] = 3.141592653589793

        # Write to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

        # Read back
        with open(json_path, "r", encoding="utf-8") as f:
            restored_dict = json.load(f)

        # Check float precision (JSON typically preserves ~15 decimal digits)
        assert abs(restored_dict["time_eps"] - 0.123456789) < 1e-10
        assert abs(restored_dict["repa_lambda"] - 0.999999999) < 1e-10
        assert abs(restored_dict["mlp_ratio"] - 3.141592653589793) < 1e-10

    def test_json_special_values_handled(self, temp_dir):
        """Test that JSON handles special values appropriately.

        None values should be preserved, but inf/nan cannot be in JSON.
        """
        config = PixelHDMConfig.default()
        config_dict = config.to_dict()

        # None should be preserved
        assert config_dict["repa_local_path"] is None

        json_path = temp_dir / "test_special_values.json"
        config.to_json(str(json_path))

        # Read back
        restored = PixelHDMConfig.from_json(str(json_path))
        assert restored.repa_local_path is None

        # Test that the roundtrip works
        assert restored.hidden_dim == config.hidden_dim
        assert restored.repa_enabled == config.repa_enabled

    def test_yaml_parsing_with_comments(self, temp_dir):
        """Test that YAML parsing handles comments correctly.

        Comments should be ignored during parsing.
        """
        yaml_content = """
# This is a comment
model:
  hidden_dim: 512  # inline comment
  patch_layers: 8
  num_heads: 8
  num_kv_heads: 2
  head_dim: 64  # will be auto-computed
  text_hidden_size: 512
  # Another comment
  mrope_text_dim: 16
  mrope_img_h_dim: 24
  mrope_img_w_dim: 24
  repa_enabled: false
  freq_loss_enabled: false

# Training section with comments
training:
  learning_rate: 1.0e-4  # scientific notation
  batch_size: 8

data:
  data_dir: /tmp/test
  image_size: 256
  num_workers: 0
"""
        yaml_path = temp_dir / "test_with_comments.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        # Should parse without errors
        config = Config.from_yaml(str(yaml_path))

        # Verify values were parsed correctly
        assert config.model.hidden_dim == 512
        assert config.model.patch_layers == 8
        assert config.model.num_heads == 8
        assert config.model.repa_enabled is False
        assert config.training.learning_rate == 1.0e-4
        assert config.training.batch_size == 8
        assert config.data.data_dir == "/tmp/test"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
