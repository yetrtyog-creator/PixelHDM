"""
Tests for PixelHDM-RPEA-DinoV3 Configuration Module.

Tests cover:
- Configuration validation (8 tests)
- Serialization to/from dict and JSON (8 tests)
- Factory methods (4 tests)
- Computed properties (5 tests)

Total: 25 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import PixelHDMConfig, TrainingConfig, DataConfig, Config


# ============================================================================
# Configuration Validation Tests (8 tests)
# ============================================================================

class TestConfigValidation:
    """Test configuration validation in __post_init__."""

    def test_head_dim_computed_correctly(self):
        """Test that head_dim is correctly computed from hidden_dim / num_heads."""
        config = PixelHDMConfig(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=2,
            text_hidden_size=512,
            # Don't set head_dim, let it be computed
        )
        # After __post_init__, head_dim should be hidden_dim // num_heads
        expected_head_dim = 512 // 8  # = 64
        assert config.head_dim == expected_head_dim, \
            f"head_dim should be {expected_head_dim}, got {config.head_dim}"

    def test_mrope_dimension_sum_validation(self):
        """Test that mRoPE dimensions sum to head_dim."""
        # Default config should pass: 16 + 24 + 24 = 64 = head_dim
        config = PixelHDMConfig.default()
        mrope_sum = config.mrope_text_dim + config.mrope_img_h_dim + config.mrope_img_w_dim
        assert mrope_sum == config.head_dim, \
            f"mRoPE dims ({mrope_sum}) should equal head_dim ({config.head_dim})"

    def test_invalid_mrope_dimension_raises(self):
        """Test that invalid mRoPE dimensions raise ValueError."""
        with pytest.raises(AssertionError, match="mRoPE dimensions"):
            PixelHDMConfig(
                hidden_dim=1024,
                num_heads=16,
                num_kv_heads=4,
                text_hidden_size=1024,
                # These don't sum to 64 (head_dim = 1024/16)
                mrope_text_dim=16,
                mrope_img_h_dim=16,  # 16+16+16=48 != 64
                mrope_img_w_dim=16,
            )

    def test_gqa_ratio_validation(self):
        """Test that num_heads is divisible by num_kv_heads."""
        # Valid: 16 % 4 == 0
        config = PixelHDMConfig(
            hidden_dim=1024,
            num_heads=16,
            num_kv_heads=4,
            text_hidden_size=1024,
        )
        assert config.num_heads % config.num_kv_heads == 0

    def test_invalid_gqa_ratio_raises(self):
        """Test that invalid GQA ratio raises AssertionError."""
        with pytest.raises(AssertionError, match="num_heads.*divisible by num_kv_heads"):
            PixelHDMConfig(
                hidden_dim=1024,
                num_heads=16,
                num_kv_heads=5,  # 16 % 5 != 0
                text_hidden_size=1024,
            )

    def test_text_hidden_size_validation(self):
        """Test that text_hidden_size must match hidden_dim."""
        # Default should pass
        config = PixelHDMConfig.default()
        assert config.text_hidden_size == config.hidden_dim

    def test_text_hidden_size_mismatch_allowed(self):
        """Test that mismatched text_hidden_size is allowed (TextProjector handles alignment)."""
        # 不再強制 text_hidden_size == hidden_dim
        # TextProjector 會處理維度對齊
        config = PixelHDMConfig(
            hidden_dim=1024,
            text_hidden_size=2048,  # Different from hidden_dim - allowed now
        )
        assert config.hidden_dim == 1024
        assert config.text_hidden_size == 2048

    def test_default_config_creation(self):
        """Test that default config can be created without errors."""
        config = PixelHDMConfig.default()
        assert config is not None
        assert config.hidden_dim == 1024
        assert config.patch_size == 16

    def test_config_post_init_runs(self):
        """Test that __post_init__ correctly adjusts head_dim if mismatched."""
        # Provide wrong head_dim, should be corrected
        config = PixelHDMConfig(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,  # Wrong, should be 512/8=64
            text_hidden_size=512,
            # Adjust mRoPE to match corrected head_dim
            mrope_text_dim=16,
            mrope_img_h_dim=24,
            mrope_img_w_dim=24,  # Sum = 64
        )
        # __post_init__ should correct head_dim to 64
        assert config.head_dim == 64, \
            f"head_dim should be corrected to 64, got {config.head_dim}"


# ============================================================================
# Serialization Tests (8 tests)
# ============================================================================

class TestConfigSerialization:
    """Test configuration serialization to/from dict and JSON."""

    def test_to_dict_contains_all_params(self):
        """Test that to_dict() includes all configuration parameters."""
        config = PixelHDMConfig.default()
        config_dict = config.to_dict()

        # Check essential parameters are present
        essential_keys = [
            "hidden_dim", "pixel_dim", "patch_size",
            "patch_layers", "pixel_layers",
            "num_heads", "num_kv_heads", "head_dim",
            "mlp_ratio", "mlp_type",
            "repa_enabled", "repa_lambda",
            "freq_loss_enabled", "freq_loss_lambda",
            "text_encoder_name", "text_max_length",
            "mrope_text_dim", "mrope_img_h_dim", "mrope_img_w_dim",
        ]

        for key in essential_keys:
            assert key in config_dict, f"Missing key '{key}' in to_dict() output"

    def test_from_dict_creates_valid_config(self):
        """Test that from_dict() correctly creates a config."""
        original = PixelHDMConfig.default()
        config_dict = original.to_dict()
        restored = PixelHDMConfig.from_dict(config_dict)

        assert restored.hidden_dim == original.hidden_dim
        assert restored.patch_size == original.patch_size
        assert restored.num_heads == original.num_heads
        assert restored.repa_enabled == original.repa_enabled

    def test_config_roundtrip_dict(self):
        """Test that dict -> config -> dict is lossless."""
        original = PixelHDMConfig.default()
        dict1 = original.to_dict()
        restored = PixelHDMConfig.from_dict(dict1)
        dict2 = restored.to_dict()

        assert dict1 == dict2, "Dict roundtrip should be lossless"

    def test_from_json_creates_valid_config(self, temp_json_file):
        """Test that from_json() correctly reads a config file."""
        # Create a config and save it
        original = PixelHDMConfig.default()
        original.to_json(str(temp_json_file))

        # Read it back
        restored = PixelHDMConfig.from_json(str(temp_json_file))

        assert restored.hidden_dim == original.hidden_dim
        assert restored.patch_layers == original.patch_layers

    def test_to_json_writes_correctly(self, temp_json_file):
        """Test that to_json() writes a valid JSON file."""
        config = PixelHDMConfig.default()
        config.to_json(str(temp_json_file))

        # Verify file exists and is valid JSON
        assert temp_json_file.exists()

        with open(temp_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "hidden_dim" in data
        assert data["hidden_dim"] == 1024

    def test_config_roundtrip_json(self, temp_json_file):
        """Test that json -> config -> json is lossless."""
        original = PixelHDMConfig.default()
        original.to_json(str(temp_json_file))

        restored = PixelHDMConfig.from_json(str(temp_json_file))

        # Compare key values
        assert restored.hidden_dim == original.hidden_dim
        assert restored.patch_size == original.patch_size
        assert restored.repa_lambda == original.repa_lambda
        assert restored.freq_loss_quality == original.freq_loss_quality

    def test_from_json_invalid_path_raises(self):
        """Test that from_json() with invalid path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PixelHDMConfig.from_json("/nonexistent/path/config.json")

    def test_from_dict_missing_param_uses_default(self):
        """Test that from_dict() uses defaults for missing parameters."""
        # Minimal dict with only required params
        partial_dict = {
            "hidden_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 2,
            "text_hidden_size": 512,
        }
        config = PixelHDMConfig.from_dict(partial_dict)

        # Should use defaults for missing params
        assert config.patch_size == 16  # default
        assert config.patch_layers == 16  # default
        assert config.repa_enabled is True  # default


# ============================================================================
# Factory Method Tests (4 tests)
# ============================================================================

class TestConfigFactoryMethods:
    """Test configuration factory methods."""

    def test_default_factory_values(self):
        """Test that default() returns expected values."""
        config = PixelHDMConfig.default()

        assert config.hidden_dim == 1024
        assert config.patch_layers == 16
        assert config.pixel_layers == 4
        assert config.num_heads == 16
        assert config.num_kv_heads == 4
        assert config.patch_size == 16

    def test_small_factory_values(self):
        """Test that small() returns expected reduced values."""
        config = PixelHDMConfig.small()

        assert config.hidden_dim == 512
        assert config.patch_layers == 8
        assert config.pixel_layers == 1
        assert config.num_heads == 8
        assert config.num_kv_heads == 2
        assert config.text_hidden_size == 512  # Must match hidden_dim

    def test_large_factory_values(self):
        """Test that large() returns expected increased values."""
        config = PixelHDMConfig.large()

        assert config.hidden_dim == 1152
        assert config.patch_layers == 26
        assert config.pixel_layers == 4
        assert config.num_heads == 16
        assert config.num_kv_heads == 4
        assert config.head_dim == 72
        assert config.text_hidden_size == 1152  # Must match hidden_dim
        # Verify mRoPE dimensions sum to head_dim=72
        mrope_sum = config.mrope_text_dim + config.mrope_img_h_dim + config.mrope_img_w_dim
        assert mrope_sum == config.head_dim == 72

    def test_for_testing_factory_values(self):
        """Test that for_testing() disables REPA and FreqLoss."""
        config = PixelHDMConfig.for_testing()

        # Check minimal dimensions
        assert config.hidden_dim == 256
        assert config.patch_layers == 2
        assert config.pixel_layers == 1

        # Check that expensive features are disabled
        assert config.repa_enabled is False
        assert config.freq_loss_enabled is False
        assert config.use_flash_attention is False
        assert config.use_gradient_checkpointing is False


# ============================================================================
# Computed Properties Tests (5 tests)
# ============================================================================

class TestConfigComputedProperties:
    """Test configuration computed properties."""

    def test_gqa_ratio_property(self):
        """Test that gqa_ratio is correctly computed."""
        config = PixelHDMConfig.default()
        expected_ratio = config.num_heads // config.num_kv_heads
        assert config.gqa_ratio == expected_ratio
        assert config.gqa_ratio == 4  # 16 / 4 = 4

    def test_mlp_hidden_dim_property(self):
        """Test that mlp_hidden_dim is correctly computed."""
        config = PixelHDMConfig.default()
        expected = int(config.hidden_dim * config.mlp_ratio)
        assert config.mlp_hidden_dim == expected
        assert config.mlp_hidden_dim == 3072  # 1024 * 3.0

    def test_pixels_per_patch_property(self):
        """Test that pixels_per_patch returns patch_size squared."""
        config = PixelHDMConfig.default()
        expected = config.patch_size ** 2
        assert config.pixels_per_patch == expected
        assert config.pixels_per_patch == 256  # 16^2

    def test_get_num_patches_square(self):
        """Test get_num_patches for square images."""
        config = PixelHDMConfig.default()

        # 256x256 with patch_size=16 -> (256/16)^2 = 16^2 = 256 patches
        num_patches = config.get_num_patches(256, 256)
        assert num_patches == 256

        # 512x512 -> (512/16)^2 = 32^2 = 1024 patches
        num_patches = config.get_num_patches(512, 512)
        assert num_patches == 1024

    def test_get_num_patches_rectangle(self):
        """Test get_num_patches for rectangular images."""
        config = PixelHDMConfig.default()

        # 512x256 with patch_size=16 -> (512/16) * (256/16) = 32 * 16 = 512 patches
        num_patches = config.get_num_patches(256, 512)
        assert num_patches == 512

        # 1024x512 -> (1024/16) * (512/16) = 64 * 32 = 2048 patches
        num_patches = config.get_num_patches(512, 1024)
        assert num_patches == 2048


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_num_patches_invalid_resolution_raises(self):
        """Test that non-divisible resolution raises AssertionError."""
        config = PixelHDMConfig.default()

        with pytest.raises(AssertionError, match="must be divisible by patch_size"):
            config.get_num_patches(255, 256)  # 255 not divisible by 16

    def test_patch_pixel_dim_property(self):
        """Test patch_pixel_dim = pixels_per_patch * pixel_dim."""
        config = PixelHDMConfig.default()
        expected = config.pixels_per_patch * config.pixel_dim
        assert config.patch_pixel_dim == expected
        # 256 * 16 = 4096
        assert config.patch_pixel_dim == 4096

    def test_num_patches_property_for_256x256(self):
        """Test num_patches property returns correct value for 256x256."""
        config = PixelHDMConfig.default()
        # num_patches property is hardcoded for 256x256
        assert config.num_patches == 256  # (256/16)^2


# ============================================================================
# TrainingConfig and DataConfig Tests
# ============================================================================

class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has correct defaults."""
        config = TrainingConfig()

        assert config.optimizer == "adamw_8bit"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 0
        assert config.batch_size == 16
        assert config.ema_decay == 0.99  # Changed from 0.999 for small batch testing
        assert config.ema_enabled is True
        assert config.mixed_precision == "bf16"

    def test_training_config_betas_tuple(self):
        """Test that betas is a tuple."""
        config = TrainingConfig()
        assert isinstance(config.betas, tuple)
        assert config.betas == (0.9, 0.999)


class TestDataConfig:
    """Test DataConfig class."""

    def test_data_config_defaults(self):
        """Test DataConfig has correct defaults."""
        config = DataConfig()

        assert config.image_size == 512
        assert config.use_bucketing is True
        assert config.min_bucket_size == 256
        assert config.max_bucket_size == 1024
        assert config.bucket_step == 64
        assert config.target_pixels == 262144  # 512*512
        assert config.sampler_mode == "buffered_shuffle"

    def test_data_config_bucketing_params(self):
        """Test bucketing parameters are properly set."""
        config = DataConfig()

        assert config.bucket_max_resolution == 1024
        assert config.bucket_follow_max_resolution is True
        assert config.max_aspect_ratio == 2.0


class TestFullConfig:
    """Test the combined Config class."""

    def test_full_config_default(self):
        """Test Config.default() creates all sub-configs."""
        config = Config.default()

        assert isinstance(config.model, PixelHDMConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)

    def test_full_config_for_testing(self):
        """Test Config.for_testing() creates minimal configs."""
        config = Config.for_testing()

        assert config.model.hidden_dim == 256
        assert config.training.batch_size == 2
        assert config.training.max_steps == 10
        assert config.data.image_size == 64
        assert config.data.num_workers == 0

    def test_config_from_yaml(self, tmp_path):
        """Test Config.from_yaml() loads config correctly."""
        yaml_content = """
model:
  hidden_dim: 512
  patch_layers: 8
  num_heads: 8
  num_kv_heads: 2
  head_dim: 64
  text_hidden_size: 512
  mrope_text_dim: 16
  mrope_img_h_dim: 24
  mrope_img_w_dim: 24

training:
  learning_rate: 2.0e-4
  optimizer:
    type: adamw
    betas: [0.9, 0.95]
  ema:
    decay: 0.999
  lr_schedule:
    warmup_steps: 100

data:
  data_dir: /tmp/test
  image_size: 256
  num_workers: 0
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(yaml_path))

        # Model config
        assert config.model.hidden_dim == 512
        assert config.model.patch_layers == 8

        # Training config (nested to flat conversion)
        assert config.training.learning_rate == 2.0e-4
        assert config.training.optimizer == "adamw"
        assert config.training.betas == (0.9, 0.95)
        assert config.training.ema_decay == 0.999
        assert config.training.warmup_steps == 100

        # Data config
        assert config.data.data_dir == "/tmp/test"
        assert config.data.image_size == 256
        assert config.data.num_workers == 0

    def test_config_from_yaml_file_not_found(self):
        """Test Config.from_yaml() raises on missing file."""
        import pytest
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent_config.yaml")


# ============================================================================
# End-to-End Integration Tests for DataLoader
# ============================================================================

class TestDataLoaderUsesYamlConfig:
    """
    End-to-End tests to verify DataLoader actually uses YAML config values.

    These tests ensure that changing data_dir in YAML actually affects
    where the DataLoader searches for data.
    """

    def test_dataloader_uses_yaml_data_dir(self, tmp_path):
        """
        Critical test: Verify DataLoader uses data_dir from YAML config.

        This test catches the bug where YAML config is parsed correctly
        but DataLoader ignores it and uses default/hardcoded paths.
        """
        from PIL import Image

        # Create custom data directory (NOT the default "datasets")
        custom_data_dir = tmp_path / "my_custom_dataset_path"
        custom_data_dir.mkdir(parents=True)

        # Create test images
        for i in range(4):
            img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
            img.save(custom_data_dir / f"test_image_{i}.png")
            (custom_data_dir / f"test_image_{i}.txt").write_text(
                f"Test caption {i}", encoding="utf-8"
            )

        # Create YAML config pointing to custom directory
        yaml_content = f"""
model:
  hidden_dim: 256
  patch_layers: 2
  num_heads: 4
  num_kv_heads: 2
  text_hidden_size: 256
  head_dim: 64
  mrope_text_dim: 16
  mrope_img_h_dim: 24
  mrope_img_w_dim: 24
  repa_enabled: false
  freq_loss_enabled: false

training:
  batch_size: 2

data:
  data_dir: "{custom_data_dir.as_posix()}"
  image_size: 256
  use_bucketing: false
  num_workers: 0
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        # Load config
        config = Config.from_yaml(str(yaml_path))

        # Verify config was parsed correctly
        assert config.data.data_dir == custom_data_dir.as_posix()

        # Now the critical part: create DataLoader and verify it uses the path
        from src.training.train import create_dataloader_from_config

        dataloader = create_dataloader_from_config(
            data_config=config.data,
            model_config=config.model,
            training_config=config.training,
        )

        # Verify dataset is using the correct path
        assert len(dataloader.dataset) == 4, \
            f"Expected 4 images from {custom_data_dir}, got {len(dataloader.dataset)}"

        # Verify we can actually iterate
        batch = next(iter(dataloader))
        assert batch["images"].shape[0] == 2  # batch_size

    def test_dataloader_fails_on_wrong_path(self, tmp_path):
        """
        Verify DataLoader fails when data_dir points to non-existent directory.

        This ensures data_dir is actually being used, not ignored.
        """
        # Create YAML with non-existent path
        yaml_content = """
model:
  hidden_dim: 256
  patch_layers: 2
  num_heads: 4
  num_kv_heads: 2
  text_hidden_size: 256
  head_dim: 64
  mrope_text_dim: 16
  mrope_img_h_dim: 24
  mrope_img_w_dim: 24
  repa_enabled: false
  freq_loss_enabled: false

training:
  batch_size: 2

data:
  data_dir: "/nonexistent/path/that/should/fail"
  image_size: 256
  use_bucketing: false
  num_workers: 0
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(yaml_path))

        # DataLoader should fail because path doesn't exist
        from src.training.train import create_dataloader_from_config

        with pytest.raises(RuntimeError, match="No images found"):
            create_dataloader_from_config(
                data_config=config.data,
                model_config=config.model,
                training_config=config.training,
            )

    def test_bucket_dataloader_uses_yaml_data_dir(self, tmp_path):
        """
        Verify bucket DataLoader also uses data_dir from YAML config.
        """
        from PIL import Image

        # Create custom data directory
        custom_data_dir = tmp_path / "bucket_test_dataset"
        custom_data_dir.mkdir(parents=True)

        # Create test images with various sizes
        sizes = [(512, 512), (512, 512), (768, 512), (768, 512)]
        for i, (w, h) in enumerate(sizes):
            img = Image.new("RGB", (w, h), color=(i * 60, i * 60, i * 60))
            img.save(custom_data_dir / f"test_image_{i}.png")
            (custom_data_dir / f"test_image_{i}.txt").write_text(
                f"Test caption {i}", encoding="utf-8"
            )

        # Create YAML config with bucketing enabled
        yaml_content = f"""
model:
  hidden_dim: 256
  patch_layers: 2
  num_heads: 4
  num_kv_heads: 2
  text_hidden_size: 256
  head_dim: 64
  mrope_text_dim: 16
  mrope_img_h_dim: 24
  mrope_img_w_dim: 24
  repa_enabled: false
  freq_loss_enabled: false

training:
  batch_size: 2

data:
  data_dir: "{custom_data_dir.as_posix()}"
  image_size: 512
  use_bucketing: true
  min_bucket_size: 256
  max_bucket_size: 1024
  num_workers: 0
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(yaml_path))

        # Create bucket DataLoader
        from src.training.train import create_dataloader_from_config

        dataloader = create_dataloader_from_config(
            data_config=config.data,
            model_config=config.model,
            training_config=config.training,
        )

        # Verify we got images
        assert len(dataloader.dataset) == 4, \
            f"Expected 4 images from {custom_data_dir}"

    def test_legacy_dataset_train_dir_format(self, tmp_path):
        """
        Verify legacy YAML format with dataset.train_dir is supported.

        This tests backward compatibility with data_config.yaml format.
        """
        from PIL import Image

        # Create custom data directory
        custom_data_dir = tmp_path / "legacy_dataset"
        custom_data_dir.mkdir(parents=True)

        # Create test images
        for i in range(4):
            img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
            img.save(custom_data_dir / f"test_image_{i}.png")
            (custom_data_dir / f"test_image_{i}.txt").write_text(
                f"Test caption {i}", encoding="utf-8"
            )

        # Create YAML config with LEGACY format (dataset.train_dir)
        yaml_content = f"""
model:
  hidden_dim: 256
  patch_layers: 2
  num_heads: 4
  num_kv_heads: 2
  text_hidden_size: 256
  head_dim: 64
  mrope_text_dim: 16
  mrope_img_h_dim: 24
  mrope_img_w_dim: 24
  repa_enabled: false
  freq_loss_enabled: false

training:
  batch_size: 2

# LEGACY FORMAT - using dataset.train_dir instead of data.data_dir
dataset:
  train_dir: "{custom_data_dir.as_posix()}"
  target_resolution: 256
  num_workers: 0
  augmentation:
    use_random_crop: false
    use_random_flip: false
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(yaml_path))

        # Verify legacy format was correctly mapped
        assert config.data.data_dir == custom_data_dir.as_posix(), \
            f"Expected data_dir={custom_data_dir.as_posix()}, got {config.data.data_dir}"
        assert config.data.image_size == 256
        assert config.data.use_random_crop is False
        assert config.data.random_flip is False

        # Create DataLoader and verify it works
        from src.training.train import create_dataloader_from_config

        dataloader = create_dataloader_from_config(
            data_config=config.data,
            model_config=config.model,
            training_config=config.training,
        )

        assert len(dataloader.dataset) == 4, \
            f"Expected 4 images from legacy format"
