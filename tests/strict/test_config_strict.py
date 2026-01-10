"""
Strict Configuration System Tests

Tests for verifying YAML configuration is correctly parsed and applied.
These tests validate the critical configuration flow from YAML to runtime.

Key issues tested:
- C-001: flow_matching YAML block parsing to model.time_p_mean/time_p_std
- Training optimizer.betas parsing as tuple
- ZClip parameters from gradient section
- data_config external reference mechanism
- output section parsing for checkpoint/logging settings
- Validation of invalid configurations

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Pre-register src package to prevent its __init__.py from triggering
# problematic imports (training module has PyTorch version-dependent imports)
import types
if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")
    sys.modules["src"].__path__ = [str(project_root / "src")]
if "src.config" not in sys.modules:
    sys.modules["src.config"] = types.ModuleType("src.config")
    sys.modules["src.config"].__path__ = [str(project_root / "src" / "config")]

# Now import config module directly - it has no problematic dependencies
# The config module only imports dataclasses, typing, and pathlib
from src.config.model_config import (
    Config,
    DataConfig,
    PixelHDMConfig,
    ResumeConfig,
    TrainingConfig,
)


class TestFlowMatchingYAMLParsing:
    """Test flow_matching YAML block parsing.

    Validates that YAML flow_matching section is correctly mapped to
    model.time_p_mean and model.time_p_std fields.
    """

    def test_flow_matching_yaml_parsing_basic(self, tmp_path: Path):
        """Test basic flow_matching YAML parsing."""
        yaml_content = """
flow_matching:
  P_mean: 0.0
  P_std: 1.0
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Verify flow_matching is mapped to model fields
        assert config.model.time_p_mean == 0.0, \
            f"Expected time_p_mean=0.0, got {config.model.time_p_mean}"
        assert config.model.time_p_std == 1.0, \
            f"Expected time_p_std=1.0, got {config.model.time_p_std}"

    def test_flow_matching_pixelhdm_params(self, tmp_path: Path):
        """Test JiT-style parameters (P_mean=-0.8, P_std=0.8)."""
        yaml_content = """
flow_matching:
  P_mean: -0.8
  P_std: 0.8
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.model.time_p_mean == -0.8
        assert config.model.time_p_std == 0.8

    def test_flow_matching_sd3_params(self, tmp_path: Path):
        """Test SD3/PixelHDM standard parameters."""
        yaml_content = """
flow_matching:
  P_mean: 0.0
  P_std: 1.0
  t_eps: 0.05
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # SD3/PixelHDM standard: P_mean=0.0, P_std=1.0
        assert config.model.time_p_mean == 0.0
        assert config.model.time_p_std == 1.0

    def test_flow_matching_overrides_model_defaults(self, tmp_path: Path):
        """Test that flow_matching overrides model section defaults."""
        yaml_content = """
model:
  time_p_mean: -0.5
  time_p_std: 0.5

flow_matching:
  P_mean: 0.0
  P_std: 1.0
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # flow_matching section should override model section
        assert config.model.time_p_mean == 0.0, \
            "flow_matching.P_mean should override model.time_p_mean"
        assert config.model.time_p_std == 1.0, \
            "flow_matching.P_std should override model.time_p_std"

    def test_flow_matching_partial_override(self, tmp_path: Path):
        """Test partial flow_matching override (only P_mean specified)."""
        yaml_content = """
flow_matching:
  P_mean: 0.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Only P_mean should be overridden
        assert config.model.time_p_mean == 0.5
        # P_std should use default (0.0 as per model_config.py line 111)
        assert config.model.time_p_std == 1.0  # Default value

    def test_flow_matching_empty_section(self, tmp_path: Path):
        """Test empty flow_matching section uses defaults."""
        yaml_content = """
flow_matching: {}
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Should use PixelHDMConfig defaults
        assert config.model.time_p_mean == 0.0  # Default
        assert config.model.time_p_std == 1.0   # Default

    def test_no_flow_matching_section(self, tmp_path: Path):
        """Test missing flow_matching section uses defaults."""
        yaml_content = """
model:
  hidden_dim: 512
  num_heads: 8
  text_hidden_size: 512
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Should use PixelHDMConfig defaults
        assert config.model.time_p_mean == 0.0
        assert config.model.time_p_std == 1.0


class TestTrainingConfigBetasParsing:
    """Test optimizer.betas parsing as tuple."""

    def test_betas_parsing_from_yaml_list(self, tmp_path: Path):
        """Test betas is parsed from YAML list to tuple."""
        yaml_content = """
training:
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Must be tuple, not list
        assert isinstance(config.training.betas, tuple), \
            f"betas should be tuple, got {type(config.training.betas)}"
        assert config.training.betas == (0.9, 0.999)

    def test_betas_custom_values(self, tmp_path: Path):
        """Test custom betas values."""
        yaml_content = """
training:
  optimizer:
    betas: [0.95, 0.98]
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.betas == (0.95, 0.98)

    def test_betas_default_when_not_specified(self, tmp_path: Path):
        """Test betas uses default when not specified."""
        yaml_content = """
training:
  optimizer:
    type: "adamw"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Default from TrainingConfig: (0.9, 0.999)
        assert config.training.betas == (0.9, 0.999)

    def test_eps_parsing(self, tmp_path: Path):
        """Test optimizer.eps parsing."""
        yaml_content = """
training:
  optimizer:
    eps: 1.0e-7
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.eps == 1e-7

    def test_optimizer_type_parsing(self, tmp_path: Path):
        """Test optimizer type parsing."""
        yaml_content = """
training:
  optimizer:
    type: "adamw_8bit"
    betas: [0.9, 0.95]
    eps: 1.0e-6
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.optimizer == "adamw_8bit"
        assert config.training.betas == (0.9, 0.95)
        assert config.training.eps == 1e-6


class TestZClipConfigParsing:
    """Test ZClip configuration parsing from gradient section."""

    def test_zclip_threshold_parsing(self, tmp_path: Path):
        """Test gradient.zclip_threshold parsing."""
        yaml_content = """
training:
  gradient:
    zclip_threshold: 2.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.zclip_threshold == 2.5

    def test_zclip_ema_decay_parsing(self, tmp_path: Path):
        """Test gradient.zclip_ema_decay parsing."""
        yaml_content = """
training:
  gradient:
    zclip_threshold: 3.0
    zclip_ema_decay: 0.95
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.zclip_threshold == 3.0
        assert config.training.zclip_ema_decay == 0.95

    def test_zclip_default_values(self, tmp_path: Path):
        """Test ZClip defaults when not specified."""
        yaml_content = """
training:
  batch_size: 16
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Defaults from TrainingConfig
        assert config.training.zclip_threshold == 2.5
        assert config.training.zclip_ema_decay == 0.99

    def test_gradient_accumulation_parsing(self, tmp_path: Path):
        """Test gradient.accumulation_steps parsing."""
        yaml_content = """
training:
  gradient:
    accumulation_steps: 4
    max_norm: 2.0
    zclip_threshold: 1.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.gradient_accumulation_steps == 4
        assert config.training.max_grad_norm == 2.0
        assert config.training.zclip_threshold == 1.5


class TestDataConfigReference:
    """Test external data configuration reference mechanism."""

    def test_data_config_reference_basic(self, tmp_path: Path):
        """Test data_config: 'path' reference loads external file."""
        # Create external data config
        data_config_content = """
dataset:
  train_dir: "/test/data/path"
  max_resolution: 1024
  min_resolution: 512
  num_workers: 8

multi_resolution:
  enabled: true
  min_size: 512
  max_size: 1024
"""
        data_config_file = tmp_path / "data_config.yaml"
        data_config_file.write_text(data_config_content, encoding="utf-8")

        # Create main config with reference
        main_config_content = f"""
data_config: "{data_config_file.name}"

model:
  hidden_dim: 512
  num_heads: 8
  text_hidden_size: 512
"""
        main_config_file = tmp_path / "train_config.yaml"
        main_config_file.write_text(main_config_content, encoding="utf-8")

        config = Config.from_yaml(str(main_config_file))

        # Verify data config was loaded from external file
        assert config.data.data_dir == "/test/data/path"
        assert config.data.max_bucket_size == 1024
        assert config.data.min_bucket_size == 512
        assert config.data.num_workers == 8
        assert config.data.use_bucketing is True

    def test_data_config_relative_path(self, tmp_path: Path):
        """Test data_config with relative path."""
        # Create subdirectory
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        # Create external data config in subdirectory
        data_config_content = """
dataset:
  train_dir: "/custom/dataset"
  num_workers: 2

multi_resolution:
  enabled: true
  target_pixels: 524288
"""
        data_config_file = configs_dir / "my_data.yaml"
        data_config_file.write_text(data_config_content, encoding="utf-8")

        # Create main config with relative reference
        main_config_content = """
data_config: "my_data.yaml"

training:
  batch_size: 32
"""
        main_config_file = configs_dir / "main.yaml"
        main_config_file.write_text(main_config_content, encoding="utf-8")

        config = Config.from_yaml(str(main_config_file))

        assert config.data.data_dir == "/custom/dataset"
        assert config.data.num_workers == 2
        assert config.data.target_pixels == 524288

    def test_data_config_missing_file(self, tmp_path: Path):
        """Test data_config reference with missing file is handled gracefully."""
        main_config_content = """
data_config: "nonexistent_data.yaml"

model:
  hidden_dim: 512
  num_heads: 8
  text_hidden_size: 512
"""
        main_config_file = tmp_path / "main.yaml"
        main_config_file.write_text(main_config_content, encoding="utf-8")

        # Should not raise, but use defaults
        config = Config.from_yaml(str(main_config_file))

        # Uses DataConfig defaults
        assert config.data.data_dir == "datasets"

    def test_data_config_overrides_inline_data(self, tmp_path: Path):
        """Test that data_config reference overrides inline data section."""
        # Create external data config
        data_config_content = """
dataset:
  train_dir: "/external/path"
  num_workers: 16
"""
        data_config_file = tmp_path / "data.yaml"
        data_config_file.write_text(data_config_content, encoding="utf-8")

        # Create main config with both inline and reference
        main_config_content = """
data_config: "data.yaml"

data:
  data_dir: "/inline/path"
  num_workers: 4
"""
        main_config_file = tmp_path / "main.yaml"
        main_config_file.write_text(main_config_content, encoding="utf-8")

        config = Config.from_yaml(str(main_config_file))

        # External reference should take precedence
        assert config.data.data_dir == "/external/path"
        assert config.data.num_workers == 16


class TestOutputSectionParsing:
    """Test output section parsing for checkpoint/logging settings."""

    def test_max_checkpoints_parsing(self, tmp_path: Path):
        """Test output.max_checkpoints parsing."""
        yaml_content = """
output:
  max_checkpoints: 5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.max_checkpoints == 5

    def test_checkpoint_dir_parsing(self, tmp_path: Path):
        """Test output.checkpoint_dir parsing."""
        yaml_content = """
output:
  checkpoint_dir: "/custom/checkpoints"
  max_checkpoints: 3
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.checkpoint_dir == "/custom/checkpoints"
        assert config.training.max_checkpoints == 3

    def test_save_interval_parsing(self, tmp_path: Path):
        """Test output.save_interval (step-based) parsing."""
        yaml_content = """
output:
  save_interval: 1000
  log_interval: 100
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.save_interval == 1000
        assert config.training.log_interval == 100

    def test_epoch_based_intervals_parsing(self, tmp_path: Path):
        """Test epoch-based interval parsing."""
        yaml_content = """
output:
  save_every_epochs: 10
  log_every_epochs: 1
  save_interval: 0
  log_interval: 0
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.save_every_epochs == 10
        assert config.training.log_every_epochs == 1
        assert config.training.save_interval == 0
        assert config.training.log_interval == 0

    def test_output_section_complete(self, tmp_path: Path):
        """Test complete output section parsing."""
        yaml_content = """
output:
  checkpoint_dir: "./ckpts"
  save_interval: 500
  save_every_epochs: 5
  max_checkpoints: 2
  log_interval: 50
  log_every_epochs: 1
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.checkpoint_dir == "./ckpts"
        assert config.training.save_interval == 500
        assert config.training.save_every_epochs == 5
        assert config.training.max_checkpoints == 2
        assert config.training.log_interval == 50
        assert config.training.log_every_epochs == 1

    def test_legacy_save_every_alias(self, tmp_path: Path):
        """Test legacy save_every alias is mapped to save_interval."""
        yaml_content = """
output:
  save_every: 2000
  log_every: 200
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Legacy aliases mapped to *_interval
        assert config.training.save_interval == 2000
        assert config.training.log_interval == 200


class TestConfigValidationFailure:
    """Test that invalid configurations are properly rejected."""

    def test_hidden_dim_not_divisible_by_num_heads(self):
        """Test hidden_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError) as exc_info:
            PixelHDMConfig(
                hidden_dim=100,  # Not divisible by 16
                num_heads=16,
            )

        assert "must be divisible by num_heads" in str(exc_info.value)

    def test_num_heads_not_divisible_by_kv_heads(self):
        """Test num_heads must be divisible by num_kv_heads."""
        with pytest.raises(AssertionError) as exc_info:
            PixelHDMConfig(
                hidden_dim=512,
                num_heads=8,
                num_kv_heads=3,  # 8 % 3 != 0
            )

        assert "must be divisible by num_kv_heads" in str(exc_info.value)

    def test_text_hidden_size_mismatch_allowed(self):
        """Test text_hidden_size mismatch is allowed (TextProjector handles alignment)."""
        # 不再強制 text_hidden_size == hidden_dim
        # TextProjector 會處理維度對齊
        config = PixelHDMConfig(
            hidden_dim=1024,  # Default hidden_dim
            text_hidden_size=2048,  # Different from hidden_dim - allowed now
        )

        assert config.hidden_dim == 1024
        assert config.text_hidden_size == 2048

    def test_mrope_dimensions_must_sum_to_head_dim(self):
        """Test mRoPE dimensions must sum to head_dim."""
        with pytest.raises(AssertionError) as exc_info:
            PixelHDMConfig(
                hidden_dim=512,
                num_heads=8,  # head_dim = 64
                text_hidden_size=512,  # Must match hidden_dim
                mrope_text_dim=20,
                mrope_img_h_dim=20,
                mrope_img_w_dim=20,  # Sum = 60, not 64
            )

        assert "must equal head_dim" in str(exc_info.value)

    def test_valid_config_passes(self):
        """Test valid configuration does not raise."""
        # Should not raise
        config = PixelHDMConfig(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=2,
            text_hidden_size=512,
            mrope_text_dim=16,
            mrope_img_h_dim=24,
            mrope_img_w_dim=24,  # Sum = 64 = head_dim
        )

        assert config.hidden_dim == 512
        assert config.gqa_ratio == 4  # 8 / 2


class TestComplexYAMLParsing:
    """Test complex YAML configurations with nested structures."""

    def test_full_config_parsing(self, tmp_path: Path):
        """Test complete configuration file parsing."""
        yaml_content = """
model:
  hidden_dim: 512
  num_heads: 8
  num_kv_heads: 2
  text_hidden_size: 512
  repa_enabled: false
  freq_loss_enabled: true

flow_matching:
  P_mean: 0.0
  P_std: 1.0

training:
  batch_size: 16
  learning_rate: 1.0e-4

  optimizer:
    type: "adamw_8bit"
    betas: [0.9, 0.999]
    eps: 1.0e-8

  gradient:
    accumulation_steps: 2
    max_norm: 1.0
    zclip_threshold: 2.5
    zclip_ema_decay: 0.99

  ema:
    enabled: true
    decay: 0.999

  lr_schedule:
    schedule_type: "cosine_restart"
    warmup_steps: 100
    training_mode: "epochs"
    num_epochs: 100
    min_lr: 1.0e-5
    restart_epochs: 25

output:
  checkpoint_dir: "./checkpoints"
  save_every_epochs: 10
  max_checkpoints: 3

resume:
  enabled: true
  checkpoint_path: "auto"
  reset_optimizer: false
  reset_scheduler: false
"""
        config_file = tmp_path / "full_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Model
        assert config.model.hidden_dim == 512
        assert config.model.num_heads == 8
        assert config.model.time_p_mean == 0.0
        assert config.model.time_p_std == 1.0

        # Training
        assert config.training.batch_size == 16
        assert config.training.learning_rate == 1e-4
        assert config.training.optimizer == "adamw_8bit"
        assert config.training.betas == (0.9, 0.999)
        assert config.training.eps == 1e-8
        assert config.training.gradient_accumulation_steps == 2
        assert config.training.zclip_threshold == 2.5
        assert config.training.ema_enabled is True
        assert config.training.ema_decay == 0.999
        assert config.training.warmup_steps == 100
        assert config.training.num_epochs == 100

        # Output
        assert config.training.checkpoint_dir == "./checkpoints"
        assert config.training.save_every_epochs == 10
        assert config.training.max_checkpoints == 3

        # Resume
        assert config.resume.enabled is True
        assert config.resume.checkpoint_path == "auto"
        assert config.resume.reset_optimizer is False

    def test_lr_schedule_steps_mode(self, tmp_path: Path):
        """Test lr_schedule with steps training mode."""
        yaml_content = """
training:
  lr_schedule:
    training_mode: "steps"
    total_steps: 500000
    warmup_steps: 1000
    min_lr: 1.0e-6
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.training_mode == "steps"
        assert config.training.max_steps == 500000
        assert config.training.warmup_steps == 1000
        assert config.training.min_lr == 1e-6

    def test_lr_schedule_epochs_mode(self, tmp_path: Path):
        """Test lr_schedule with epochs training mode."""
        yaml_content = """
training:
  lr_schedule:
    training_mode: "epochs"
    num_epochs: 200
    restart_epochs: 50
    lr_decay_per_cycle: 0.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.training_mode == "epochs"
        assert config.training.num_epochs == 200
        assert config.training.restart_epochs == 50
        assert config.training.lr_decay_per_cycle == 0.5

    def test_multi_resolution_config_parsing(self, tmp_path: Path):
        """Test multi_resolution section parsing to DataConfig."""
        yaml_content = """
multi_resolution:
  enabled: true
  min_size: 256
  max_size: 1024
  step_size: 64
  max_aspect_ratio: 2.0
  target_pixels: 262144
  sampler_mode: "buffered_shuffle"
  chunk_size: 8
  shuffle_chunks: true
  shuffle_within_bucket: false
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.data.use_bucketing is True
        assert config.data.min_bucket_size == 256
        assert config.data.max_bucket_size == 1024
        assert config.data.bucket_step == 64
        assert config.data.max_aspect_ratio == 2.0
        assert config.data.target_pixels == 262144
        assert config.data.sampler_mode == "buffered_shuffle"
        assert config.data.chunk_size == 8
        assert config.data.shuffle_chunks is True
        assert config.data.shuffle_within_bucket is False


class TestConfigDefaults:
    """Test default configuration values."""

    def test_pixelhdm_config_defaults(self):
        """Test PixelHDMConfig default values."""
        config = PixelHDMConfig()

        # Flow matching defaults (SD3/PixelHDM standard)
        assert config.time_p_mean == 0.0
        assert config.time_p_std == 1.0
        assert config.time_eps == 0.05

        # Architecture defaults
        assert config.hidden_dim == 1024
        assert config.patch_size == 16
        assert config.num_heads == 16
        assert config.num_kv_heads == 4

        # REPA defaults
        assert config.repa_enabled is True
        assert config.repa_encoder == "dinov3-vit-b"
        assert config.repa_lambda == 0.5

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()

        assert config.optimizer == "adamw_8bit"
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.zclip_threshold == 2.5
        assert config.zclip_ema_decay == 0.99
        assert config.max_checkpoints == 1

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()

        assert config.data_dir == "datasets"
        assert config.image_size == 512
        assert config.use_bucketing is True
        assert config.sampler_mode == "buffered_shuffle"

    def test_config_for_testing(self):
        """Test Config.for_testing() creates minimal config."""
        config = Config.for_testing()

        assert config.model.hidden_dim == 256
        assert config.model.patch_layers == 2
        assert config.model.repa_enabled is False
        assert config.training.batch_size == 2
        assert config.data.image_size == 64


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_to_dict_roundtrip(self):
        """Test config can be converted to dict and back."""
        original = PixelHDMConfig(
            hidden_dim=512,
            num_heads=8,
            text_hidden_size=512,  # Must match hidden_dim
            time_p_mean=0.5,
            time_p_std=0.8,
        )

        config_dict = original.to_dict()
        restored = PixelHDMConfig.from_dict(config_dict)

        assert restored.hidden_dim == original.hidden_dim
        assert restored.num_heads == original.num_heads
        assert restored.time_p_mean == original.time_p_mean
        assert restored.time_p_std == original.time_p_std

    def test_to_json_roundtrip(self, tmp_path: Path):
        """Test config can be saved to JSON and loaded back."""
        original = PixelHDMConfig(
            hidden_dim=512,
            num_heads=8,
            text_hidden_size=512,  # Must match hidden_dim
            repa_enabled=False,
        )

        json_path = tmp_path / "config.json"
        original.to_json(str(json_path))
        restored = PixelHDMConfig.from_json(str(json_path))

        assert restored.hidden_dim == original.hidden_dim
        assert restored.num_heads == original.num_heads
        assert restored.repa_enabled == original.repa_enabled


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_yaml(self, tmp_path: Path):
        """Test empty YAML file uses all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        # Should use all defaults
        assert config.model.hidden_dim == 1024
        assert config.training.batch_size == 16
        assert config.data.image_size == 512

    def test_yaml_with_only_comments(self, tmp_path: Path):
        """Test YAML with only comments uses defaults."""
        yaml_content = """
# This is a comment
# Another comment
"""
        config_file = tmp_path / "comments.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.model.hidden_dim == 1024

    def test_nonexistent_config_file(self):
        """Test loading nonexistent config file raises error."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path/config.yaml")

    def test_float_precision_preserved(self, tmp_path: Path):
        """Test float values maintain precision."""
        yaml_content = """
training:
  learning_rate: 2.5e-5
  gradient:
    zclip_threshold: 2.123456789
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.training.learning_rate == 2.5e-5
        assert abs(config.training.zclip_threshold - 2.123456789) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
