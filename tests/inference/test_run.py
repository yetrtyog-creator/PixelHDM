"""
Tests for inference run module

Tests checkpoint loading logic, especially EMA weight loading.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn


class TestLoadCheckpointEMAFormat:
    """Test EMA checkpoint format handling"""

    def test_ema_new_format_has_shadow_key(self):
        """Test that new EMA format has shadow key"""
        from src.training.optimization.ema import EMA
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        # Create a small model
        config = PixelHDMConfig()
        model = create_pixelhdm(config)

        # Create EMA
        ema = EMA(model, decay=0.9999)

        # Get state dict
        state_dict = ema.state_dict()

        # Should have shadow key
        assert "shadow" in state_dict
        assert "decay" in state_dict
        assert "num_updates" in state_dict

        # Shadow should contain model parameters
        assert len(state_dict["shadow"]) > 0

    def test_ema_shadow_keys_match_model(self):
        """Test that EMA shadow keys match model parameters"""
        from src.training.optimization.ema import EMA
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        config = PixelHDMConfig()
        model = create_pixelhdm(config)

        ema = EMA(model, decay=0.9999)
        state_dict = ema.state_dict()

        # Shadow keys should match trainable parameters
        model_param_names = {
            name for name, p in model.named_parameters() if p.requires_grad
        }
        shadow_keys = set(state_dict["shadow"].keys())

        assert shadow_keys == model_param_names


class TestLoadCheckpointLogic:
    """Test checkpoint loading logic in run.py"""

    def test_load_checkpoint_ema_branch_new_format(self, tmp_path):
        """Test EMA loading with new format (shadow nested)"""
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        # Create model and get its state dict
        config = PixelHDMConfig()
        model = create_pixelhdm(config)
        model_state = model.state_dict()

        # Create checkpoint with new EMA format
        checkpoint = {
            "model": model_state,
            "ema": {
                "shadow": model_state,  # New format
                "decay": 0.9999,
                "num_updates": 100,
            },
            "config": config.to_dict(),
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Test the loading logic
        from src.inference.run import load_checkpoint

        loaded_model, loaded_config = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
            use_ema=True,
        )

        assert loaded_model is not None

    def test_load_checkpoint_ema_branch_old_format(self, tmp_path):
        """Test EMA loading with old format (direct state dict)"""
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        config = PixelHDMConfig()
        model = create_pixelhdm(config)
        model_state = model.state_dict()

        # Create checkpoint with old EMA format
        checkpoint = {
            "model": model_state,
            "ema": model_state,  # Old format: direct state dict
            "config": config.to_dict(),
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        from src.inference.run import load_checkpoint

        loaded_model, loaded_config = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
            use_ema=True,
        )

        assert loaded_model is not None

    def test_load_checkpoint_model_branch(self, tmp_path):
        """Test loading model weights when no EMA"""
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        config = PixelHDMConfig()
        model = create_pixelhdm(config)
        model_state = model.state_dict()

        checkpoint = {
            "model": model_state,
            "config": config.to_dict(),
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        from src.inference.run import load_checkpoint

        # use_ema=True but no EMA in checkpoint
        loaded_model, _ = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
            use_ema=True,
        )

        assert loaded_model is not None

    def test_load_checkpoint_no_ema_flag(self, tmp_path):
        """Test loading without EMA even when available"""
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        config = PixelHDMConfig()
        model = create_pixelhdm(config)
        model_state = model.state_dict()

        checkpoint = {
            "model": model_state,
            "ema": {
                "shadow": model_state,
                "decay": 0.9999,
            },
            "config": config.to_dict(),
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        from src.inference.run import load_checkpoint

        # use_ema=False should use model weights
        loaded_model, _ = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
            use_ema=False,
        )

        assert loaded_model is not None

    def test_load_checkpoint_file_not_found(self, tmp_path):
        """Test loading non-existent checkpoint raises error"""
        from src.inference.run import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint(str(tmp_path / "nonexistent.pt"), torch.device("cpu"))

    def test_load_checkpoint_with_config(self, tmp_path):
        """Test config is loaded from checkpoint"""
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        # Create config with non-default values
        config = PixelHDMConfig(patch_layers=8, pixel_layers=2)
        model = create_pixelhdm(config)
        model_state = model.state_dict()

        checkpoint = {
            "model": model_state,
            "config": config.to_dict(),
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        from src.inference.run import load_checkpoint

        _, loaded_config = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
        )

        assert loaded_config.patch_layers == 8
        assert loaded_config.pixel_layers == 2

    def test_load_checkpoint_default_config(self, tmp_path):
        """Test default config used when not in checkpoint"""
        from src.config import PixelHDMConfig
        from src.models.pixelhdm import create_pixelhdm

        config = PixelHDMConfig()
        model = create_pixelhdm(config)
        model_state = model.state_dict()

        # No config in checkpoint
        checkpoint = {
            "model": model_state,
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        from src.inference.run import load_checkpoint

        _, loaded_config = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
        )

        # Should use default config
        assert loaded_config.patch_layers == PixelHDMConfig().patch_layers


class TestParseArgs:
    """Test argument parsing"""

    def test_parse_args_required_prompt(self, monkeypatch):
        """Test that prompt is required"""
        import sys
        from src.inference.run import parse_args

        monkeypatch.setattr(sys, "argv", ["run.py"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_parse_args_with_prompt(self, monkeypatch):
        """Test parsing with valid prompt"""
        import sys
        from src.inference.run import parse_args

        monkeypatch.setattr(sys, "argv", ["run.py", "--prompt", "test prompt"])
        args = parse_args()

        assert args.prompt == "test prompt"
        assert args.checkpoint is None  # default (auto-search)
        assert args.width == 512  # default
        assert args.height == 512  # default

    def test_parse_args_all_options(self, monkeypatch):
        """Test parsing with all options"""
        import sys
        from src.inference.run import parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--prompt", "a cat",
                "--checkpoint", "model.pt",
                "--output", "results",
                "--width", "256",
                "--height", "256",
                "--steps", "25",
                "--cfg", "5.0",
                "--seed", "42",
                "--sampler", "euler",
                "--no-ema",
                "--device", "cpu",
            ],
        )
        args = parse_args()

        assert args.prompt == "a cat"
        assert args.checkpoint == "model.pt"
        assert args.output == "results"
        assert args.width == 256
        assert args.height == 256
        assert args.steps == 25
        assert args.cfg == 5.0
        assert args.seed == 42
        assert args.sampler == "euler"
        assert args.no_ema is True
        assert args.device == "cpu"


class TestTrainerCheckpointIntegration:
    """Test trainer saves checkpoint in format run.py can load"""

    def test_trainer_saves_ema_in_expected_format(self, tmp_path):
        """Test trainer checkpoint format matches run.py expectations"""
        from src.config import PixelHDMConfig, TrainingConfig
        from src.models.pixelhdm import create_pixelhdm
        from src.training.trainer import Trainer

        config = PixelHDMConfig()
        training_config = TrainingConfig(ema_decay=0.9999)
        model = create_pixelhdm(config)

        trainer = Trainer(
            model=model,
            config=config,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Save checkpoint
        trainer.save_checkpoint(tmp_path)

        # Load and verify format
        checkpoint_path = list(tmp_path.glob("*.pt"))[0]
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        # Should have model, ema, and config
        assert "model" in checkpoint
        assert "ema" in checkpoint
        assert "config" in checkpoint

        # EMA should have shadow key
        assert "shadow" in checkpoint["ema"]
        assert "decay" in checkpoint["ema"]

    def test_trainer_checkpoint_loads_in_run(self, tmp_path):
        """Test trainer checkpoint can be loaded by run.py"""
        from src.config import PixelHDMConfig, TrainingConfig
        from src.models.pixelhdm import create_pixelhdm
        from src.training.trainer import Trainer
        from src.inference.run import load_checkpoint

        config = PixelHDMConfig()
        training_config = TrainingConfig(ema_decay=0.9999)
        model = create_pixelhdm(config)

        trainer = Trainer(
            model=model,
            config=config,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Simulate some updates
        trainer.ema.update(model, step=0)

        # Save checkpoint
        trainer.save_checkpoint(tmp_path)

        # Load with run.py
        checkpoint_path = list(tmp_path.glob("*.pt"))[0]
        loaded_model, loaded_config = load_checkpoint(
            str(checkpoint_path),
            torch.device("cpu"),
            use_ema=True,
        )

        assert loaded_model is not None
        assert loaded_config.patch_layers == config.patch_layers
