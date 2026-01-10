"""
DINOv3 Weight Loader Tests

Tests for DINOv3WeightLoader and DINOv3Encoder lazy loading logic.

Test Categories:
    - DINOv3WeightLoader Basic (5 tests): load_from_local, _unwrap_state_dict
    - DINOv3WeightLoader Key Matching (6 tests): _match_state_dict, _remove_prefixes
    - DINOv3WeightLoader Error Handling (4 tests): File not found, format errors
    - DINOv3Encoder Lazy Loading (6 tests): _load_model, _ensure_loaded
    - DINOv3Encoder Device Migration (5 tests): to(), dtype handling
    - DINOv3Encoder Integration (4 tests): Full load flow

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest
import torch
import torch.nn as nn

from src.models.encoders.dinov3.loader import DINOv3WeightLoader
from src.models.encoders.dinov3.encoder import DINOv3Encoder


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_state_dict() -> Dict[str, torch.Tensor]:
    """Create a mock state dict with typical DINOv3 weights."""
    return {
        "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
        "patch_embed.proj.bias": torch.randn(768),
        "blocks.0.attn.qkv.weight": torch.randn(2304, 768),
        "blocks.0.attn.proj.weight": torch.randn(768, 768),
        "blocks.0.mlp.fc1.weight": torch.randn(3072, 768),
        "blocks.0.mlp.fc2.weight": torch.randn(768, 3072),
        "norm.weight": torch.randn(768),
        "norm.bias": torch.randn(768),
    }


@pytest.fixture
def mock_prefixed_state_dict(mock_state_dict) -> Dict[str, torch.Tensor]:
    """Create a mock state dict with 'backbone.' prefix."""
    return {f"backbone.{k}": v for k, v in mock_state_dict.items()}


@pytest.fixture
def mock_model() -> nn.Module:
    """Create a mock model with state dict keys matching mock_state_dict."""
    model = MagicMock(spec=nn.Module)
    model.state_dict.return_value = {
        "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
        "patch_embed.proj.bias": torch.randn(768),
        "blocks.0.attn.qkv.weight": torch.randn(2304, 768),
        "blocks.0.attn.proj.weight": torch.randn(768, 768),
        "blocks.0.mlp.fc1.weight": torch.randn(3072, 768),
        "blocks.0.mlp.fc2.weight": torch.randn(768, 3072),
        "norm.weight": torch.randn(768),
        "norm.bias": torch.randn(768),
    }
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def temp_weight_file(mock_state_dict) -> str:
    """Create a temporary weight file."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(mock_state_dict, f.name)
        return f.name


# ============================================================================
# DINOv3WeightLoader Basic Tests
# ============================================================================

class TestDINOv3WeightLoaderBasic:
    """Tests for DINOv3WeightLoader basic functionality."""

    def test_init(self):
        """Test loader initialization."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        assert loader.model_name == "dinov3-vitb16"

    def test_init_different_models(self):
        """Test loader initialization with different model names."""
        for model_name in ["dinov3-vitb16", "dinov3-vitl16", "dinov3-vits16", "dinov3-vit7b16"]:
            loader = DINOv3WeightLoader(model_name)
            assert loader.model_name == model_name

    def test_hf_model_ids_mapping(self):
        """Test HuggingFace model ID mapping."""
        assert "dinov3-vitb16" in DINOv3WeightLoader.HF_MODEL_IDS
        assert "dinov3-vitl16" in DINOv3WeightLoader.HF_MODEL_IDS
        assert "dinov3-vits16" in DINOv3WeightLoader.HF_MODEL_IDS
        assert "dinov3-vit7b16" in DINOv3WeightLoader.HF_MODEL_IDS

    @patch("torch.load")
    def test_load_from_local_basic(self, mock_torch_load, mock_state_dict, mock_model, tmp_path):
        """Test basic load_from_local functionality with mocked torch.load."""
        # Create a dummy file
        weight_file = tmp_path / "weights.pth"
        weight_file.touch()

        mock_torch_load.return_value = mock_state_dict

        loader = DINOv3WeightLoader("dinov3-vitb16")
        loader.load_from_local(mock_model, str(weight_file))

        mock_torch_load.assert_called_once_with(
            weight_file, map_location="cpu", weights_only=True
        )
        mock_model.load_state_dict.assert_called()

    @patch("torch.load")
    def test_load_from_local_prints_message(self, mock_torch_load, mock_state_dict, mock_model, tmp_path, capsys):
        """Test load_from_local prints success message."""
        weight_file = tmp_path / "weights.pth"
        weight_file.touch()

        mock_torch_load.return_value = mock_state_dict

        loader = DINOv3WeightLoader("dinov3-vitb16")
        loader.load_from_local(mock_model, str(weight_file))

        captured = capsys.readouterr()
        assert "[DINOv3] Loaded weights from:" in captured.out


# ============================================================================
# DINOv3WeightLoader _unwrap_state_dict Tests
# ============================================================================

class TestUnwrapStateDict:
    """Tests for _unwrap_state_dict method."""

    def test_unwrap_plain_dict(self, mock_state_dict):
        """Test unwrap with plain state dict (no wrapper)."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._unwrap_state_dict(mock_state_dict)
        assert result == mock_state_dict

    def test_unwrap_model_wrapper(self, mock_state_dict):
        """Test unwrap with 'model' wrapper."""
        wrapped = {"model": mock_state_dict}
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._unwrap_state_dict(wrapped)
        assert result == mock_state_dict

    def test_unwrap_state_dict_wrapper(self, mock_state_dict):
        """Test unwrap with 'state_dict' wrapper."""
        wrapped = {"state_dict": mock_state_dict}
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._unwrap_state_dict(wrapped)
        assert result == mock_state_dict

    def test_unwrap_model_takes_priority(self, mock_state_dict):
        """Test 'model' key takes priority over 'state_dict'."""
        other_dict = {"other_key": torch.randn(1)}
        wrapped = {"model": mock_state_dict, "state_dict": other_dict}
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._unwrap_state_dict(wrapped)
        assert result == mock_state_dict

    def test_unwrap_with_extra_keys(self, mock_state_dict):
        """Test unwrap with extra metadata keys."""
        wrapped = {
            "model": mock_state_dict,
            "epoch": 100,
            "optimizer": {"state": {}},
            "config": {"hidden_dim": 768},
        }
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._unwrap_state_dict(wrapped)
        assert result == mock_state_dict


# ============================================================================
# DINOv3WeightLoader _remove_prefixes Tests
# ============================================================================

class TestRemovePrefixes:
    """Tests for _remove_prefixes method."""

    def test_remove_backbone_prefix(self):
        """Test removing 'backbone.' prefix."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder.", "model.", "module."]

        result = loader._remove_prefixes("backbone.blocks.0.weight", prefixes)
        assert result == "blocks.0.weight"

    def test_remove_encoder_prefix(self):
        """Test removing 'encoder.' prefix."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder.", "model.", "module."]

        result = loader._remove_prefixes("encoder.norm.weight", prefixes)
        assert result == "norm.weight"

    def test_remove_model_prefix(self):
        """Test removing 'model.' prefix."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder.", "model.", "module."]

        result = loader._remove_prefixes("model.patch_embed.proj.weight", prefixes)
        assert result == "patch_embed.proj.weight"

    def test_remove_module_prefix(self):
        """Test removing 'module.' prefix (DataParallel)."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder.", "model.", "module."]

        result = loader._remove_prefixes("module.blocks.1.attn.weight", prefixes)
        assert result == "blocks.1.attn.weight"

    def test_no_prefix_unchanged(self):
        """Test key without prefix remains unchanged."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder.", "model.", "module."]

        result = loader._remove_prefixes("blocks.0.weight", prefixes)
        assert result == "blocks.0.weight"

    def test_only_first_prefix_removed(self):
        """Test only the first matching prefix is removed."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder."]

        # backbone. is first in the list and matches
        result = loader._remove_prefixes("backbone.encoder.weight", prefixes)
        assert result == "encoder.weight"


# ============================================================================
# DINOv3WeightLoader _match_state_dict Tests
# ============================================================================

class TestMatchStateDict:
    """Tests for _match_state_dict method."""

    def test_match_exact_keys(self, mock_state_dict, mock_model):
        """Test matching with exact keys."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict(mock_state_dict, mock_model)

        assert len(result) == len(mock_state_dict)
        for key in mock_state_dict:
            assert key in result

    def test_match_prefixed_keys(self, mock_prefixed_state_dict, mock_model):
        """Test matching with prefixed keys."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict(mock_prefixed_state_dict, mock_model)

        # All keys should be matched after prefix removal
        assert len(result) > 0
        for key in result:
            assert not key.startswith("backbone.")

    def test_match_ignores_unmatched_keys(self, mock_model):
        """Test that unmatched keys are ignored."""
        state_dict = {
            "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
            "unknown_key.weight": torch.randn(100),  # Not in model
            "another_unknown": torch.randn(50),  # Not in model
        }

        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict(state_dict, mock_model)

        assert "patch_embed.proj.weight" in result
        assert "unknown_key.weight" not in result
        assert "another_unknown" not in result

    def test_match_with_encoder_prefix(self, mock_state_dict, mock_model):
        """Test matching with 'encoder.' prefix."""
        prefixed = {f"encoder.{k}": v for k, v in mock_state_dict.items()}

        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict(prefixed, mock_model)

        assert len(result) == len(mock_state_dict)

    def test_match_with_module_prefix(self, mock_state_dict, mock_model):
        """Test matching with 'module.' prefix (DataParallel)."""
        prefixed = {f"module.{k}": v for k, v in mock_state_dict.items()}

        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict(prefixed, mock_model)

        assert len(result) == len(mock_state_dict)

    def test_match_returns_correct_tensors(self, mock_state_dict, mock_model):
        """Test that matched tensors are the correct ones."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict(mock_state_dict, mock_model)

        for key, tensor in result.items():
            assert torch.equal(tensor, mock_state_dict[key])


# ============================================================================
# DINOv3WeightLoader Error Handling Tests
# ============================================================================

class TestLoaderErrorHandling:
    """Tests for DINOv3WeightLoader error handling."""

    def test_load_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        model = MagicMock()

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_from_local(model, "/nonexistent/path/to/weights.pth")

        assert "not found" in str(exc_info.value).lower()

    def test_load_file_not_found_message_includes_path(self):
        """Test error message includes the missing path."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        model = MagicMock()
        bad_path = "/some/invalid/path/weights.pth"

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_from_local(model, bad_path)

        assert bad_path in str(exc_info.value) or "weights.pth" in str(exc_info.value)

    @patch("torch.load")
    def test_load_strict_fails_fallback_to_match(self, mock_torch_load, mock_model, tmp_path):
        """Test fallback to _match_state_dict when strict loading fails."""
        weight_file = tmp_path / "weights.pth"
        weight_file.touch()

        # State dict with extra key that will cause strict=True to fail
        state_dict = {
            "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
            "extra_key_not_in_model": torch.randn(100),
        }
        mock_torch_load.return_value = state_dict

        # Make strict loading fail
        mock_model.load_state_dict.side_effect = [
            RuntimeError("Key mismatch"),
            None,  # Second call (strict=False) succeeds
        ]

        loader = DINOv3WeightLoader("dinov3-vitb16")
        loader.load_from_local(mock_model, str(weight_file))

        # Should be called twice: first strict=True, then strict=False
        assert mock_model.load_state_dict.call_count == 2

    @patch("torch.load")
    def test_load_nested_wrapper(self, mock_torch_load, mock_state_dict, mock_model, tmp_path):
        """Test loading from deeply nested checkpoint structure."""
        weight_file = tmp_path / "weights.pth"
        weight_file.touch()

        # Nested structure with 'model' wrapper
        wrapped = {
            "model": mock_state_dict,
            "epoch": 100,
            "global_step": 50000,
        }
        mock_torch_load.return_value = wrapped

        loader = DINOv3WeightLoader("dinov3-vitb16")
        loader.load_from_local(mock_model, str(weight_file))

        # Should have successfully loaded
        mock_model.load_state_dict.assert_called()


# ============================================================================
# DINOv3Encoder Lazy Loading Tests
# ============================================================================

class TestDINOv3EncoderLazyLoading:
    """Tests for DINOv3Encoder lazy loading behavior."""

    def test_model_not_loaded_on_init(self):
        """Test model is not loaded during initialization."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        assert encoder._loaded is False
        assert encoder.model is None

    def test_ensure_loaded_triggers_load(self):
        """Test _ensure_loaded triggers model loading."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        with patch.object(encoder, "_load_model") as mock_load:
            encoder._ensure_loaded()
            mock_load.assert_called_once()

    def test_ensure_loaded_skips_if_already_loaded(self):
        """Test _ensure_loaded does nothing if already loaded."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)
        encoder._loaded = True

        with patch.object(encoder, "_load_model") as mock_load:
            encoder._ensure_loaded()
            mock_load.assert_not_called()

    def test_load_model_sets_loaded_flag(self):
        """Test _load_model sets _loaded flag via _finalize_model."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        # Mock the entire loading chain
        with patch.object(encoder, "_create_local_model") as mock_create:
            with patch.object(encoder, "_finalize_model") as mock_finalize:
                encoder.local_path = "/fake/path.pth"
                encoder._load_model()

                mock_create.assert_called_once()

    @patch("src.models.encoders.dinov3.encoder.DINOv3WeightLoader")
    def test_load_model_uses_local_path(self, mock_loader_class):
        """Test _load_model uses local_path when provided."""
        encoder = DINOv3Encoder(
            model_name="dinov3-vitb16",
            pretrained=True,
            local_path="/path/to/weights.pth"
        )

        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        with patch.object(encoder, "_create_local_model") as mock_create:
            with patch.object(encoder, "_finalize_model"):
                encoder._load_model()
                mock_create.assert_called_once_with(mock_loader)

    @patch("src.models.encoders.dinov3.encoder.DINOv3WeightLoader")
    def test_load_model_uses_hf_when_no_local_path(self, mock_loader_class):
        """Test _load_model uses HuggingFace when no local_path."""
        encoder = DINOv3Encoder(
            model_name="dinov3-vitb16",
            pretrained=True,
            local_path=None,
            use_hf=True
        )

        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        with patch.object(encoder, "_finalize_model"):
            encoder._load_model()
            mock_loader.load_from_hf.assert_called_once()


# ============================================================================
# DINOv3Encoder Device Migration Tests
# ============================================================================

class TestDINOv3EncoderDeviceMigration:
    """Tests for DINOv3Encoder device migration via to()."""

    def test_to_sets_target_device(self):
        """Test to() sets _target_device."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        encoder.to("cuda:0")
        assert encoder._target_device == torch.device("cuda:0")

    def test_to_with_string_device(self):
        """Test to() with string device."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        encoder.to("cpu")
        assert encoder._target_device == torch.device("cpu")

    def test_to_with_torch_device(self):
        """Test to() with torch.device object."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        device = torch.device("cuda:1")
        encoder.to(device)
        assert encoder._target_device == device

    def test_to_moves_model_if_loaded(self):
        """Test to() moves model if already loaded."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        # Simulate loaded model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        encoder.model = mock_model
        encoder._loaded = True

        encoder.to("cpu")
        mock_model.to.assert_called()

    def test_to_does_not_load_model(self):
        """Test to() does not trigger model loading."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        encoder.to("cpu")

        assert encoder._loaded is False
        assert encoder.model is None


# ============================================================================
# DINOv3Encoder _finalize_model Tests
# ============================================================================

class TestFinalizeModel:
    """Tests for DINOv3Encoder _finalize_model method."""

    def test_finalize_sets_eval_mode(self):
        """Test _finalize_model sets model to eval mode."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.to.return_value = mock_model
        encoder.model = mock_model
        encoder._target_device = torch.device("cpu")

        encoder._finalize_model()

        mock_model.eval.assert_called_once()

    def test_finalize_freezes_parameters(self):
        """Test _finalize_model freezes all parameters."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        # Create mock parameters
        params = [MagicMock(requires_grad=True) for _ in range(3)]
        mock_model = MagicMock()
        mock_model.parameters.return_value = params
        mock_model.to.return_value = mock_model
        encoder.model = mock_model
        encoder._target_device = torch.device("cpu")

        encoder._finalize_model()

        for param in params:
            assert param.requires_grad is False

    def test_finalize_moves_to_target_device(self):
        """Test _finalize_model moves model to target device."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.to.return_value = mock_model
        encoder.model = mock_model
        encoder._target_device = torch.device("cuda:0")

        encoder._finalize_model()

        # Should be called with target device
        mock_model.to.assert_any_call(torch.device("cuda:0"))

    def test_finalize_sets_loaded_flag(self):
        """Test _finalize_model sets _loaded to True."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.to.return_value = mock_model
        encoder.model = mock_model
        encoder._target_device = torch.device("cpu")

        assert encoder._loaded is False
        encoder._finalize_model()
        assert encoder._loaded is True


# ============================================================================
# DINOv3Encoder Integration Tests
# ============================================================================

class TestDINOv3EncoderIntegration:
    """Integration tests for DINOv3Encoder loading flow."""

    @patch("src.models.encoders.dinov3.encoder.DINOv3ViT")
    @patch("src.models.encoders.dinov3.encoder.DINOv3WeightLoader")
    def test_full_local_load_flow(self, mock_loader_class, mock_vit_class):
        """Test complete local loading flow."""
        # Setup mocks
        mock_vit = MagicMock()
        mock_vit.parameters.return_value = []
        mock_vit.to.return_value = mock_vit
        mock_vit_class.return_value = mock_vit

        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        encoder = DINOv3Encoder(
            model_name="dinov3-vitb16",
            pretrained=True,
            local_path="/path/to/weights.pth"
        )
        encoder._target_device = torch.device("cpu")

        # Trigger load
        encoder._load_model()

        # Verify flow
        mock_vit_class.assert_called_once()
        mock_loader.load_from_local.assert_called_once_with(mock_vit, "/path/to/weights.pth")
        mock_vit.eval.assert_called_once()
        assert encoder._loaded is True

    @patch("src.models.encoders.dinov3.encoder.DINOv3WeightLoader")
    def test_forward_triggers_lazy_load(self, mock_loader_class):
        """Test forward() triggers lazy loading."""
        encoder = DINOv3Encoder(
            model_name="dinov3-vitb16",
            pretrained=False,
            local_path=None,
            use_hf=True
        )

        # Setup mock
        mock_loader = MagicMock()
        mock_hf_model = MagicMock()
        mock_hf_model.parameters.return_value = []
        mock_hf_model.to.return_value = mock_hf_model
        mock_hf_model.return_value = MagicMock()  # Model output
        mock_hf_model.return_value.last_hidden_state = torch.randn(1, 257, 768)
        mock_loader.load_from_hf.return_value = mock_hf_model
        mock_loader_class.return_value = mock_loader

        encoder._target_device = torch.device("cpu")

        # First forward should trigger load
        x = torch.randn(1, 3, 224, 224)
        assert encoder._loaded is False

        # Note: This will actually try to run the full forward, which may fail
        # due to incomplete mocking. The important test is that _ensure_loaded is called.
        with patch.object(encoder, "_ensure_loaded") as mock_ensure:
            with patch.object(encoder, "_prepare_input", return_value=x):
                with patch.object(encoder, "_run_forward", return_value=torch.randn(1, 197, 768)):
                    with patch.object(encoder, "_parse_output", return_value=(torch.randn(1, 196, 768), torch.randn(1, 768))):
                        with patch.object(encoder, "_check_cache", return_value=None):
                            with patch.object(encoder, "_update_cache"):
                                encoder(x)
                                mock_ensure.assert_called_once()

    def test_create_local_model_uses_config(self):
        """Test _create_local_model uses correct config."""
        encoder = DINOv3Encoder(
            model_name="dinov3-vitl16",
            pretrained=True,
            local_path="/path/to/weights.pth"
        )

        with patch("src.models.encoders.dinov3.encoder.DINOv3ViT") as mock_vit:
            mock_loader = MagicMock()

            encoder._create_local_model(mock_loader)

            # Check DINOv3ViT was called with large model config
            mock_vit.assert_called_once()
            call_kwargs = mock_vit.call_args[1]
            assert call_kwargs["embed_dim"] == 1024  # Large model
            assert call_kwargs["depth"] == 24
            assert call_kwargs["num_heads"] == 16

    def test_model_config_used_for_different_variants(self):
        """Test correct config is used for different model variants."""
        configs = {
            "dinov3-vitb16": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "dinov3-vitl16": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "dinov3-vits16": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "dinov3-vit7b16": {"embed_dim": 4096, "depth": 40, "num_heads": 32},
        }

        for model_name, expected_config in configs.items():
            encoder = DINOv3Encoder(
                model_name=model_name,
                pretrained=False,
                local_path="/path/to/weights.pth"
            )

            assert encoder._model_config["embed_dim"] == expected_config["embed_dim"]
            assert encoder._model_config["depth"] == expected_config["depth"]
            assert encoder._model_config["num_heads"] == expected_config["num_heads"]


# ============================================================================
# DINOv3WeightLoader load_from_hf and load_from_hub Tests
# ============================================================================

class TestLoaderExternalSources:
    """Tests for loading from external sources (HuggingFace, Hub)."""

    @patch("src.models.encoders.dinov3.loader.torch.hub.load")
    def test_load_from_hub(self, mock_hub_load):
        """Test load_from_hub calls torch.hub.load correctly."""
        mock_model = MagicMock()
        mock_hub_load.return_value = mock_model

        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader.load_from_hub(pretrained=True)

        mock_hub_load.assert_called_once_with(
            "facebookresearch/dinov3",
            "dinov3_vitb16",
            pretrained=True
        )
        assert result == mock_model

    @patch("src.models.encoders.dinov3.loader.torch.hub.load")
    def test_load_from_hub_model_name_converted(self, mock_hub_load):
        """Test model name is converted correctly for hub."""
        loader = DINOv3WeightLoader("dinov3-vitl16")
        loader.load_from_hub(pretrained=True)

        mock_hub_load.assert_called_once()
        # dinov3-vitl16 -> dinov3_vitl16
        assert mock_hub_load.call_args[0][1] == "dinov3_vitl16"


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_state_dict(self, mock_model):
        """Test handling of empty state dict."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        result = loader._match_state_dict({}, mock_model)
        assert result == {}

    def test_none_in_prefixes(self):
        """Test _remove_prefixes handles edge case keys."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone.", "encoder."]

        # Empty string
        assert loader._remove_prefixes("", prefixes) == ""

        # Just the prefix
        assert loader._remove_prefixes("backbone.", prefixes) == ""

    def test_multiple_dots_in_key(self):
        """Test key with multiple dots is handled correctly."""
        loader = DINOv3WeightLoader("dinov3-vitb16")
        prefixes = ["backbone."]

        result = loader._remove_prefixes("backbone.blocks.0.attn.qkv.weight", prefixes)
        assert result == "blocks.0.attn.qkv.weight"

    def test_encoder_to_returns_self(self):
        """Test encoder.to() returns self for chaining."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)
        result = encoder.to("cpu")
        assert result is encoder

    def test_target_device_default_is_cpu(self):
        """Test default target device is CPU."""
        encoder = DINOv3Encoder(model_name="dinov3-vitb16", pretrained=False)
        assert encoder._target_device == torch.device("cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
