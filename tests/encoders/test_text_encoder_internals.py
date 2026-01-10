"""
Text Encoder Internals Tests

Tests for TokenizerWrapper, Pooling strategies, and Qwen3TextEncoder lazy loading mechanisms.

Test Categories:
    - TokenizerWrapper (8 tests): Lazy loading, tokenize, pad token handling
    - LastTokenPooling (6 tests): Different sequence lengths, edge cases
    - MeanPooling (5 tests): Mean calculation, mask handling
    - get_pooler Factory (3 tests): Factory function validation
    - Qwen3TextEncoder Lazy Loading (8 tests): _load_model, to(), device tracking
    - Qwen3TextEncoder Input Handling (5 tests): _prepare_inputs, _move_to_device

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any

from src.models.encoders.text.tokenizer import TokenizerWrapper
from src.models.encoders.text.pooling import (
    LastTokenPooling,
    MeanPooling,
    get_pooler,
)
from src.models.encoders.text.encoder import Qwen3TextEncoder


# ============================================================================
# TokenizerWrapper Tests
# ============================================================================

class TestTokenizerWrapperInit:
    """Tests for TokenizerWrapper initialization and properties."""

    def test_init_default_params(self):
        """Test TokenizerWrapper initializes with default parameters."""
        wrapper = TokenizerWrapper()

        assert wrapper.model_name == "Qwen/Qwen3-0.6B"
        assert wrapper.max_length == 256
        assert wrapper._tokenizer is None
        assert wrapper._loaded is False

    def test_init_custom_params(self):
        """Test TokenizerWrapper with custom parameters."""
        wrapper = TokenizerWrapper(
            model_name="Qwen/Qwen3-1.7B",
            max_length=512,
        )

        assert wrapper.model_name == "Qwen/Qwen3-1.7B"
        assert wrapper.max_length == 512

    def test_is_loaded_property_false_initially(self):
        """Test is_loaded property returns False before loading."""
        wrapper = TokenizerWrapper()
        assert wrapper.is_loaded is False


class TestTokenizerWrapperLoad:
    """Tests for TokenizerWrapper loading mechanism."""

    def test_load_sets_loaded_flag(self):
        """Test load() sets _loaded flag after successful loading."""
        wrapper = TokenizerWrapper()

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            wrapper.load()

        assert wrapper._loaded is True
        assert wrapper.is_loaded is True

    def test_load_is_idempotent(self):
        """Test load() is idempotent - only loads once."""
        wrapper = TokenizerWrapper()

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer

            wrapper.load()
            wrapper.load()  # Second call should be no-op

            assert mock_auto.from_pretrained.call_count == 1

    def test_load_raises_runtime_error_on_failure(self):
        """Test load() raises RuntimeError on failure."""
        wrapper = TokenizerWrapper()

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("Network error")

            with pytest.raises(RuntimeError, match="Network error"):
                wrapper.load()

    def test_ensure_pad_token_uses_eos_token(self):
        """Test _ensure_pad_token uses eos_token when pad_token is None."""
        wrapper = TokenizerWrapper()

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            wrapper.load()

        assert mock_tokenizer.pad_token == "<eos>"


class TestTokenizerWrapperTokenize:
    """Tests for TokenizerWrapper tokenization."""

    def test_tokenize_triggers_lazy_load(self):
        """Test tokenize() triggers load() if not loaded."""
        wrapper = TokenizerWrapper()

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 256)),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            wrapper.tokenize("Hello world")

        assert wrapper._loaded is True

    def test_tokenize_single_string(self):
        """Test tokenize() with a single string input."""
        wrapper = TokenizerWrapper(max_length=128)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            result = wrapper.tokenize("Hello world")

        # Verify tokenizer was called with list (single string converted)
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == ["Hello world"]

    def test_tokenize_list_of_strings(self):
        """Test tokenize() with a list of strings."""
        wrapper = TokenizerWrapper(max_length=128)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128, dtype=torch.long),
        }

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            result = wrapper.tokenize(["Hello", "World"])

        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == ["Hello", "World"]

    def test_callable_interface(self):
        """Test __call__ is equivalent to tokenize."""
        wrapper = TokenizerWrapper(max_length=128)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        expected_result = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }
        mock_tokenizer.return_value = expected_result

        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            result = wrapper("Test text")

        assert "input_ids" in result
        assert "attention_mask" in result


# ============================================================================
# LastTokenPooling Tests
# ============================================================================

class TestLastTokenPooling:
    """Tests for LastTokenPooling strategy."""

    def test_init(self):
        """Test LastTokenPooling initialization."""
        pooler = LastTokenPooling()
        assert isinstance(pooler, nn.Module)

    def test_forward_uniform_mask(self):
        """Test forward with uniform attention mask (all ones)."""
        pooler = LastTokenPooling()

        B, T, D = 2, 10, 64
        hidden_states = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (B, D)
        # Last token is at index T-1 = 9
        expected = hidden_states[:, T - 1, :]
        assert torch.allclose(output, expected)

    def test_forward_varying_lengths(self):
        """Test forward with varying sequence lengths."""
        pooler = LastTokenPooling()

        B, T, D = 3, 10, 64
        hidden_states = torch.randn(B, T, D)
        # Sequence lengths: 5, 8, 10
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        attention_mask[0, :5] = 1  # Length 5, last valid index = 4
        attention_mask[1, :8] = 1  # Length 8, last valid index = 7
        attention_mask[2, :10] = 1  # Length 10, last valid index = 9

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (B, D)
        assert torch.allclose(output[0], hidden_states[0, 4, :])
        assert torch.allclose(output[1], hidden_states[1, 7, :])
        assert torch.allclose(output[2], hidden_states[2, 9, :])

    def test_forward_single_token(self):
        """Test forward with single token sequences."""
        pooler = LastTokenPooling()

        B, T, D = 2, 10, 32
        hidden_states = torch.randn(B, T, D)
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        attention_mask[:, 0] = 1  # Only first token is valid

        output = pooler(hidden_states, attention_mask)

        # Last valid index = 0
        assert torch.allclose(output, hidden_states[:, 0, :])

    def test_forward_batch_size_one(self):
        """Test forward with batch size 1."""
        pooler = LastTokenPooling()

        B, T, D = 1, 20, 128
        hidden_states = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        attention_mask[0, 15:] = 0  # Length 15

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (1, D)
        assert torch.allclose(output[0], hidden_states[0, 14, :])

    def test_forward_preserves_dtype(self):
        """Test forward preserves input dtype."""
        pooler = LastTokenPooling()

        hidden_states = torch.randn(2, 10, 64, dtype=torch.float16)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        output = pooler(hidden_states, attention_mask)

        assert output.dtype == torch.float16


# ============================================================================
# MeanPooling Tests
# ============================================================================

class TestMeanPooling:
    """Tests for MeanPooling strategy."""

    def test_init(self):
        """Test MeanPooling initialization."""
        pooler = MeanPooling()
        assert isinstance(pooler, nn.Module)

    def test_forward_uniform_mask(self):
        """Test forward with uniform attention mask."""
        pooler = MeanPooling()

        B, T, D = 2, 10, 64
        hidden_states = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (B, D)
        expected = hidden_states.mean(dim=1)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_varying_lengths(self):
        """Test forward with varying sequence lengths."""
        pooler = MeanPooling()

        B, T, D = 2, 8, 32
        hidden_states = torch.randn(B, T, D)
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        attention_mask[0, :4] = 1  # Length 4
        attention_mask[1, :8] = 1  # Length 8

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (B, D)
        # Manual calculation for verification
        expected_0 = hidden_states[0, :4, :].mean(dim=0)
        expected_1 = hidden_states[1, :8, :].mean(dim=0)
        assert torch.allclose(output[0], expected_0, atol=1e-6)
        assert torch.allclose(output[1], expected_1, atol=1e-6)

    def test_forward_single_token(self):
        """Test forward with single token (mean = that token)."""
        pooler = MeanPooling()

        B, T, D = 2, 10, 64
        hidden_states = torch.randn(B, T, D)
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        attention_mask[:, 0] = 1  # Only first token

        output = pooler(hidden_states, attention_mask)

        # Mean of single token = that token
        assert torch.allclose(output, hidden_states[:, 0, :], atol=1e-6)

    def test_forward_handles_empty_mask_gracefully(self):
        """Test forward handles near-empty mask without NaN."""
        pooler = MeanPooling()

        B, T, D = 2, 10, 64
        hidden_states = torch.randn(B, T, D)
        # Near-zero mask (clamped to avoid division by zero)
        attention_mask = torch.zeros(B, T, dtype=torch.float)

        output = pooler(hidden_states, attention_mask)

        # Should not produce NaN due to clamp(min=1e-9)
        assert not torch.isnan(output).any()

    def test_forward_dtype_float32_output(self):
        """Test forward outputs float32 due to internal .float() conversion."""
        pooler = MeanPooling()

        # MeanPooling uses .float() on attention_mask, which promotes to float32
        hidden_states = torch.randn(2, 10, 64, dtype=torch.float16)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        output = pooler(hidden_states, attention_mask)

        # Due to .float() in the implementation, output is float32
        assert output.dtype == torch.float32


# ============================================================================
# get_pooler Factory Tests
# ============================================================================

class TestGetPooler:
    """Tests for get_pooler factory function."""

    def test_get_pooler_last_token(self):
        """Test get_pooler returns LastTokenPooling for 'last_token'."""
        pooler = get_pooler("last_token")
        assert isinstance(pooler, LastTokenPooling)

    def test_get_pooler_mean(self):
        """Test get_pooler returns MeanPooling for 'mean'."""
        pooler = get_pooler("mean")
        assert isinstance(pooler, MeanPooling)

    def test_get_pooler_invalid_strategy_raises(self):
        """Test get_pooler raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="last_token"):
            get_pooler("invalid_strategy")

        with pytest.raises(ValueError, match="mean"):
            get_pooler("unknown")


# ============================================================================
# Qwen3TextEncoder Lazy Loading Tests
# ============================================================================

class TestQwen3TextEncoderLazyLoading:
    """Tests for Qwen3TextEncoder lazy loading mechanism."""

    def test_model_is_none_before_loading(self):
        """Test model attribute is None before lazy loading."""
        encoder = Qwen3TextEncoder()

        assert encoder.model is None
        assert encoder._loaded is False

    def test_load_model_sets_loaded_flag(self):
        """Test _load_model sets _loaded flag."""
        encoder = Qwen3TextEncoder()

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_auto.from_pretrained.return_value = mock_model
                mock_tok.from_pretrained.return_value = mock_tokenizer

                encoder._load_model()

        assert encoder._loaded is True
        assert encoder.model is not None

    def test_load_model_is_idempotent(self):
        """Test _load_model only loads once."""
        encoder = Qwen3TextEncoder()

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_auto.from_pretrained.return_value = mock_model
                mock_tok.from_pretrained.return_value = mock_tokenizer

                encoder._load_model()
                encoder._load_model()

                assert mock_auto.from_pretrained.call_count == 1

    def test_load_model_freezes_params_when_freeze_true(self):
        """Test _load_model freezes parameters when freeze=True."""
        encoder = Qwen3TextEncoder(freeze=True)

        # Create mock parameters
        param1 = MagicMock()
        param1.requires_grad = True
        param2 = MagicMock()
        param2.requires_grad = True

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([param1, param2]))

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_auto.from_pretrained.return_value = mock_model
                mock_tok.from_pretrained.return_value = mock_tokenizer

                encoder._load_model()

        assert param1.requires_grad is False
        assert param2.requires_grad is False

    def test_load_model_raises_on_failure(self):
        """Test _load_model raises RuntimeError on failure."""
        encoder = Qwen3TextEncoder()

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_tok.from_pretrained.return_value = mock_tokenizer
                mock_auto.from_pretrained.side_effect = Exception("Download failed")

                with pytest.raises(RuntimeError, match="Download failed"):
                    encoder._load_model()

    def test_ensure_loaded_triggers_load(self):
        """Test _ensure_loaded triggers _load_model if not loaded."""
        encoder = Qwen3TextEncoder()

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_auto.from_pretrained.return_value = mock_model
                mock_tok.from_pretrained.return_value = mock_tokenizer

                encoder._ensure_loaded()

        assert encoder._loaded is True

    def test_tokenizer_property_triggers_load(self):
        """Test tokenizer property triggers lazy loading."""
        encoder = Qwen3TextEncoder()

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_auto.from_pretrained.return_value = mock_model
                mock_tok.from_pretrained.return_value = mock_tokenizer

                _ = encoder.tokenizer

        assert encoder._loaded is True


# ============================================================================
# Qwen3TextEncoder Device Tracking Tests
# ============================================================================

class TestQwen3TextEncoderDeviceTracking:
    """Tests for Qwen3TextEncoder device tracking mechanism."""

    def test_target_device_is_none_initially(self):
        """Test _target_device is None after initialization."""
        encoder = Qwen3TextEncoder()
        assert encoder._target_device is None

    def test_to_method_records_device_string(self):
        """Test to() method records device when passed as string."""
        encoder = Qwen3TextEncoder()
        encoder.to("cpu")

        assert encoder._target_device is not None
        assert encoder._target_device == torch.device("cpu")

    def test_to_method_records_device_object(self):
        """Test to() method records device when passed as torch.device."""
        encoder = Qwen3TextEncoder()
        device = torch.device("cpu")
        encoder.to(device)

        assert encoder._target_device == device

    def test_load_model_uses_target_device(self):
        """Test _load_model moves model to _target_device."""
        encoder = Qwen3TextEncoder()
        encoder._target_device = torch.device("cpu")

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))
        mock_model.to = MagicMock(return_value=mock_model)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("transformers.AutoModel") as mock_auto:
            with patch("transformers.AutoTokenizer") as mock_tok:
                mock_auto.from_pretrained.return_value = mock_model
                mock_tok.from_pretrained.return_value = mock_tokenizer

                encoder._load_model()

        mock_model.to.assert_called_once_with(torch.device("cpu"))


# ============================================================================
# Qwen3TextEncoder Input Handling Tests
# ============================================================================

class TestQwen3TextEncoderInputHandling:
    """Tests for Qwen3TextEncoder input preparation and device handling."""

    def _create_loaded_encoder(self):
        """Helper to create a loaded encoder with mocked model."""
        encoder = Qwen3TextEncoder()

        # Create mock model with device attribute
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.eval = MagicMock()

        encoder.model = mock_model
        encoder._loaded = True

        return encoder

    def test_prepare_inputs_from_texts(self):
        """Test _prepare_inputs converts texts to tensors."""
        encoder = self._create_loaded_encoder()

        # Mock tokenizer wrapper
        mock_tokens = {
            "input_ids": torch.randint(0, 1000, (2, 77)),
            "attention_mask": torch.ones(2, 77, dtype=torch.long),
        }
        encoder._tokenizer_wrapper._tokenizer = MagicMock(return_value=mock_tokens)
        encoder._tokenizer_wrapper._loaded = True

        input_ids, attention_mask = encoder._prepare_inputs(
            input_ids=None,
            attention_mask=None,
            texts=["Hello", "World"],
        )

        assert input_ids.shape == (2, 77)
        assert attention_mask.shape == (2, 77)

    def test_prepare_inputs_from_tensors(self):
        """Test _prepare_inputs passes through tensors."""
        encoder = self._create_loaded_encoder()

        input_ids = torch.randint(0, 1000, (2, 50))
        attention_mask = torch.ones(2, 50, dtype=torch.long)

        result_ids, result_mask = encoder._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=None,
        )

        assert torch.equal(result_ids, input_ids)
        assert torch.equal(result_mask, attention_mask)

    def test_prepare_inputs_creates_default_mask(self):
        """Test _prepare_inputs creates default attention mask."""
        encoder = self._create_loaded_encoder()

        input_ids = torch.randint(0, 1000, (2, 50))

        result_ids, result_mask = encoder._prepare_inputs(
            input_ids=input_ids,
            attention_mask=None,
            texts=None,
        )

        assert result_mask.shape == input_ids.shape
        assert (result_mask == 1).all()

    def test_prepare_inputs_raises_without_input(self):
        """Test _prepare_inputs raises ValueError without input."""
        encoder = self._create_loaded_encoder()

        with pytest.raises(ValueError, match="input_ids"):
            encoder._prepare_inputs(
                input_ids=None,
                attention_mask=None,
                texts=None,
            )

    def test_move_to_device_with_device_attribute(self):
        """Test _move_to_device uses model.device if available."""
        encoder = self._create_loaded_encoder()
        encoder.model.device = torch.device("cpu")

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        result_ids, result_mask = encoder._move_to_device(input_ids, attention_mask)

        assert result_ids.device.type == "cpu"
        assert result_mask.device.type == "cpu"

    def test_move_to_device_fallback_to_parameters(self):
        """Test _move_to_device falls back to parameters() for device."""
        encoder = self._create_loaded_encoder()

        # Remove device attribute, add parameters
        delattr(encoder.model, "device")
        cpu_param = nn.Parameter(torch.randn(10))  # On CPU by default
        encoder.model.parameters = MagicMock(return_value=iter([cpu_param]))

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        result_ids, result_mask = encoder._move_to_device(input_ids, attention_mask)

        assert result_ids.device.type == "cpu"
        assert result_mask.device.type == "cpu"


# ============================================================================
# Qwen3TextEncoder Format Output Tests
# ============================================================================

class TestQwen3TextEncoderFormatOutput:
    """Tests for Qwen3TextEncoder output formatting."""

    def test_format_output_tuple_with_pooled(self):
        """Test _format_output returns tuple with pooled output."""
        encoder = Qwen3TextEncoder()

        hidden_states = torch.randn(2, 10, 1024)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        result = encoder._format_output(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=False,
            return_pooled=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0].shape == (2, 10, 1024)  # hidden_states
        assert result[1].shape == (2, 10)  # attention_mask
        assert result[2].shape == (2, 1024)  # pooled_output

    def test_format_output_tuple_without_pooled(self):
        """Test _format_output returns tuple without pooled output."""
        encoder = Qwen3TextEncoder()

        hidden_states = torch.randn(2, 10, 1024)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        result = encoder._format_output(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=False,
            return_pooled=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 10, 1024)
        assert result[1].shape == (2, 10)

    def test_format_output_dict_with_pooled(self):
        """Test _format_output returns dict with pooled output."""
        encoder = Qwen3TextEncoder(use_pooler=True)

        hidden_states = torch.randn(2, 10, 1024)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        result = encoder._format_output(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
            return_pooled=True,
        )

        assert isinstance(result, dict)
        assert "hidden_states" in result
        assert "attention_mask" in result
        assert "pooled_output" in result
        assert result["pooled_output"].shape == (2, 1024)

    def test_format_output_dict_use_pooler_flag(self):
        """Test _format_output respects use_pooler flag."""
        encoder = Qwen3TextEncoder(use_pooler=True)

        hidden_states = torch.randn(2, 10, 1024)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        result = encoder._format_output(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
            return_pooled=False,  # False but use_pooler=True
        )

        # Should still have pooled_output due to use_pooler=True
        assert "pooled_output" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
