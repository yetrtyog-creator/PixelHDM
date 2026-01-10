"""
PixelHDM-RPEA-DinoV3 Device & Critical Fixes Strict Tests

These tests verify the critical fixes documented in CLAUDE.md:
1. Text Encoder device fix (train.py:71-72)
2. Text Embedding device mismatch fix (step.py:125-133)
3. Pixel-level position encoding fix (core.py:226-237)
4. Sampler CFG pooled_text_embed fix (euler.py, dpm.py, full_heun.py)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Test 1: Text Encoder Device Fix
# =============================================================================

class TestTextEncoderDeviceFix:
    """
    Verify that _load_text_encoder() moves encoder to the correct device.

    Fix location: src/training/train.py:71-72

    Problem: encoder was created but never moved to device, causing CPU inference.
    Solution: encoder = encoder.to(device)
    """

    def test_text_encoder_device_in_code(self):
        """Verify that the .to(device) call exists in the code."""
        import inspect
        from src.training import train

        source = inspect.getsource(train._load_text_encoder)

        # Verify the fix exists: encoder.to(device) or encoder = encoder.to(device)
        assert ".to(device)" in source, (
            "_load_text_encoder must call .to(device) to move encoder to GPU"
        )

    def test_load_text_encoder_returns_none_when_no_encoder_name(self):
        """Verify _load_text_encoder returns None when no encoder name."""
        from src.config import PixelHDMConfig
        from src.training.train import _load_text_encoder

        config = PixelHDMConfig(text_encoder_name=None)
        device = torch.device("cpu")

        result = _load_text_encoder(config, device)
        assert result is None

    def test_qwen3_encoder_lazy_load_device_fix(self):
        """Verify Qwen3TextEncoder correctly handles lazy loading with .to(device).

        This is the CRITICAL fix for CPU usage explosion.

        Problem:
            1. Qwen3TextEncoder uses lazy loading (self.model=None initially)
            2. .to(device) was called before model loaded, so it had no effect
            3. When _load_model() was triggered, model loaded to CPU

        Fix:
            1. Override to() to record target device in self._target_device
            2. In _load_model(), move model to _target_device after loading
        """
        from src.models.encoders.text.encoder import Qwen3TextEncoder

        # Create encoder (lazy load, model not loaded yet)
        encoder = Qwen3TextEncoder(
            model_name="Qwen/Qwen3-0.6B",
            max_length=64,
            freeze=True,
        )

        # Verify model not loaded yet
        assert encoder.model is None
        assert encoder._loaded is False

        # Call .to(device) - this should record the target device
        target_device = torch.device("cpu")  # Use CPU for testing
        encoder = encoder.to(target_device)

        # Verify target device was recorded
        assert encoder._target_device == target_device, (
            "Qwen3TextEncoder.to() must record target device in _target_device"
        )

    def test_qwen3_encoder_to_method_exists(self):
        """Verify Qwen3TextEncoder has overridden to() method."""
        import inspect
        from src.models.encoders.text.encoder import Qwen3TextEncoder

        # Check that to() is defined in Qwen3TextEncoder (not just inherited)
        source = inspect.getsource(Qwen3TextEncoder)
        assert "def to(self" in source, (
            "Qwen3TextEncoder must override to() method to record target device"
        )
        assert "_target_device" in source, (
            "Qwen3TextEncoder.to() must save target device to _target_device"
        )

    def test_qwen3_encoder_load_model_uses_target_device(self):
        """Verify _load_model uses _target_device."""
        import inspect
        from src.models.encoders.text.encoder import Qwen3TextEncoder

        source = inspect.getsource(Qwen3TextEncoder._load_model)
        assert "_target_device" in source, (
            "_load_model must use _target_device to move model after loading"
        )


# =============================================================================
# Test 2: Text Embedding Device Mismatch Fix
# =============================================================================

class TestTextEmbeddingDeviceFix:
    """
    Verify that _encode_captions() returns tensors on the correct device.

    Fix location: src/training/trainer/step.py:125-133

    Problem: Encoded tensors were on CPU, but training expects them on GPU.
    Solution: Move tensors to self.device after encoding.
    """

    def test_prepare_batch_moves_text_embeddings_to_device(self):
        """Verify _prepare_batch moves text embeddings to training device."""
        import inspect
        from src.training.trainer import step

        source = inspect.getsource(step.StepExecutor._prepare_batch)

        # Verify the fix exists - should have .to(self.device) for text_embeddings
        assert ".to(self.device)" in source, (
            "_prepare_batch must move tensors to self.device"
        )

    def test_prepare_batch_handles_text_encoder_path(self):
        """Verify _prepare_batch has text_encoder encoding path."""
        import inspect
        from src.training.trainer import step

        source = inspect.getsource(step.StepExecutor._prepare_batch)

        # Verify the text_encoder path exists
        assert "self.text_encoder" in source, (
            "_prepare_batch must handle text_encoder encoding"
        )
        assert "_encode_captions" in source, (
            "_prepare_batch must call _encode_captions when text_encoder exists"
        )

    def test_encode_captions_method_exists(self):
        """Verify _encode_captions method exists in StepExecutor."""
        from src.training.trainer.step import StepExecutor

        assert hasattr(StepExecutor, "_encode_captions"), (
            "StepExecutor must have _encode_captions method"
        )


# =============================================================================
# Test 3: Pixel-Level Position Encoding Fix
# =============================================================================

class TestPixelPositionEncodingFix:
    """
    Verify that PixelTransformerBlock receives rope_fn and img_positions.

    Fix location: src/models/pixeldit/core.py:226-237

    Problem: Token Compaction attention lacked position information.
    Solution: Create pixel_rope_fn and img_positions, pass to blocks.
    """

    def test_pixel_blocks_receive_position_info_in_code(self):
        """Verify that pixel_blocks are called with rope_fn and img_positions."""
        import inspect
        from src.models.pixelhdm import core

        source = inspect.getsource(core.PixelHDM.forward)

        # Verify position encoding is created
        assert "create_image_positions_batched" in source, (
            "PixelHDM.forward must create img_positions for pixel blocks"
        )

        # Verify pixel_rope_fn is created
        assert "pixel_rope_fn" in source or "_create_rope_fn" in source, (
            "PixelHDM.forward must create pixel_rope_fn for pixel blocks"
        )

        # Verify blocks receive the position info
        assert "rope_fn=" in source and "img_positions=" in source, (
            "pixel_blocks must receive rope_fn and img_positions arguments"
        )

    def test_pixel_rope_uses_text_len_zero(self):
        """Verify that pixel-level rope_fn uses text_len=0 (no text tokens)."""
        import inspect
        from src.models.pixelhdm import core

        source = inspect.getsource(core.PixelHDM.forward)

        # The fix should use text_len=0 for pixel-level RoPE
        assert "text_len=0" in source, (
            "Pixel-level rope_fn must use text_len=0 (no text tokens in pixel stage)"
        )


# =============================================================================
# Test 4: Sampler CFG pooled_text_embed Fix
# =============================================================================

class TestSamplerCFGPooledEmbedFix:
    """
    Verify that Sampler CFG correctly separates cond/uncond pooled_text_embed.

    Fix locations:
    - src/inference/sampler/euler.py
    - src/inference/sampler/dpm.py
    - src/inference/sampler/full_heun.py

    Problem: Uncond branch incorrectly used conditional pooled_text_embed.
    Solution: Pop both pooled_text_embed and null_pooled_text_embed from kwargs.
    """

    def test_euler_sampler_separates_pooled_embeds(self):
        """Verify EulerSampler correctly handles pooled embeddings in CFG."""
        import inspect
        from src.inference.sampler import euler

        source = inspect.getsource(euler.EulerSampler._predict_v)

        # Verify both pooled embeddings are extracted
        assert 'pooled_text_embed = model_kwargs.pop("pooled_text_embed"' in source, (
            "EulerSampler must pop pooled_text_embed from model_kwargs"
        )
        assert 'null_pooled_text_embed = model_kwargs.pop("null_pooled_text_embed"' in source, (
            "EulerSampler must pop null_pooled_text_embed from model_kwargs"
        )

        # Verify uncond branch uses null_pooled_text_embed
        assert "pooled_text_embed=null_pooled_text_embed" in source, (
            "Uncond branch must use null_pooled_text_embed"
        )

    def test_euler_cfg_uses_correct_pooled_embeds(self):
        """Test that CFG branches use the correct pooled embeddings."""
        from src.inference.sampler import EulerSampler

        sampler = EulerSampler(num_steps=10)

        # Create mock model that records its inputs
        call_records = []

        def mock_model(z, t, text_embed=None, pooled_text_embed=None, **kwargs):
            call_records.append({
                "text_embed_is_none": text_embed is None,
                "pooled_text_embed_id": id(pooled_text_embed) if pooled_text_embed is not None else None,
            })
            return torch.zeros_like(z)

        z = torch.randn(2, 4, 4, 3)
        t = torch.tensor(0.5)
        t_next = torch.tensor(0.6)

        # Create distinct embeddings
        text_embeddings = torch.randn(2, 32, 256)
        null_text_embeddings = torch.zeros(2, 32, 256)
        pooled_text_embed = torch.randn(2, 256)
        null_pooled_text_embed = torch.zeros(2, 256)

        pooled_id = id(pooled_text_embed)
        null_pooled_id = id(null_pooled_text_embed)

        # Call with CFG
        sampler.step(
            model=mock_model,
            z=z,
            t=t,
            t_next=t_next,
            text_embeddings=text_embeddings,
            guidance_scale=7.5,  # > 1.0 triggers CFG
            null_text_embeddings=null_text_embeddings,
            pooled_text_embed=pooled_text_embed,
            null_pooled_text_embed=null_pooled_text_embed,
        )

        # Should have 2 calls (uncond + cond)
        assert len(call_records) == 2, f"Expected 2 model calls for CFG, got {len(call_records)}"

        # First call (uncond) should use null_pooled_text_embed
        assert call_records[0]["pooled_text_embed_id"] == null_pooled_id, (
            "Uncond branch must use null_pooled_text_embed"
        )

        # Second call (cond) should use pooled_text_embed
        assert call_records[1]["pooled_text_embed_id"] == pooled_id, (
            "Cond branch must use pooled_text_embed"
        )

    def test_dpm_sampler_separates_pooled_embeds(self):
        """Verify DPMPPSampler correctly handles pooled embeddings in CFG."""
        import inspect
        from src.inference.sampler import dpm

        source = inspect.getsource(dpm)

        # Verify both pooled embeddings are handled
        assert 'null_pooled_text_embed' in source, (
            "DPMPPSampler must handle null_pooled_text_embed"
        )

    def test_full_heun_sampler_separates_pooled_embeds(self):
        """Verify FullHeunSampler correctly handles pooled embeddings in CFG."""
        import inspect
        from src.inference.sampler import full_heun

        source = inspect.getsource(full_heun)

        # Verify both pooled embeddings are handled
        assert 'null_pooled_text_embed' in source, (
            "FullHeunSampler must handle null_pooled_text_embed"
        )


# =============================================================================
# Test 5: V-Prediction Integration
# =============================================================================

class TestVPredictionIntegration:
    """Verify V-Prediction is correctly implemented throughout the pipeline."""

    def test_vloss_uses_v_prediction_formula(self):
        """Verify VLoss uses v_target = x_clean - noise."""
        import inspect
        from src.training.losses import vloss

        source = inspect.getsource(vloss)

        # V-Prediction: v_target = x_clean - noise (not involving division)
        # The old X-Prediction had: v_theta = (x_pred - z_t) / (1 - t)
        assert "1 - t" not in source or "/ (1 - t)" not in source, (
            "V-Prediction should NOT divide by (1-t), that was X-Prediction"
        )

    def test_sampler_uses_v_directly(self):
        """Verify samplers use model output as velocity directly."""
        import inspect
        from src.inference.sampler import euler

        source = inspect.getsource(euler.EulerSampler.step)

        # V-Prediction: z_next = z + dt * v (no conversion)
        assert "z + dt * v" in source or "dt * v" in source, (
            "Euler step should use velocity directly: z + dt * v"
        )


# =============================================================================
# Integration Test
# =============================================================================

class TestDeviceFixesIntegration:
    """Integration tests combining multiple fixes."""

    def test_training_step_with_text_encoder_no_device_error(self):
        """
        Verify that training with text encoder doesn't raise device mismatch errors.

        This tests the combination of:
        - Text encoder device fix
        - Text embedding device fix
        """
        from src.training.trainer import Trainer
        from src.config import TrainingConfig

        # Simple model that accepts text_embed (small for memory efficiency)
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Use small dimensions to avoid memory issues
                self.linear = nn.Linear(32 * 32 * 3, 32 * 32 * 3)

            def forward(self, x, t, text_embed=None, pooled_text_embed=None, **kwargs):
                B = x.shape[0]
                H, W = x.shape[1], x.shape[2]
                x_flat = x.reshape(B, -1)
                # Match input size
                if x_flat.shape[1] != 32 * 32 * 3:
                    # Just return a tensor of same shape for testing
                    return x
                out = self.linear(x_flat)
                return out.reshape(B, H, W, 3)

        # Mock text encoder on CPU
        class MockCPUTextEncoder(nn.Module):
            def forward(self, texts, **kwargs):
                batch_size = len(texts)
                # Return CPU tensors
                return (
                    torch.randn(batch_size, 32, 256),  # text_embeddings
                    torch.ones(batch_size, 32),        # text_mask
                    torch.randn(batch_size, 256),      # pooled
                )

        model = SimpleModel()
        trainer = Trainer(
            model=model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )
        trainer.text_encoder = MockCPUTextEncoder()

        # Create batch with captions (small images)
        batch = {
            "images": torch.randn(2, 32, 32, 3),
            "captions": ["a cat", "a dog"],
        }

        # Should NOT raise RuntimeError about device mismatch
        try:
            metrics = trainer.train_step(batch)
            assert metrics is not None
        except RuntimeError as e:
            if "device" in str(e).lower():
                pytest.fail(f"Device mismatch error: {e}")
            raise


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
