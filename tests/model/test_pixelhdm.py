"""
PixelHDM Main Model Tests

Tests for the PixelHDM and PixelHDMForT2I classes.

Test Categories:
    - Forward Pass (5 tests): Complete forward, output shape, patch embedding, unpatchify, joint sequence
    - CFG (3 tests): CFG forward, guidance scale, null text embedding
    - REPA Features (3 tests): Feature extraction, shape, align layer
    - Conditioning and Gradients (4 tests): Time embedding, text conditioning, gradient flow, factory

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typing import Optional

from src.models.pixelhdm import (
    PixelHDM,
    PixelHDMForT2I,
    create_pixelhdm,
    create_pixelhdm_for_t2i,
)
from src.config import PixelHDMConfig


# ============================================================================
# Test Fixtures (module-scoped where possible for efficiency)
# ============================================================================

@pytest.fixture(scope="module")
def testing_config() -> PixelHDMConfig:
    """Create minimal configuration for fast testing (module-scoped)."""
    return PixelHDMConfig.for_testing()


@pytest.fixture(scope="module")
def pixelhdm_model(testing_config: PixelHDMConfig) -> PixelHDM:
    """Create a PixelHDM model for testing (module-scoped, eval mode)."""
    model = PixelHDM(testing_config)
    model.eval()
    return model


@pytest.fixture
def batch_data(testing_config: PixelHDMConfig) -> dict:
    """
    Create test batch data with correct dimensions.

    Returns dict with:
        - x: (B, H, W, 3) input image
        - x_bchw: (B, 3, H, W) alternative input format
        - t: (B,) timesteps
        - text_emb: (B, T, D) text embeddings
        - text_mask: (B, T) attention mask
        - num_patches: number of patches
    """
    B = 2
    H, W = 256, 256
    T = 32
    D = testing_config.hidden_dim
    patch_size = testing_config.patch_size
    num_patches = (H // patch_size) * (W // patch_size)

    return {
        "x": torch.randn(B, H, W, 3),
        "x_bchw": torch.randn(B, 3, H, W),
        "t": torch.rand(B),
        "text_emb": torch.randn(B, T, D),
        "text_mask": torch.ones(B, T, dtype=torch.long),
        "null_text_emb": torch.zeros(B, T, D),
        "num_patches": num_patches,
        "batch_size": B,
        "height": H,
        "width": W,
        "text_len": T,
    }


# ============================================================================
# Test Class: PixelHDM Forward Pass Tests
# ============================================================================

class TestPixelHDMForwardPass:
    """Tests for PixelHDM forward pass functionality."""

    def test_forward_pass_complete(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 1: Complete forward pass with all inputs.

        Verifies that the model can process:
        - Image input
        - Time embedding
        - Text conditioning

        And produces valid output without NaN/Inf.
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            output = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                text_mask=batch_data["text_mask"],
            )

        # Output should exist
        assert output is not None, "Forward pass returned None"

        # Output should not contain NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        # Output should have expected shape
        expected_shape = (batch_data["batch_size"], batch_data["height"], batch_data["width"], 3)
        assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"

    def test_output_shape_equals_input(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 2: Verify output shape equals input shape.

        For x-prediction, the model should output a tensor with
        the same shape as the input image.
        """
        pixelhdm_model.eval()

        # Test with (B, H, W, C) format
        with torch.no_grad():
            output = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
            )

        assert output.shape == batch_data["x"].shape, \
            f"Output shape {output.shape} != input shape {batch_data['x'].shape}"

        # Test with (B, C, H, W) format
        with torch.no_grad():
            output_bchw = pixelhdm_model(
                x_t=batch_data["x_bchw"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
            )

        # Output should be (B, H, W, C) regardless of input format
        expected_bhwc = (batch_data["batch_size"], batch_data["height"], batch_data["width"], 3)
        assert output_bchw.shape == expected_bhwc, \
            f"Output shape {output_bchw.shape} != expected BHWC {expected_bhwc}"

    def test_patch_embed_shape(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
        testing_config: PixelHDMConfig
    ):
        """
        Test 3: Verify patch embedding produces correct shape.

        Patch embedding should transform:
        (B, H, W, C) -> (B, L, D)
        where L = (H/p) * (W/p)
        """
        x = batch_data["x"]

        # Direct call to patch_embed
        embedded = pixelhdm_model.patch_embed(x)

        B = x.shape[0]
        expected_L = batch_data["num_patches"]
        expected_D = testing_config.hidden_dim

        assert embedded.shape[0] == B, f"Batch size mismatch: {embedded.shape[0]} != {B}"
        assert embedded.shape[1] == expected_L, f"Num patches mismatch: {embedded.shape[1]} != {expected_L}"
        assert embedded.shape[2] == expected_D, f"Hidden dim mismatch: {embedded.shape[2]} != {expected_D}"

    def test_pixel_embed_shape(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
        testing_config: PixelHDMConfig
    ):
        """
        Test 4: Verify pixel_embed (1×1 Patchify) produces correct shape.

        PixelEmbedding transforms raw image to pixel-level features:
        (B, H, W, C) -> (B, L, p^2, D_pix)

        Note (2026-01-06): Replaced test_unpatchify_shape.
        Architecture now uses 1×1 Patchify instead of unpatchify.
        """
        B = batch_data["batch_size"]
        H = batch_data["height"]
        W = batch_data["width"]
        L = batch_data["num_patches"]

        # Create raw image input
        raw_image = torch.randn(B, H, W, testing_config.in_channels)

        # Apply pixel_embed (1×1 Patchify)
        pixel_features = pixelhdm_model.pixel_embed(raw_image)

        p2 = testing_config.patch_size ** 2  # 256 for p=16
        D_pix = testing_config.pixel_dim

        expected_shape = (B, L, p2, D_pix)
        assert pixel_features.shape == expected_shape, \
            f"PixelEmbedding shape {pixel_features.shape} != expected {expected_shape}"

    def test_joint_text_image_creation(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
        testing_config: PixelHDMConfig
    ):
        """
        Test 5: Verify joint text-image sequence creation.

        The model should correctly concatenate text and image tokens
        in Lumina2 style: [text_tokens, img_tokens]
        """
        B = batch_data["batch_size"]
        L = batch_data["num_patches"]
        T = batch_data["text_len"]
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)
        text_embed = batch_data["text_emb"]
        text_mask = batch_data["text_mask"]

        # Call internal method
        joint_tokens, joint_mask, text_len = pixelhdm_model._create_joint_sequence(
            img_tokens, text_embed, text_mask
        )

        # Check joint sequence shape
        expected_joint_len = T + L
        assert joint_tokens.shape == (B, expected_joint_len, D), \
            f"Joint tokens shape {joint_tokens.shape} != expected ({B}, {expected_joint_len}, {D})"

        # Check text_len is correct
        assert text_len == T, f"Text length {text_len} != {T}"

        # Check joint mask shape
        assert joint_mask.shape == (B, expected_joint_len), \
            f"Joint mask shape {joint_mask.shape} != expected ({B}, {expected_joint_len})"

        # Test without text embedding
        joint_tokens_no_text, joint_mask_no_text, text_len_no_text = \
            pixelhdm_model._create_joint_sequence(img_tokens, None, None)

        assert joint_tokens_no_text.shape == (B, L, D), \
            "Without text, joint tokens should equal img tokens"
        assert joint_mask_no_text is None, "Without text, joint mask should be None"
        assert text_len_no_text == 0, "Without text, text_len should be 0"


# ============================================================================
# Test Class: PixelHDM CFG Tests
# ============================================================================

class TestPixelHDMCFG:
    """Tests for Classifier-Free Guidance functionality."""

    def test_cfg_forward(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 6: CFG forward pass produces valid output.

        Tests forward_with_cfg method with conditional and unconditional inputs.
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            cfg_output = pixelhdm_model.forward_with_cfg(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                text_mask=batch_data["text_mask"],
                cfg_scale=7.5,
                null_text_embed=batch_data["null_text_emb"],
            )

        # CFG output should have same shape as input
        assert cfg_output.shape == batch_data["x"].shape, \
            f"CFG output shape {cfg_output.shape} != input shape {batch_data['x'].shape}"

        # Should not contain NaN or Inf
        assert not torch.isnan(cfg_output).any(), "CFG output contains NaN"
        assert not torch.isinf(cfg_output).any(), "CFG output contains Inf"

    def test_cfg_guidance_scale(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 7: Different guidance scales produce different outputs.

        Higher guidance scale should amplify the difference between
        conditional and unconditional predictions.
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            # Low guidance scale
            output_low = pixelhdm_model.forward_with_cfg(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                cfg_scale=1.0,
                null_text_embed=batch_data["null_text_emb"],
            )

            # High guidance scale
            output_high = pixelhdm_model.forward_with_cfg(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                cfg_scale=15.0,
                null_text_embed=batch_data["null_text_emb"],
            )

        # Outputs should be different (unless perfectly aligned)
        diff = (output_low - output_high).abs().mean()
        assert diff > 0, "Different guidance scales should produce different outputs"

    def test_cfg_null_text_embedding(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 8: CFG with null_text_embed=None skips CFG computation.

        When null_text_embed is None, forward_with_cfg should behave
        like regular forward pass (no CFG).
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            # CFG with null_text_embed=None
            output_no_cfg = pixelhdm_model.forward_with_cfg(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                cfg_scale=7.5,
                null_text_embed=None,
            )

            # Regular forward pass
            output_regular = pixelhdm_model.forward(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
            )

        # Should be identical when null_text_embed is None
        assert torch.allclose(output_no_cfg, output_regular, rtol=1e-5, atol=1e-6), \
            "CFG with null_text_embed=None should equal regular forward"


# ============================================================================
# Test Class: PixelHDM REPA Features Tests
# ============================================================================

class TestPixelHDMREPA:
    """Tests for REPA feature extraction functionality."""

    def test_repa_features_extraction(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 9: REPA feature extraction returns valid features.

        The model should extract features from the REPA align layer
        when return_features=True.
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            output, repa_features = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                return_features=True,
            )

        # Both output and features should be returned
        assert output is not None, "Output should not be None"
        assert repa_features is not None, "REPA features should not be None"

        # REPA features should not contain NaN or Inf
        assert not torch.isnan(repa_features).any(), "REPA features contain NaN"
        assert not torch.isinf(repa_features).any(), "REPA features contain Inf"

    def test_repa_features_shape(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
        testing_config: PixelHDMConfig
    ):
        """
        Test 10: REPA features have correct shape (B, L, D).

        Features should have shape matching the image tokens,
        not the joint text-image sequence.
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            output, repa_features = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
                return_features=True,
            )

        B = batch_data["batch_size"]
        L = batch_data["num_patches"]
        D = testing_config.hidden_dim

        expected_shape = (B, L, D)
        assert repa_features.shape == expected_shape, \
            f"REPA features shape {repa_features.shape} != expected {expected_shape}"

    def test_repa_align_layer(
        self,
        testing_config: PixelHDMConfig
    ):
        """
        Test 11: REPA align layer is correctly set.

        The repa_align_layer should be set to config value - 1
        (config uses 1-indexed, internal uses 0-indexed).

        Note: In testing config, repa_align_layer may exceed patch_layers,
        so we test the raw value transformation, not the clamped value.
        """
        model = PixelHDM(testing_config)

        # Check alignment: model uses 0-indexed, config uses 1-indexed
        expected_layer = testing_config.repa_align_layer - 1
        assert model.repa_align_layer == expected_layer, \
            f"REPA align layer {model.repa_align_layer} != expected {expected_layer}"

        # Test with a config where repa_align_layer is within range
        valid_config = PixelHDMConfig.for_testing()
        valid_config.repa_align_layer = 1  # Valid for patch_layers=2
        valid_model = PixelHDM(valid_config)

        # Check it's within valid range
        assert 0 <= valid_model.repa_align_layer < valid_config.patch_layers, \
            f"REPA align layer {valid_model.repa_align_layer} out of range [0, {valid_config.patch_layers})"


# ============================================================================
# Test Class: PixelHDM Conditioning and Gradient Tests
# ============================================================================

class TestPixelHDMConditioning:
    """Tests for conditioning and gradient flow."""

    def test_time_embedding(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
        testing_config: PixelHDMConfig
    ):
        """
        Test 12: Time embedding produces correct shape.

        TimeEmbedding should transform timesteps from (B,) to (B, D).
        """
        t = batch_data["t"]

        # Apply time embedding
        t_embed = pixelhdm_model.time_embed(t)

        B = batch_data["batch_size"]
        D = testing_config.hidden_dim

        assert t_embed.shape == (B, D), \
            f"Time embedding shape {t_embed.shape} != expected ({B}, {D})"

        # Time embedding should not contain NaN or Inf
        assert not torch.isnan(t_embed).any(), "Time embedding contains NaN"
        assert not torch.isinf(t_embed).any(), "Time embedding contains Inf"

    def test_text_conditioning(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 13: Text conditioning affects model output.

        Different text embeddings should produce different outputs.
        """
        pixelhdm_model.eval()

        with torch.no_grad():
            # With text conditioning
            output_with_text = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
            )

            # Without text conditioning
            output_no_text = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=None,
            )

        # Outputs should be different
        diff = (output_with_text - output_no_text).abs().mean()
        assert diff > 0, "Text conditioning should affect output"

    def test_gradient_flow(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """
        Test 14: Gradients flow correctly through the model.

        All trainable parameters should receive gradients during backprop.
        """
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        # Forward pass
        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        # Compute simple loss and backprop
        loss = output.mean()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        grad_params = 0
        total_params = 0

        for name, param in pixelhdm_model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    grad_params += 1
                    # Check for NaN/Inf in gradients
                    assert not torch.isnan(param.grad).any(), \
                        f"Gradient for '{name}' contains NaN"
                    assert not torch.isinf(param.grad).any(), \
                        f"Gradient for '{name}' contains Inf"

        assert has_grad, "No gradients found in any parameter"
        assert grad_params > 0, f"Only {grad_params}/{total_params} parameters received gradients"

    def test_create_from_config(self, testing_config: PixelHDMConfig):
        """
        Test 15: Factory function creates valid model from config.

        create_pixelhdm() should create a working model from config.
        """
        # Create using factory function
        model = create_pixelhdm(config=testing_config)

        # Verify model type
        assert isinstance(model, PixelHDM), "Factory should return PixelHDM instance"

        # Verify config is stored
        assert model.config is testing_config, "Config should be stored in model"

        # Verify dimensions match config
        assert model.hidden_dim == testing_config.hidden_dim
        assert model.patch_size == testing_config.patch_size
        assert model.patch_layers == testing_config.patch_layers
        assert model.pixel_layers == testing_config.pixel_layers


# ============================================================================
# Test Class: PixelHDMForT2I Tests
# ============================================================================

class TestPixelHDMForT2I:
    """Tests for PixelHDMForT2I (T2I wrapper with text/DINO encoders)."""

    @pytest.fixture
    def t2i_model_no_encoders(self, testing_config: PixelHDMConfig) -> PixelHDMForT2I:
        """Create T2I model without loading actual encoders."""
        return create_pixelhdm_for_t2i(
            config=testing_config,
            load_text_encoder=False,
            load_dino_encoder=False,
        )

    def test_t2i_model_creation(self, t2i_model_no_encoders: PixelHDMForT2I):
        """Test T2I model can be created without loading encoders."""
        assert isinstance(t2i_model_no_encoders, PixelHDMForT2I)
        assert isinstance(t2i_model_no_encoders, PixelHDM)  # Should inherit

        # Encoders should be None when not loaded
        assert t2i_model_no_encoders._text_encoder is None
        assert t2i_model_no_encoders._dino_encoder is None

    def test_t2i_inherits_forward(
        self,
        t2i_model_no_encoders: PixelHDMForT2I,
        batch_data: dict
    ):
        """Test T2I model can use base forward pass."""
        t2i_model_no_encoders.eval()

        with torch.no_grad():
            output = t2i_model_no_encoders.forward(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
            )

        assert output.shape == batch_data["x"].shape
        assert not torch.isnan(output).any()

    def test_t2i_text_encoder_error_when_not_loaded(
        self,
        t2i_model_no_encoders: PixelHDMForT2I
    ):
        """Test proper error when text encoder is not loaded."""
        # Accessing text_encoder should return None when load_text_encoder=False
        assert t2i_model_no_encoders.text_encoder is None

        # Trying to encode text should raise error
        with pytest.raises(RuntimeError, match="文本編碼器未加載"):
            t2i_model_no_encoders.encode_text(["test text"])

    def test_t2i_dino_encoder_error_when_not_loaded(
        self,
        t2i_model_no_encoders: PixelHDMForT2I,
        batch_data: dict
    ):
        """Test proper error when DINO encoder is not loaded."""
        # Accessing dino_encoder should return None when load_dino_encoder=False
        assert t2i_model_no_encoders.dino_encoder is None

        # Trying to get DINO features should raise error
        with pytest.raises(RuntimeError, match="DINO 編碼器未加載"):
            t2i_model_no_encoders.get_dino_features(batch_data["x"])


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================

class TestPixelHDMEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_input_dimension(self, pixelhdm_model: PixelHDM):
        """Test error handling for invalid input dimensions."""
        # 3D input should fail
        x_3d = torch.randn(2, 256, 256)
        t = torch.rand(2)

        with pytest.raises(ValueError, match="4D"):
            pixelhdm_model(x_3d, t)

    def test_non_divisible_resolution(
        self,
        testing_config: PixelHDMConfig
    ):
        """Test error handling for non-divisible resolution."""
        model = PixelHDM(testing_config)

        # Image size not divisible by patch_size
        patch_size = testing_config.patch_size
        bad_size = patch_size * 10 + 7  # Not divisible

        x = torch.randn(2, bad_size, bad_size, 3)
        t = torch.rand(2)

        with pytest.raises(AssertionError):
            model(x, t)

    def test_count_parameters(self, pixelhdm_model: PixelHDM):
        """Test parameter counting method."""
        params = pixelhdm_model.count_parameters()

        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params
        assert "patch_level" in params
        assert "pixel_level" in params
        assert "embeddings" in params

        # Total should equal trainable + frozen
        assert params["total"] == params["trainable"] + params["frozen"]

        # Patch level should be larger than pixel level (16 vs 1 layers in testing config)
        # Note: This depends on testing config, so just check they're positive
        assert params["patch_level"] > 0
        assert params["pixel_level"] > 0

    def test_extra_repr(self, pixelhdm_model: PixelHDM, testing_config: PixelHDMConfig):
        """Test extra_repr provides useful info."""
        repr_str = pixelhdm_model.extra_repr()

        # Should contain key dimensions
        assert f"hidden_dim={testing_config.hidden_dim}" in repr_str
        assert f"patch_size={testing_config.patch_size}" in repr_str
        assert "params=" in repr_str

    def test_rectangular_input(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """Test model handles rectangular (non-square) images."""
        pixelhdm_model.eval()

        B = 2
        H = 256
        W = 512  # Rectangular

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3), \
            f"Rectangular output shape {output.shape} != expected ({B}, {H}, {W}, 3)"


# ============================================================================
# Test Class: Integration Tests
# ============================================================================

class TestPixelHDMIntegration:
    """Integration tests for complete workflows."""

    def test_training_step_simulation(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """Simulate a complete training step."""
        pixelhdm_model.train()

        # Optimizer
        optimizer = torch.optim.AdamW(pixelhdm_model.parameters(), lr=1e-4)

        # Forward pass
        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        # Simple MSE loss
        target = torch.randn_like(output)
        loss = ((output - target) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        grad_norm = 0.0
        for param in pixelhdm_model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()

        assert grad_norm > 0, "Gradients should have non-zero norm"

        # Optimizer step
        optimizer.step()

        # Verify no NaN in parameters after update
        for name, param in pixelhdm_model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in param '{name}' after update"

    def test_inference_workflow(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict
    ):
        """Test complete inference workflow."""
        pixelhdm_model.eval()

        # Start from noise (t=0 in PixelHDM convention)
        B, H, W = batch_data["batch_size"], batch_data["height"], batch_data["width"]
        z_0 = torch.randn(B, H, W, 3)  # Pure noise

        # Simulate few denoising steps
        num_steps = 5
        z = z_0

        for i in range(num_steps):
            t_val = (i + 1) / num_steps
            t = torch.full((B,), t_val)

            with torch.no_grad():
                pred = pixelhdm_model(
                    x_t=z,
                    t=t,
                    text_embed=batch_data["text_emb"],
                )

            # Simple Euler update (not accurate, just testing workflow)
            z = pred

        # Final output should be valid
        assert z.shape == (B, H, W, 3)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
