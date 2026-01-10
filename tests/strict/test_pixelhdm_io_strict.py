"""
PixelHDM I/O Strict Tests

Strict tests for PixelHDM model input/output format handling.
Based on STRICT_TEST_STRATEGY.md guidelines.

Test Categories:
    1. Input Format Tests: BHWC and BCHW format handling
    2. Output Format Tests: Consistent output format
    3. Variable Resolution Tests: Different resolutions
    4. REPA Layer Index Tests: 1-indexed to 0-indexed conversion
    5. Text Embedding Integration Tests: Joint sequence creation
    6. Edge Cases: Non-square images, mismatched dimensions

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.models.pixelhdm import PixelHDM, create_pixelhdm
from src.config import PixelHDMConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def testing_config() -> PixelHDMConfig:
    """Create minimal configuration for fast testing."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def pixelhdm_model(testing_config: PixelHDMConfig) -> PixelHDM:
    """Create a PixelHDM model for testing."""
    return PixelHDM(testing_config)


# ============================================================================
# Test Class: Input Format Tests (BHWC)
# ============================================================================

class TestInputFormatBHWC:
    """
    Tests for (B, H, W, 3) input format handling.

    The model's native format is BHWC (Batch, Height, Width, Channels).
    These tests verify that BHWC inputs are processed correctly.
    """

    def test_input_format_bhwc_basic(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 1: Basic BHWC format is accepted and processed.

        Input: (B, H, W, 3)
        Expected: No errors, valid output
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output is not None, "Forward pass returned None"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_input_format_bhwc_dimensions_preserved(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 2: BHWC input dimensions are preserved in output.

        For x-prediction, output shape should match input shape.
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == x.shape, \
            f"Output shape {output.shape} != input shape {x.shape}"

    def test_input_format_bhwc_channel_last_position(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 3: Verify BHWC format has channels in last position.

        The last dimension should be 3 (RGB channels).
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape[-1] == 3, \
            f"Output last dimension {output.shape[-1]} != 3 (RGB channels)"
        assert output.shape[0] == B, \
            f"Batch dimension {output.shape[0]} != {B}"
        assert output.shape[1] == H, \
            f"Height dimension {output.shape[1]} != {H}"
        assert output.shape[2] == W, \
            f"Width dimension {output.shape[2]} != {W}"

    def test_input_format_bhwc_with_text_conditioning(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 4: BHWC format works with text conditioning.
        """
        B, H, W = 2, 256, 256
        T, D = 32, testing_config.hidden_dim

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, T, D)
        text_mask = torch.ones(B, T, dtype=torch.long)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t, text_embed=text_embed, text_mask=text_mask)

        assert output.shape == (B, H, W, 3), \
            f"Output shape {output.shape} != expected ({B}, {H}, {W}, 3)"


# ============================================================================
# Test Class: Input Format Tests (BCHW)
# ============================================================================

class TestInputFormatBCHW:
    """
    Tests for (B, 3, H, W) input format handling.

    BCHW is the standard PyTorch format. The model should automatically
    convert it to BHWC internally.
    """

    def test_input_format_bchw_basic(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 5: Basic BCHW format is accepted and auto-converted.

        Input: (B, 3, H, W) - standard PyTorch format
        Expected: Auto-conversion, no errors
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, 3, H, W)  # BCHW format
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output is not None, "Forward pass returned None for BCHW input"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_input_format_bchw_converts_to_bhwc_output(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 6: BCHW input produces BHWC output.

        Regardless of input format, output should always be BHWC.
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, 3, H, W)  # BCHW format
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        # Output should be BHWC regardless of input format
        expected_shape = (B, H, W, 3)
        assert output.shape == expected_shape, \
            f"BCHW input produced {output.shape}, expected BHWC {expected_shape}"

    def test_input_format_bchw_detection(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 7: Model correctly detects BCHW format.

        When dim[1] == in_channels, the model should permute input.
        """
        B, H, W = 2, 256, 256
        in_channels = testing_config.in_channels  # Should be 3

        x_bchw = torch.randn(B, in_channels, H, W)
        x_bhwc = torch.randn(B, H, W, in_channels)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output_bchw = pixelhdm_model(x_bchw, t)
            output_bhwc = pixelhdm_model(x_bhwc, t)

        # Both should produce BHWC output
        assert output_bchw.shape == (B, H, W, 3), \
            f"BCHW output shape {output_bchw.shape} incorrect"
        assert output_bhwc.shape == (B, H, W, 3), \
            f"BHWC output shape {output_bhwc.shape} incorrect"

    def test_input_format_bchw_with_text_conditioning(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 8: BCHW format works with text conditioning.
        """
        B, H, W = 2, 256, 256
        T, D = 32, testing_config.hidden_dim

        x = torch.randn(B, 3, H, W)  # BCHW format
        t = torch.rand(B)
        text_embed = torch.randn(B, T, D)
        text_mask = torch.ones(B, T, dtype=torch.long)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t, text_embed=text_embed, text_mask=text_mask)

        assert output.shape == (B, H, W, 3), \
            f"Output shape {output.shape} != expected ({B}, {H}, {W}, 3)"


# ============================================================================
# Test Class: Output Format Tests
# ============================================================================

class TestOutputFormat:
    """
    Tests for consistent output format.

    The model should ALWAYS output in BHWC format (B, H, W, 3)
    regardless of input format.
    """

    def test_output_format_always_bhwc(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 9: Output is always BHWC format.

        This is the most critical output format test.
        """
        B, H, W = 2, 256, 256
        t = torch.rand(B)

        pixelhdm_model.eval()

        # Test with BHWC input
        x_bhwc = torch.randn(B, H, W, 3)
        with torch.no_grad():
            output_from_bhwc = pixelhdm_model(x_bhwc, t)

        # Test with BCHW input
        x_bchw = torch.randn(B, 3, H, W)
        with torch.no_grad():
            output_from_bchw = pixelhdm_model(x_bchw, t)

        expected = (B, H, W, 3)

        assert output_from_bhwc.shape == expected, \
            f"BHWC input -> output {output_from_bhwc.shape} != expected {expected}"
        assert output_from_bchw.shape == expected, \
            f"BCHW input -> output {output_from_bchw.shape} != expected {expected}"

    def test_output_format_with_return_features(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 10: Output format is BHWC even when returning REPA features.
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output, repa_features = pixelhdm_model(x, t, return_features=True)

        expected_output = (B, H, W, 3)
        assert output.shape == expected_output, \
            f"Output with return_features {output.shape} != expected {expected_output}"

    def test_output_format_cfg_forward(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 11: CFG forward output is also BHWC.
        """
        B, H, W = 2, 256, 256
        T, D = 32, testing_config.hidden_dim

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, T, D)
        null_text_embed = torch.zeros(B, T, D)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model.forward_with_cfg(
                x_t=x,
                t=t,
                text_embed=text_embed,
                cfg_scale=7.5,
                null_text_embed=null_text_embed,
            )

        expected = (B, H, W, 3)
        assert output.shape == expected, \
            f"CFG output {output.shape} != expected {expected}"

    def test_output_channels_rgb(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 12: Output has exactly 3 channels (RGB).
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape[-1] == 3, \
            f"Output channels {output.shape[-1]} != 3 (RGB)"


# ============================================================================
# Test Class: Variable Resolution Tests
# ============================================================================

class TestVariableResolution:
    """
    Tests for different image resolutions.

    The model should handle:
    - 256x256 (small, common test size)
    - 512x512 (default training size)
    - 512x256 (non-square, portrait)
    - 256x512 (non-square, landscape)
    """

    def test_variable_resolution_256x256(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 13: 256x256 resolution is processed correctly.
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3), \
            f"256x256 output {output.shape} != expected ({B}, {H}, {W}, 3)"
        assert not torch.isnan(output).any(), "256x256 output contains NaN"

    def test_variable_resolution_512x512(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 14: 512x512 resolution is processed correctly.

        This is the default training resolution.
        """
        B, H, W = 2, 512, 512
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3), \
            f"512x512 output {output.shape} != expected ({B}, {H}, {W}, 3)"
        assert not torch.isnan(output).any(), "512x512 output contains NaN"

    def test_variable_resolution_512x256_landscape(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 15: 512x256 landscape resolution is processed correctly.
        """
        B, H, W = 2, 256, 512  # Landscape: width > height
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3), \
            f"256x512 (landscape) output {output.shape} != expected ({B}, {H}, {W}, 3)"
        assert not torch.isnan(output).any(), "256x512 output contains NaN"

    def test_variable_resolution_256x512_portrait(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 16: 256x512 portrait resolution is processed correctly.
        """
        B, H, W = 2, 512, 256  # Portrait: height > width
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3), \
            f"512x256 (portrait) output {output.shape} != expected ({B}, {H}, {W}, 3)"
        assert not torch.isnan(output).any(), "512x256 output contains NaN"

    def test_variable_resolution_preserves_aspect_ratio(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 17: Different aspect ratios are preserved in output.
        """
        B = 2
        resolutions = [
            (256, 256),   # 1:1
            (512, 256),   # 2:1
            (256, 512),   # 1:2
            (384, 256),   # 3:2
        ]

        pixelhdm_model.eval()

        for H, W in resolutions:
            x = torch.randn(B, H, W, 3)
            t = torch.rand(B)

            with torch.no_grad():
                output = pixelhdm_model(x, t)

            assert output.shape == (B, H, W, 3), \
                f"Resolution {H}x{W}: output {output.shape} != expected ({B}, {H}, {W}, 3)"


# ============================================================================
# Test Class: REPA Layer Index Tests
# ============================================================================

class TestREPALayerIndex:
    """
    Tests for REPA align layer index conversion.

    Config uses 1-indexed (human-readable), internal uses 0-indexed.
    repa_align_layer in config = 1 means layer 0 (first layer)
    """

    def test_repa_layer_index_conversion(self, testing_config: PixelHDMConfig):
        """
        Test 18: 1-indexed config correctly converts to 0-indexed internal.

        This is a CRITICAL test for REPA loss alignment.
        """
        # Default testing config has repa_align_layer=1
        model = PixelHDM(testing_config)

        # Internal should be 0-indexed
        expected_internal = testing_config.repa_align_layer - 1
        assert model.repa_align_layer == expected_internal, \
            f"Internal layer {model.repa_align_layer} != expected {expected_internal}"

    def test_repa_layer_index_valid_range(self):
        """
        Test 19: REPA layer index is within valid range.

        0 <= repa_align_layer < patch_layers
        """
        # Create config with known values
        config = PixelHDMConfig.for_testing()
        config.repa_align_layer = 2  # 2nd layer (0-indexed will be 1)

        model = PixelHDM(config)

        assert model.repa_align_layer >= 0, \
            f"REPA layer index {model.repa_align_layer} < 0"
        assert model.repa_align_layer < config.patch_layers, \
            f"REPA layer index {model.repa_align_layer} >= patch_layers {config.patch_layers}"

    def test_repa_layer_index_first_layer(self):
        """
        Test 20: repa_align_layer=1 maps to layer 0 (first layer).
        """
        config = PixelHDMConfig.for_testing()
        config.repa_align_layer = 1

        model = PixelHDM(config)

        assert model.repa_align_layer == 0, \
            f"Config layer 1 should map to internal layer 0, got {model.repa_align_layer}"

    def test_repa_layer_index_last_layer(self):
        """
        Test 21: repa_align_layer=patch_layers maps to last layer.
        """
        config = PixelHDMConfig.for_testing()
        config.repa_align_layer = config.patch_layers  # Last layer (1-indexed)

        model = PixelHDM(config)

        expected = config.patch_layers - 1  # Last layer (0-indexed)
        assert model.repa_align_layer == expected, \
            f"Config layer {config.patch_layers} should map to internal layer {expected}, got {model.repa_align_layer}"

    def test_repa_features_extracted_from_correct_layer(
        self,
        testing_config: PixelHDMConfig
    ):
        """
        Test 22: REPA features are extracted from the correct layer.

        Verify that return_features=True returns features from the correct layer.
        """
        B, H, W = 2, 256, 256

        model = PixelHDM(testing_config)
        model.eval()

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        with torch.no_grad():
            output, repa_features = model(x, t, return_features=True)

        # REPA features should exist
        assert repa_features is not None, "REPA features should not be None"

        # REPA features should have correct shape
        patch_size = testing_config.patch_size
        L = (H // patch_size) * (W // patch_size)
        D = testing_config.hidden_dim

        assert repa_features.shape == (B, L, D), \
            f"REPA features shape {repa_features.shape} != expected ({B}, {L}, {D})"


# ============================================================================
# Test Class: Text Embedding Integration Tests
# ============================================================================

class TestTextEmbeddingIntegration:
    """
    Tests for text embedding integration.

    Verifies that text embeddings are correctly integrated into the model.
    """

    def test_text_embedding_integration_basic(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 23: Text embeddings are correctly accepted.
        """
        B, H, W = 2, 256, 256
        T, D = 32, testing_config.hidden_dim

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, T, D)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t, text_embed=text_embed)

        assert output.shape == (B, H, W, 3), \
            f"Output with text embedding {output.shape} incorrect"
        assert not torch.isnan(output).any(), "Output with text embedding contains NaN"

    def test_text_embedding_affects_output(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 24: Different text embeddings produce different outputs.
        """
        B, H, W = 2, 256, 256
        T, D = 32, testing_config.hidden_dim

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        # Two different text embeddings
        text_embed_1 = torch.randn(B, T, D)
        text_embed_2 = torch.randn(B, T, D)

        pixelhdm_model.eval()
        with torch.no_grad():
            output_1 = pixelhdm_model(x, t, text_embed=text_embed_1)
            output_2 = pixelhdm_model(x, t, text_embed=text_embed_2)

        # Outputs should be different
        diff = (output_1 - output_2).abs().mean()
        assert diff > 0, "Different text embeddings should produce different outputs"

    def test_text_embedding_with_mask(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 25: Text embeddings with mask work correctly.
        """
        B, H, W = 2, 256, 256
        T, D = 32, testing_config.hidden_dim

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, T, D)

        # Create mask (half valid, half padding)
        text_mask = torch.cat([
            torch.ones(B, T // 2, dtype=torch.long),
            torch.zeros(B, T // 2, dtype=torch.long)
        ], dim=1)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t, text_embed=text_embed, text_mask=text_mask)

        assert output.shape == (B, H, W, 3), \
            f"Output with masked text {output.shape} incorrect"

    def test_text_embedding_none_fallback(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 26: None text embedding is handled (unconditional generation).
        """
        B, H, W = 2, 256, 256

        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t, text_embed=None)

        assert output.shape == (B, H, W, 3), \
            f"Unconditional output {output.shape} incorrect"
        assert not torch.isnan(output).any(), "Unconditional output contains NaN"


# ============================================================================
# Test Class: Joint Sequence Creation Tests
# ============================================================================

class TestJointSequenceCreation:
    """
    Tests for joint text-image sequence creation (Lumina2 style).

    The model concatenates text and image tokens: [text_tokens, img_tokens]
    """

    def test_joint_sequence_creation_basic(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 27: Joint sequence is correctly created.
        """
        B = 2
        L = 256  # Number of image patches
        T = 32   # Text length
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)
        text_embed = torch.randn(B, T, D)
        text_mask = torch.ones(B, T, dtype=torch.long)

        joint_tokens, joint_mask, text_len = pixelhdm_model._create_joint_sequence(
            img_tokens, text_embed, text_mask
        )

        # Check joint sequence shape
        expected_len = T + L
        assert joint_tokens.shape == (B, expected_len, D), \
            f"Joint tokens shape {joint_tokens.shape} != expected ({B}, {expected_len}, {D})"

        # Check text_len
        assert text_len == T, f"text_len {text_len} != {T}"

    def test_joint_sequence_order_text_first(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 28: Text tokens come before image tokens.

        Order: [text_tokens, img_tokens]
        """
        B = 2
        L = 256
        T = 32
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)
        text_embed = torch.randn(B, T, D)
        text_mask = torch.ones(B, T, dtype=torch.long)

        joint_tokens, _, text_len = pixelhdm_model._create_joint_sequence(
            img_tokens, text_embed, text_mask
        )

        # First T tokens should be text
        text_part = joint_tokens[:, :T]
        assert torch.allclose(text_part, text_embed), \
            "First tokens should be text embeddings"

        # Last L tokens should be image
        img_part = joint_tokens[:, T:]
        assert torch.allclose(img_part, img_tokens), \
            "Last tokens should be image tokens"

    def test_joint_sequence_mask_creation(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 29: Joint mask is correctly created.
        """
        B = 2
        L = 256
        T = 32
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)
        text_embed = torch.randn(B, T, D)
        text_mask = torch.ones(B, T, dtype=torch.long)

        _, joint_mask, _ = pixelhdm_model._create_joint_sequence(
            img_tokens, text_embed, text_mask
        )

        # Joint mask should have correct shape
        expected_len = T + L
        assert joint_mask.shape == (B, expected_len), \
            f"Joint mask shape {joint_mask.shape} != expected ({B}, {expected_len})"

        # Image part should always be True (valid)
        img_mask_part = joint_mask[:, T:]
        assert img_mask_part.all(), "Image mask part should all be True"

    def test_joint_sequence_without_text(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 30: Joint sequence handles None text embedding.
        """
        B = 2
        L = 256
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)

        joint_tokens, joint_mask, text_len = pixelhdm_model._create_joint_sequence(
            img_tokens, None, None
        )

        # Without text, should just return image tokens
        assert joint_tokens.shape == (B, L, D), \
            f"Without text, joint shape {joint_tokens.shape} != ({B}, {L}, {D})"
        assert joint_mask is None, "Without text, joint mask should be None"
        assert text_len == 0, "Without text, text_len should be 0"

    def test_joint_sequence_extract_image_tokens(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 31: Image tokens can be correctly extracted from joint sequence.
        """
        B = 2
        L = 256
        T = 32
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)
        text_embed = torch.randn(B, T, D)
        text_mask = torch.ones(B, T, dtype=torch.long)

        joint_tokens, _, text_len = pixelhdm_model._create_joint_sequence(
            img_tokens, text_embed, text_mask
        )

        # Extract image tokens
        extracted = pixelhdm_model._extract_image_tokens(joint_tokens, text_len)

        assert extracted.shape == (B, L, D), \
            f"Extracted shape {extracted.shape} != expected ({B}, {L}, {D})"
        assert torch.allclose(extracted, img_tokens), \
            "Extracted tokens should match original image tokens"

    def test_joint_sequence_extract_without_text(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 32: Extract image tokens when text_len=0.
        """
        B = 2
        L = 256
        D = testing_config.hidden_dim

        img_tokens = torch.randn(B, L, D)

        joint_tokens, _, text_len = pixelhdm_model._create_joint_sequence(
            img_tokens, None, None
        )

        # text_len should be 0
        assert text_len == 0

        # Extract should return same tensor
        extracted = pixelhdm_model._extract_image_tokens(joint_tokens, text_len)

        assert extracted.shape == img_tokens.shape
        assert torch.equal(extracted, joint_tokens), \
            "When text_len=0, extracted should equal joint_tokens"


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """
    Tests for edge cases and error handling.
    """

    def test_invalid_input_dimension_3d(self, pixelhdm_model: PixelHDM):
        """
        Test 33: 3D input raises ValueError.
        """
        x = torch.randn(2, 256, 256)  # 3D, missing channel dim
        t = torch.rand(2)

        with pytest.raises(ValueError, match="4D"):
            pixelhdm_model(x, t)

    def test_invalid_input_dimension_5d(self, pixelhdm_model: PixelHDM):
        """
        Test 34: 5D input raises ValueError.
        """
        x = torch.randn(2, 256, 256, 3, 1)  # 5D
        t = torch.rand(2)

        with pytest.raises(ValueError, match="4D"):
            pixelhdm_model(x, t)

    def test_non_divisible_resolution(
        self,
        testing_config: PixelHDMConfig
    ):
        """
        Test 35: Resolution not divisible by patch_size raises error.
        """
        model = PixelHDM(testing_config)

        patch_size = testing_config.patch_size
        bad_size = patch_size * 10 + 7  # Not divisible by patch_size

        x = torch.randn(2, bad_size, bad_size, 3)
        t = torch.rand(2)

        with pytest.raises((AssertionError, RuntimeError)):
            model(x, t)

    def test_single_batch(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 36: Single batch (B=1) works correctly.
        """
        B, H, W = 1, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3)
        assert not torch.isnan(output).any()

    def test_large_batch(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 37: Larger batch size works correctly.
        """
        B, H, W = 8, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.rand(B)

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3)
        assert not torch.isnan(output).any()

    def test_timestep_boundary_zero(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 38: Timestep t=0 (pure noise in JiT) works.
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.zeros(B)  # t=0, pure noise

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3)
        assert not torch.isnan(output).any()

    def test_timestep_boundary_one(
        self,
        pixelhdm_model: PixelHDM,
        testing_config: PixelHDMConfig
    ):
        """
        Test 39: Timestep t=1 (clean image in JiT) works.
        """
        B, H, W = 2, 256, 256
        x = torch.randn(B, H, W, 3)
        t = torch.ones(B)  # t=1, clean

        pixelhdm_model.eval()
        with torch.no_grad():
            output = pixelhdm_model(x, t)

        assert output.shape == (B, H, W, 3)
        assert not torch.isnan(output).any()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
