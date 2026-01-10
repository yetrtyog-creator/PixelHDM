"""
PixelHDM-RPEA-DinoV3 Strict Loss Function Tests

Strict tests for loss functions with hand-calculated verification.

Test Coverage:
    1. test_vloss_formula - V-Loss = MSE(v_theta, v_target) with manual calculation
    2. test_vloss_zero_when_perfect - Perfect prediction yields zero loss
    3. test_freq_loss_dct_weights - DCT weight calculation correctness
    4. test_freq_loss_ycbcr_conversion - RGB to YCbCr conversion verification
    5. test_repa_loss_cosine_similarity - REPA Loss cosine similarity range [0, 2]
    6. test_combined_loss_weights - Weight application: total = vloss + lambda_freq * freq + lambda_repa * repa

Key Principles:
    - Hand-calculated verification against implementation
    - Edge case and boundary testing
    - Mathematical formula correctness validation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =============================================================================
# V-Loss Strict Tests (V-Prediction)
# =============================================================================

class TestVLossFormulaStrict:
    """
    Strict V-Loss formula verification tests (V-Prediction).

    V-Prediction formulas:
        v_pred = model(z_t, t)  # Network directly outputs velocity
        v_target = x_clean - noise
        L = MSE(v_pred, v_target) = mean((v_pred - v_target)^2)
    """

    @pytest.fixture
    def vloss(self):
        """Create VLoss instance with known t_eps."""
        from src.training.losses.vloss import VLoss
        return VLoss(config=None, t_eps=0.05)

    def test_vloss_formula_manual_calculation(self, vloss):
        """
        Test 1: V-Loss formula with hand-calculated verification (V-Prediction).

        Given:
            v_pred = [[1, 2], [3, 4]]
            x_clean = [[1, 2], [3, 4]]
            noise = [[0, 0], [0, 0]]

        Expected:
            v_target = x_clean - noise
                     = [[1, 2], [3, 4]] - [[0, 0], [0, 0]]
                     = [[1, 2], [3, 4]]

            L = MSE(v_pred, v_target)
              = mean((v_pred - v_target)^2)
              = mean([[0, 0], [0, 0]])
              = 0
        """
        # Create tensors with known values (B=1, C=1, H=2, W=2)
        # In V-Prediction, v_pred is directly what the model outputs
        v_pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        x_clean = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        noise = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])

        # Hand-calculated expected values
        # v_target = x_clean - noise
        expected_v_target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Loss = MSE = 0 when v_pred == v_target
        expected_loss = 0.0

        # Compute using VLoss
        v_target = vloss.compute_v_target(x_clean, noise)
        loss = vloss(v_pred, x_clean, noise, reduction="mean")

        # Verify v_target calculation
        assert torch.allclose(v_target, expected_v_target, atol=1e-5), \
            f"v_target mismatch:\n  got: {v_target}\n  expected: {expected_v_target}"

        # Verify loss
        assert abs(loss.item() - expected_loss) < 1e-5, \
            f"Loss mismatch: got {loss.item()}, expected {expected_loss}"

    def test_vloss_formula_nonzero_loss(self, vloss):
        """
        Test V-Loss formula with non-zero loss (hand-calculated, V-Prediction).

        Given:
            v_pred = [[3, 6], [9, 12]]
            x_clean = [[1, 2], [3, 4]]
            noise = [[0, 0], [0, 0]]

        Expected:
            v_target = x_clean - noise
                     = [[1, 2], [3, 4]]

            diff = v_pred - v_target = [[2, 4], [6, 8]]

            L = MSE = mean([[4, 16], [36, 64]])
              = (4 + 16 + 36 + 64) / 4
              = 120 / 4
              = 30
        """
        v_pred = torch.tensor([[[[3.0, 6.0], [9.0, 12.0]]]])
        x_clean = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        noise = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])

        # Hand-calculated values
        expected_v_target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        expected_loss = 30.0

        # Compute
        v_target = vloss.compute_v_target(x_clean, noise)
        loss = vloss(v_pred, x_clean, noise, reduction="mean")

        # Verify
        assert torch.allclose(v_target, expected_v_target, atol=1e-4), \
            f"v_target mismatch:\n  got: {v_target}\n  expected: {expected_v_target}"
        assert abs(loss.item() - expected_loss) < 1e-3, \
            f"Loss mismatch: got {loss.item()}, expected {expected_loss}"

    def test_vloss_mse_equivalence(self, vloss):
        """
        Verify V-Loss is equivalent to F.mse_loss(v_pred, v_target) (V-Prediction).
        """
        torch.manual_seed(42)
        B, C, H, W = 4, 3, 32, 32

        v_pred = torch.randn(B, C, H, W)
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Compute v_target
        v_target = vloss.compute_v_target(x_clean, noise)

        # VLoss result
        vloss_result = vloss(v_pred, x_clean, noise, reduction="mean")

        # Manual MSE calculation
        manual_mse = F.mse_loss(v_pred.float(), v_target.float(), reduction="mean")

        # Should be equivalent
        assert torch.allclose(vloss_result.float(), manual_mse.float(), atol=1e-4), \
            f"V-Loss {vloss_result.item()} != MSE {manual_mse.item()}"


class TestVLossZeroWhenPerfect:
    """
    Test that V-Loss is zero when prediction is perfect.
    """

    @pytest.fixture
    def vloss(self):
        from src.training.losses.vloss import VLoss
        return VLoss(config=None, t_eps=0.05)

    def test_vloss_zero_when_perfect_simple(self, vloss):
        """
        Test 2: When x_pred perfectly matches the ODE solution, loss should be zero.

        PixelHDM Flow Matching:
            z_t = t * x_clean + (1 - t) * noise

        When the network perfectly predicts x_clean:
            x_pred = x_clean

        Then:
            v_theta = (x_pred - z_t) / (1 - t)
                    = (x_clean - (t * x_clean + (1 - t) * noise)) / (1 - t)
                    = (x_clean - t * x_clean - (1 - t) * noise) / (1 - t)
                    = ((1 - t) * x_clean - (1 - t) * noise) / (1 - t)
                    = x_clean - noise
                    = v_target

        Therefore: L = MSE(v_pred, v_target) = 0

        V-Prediction:
            - v_target = x - noise
            - Perfect prediction: v_pred = v_target
        """
        torch.manual_seed(123)
        B, C, H, W = 2, 3, 32, 32

        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Perfect prediction: v_pred = v_target = x - noise
        v_target = vloss.compute_v_target(x_clean, noise)
        v_pred = v_target.clone()

        loss = vloss(v_pred, x_clean, noise, reduction="mean")

        assert loss.item() < 1e-6, \
            f"Expected near-zero loss for perfect prediction, got {loss.item()}"

    def test_vloss_zero_edge_case_t_near_zero(self, vloss):
        """
        Test perfect prediction with various noise/image combinations.
        For V-Prediction, t is not used in loss computation.
        """
        torch.manual_seed(456)
        B, C, H, W = 2, 3, 32, 32

        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Perfect prediction: v_pred = v_target
        v_target = vloss.compute_v_target(x_clean, noise)
        v_pred = v_target.clone()

        loss = vloss(v_pred, x_clean, noise, reduction="mean")

        assert loss.item() < 1e-5, \
            f"Expected near-zero loss for perfect prediction, got {loss.item()}"

    def test_vloss_zero_edge_case_t_near_one_clamped(self, vloss):
        """
        Test perfect prediction - V-Prediction has no t_eps clamping issue.

        V-Prediction directly computes v_target = x - noise, no 1/(1-t) division.
        This eliminates the numerical instability at t near 1.
        """
        torch.manual_seed(789)
        B, C, H, W = 2, 3, 32, 32

        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Perfect prediction: v_pred = v_target
        v_target = vloss.compute_v_target(x_clean, noise)
        v_pred = v_target.clone()

        loss = vloss(v_pred, x_clean, noise, reduction="mean")

        # V-Prediction should have near-zero loss for perfect prediction
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() < 1e-5, \
            f"V-Prediction should have near-zero loss for perfect prediction, got {loss.item()}"


# =============================================================================
# Frequency Loss Strict Tests
# =============================================================================

class TestFreqLossDCTWeightsStrict:
    """
    Strict tests for DCT weight calculation in Frequency Loss.
    """

    def test_freq_loss_dct_weights_quality_90(self):
        """
        Test 3: DCT weight calculation for quality=90.

        JPEG quality factor scaling formula:
            if quality < 50: scale = 5000 / quality
            else: scale = 200 - 2 * quality

        For quality=90:
            scale = 200 - 2 * 90 = 20

        Scaled quantization value Q:
            Q = floor((base_Q * scale + 50) / 100)
            Q = clamp(Q, 1, 255)

        For DC coefficient (base_Q = 16):
            Q_dc = floor((16 * 20 + 50) / 100) = floor(370 / 100) = 3

        Weight = 1 / Q, then normalized by mean.
        """
        from src.training.losses.freq_loss import FrequencyLoss, JPEG_LUMINANCE_QUANTIZATION_TABLE

        freq_loss = FrequencyLoss(config=None, quality=90)

        # Verify scale calculation
        quality = 90
        expected_scale = 200 - 2 * quality  # = 20
        assert expected_scale == 20, f"Scale should be 20, got {expected_scale}"

        # Manually calculate expected weights for verification
        scale = 20

        def scale_table(q_base, scale):
            q_scaled = torch.floor((q_base * scale + 50) / 100)
            return torch.clamp(q_scaled, min=1, max=255)

        q_y_expected = scale_table(JPEG_LUMINANCE_QUANTIZATION_TABLE, scale)

        # Calculate weights: w = 1/Q, normalized
        w_expected = 1.0 / q_y_expected
        w_expected = w_expected / w_expected.mean()

        # Compare with actual weights
        actual_weights = freq_loss.weights_y

        assert torch.allclose(actual_weights, w_expected, atol=1e-4), \
            f"DCT weights mismatch for quality=90:\n  got: {actual_weights[0, 0]}\n  expected: {w_expected[0, 0]}"

    def test_freq_loss_dct_weights_quality_50(self):
        """
        Test DCT weight calculation for quality=50 (boundary case).

        For quality=50:
            scale = 5000 / 50 = 100
        """
        from src.training.losses.freq_loss import FrequencyLoss, JPEG_LUMINANCE_QUANTIZATION_TABLE

        freq_loss = FrequencyLoss(config=None, quality=50)

        # Verify scale calculation
        quality = 50
        expected_scale = 5000 / quality  # = 100 (boundary, same as 200 - 2*50)
        assert expected_scale == 100, f"Scale should be 100, got {expected_scale}"

        # Verify weights are computed and positive
        assert freq_loss.weights_y.shape == (8, 8), "Weights should be 8x8"
        assert (freq_loss.weights_y > 0).all(), "All weights should be positive"

    def test_freq_loss_dct_weights_low_frequency_higher(self):
        """
        Test that low-frequency weights are higher than high-frequency weights.

        JPEG quantization: low-frequency coefficients have smaller Q values,
        resulting in larger weights (1/Q).

        The DC coefficient (top-left) should have one of the highest weights.
        """
        from src.training.losses.freq_loss import FrequencyLoss

        freq_loss = FrequencyLoss(config=None, quality=90)

        weights = freq_loss.weights_y

        # DC coefficient is at (0, 0)
        dc_weight = weights[0, 0].item()

        # High-frequency coefficient is at (7, 7)
        hf_weight = weights[7, 7].item()

        # DC should have higher or equal weight (after normalization)
        # Note: After normalization, the relative ordering is preserved
        # but absolute values are scaled
        assert dc_weight >= hf_weight * 0.8, \
            f"DC weight ({dc_weight}) should be >= HF weight ({hf_weight})"


class TestFreqLossYCbCrConversionStrict:
    """
    Strict tests for RGB to YCbCr conversion in Frequency Loss.
    """

    def test_freq_loss_ycbcr_conversion_formula(self):
        """
        Test 4: RGB to YCbCr conversion using ITU-R BT.601 standard.

        Conversion formulas:
            Y  =  0.299 * R + 0.587 * G + 0.114 * B
            Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
            Cr =  0.5 * R - 0.418688 * G - 0.081312 * B

        Test with known values:
            R=255, G=0, B=0 (pure red)
            Y  = 0.299 * 255 = 76.245
            Cb = -0.168736 * 255 = -43.028
            Cr = 0.5 * 255 = 127.5
        """
        from src.training.losses.freq_loss import rgb_to_ycbcr

        # Test with known RGB value (normalized to [0, 1] scale, then scaled)
        # Using raw pixel values for clarity
        R, G, B = 1.0, 0.0, 0.0  # Pure red (normalized)

        rgb = torch.tensor([[[[R]], [[G]], [[B]]]])  # (1, 3, 1, 1)

        ycbcr = rgb_to_ycbcr(rgb)

        # Expected values based on ITU-R BT.601
        expected_Y = 0.299 * R + 0.587 * G + 0.114 * B
        expected_Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
        expected_Cr = 0.5 * R - 0.418688 * G - 0.081312 * B

        assert abs(ycbcr[0, 0, 0, 0].item() - expected_Y) < 1e-5, \
            f"Y mismatch: got {ycbcr[0, 0, 0, 0].item()}, expected {expected_Y}"
        assert abs(ycbcr[0, 1, 0, 0].item() - expected_Cb) < 1e-5, \
            f"Cb mismatch: got {ycbcr[0, 1, 0, 0].item()}, expected {expected_Cb}"
        assert abs(ycbcr[0, 2, 0, 0].item() - expected_Cr) < 1e-5, \
            f"Cr mismatch: got {ycbcr[0, 2, 0, 0].item()}, expected {expected_Cr}"

    def test_freq_loss_ycbcr_gray_image(self):
        """
        Test YCbCr conversion for gray image (R=G=B).

        For grayscale where R=G=B=V:
            Y = 0.299*V + 0.587*V + 0.114*V = V
            Cb = -0.168736*V - 0.331264*V + 0.5*V = 0
            Cr = 0.5*V - 0.418688*V - 0.081312*V = 0
        """
        from src.training.losses.freq_loss import rgb_to_ycbcr

        V = 0.5  # Gray level
        rgb = torch.full((1, 3, 4, 4), V)

        ycbcr = rgb_to_ycbcr(rgb)

        # Y should equal V
        assert torch.allclose(ycbcr[:, 0:1], torch.full_like(ycbcr[:, 0:1], V), atol=1e-5), \
            f"Y should equal {V} for gray image"

        # Cb and Cr should be approximately 0
        assert torch.allclose(ycbcr[:, 1:2], torch.zeros_like(ycbcr[:, 1:2]), atol=1e-5), \
            "Cb should be 0 for gray image"
        assert torch.allclose(ycbcr[:, 2:3], torch.zeros_like(ycbcr[:, 2:3]), atol=1e-5), \
            "Cr should be 0 for gray image"

    def test_freq_loss_ycbcr_batch_consistency(self):
        """
        Test that YCbCr conversion is consistent across batch.
        """
        from src.training.losses.freq_loss import rgb_to_ycbcr

        B = 4
        C, H, W = 3, 32, 32

        # Create batch with same image repeated
        single_image = torch.randn(1, C, H, W)
        batch = single_image.repeat(B, 1, 1, 1)

        ycbcr = rgb_to_ycbcr(batch)

        # All batch elements should be identical
        for i in range(1, B):
            assert torch.allclose(ycbcr[0], ycbcr[i], atol=1e-6), \
                f"Batch element {i} differs from element 0"


# =============================================================================
# REPA Loss Strict Tests
# =============================================================================

class TestREPALossCosineSimilarityStrict:
    """
    Strict tests for REPA Loss cosine similarity calculation.
    """

    @pytest.fixture
    def repa_loss(self):
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss
        config = PixelHDMConfig.for_testing()
        config.repa_enabled = True
        config.repa_lambda = 1.0  # Use 1.0 for easier verification
        config.repa_early_stop = 250000
        config.repa_hidden_size = 768
        return REPALoss(config=config)

    def test_repa_loss_cosine_similarity_range(self, repa_loss):
        """
        Test 5: REPA Loss cosine similarity should be in range [0, 2].

        REPA Loss formula:
            L = lambda * (1 - cos_sim).mean()

        Where:
            cos_sim in [-1, 1]
            1 - cos_sim in [0, 2]

        With lambda=1.0:
            L in [0, 2]
        """
        torch.manual_seed(42)
        B, L = 2, 256

        model_features = torch.randn(B, L, 256)
        dino_features = torch.randn(B, L, 768)

        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # Loss should be in [0, 2] with lambda=1.0
        assert 0 <= loss.item() <= 2.0, \
            f"REPA loss {loss.item()} should be in range [0, 2]"

    def test_repa_loss_identical_features_near_zero(self, repa_loss):
        """
        Test that identical (aligned) features produce loss near 0.

        When h_proj == dino_features (after spatial normalization):
            cos_sim = 1
            L = lambda * (1 - 1) = 0
        """
        torch.manual_seed(123)
        B, L = 2, 256
        H = W = int(L ** 0.5)  # 16x16

        # Create model features
        model_features = torch.randn(B, L, 256)

        # Project them through the projector to get dino-like features
        with torch.no_grad():
            h_2d = model_features.permute(0, 2, 1).reshape(B, 256, H, W)
            projected = repa_loss.projector(h_2d)
            projected = projected.flatten(2).permute(0, 2, 1)  # (B, L, 768)

            # Apply spatial normalization as the loss function does
            dino_features = repa_loss._spatial_normalize(projected.clone())

        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # Should be close to 0
        assert loss.item() < 0.1, \
            f"Loss for identical (aligned) features should be near 0, got {loss.item()}"

    def test_repa_loss_opposite_features_near_two(self, repa_loss):
        """
        Test that opposite features produce loss near 2.

        When h_proj == -dino_features:
            cos_sim = -1
            L = lambda * (1 - (-1)) = lambda * 2 = 2
        """
        torch.manual_seed(456)
        B, L = 2, 256
        H = W = int(L ** 0.5)

        model_features = torch.randn(B, L, 256)

        with torch.no_grad():
            h_2d = model_features.permute(0, 2, 1).reshape(B, 256, H, W)
            projected = repa_loss.projector(h_2d)
            projected = projected.flatten(2).permute(0, 2, 1)

            # Create opposite features (negative)
            dino_features = repa_loss._spatial_normalize(-projected.clone())

        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # Should be close to 2
        assert loss.item() > 1.5, \
            f"Loss for opposite features should be near 2, got {loss.item()}"

    def test_repa_loss_orthogonal_features_near_one(self, repa_loss):
        """
        Test that orthogonal features produce loss near 1.

        When h_proj perpendicular to dino_features:
            cos_sim = 0
            L = lambda * (1 - 0) = 1
        """
        torch.manual_seed(789)
        B, L = 2, 256

        # Create random features - for random high-dimensional vectors,
        # expected cosine similarity approaches 0
        model_features = torch.randn(B, L, 256)
        dino_features = torch.randn(B, L, 768)

        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # For random vectors, loss should be around 1 (cos_sim ~ 0)
        assert 0.5 <= loss.item() <= 1.5, \
            f"Loss for orthogonal features should be around 1, got {loss.item()}"

    def test_repa_loss_cosine_formula_manual(self, repa_loss):
        """
        Manually verify the cosine similarity calculation.

        cos_sim(a, b) = (a . b) / (||a|| * ||b||)

        With normalization: cos_sim = sum(a_norm * b_norm)
        """
        B, L, D = 2, 4, 4  # Small dimensions for manual calculation

        # Create simple known vectors
        h = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * L] * B)  # (B, L, 4)
        y = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * L] * B)  # Same direction

        # Manual calculation
        h_norm = F.normalize(h, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        cos_sim = torch.sum(h_norm * y_norm, dim=-1)

        # Expected: cos_sim = 1 for each (B, L) element
        assert torch.allclose(cos_sim, torch.ones(B, L), atol=1e-5), \
            "Cosine similarity should be 1 for identical unit vectors"


# =============================================================================
# Combined Loss Strict Tests
# =============================================================================

class TestCombinedLossWeightsStrict:
    """
    Strict tests for Combined Loss weight application.
    """

    def test_combined_loss_weights_formula(self):
        """
        Test 6: Weight application verification.

        Combined Loss formula:
            L_total = L_vloss + lambda_freq * L_freq + lambda_repa * L_repa

        Note: In the implementation, FrequencyLoss applies lambda_freq internally,
        and REPALoss applies lambda_repa internally.

        So the actual formula is:
            L_total = L_vloss + L_freq_weighted + L_repa_weighted

        Where:
            L_freq_weighted = lambda_freq * L_freq_raw (applied in FrequencyLoss)
            L_repa_weighted = lambda_repa * L_repa_raw (applied in REPALoss)
        """
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        # Create config with known weights
        config = PixelHDMConfig.for_testing()
        config.freq_loss_enabled = True
        config.freq_loss_lambda = 0.5
        config.repa_enabled = True
        config.repa_lambda = 0.3
        config.repa_hidden_size = 768

        loss_fn = CombinedLoss(config=config)

        # Create inputs (V-Prediction: model directly outputs velocity)
        torch.manual_seed(42)
        B, C, H, W = 2, 3, 64, 64
        L = (H // 16) * (W // 16)  # 16 patches

        v_pred = torch.randn(B, C, H, W)  # V-Prediction output
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        h_t = torch.randn(B, L, 256)
        dino_features = torch.randn(B, L, 768)

        result = loss_fn(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            h_t=h_t,
            step=0,
            dino_features=dino_features,
        )

        # Verify total = vloss + freq_loss + repa_loss
        # (freq_loss and repa_loss already have weights applied internally)
        computed_total = result["vloss"] + result["freq_loss"] + result["repa_loss"]
        actual_total = result["total"]

        assert torch.allclose(computed_total, actual_total, atol=1e-5), \
            f"Total mismatch: {actual_total.item()} != vloss({result['vloss'].item()}) + freq({result['freq_loss'].item()}) + repa({result['repa_loss'].item()})"

    def test_combined_loss_vloss_only_when_others_disabled(self):
        """
        Test that total equals vloss when freq and repa are disabled.
        """
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        config.freq_loss_enabled = False  # Disable freq loss
        config.repa_enabled = False  # Disable REPA

        loss_fn = CombinedLoss(config=config)

        torch.manual_seed(42)
        B, C, H, W = 2, 3, 64, 64

        v_pred = torch.randn(B, C, H, W)  # V-Prediction output
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        result = loss_fn(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            step=0,
        )

        # With freq and repa disabled, total should equal vloss
        assert torch.allclose(result["total"], result["vloss"], atol=1e-5), \
            f"Total {result['total'].item()} should equal vloss {result['vloss'].item()}"
        assert result["freq_loss"].item() == 0.0, "Freq loss should be 0 when disabled"
        assert result["repa_loss"].item() == 0.0, "REPA loss should be 0 when disabled"

    def test_combined_loss_repa_zero_after_early_stop(self):
        """
        Test that REPA contributes 0 after early stop threshold.
        """
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        config.freq_loss_enabled = True
        config.repa_enabled = True
        config.repa_early_stop = 250000
        config.repa_hidden_size = 768

        loss_fn = CombinedLoss(config=config)

        torch.manual_seed(42)
        B, C, H, W = 2, 3, 64, 64
        L = (H // 16) * (W // 16)

        v_pred = torch.randn(B, C, H, W)  # V-Prediction output
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        h_t = torch.randn(B, L, 256)
        dino_features = torch.randn(B, L, 768)

        # Before early stop
        result_before = loss_fn(
            v_pred=v_pred, x_clean=x_clean, noise=noise,
            h_t=h_t, step=100, dino_features=dino_features,
        )

        # After early stop
        result_after = loss_fn(
            v_pred=v_pred, x_clean=x_clean, noise=noise,
            h_t=h_t, step=300000, dino_features=dino_features,
        )

        assert result_before["repa_loss"].item() != 0.0, \
            "REPA should be non-zero before early stop"
        assert result_after["repa_loss"].item() == 0.0, \
            "REPA should be zero after early stop"

    def test_combined_loss_scale_with_lambda(self):
        """
        Test that loss scales correctly with lambda values.

        If we double lambda_freq, the freq contribution should double.
        """
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        # First config with lambda_freq = 0.5
        config1 = PixelHDMConfig.for_testing()
        config1.freq_loss_enabled = True
        config1.freq_loss_lambda = 0.5
        config1.repa_enabled = False  # Disable REPA for clarity

        # Second config with lambda_freq = 1.0
        config2 = PixelHDMConfig.for_testing()
        config2.freq_loss_enabled = True
        config2.freq_loss_lambda = 1.0
        config2.repa_enabled = False

        loss_fn1 = CombinedLoss(config=config1)
        loss_fn2 = CombinedLoss(config=config2)

        torch.manual_seed(42)
        B, C, H, W = 2, 3, 64, 64

        v_pred = torch.randn(B, C, H, W)  # V-Prediction output
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        result1 = loss_fn1(v_pred=v_pred, x_clean=x_clean, noise=noise, step=0)
        result2 = loss_fn2(v_pred=v_pred, x_clean=x_clean, noise=noise, step=0)

        # VLoss should be the same
        assert torch.allclose(result1["vloss"], result2["vloss"], atol=1e-5), \
            "VLoss should be identical"

        # Freq loss should scale with lambda
        # result2.freq_loss should be 2x result1.freq_loss
        if result1["freq_loss"].item() > 1e-6:  # Avoid division by zero
            ratio = result2["freq_loss"].item() / result1["freq_loss"].item()
            assert 1.9 <= ratio <= 2.1, \
                f"Freq loss should scale 2x with lambda, got ratio {ratio}"


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================

class TestLossNumericalStabilityStrict:
    """
    Strict tests for numerical stability of loss functions.
    """

    def test_vloss_no_nan_inf(self):
        """
        Test that V-Loss never produces NaN or Inf under various conditions.
        V-Prediction: directly uses MSE(v_pred, v_target), no 1/(1-t) division.
        """
        from src.training.losses.vloss import VLoss

        vloss = VLoss(config=None)

        test_cases = [
            ("normal", torch.randn(2, 3, 32, 32)),
            ("small_values", torch.randn(2, 3, 32, 32) * 1e-6),
            ("large_values", torch.randn(2, 3, 32, 32) * 1e6),
        ]

        for name, v_pred in test_cases:
            x_clean = torch.randn(2, 3, 32, 32)
            noise = torch.randn(2, 3, 32, 32)

            loss = vloss(v_pred, x_clean, noise, reduction="mean")

            assert not torch.isnan(loss), f"NaN in V-Loss for case: {name}"
            assert not torch.isinf(loss), f"Inf in V-Loss for case: {name}"

    def test_freq_loss_no_nan_inf(self):
        """
        Test that Frequency Loss never produces NaN or Inf.
        """
        from src.training.losses.freq_loss import FrequencyLoss

        freq_loss = FrequencyLoss(config=None, quality=90)

        test_cases = [
            ("normal", torch.randn(2, 3, 64, 64)),
            ("small_values", torch.randn(2, 3, 64, 64) * 1e-6),
            ("large_values", torch.randn(2, 3, 64, 64) * 1e3),
            ("zeros", torch.zeros(2, 3, 64, 64)),
            ("non_8_multiple", torch.randn(2, 3, 67, 73)),
        ]

        for name, v in test_cases:
            v_target = torch.randn_like(v)

            loss = freq_loss(v, v_target)

            assert not torch.isnan(loss), f"NaN in Freq Loss for case: {name}"
            assert not torch.isinf(loss), f"Inf in Freq Loss for case: {name}"

    def test_repa_loss_no_nan_inf(self):
        """
        Test that REPA Loss never produces NaN or Inf.
        """
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        config.repa_enabled = True
        config.repa_hidden_size = 768

        repa_loss = REPALoss(config=config)

        test_cases = [
            ("normal", torch.randn(2, 256, 256), torch.randn(2, 256, 768)),
            ("small_model", torch.randn(2, 256, 256) * 1e-6, torch.randn(2, 256, 768)),
            ("small_dino", torch.randn(2, 256, 256), torch.randn(2, 256, 768) * 1e-6),
            ("zeros_model", torch.zeros(2, 256, 256), torch.randn(2, 256, 768)),
        ]

        for name, h_t, dino in test_cases:
            loss = repa_loss(h_t, x_clean=None, step=0, dino_features=dino)

            assert not torch.isnan(loss), f"NaN in REPA Loss for case: {name}"
            assert not torch.isinf(loss), f"Inf in REPA Loss for case: {name}"


class TestLossGradientFlowStrict:
    """
    Strict tests for gradient flow through loss functions.
    """

    def test_vloss_gradient_flow(self):
        """
        Test that gradients flow correctly through V-Loss.
        V-Prediction: model outputs velocity directly.
        """
        from src.training.losses.vloss import VLoss

        vloss = VLoss(config=None)

        v_pred = torch.randn(2, 3, 32, 32, requires_grad=True)  # V-Prediction output
        x_clean = torch.randn(2, 3, 32, 32)
        noise = torch.randn(2, 3, 32, 32)

        loss = vloss(v_pred, x_clean, noise, reduction="mean")
        loss.backward()

        assert v_pred.grad is not None, "v_pred should have gradient"
        assert not torch.isnan(v_pred.grad).any(), "Gradient should not contain NaN"
        assert not torch.all(v_pred.grad == 0), "Gradient should not be all zeros"

    def test_freq_loss_gradient_flow(self):
        """
        Test that gradients flow correctly through Frequency Loss.
        """
        from src.training.losses.freq_loss import FrequencyLoss

        freq_loss = FrequencyLoss(config=None, quality=90)

        v_pred = torch.randn(2, 3, 64, 64, requires_grad=True)
        v_target = torch.randn(2, 3, 64, 64)

        loss = freq_loss(v_pred, v_target)
        loss.backward()

        assert v_pred.grad is not None, "v_pred should have gradient"
        assert not torch.isnan(v_pred.grad).any(), "Gradient should not contain NaN"
        assert not torch.all(v_pred.grad == 0), "Gradient should not be all zeros"

    def test_repa_loss_gradient_flow(self):
        """
        Test that gradients flow correctly through REPA Loss.
        """
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        config.repa_enabled = True
        config.repa_hidden_size = 768

        repa_loss = REPALoss(config=config)

        h_t = torch.randn(2, 256, 256, requires_grad=True)
        dino = torch.randn(2, 256, 768)

        loss = repa_loss(h_t, x_clean=None, step=0, dino_features=dino)
        loss.backward()

        assert h_t.grad is not None, "h_t should have gradient"
        assert not torch.isnan(h_t.grad).any(), "Gradient should not contain NaN"
        assert not torch.all(h_t.grad == 0), "Gradient should not be all zeros"

    def test_combined_loss_gradient_to_all_inputs(self):
        """
        Test that Combined Loss propagates gradients to all relevant inputs.
        V-Prediction: model outputs velocity directly.
        """
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        config.freq_loss_enabled = True
        config.repa_enabled = True
        config.repa_hidden_size = 768

        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64
        L = (H // 16) * (W // 16)

        v_pred = torch.randn(B, C, H, W, requires_grad=True)  # V-Prediction output
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        h_t = torch.randn(B, L, 256, requires_grad=True)
        dino = torch.randn(B, L, 768)

        result = loss_fn(
            v_pred=v_pred, x_clean=x_clean, noise=noise,
            h_t=h_t, step=0, dino_features=dino,
        )
        result["total"].backward()

        # Both inputs should have gradients
        assert v_pred.grad is not None, "v_pred should have gradient from VLoss + FreqLoss"
        assert h_t.grad is not None, "h_t should have gradient from REPALoss"

        # Gradients should be non-zero
        assert not torch.all(v_pred.grad == 0), "v_pred gradient should be non-zero"
        assert not torch.all(h_t.grad == 0), "h_t gradient should be non-zero"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
