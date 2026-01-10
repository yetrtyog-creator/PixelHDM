"""
Gradient Flow Strict Tests

CRITICAL tests verifying gradient flow through the PixelHDM model.

Background:
    A previous bug caused output_proj.proj.weight to be zero-initialized,
    which completely blocked gradient flow:

        grad_input = grad_output @ weight.T

    When weight = 0, grad_input = 0, causing gradient flow to stop.

    This was fixed by using xavier_uniform_(gain=0.02) instead of zeros_().

This test suite ensures:
    1. output_proj weights are never zero-initialized
    2. Full model gradient flow is working (>95% params have gradients)
    3. Gradient magnitudes are reasonable (no vanishing/exploding)
    4. All major components receive gradients

Related files:
    - src/models/pixeldit.py (PixelHDM._init_weights)
    - src/models/layers/embedding.py (PixelPatchify._init_weights)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

from src.config import PixelHDMConfig
from src.models.pixelhdm import PixelHDM, create_pixelhdm
from src.models.layers.embedding import PixelPatchify, create_pixel_patchify


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def testing_config() -> PixelHDMConfig:
    """Create minimal configuration for fast testing.

    Uses PixelHDMConfig.for_testing() which has:
        - hidden_dim=256
        - pixel_dim=8
        - patch_layers=2
        - pixel_layers=1
    """
    return PixelHDMConfig.for_testing()


@pytest.fixture
def pixelhdm_model(testing_config: PixelHDMConfig) -> PixelHDM:
    """Create a fresh PixelHDM model for testing."""
    return PixelHDM(testing_config)


@pytest.fixture
def batch_data(testing_config: PixelHDMConfig) -> dict:
    """Create test batch data with correct dimensions."""
    B = 2
    H, W = 256, 256
    T = 16
    D = testing_config.hidden_dim

    return {
        "x": torch.randn(B, H, W, 3),
        "t": torch.rand(B),
        "text_emb": torch.randn(B, T, D),
        "text_mask": torch.ones(B, T, dtype=torch.long),
        "batch_size": B,
        "height": H,
        "width": W,
    }


# ============================================================================
# Helper Functions
# ============================================================================

def count_gradient_stats(model: nn.Module) -> Dict[str, int]:
    """Count parameters with and without gradients.

    Returns:
        {
            "total": total trainable parameters,
            "with_grad": parameters with non-zero gradients,
            "without_grad": parameters without gradients,
            "grad_is_none": parameters where grad is None,
            "grad_is_zero": parameters where grad exists but is zero
        }
    """
    stats = {
        "total": 0,
        "with_grad": 0,
        "without_grad": 0,
        "grad_is_none": 0,
        "grad_is_zero": 0,
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            stats["total"] += 1
            if param.grad is None:
                stats["grad_is_none"] += 1
                stats["without_grad"] += 1
            elif param.grad.abs().sum() == 0:
                stats["grad_is_zero"] += 1
                stats["without_grad"] += 1
            else:
                stats["with_grad"] += 1

    return stats


def get_params_without_gradients(model: nn.Module) -> List[str]:
    """Get list of parameter names without gradients."""
    no_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or param.grad.abs().sum() == 0:
                no_grad_params.append(name)

    return no_grad_params


def get_gradient_magnitude_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Get gradient magnitude statistics for each parameter.

    Returns:
        {param_name: {"min": ..., "max": ..., "mean": ..., "std": ...}}
    """
    stats = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            stats[name] = {
                "min": grad.abs().min().item(),
                "max": grad.abs().max().item(),
                "mean": grad.abs().mean().item(),
                "std": grad.std().item(),
            }

    return stats


# ============================================================================
# Test 1: output_proj Not Zero Initialized
# ============================================================================

class TestOutputProjNotZeroInitialized:
    """
    CRITICAL: Verify output_proj.proj.weight is not zero-initialized.

    Background:
        Zero initialization of output_proj was the root cause of
        gradient flow blockage. The fix uses xavier_uniform_(gain=0.02)
        which ensures:
        1. Gradients can flow (weight != 0)
        2. Initial outputs are still small (gain=0.02)
        3. Truncation prevents extreme values
    """

    def test_pixelpatchify_weight_not_zero(self):
        """Test PixelPatchify proj.weight is not all zeros."""
        patchify = create_pixel_patchify(
            pixel_dim=16,
            patch_size=16,
            out_channels=3,
        )

        # Weight should NOT be all zeros
        weight = patchify.proj.weight

        assert not torch.allclose(weight, torch.zeros_like(weight)), \
            "CRITICAL: PixelPatchify.proj.weight is all zeros! " \
            "This will block gradient flow. " \
            "Expected xavier_uniform_ or trunc_normal_ initialization."

    def test_pixelpatchify_weight_has_variance(self):
        """Test PixelPatchify proj.weight has non-zero variance."""
        patchify = create_pixel_patchify(
            pixel_dim=16,
            patch_size=16,
            out_channels=3,
        )

        weight = patchify.proj.weight
        weight_std = weight.std().item()

        assert weight_std > 1e-6, \
            f"CRITICAL: PixelPatchify.proj.weight has near-zero std ({weight_std}). " \
            "This indicates improper initialization."

    def test_pixelpatchify_weight_reasonable_magnitude(self):
        """Test PixelPatchify proj.weight has reasonable magnitude."""
        patchify = create_pixel_patchify(
            pixel_dim=16,
            patch_size=16,
            out_channels=3,
        )

        weight = patchify.proj.weight
        weight_norm = weight.norm().item()

        # With trunc_normal(std=0.02), norm should be relatively small
        # For a (3, 16) weight matrix, expected norm ~ sqrt(48) * 0.02 ~ 0.14
        assert 0 < weight_norm < 10.0, \
            f"Weight norm {weight_norm} is outside reasonable range (0, 10). " \
            "Check initialization."

    def test_pixelhdm_output_proj_not_zero(self, pixelhdm_model: PixelHDM):
        """Test PixelHDM output_proj.proj.weight is not all zeros."""
        output_proj = pixelhdm_model.output_proj
        weight = output_proj.proj.weight

        assert not torch.allclose(weight, torch.zeros_like(weight)), \
            "CRITICAL: PixelHDM.output_proj.proj.weight is all zeros! " \
            "This will block gradient flow."

    def test_multiple_model_instances_have_nonzero_weights(self, testing_config: PixelHDMConfig):
        """Test that multiple model instances all have non-zero output_proj weights."""
        # Create 5 different instances
        for i in range(5):
            model = PixelHDM(testing_config)
            weight = model.output_proj.proj.weight

            assert not torch.allclose(weight, torch.zeros_like(weight)), \
                f"Instance {i}: output_proj.proj.weight is all zeros!"

            # Also check variance
            assert weight.std().item() > 1e-6, \
                f"Instance {i}: output_proj.proj.weight has near-zero variance!"


# ============================================================================
# Test 2: Full Model Gradient Flow
# ============================================================================

class TestFullModelGradientFlow:
    """
    CRITICAL: Verify gradients flow through the complete model.

    Requirement: >95% of trainable parameters should have non-zero gradients.
    """

    def test_gradient_flow_percentage(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that >95% of parameters receive gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        # Forward pass
        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Count gradient statistics
        stats = count_gradient_stats(pixelhdm_model)

        # Calculate percentage
        if stats["total"] > 0:
            grad_percentage = (stats["with_grad"] / stats["total"]) * 100
        else:
            grad_percentage = 0

        # CRITICAL: >95% should have gradients
        assert grad_percentage > 95.0, \
            f"CRITICAL: Only {grad_percentage:.1f}% parameters have gradients! " \
            f"(Expected >95%)\n" \
            f"Stats: total={stats['total']}, with_grad={stats['with_grad']}, " \
            f"without_grad={stats['without_grad']}"

    def test_no_parameters_with_none_grad(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that no trainable parameters have grad=None after backward."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        stats = count_gradient_stats(pixelhdm_model)

        assert stats["grad_is_none"] == 0, \
            f"Found {stats['grad_is_none']} parameters with grad=None! " \
            "This indicates disconnected computation graph."

    def test_list_parameters_without_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test and list any parameters without gradients for debugging."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        no_grad_params = get_params_without_gradients(pixelhdm_model)

        # At most 5% of parameters can lack gradients
        stats = count_gradient_stats(pixelhdm_model)
        max_allowed = int(stats["total"] * 0.05)

        assert len(no_grad_params) <= max_allowed, \
            f"Too many parameters without gradients ({len(no_grad_params)} > {max_allowed}):\n" + \
            "\n".join(f"  - {name}" for name in no_grad_params[:20])

    def test_gradient_flow_with_text_conditioning(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test gradient flow when using text conditioning."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
            text_mask=batch_data["text_mask"],
        )

        loss = output.mean()
        loss.backward()

        stats = count_gradient_stats(pixelhdm_model)
        grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0

        assert grad_percentage > 95.0, \
            f"With text conditioning: only {grad_percentage:.1f}% have gradients"

    def test_gradient_flow_without_text_conditioning(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test gradient flow without text conditioning."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=None,  # No text
        )

        loss = output.mean()
        loss.backward()

        stats = count_gradient_stats(pixelhdm_model)
        grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0

        assert grad_percentage > 95.0, \
            f"Without text conditioning: only {grad_percentage:.1f}% have gradients"


# ============================================================================
# Test 3: Gradient Magnitude Reasonable
# ============================================================================

class TestGradientMagnitudeReasonable:
    """
    Verify gradient magnitudes are within reasonable bounds.

    Detects:
        - Vanishing gradients (too small, < 1e-10)
        - Exploding gradients (too large, > 100)
    """

    def test_no_vanishing_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that no gradients are vanishingly small."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        vanishing_params = []
        threshold = 1e-10

        for name, param in pixelhdm_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_max = param.grad.abs().max().item()
                if grad_max < threshold:
                    vanishing_params.append((name, grad_max))

        assert len(vanishing_params) == 0, \
            f"Found {len(vanishing_params)} parameters with vanishing gradients (<{threshold}):\n" + \
            "\n".join(f"  - {name}: max_grad={val:.2e}" for name, val in vanishing_params[:10])

    def test_no_exploding_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that no gradients are explosively large."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        exploding_params = []
        threshold = 100.0

        for name, param in pixelhdm_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_max = param.grad.abs().max().item()
                if grad_max > threshold:
                    exploding_params.append((name, grad_max))

        assert len(exploding_params) == 0, \
            f"Found {len(exploding_params)} parameters with exploding gradients (>{threshold}):\n" + \
            "\n".join(f"  - {name}: max_grad={val:.2e}" for name, val in exploding_params[:10])

    def test_no_nan_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that no gradients are NaN."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        nan_params = []

        for name, param in pixelhdm_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(name)

        assert len(nan_params) == 0, \
            f"Found {len(nan_params)} parameters with NaN gradients:\n" + \
            "\n".join(f"  - {name}" for name in nan_params[:10])

    def test_no_inf_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that no gradients are Inf."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        inf_params = []

        for name, param in pixelhdm_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isinf(param.grad).any():
                    inf_params.append(name)

        assert len(inf_params) == 0, \
            f"Found {len(inf_params)} parameters with Inf gradients:\n" + \
            "\n".join(f"  - {name}" for name in inf_params[:10])

    def test_gradient_magnitude_statistics(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test overall gradient magnitude statistics are reasonable."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Collect all gradient magnitudes
        all_grads = []
        for param in pixelhdm_model.parameters():
            if param.requires_grad and param.grad is not None:
                all_grads.append(param.grad.abs().mean().item())

        if len(all_grads) > 0:
            avg_grad = sum(all_grads) / len(all_grads)
            max_grad = max(all_grads)
            min_grad = min(all_grads)

            # Average gradient should be reasonable
            assert avg_grad > 1e-10, f"Average gradient too small: {avg_grad:.2e}"
            assert avg_grad < 10.0, f"Average gradient too large: {avg_grad:.2e}"

            # Min/max ratio should not be too extreme
            if min_grad > 0:
                ratio = max_grad / min_grad
                assert ratio < 1e12, \
                    f"Gradient magnitude range too large: max/min = {ratio:.2e}"


# ============================================================================
# Test 4: Patch Blocks Gradient Flow
# ============================================================================

class TestPatchBlocksGradientFlow:
    """Verify each patch transformer block receives gradients."""

    def test_all_patch_blocks_have_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that all patch blocks receive gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Check each patch block
        blocks_without_grad = []

        for i, block in enumerate(pixelhdm_model.patch_blocks):
            block_has_grad = False

            for name, param in block.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.abs().sum() > 0:
                        block_has_grad = True
                        break

            if not block_has_grad:
                blocks_without_grad.append(i)

        assert len(blocks_without_grad) == 0, \
            f"Patch blocks without gradients: {blocks_without_grad}"

    def test_patch_block_attention_has_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that attention layers in patch blocks have gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Check attention parameters
        for i, block in enumerate(pixelhdm_model.patch_blocks):
            # Look for attention-related parameters
            attention_params = [
                (name, param)
                for name, param in block.named_parameters()
                if "attn" in name.lower() or "q_proj" in name or "k_proj" in name or "v_proj" in name
            ]

            if len(attention_params) > 0:
                has_attn_grad = any(
                    param.grad is not None and param.grad.abs().sum() > 0
                    for _, param in attention_params
                )

                assert has_attn_grad, \
                    f"Patch block {i} attention has no gradients!"

    def test_patch_block_mlp_has_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that MLP layers in patch blocks have gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        for i, block in enumerate(pixelhdm_model.patch_blocks):
            # Look for MLP-related parameters
            mlp_params = [
                (name, param)
                for name, param in block.named_parameters()
                if "mlp" in name.lower() or "ffn" in name.lower() or "feed_forward" in name.lower()
            ]

            if len(mlp_params) > 0:
                has_mlp_grad = any(
                    param.grad is not None and param.grad.abs().sum() > 0
                    for _, param in mlp_params
                )

                assert has_mlp_grad, \
                    f"Patch block {i} MLP has no gradients!"


# ============================================================================
# Test 5: Pixel Blocks Gradient Flow
# ============================================================================

class TestPixelBlocksGradientFlow:
    """Verify each pixel transformer block receives gradients."""

    def test_all_pixel_blocks_have_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that all pixel blocks receive gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Check each pixel block
        blocks_without_grad = []

        for i, block in enumerate(pixelhdm_model.pixel_blocks):
            block_has_grad = False

            for name, param in block.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.abs().sum() > 0:
                        block_has_grad = True
                        break

            if not block_has_grad:
                blocks_without_grad.append(i)

        assert len(blocks_without_grad) == 0, \
            f"Pixel blocks without gradients: {blocks_without_grad}"

    def test_pixel_block_token_compaction_has_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that token compaction in pixel blocks has gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        for i, block in enumerate(pixelhdm_model.pixel_blocks):
            # Look for token compaction parameters
            compaction_params = [
                (name, param)
                for name, param in block.named_parameters()
                if "compact" in name.lower() or "expand" in name.lower()
            ]

            if len(compaction_params) > 0:
                has_compaction_grad = any(
                    param.grad is not None and param.grad.abs().sum() > 0
                    for _, param in compaction_params
                )

                assert has_compaction_grad, \
                    f"Pixel block {i} token compaction has no gradients!"


# ============================================================================
# Test 6: AdaLN Gradient Flow
# ============================================================================

class TestAdaLNGradientFlow:
    """Verify AdaLN (Adaptive LayerNorm) receives gradients."""

    def test_adaln_in_patch_blocks_has_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that AdaLN in patch blocks receives gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Check AdaLN parameters in patch blocks
        for i, block in enumerate(pixelhdm_model.patch_blocks):
            adaln_params = [
                (name, param)
                for name, param in block.named_parameters()
                if "adaln" in name.lower() or "ada_ln" in name.lower()
            ]

            if len(adaln_params) > 0:
                has_adaln_grad = any(
                    param.grad is not None and param.grad.abs().sum() > 0
                    for _, param in adaln_params
                )

                assert has_adaln_grad, \
                    f"Patch block {i} AdaLN has no gradients!"

    def test_adaln_in_pixel_blocks_has_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that AdaLN in pixel blocks receives gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Check AdaLN parameters in pixel blocks
        for i, block in enumerate(pixelhdm_model.pixel_blocks):
            adaln_params = [
                (name, param)
                for name, param in block.named_parameters()
                if "adaln" in name.lower() or "ada_ln" in name.lower()
            ]

            if len(adaln_params) > 0:
                has_adaln_grad = any(
                    param.grad is not None and param.grad.abs().sum() > 0
                    for _, param in adaln_params
                )

                assert has_adaln_grad, \
                    f"Pixel block {i} AdaLN has no gradients!"

    def test_time_embedding_affects_adaln(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test that time embedding (AdaLN condition) receives gradients."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Time embedding should have gradients (it conditions AdaLN)
        time_embed_has_grad = False

        for name, param in pixelhdm_model.time_embed.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    time_embed_has_grad = True
                    break

        assert time_embed_has_grad, \
            "Time embedding has no gradients! " \
            "This suggests AdaLN is not properly connected."


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestGradientFlowEdgeCases:
    """Edge case tests for gradient flow."""

    def test_gradient_flow_with_return_features(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test gradient flow when return_features=True."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output, repa_features = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
            return_features=True,
        )

        # Use both outputs in loss
        loss = output.mean() + repa_features.mean()
        loss.backward()

        stats = count_gradient_stats(pixelhdm_model)
        grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0

        assert grad_percentage > 95.0, \
            f"With return_features=True: only {grad_percentage:.1f}% have gradients"

    def test_gradient_flow_with_different_batch_sizes(
        self,
        testing_config: PixelHDMConfig,
    ):
        """Test gradient flow with different batch sizes."""
        for batch_size in [1, 2, 4]:
            model = PixelHDM(testing_config)
            model.train()
            model.zero_grad()

            x = torch.randn(batch_size, 256, 256, 3)
            t = torch.rand(batch_size)
            text_emb = torch.randn(batch_size, 16, testing_config.hidden_dim)

            output = model(x_t=x, t=t, text_embed=text_emb)
            loss = output.mean()
            loss.backward()

            stats = count_gradient_stats(model)
            grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0

            assert grad_percentage > 95.0, \
                f"Batch size {batch_size}: only {grad_percentage:.1f}% have gradients"

    def test_gradient_flow_with_different_resolutions(
        self,
        testing_config: PixelHDMConfig,
    ):
        """Test gradient flow with different image resolutions."""
        patch_size = testing_config.patch_size

        for resolution in [64, 128, 256]:
            # Skip if not divisible by patch size
            if resolution % patch_size != 0:
                continue

            model = PixelHDM(testing_config)
            model.train()
            model.zero_grad()

            x = torch.randn(2, resolution, resolution, 3)
            t = torch.rand(2)
            text_emb = torch.randn(2, 16, testing_config.hidden_dim)

            output = model(x_t=x, t=t, text_embed=text_emb)
            loss = output.mean()
            loss.backward()

            stats = count_gradient_stats(model)
            grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0

            assert grad_percentage > 95.0, \
                f"Resolution {resolution}x{resolution}: only {grad_percentage:.1f}% have gradients"

    def test_gradient_flow_multiple_backward_passes(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """Test gradient accumulation over multiple backward passes."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        # Multiple forward-backward passes (simulating gradient accumulation)
        for _ in range(3):
            output = pixelhdm_model(
                x_t=batch_data["x"],
                t=batch_data["t"],
                text_embed=batch_data["text_emb"],
            )

            loss = output.mean()
            loss.backward()

        stats = count_gradient_stats(pixelhdm_model)
        grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0

        assert grad_percentage > 95.0, \
            f"After 3 backward passes: only {grad_percentage:.1f}% have gradients"


# ============================================================================
# Regression Test
# ============================================================================

class TestGradientFlowRegression:
    """
    Regression tests to ensure the zero-initialization bug doesn't reoccur.

    These tests are specifically designed to catch the bug that was fixed:
    output_proj.proj.weight was zeros_, which blocked gradients.
    """

    def test_output_proj_receives_gradients(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """REGRESSION: output_proj must receive gradients (was blocked before fix)."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # output_proj is the final projection layer
        output_proj = pixelhdm_model.output_proj

        output_proj_has_grad = False
        for name, param in output_proj.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    output_proj_has_grad = True
                    break

        assert output_proj_has_grad, \
            "REGRESSION: output_proj has no gradients! " \
            "This was the original bug (zero initialization)."

    def test_gradient_flows_to_early_layers(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """REGRESSION: gradients must flow to early layers (was blocked before fix)."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        # Check patch_embed (earliest layer)
        patch_embed_has_grad = False
        for name, param in pixelhdm_model.patch_embed.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    patch_embed_has_grad = True
                    break

        assert patch_embed_has_grad, \
            "REGRESSION: patch_embed has no gradients! " \
            "Gradient flow is blocked somewhere in the network."

        # Check time_embed
        time_embed_has_grad = False
        for name, param in pixelhdm_model.time_embed.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    time_embed_has_grad = True
                    break

        assert time_embed_has_grad, \
            "REGRESSION: time_embed has no gradients! " \
            "Gradient flow is blocked somewhere in the network."

    def test_before_fix_would_have_only_2_params_with_grads(
        self,
        pixelhdm_model: PixelHDM,
        batch_data: dict,
    ):
        """
        REGRESSION: Before the fix, only 2/342 parameters had gradients.
        After the fix, >95% should have gradients.
        """
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        output = pixelhdm_model(
            x_t=batch_data["x"],
            t=batch_data["t"],
            text_embed=batch_data["text_emb"],
        )

        loss = output.mean()
        loss.backward()

        stats = count_gradient_stats(pixelhdm_model)

        # Before fix: only 2 params had gradients
        # After fix: should be >> 2
        assert stats["with_grad"] > 10, \
            f"REGRESSION: Only {stats['with_grad']} parameters have gradients! " \
            f"Before fix, only 2/342 had gradients. This suggests the fix regressed."

        # Verify percentage is high
        grad_percentage = (stats["with_grad"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        assert grad_percentage > 95.0, \
            f"REGRESSION: Only {grad_percentage:.1f}% parameters have gradients! " \
            f"Before fix, it was ~0.6%. After fix, should be >95%."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
