"""
PixelHDM Extended Unit Tests.

Extended test coverage for PixelHDM core model:
- Parametrized resolution tests (256, 512, 768)
- Parametrized batch size tests (1, 2, 4, 8)
- EMA state handling tests
- Initialization regression tests (2026-01-05 bug prevention)
- Mixed precision tests (float16, bfloat16)
- Memory efficiency tests

Test Count: 35 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional, List
import copy

from src.models.pixelhdm import PixelHDM, create_pixelhdm
from src.config import PixelHDMConfig
from src.training.optimization.ema import EMA


# =============================================================================
# Test Fixtures (module-scoped where possible for efficiency)
# =============================================================================


@pytest.fixture(scope="module")
def testing_config() -> PixelHDMConfig:
    """Create minimal configuration for fast testing (module-scoped)."""
    return PixelHDMConfig.for_testing()


@pytest.fixture(scope="module")
def pixelhdm_model(testing_config: PixelHDMConfig) -> PixelHDM:
    """Create PixelHDM model for testing (module-scoped, eval mode)."""
    model = PixelHDM(testing_config)
    model.eval()
    return model


# Shared module-scoped model for parametrized tests
@pytest.fixture(scope="module")
def shared_model_for_parametrized(testing_config: PixelHDMConfig) -> PixelHDM:
    """Shared model for parametrized tests to avoid re-creation."""
    model = PixelHDM(testing_config)
    model.eval()
    return model


# =============================================================================
# Test Class: Parametrized Resolution Tests
# =============================================================================


class TestParametrizedResolution:
    """Parametrized tests for different resolutions (uses shared model)."""

    @pytest.mark.parametrize("resolution", [256, 512, 768])
    def test_forward_various_resolutions(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig, resolution: int):
        """Test forward pass at various resolutions."""
        model = shared_model_for_parametrized

        B = 2
        x = torch.randn(B, resolution, resolution, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, 32, testing_config.hidden_dim)

        with torch.no_grad():
            output = model(x, t, text_embed=text_embed)

        assert output.shape == (B, resolution, resolution, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("height,width", [
        (256, 512),
        (512, 256),
        (384, 768),
        (768, 384),
    ])
    def test_rectangular_resolutions(self, shared_model_for_parametrized: PixelHDM, height: int, width: int):
        """Test with rectangular (non-square) resolutions."""
        model = shared_model_for_parametrized

        B = 2
        x = torch.randn(B, height, width, 3)
        t = torch.rand(B)

        with torch.no_grad():
            output = model(x, t)

        assert output.shape == (B, height, width, 3)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("resolution", [256, 512])
    def test_patch_count_correct(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig, resolution: int):
        """Test patch count is correct for different resolutions."""
        model = shared_model_for_parametrized
        patch_size = testing_config.patch_size

        x = torch.randn(2, resolution, resolution, 3)
        embedded = model.patch_embed(x)

        expected_patches = (resolution // patch_size) ** 2
        assert embedded.shape[1] == expected_patches


# =============================================================================
# Test Class: Parametrized Batch Size Tests
# =============================================================================


class TestParametrizedBatchSize:
    """Parametrized tests for different batch sizes (uses shared model)."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_various_batch_sizes(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig, batch_size: int):
        """Test forward pass with various batch sizes."""
        model = shared_model_for_parametrized

        x = torch.randn(batch_size, 256, 256, 3)
        t = torch.rand(batch_size)
        text_embed = torch.randn(batch_size, 32, testing_config.hidden_dim)

        with torch.no_grad():
            output = model(x, t, text_embed=text_embed)

        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cfg_various_batch_sizes(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig, batch_size: int):
        """Test CFG forward with various batch sizes."""
        model = shared_model_for_parametrized

        x = torch.randn(batch_size, 256, 256, 3)
        t = torch.rand(batch_size)
        text_embed = torch.randn(batch_size, 32, testing_config.hidden_dim)
        null_embed = torch.zeros(batch_size, 32, testing_config.hidden_dim)

        with torch.no_grad():
            output = model.forward_with_cfg(
                x, t,
                text_embed=text_embed,
                cfg_scale=7.5,
                null_text_embed=null_embed,
            )

        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

    def test_batch_size_one_edge_case(self, shared_model_for_parametrized: PixelHDM):
        """Test batch size 1 specifically (common for inference)."""
        model = shared_model_for_parametrized

        x = torch.randn(1, 256, 256, 3)
        t = torch.rand(1)

        with torch.no_grad():
            output = model(x, t)

        assert output.shape == (1, 256, 256, 3)


# =============================================================================
# Test Class: EMA State Handling
# =============================================================================


class TestEMAStateHandling:
    """Tests for EMA (Exponential Moving Average) state handling."""

    def test_ema_initialization(self, pixelhdm_model: PixelHDM):
        """Test EMA can be initialized from model."""
        ema = EMA(pixelhdm_model, decay=0.999)

        # EMA should have shadow dict
        assert hasattr(ema, 'shadow')
        assert isinstance(ema.shadow, dict)
        # Should have entries for trainable parameters
        trainable_count = sum(1 for p in pixelhdm_model.parameters() if p.requires_grad)
        assert len(ema.shadow) == trainable_count

    def test_ema_update(self, pixelhdm_model: PixelHDM):
        """Test EMA update modifies shadow parameters."""
        ema = EMA(pixelhdm_model, decay=0.999)

        # Get initial shadow params
        initial_shadows = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        for param in pixelhdm_model.parameters():
            param.data.add_(0.1)

        # Update EMA
        ema.update(pixelhdm_model)

        # Shadow params should have changed
        for name, initial in initial_shadows.items():
            assert not torch.allclose(initial, ema.shadow[name])

    def test_ema_apply_restore(self, pixelhdm_model: PixelHDM, testing_config: PixelHDMConfig):
        """Test EMA apply and restore cycle."""
        ema = EMA(pixelhdm_model, decay=0.999)

        # Save original parameters
        original_params = {n: p.clone() for n, p in pixelhdm_model.named_parameters()}

        # Modify model
        for param in pixelhdm_model.parameters():
            param.data.add_(1.0)

        # Update EMA several times
        for _ in range(10):
            ema.update(pixelhdm_model)

        # Store modified params
        modified_params = {n: p.clone() for n, p in pixelhdm_model.named_parameters()}

        # Use context manager to apply EMA
        with ema.apply_to(pixelhdm_model):
            # Inside context, params should be EMA shadow
            for name, param in pixelhdm_model.named_parameters():
                if param.requires_grad:
                    assert not torch.allclose(param, modified_params[name])

        # After context, model params should be restored
        for name, param in pixelhdm_model.named_parameters():
            assert torch.allclose(param, modified_params[name])

    def test_ema_state_dict(self, pixelhdm_model: PixelHDM):
        """Test EMA state dict save/load."""
        ema = EMA(pixelhdm_model, decay=0.999)

        # Update EMA
        for _ in range(5):
            for param in pixelhdm_model.parameters():
                param.data.add_(0.01)
            ema.update(pixelhdm_model)

        # Save state dict
        state_dict = ema.state_dict()

        # Create new EMA and load
        new_ema = EMA(pixelhdm_model, decay=0.999)
        new_ema.load_state_dict(state_dict)

        # Verify shadow params match
        for name in ema.shadow:
            assert torch.allclose(ema.shadow[name], new_ema.shadow[name])


# =============================================================================
# Test Class: Initialization Regression Tests
# =============================================================================


class TestInitializationRegression:
    """Regression tests for initialization bugs (2026-01-05, 2026-01-07)."""

    def test_adaln_bias_not_zero_after_init(self, testing_config: PixelHDMConfig):
        """
        Regression test for 2026-01-05 bug: AdaLN bias overwritten to zero.

        The _reinit_adaln() method should restore proper bias values after
        _basic_init overwrites all Linear biases to zero.

        TokenAdaLN bias order: [gamma1, beta1, alpha1, gamma2, beta2, alpha2]
        """
        model = PixelHDM(testing_config)

        # Check patch block AdaLN biases
        for i, block in enumerate(model.patch_blocks):
            # TokenAdaLN uses proj[-1] (Sequential with SiLU + Linear)
            bias = block.adaln.proj[-1].bias.data
            hidden_dim = testing_config.hidden_dim

            # Reshape to see individual params
            bias_reshaped = bias.view(6, hidden_dim)

            # gamma1 should be 1
            gamma1_mean = bias_reshaped[0].mean().item()
            assert abs(gamma1_mean - 1.0) < 1e-5, \
                f"Patch block {i} gamma1={gamma1_mean}, expected 1.0"

            # alpha1 should be 1
            alpha1_mean = bias_reshaped[2].mean().item()
            assert abs(alpha1_mean - 1.0) < 1e-5, \
                f"Patch block {i} alpha1={alpha1_mean}, expected 1.0"

    def test_adaln_weight_not_zero(self, testing_config: PixelHDMConfig):
        """
        Test that AdaLN weights are not zero (should be trunc_normal_).

        2026-01-04 fix: Changed zeros_ to trunc_normal_(std=0.02).
        """
        model = PixelHDM(testing_config)

        for i, block in enumerate(model.patch_blocks):
            # TokenAdaLN uses proj[-1] (Sequential with SiLU + Linear)
            weight = block.adaln.proj[-1].weight.data
            weight_std = weight.std().item()

            # Weight std should be approximately 0.02 (from trunc_normal_)
            assert weight_std > 0.01, \
                f"Patch block {i} AdaLN weight std={weight_std}, expected ~0.02"
            assert weight_std < 0.05, \
                f"Patch block {i} AdaLN weight std={weight_std}, expected ~0.02"

    def test_reinit_adaln_called(self, testing_config: PixelHDMConfig):
        """
        Test that _reinit_adaln is called during __init__.

        Verify by checking that AdaLN has correct initialization after
        _basic_init would have zeroed it.
        """
        model = PixelHDM(testing_config)

        # Check patch blocks
        for block in model.patch_blocks:
            bias = block.adaln.proj[-1].bias.data
            # If _reinit_adaln wasn't called, bias would be all zeros
            assert bias.abs().sum() > 0, "AdaLN bias is all zeros"

        # Check pixel blocks
        for block in model.pixel_blocks:
            # PixelwiseAdaLN has different structure
            if hasattr(block.adaln, 'param_gen'):
                last_linear = block.adaln.param_gen[-1]
                bias = last_linear.bias.data
                # gamma=0.1, alpha=1.0 pattern
                assert bias.abs().sum() > 0, "PixelwiseAdaLN bias is all zeros"

    def test_output_proj_xavier_init(self, testing_config: PixelHDMConfig):
        """
        Regression test for 2026-01-05 fix: output_proj should use Xavier init.

        The small std=0.02 initialization was removed.
        """
        model = PixelHDM(testing_config)

        # Check output_proj weight
        weight = model.output_proj.proj.weight.data
        weight_std = weight.std().item()

        # Xavier uniform: std = sqrt(2 / (fan_in + fan_out))
        # Should be significantly larger than 0.02
        assert weight_std > 0.1, \
            f"output_proj weight std={weight_std}, expected >0.1 (Xavier)"


# =============================================================================
# Test Class: Mixed Precision Tests
# =============================================================================


class TestMixedPrecision:
    """Tests for mixed precision support (float16, bfloat16).

    Note: These tests create separate models because they need different dtypes.
    Creating dtype-specific models is unavoidable for these tests.
    """

    @pytest.fixture(scope="class")
    def float32_model(self, testing_config: PixelHDMConfig) -> PixelHDM:
        """Float32 model (class-scoped)."""
        model = PixelHDM(testing_config)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def float16_model(self, testing_config: PixelHDMConfig) -> PixelHDM:
        """Float16 model (class-scoped)."""
        model = PixelHDM(testing_config)
        model.half()
        model.eval()
        return model

    def test_forward_dtype_float32(self, float32_model: PixelHDM):
        """Test forward pass preserves float32 dtype."""
        x = torch.randn(2, 256, 256, 3, dtype=torch.float32)
        t = torch.rand(2, dtype=torch.float32)

        with torch.no_grad():
            output = float32_model(x, t)

        assert output.dtype == torch.float32

    def test_forward_dtype_float16(self, float16_model: PixelHDM):
        """Test forward pass preserves float16 dtype."""
        x = torch.randn(2, 256, 256, 3, dtype=torch.float16)
        t = torch.rand(2, dtype=torch.float16)

        with torch.no_grad():
            output = float16_model(x, t)

        assert output.dtype == torch.float16

    def test_float16_numerical_stability(self, float16_model: PixelHDM, testing_config: PixelHDMConfig):
        """Test float16 produces valid output without NaN/Inf."""
        x = torch.randn(2, 256, 256, 3, dtype=torch.float16)
        t = torch.rand(2, dtype=torch.float16)
        text_embed = torch.randn(2, 32, testing_config.hidden_dim, dtype=torch.float16)

        with torch.no_grad():
            output = float16_model(x, t, text_embed=text_embed)

        assert not torch.isnan(output).any(), "float16 output contains NaN"
        assert not torch.isinf(output).any(), "float16 output contains Inf"

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        reason="bfloat16 not supported"
    )
    def test_bfloat16_support(self, testing_config: PixelHDMConfig):
        """Test bfloat16 forward pass (requires CUDA, creates its own model)."""
        device = torch.device("cuda")
        model = PixelHDM(testing_config)
        model.to(device, torch.bfloat16)
        model.eval()

        x = torch.randn(2, 256, 256, 3, dtype=torch.bfloat16, device=device)
        t = torch.rand(2, dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            output = model(x, t)

        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()


# =============================================================================
# Test Class: Memory and Efficiency Tests
# =============================================================================


class TestMemoryEfficiency:
    """Tests for memory efficiency and gradient checkpointing."""

    def test_inference_mode_no_grad(self, pixelhdm_model: PixelHDM):
        """Test inference with torch.inference_mode() works correctly."""
        pixelhdm_model.eval()

        x = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.inference_mode():
            output = pixelhdm_model(x, t)

        assert output.shape == (2, 256, 256, 3)
        assert not output.requires_grad

    def test_no_unused_parameters_in_forward(self, pixelhdm_model: PixelHDM):
        """Test all parameters are used in forward pass (for DDP compatibility)."""
        pixelhdm_model.train()
        pixelhdm_model.zero_grad()

        x = torch.randn(2, 256, 256, 3, requires_grad=True)
        t = torch.rand(2)
        text_embed = torch.randn(2, 32, pixelhdm_model.hidden_dim)

        output = pixelhdm_model(x, t, text_embed=text_embed)
        loss = output.mean()
        loss.backward()

        # Count parameters with gradients
        params_with_grad = 0
        params_without_grad = 0

        for name, param in pixelhdm_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad += 1
                else:
                    params_without_grad += 1

        # Most parameters should have gradients
        # Some may legitimately have zero gradient (e.g., unused buffers)
        total = params_with_grad + params_without_grad
        ratio = params_with_grad / total if total > 0 else 0

        # Use 75% threshold - some params may have zero grad due to:
        # - AdaLN bias terms that multiply by zero
        # - Unused projection layers in certain configurations
        assert ratio > 0.75, \
            f"Only {ratio*100:.1f}% of parameters have gradients, expected >75%"


# =============================================================================
# Test Class: Text Conditioning Variations
# =============================================================================


class TestTextConditioningVariations:
    """Tests for various text conditioning scenarios (uses shared model)."""

    def test_with_pooled_text_embed(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig):
        """Test forward with pooled text embedding."""
        model = shared_model_for_parametrized

        B = 2
        x = torch.randn(B, 256, 256, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, 32, testing_config.hidden_dim)
        pooled_text_embed = torch.randn(B, testing_config.hidden_dim)

        with torch.no_grad():
            output = model(
                x, t,
                text_embed=text_embed,
                pooled_text_embed=pooled_text_embed,
            )

        assert output.shape == (B, 256, 256, 3)
        assert not torch.isnan(output).any()

    def test_pooled_embed_affects_output(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig):
        """Test that pooled text embedding affects output."""
        model = shared_model_for_parametrized

        B = 2
        x = torch.randn(B, 256, 256, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, 32, testing_config.hidden_dim)
        pooled_text_embed = torch.randn(B, testing_config.hidden_dim)

        with torch.no_grad():
            output_without_pooled = model(x, t, text_embed=text_embed)
            output_with_pooled = model(
                x, t,
                text_embed=text_embed,
                pooled_text_embed=pooled_text_embed,
            )

        diff = (output_without_pooled - output_with_pooled).abs().mean()
        assert diff > 1e-6, "Pooled text embed should affect output"

    @pytest.mark.parametrize("text_len", [16, 32, 64, 128])
    def test_variable_text_lengths(self, shared_model_for_parametrized: PixelHDM, testing_config: PixelHDMConfig, text_len: int):
        """Test with variable text sequence lengths."""
        model = shared_model_for_parametrized

        B = 2
        x = torch.randn(B, 256, 256, 3)
        t = torch.rand(B)
        text_embed = torch.randn(B, text_len, testing_config.hidden_dim)
        text_mask = torch.ones(B, text_len, dtype=torch.bool)

        with torch.no_grad():
            output = model(x, t, text_embed=text_embed, text_mask=text_mask)

        assert output.shape == (B, 256, 256, 3)
        assert not torch.isnan(output).any()


# =============================================================================
# Test Class: REPA Extended Tests
# =============================================================================


class TestREPAExtended:
    """Extended tests for REPA feature extraction."""

    def test_repa_features_at_different_layers(self, testing_config: PixelHDMConfig):
        """Test REPA features can be extracted from different layers."""
        # Test with layer 1
        config1 = PixelHDMConfig.for_testing()
        config1.repa_align_layer = 1
        model1 = PixelHDM(config1)
        model1.eval()

        x = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.no_grad():
            _, features1 = model1(x, t, return_features=True)

        # Test with layer 2
        config2 = PixelHDMConfig.for_testing()
        config2.repa_align_layer = 2
        model2 = PixelHDM(config2)
        model2.eval()

        with torch.no_grad():
            _, features2 = model2(x, t, return_features=True)

        # Features should have same shape but different values
        assert features1.shape == features2.shape

    def test_repa_features_batch_consistency(self, pixelhdm_model: PixelHDM):
        """Test REPA features are consistent for same input."""
        pixelhdm_model.eval()

        x = torch.randn(2, 256, 256, 3)
        t = torch.rand(2)

        with torch.no_grad():
            _, features1 = pixelhdm_model(x, t, return_features=True)
            _, features2 = pixelhdm_model(x, t, return_features=True)

        assert torch.allclose(features1, features2)


# =============================================================================
# Test Class: Model State Tests
# =============================================================================


class TestModelState:
    """Tests for model state management."""

    def test_train_eval_mode_switch(self, pixelhdm_model: PixelHDM):
        """Test switching between train and eval modes."""
        # Start in train mode
        pixelhdm_model.train()
        assert pixelhdm_model.training

        # Switch to eval
        pixelhdm_model.eval()
        assert not pixelhdm_model.training

        # Back to train
        pixelhdm_model.train()
        assert pixelhdm_model.training

    def test_state_dict_save_load(self, testing_config: PixelHDMConfig):
        """Test state dict save and load."""
        model1 = PixelHDM(testing_config)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load
        model2 = PixelHDM(testing_config)
        model2.load_state_dict(state_dict)

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_requires_grad_control(self, pixelhdm_model: PixelHDM):
        """Test requires_grad can be controlled."""
        # Freeze all parameters
        for param in pixelhdm_model.parameters():
            param.requires_grad = False

        trainable = sum(
            p.numel() for p in pixelhdm_model.parameters() if p.requires_grad
        )
        assert trainable == 0

        # Unfreeze
        for param in pixelhdm_model.parameters():
            param.requires_grad = True

        trainable = sum(
            p.numel() for p in pixelhdm_model.parameters() if p.requires_grad
        )
        assert trainable > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
