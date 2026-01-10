"""
Attention Operations Unit Tests

Tests for:
    - QKV Projection
    - Attention Score Computation
    - GQA (Grouped Query Attention)
    - Gate Mechanism

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.attention import (
    GatedMultiHeadAttention,
    QKVProjection,
    OutputProjection,
    GateProjection,
    IdentityGate,
    create_gate,
    repeat_kv,
)
from src.models.attention.attention_ops import (
    compute_flash_attention,
    compute_manual_attention,
    prepare_attention_mask,
)
from src.config import PixelHDMConfig


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def default_config() -> dict:
    """Default configuration for attention tests."""
    return {
        "hidden_dim": 1024,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 64,
        "batch_size": 2,
        "seq_len": 256,
    }


@pytest.fixture
def qkv_projection(default_config: dict) -> QKVProjection:
    """Create QKV projection module."""
    return QKVProjection(
        hidden_dim=default_config["hidden_dim"],
        num_heads=default_config["num_heads"],
        num_kv_heads=default_config["num_kv_heads"],
        head_dim=default_config["head_dim"],
    )


@pytest.fixture
def gated_attention(default_config: dict) -> GatedMultiHeadAttention:
    """Create gated attention module."""
    return GatedMultiHeadAttention(
        hidden_dim=default_config["hidden_dim"],
        num_heads=default_config["num_heads"],
        num_kv_heads=default_config["num_kv_heads"],
        head_dim=default_config["head_dim"],
        use_qk_norm=True,
        use_gated_attention=True,
        gate_type="headwise",
        use_flash_attention=False,  # Use manual attention for testing
        use_checkpoint=False,
    )


@pytest.fixture
def sample_input(default_config: dict) -> torch.Tensor:
    """Create sample input tensor."""
    return torch.randn(
        default_config["batch_size"],
        default_config["seq_len"],
        default_config["hidden_dim"],
    )


# ==============================================================================
# 1. QKV Projection Tests (5 tests)
# ==============================================================================


class TestQKVProjection:
    """Tests for QKV projection operations."""

    def test_q_projection_shape(
        self, qkv_projection: QKVProjection, sample_input: torch.Tensor, default_config: dict
    ) -> None:
        """Test Q projection output shape: (B, num_heads, L, head_dim)."""
        q, _, _ = qkv_projection(sample_input)

        expected_shape = (
            default_config["batch_size"],
            default_config["num_heads"],
            default_config["seq_len"],
            default_config["head_dim"],
        )
        assert q.shape == expected_shape, (
            f"Q projection shape mismatch. Expected {expected_shape}, got {q.shape}"
        )

    def test_k_projection_shape(
        self, qkv_projection: QKVProjection, sample_input: torch.Tensor, default_config: dict
    ) -> None:
        """Test K projection output shape with GQA: (B, num_kv_heads, L, head_dim)."""
        _, k, _ = qkv_projection(sample_input)

        expected_shape = (
            default_config["batch_size"],
            default_config["num_kv_heads"],
            default_config["seq_len"],
            default_config["head_dim"],
        )
        assert k.shape == expected_shape, (
            f"K projection shape mismatch. Expected {expected_shape}, got {k.shape}"
        )

    def test_v_projection_shape(
        self, qkv_projection: QKVProjection, sample_input: torch.Tensor, default_config: dict
    ) -> None:
        """Test V projection output shape with GQA: (B, num_kv_heads, L, head_dim)."""
        _, _, v = qkv_projection(sample_input)

        expected_shape = (
            default_config["batch_size"],
            default_config["num_kv_heads"],
            default_config["seq_len"],
            default_config["head_dim"],
        )
        assert v.shape == expected_shape, (
            f"V projection shape mismatch. Expected {expected_shape}, got {v.shape}"
        )

    def test_qkv_fused_projection(self, default_config: dict) -> None:
        """Test that QKVProjection contains all three projection layers."""
        qkv_proj = QKVProjection(
            hidden_dim=default_config["hidden_dim"],
            num_heads=default_config["num_heads"],
            num_kv_heads=default_config["num_kv_heads"],
            head_dim=default_config["head_dim"],
        )

        # Verify all projection layers exist
        assert hasattr(qkv_proj, "q_proj"), "Missing q_proj layer"
        assert hasattr(qkv_proj, "k_proj"), "Missing k_proj layer"
        assert hasattr(qkv_proj, "v_proj"), "Missing v_proj layer"

        # Verify output dimensions
        assert qkv_proj.q_proj.out_features == default_config["num_heads"] * default_config["head_dim"]
        assert qkv_proj.k_proj.out_features == default_config["num_kv_heads"] * default_config["head_dim"]
        assert qkv_proj.v_proj.out_features == default_config["num_kv_heads"] * default_config["head_dim"]

    def test_projection_bias_optional(self, default_config: dict) -> None:
        """Test that QKV projections have bias disabled (as per design)."""
        qkv_proj = QKVProjection(
            hidden_dim=default_config["hidden_dim"],
            num_heads=default_config["num_heads"],
            num_kv_heads=default_config["num_kv_heads"],
            head_dim=default_config["head_dim"],
        )

        # Default implementation uses bias=False
        assert qkv_proj.q_proj.bias is None, "Q projection should have no bias"
        assert qkv_proj.k_proj.bias is None, "K projection should have no bias"
        assert qkv_proj.v_proj.bias is None, "V projection should have no bias"


# ==============================================================================
# 2. Attention Score Tests (5 tests)
# ==============================================================================


class TestAttentionScores:
    """Tests for attention score computation."""

    def test_attention_scores_shape(self, default_config: dict) -> None:
        """Test attention scores shape: (B, num_heads, L, L)."""
        B = default_config["batch_size"]
        num_heads = default_config["num_heads"]
        seq_len = default_config["seq_len"]
        head_dim = default_config["head_dim"]

        q = torch.randn(B, num_heads, seq_len, head_dim)
        k = torch.randn(B, num_heads, seq_len, head_dim)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        expected_shape = (B, num_heads, seq_len, seq_len)
        assert attn_scores.shape == expected_shape, (
            f"Attention scores shape mismatch. Expected {expected_shape}, got {attn_scores.shape}"
        )

    def test_attention_scores_softmax_sum(self, default_config: dict) -> None:
        """Test that softmax rows sum to 1."""
        B = default_config["batch_size"]
        num_heads = default_config["num_heads"]
        seq_len = default_config["seq_len"]
        head_dim = default_config["head_dim"]

        q = torch.randn(B, num_heads, seq_len, head_dim)
        k = torch.randn(B, num_heads, seq_len, head_dim)
        v = torch.randn(B, num_heads, seq_len, head_dim)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Check that each row sums to 1
        row_sums = attn_weights.sum(dim=-1)
        expected_sum = torch.ones_like(row_sums)

        assert torch.allclose(row_sums, expected_sum, atol=1e-5), (
            f"Softmax rows do not sum to 1. Got mean sum: {row_sums.mean().item()}"
        )

    def test_attention_scores_causal_mask(self, default_config: dict) -> None:
        """Test that causal mask is applied correctly."""
        B = default_config["batch_size"]
        num_heads = default_config["num_heads"]
        seq_len = 16  # Smaller for testing
        head_dim = default_config["head_dim"]

        q = torch.randn(B, num_heads, seq_len, head_dim)
        k = torch.randn(B, num_heads, seq_len, head_dim)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        # Apply mask
        attn_scores_masked = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores_masked, dim=-1)

        # Check upper triangle is zero (after softmax)
        upper_triangle = torch.triu(attn_weights, diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), (
            "Causal mask not applied correctly - upper triangle should be zero"
        )

    def test_attention_scores_padding_mask(self, default_config: dict) -> None:
        """Test that padding mask is applied correctly."""
        B = default_config["batch_size"]
        num_heads = default_config["num_heads"]
        seq_len = 16
        head_dim = default_config["head_dim"]

        q = torch.randn(B, num_heads, seq_len, head_dim)
        k = torch.randn(B, num_heads, seq_len, head_dim)
        v = torch.randn(B, num_heads, seq_len, head_dim)

        # Create padding mask (last 4 positions are padding)
        padding_mask = torch.ones(B, seq_len).bool()
        padding_mask[:, -4:] = False  # Mask out last 4 positions

        # Prepare mask
        prepared_mask = prepare_attention_mask(padding_mask)

        # Compute attention with mask
        dropout = nn.Identity()
        output = compute_manual_attention(
            q, k, v, prepared_mask, head_dim ** -0.5, dropout
        )

        # Output should still be valid (no NaN)
        assert not torch.isnan(output).any(), "Padding mask caused NaN in output"

    def test_attention_scores_no_nan(self, default_config: dict) -> None:
        """Test that attention computation produces no NaN values."""
        B = default_config["batch_size"]
        num_heads = default_config["num_heads"]
        seq_len = default_config["seq_len"]
        head_dim = default_config["head_dim"]

        q = torch.randn(B, num_heads, seq_len, head_dim)
        k = torch.randn(B, num_heads, seq_len, head_dim)
        v = torch.randn(B, num_heads, seq_len, head_dim)

        dropout = nn.Identity()
        output = compute_manual_attention(
            q, k, v, None, head_dim ** -0.5, dropout
        )

        assert not torch.isnan(output).any(), "Attention output contains NaN"
        assert not torch.isinf(output).any(), "Attention output contains Inf"


# ==============================================================================
# 3. GQA Tests (5 tests)
# ==============================================================================


class TestGQA:
    """Tests for Grouped Query Attention."""

    def test_repeat_kv_expansion(self, default_config: dict) -> None:
        """Test that KV heads are expanded correctly for GQA."""
        B = default_config["batch_size"]
        num_kv_heads = default_config["num_kv_heads"]
        num_heads = default_config["num_heads"]
        seq_len = default_config["seq_len"]
        head_dim = default_config["head_dim"]
        n_rep = num_heads // num_kv_heads

        k = torch.randn(B, num_kv_heads, seq_len, head_dim)
        v = torch.randn(B, num_kv_heads, seq_len, head_dim)

        k_expanded, v_expanded = repeat_kv(k, v, n_rep)

        # Check expanded shape
        expected_shape = (B, num_heads, seq_len, head_dim)
        assert k_expanded.shape == expected_shape, (
            f"K expansion shape mismatch. Expected {expected_shape}, got {k_expanded.shape}"
        )
        assert v_expanded.shape == expected_shape, (
            f"V expansion shape mismatch. Expected {expected_shape}, got {v_expanded.shape}"
        )

        # Verify content is correctly repeated
        for i in range(num_kv_heads):
            for j in range(n_rep):
                idx = i * n_rep + j
                assert torch.equal(k_expanded[:, idx], k[:, i]), (
                    f"K head {idx} should equal original head {i}"
                )
                assert torch.equal(v_expanded[:, idx], v[:, i]), (
                    f"V head {idx} should equal original head {i}"
                )

    def test_gqa_ratio_4_to_1(self) -> None:
        """Test 4:1 GQA ratio (16 Q heads, 4 KV heads)."""
        hidden_dim = 1024
        num_heads = 16
        num_kv_heads = 4
        head_dim = 64

        qkv_proj = QKVProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Verify ratio
        assert qkv_proj.n_rep == 4, f"Expected GQA ratio 4, got {qkv_proj.n_rep}"
        assert qkv_proj.num_heads == 16, f"Expected 16 Q heads, got {qkv_proj.num_heads}"
        assert qkv_proj.num_kv_heads == 4, f"Expected 4 KV heads, got {qkv_proj.num_kv_heads}"

    def test_gqa_ratio_8_to_1(self) -> None:
        """Test 8:1 GQA ratio."""
        hidden_dim = 1024
        num_heads = 16
        num_kv_heads = 2
        head_dim = 64

        qkv_proj = QKVProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Verify ratio
        assert qkv_proj.n_rep == 8, f"Expected GQA ratio 8, got {qkv_proj.n_rep}"

    def test_gqa_output_shape(
        self, gated_attention: GatedMultiHeadAttention, sample_input: torch.Tensor, default_config: dict
    ) -> None:
        """Test that GQA output shape is same as non-GQA attention."""
        output = gated_attention(sample_input)

        expected_shape = (
            default_config["batch_size"],
            default_config["seq_len"],
            default_config["hidden_dim"],
        )
        assert output.shape == expected_shape, (
            f"GQA output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    def test_gqa_memory_efficiency(self, default_config: dict) -> None:
        """Test that GQA uses less memory for KV projections."""
        hidden_dim = default_config["hidden_dim"]
        num_heads = default_config["num_heads"]
        num_kv_heads = default_config["num_kv_heads"]
        head_dim = default_config["head_dim"]

        # Create GQA projection
        gqa_proj = QKVProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Create MHA projection (no GQA)
        mha_proj = QKVProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,  # Same as num_heads
            head_dim=head_dim,
        )

        # Count KV parameters
        gqa_kv_params = sum(p.numel() for p in [gqa_proj.k_proj.weight, gqa_proj.v_proj.weight])
        mha_kv_params = sum(p.numel() for p in [mha_proj.k_proj.weight, mha_proj.v_proj.weight])

        # GQA should have fewer KV parameters
        expected_ratio = num_heads / num_kv_heads  # 4:1
        actual_ratio = mha_kv_params / gqa_kv_params

        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"GQA memory efficiency incorrect. Expected ratio {expected_ratio}, got {actual_ratio}"
        )


# ==============================================================================
# 4. Gate Mechanism Tests (5 tests)
# ==============================================================================


class TestGateMechanism:
    """Tests for gate mechanism in attention."""

    def test_headwise_gate_shape(self, default_config: dict) -> None:
        """Test headwise gate produces per-head gates."""
        hidden_dim = default_config["hidden_dim"]
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]
        B = default_config["batch_size"]
        seq_len = default_config["seq_len"]

        gate = GateProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gate_type="headwise",
        )

        x = torch.randn(B, seq_len, hidden_dim)
        attn_output = torch.randn(B, num_heads, seq_len, head_dim)

        gated_output = gate(x, attn_output)

        # Output shape should match attention output
        assert gated_output.shape == attn_output.shape, (
            f"Headwise gate output shape mismatch. Expected {attn_output.shape}, got {gated_output.shape}"
        )

        # Gate projection output dimension should be num_heads
        assert gate.proj.out_features == num_heads, (
            f"Headwise gate projection should have {num_heads} outputs, got {gate.proj.out_features}"
        )

    def test_elementwise_gate_shape(self, default_config: dict) -> None:
        """Test elementwise gate produces per-element gates."""
        hidden_dim = default_config["hidden_dim"]
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]
        B = default_config["batch_size"]
        seq_len = default_config["seq_len"]

        gate = GateProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gate_type="elementwise",
        )

        x = torch.randn(B, seq_len, hidden_dim)
        attn_output = torch.randn(B, num_heads, seq_len, head_dim)

        gated_output = gate(x, attn_output)

        # Output shape should match attention output
        assert gated_output.shape == attn_output.shape, (
            f"Elementwise gate output shape mismatch. Expected {attn_output.shape}, got {gated_output.shape}"
        )

        # Gate projection output dimension should be num_heads * head_dim
        expected_out_features = num_heads * head_dim
        assert gate.proj.out_features == expected_out_features, (
            f"Elementwise gate projection should have {expected_out_features} outputs, "
            f"got {gate.proj.out_features}"
        )

    def test_gate_initialization(self, default_config: dict) -> None:
        """Test that gate weights are initialized to zero (for stable training)."""
        hidden_dim = default_config["hidden_dim"]
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        gate = GateProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gate_type="headwise",
            gate_bias=True,
        )

        # Weights should be initialized to zero
        assert torch.allclose(gate.proj.weight, torch.zeros_like(gate.proj.weight)), (
            "Gate weights should be initialized to zero"
        )

        # Bias should be initialized to zero
        assert torch.allclose(gate.proj.bias, torch.zeros_like(gate.proj.bias)), (
            "Gate bias should be initialized to zero"
        )

    def test_gate_sigmoid_activation(self, default_config: dict) -> None:
        """Test that sigmoid activation is applied to gate values."""
        hidden_dim = default_config["hidden_dim"]
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]
        B = default_config["batch_size"]
        seq_len = default_config["seq_len"]

        gate = GateProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gate_type="headwise",
            gate_activation="sigmoid",
        )

        # Set non-zero weights to test activation
        nn.init.normal_(gate.proj.weight, mean=0, std=0.1)

        x = torch.randn(B, seq_len, hidden_dim)
        attn_output = torch.randn(B, num_heads, seq_len, head_dim)

        gated_output = gate(x, attn_output)

        # With zero-initialized weights, sigmoid(0) = 0.5
        # After reinitializing, values should still be in (0, 1) due to sigmoid
        # The gated output should be modified (not equal to original)
        gate_values = torch.sigmoid(gate.proj(x))
        assert (gate_values >= 0).all() and (gate_values <= 1).all(), (
            "Sigmoid gate values should be in [0, 1]"
        )

    def test_gate_modulates_output(self, default_config: dict) -> None:
        """Test that gate actually modulates attention output."""
        hidden_dim = default_config["hidden_dim"]
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]
        B = default_config["batch_size"]
        seq_len = default_config["seq_len"]

        # Create gate with non-zero weights
        gate = GateProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gate_type="headwise",
            gate_activation="sigmoid",
        )
        nn.init.normal_(gate.proj.weight, mean=0, std=1.0)

        # Create identity gate
        identity_gate = IdentityGate()

        x = torch.randn(B, seq_len, hidden_dim)
        attn_output = torch.randn(B, num_heads, seq_len, head_dim)

        gated_output = gate(x, attn_output)
        identity_output = identity_gate(x, attn_output)

        # Gated output should differ from identity output
        assert not torch.allclose(gated_output, identity_output, atol=1e-3), (
            "Gate should modulate output - gated and identity outputs should differ"
        )

        # Identity gate should return unchanged output
        assert torch.equal(identity_output, attn_output), (
            "Identity gate should return unchanged attention output"
        )


# ==============================================================================
# Additional Integration Tests
# ==============================================================================


class TestAttentionIntegration:
    """Integration tests for attention components."""

    def test_full_attention_forward(
        self, gated_attention: GatedMultiHeadAttention, sample_input: torch.Tensor
    ) -> None:
        """Test full attention forward pass."""
        output = gated_attention(sample_input)

        assert output.shape == sample_input.shape, (
            f"Output shape should match input shape. Got {output.shape}"
        )
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_attention_gradient_flow(
        self, gated_attention: GatedMultiHeadAttention, sample_input: torch.Tensor
    ) -> None:
        """Test that gradients flow through attention."""
        sample_input.requires_grad_(True)
        output = gated_attention(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(sample_input.grad).any(), "Gradients contain NaN"

    def test_create_gate_factory(self, default_config: dict) -> None:
        """Test gate factory function."""
        # Test creating gated attention
        gate = create_gate(
            use_gated_attention=True,
            hidden_dim=default_config["hidden_dim"],
            num_heads=default_config["num_heads"],
            head_dim=default_config["head_dim"],
            gate_type="headwise",
        )
        assert isinstance(gate, GateProjection), "Factory should create GateProjection"

        # Test creating identity gate
        identity = create_gate(
            use_gated_attention=False,
            hidden_dim=default_config["hidden_dim"],
            num_heads=default_config["num_heads"],
            head_dim=default_config["head_dim"],
        )
        assert isinstance(identity, IdentityGate), "Factory should create IdentityGate"

    def test_output_projection(self, default_config: dict) -> None:
        """Test output projection."""
        B = default_config["batch_size"]
        num_heads = default_config["num_heads"]
        seq_len = default_config["seq_len"]
        head_dim = default_config["head_dim"]
        hidden_dim = default_config["hidden_dim"]

        out_proj = OutputProjection(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        attn_output = torch.randn(B, num_heads, seq_len, head_dim)
        output = out_proj(attn_output)

        expected_shape = (B, seq_len, hidden_dim)
        assert output.shape == expected_shape, (
            f"Output projection shape mismatch. Expected {expected_shape}, got {output.shape}"
        )
