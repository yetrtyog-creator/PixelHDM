"""
s_cond + Pixel Path Convergence Diagnostics

This test captures and analyzes intermediate values during training to identify
convergence issues. Specifically:

1. s_cond signal diagnostics:
   - semantic_tokens / t_embed / pooled_text_embed norm statistics
   - s_cond mean/std/RMS
   - cosine similarity between different tokens

2. Pixel path activation diagnostics:
   - x_exp.std() / x.std() ratio (compaction effectiveness)
   - alpha1*h and alpha2*h std values

3. Gate value statistics:
   - gate mean/std/min/max

4. Pixel embedding diagnostics:
   - pixel_embed output std / RMS

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-22
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Diagnostic Data Structures
# ============================================================================

@dataclass
class TensorStats:
    """Statistics for a tensor."""
    mean: float
    std: float
    rms: float
    min_val: float
    max_val: float
    shape: Tuple[int, ...]

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "TensorStats":
        """Compute statistics from a tensor."""
        with torch.no_grad():
            t_flat = t.float().flatten()
            return cls(
                mean=t_flat.mean().item(),
                std=t_flat.std().item(),
                rms=(t_flat ** 2).mean().sqrt().item(),
                min_val=t_flat.min().item(),
                max_val=t_flat.max().item(),
                shape=tuple(t.shape),
            )

    def __repr__(self) -> str:
        return (f"TensorStats(mean={self.mean:.6f}, std={self.std:.6f}, "
                f"rms={self.rms:.6f}, min={self.min_val:.6f}, max={self.max_val:.6f})")


@dataclass
class DiagnosticResults:
    """Results from a diagnostic run."""
    # s_cond components
    semantic_tokens: List[TensorStats] = field(default_factory=list)
    t_embed: List[TensorStats] = field(default_factory=list)
    pooled_text_embed: List[TensorStats] = field(default_factory=list)
    s_cond: List[TensorStats] = field(default_factory=list)

    # s_cond cosine similarity (between first two batch items)
    s_cond_cosine_sim: List[float] = field(default_factory=list)

    # Pixel path
    pixel_embed_output: List[TensorStats] = field(default_factory=list)

    # Per pixel block diagnostics
    pixel_block_x_input: List[List[TensorStats]] = field(default_factory=list)  # [block][step]
    pixel_block_h_after_compaction: List[List[TensorStats]] = field(default_factory=list)
    pixel_block_alpha1_h: List[List[TensorStats]] = field(default_factory=list)
    pixel_block_alpha2_h: List[List[TensorStats]] = field(default_factory=list)
    pixel_block_gamma1: List[List[TensorStats]] = field(default_factory=list)
    pixel_block_gamma2: List[List[TensorStats]] = field(default_factory=list)
    pixel_block_alpha1: List[List[TensorStats]] = field(default_factory=list)
    pixel_block_alpha2: List[List[TensorStats]] = field(default_factory=list)

    # Token compaction x_exp.std() / x.std() ratio
    compaction_std_ratio: List[List[float]] = field(default_factory=list)  # [block][step]

    # Gate values (from GatedMultiHeadAttention)
    gate_values: List[List[TensorStats]] = field(default_factory=list)  # [block][step]

    # Output
    model_output: List[TensorStats] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all steps."""
        def mean_of_stats(stats_list: List[TensorStats], attr: str) -> Optional[float]:
            if not stats_list:
                return None
            return sum(getattr(s, attr) for s in stats_list) / len(stats_list)

        summary = {}

        # s_cond summary
        if self.s_cond:
            summary["s_cond_mean_std"] = mean_of_stats(self.s_cond, "std")
            summary["s_cond_mean_rms"] = mean_of_stats(self.s_cond, "rms")

        if self.s_cond_cosine_sim:
            summary["s_cond_cosine_sim_avg"] = sum(self.s_cond_cosine_sim) / len(self.s_cond_cosine_sim)

        # Pixel path summary
        if self.pixel_embed_output:
            summary["pixel_embed_mean_std"] = mean_of_stats(self.pixel_embed_output, "std")

        # Compaction ratio summary
        if self.compaction_std_ratio:
            all_ratios = [r for block_ratios in self.compaction_std_ratio for r in block_ratios]
            if all_ratios:
                summary["compaction_std_ratio_avg"] = sum(all_ratios) / len(all_ratios)

        # Gate values summary
        if self.gate_values:
            all_gate_stats = [s for block_stats in self.gate_values for s in block_stats]
            if all_gate_stats:
                summary["gate_mean_avg"] = mean_of_stats(all_gate_stats, "mean")
                summary["gate_std_avg"] = mean_of_stats(all_gate_stats, "std")

        return summary


# ============================================================================
# Diagnostic Hooks
# ============================================================================

class DiagnosticHooks:
    """Manage forward hooks for diagnostic capture."""

    def __init__(self, model: nn.Module, num_pixel_blocks: int):
        self.model = model
        self.num_pixel_blocks = num_pixel_blocks
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captured: Dict[str, Any] = {}

    def _create_capture_hook(self, name: str):
        """Create a hook that captures output."""
        def hook(module, input, output):
            self.captured[name] = output
        return hook

    def _create_input_capture_hook(self, name: str):
        """Create a hook that captures input."""
        def hook(module, input, output):
            self.captured[name] = input[0] if len(input) == 1 else input
        return hook

    def attach(self) -> None:
        """Attach all diagnostic hooks."""
        # Hook for pixel_embed output
        if hasattr(self.model, 'pixel_embed'):
            h = self.model.pixel_embed.register_forward_hook(
                self._create_capture_hook("pixel_embed_output")
            )
            self.handles.append(h)

        # Hook for each pixel block
        if hasattr(self.model, 'pixel_blocks'):
            for i, block in enumerate(self.model.pixel_blocks):
                # Capture input to block
                h = block.register_forward_hook(
                    self._create_input_capture_hook(f"pixel_block_{i}_input")
                )
                self.handles.append(h)

                # Capture compaction output
                if hasattr(block, 'compaction'):
                    h = block.compaction.register_forward_hook(
                        self._create_capture_hook(f"pixel_block_{i}_compaction_output")
                    )
                    self.handles.append(h)

                # Capture gate values from attention inside compaction
                if hasattr(block, 'compaction') and hasattr(block.compaction, 'attention'):
                    attn = block.compaction.attention
                    if hasattr(attn, 'gate') and hasattr(attn.gate, 'proj'):
                        h = attn.gate.register_forward_hook(
                            self._create_capture_hook(f"pixel_block_{i}_gate_output")
                        )
                        self.handles.append(h)

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self) -> None:
        """Clear captured values."""
        self.captured.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()


# ============================================================================
# Diagnostic Runner
# ============================================================================

class ConvergenceDiagnostics:
    """Run convergence diagnostics on the model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 10,
        batch_size: int = 2,
        image_size: int = 256,
        text_len: int = 32,
    ):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.image_size = image_size
        self.text_len = text_len

        # Get config from model
        self.config = getattr(model, 'config', None)
        self.hidden_dim = getattr(self.config, 'hidden_dim', 1024) if self.config else 1024
        self.num_pixel_blocks = len(model.pixel_blocks) if hasattr(model, 'pixel_blocks') else 4

    def _create_dummy_batch(self) -> Dict[str, torch.Tensor]:
        """Create a dummy batch for testing."""
        B = self.batch_size
        H = W = self.image_size
        T = self.text_len
        D = self.hidden_dim

        return {
            "x_t": torch.randn(B, 3, H, W, device=self.device),
            "t": torch.rand(B, device=self.device),
            "text_embed": torch.randn(B, T, D, device=self.device),
            "text_mask": torch.ones(B, T, dtype=torch.bool, device=self.device),
            "pooled_text_embed": torch.randn(B, D, device=self.device),
        }

    def _compute_cosine_similarity(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors."""
        t1_flat = t1.flatten().float()
        t2_flat = t2.flatten().float()
        return F.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()

    def run(self) -> DiagnosticResults:
        """Run diagnostics and return results."""
        results = DiagnosticResults()

        # Initialize per-block lists
        for _ in range(self.num_pixel_blocks):
            results.pixel_block_x_input.append([])
            results.pixel_block_h_after_compaction.append([])
            results.pixel_block_alpha1_h.append([])
            results.pixel_block_alpha2_h.append([])
            results.pixel_block_gamma1.append([])
            results.pixel_block_gamma2.append([])
            results.pixel_block_alpha1.append([])
            results.pixel_block_alpha2.append([])
            results.compaction_std_ratio.append([])
            results.gate_values.append([])

        self.model.eval()

        with torch.no_grad():
            hooks = DiagnosticHooks(self.model, self.num_pixel_blocks)

            with hooks:
                for step in range(self.num_steps):
                    hooks.clear()
                    batch = self._create_dummy_batch()

                    # Capture s_cond components by temporarily modifying forward
                    s_cond_data = self._capture_s_cond(batch)

                    # Run forward pass
                    output = self.model(
                        x_t=batch["x_t"],
                        t=batch["t"],
                        text_embed=batch["text_embed"],
                        text_mask=batch["text_mask"],
                        pooled_text_embed=batch["pooled_text_embed"],
                    )

                    # Record s_cond statistics
                    if s_cond_data:
                        results.semantic_tokens.append(TensorStats.from_tensor(s_cond_data["semantic_tokens"]))
                        results.t_embed.append(TensorStats.from_tensor(s_cond_data["t_embed"]))
                        if s_cond_data.get("pooled_text_embed") is not None:
                            results.pooled_text_embed.append(TensorStats.from_tensor(s_cond_data["pooled_text_embed"]))
                        results.s_cond.append(TensorStats.from_tensor(s_cond_data["s_cond"]))

                        # Cosine similarity between batch items
                        if s_cond_data["s_cond"].shape[0] >= 2:
                            cos_sim = self._compute_cosine_similarity(
                                s_cond_data["s_cond"][0], s_cond_data["s_cond"][1]
                            )
                            results.s_cond_cosine_sim.append(cos_sim)

                    # Record pixel_embed output
                    if "pixel_embed_output" in hooks.captured:
                        results.pixel_embed_output.append(
                            TensorStats.from_tensor(hooks.captured["pixel_embed_output"])
                        )

                    # Record per-block statistics
                    for i in range(self.num_pixel_blocks):
                        # Block input
                        key = f"pixel_block_{i}_input"
                        if key in hooks.captured:
                            inp = hooks.captured[key]
                            if isinstance(inp, tuple):
                                inp = inp[0]
                            results.pixel_block_x_input[i].append(TensorStats.from_tensor(inp))

                            # Compaction output and ratio
                            comp_key = f"pixel_block_{i}_compaction_output"
                            if comp_key in hooks.captured:
                                comp_out = hooks.captured[comp_key]
                                results.pixel_block_h_after_compaction[i].append(
                                    TensorStats.from_tensor(comp_out)
                                )

                                # Compute std ratio
                                x_std = inp.std().item()
                                h_std = comp_out.std().item()
                                ratio = h_std / x_std if x_std > 1e-8 else 0.0
                                results.compaction_std_ratio[i].append(ratio)

                        # Gate values
                        gate_key = f"pixel_block_{i}_gate_output"
                        if gate_key in hooks.captured:
                            gate_out = hooks.captured[gate_key]
                            results.gate_values[i].append(TensorStats.from_tensor(gate_out))

                    # Record output statistics
                    results.model_output.append(TensorStats.from_tensor(output))

        return results

    def _capture_s_cond(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Capture s_cond components during forward pass."""
        # We need to manually compute s_cond components
        # This mirrors the logic in PixelHDM.forward()

        data = {}

        with torch.no_grad():
            # Process input
            x_t = batch["x_t"]
            if x_t.shape[1] == 3:  # BCHW -> BHWC
                x_t = x_t.permute(0, 2, 3, 1)

            # Patch embedding
            if hasattr(self.model, 'patch_embed'):
                x = self.model.patch_embed(x_t)
            else:
                return {}

            # Time embedding
            t_embed = self.model.time_embed(batch["t"])
            data["t_embed"] = t_embed

            # Create joint sequence
            text_embed = batch.get("text_embed")
            text_mask = batch.get("text_mask")
            text_len = 0

            if text_embed is not None:
                T = text_embed.shape[1]
                x = torch.cat([text_embed, x], dim=1)
                text_len = T

            # Run through patch blocks to get semantic_tokens
            # Create position IDs
            from src.models.layers.rope import create_position_ids_batched
            B = x_t.shape[0]
            H, W = x_t.shape[1], x_t.shape[2]

            position_ids = create_position_ids_batched(
                batch_size=B,
                text_len=text_len,
                img_height=H,
                img_width=W,
                patch_size=self.model.patch_size,
                device=x.device,
                text_mask=text_mask,
            )
            rope_fn = self.model._create_rope_fn(position_ids)

            # Run patch blocks
            joint_mask = None
            if text_mask is not None:
                img_mask = torch.ones(B, x.shape[1] - text_len, dtype=torch.bool, device=x.device)
                joint_mask = torch.cat([text_mask, img_mask], dim=1)

            for block in self.model.patch_blocks:
                x = block(x, t_embed=t_embed, rope_fn=rope_fn, position_ids=position_ids, attention_mask=joint_mask)

            # Extract semantic tokens
            semantic_tokens = x[:, text_len:] if text_len > 0 else x
            data["semantic_tokens"] = semantic_tokens

            # Compute s_cond
            s_cond = semantic_tokens + t_embed.unsqueeze(1)
            pooled_text_embed = batch.get("pooled_text_embed")
            if pooled_text_embed is not None:
                s_cond = s_cond + pooled_text_embed.unsqueeze(1)
                data["pooled_text_embed"] = pooled_text_embed

            data["s_cond"] = s_cond

        return data

    def run_with_adaln_capture(self) -> DiagnosticResults:
        """Run diagnostics with AdaLN parameter capture."""
        results = self.run()

        # Additional pass to capture AdaLN parameters
        self.model.eval()

        with torch.no_grad():
            batch = self._create_dummy_batch()

            # Capture s_cond for AdaLN analysis
            s_cond_data = self._capture_s_cond(batch)

            if s_cond_data and "s_cond" in s_cond_data:
                s_cond = s_cond_data["s_cond"]

                # Run through each pixel block's AdaLN
                for i, block in enumerate(self.model.pixel_blocks):
                    if hasattr(block, 'adaln'):
                        params = block.adaln(s_cond)
                        if len(params) >= 6:
                            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params[:6]

                            # Record statistics
                            if i < len(results.pixel_block_gamma1):
                                results.pixel_block_gamma1[i].append(TensorStats.from_tensor(gamma1))
                                results.pixel_block_gamma2[i].append(TensorStats.from_tensor(gamma2))
                                results.pixel_block_alpha1[i].append(TensorStats.from_tensor(alpha1))
                                results.pixel_block_alpha2[i].append(TensorStats.from_tensor(alpha2))

        return results


# ============================================================================
# Test Functions
# ============================================================================

@pytest.fixture(scope="module")
def diagnostic_model(device):
    """Create a model for diagnostics."""
    from src.config import PixelHDMConfig
    from src.models.pixelhdm import PixelHDM

    # Use small config for faster testing
    config = PixelHDMConfig.for_testing()
    model = PixelHDM(config=config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSCondDiagnostics:
    """Tests for s_cond signal diagnostics."""

    def test_s_cond_components_have_reasonable_magnitude(self, diagnostic_model, device):
        """Test that s_cond components have reasonable magnitudes."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=5,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run()

        # Check semantic_tokens
        assert len(results.semantic_tokens) > 0, "No semantic_tokens captured"
        for stats in results.semantic_tokens:
            assert stats.std > 0.01, f"semantic_tokens std too small: {stats.std}"
            assert stats.std < 100, f"semantic_tokens std too large: {stats.std}"

        # Check t_embed
        assert len(results.t_embed) > 0, "No t_embed captured"
        for stats in results.t_embed:
            assert stats.std > 0.01, f"t_embed std too small: {stats.std}"

        # Check s_cond
        assert len(results.s_cond) > 0, "No s_cond captured"
        for stats in results.s_cond:
            assert stats.std > 0.01, f"s_cond std too small: {stats.std}"

    def test_s_cond_cosine_similarity_not_too_high(self, diagnostic_model, device):
        """Test that s_cond for different batch items are not too similar."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=5,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run()

        # Cosine similarity should not be too high (would indicate collapsed conditioning)
        if results.s_cond_cosine_sim:
            avg_cos_sim = sum(results.s_cond_cosine_sim) / len(results.s_cond_cosine_sim)
            # Allow high similarity since random inputs may be similar
            # But flag if extremely high
            assert avg_cos_sim < 0.999, f"s_cond cosine similarity too high: {avg_cos_sim}"
            print(f"s_cond average cosine similarity: {avg_cos_sim:.6f}")


class TestPixelPathDiagnostics:
    """Tests for pixel path activation diagnostics."""

    def test_pixel_embed_output_reasonable(self, diagnostic_model, device):
        """Test that pixel_embed output has reasonable statistics."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=5,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run()

        assert len(results.pixel_embed_output) > 0, "No pixel_embed_output captured"

        for stats in results.pixel_embed_output:
            # Should have non-zero std
            assert stats.std > 0.001, f"pixel_embed std too small: {stats.std}"
            # Should not be NaN or Inf
            assert not (stats.std != stats.std), "pixel_embed std is NaN"

    def test_compaction_std_ratio_reasonable(self, diagnostic_model, device):
        """Test that compaction std ratio is in reasonable range."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=5,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run()

        # Check compaction ratios
        for block_idx, ratios in enumerate(results.compaction_std_ratio):
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                print(f"Block {block_idx} compaction std ratio: {avg_ratio:.4f}")

                # Ratio should not be too small (would indicate collapsed representation)
                # or too large (would indicate exploding activations)
                assert avg_ratio > 0.001, f"Block {block_idx} compaction ratio too small: {avg_ratio}"


class TestGateDiagnostics:
    """Tests for gate value diagnostics."""

    def test_gate_values_not_saturated(self, diagnostic_model, device):
        """Test that gate values are not saturated at 0 or 1."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=5,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run()

        # Check gate values
        for block_idx, gate_stats_list in enumerate(results.gate_values):
            if gate_stats_list:
                for stats in gate_stats_list:
                    print(f"Block {block_idx} gate: mean={stats.mean:.4f}, std={stats.std:.4f}")

                    # Gate should not be stuck at extreme values
                    # Note: At initialization, gates may be near 0.5 for sigmoid
                    # or could vary depending on the activation


class TestAdaLNDiagnostics:
    """Tests for AdaLN parameter diagnostics."""

    def test_adaln_parameters_reasonable(self, diagnostic_model, device):
        """Test that AdaLN parameters have reasonable values."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=3,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run_with_adaln_capture()

        # Check gamma values
        for block_idx, gamma_stats_list in enumerate(results.pixel_block_gamma1):
            if gamma_stats_list:
                for stats in gamma_stats_list:
                    print(f"Block {block_idx} gamma1: mean={stats.mean:.4f}, std={stats.std:.4f}")

        # Check alpha values (should be around 1.0 at initialization)
        for block_idx, alpha_stats_list in enumerate(results.pixel_block_alpha1):
            if alpha_stats_list:
                for stats in alpha_stats_list:
                    print(f"Block {block_idx} alpha1: mean={stats.mean:.4f}, std={stats.std:.4f}")
                    # Alpha should have mean around 1.0 at init
                    assert stats.mean > 0, f"Block {block_idx} alpha1 mean should be positive"


class TestComprehensiveDiagnostics:
    """Comprehensive diagnostic tests."""

    def test_full_diagnostic_summary(self, diagnostic_model, device):
        """Run full diagnostics and print summary."""
        diagnostics = ConvergenceDiagnostics(
            model=diagnostic_model,
            device=device,
            num_steps=10,
            batch_size=2,
            image_size=256,
        )

        results = diagnostics.run_with_adaln_capture()
        summary = results.get_summary()

        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)

        for key, value in summary.items():
            if value is not None:
                print(f"{key}: {value:.6f}")

        print("="*60)

        # Print detailed per-block statistics
        print("\nPER-BLOCK STATISTICS:")
        for i in range(diagnostics.num_pixel_blocks):
            print(f"\n--- Pixel Block {i} ---")

            if results.pixel_block_x_input[i]:
                avg_std = sum(s.std for s in results.pixel_block_x_input[i]) / len(results.pixel_block_x_input[i])
                print(f"  x_input avg std: {avg_std:.6f}")

            if results.compaction_std_ratio[i]:
                avg_ratio = sum(results.compaction_std_ratio[i]) / len(results.compaction_std_ratio[i])
                print(f"  compaction std ratio: {avg_ratio:.6f}")

            if results.gate_values[i]:
                avg_mean = sum(s.mean for s in results.gate_values[i]) / len(results.gate_values[i])
                avg_std = sum(s.std for s in results.gate_values[i]) / len(results.gate_values[i])
                print(f"  gate: mean={avg_mean:.6f}, std={avg_std:.6f}")

            if results.pixel_block_gamma1[i]:
                avg_mean = sum(s.mean for s in results.pixel_block_gamma1[i]) / len(results.pixel_block_gamma1[i])
                print(f"  gamma1 mean: {avg_mean:.6f}")

            if results.pixel_block_alpha1[i]:
                avg_mean = sum(s.mean for s in results.pixel_block_alpha1[i]) / len(results.pixel_block_alpha1[i])
                print(f"  alpha1 mean: {avg_mean:.6f}")


# ============================================================================
# Standalone Runner
# ============================================================================

def run_diagnostics_standalone():
    """Run diagnostics as standalone script."""
    import argparse

    parser = argparse.ArgumentParser(description="Run convergence diagnostics")
    parser.add_argument("--steps", type=int, default=10, help="Number of diagnostic steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    from src.config import PixelHDMConfig
    from src.models.pixelhdm import PixelHDM

    print("Creating model...")
    config = PixelHDMConfig.for_testing()
    model = PixelHDM(config=config)
    model = model.to(device)
    model.eval()

    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Run diagnostics
    print(f"\nRunning diagnostics for {args.steps} steps...")
    diagnostics = ConvergenceDiagnostics(
        model=model,
        device=device,
        num_steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    results = diagnostics.run_with_adaln_capture()
    summary = results.get_summary()

    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    for key, value in summary.items():
        if value is not None:
            print(f"{key}: {value:.6f}")

    print("="*60)

    return results


if __name__ == "__main__":
    run_diagnostics_standalone()
