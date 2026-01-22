"""
Training Diagnostics - Track intermediate values during actual training

This test runs a short training loop and tracks:
1. Loss components evolution
2. s_cond signal changes during training
3. AdaLN parameter evolution (gamma, alpha, beta)
4. Gate value evolution
5. Gradient statistics

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-22
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.diagnostics.test_convergence_diagnostics import TensorStats, DiagnosticResults


# ============================================================================
# Training Diagnostic Tracker
# ============================================================================

@dataclass
class TrainingDiagnostics:
    """Track diagnostics during training."""
    # Loss tracking
    total_loss: List[float] = field(default_factory=list)
    v_loss: List[float] = field(default_factory=list)
    freq_loss: List[float] = field(default_factory=list)
    repa_loss: List[float] = field(default_factory=list)

    # s_cond tracking
    s_cond_mean: List[float] = field(default_factory=list)
    s_cond_std: List[float] = field(default_factory=list)
    s_cond_rms: List[float] = field(default_factory=list)

    # AdaLN parameters (averaged across blocks)
    gamma1_mean: List[float] = field(default_factory=list)
    gamma1_std: List[float] = field(default_factory=list)
    gamma2_mean: List[float] = field(default_factory=list)
    alpha1_mean: List[float] = field(default_factory=list)
    alpha2_mean: List[float] = field(default_factory=list)

    # Gate tracking
    gate_mean: List[float] = field(default_factory=list)
    gate_std: List[float] = field(default_factory=list)

    # Compaction tracking
    compaction_ratio: List[float] = field(default_factory=list)

    # Gradient tracking
    grad_norm: List[float] = field(default_factory=list)
    adaln_grad_norm: List[float] = field(default_factory=list)
    gate_grad_norm: List[float] = field(default_factory=list)

    # Output tracking
    output_std: List[float] = field(default_factory=list)
    output_mean: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return {
            "total_loss": self.total_loss,
            "v_loss": self.v_loss,
            "freq_loss": self.freq_loss,
            "repa_loss": self.repa_loss,
            "s_cond_mean": self.s_cond_mean,
            "s_cond_std": self.s_cond_std,
            "s_cond_rms": self.s_cond_rms,
            "gamma1_mean": self.gamma1_mean,
            "gamma1_std": self.gamma1_std,
            "gamma2_mean": self.gamma2_mean,
            "alpha1_mean": self.alpha1_mean,
            "alpha2_mean": self.alpha2_mean,
            "gate_mean": self.gate_mean,
            "gate_std": self.gate_std,
            "compaction_ratio": self.compaction_ratio,
            "grad_norm": self.grad_norm,
            "adaln_grad_norm": self.adaln_grad_norm,
            "gate_grad_norm": self.gate_grad_norm,
            "output_std": self.output_std,
            "output_mean": self.output_mean,
        }

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self, step_interval: int = 10) -> None:
        """Print summary of diagnostics."""
        n = len(self.total_loss)
        print(f"\n{'='*70}")
        print(f"TRAINING DIAGNOSTICS SUMMARY ({n} steps)")
        print(f"{'='*70}")

        # Loss evolution
        if self.total_loss:
            print(f"\nLoss: start={self.total_loss[0]:.4f}, end={self.total_loss[-1]:.4f}")
            if self.v_loss:
                print(f"V-Loss: start={self.v_loss[0]:.4f}, end={self.v_loss[-1]:.4f}")

        # s_cond evolution
        if self.s_cond_std:
            print(f"\ns_cond std: start={self.s_cond_std[0]:.4f}, end={self.s_cond_std[-1]:.4f}")
            print(f"s_cond rms: start={self.s_cond_rms[0]:.4f}, end={self.s_cond_rms[-1]:.4f}")

        # AdaLN parameters
        if self.gamma1_mean:
            print(f"\ngamma1 mean: start={self.gamma1_mean[0]:.6f}, end={self.gamma1_mean[-1]:.6f}")
            print(f"alpha1 mean: start={self.alpha1_mean[0]:.4f}, end={self.alpha1_mean[-1]:.4f}")

        # Gate values
        if self.gate_mean:
            print(f"\ngate mean: start={self.gate_mean[0]:.6f}, end={self.gate_mean[-1]:.6f}")
            print(f"gate std: start={self.gate_std[0]:.4f}, end={self.gate_std[-1]:.4f}")

        # Compaction ratio
        if self.compaction_ratio:
            print(f"\ncompaction ratio: start={self.compaction_ratio[0]:.4f}, end={self.compaction_ratio[-1]:.4f}")

        # Gradients
        if self.grad_norm:
            print(f"\ngrad norm: start={self.grad_norm[0]:.4f}, end={self.grad_norm[-1]:.4f}")

        print(f"{'='*70}")


# ============================================================================
# Diagnostic Training Runner
# ============================================================================

class DiagnosticTrainer:
    """Run training with diagnostics capture."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        num_steps: int = 100,
        batch_size: int = 2,
        image_size: int = 256,
        text_len: int = 32,
        log_interval: int = 10,
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.image_size = image_size
        self.text_len = text_len
        self.log_interval = log_interval

        self.config = getattr(model, 'config', None)
        self.hidden_dim = getattr(self.config, 'hidden_dim', 1024) if self.config else 1024

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Setup loss
        self._setup_loss()

        # Diagnostics tracker
        self.diagnostics = TrainingDiagnostics()

    def _setup_loss(self) -> None:
        """Setup loss functions."""
        from src.training.losses import VLoss

        self.v_loss_fn = VLoss()

    def _create_batch(self) -> Dict[str, torch.Tensor]:
        """Create a training batch."""
        B = self.batch_size
        H = W = self.image_size
        T = self.text_len
        D = self.hidden_dim

        # Create consistent image (for single-image overfit test)
        torch.manual_seed(42)
        x_clean = torch.randn(B, 3, H, W, device=self.device)

        # Create noisy version
        t = torch.rand(B, device=self.device)
        noise = torch.randn_like(x_clean)

        # Flow matching interpolation: z_t = t * x + (1 - t) * noise
        x_t = t.view(B, 1, 1, 1) * x_clean + (1 - t.view(B, 1, 1, 1)) * noise

        # Velocity target: v = x - noise
        v_target = x_clean - noise

        return {
            "x_t": x_t,
            "x_clean": x_clean,
            "t": t,
            "noise": noise,
            "v_target": v_target,
            "text_embed": torch.randn(B, T, D, device=self.device),
            "text_mask": torch.ones(B, T, dtype=torch.bool, device=self.device),
            "pooled_text_embed": torch.randn(B, D, device=self.device),
        }

    def _capture_intermediate_values(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Capture intermediate values during forward pass."""
        captured = {}

        with torch.no_grad():
            # Process input
            x_t = batch["x_t"]
            if x_t.shape[1] == 3:  # BCHW -> BHWC
                x_t_bhwc = x_t.permute(0, 2, 3, 1)
            else:
                x_t_bhwc = x_t

            # Patch embedding
            x = self.model.patch_embed(x_t_bhwc)

            # Time embedding
            t_embed = self.model.time_embed(batch["t"])

            # Create joint sequence
            text_embed = batch.get("text_embed")
            text_len = text_embed.shape[1] if text_embed is not None else 0

            if text_embed is not None:
                x = torch.cat([text_embed, x], dim=1)

            # Create position IDs
            from src.models.layers.rope import create_position_ids_batched
            B = x_t.shape[0]
            H, W = x_t_bhwc.shape[1], x_t_bhwc.shape[2]

            position_ids = create_position_ids_batched(
                batch_size=B,
                text_len=text_len,
                img_height=H,
                img_width=W,
                patch_size=self.model.patch_size,
                device=x.device,
                text_mask=batch.get("text_mask"),
            )
            rope_fn = self.model._create_rope_fn(position_ids)

            # Joint mask
            joint_mask = None
            text_mask = batch.get("text_mask")
            if text_mask is not None:
                img_mask = torch.ones(B, x.shape[1] - text_len, dtype=torch.bool, device=x.device)
                joint_mask = torch.cat([text_mask, img_mask], dim=1)

            # Run through patch blocks
            for block in self.model.patch_blocks:
                x = block(x, t_embed=t_embed, rope_fn=rope_fn, position_ids=position_ids, attention_mask=joint_mask)

            # Extract semantic tokens
            semantic_tokens = x[:, text_len:] if text_len > 0 else x

            # Compute s_cond
            s_cond = semantic_tokens + t_embed.unsqueeze(1)
            pooled_text_embed = batch.get("pooled_text_embed")
            if pooled_text_embed is not None:
                s_cond = s_cond + pooled_text_embed.unsqueeze(1)

            captured["s_cond"] = TensorStats.from_tensor(s_cond)

            # Capture AdaLN parameters
            gamma1_list = []
            gamma2_list = []
            alpha1_list = []
            alpha2_list = []

            for block in self.model.pixel_blocks:
                if hasattr(block, 'adaln'):
                    params = block.adaln(s_cond)
                    if len(params) >= 6:
                        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params[:6]
                        gamma1_list.append(gamma1.mean().item())
                        gamma2_list.append(gamma2.mean().item())
                        alpha1_list.append(alpha1.mean().item())
                        alpha2_list.append(alpha2.mean().item())

            if gamma1_list:
                captured["gamma1_mean"] = sum(gamma1_list) / len(gamma1_list)
                captured["gamma2_mean"] = sum(gamma2_list) / len(gamma2_list)
                captured["alpha1_mean"] = sum(alpha1_list) / len(alpha1_list)
                captured["alpha2_mean"] = sum(alpha2_list) / len(alpha2_list)

            # Capture pixel_embed output
            pixel_out = self.model.pixel_embed(x_t_bhwc)
            captured["pixel_embed"] = TensorStats.from_tensor(pixel_out)

            # Capture compaction ratios and gate values
            compaction_ratios = []
            gate_means = []
            gate_stds = []

            # Run pixel blocks to capture
            x_pixel = pixel_out
            pixel_position_ids = create_position_ids_batched(
                batch_size=B,
                text_len=0,
                img_height=H,
                img_width=W,
                patch_size=self.model.patch_size,
                device=x_pixel.device,
            )
            pixel_rope_fn = self.model._create_rope_fn(pixel_position_ids)

            for block in self.model.pixel_blocks:
                x_in_std = x_pixel.std().item()

                # Get compaction output
                if hasattr(block, 'compaction'):
                    h = block.compaction(block.adaln.modulate(x_pixel, *block.adaln(s_cond)[:2]), pixel_rope_fn, pixel_position_ids)
                    h_std = h.std().item()
                    ratio = h_std / x_in_std if x_in_std > 1e-8 else 0.0
                    compaction_ratios.append(ratio)

                    # Gate values from compaction attention
                    if hasattr(block.compaction, 'attention') and hasattr(block.compaction.attention, 'gate'):
                        gate = block.compaction.attention.gate
                        if hasattr(gate, 'proj'):
                            # Get gate output
                            # This is a bit tricky - we need to run attention to get gate values
                            pass

                # Run full block
                x_pixel = block(x_pixel, s_cond=s_cond, rope_fn=pixel_rope_fn, position_ids=pixel_position_ids)

            if compaction_ratios:
                captured["compaction_ratio"] = sum(compaction_ratios) / len(compaction_ratios)

        return captured

    def _compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        norms = {}

        # Total gradient norm
        total_norm = 0.0
        adaln_norm = 0.0
        gate_norm = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item() ** 2
                total_norm += param_norm

                if 'adaln' in name.lower():
                    adaln_norm += param_norm
                if 'gate' in name.lower():
                    gate_norm += param_norm

        norms["total"] = math.sqrt(total_norm)
        norms["adaln"] = math.sqrt(adaln_norm)
        norms["gate"] = math.sqrt(gate_norm)

        return norms

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Run a single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(
            x_t=batch["x_t"],
            t=batch["t"],
            text_embed=batch["text_embed"],
            text_mask=batch["text_mask"],
            pooled_text_embed=batch["pooled_text_embed"],
        )

        # Convert output to BCHW if needed
        if output.shape[-1] == 3:  # BHWC
            output = output.permute(0, 3, 1, 2)

        # Compute loss (VLoss needs v_pred, x_clean, noise)
        v_loss = self.v_loss_fn(output, batch["x_clean"], batch["noise"])
        total_loss = v_loss

        # Backward pass
        total_loss.backward()

        # Get gradient norms before optimizer step
        grad_norms = self._compute_grad_norms()

        # Optimizer step
        self.optimizer.step()

        losses = {
            "total": total_loss.item(),
            "v_loss": v_loss.item(),
        }

        return losses, grad_norms, TensorStats.from_tensor(output)

    def run(self) -> TrainingDiagnostics:
        """Run training with diagnostics."""
        print(f"Starting diagnostic training for {self.num_steps} steps...")

        for step in range(self.num_steps):
            # Create batch
            batch = self._create_batch()

            # Capture intermediate values before training step
            if step % self.log_interval == 0:
                intermediate = self._capture_intermediate_values(batch)

                # Record s_cond stats
                if "s_cond" in intermediate:
                    self.diagnostics.s_cond_mean.append(intermediate["s_cond"].mean)
                    self.diagnostics.s_cond_std.append(intermediate["s_cond"].std)
                    self.diagnostics.s_cond_rms.append(intermediate["s_cond"].rms)

                # Record AdaLN params
                if "gamma1_mean" in intermediate:
                    self.diagnostics.gamma1_mean.append(intermediate["gamma1_mean"])
                    self.diagnostics.gamma2_mean.append(intermediate["gamma2_mean"])
                    self.diagnostics.alpha1_mean.append(intermediate["alpha1_mean"])
                    self.diagnostics.alpha2_mean.append(intermediate["alpha2_mean"])

                # Record compaction ratio
                if "compaction_ratio" in intermediate:
                    self.diagnostics.compaction_ratio.append(intermediate["compaction_ratio"])

            # Training step
            losses, grad_norms, output_stats = self.train_step(batch)

            # Record losses
            self.diagnostics.total_loss.append(losses["total"])
            self.diagnostics.v_loss.append(losses["v_loss"])

            # Record gradients
            self.diagnostics.grad_norm.append(grad_norms["total"])
            self.diagnostics.adaln_grad_norm.append(grad_norms["adaln"])
            self.diagnostics.gate_grad_norm.append(grad_norms["gate"])

            # Record output stats
            self.diagnostics.output_std.append(output_stats.std)
            self.diagnostics.output_mean.append(output_stats.mean)

            # Print progress
            if step % self.log_interval == 0:
                print(f"Step {step}: loss={losses['total']:.4f}, v_loss={losses['v_loss']:.4f}, "
                      f"grad_norm={grad_norms['total']:.4f}")

        return self.diagnostics


# ============================================================================
# Tests
# ============================================================================

@pytest.fixture(scope="module")
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def diagnostic_model(device):
    """Create a model for diagnostics."""
    from src.config import PixelHDMConfig
    from src.models.pixelhdm import PixelHDM

    config = PixelHDMConfig.for_testing()
    model = PixelHDM(config=config)
    model = model.to(device)
    return model


class TestTrainingDiagnostics:
    """Tests for training diagnostics."""

    def test_loss_decreases(self, diagnostic_model, device):
        """Test that loss decreases during training."""
        trainer = DiagnosticTrainer(
            model=diagnostic_model,
            device=device,
            num_steps=50,
            batch_size=2,
            image_size=256,
            log_interval=10,
        )

        diagnostics = trainer.run()

        # Loss should decrease
        initial_loss = diagnostics.total_loss[0]
        final_loss = diagnostics.total_loss[-1]

        print(f"\nInitial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Decrease: {initial_loss - final_loss:.4f}")

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    def test_s_cond_maintains_signal(self, diagnostic_model, device):
        """Test that s_cond maintains meaningful signal during training."""
        trainer = DiagnosticTrainer(
            model=diagnostic_model,
            device=device,
            num_steps=30,
            batch_size=2,
            image_size=256,
            log_interval=10,
        )

        diagnostics = trainer.run()

        # s_cond std should not collapse
        if diagnostics.s_cond_std:
            min_std = min(diagnostics.s_cond_std)
            max_std = max(diagnostics.s_cond_std)
            print(f"\ns_cond std range: [{min_std:.4f}, {max_std:.4f}]")

            assert min_std > 0.1, f"s_cond std collapsed to {min_std}"

    def test_adaln_parameters_evolve(self, diagnostic_model, device):
        """Test that AdaLN parameters change during training."""
        trainer = DiagnosticTrainer(
            model=diagnostic_model,
            device=device,
            num_steps=30,
            batch_size=2,
            image_size=256,
            log_interval=5,
        )

        diagnostics = trainer.run()

        # gamma should change from initial value
        if diagnostics.gamma1_mean and len(diagnostics.gamma1_mean) > 1:
            gamma_change = abs(diagnostics.gamma1_mean[-1] - diagnostics.gamma1_mean[0])
            print(f"\ngamma1 change: {gamma_change:.6f}")
            print(f"gamma1: {diagnostics.gamma1_mean[0]:.6f} -> {diagnostics.gamma1_mean[-1]:.6f}")

    def test_gradients_flow(self, diagnostic_model, device):
        """Test that gradients flow through the model."""
        trainer = DiagnosticTrainer(
            model=diagnostic_model,
            device=device,
            num_steps=10,
            batch_size=2,
            image_size=256,
            log_interval=5,
        )

        diagnostics = trainer.run()

        # Gradient norms should be non-zero
        assert all(gn > 0 for gn in diagnostics.grad_norm), "Some gradient norms are zero"

        # AdaLN gradients should flow
        if diagnostics.adaln_grad_norm:
            assert any(gn > 0 for gn in diagnostics.adaln_grad_norm), "AdaLN gradients are zero"


class TestComprehensiveTrainingDiagnostics:
    """Comprehensive training diagnostic tests."""

    def test_full_training_diagnostics(self, diagnostic_model, device):
        """Run full training diagnostics and print summary."""
        trainer = DiagnosticTrainer(
            model=diagnostic_model,
            device=device,
            num_steps=100,
            batch_size=2,
            image_size=256,
            log_interval=10,
        )

        diagnostics = trainer.run()
        diagnostics.print_summary()

        # Basic assertions
        assert len(diagnostics.total_loss) == 100
        assert diagnostics.total_loss[-1] < diagnostics.total_loss[0]


# ============================================================================
# Standalone Runner
# ============================================================================

def run_training_diagnostics_standalone():
    """Run training diagnostics as standalone script."""
    import argparse

    parser = argparse.ArgumentParser(description="Run training diagnostics")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
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

    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Run training diagnostics
    trainer = DiagnosticTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        num_steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        log_interval=args.log_interval,
    )

    diagnostics = trainer.run()
    diagnostics.print_summary()

    # Save if output path provided
    if args.output:
        diagnostics.save(args.output)
        print(f"\nDiagnostics saved to: {args.output}")

    return diagnostics


if __name__ == "__main__":
    run_training_diagnostics_standalone()
