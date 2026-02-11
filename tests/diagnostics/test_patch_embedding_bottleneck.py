"""
Patch Embedding Bottleneck Diagnostic - With vs Without Bottleneck Comparison

This script tests the impact of the bottleneck design in PatchEmbedding:
- WITH bottleneck: input(768) → down(256) → SiLU → up(1024)
- WITHOUT bottleneck: input(768) → Linear → hidden(1024)

Test parameters:
- 1024 training steps
- 5e-4 learning rate
- Single image overfit test
- Output visualization comparison

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-23
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PIL import Image
import math
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Custom PatchEmbedding without Bottleneck
# ============================================================================

class PatchEmbeddingDirect(nn.Module):
    """
    Patch Embedding WITHOUT Bottleneck (Direct Linear Projection)

    Flow: (B, H, W, 3) -> unfold -> (B, L, p^2 * 3) -> Linear -> (B, L, D)

    This is the comparison baseline to test if bottleneck affects convergence.
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.p2 = patch_size ** 2

        input_dim = in_channels * self.p2  # 3 * 256 = 768
        # Direct projection: no bottleneck
        self.proj = nn.Linear(input_dim, hidden_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        B, H, W, C = x.shape

        self._validate_dimensions(H, W)

        x = self._unfold_patches(x, B, H, W)
        x = self.proj(x)  # Direct linear projection

        return x

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input should be 4D, got: {x.dim()}")

        if x.shape[-1] == self.in_channels:
            return x
        return x.permute(0, 2, 3, 1)

    def _validate_dimensions(self, H: int, W: int) -> None:
        assert H % self.patch_size == 0, f"H={H} not divisible by patch_size={self.patch_size}"
        assert W % self.patch_size == 0, f"W={W} not divisible by patch_size={self.patch_size}"

    def _unfold_patches(self, x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        L = h_patches * w_patches

        x = x.reshape(B, h_patches, p, w_patches, p, self.in_channels)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, L, self.p2 * self.in_channels)

        return x


# ============================================================================
# Monkey-patch helper to swap PatchEmbedding
# ============================================================================

def create_model_with_direct_embedding(config):
    """Create PixelHDM with direct (no bottleneck) patch embedding."""
    from src.models.pixelhdm import PixelHDM
    from src.models.layers.embedding import PatchEmbedding

    # Create normal model
    model = PixelHDM(config=config)

    # Replace patch_embed with direct version
    model.patch_embed = PatchEmbeddingDirect(
        patch_size=config.patch_size,
        hidden_dim=config.hidden_dim,
        in_channels=config.in_channels,
    )

    return model


# ============================================================================
# Diagnostic Results
# ============================================================================

@dataclass
class DiagnosticStep:
    """Metrics for a single training step."""
    step: int
    loss: float
    correlation: float


@dataclass
class DiagnosticResults:
    """Complete diagnostic results for a run."""
    config_name: str
    has_bottleneck: bool
    patch_layers: int
    pixel_layers: int
    hidden_dim: int
    total_params: int
    steps: List[DiagnosticStep] = field(default_factory=list)
    elapsed_time: float = 0.0
    final_loss: float = 0.0
    final_correlation: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "has_bottleneck": self.has_bottleneck,
            "patch_layers": self.patch_layers,
            "pixel_layers": self.pixel_layers,
            "hidden_dim": self.hidden_dim,
            "total_params": self.total_params,
            "elapsed_time": self.elapsed_time,
            "final_loss": self.final_loss,
            "final_correlation": self.final_correlation,
            "steps": [{"step": s.step, "loss": s.loss, "correlation": s.correlation} for s in self.steps]
        }


# ============================================================================
# Bottleneck Diagnostic Runner
# ============================================================================

class BottleneckDiagnostic:
    """Diagnostic class for bottleneck comparison."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_path: str,
        config_name: str = "unknown",
        has_bottleneck: bool = True,
        image_size: int = 256,
        learning_rate: float = 5e-4,
        num_steps: int = 1024,
        log_interval: int = 10,
        seed: int = 42,
    ):
        self.model = model
        self.device = device
        self.image_path = image_path
        self.config_name = config_name
        self.has_bottleneck = has_bottleneck
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.seed = seed

        # Get model config
        self.config = getattr(model, 'config', None)
        self.hidden_dim = getattr(self.config, 'hidden_dim', 1024) if self.config else 1024
        self.patch_layers = len(model.patch_blocks) if hasattr(model, 'patch_blocks') else 0
        self.pixel_layers = len(model.pixel_blocks) if hasattr(model, 'pixel_blocks') else 0

        # Load image
        self.image = self._load_image()

        # Fixed text embedding (for reproducibility)
        torch.manual_seed(seed)
        self.text_embed = torch.randn(1, 32, self.hidden_dim, device=device)
        self.text_mask = torch.ones(1, 32, dtype=torch.bool, device=device)
        self.pooled_text_embed = torch.randn(1, self.hidden_dim, device=device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Loss function
        from src.training.losses import VLoss
        self.v_loss_fn = VLoss()

    def _load_image(self) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(self.image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5  # [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor.unsqueeze(0).to(self.device)

    def _create_batch(self, step: int = 0) -> Dict[str, torch.Tensor]:
        """Create training batch with fixed noise for reproducibility."""
        x_clean = self.image

        # Use step-based seed for reproducibility
        torch.manual_seed(self.seed + step * 7919)
        t = torch.rand(1, device=self.device) * 0.9 + 0.05

        # Fixed noise based on step
        torch.manual_seed(self.seed + step * 7919 + 1)
        noise = torch.randn_like(x_clean)

        # Flow matching: z_t = t * x + (1-t) * noise
        x_t = t.view(1, 1, 1, 1) * x_clean + (1 - t.view(1, 1, 1, 1)) * noise
        v_target = x_clean - noise

        return {
            "x_t": x_t,
            "x_clean": x_clean,
            "t": t,
            "noise": noise,
            "v_target": v_target,
            "text_embed": self.text_embed,
            "text_mask": self.text_mask,
            "pooled_text_embed": self.pooled_text_embed,
        }

    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Pearson correlation between prediction and target."""
        pred_flat = pred.flatten().float()
        target_flat = target.flatten().float()
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr = (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm() + 1e-8
        )
        return corr.item()

    def run(self) -> DiagnosticResults:
        """Run the diagnostic test."""
        results = DiagnosticResults(
            config_name=self.config_name,
            has_bottleneck=self.has_bottleneck,
            patch_layers=self.patch_layers,
            pixel_layers=self.pixel_layers,
            hidden_dim=self.hidden_dim,
            total_params=sum(p.numel() for p in self.model.parameters()),
        )

        bottleneck_status = "WITH bottleneck" if self.has_bottleneck else "WITHOUT bottleneck (direct)"

        print(f"\n{'='*60}")
        print(f"BOTTLENECK DIAGNOSTIC: {self.config_name}")
        print(f"{'='*60}")
        print(f"Mode: {bottleneck_status}")
        print(f"Patch layers: {self.patch_layers}")
        print(f"Pixel layers: {self.pixel_layers}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Total params: {results.total_params/1e6:.2f}M")
        print(f"Image: {self.image_path}")
        print(f"Size: {self.image_size}x{self.image_size}")
        print(f"LR: {self.learning_rate}")
        print(f"Steps: {self.num_steps}")
        print(f"Seed: {self.seed}")
        print(f"{'='*60}\n")

        start_time = time.time()
        self.model.train()

        for step in range(self.num_steps):
            batch = self._create_batch(step=step)

            # Forward
            self.optimizer.zero_grad()
            output = self.model(
                x_t=batch["x_t"],
                t=batch["t"],
                text_embed=batch["text_embed"],
                text_mask=batch["text_mask"],
                pooled_text_embed=batch["pooled_text_embed"],
            )

            if output.shape[-1] == 3:
                output = output.permute(0, 3, 1, 2)

            loss = self.v_loss_fn(output, batch["x_clean"], batch["noise"])
            loss.backward()
            self.optimizer.step()

            # Record metrics at log intervals
            if step % self.log_interval == 0 or step == self.num_steps - 1:
                with torch.no_grad():
                    corr = self._compute_correlation(output, batch["v_target"])

                    step_metrics = DiagnosticStep(
                        step=step,
                        loss=loss.item(),
                        correlation=corr,
                    )
                    results.steps.append(step_metrics)

                    print(f"Step {step:4d}: loss={loss.item():.6f}, corr={corr:.4f}")

        results.elapsed_time = time.time() - start_time
        results.final_loss = results.steps[-1].loss
        results.final_correlation = results.steps[-1].correlation

        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC COMPLETE: {self.config_name}")
        print(f"Elapsed: {results.elapsed_time:.1f}s")
        print(f"Final loss: {results.final_loss:.6f}")
        print(f"Final correlation: {results.final_correlation:.4f}")
        print(f"{'='*60}")

        return results

    def generate_and_save(self, output_dir: str, num_steps: int = 50) -> str:
        """Generate image using Euler sampling and save."""
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        print(f"\nGenerating with {num_steps}-step Euler sampling...")

        with torch.no_grad():
            # Start from pure noise at t=0
            torch.manual_seed(self.seed)
            z = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)

            # Time steps from 0 to 1
            dt = 1.0 / num_steps

            for i in range(num_steps):
                t = torch.tensor([i * dt], device=self.device)

                # Predict velocity
                v_pred = self.model(
                    x_t=z,
                    t=t,
                    text_embed=self.text_embed,
                    text_mask=self.text_mask,
                    pooled_text_embed=self.pooled_text_embed,
                )

                if v_pred.shape[-1] == 3:  # BHWC -> BCHW
                    v_pred = v_pred.permute(0, 3, 1, 2)

                # Euler step: z_{t+dt} = z_t + dt * v_pred
                z = z + dt * v_pred

            # Final result
            x_pred = z.clamp(-1, 1)
            x_pred_np = ((x_pred[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            pred_img = Image.fromarray(x_pred_np)

            # Save image
            pred_path = f"{output_dir}/{self.config_name}_sampled.png"
            pred_img.save(pred_path)

            print(f"Sampled image saved: {pred_path}")

            return pred_path


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(
    results_with: DiagnosticResults,
    results_without: DiagnosticResults,
    output_dir: str,
):
    """Plot comparison of convergence curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1 = axes[0]
    steps_with = [s.step for s in results_with.steps]
    loss_with = [s.loss for s in results_with.steps]
    steps_without = [s.step for s in results_without.steps]
    loss_without = [s.loss for s in results_without.steps]

    ax1.plot(steps_with, loss_with, 'b-', label='With Bottleneck', linewidth=2)
    ax1.plot(steps_without, loss_without, 'r-', label='Without Bottleneck (Direct)', linewidth=2)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Correlation curves
    ax2 = axes[1]
    corr_with = [s.correlation for s in results_with.steps]
    corr_without = [s.correlation for s in results_without.steps]

    ax2.plot(steps_with, corr_with, 'b-', label='With Bottleneck', linewidth=2)
    ax2.plot(steps_without, corr_without, 'r-', label='Without Bottleneck (Direct)', linewidth=2)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Correlation Curve Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Target (0.95)')

    plt.tight_layout()
    plot_path = f"{output_dir}/convergence_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Convergence plot saved: {plot_path}")
    return plot_path


def create_final_comparison_image(
    image_path: str,
    pred_with_path: str,
    pred_without_path: str,
    output_dir: str,
    image_size: int = 256,
):
    """Create side-by-side comparison image: [Original | With Bottleneck | Without Bottleneck]"""
    orig = Image.open(image_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
    pred_with = Image.open(pred_with_path).convert('RGB')
    pred_without = Image.open(pred_without_path).convert('RGB')

    # Create combined image
    combined = Image.new('RGB', (image_size * 3, image_size))
    combined.paste(orig, (0, 0))
    combined.paste(pred_with, (image_size, 0))
    combined.paste(pred_without, (image_size * 2, 0))

    combined_path = f"{output_dir}/final_comparison.png"
    combined.save(combined_path)

    print(f"Final comparison saved: {combined_path}")
    print("  [Original | With Bottleneck | Without Bottleneck]")

    return combined_path


# ============================================================================
# Configuration
# ============================================================================

def create_test_config(hidden_dim: int = 512, patch_layers: int = 4, pixel_layers: int = 2):
    """Create test configuration with moderate size for faster testing."""
    from src.config import PixelHDMConfig
    return PixelHDMConfig(
        hidden_dim=hidden_dim,
        pixel_dim=16,
        patch_size=16,
        patch_layers=patch_layers,
        pixel_layers=pixel_layers,
        num_heads=8,
        num_kv_heads=4,
        head_dim=64,
        mlp_ratio=2.0,
        bottleneck_dim=128,  # Used by bottleneck version
        text_hidden_size=hidden_dim,
        repa_enabled=False,
        freq_loss_enabled=False,
        use_flash_attention=False,
        use_gradient_checkpointing=False,
        zero_init_output=False,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def run_bottleneck_comparison(
    image_path: str = r"G:\Experimental_H_model\單圖測試\01.jpg",
    num_steps: int = 1024,
    image_size: int = 256,
    learning_rate: float = 5e-4,
    output_dir: str = "outputs/bottleneck_diagnostic",
    seed: int = 42,
    hidden_dim: int = 512,
    patch_layers: int = 4,
    pixel_layers: int = 2,
):
    """Run bottleneck comparison diagnostic."""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src.models.pixelhdm import PixelHDM

    all_results = {}

    # ========================================
    # Run WITH bottleneck
    # ========================================
    print("\n" + "="*70)
    print("RUNNING WITH BOTTLENECK")
    print("="*70)

    config = create_test_config(hidden_dim=hidden_dim, patch_layers=patch_layers, pixel_layers=pixel_layers)

    # Reset seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_with = PixelHDM(config=config).to(device)

    diagnostic_with = BottleneckDiagnostic(
        model=model_with,
        device=device,
        image_path=image_path,
        config_name="with_bottleneck",
        has_bottleneck=True,
        image_size=image_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
    )
    results_with = diagnostic_with.run()
    pred_with_path = diagnostic_with.generate_and_save(output_dir)
    all_results["with_bottleneck"] = results_with.to_dict()

    del model_with
    torch.cuda.empty_cache()

    # ========================================
    # Run WITHOUT bottleneck
    # ========================================
    print("\n" + "="*70)
    print("RUNNING WITHOUT BOTTLENECK (DIRECT)")
    print("="*70)

    # Reset seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_without = create_model_with_direct_embedding(config).to(device)

    diagnostic_without = BottleneckDiagnostic(
        model=model_without,
        device=device,
        image_path=image_path,
        config_name="without_bottleneck",
        has_bottleneck=False,
        image_size=image_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
    )
    results_without = diagnostic_without.run()
    pred_without_path = diagnostic_without.generate_and_save(output_dir)
    all_results["without_bottleneck"] = results_without.to_dict()

    # ========================================
    # Generate comparison plots and images
    # ========================================
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    # Plot convergence curves
    plot_comparison(results_with, results_without, output_dir)

    # Create final comparison image
    create_final_comparison_image(
        image_path=image_path,
        pred_with_path=pred_with_path,
        pred_without_path=pred_without_path,
        output_dir=output_dir,
        image_size=image_size,
    )

    # Save original for reference
    orig = Image.open(image_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
    orig.save(f"{output_dir}/original.png")

    # Save results JSON
    output_file = f"{output_dir}/comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<30} {'With Bottleneck':<20} {'Without Bottleneck':<20}")
    print("-"*70)
    print(f"{'Total params':<30} {results_with.total_params/1e6:.2f}M{'':<13} {results_without.total_params/1e6:.2f}M")
    print(f"{'Final loss':<30} {results_with.final_loss:.6f}{'':<13} {results_without.final_loss:.6f}")
    print(f"{'Final correlation':<30} {results_with.final_correlation:.4f}{'':<15} {results_without.final_correlation:.4f}")
    print(f"{'Elapsed time':<30} {results_with.elapsed_time:.1f}s{'':<15} {results_without.elapsed_time:.1f}s")
    print("="*70)

    # Determine winner
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    loss_diff = results_with.final_loss - results_without.final_loss
    corr_diff = results_with.final_correlation - results_without.final_correlation

    if abs(loss_diff) < 0.001 and abs(corr_diff) < 0.01:
        print("Result: SIMILAR - Both methods show comparable convergence")
    elif results_without.final_loss < results_with.final_loss and results_without.final_correlation > results_with.final_correlation:
        print("Result: WITHOUT BOTTLENECK IS BETTER")
        print(f"  Loss improvement: {loss_diff:.6f} ({100*loss_diff/results_with.final_loss:.1f}%)")
        print(f"  Correlation improvement: {corr_diff:.4f}")
    elif results_with.final_loss < results_without.final_loss and results_with.final_correlation > results_without.final_correlation:
        print("Result: WITH BOTTLENECK IS BETTER")
        print(f"  Loss advantage: {-loss_diff:.6f} ({100*(-loss_diff)/results_without.final_loss:.1f}%)")
        print(f"  Correlation advantage: {-corr_diff:.4f}")
    else:
        print("Result: MIXED - Results are inconclusive")
        print(f"  Loss difference: {loss_diff:.6f}")
        print(f"  Correlation difference: {corr_diff:.4f}")

    print("="*70)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patch Embedding Bottleneck Diagnostic")
    parser.add_argument("--image", type=str, default=r"G:\Experimental_H_model\單圖測試\01.jpg")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", type=str, default="outputs/bottleneck_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--patch_layers", type=int, default=4)
    parser.add_argument("--pixel_layers", type=int, default=2)
    args = parser.parse_args()

    run_bottleneck_comparison(
        image_path=args.image,
        num_steps=args.steps,
        image_size=args.size,
        learning_rate=args.lr,
        output_dir=args.output,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        patch_layers=args.patch_layers,
        pixel_layers=args.pixel_layers,
    )
