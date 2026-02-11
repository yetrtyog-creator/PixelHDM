"""
Bottleneck Dimension Sweep Diagnostic

Tests different bottleneck_dim values for PatchEmbedding:
- bottleneck_dim: 16, 32, 64, 128, 256

The bottleneck design: input(768) -> down(bottleneck_dim) -> SiLU -> up(hidden_dim)

Test parameters:
- 1024 training steps
- 5e-4 learning rate
- Single image overfit test
- Output visualization and comparison

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-24
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PIL import Image
import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
    bottleneck_dim: int
    patch_layers: int
    pixel_layers: int
    hidden_dim: int
    total_params: int
    steps: List[DiagnosticStep] = field(default_factory=list)
    elapsed_time: float = 0.0
    final_loss: float = 0.0
    final_correlation: float = 0.0
    sampled_image_path: str = ""

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "bottleneck_dim": self.bottleneck_dim,
            "patch_layers": self.patch_layers,
            "pixel_layers": self.pixel_layers,
            "hidden_dim": self.hidden_dim,
            "total_params": self.total_params,
            "elapsed_time": self.elapsed_time,
            "final_loss": self.final_loss,
            "final_correlation": self.final_correlation,
            "sampled_image_path": self.sampled_image_path,
            "steps": [{"step": s.step, "loss": s.loss, "correlation": s.correlation} for s in self.steps]
        }


# ============================================================================
# Bottleneck Dimension Diagnostic Runner
# ============================================================================

class BottleneckDimDiagnostic:
    """Diagnostic class for bottleneck dimension comparison."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_path: str,
        config_name: str = "unknown",
        bottleneck_dim: int = 256,
        image_size: int = 256,
        learning_rate: float = 5e-4,
        num_steps: int = 1024,
        log_interval: int = 50,
        seed: int = 42,
    ):
        self.model = model
        self.device = device
        self.image_path = image_path
        self.config_name = config_name
        self.bottleneck_dim = bottleneck_dim
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
            bottleneck_dim=self.bottleneck_dim,
            patch_layers=self.patch_layers,
            pixel_layers=self.pixel_layers,
            hidden_dim=self.hidden_dim,
            total_params=sum(p.numel() for p in self.model.parameters()),
        )

        print(f"\n{'='*60}")
        print(f"BOTTLENECK DIM DIAGNOSTIC: {self.config_name}")
        print(f"{'='*60}")
        print(f"Bottleneck dim: {self.bottleneck_dim}")
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
    all_results: Dict[int, DiagnosticResults],
    output_dir: str,
):
    """Plot comparison of convergence curves for all bottleneck dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bottleneck_dims = sorted(all_results.keys())

    # Loss curves
    ax1 = axes[0]
    for idx, bn_dim in enumerate(bottleneck_dims):
        results = all_results[bn_dim]
        steps = [s.step for s in results.steps]
        losses = [s.loss for s in results.steps]
        ax1.plot(steps, losses, color=colors[idx], label=f'bn_dim={bn_dim}', linewidth=2)

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve Comparison by Bottleneck Dim', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Correlation curves
    ax2 = axes[1]
    for idx, bn_dim in enumerate(bottleneck_dims):
        results = all_results[bn_dim]
        steps = [s.step for s in results.steps]
        corrs = [s.correlation for s in results.steps]
        ax2.plot(steps, corrs, color=colors[idx], label=f'bn_dim={bn_dim}', linewidth=2)

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Correlation Curve Comparison by Bottleneck Dim', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Target (0.95)')

    plt.tight_layout()
    plot_path = f"{output_dir}/convergence_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Convergence plot saved: {plot_path}")
    return plot_path


def create_final_comparison_grid(
    image_path: str,
    all_results: Dict[int, DiagnosticResults],
    output_dir: str,
    image_size: int = 256,
):
    """Create side-by-side comparison image: [Original | bn_dim=16 | 32 | 64 | 128 | 256]"""
    orig = Image.open(image_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)

    bottleneck_dims = sorted(all_results.keys())
    num_images = 1 + len(bottleneck_dims)  # Original + all bottleneck variants

    # Create combined image
    combined = Image.new('RGB', (image_size * num_images, image_size))
    combined.paste(orig, (0, 0))

    for idx, bn_dim in enumerate(bottleneck_dims):
        pred_path = all_results[bn_dim].sampled_image_path
        if pred_path and os.path.exists(pred_path):
            pred_img = Image.open(pred_path).convert('RGB')
            combined.paste(pred_img, ((idx + 1) * image_size, 0))

    combined_path = f"{output_dir}/final_comparison_grid.png"
    combined.save(combined_path)

    labels = ["Original"] + [f"bn={d}" for d in bottleneck_dims]
    print(f"Final comparison saved: {combined_path}")
    print(f"  [{' | '.join(labels)}]")

    return combined_path


# ============================================================================
# Configuration
# ============================================================================

def create_test_config(
    hidden_dim: int = 512,
    patch_layers: int = 4,
    pixel_layers: int = 2,
    bottleneck_dim: int = 256,
):
    """Create test configuration with specified bottleneck_dim."""
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
        bottleneck_dim=bottleneck_dim,
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

def run_bottleneck_dim_sweep(
    image_path: str = r"G:\Experimental_H_model\單圖測試\01.jpg",
    bottleneck_dims: List[int] = [16, 32, 64, 128, 256],
    num_steps: int = 1024,
    image_size: int = 256,
    learning_rate: float = 5e-4,
    output_dir: str = "outputs/bottleneck_dim_sweep",
    seed: int = 42,
    hidden_dim: int = 512,
    patch_layers: int = 4,
    pixel_layers: int = 2,
):
    """Run bottleneck dimension sweep diagnostic."""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bottleneck dimensions to test: {bottleneck_dims}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num steps: {num_steps}")
    print()

    from src.models.pixelhdm import PixelHDM

    all_results: Dict[int, DiagnosticResults] = {}

    for bn_dim in bottleneck_dims:
        print("\n" + "="*70)
        print(f"TESTING BOTTLENECK_DIM = {bn_dim}")
        print("="*70)

        config = create_test_config(
            hidden_dim=hidden_dim,
            patch_layers=patch_layers,
            pixel_layers=pixel_layers,
            bottleneck_dim=bn_dim,
        )

        # Reset seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PixelHDM(config=config).to(device)

        diagnostic = BottleneckDimDiagnostic(
            model=model,
            device=device,
            image_path=image_path,
            config_name=f"bn_dim_{bn_dim}",
            bottleneck_dim=bn_dim,
            image_size=image_size,
            learning_rate=learning_rate,
            num_steps=num_steps,
            seed=seed,
        )

        results = diagnostic.run()
        pred_path = diagnostic.generate_and_save(output_dir)
        results.sampled_image_path = pred_path

        all_results[bn_dim] = results

        # Clean up
        del model
        del diagnostic
        torch.cuda.empty_cache()

    # ========================================
    # Generate comparison plots and images
    # ========================================
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    # Plot convergence curves
    plot_comparison(all_results, output_dir)

    # Create final comparison image
    create_final_comparison_grid(
        image_path=image_path,
        all_results=all_results,
        output_dir=output_dir,
        image_size=image_size,
    )

    # Save original for reference
    orig = Image.open(image_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
    orig.save(f"{output_dir}/original.png")

    # Save results JSON
    results_dict = {str(k): v.to_dict() for k, v in all_results.items()}
    output_file = f"{output_dir}/sweep_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Bottleneck Dim':<15} {'Params (M)':<12} {'Final Loss':<15} {'Final Corr':<12} {'Time (s)':<10}")
    print("-"*70)

    for bn_dim in bottleneck_dims:
        r = all_results[bn_dim]
        print(f"{bn_dim:<15} {r.total_params/1e6:<12.2f} {r.final_loss:<15.6f} {r.final_correlation:<12.4f} {r.elapsed_time:<10.1f}")

    print("="*70)

    # Determine best configuration
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Find best by loss
    best_by_loss = min(all_results.items(), key=lambda x: x[1].final_loss)
    # Find best by correlation
    best_by_corr = max(all_results.items(), key=lambda x: x[1].final_correlation)

    print(f"Best by loss: bottleneck_dim={best_by_loss[0]} (loss={best_by_loss[1].final_loss:.6f})")
    print(f"Best by correlation: bottleneck_dim={best_by_corr[0]} (corr={best_by_corr[1].final_correlation:.4f})")

    # Check if all converged well (corr > 0.95)
    converged = [bn for bn, r in all_results.items() if r.final_correlation > 0.95]
    if converged:
        print(f"\nConfigurations with corr > 0.95: {converged}")
        # Among converged, find smallest bottleneck (most compressed)
        smallest_converged = min(converged)
        print(f"Smallest converged bottleneck_dim: {smallest_converged}")
    else:
        print("\nNo configuration achieved corr > 0.95")

    print("="*70)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bottleneck Dimension Sweep Diagnostic")
    parser.add_argument("--image", type=str, default=r"G:\Experimental_H_model\單圖測試\01.jpg")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", type=str, default="outputs/bottleneck_dim_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--patch_layers", type=int, default=4)
    parser.add_argument("--pixel_layers", type=int, default=2)
    parser.add_argument("--bottleneck_dims", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = parser.parse_args()

    run_bottleneck_dim_sweep(
        image_path=args.image,
        bottleneck_dims=args.bottleneck_dims,
        num_steps=args.steps,
        image_size=args.size,
        learning_rate=args.lr,
        output_dir=args.output,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        patch_layers=args.patch_layers,
        pixel_layers=args.pixel_layers,
    )
