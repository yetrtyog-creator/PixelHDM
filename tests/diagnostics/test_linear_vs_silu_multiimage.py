"""
Linear vs SiLU Bottleneck Comparison - Multi-Image Version

Tests the impact of SiLU activation in PatchEmbedding bottleneck with multiple images:
- Nonlinear (current): Linear(768->256) -> SiLU -> Linear(256->D)
- Linear (JiT-style): Linear(768->256) -> Linear(256->D) (no SiLU)

Test parameters:
- 1024 total training steps (cycling through 5 images)
- Batch size = 1
- 5e-4 learning rate
- Multi-image test (5 images)
- Output visualization and comparison for all 5 images

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-24
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json
import os
import copy
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
# Local PatchEmbedding Variants (isolated from src)
# ============================================================================

class PatchEmbeddingWithSiLU(nn.Module):
    """
    Patch Embedding with Nonlinear Bottleneck (Current Design)

    Flow: input(768) -> Linear(768->r) -> SiLU -> Linear(r->D)
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        in_channels: int = 3,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.p2 = patch_size ** 2
        self.bottleneck_dim = bottleneck_dim

        input_dim = in_channels * self.p2
        self.proj_down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.act = nn.SiLU()
        self.proj_up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        B, H, W, C = x.shape
        self._validate_dimensions(H, W)
        x = self._unfold_patches(x, B, H, W)
        x = self._apply_projection(x)
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

    def _apply_projection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_down(x)
        x = self.act(x)  # SiLU activation
        x = self.proj_up(x)
        return x


class PatchEmbeddingLinear(nn.Module):
    """
    Patch Embedding with Linear Bottleneck (JiT-style)

    Flow: input(768) -> Linear(768->r) -> Linear(r->D) (NO activation)

    This is a pure low-rank reparameterization.
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        in_channels: int = 3,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.p2 = patch_size ** 2
        self.bottleneck_dim = bottleneck_dim

        input_dim = in_channels * self.p2
        self.proj_down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        # NO activation function
        self.proj_up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        B, H, W, C = x.shape
        self._validate_dimensions(H, W)
        x = self._unfold_patches(x, B, H, W)
        x = self._apply_projection(x)
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

    def _apply_projection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_down(x)
        # NO SiLU activation - pure linear (low-rank)
        x = self.proj_up(x)
        return x


# ============================================================================
# Diagnostic Results
# ============================================================================

@dataclass
class ImageMetrics:
    """Metrics for a single image."""
    image_name: str
    losses: List[float] = field(default_factory=list)
    correlations: List[float] = field(default_factory=list)
    final_loss: float = 0.0
    final_correlation: float = 0.0


@dataclass
class DiagnosticStep:
    """Metrics for a single training step."""
    step: int
    loss: float
    correlation: float
    image_idx: int


@dataclass
class MultiImageDiagnosticResults:
    """Complete diagnostic results for a multi-image run."""
    config_name: str
    bottleneck_type: str  # "silu" or "linear"
    bottleneck_dim: int
    patch_layers: int
    pixel_layers: int
    hidden_dim: int
    total_params: int
    num_images: int
    steps: List[DiagnosticStep] = field(default_factory=list)
    image_metrics: Dict[str, ImageMetrics] = field(default_factory=dict)
    elapsed_time: float = 0.0
    final_avg_loss: float = 0.0
    final_avg_correlation: float = 0.0
    sampled_image_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "bottleneck_type": self.bottleneck_type,
            "bottleneck_dim": self.bottleneck_dim,
            "patch_layers": self.patch_layers,
            "pixel_layers": self.pixel_layers,
            "hidden_dim": self.hidden_dim,
            "total_params": self.total_params,
            "num_images": self.num_images,
            "elapsed_time": self.elapsed_time,
            "final_avg_loss": self.final_avg_loss,
            "final_avg_correlation": self.final_avg_correlation,
            "sampled_image_paths": self.sampled_image_paths,
            "image_metrics": {
                name: {
                    "final_loss": m.final_loss,
                    "final_correlation": m.final_correlation,
                }
                for name, m in self.image_metrics.items()
            },
            "steps": [
                {"step": s.step, "loss": s.loss, "correlation": s.correlation, "image_idx": s.image_idx}
                for s in self.steps
            ]
        }


# ============================================================================
# Modified PixelHDM for Testing (with swappable PatchEmbedding)
# ============================================================================

def create_model_with_patch_embedding(
    hidden_dim: int = 512,
    patch_layers: int = 4,
    pixel_layers: int = 2,
    bottleneck_dim: int = 256,
    bottleneck_type: str = "silu",  # "silu" or "linear"
    device: torch.device = None,
) -> nn.Module:
    """Create PixelHDM model and replace PatchEmbedding with specified type."""
    from src.config import PixelHDMConfig
    from src.models.pixelhdm import PixelHDM

    config = PixelHDMConfig(
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

    model = PixelHDM(config=config)

    # Replace the patch_embed module with our test variant
    if bottleneck_type == "silu":
        new_patch_embed = PatchEmbeddingWithSiLU(
            patch_size=16,
            hidden_dim=hidden_dim,
            in_channels=3,
            bottleneck_dim=bottleneck_dim,
        )
    elif bottleneck_type == "linear":
        new_patch_embed = PatchEmbeddingLinear(
            patch_size=16,
            hidden_dim=hidden_dim,
            in_channels=3,
            bottleneck_dim=bottleneck_dim,
        )
    else:
        raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}")

    # Replace the module
    model.patch_embed = new_patch_embed

    if device is not None:
        model = model.to(device)

    return model


# ============================================================================
# Multi-Image Diagnostic Runner
# ============================================================================

class MultiImageDiagnostic:
    """Diagnostic class for Linear vs SiLU bottleneck comparison with multiple images."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_paths: List[str],
        config_name: str = "unknown",
        bottleneck_type: str = "silu",
        bottleneck_dim: int = 256,
        image_size: int = 256,
        learning_rate: float = 5e-4,
        num_steps: int = 1024,
        log_interval: int = 50,
        seed: int = 42,
    ):
        self.model = model
        self.device = device
        self.image_paths = image_paths
        self.num_images = len(image_paths)
        self.config_name = config_name
        self.bottleneck_type = bottleneck_type
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

        # Load all images
        self.images = [self._load_image(p) for p in image_paths]
        self.image_names = [Path(p).stem for p in image_paths]

        # Fixed text embeddings (for reproducibility) - one per image
        torch.manual_seed(seed)
        self.text_embeds = [
            torch.randn(1, 32, self.hidden_dim, device=device)
            for _ in range(self.num_images)
        ]
        self.text_masks = [
            torch.ones(1, 32, dtype=torch.bool, device=device)
            for _ in range(self.num_images)
        ]
        self.pooled_text_embeds = [
            torch.randn(1, self.hidden_dim, device=device)
            for _ in range(self.num_images)
        ]

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Loss function
        from src.training.losses import VLoss
        self.v_loss_fn = VLoss()

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5  # [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor.unsqueeze(0).to(self.device)

    def _create_batch(self, image_idx: int, step: int) -> Dict[str, torch.Tensor]:
        """Create training batch with fixed noise for reproducibility."""
        x_clean = self.images[image_idx]

        # Use step-based seed for reproducibility
        torch.manual_seed(self.seed + step * 7919 + image_idx * 1000)
        t = torch.rand(1, device=self.device) * 0.9 + 0.05

        # Fixed noise based on step
        torch.manual_seed(self.seed + step * 7919 + image_idx * 1000 + 1)
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
            "text_embed": self.text_embeds[image_idx],
            "text_mask": self.text_masks[image_idx],
            "pooled_text_embed": self.pooled_text_embeds[image_idx],
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

    def run(self) -> MultiImageDiagnosticResults:
        """Run the diagnostic test."""
        results = MultiImageDiagnosticResults(
            config_name=self.config_name,
            bottleneck_type=self.bottleneck_type,
            bottleneck_dim=self.bottleneck_dim,
            patch_layers=self.patch_layers,
            pixel_layers=self.pixel_layers,
            hidden_dim=self.hidden_dim,
            total_params=sum(p.numel() for p in self.model.parameters()),
            num_images=self.num_images,
        )

        # Initialize image metrics
        for name in self.image_names:
            results.image_metrics[name] = ImageMetrics(image_name=name)

        print(f"\n{'='*60}")
        print(f"MULTI-IMAGE LINEAR vs SiLU DIAGNOSTIC: {self.config_name}")
        print(f"{'='*60}")
        print(f"Bottleneck type: {self.bottleneck_type.upper()}")
        print(f"Bottleneck dim: {self.bottleneck_dim}")
        print(f"Patch layers: {self.patch_layers}")
        print(f"Pixel layers: {self.pixel_layers}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Total params: {results.total_params/1e6:.2f}M")
        print(f"Number of images: {self.num_images}")
        print(f"Images: {self.image_names}")
        print(f"Size: {self.image_size}x{self.image_size}")
        print(f"LR: {self.learning_rate}")
        print(f"Total steps: {self.num_steps}")
        print(f"Seed: {self.seed}")
        print(f"{'='*60}\n")

        start_time = time.time()
        self.model.train()

        # Track per-image metrics for final evaluation
        recent_losses = {name: [] for name in self.image_names}
        recent_corrs = {name: [] for name in self.image_names}

        for step in range(self.num_steps):
            # Cycle through images
            image_idx = step % self.num_images
            image_name = self.image_names[image_idx]

            batch = self._create_batch(image_idx, step)

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

            # Compute correlation
            with torch.no_grad():
                corr = self._compute_correlation(output, batch["v_target"])

            # Track recent metrics for final evaluation
            recent_losses[image_name].append(loss.item())
            recent_corrs[image_name].append(corr)
            # Keep only last 50 for final average
            if len(recent_losses[image_name]) > 50:
                recent_losses[image_name].pop(0)
                recent_corrs[image_name].pop(0)

            # Record metrics at log intervals
            if step % self.log_interval == 0 or step == self.num_steps - 1:
                step_metrics = DiagnosticStep(
                    step=step,
                    loss=loss.item(),
                    correlation=corr,
                    image_idx=image_idx,
                )
                results.steps.append(step_metrics)

                # Store in image metrics
                results.image_metrics[image_name].losses.append(loss.item())
                results.image_metrics[image_name].correlations.append(corr)

                print(f"Step {step:4d} [{image_name:>5}]: loss={loss.item():.6f}, corr={corr:.4f}")

        results.elapsed_time = time.time() - start_time

        # Compute final metrics per image (average of last 50 steps for that image)
        for name in self.image_names:
            if recent_losses[name]:
                results.image_metrics[name].final_loss = np.mean(recent_losses[name])
                results.image_metrics[name].final_correlation = np.mean(recent_corrs[name])

        # Compute overall average
        results.final_avg_loss = np.mean([m.final_loss for m in results.image_metrics.values()])
        results.final_avg_correlation = np.mean([m.final_correlation for m in results.image_metrics.values()])

        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC COMPLETE: {self.config_name}")
        print(f"Elapsed: {results.elapsed_time:.1f}s")
        print(f"\nPer-image final metrics:")
        for name, m in results.image_metrics.items():
            print(f"  {name}: loss={m.final_loss:.6f}, corr={m.final_correlation:.4f}")
        print(f"\nOverall average:")
        print(f"  Avg loss: {results.final_avg_loss:.6f}")
        print(f"  Avg correlation: {results.final_avg_correlation:.4f}")
        print(f"{'='*60}")

        return results

    def generate_and_save_all(self, output_dir: str, num_steps: int = 50) -> Dict[str, str]:
        """Generate images using Euler sampling and save all."""
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        print(f"\nGenerating {self.num_images} images with {num_steps}-step Euler sampling...")

        paths = {}

        for image_idx, (image_name, text_embed, text_mask, pooled) in enumerate(
            zip(self.image_names, self.text_embeds, self.text_masks, self.pooled_text_embeds)
        ):
            with torch.no_grad():
                # Start from pure noise at t=0 (use same seed pattern as training)
                torch.manual_seed(self.seed + image_idx * 12345)
                z = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)

                # Time steps from 0 to 1
                dt = 1.0 / num_steps

                for i in range(num_steps):
                    t = torch.tensor([i * dt], device=self.device)

                    # Predict velocity
                    v_pred = self.model(
                        x_t=z,
                        t=t,
                        text_embed=text_embed,
                        text_mask=text_mask,
                        pooled_text_embed=pooled,
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
                pred_path = f"{output_dir}/{self.config_name}_{image_name}_sampled.png"
                pred_img.save(pred_path)

                paths[image_name] = pred_path
                print(f"  {image_name}: {pred_path}")

        return paths


# ============================================================================
# Visualization
# ============================================================================

def plot_multiimage_comparison(
    all_results: Dict[str, MultiImageDiagnosticResults],
    output_dir: str,
):
    """Plot comparison of convergence curves for Linear vs SiLU bottleneck (multi-image)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'silu': '#1f77b4', 'linear': '#ff7f0e'}
    labels = {'silu': 'SiLU (Nonlinear)', 'linear': 'Linear (JiT-style)'}

    # Loss curves (all steps)
    ax1 = axes[0]
    for bn_type, results in all_results.items():
        steps = [s.step for s in results.steps]
        losses = [s.loss for s in results.steps]
        ax1.plot(steps, losses, color=colors[bn_type], label=labels[bn_type], linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve Comparison (Multi-Image)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Correlation curves
    ax2 = axes[1]
    for bn_type, results in all_results.items():
        steps = [s.step for s in results.steps]
        corrs = [s.correlation for s in results.steps]
        ax2.plot(steps, corrs, color=colors[bn_type], label=labels[bn_type], linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Correlation Curve Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = f"{output_dir}/convergence_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Convergence plot saved: {plot_path}")
    return plot_path


def plot_per_image_comparison(
    all_results: Dict[str, MultiImageDiagnosticResults],
    image_names: List[str],
    output_dir: str,
):
    """Plot per-image final metrics comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(image_names))
    width = 0.35

    colors = {'silu': '#1f77b4', 'linear': '#ff7f0e'}
    labels = {'silu': 'SiLU', 'linear': 'Linear'}

    # Final loss per image
    ax1 = axes[0]
    for i, (bn_type, results) in enumerate(all_results.items()):
        losses = [results.image_metrics[name].final_loss for name in image_names]
        offset = (i - 0.5) * width
        ax1.bar(x + offset, losses, width, label=labels[bn_type], color=colors[bn_type])

    ax1.set_xlabel('Image', fontsize=12)
    ax1.set_ylabel('Final Loss', fontsize=12)
    ax1.set_title('Final Loss per Image', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(image_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Final correlation per image
    ax2 = axes[1]
    for i, (bn_type, results) in enumerate(all_results.items()):
        corrs = [results.image_metrics[name].final_correlation for name in image_names]
        offset = (i - 0.5) * width
        ax2.bar(x + offset, corrs, width, label=labels[bn_type], color=colors[bn_type])

    ax2.set_xlabel('Image', fontsize=12)
    ax2.set_ylabel('Final Correlation', fontsize=12)
    ax2.set_title('Final Correlation per Image', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(image_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = f"{output_dir}/per_image_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-image comparison plot saved: {plot_path}")
    return plot_path


def create_multiimage_comparison_grid(
    image_paths: List[str],
    all_results: Dict[str, MultiImageDiagnosticResults],
    output_dir: str,
    image_size: int = 256,
):
    """Create comparison grid: rows = images, cols = [Original | SiLU | Linear]"""
    num_images = len(image_paths)
    image_names = [Path(p).stem for p in image_paths]

    # Create combined image: 3 columns (Original, SiLU, Linear), N rows
    combined = Image.new('RGB', (image_size * 3, image_size * num_images))

    for row, (img_path, img_name) in enumerate(zip(image_paths, image_names)):
        # Original
        orig = Image.open(img_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
        combined.paste(orig, (0, row * image_size))

        # SiLU result
        if 'silu' in all_results:
            silu_path = all_results['silu'].sampled_image_paths.get(img_name)
            if silu_path and os.path.exists(silu_path):
                silu_img = Image.open(silu_path).convert('RGB')
                combined.paste(silu_img, (image_size, row * image_size))

        # Linear result
        if 'linear' in all_results:
            linear_path = all_results['linear'].sampled_image_paths.get(img_name)
            if linear_path and os.path.exists(linear_path):
                linear_img = Image.open(linear_path).convert('RGB')
                combined.paste(linear_img, (2 * image_size, row * image_size))

    combined_path = f"{output_dir}/multiimage_comparison_grid.png"
    combined.save(combined_path)

    print(f"Multi-image comparison grid saved: {combined_path}")
    print(f"  Columns: [Original | SiLU | Linear]")
    print(f"  Rows: {image_names}")

    return combined_path


# ============================================================================
# Main Entry Point
# ============================================================================

def run_multiimage_comparison(
    image_dir: str = r"G:\Experimental_H_model\簡單多圖測試",
    image_indices: List[int] = [1, 5, 10, 15, 20],  # Image numbers to use
    bottleneck_dim: int = 256,
    num_steps: int = 1024,
    image_size: int = 256,
    learning_rate: float = 5e-4,
    output_dir: str = "outputs/linear_vs_silu_multiimage",
    seed: int = 42,
    hidden_dim: int = 512,
    patch_layers: int = 4,
    pixel_layers: int = 2,
):
    """Run Linear vs SiLU bottleneck comparison diagnostic with multiple images."""
    # Add timestamp to output dir
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Build image paths
    image_paths = []
    for idx in image_indices:
        # Try .jpg first, then .jpeg
        jpg_path = os.path.join(image_dir, f"{idx:02d}.jpg")
        jpeg_path = os.path.join(image_dir, f"{idx:02d}.jpeg")
        if os.path.exists(jpg_path):
            image_paths.append(jpg_path)
        elif os.path.exists(jpeg_path):
            image_paths.append(jpeg_path)
        else:
            raise FileNotFoundError(f"Image not found: {jpg_path} or {jpeg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bottleneck dimension: {bottleneck_dim}")
    print(f"Comparing: SiLU (Nonlinear) vs Linear (JiT-style)")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {num_steps}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Images: {[Path(p).stem for p in image_paths]}")
    print()

    bottleneck_types = ["silu", "linear"]
    all_results: Dict[str, MultiImageDiagnosticResults] = {}

    for bn_type in bottleneck_types:
        print("\n" + "="*70)
        print(f"TESTING BOTTLENECK TYPE: {bn_type.upper()}")
        print("="*70)

        # Reset seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model_with_patch_embedding(
            hidden_dim=hidden_dim,
            patch_layers=patch_layers,
            pixel_layers=pixel_layers,
            bottleneck_dim=bottleneck_dim,
            bottleneck_type=bn_type,
            device=device,
        )

        diagnostic = MultiImageDiagnostic(
            model=model,
            device=device,
            image_paths=image_paths,
            config_name=f"bottleneck_{bn_type}",
            bottleneck_type=bn_type,
            bottleneck_dim=bottleneck_dim,
            image_size=image_size,
            learning_rate=learning_rate,
            num_steps=num_steps,
            seed=seed,
        )

        results = diagnostic.run()
        pred_paths = diagnostic.generate_and_save_all(output_dir)
        results.sampled_image_paths = pred_paths

        all_results[bn_type] = results

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

    image_names = [Path(p).stem for p in image_paths]

    # Plot convergence curves
    plot_multiimage_comparison(all_results, output_dir)

    # Plot per-image metrics
    plot_per_image_comparison(all_results, image_names, output_dir)

    # Create final comparison grid
    create_multiimage_comparison_grid(
        image_paths=image_paths,
        all_results=all_results,
        output_dir=output_dir,
        image_size=image_size,
    )

    # Save originals for reference
    for img_path in image_paths:
        img_name = Path(img_path).stem
        orig = Image.open(img_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
        orig.save(f"{output_dir}/original_{img_name}.png")

    # Save results JSON
    results_dict = {k: v.to_dict() for k, v in all_results.items()}
    output_file = f"{output_dir}/comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Bottleneck Type':<20} {'Params (M)':<12} {'Avg Loss':<15} {'Avg Corr':<12} {'Time (s)':<10}")
    print("-"*70)

    for bn_type in bottleneck_types:
        r = all_results[bn_type]
        type_label = "SiLU (Nonlinear)" if bn_type == "silu" else "Linear (JiT)"
        print(f"{type_label:<20} {r.total_params/1e6:<12.2f} {r.final_avg_loss:<15.6f} {r.final_avg_correlation:<12.4f} {r.elapsed_time:<10.1f}")

    print("="*70)

    # Per-image summary
    print("\n" + "="*70)
    print("PER-IMAGE FINAL METRICS")
    print("="*70)
    print(f"{'Image':<10} {'SiLU Loss':<15} {'Linear Loss':<15} {'SiLU Corr':<12} {'Linear Corr':<12} {'Winner':<10}")
    print("-"*70)

    silu_wins = 0
    linear_wins = 0

    for name in image_names:
        silu_loss = all_results['silu'].image_metrics[name].final_loss
        linear_loss = all_results['linear'].image_metrics[name].final_loss
        silu_corr = all_results['silu'].image_metrics[name].final_correlation
        linear_corr = all_results['linear'].image_metrics[name].final_correlation

        # Winner by correlation
        if silu_corr > linear_corr:
            winner = "SiLU"
            silu_wins += 1
        else:
            winner = "Linear"
            linear_wins += 1

        print(f"{name:<10} {silu_loss:<15.6f} {linear_loss:<15.6f} {silu_corr:<12.4f} {linear_corr:<12.4f} {winner:<10}")

    print("-"*70)
    print(f"SiLU wins: {silu_wins}/{len(image_names)}, Linear wins: {linear_wins}/{len(image_names)}")
    print("="*70)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    silu_results = all_results['silu']
    linear_results = all_results['linear']

    loss_diff = silu_results.final_avg_loss - linear_results.final_avg_loss
    corr_diff = silu_results.final_avg_correlation - linear_results.final_avg_correlation

    print(f"Average loss difference (SiLU - Linear): {loss_diff:+.6f}")
    print(f"Average correlation difference (SiLU - Linear): {corr_diff:+.4f}")

    if loss_diff > 0:
        print("\n=> Linear bottleneck achieved LOWER average loss (better)")
    else:
        print("\n=> SiLU bottleneck achieved LOWER average loss (better)")

    if corr_diff > 0:
        print("=> SiLU bottleneck achieved HIGHER average correlation (better)")
    else:
        print("=> Linear bottleneck achieved HIGHER average correlation (better)")

    # Overall winner
    print("\n" + "-"*70)
    print("OVERALL CONCLUSION (Multi-Image Test)")
    print("-"*70)
    if silu_wins > linear_wins:
        print(f"SiLU bottleneck won on {silu_wins}/{len(image_names)} images")
        print("SiLU is preferred for this multi-image training scenario.")
    elif linear_wins > silu_wins:
        print(f"Linear bottleneck won on {linear_wins}/{len(image_names)} images")
        print("Linear bottleneck is preferred for this multi-image training scenario.")
    else:
        print("Tie - both bottleneck types performed similarly.")
        if corr_diff > 0:
            print("SiLU has slightly better average correlation.")
        else:
            print("Linear has slightly better average correlation.")

    print("="*70)
    print(f"\nOutput directory: {output_dir}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linear vs SiLU Bottleneck Multi-Image Comparison")
    parser.add_argument("--image_dir", type=str, default=r"G:\Experimental_H_model\簡單多圖測試")
    parser.add_argument("--image_indices", type=int, nargs="+", default=[1, 5, 10, 15, 20])
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", type=str, default="outputs/linear_vs_silu_multiimage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--patch_layers", type=int, default=4)
    parser.add_argument("--pixel_layers", type=int, default=2)
    parser.add_argument("--bottleneck_dim", type=int, default=256)
    args = parser.parse_args()

    run_multiimage_comparison(
        image_dir=args.image_dir,
        image_indices=args.image_indices,
        bottleneck_dim=args.bottleneck_dim,
        num_steps=args.steps,
        image_size=args.size,
        learning_rate=args.lr,
        output_dir=args.output,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        patch_layers=args.patch_layers,
        pixel_layers=args.pixel_layers,
    )
