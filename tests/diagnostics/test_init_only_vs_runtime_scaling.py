"""
Init-Only Scaling vs Runtime Depth Scaling Comparison Test

This script compares two residual scaling strategies:
A) Runtime Depth Scaling (current): residual_scale = 1/sqrt(2*L) at every step
B) Init-Only Scaling (proposed): runtime 1/sqrt(2), init residual projections *= 1/sqrt(2*L)
C) Init-Only Aggressive: runtime 1.0, init *= 1/sqrt(2*L)

Test configurations:
- Shallow: 2 patch + 1 pixel layers (baseline)
- Deep: 8 patch + 2 pixel layers (stress test)

Metrics:
- Loss convergence speed (steps to reach threshold)
- NaN/Inf occurrence
- Update ratio per layer
- Final loss value

Reference: G:\Experimental_H_model\參考chatgpt\init_only_residual_scaling_thought_report_20260123.md

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-23
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json
import copy
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Literal
from collections import defaultdict
from PIL import Image
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Scaling Strategy Types
# ============================================================================

ScalingStrategy = Literal["runtime_depth", "init_only", "init_only_aggressive"]


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LayerMetrics:
    """Metrics for a single layer at a single step."""
    update_ratio: float = 0.0  # std(x_out - x_in) / std(x_in)
    grad_norm: float = 0.0
    weight_norm: float = 0.0


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    loss: float
    correlation: float
    has_nan: bool = False
    has_inf: bool = False
    patch_metrics: List[LayerMetrics] = field(default_factory=list)
    pixel_metrics: List[LayerMetrics] = field(default_factory=list)


@dataclass
class ExperimentResults:
    """Results for a single experiment."""
    strategy: str
    config_name: str
    patch_layers: int
    pixel_layers: int
    hidden_dim: int
    total_params: int
    steps: List[StepMetrics] = field(default_factory=list)
    elapsed_time: float = 0.0
    converged: bool = False
    convergence_step: Optional[int] = None
    crashed: bool = False
    crash_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "config_name": self.config_name,
            "patch_layers": self.patch_layers,
            "pixel_layers": self.pixel_layers,
            "hidden_dim": self.hidden_dim,
            "total_params": self.total_params,
            "elapsed_time": self.elapsed_time,
            "converged": self.converged,
            "convergence_step": self.convergence_step,
            "crashed": self.crashed,
            "crash_reason": self.crash_reason,
            "final_loss": self.steps[-1].loss if self.steps else float('nan'),
            "final_correlation": self.steps[-1].correlation if self.steps else 0.0,
            "steps": [
                {
                    "step": s.step,
                    "loss": s.loss,
                    "correlation": s.correlation,
                    "has_nan": s.has_nan,
                    "has_inf": s.has_inf,
                }
                for s in self.steps
            ],
        }


# ============================================================================
# Mock Blocks with Configurable Scaling
# ============================================================================

class MockPatchBlock(nn.Module):
    """
    Simplified Patch Transformer Block for testing scaling strategies.

    Implements: x = x + (alpha * residual_scale) * h

    Supports:
    - runtime_depth: residual_scale = 1/sqrt(2*L) (current)
    - init_only: residual_scale = 1/sqrt(2), init out_proj/down_proj *= 1/sqrt(2*L)
    - init_only_aggressive: residual_scale = 1.0, init *= 1/sqrt(2*L)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_layers: int = 1,
        strategy: ScalingStrategy = "runtime_depth",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.strategy = strategy

        # Compute residual scale based on strategy
        k = 2  # 2 branches per block (attn + mlp)
        if strategy == "runtime_depth":
            self.residual_scale = 1.0 / math.sqrt(k * num_layers)
        elif strategy == "init_only":
            self.residual_scale = 1.0 / math.sqrt(k)  # Only branch count
        else:  # init_only_aggressive
            self.residual_scale = 1.0

        # Attention-like branch
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # MLP branch
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

        # AdaLN-like gating (simplified: just learnable scalars per branch)
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

        # Apply init-only scaling to residual projections
        if strategy in ("init_only", "init_only_aggressive"):
            init_scale = 1.0 / math.sqrt(k * num_layers)
            with torch.no_grad():
                self.out_proj.weight.mul_(init_scale)
                self.down_proj.weight.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Attention branch (simplified)
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(D), dim=-1)
        h = attn @ v
        h = self.out_proj(h)
        x = x + (self.alpha1 * self.residual_scale) * h

        # MLP branch
        h = self.norm2(x)
        h = F.silu(self.up_proj(h))
        h = self.down_proj(h)
        x = x + (self.alpha2 * self.residual_scale) * h

        return x


class MockPixelBlock(nn.Module):
    """
    Simplified Pixel Transformer Block for testing scaling strategies.

    Includes compress-attend-expand pattern.
    """

    def __init__(
        self,
        hidden_dim: int,
        pixel_dim: int,
        patch_size: int = 16,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_layers: int = 1,
        strategy: ScalingStrategy = "runtime_depth",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.p2 = patch_size ** 2
        self.p2_d_pix = self.p2 * pixel_dim
        self.num_layers = num_layers
        self.strategy = strategy

        # Residual scale
        k = 2
        if strategy == "runtime_depth":
            self.residual_scale = 1.0 / math.sqrt(k * num_layers)
        elif strategy == "init_only":
            self.residual_scale = 1.0 / math.sqrt(k)
        else:
            self.residual_scale = 1.0

        # Compaction branch (compress -> attn -> expand)
        self.compress = nn.Linear(self.p2_d_pix, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.expand = nn.Linear(hidden_dim, self.p2_d_pix, bias=False)

        # MLP branch (on pixel_dim)
        self.norm2 = nn.LayerNorm(pixel_dim)
        mlp_dim = int(pixel_dim * mlp_ratio)
        self.up_proj = nn.Linear(pixel_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, pixel_dim, bias=False)

        # Alpha gating
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

        # Init-only scaling for residual projections
        if strategy in ("init_only", "init_only_aggressive"):
            init_scale = 1.0 / math.sqrt(k * num_layers)
            with torch.no_grad():
                # Compaction expand is the residual projection
                self.expand.weight.mul_(init_scale)
                # MLP down_proj
                self.down_proj.weight.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, p^2, D_pix)
        Returns:
            (B, L, p^2, D_pix)
        """
        B, L, P2, D = x.shape

        # Compaction branch
        x_flat = x.reshape(B, L, self.p2_d_pix)
        h = self.compress(x_flat)  # (B, L, hidden_dim)
        h = self.norm1(h)

        # Simplified attention
        qkv = self.qkv(h).reshape(B, L, 3, self.hidden_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.hidden_dim), dim=-1)
        h = attn @ v
        h = self.out_proj(h)
        h = self.expand(h)  # (B, L, p2_d_pix)
        h = h.reshape(B, L, P2, D)
        x = x + (self.alpha1 * self.residual_scale) * h

        # MLP branch (on pixel dim)
        h = self.norm2(x)
        h = F.silu(self.up_proj(h))
        h = self.down_proj(h)
        x = x + (self.alpha2 * self.residual_scale) * h

        return x


class MockPixelHDM(nn.Module):
    """
    Simplified PixelHDM for testing scaling strategies.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        pixel_dim: int = 8,
        patch_size: int = 16,
        patch_layers: int = 2,
        pixel_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        strategy: ScalingStrategy = "runtime_depth",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.patch_layers = patch_layers
        self.pixel_layers = pixel_layers
        self.strategy = strategy

        # Input projection (image -> patches)
        self.patch_embed = nn.Conv2d(
            3, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Patch blocks
        self.patch_blocks = nn.ModuleList([
            MockPatchBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_layers=patch_layers,
                strategy=strategy,
            )
            for _ in range(patch_layers)
        ])

        # Pixel unflatten: (B, L, D) -> (B, L, p^2, D_pix)
        self.pixel_unfold = nn.Linear(hidden_dim, patch_size**2 * pixel_dim, bias=False)

        # Pixel blocks
        self.pixel_blocks = nn.ModuleList([
            MockPixelBlock(
                hidden_dim=hidden_dim,
                pixel_dim=pixel_dim,
                patch_size=patch_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_layers=pixel_layers,
                strategy=strategy,
            )
            for _ in range(pixel_layers)
        ])

        # Output projection: (B, L, p^2, D_pix) -> (B, H, W, 3)
        self.output_proj = nn.Linear(pixel_dim, 3, bias=False)

        # Time embedding (simplified)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, 3, H, W) or (B, H, W, 3)
            t: (B,)
        Returns:
            v_pred: (B, 3, H, W)
        """
        # Handle BHWC input
        if x_t.shape[-1] == 3:
            x_t = x_t.permute(0, 3, 1, 2)

        B, C, H, W = x_t.shape

        # Patch embedding
        x = self.patch_embed(x_t)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        L = x.shape[1]

        # Time embedding (add to all tokens)
        t_embed = self.time_embed(t.view(-1, 1))  # (B, D)
        x = x + t_embed.unsqueeze(1)

        # Patch blocks
        for block in self.patch_blocks:
            x = block(x)

        # Pixel unfold
        x = self.pixel_unfold(x)  # (B, L, p^2 * D_pix)
        x = x.reshape(B, L, self.patch_size**2, self.pixel_dim)  # (B, L, p^2, D_pix)

        # Pixel blocks
        for block in self.pixel_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)  # (B, L, p^2, 3)

        # Reshape to image
        pH = H // self.patch_size
        pW = W // self.patch_size
        x = x.reshape(B, pH, pW, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
        x = x.permute(0, 3, 1, 2)  # (B, 3, H, W)

        return x


# ============================================================================
# Experiment Runner
# ============================================================================

class ScalingExperiment:
    """Runs scaling strategy comparison experiments."""

    def __init__(
        self,
        image_path: str,
        device: torch.device,
        image_size: int = 256,
        learning_rate: float = 1e-4,
        num_steps: int = 300,
        log_interval: int = 10,
        convergence_threshold: float = 0.01,
        seed: int = 42,
    ):
        self.image_path = image_path
        self.device = device
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.convergence_threshold = convergence_threshold
        self.seed = seed

        # Load image
        self.image = self._load_image()

    def _load_image(self) -> torch.Tensor:
        img = Image.open(self.image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5  # [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor.unsqueeze(0).to(self.device)

    def _create_batch(self, step: int) -> Dict[str, torch.Tensor]:
        x_clean = self.image

        torch.manual_seed(self.seed + step * 7919)
        t = torch.rand(1, device=self.device) * 0.9 + 0.05

        torch.manual_seed(self.seed + step * 7919 + 1)
        noise = torch.randn_like(x_clean)

        x_t = t.view(1, 1, 1, 1) * x_clean + (1 - t.view(1, 1, 1, 1)) * noise
        v_target = x_clean - noise

        return {
            "x_t": x_t,
            "t": t,
            "v_target": v_target,
        }

    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_flat = pred.flatten().float()
        target_flat = target.flatten().float()
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr = (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm() + 1e-8
        )
        return corr.item()

    def run_single(
        self,
        model: nn.Module,
        strategy: str,
        config_name: str,
    ) -> ExperimentResults:
        """Run a single experiment."""
        patch_layers = len(model.patch_blocks)
        pixel_layers = len(model.pixel_blocks)

        results = ExperimentResults(
            strategy=strategy,
            config_name=config_name,
            patch_layers=patch_layers,
            pixel_layers=pixel_layers,
            hidden_dim=model.hidden_dim,
            total_params=sum(p.numel() for p in model.parameters()),
        )

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy} | Config: {config_name}")
        print(f"{'='*60}")
        print(f"Patch layers: {patch_layers}, Pixel layers: {pixel_layers}")
        print(f"Total params: {results.total_params/1e6:.2f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        try:
            for step in range(self.num_steps):
                batch = self._create_batch(step)

                optimizer.zero_grad()
                output = model(batch["x_t"], batch["t"])

                # V-loss
                loss = F.mse_loss(output, batch["v_target"])

                # Check for NaN/Inf
                has_nan = torch.isnan(loss).item()
                has_inf = torch.isinf(loss).item()

                if has_nan or has_inf:
                    print(f"Step {step}: {'NaN' if has_nan else 'Inf'} detected!")
                    results.crashed = True
                    results.crash_reason = f"{'NaN' if has_nan else 'Inf'} at step {step}"
                    break

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Log metrics
                if step % self.log_interval == 0:
                    with torch.no_grad():
                        corr = self._compute_correlation(output, batch["v_target"])

                        step_metrics = StepMetrics(
                            step=step,
                            loss=loss.item(),
                            correlation=corr,
                            has_nan=has_nan,
                            has_inf=has_inf,
                        )
                        results.steps.append(step_metrics)

                        print(f"Step {step:4d}: loss={loss.item():.6f}, corr={corr:.4f}")

                # Check convergence
                if loss.item() < self.convergence_threshold:
                    results.converged = True
                    results.convergence_step = step
                    print(f"\nCONVERGED at step {step}!")
                    break

        except Exception as e:
            results.crashed = True
            results.crash_reason = str(e)
            print(f"CRASHED: {e}")
            traceback.print_exc()

        results.elapsed_time = time.time() - start_time

        print(f"\nElapsed: {results.elapsed_time:.1f}s")
        if results.steps:
            print(f"Final loss: {results.steps[-1].loss:.6f}")
        print(f"Converged: {'YES' if results.converged else 'NO'}")
        print(f"Crashed: {'YES - ' + results.crash_reason if results.crashed else 'NO'}")

        return results


# ============================================================================
# Main Comparison
# ============================================================================

def create_model(
    patch_layers: int,
    pixel_layers: int,
    strategy: ScalingStrategy,
    device: torch.device,
) -> MockPixelHDM:
    """Create a MockPixelHDM with specified configuration."""
    return MockPixelHDM(
        hidden_dim=256,
        pixel_dim=8,
        patch_size=16,
        patch_layers=patch_layers,
        pixel_layers=pixel_layers,
        num_heads=4,
        mlp_ratio=2.0,
        strategy=strategy,
    ).to(device)


def run_comparison(
    image_path: str = r"G:\Experimental_H_model\單圖測試\01.jpg",
    num_steps: int = 300,
    image_size: int = 256,
    learning_rate: float = 1e-4,
    output_dir: str = "outputs/init_only_vs_runtime",
    seed: int = 42,
):
    """Run full comparison of scaling strategies."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    experiment = ScalingExperiment(
        image_path=image_path,
        device=device,
        image_size=image_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
    )

    # Test configurations
    configs = [
        ("shallow", 2, 1),   # 2 patch + 1 pixel (baseline)
        ("deep", 8, 2),      # 8 patch + 2 pixel (stress test)
    ]

    strategies = [
        "runtime_depth",        # Current: 1/sqrt(2*L)
        "init_only",            # Proposed: runtime 1/sqrt(2), init *= 1/sqrt(2*L)
        "init_only_aggressive", # Aggressive: runtime 1.0, init *= 1/sqrt(2*L)
    ]

    all_results = {}

    for config_name, patch_layers, pixel_layers in configs:
        for strategy in strategies:
            key = f"{config_name}_{strategy}"
            print(f"\n{'#'*70}")
            print(f"# Running: {key}")
            print(f"{'#'*70}")

            # Reset seed for fair comparison
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = create_model(patch_layers, pixel_layers, strategy, device)
            result = experiment.run_single(model, strategy, config_name)
            all_results[key] = result.to_dict()

            del model
            torch.cuda.empty_cache()

    # Save results
    output_file = f"{output_dir}/comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    # Generate comparison report
    report = generate_report(all_results)
    report_file = f"{output_dir}/comparison_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    # Print summary
    print_summary(all_results)

    return all_results


def generate_report(results: Dict) -> str:
    """Generate markdown report from results."""
    lines = [
        "# Init-Only vs Runtime Depth Scaling Comparison Report",
        "",
        f"Date: 2026-01-23",
        "",
        "## Experiment Setup",
        "",
        "| Strategy | Runtime Scale | Init Scale |",
        "|----------|---------------|------------|",
        "| runtime_depth | 1/sqrt(2*L) | None |",
        "| init_only | 1/sqrt(2) | residual_proj *= 1/sqrt(2*L) |",
        "| init_only_aggressive | 1.0 | residual_proj *= 1/sqrt(2*L) |",
        "",
        "## Results Summary",
        "",
        "| Config | Strategy | Final Loss | Converged | Crashed | Time (s) |",
        "|--------|----------|------------|-----------|---------|----------|",
    ]

    for key, r in results.items():
        lines.append(
            f"| {r['config_name']} | {r['strategy']} | "
            f"{r['final_loss']:.6f} | {'Yes' if r['converged'] else 'No'} | "
            f"{'Yes' if r['crashed'] else 'No'} | {r['elapsed_time']:.1f} |"
        )

    lines.extend([
        "",
        "## Analysis",
        "",
        "### Shallow Configuration (2+1)",
        "",
    ])

    # Shallow analysis
    shallow_results = {k: v for k, v in results.items() if "shallow" in k}
    if shallow_results:
        baseline = shallow_results.get("shallow_runtime_depth", {})
        init_only = shallow_results.get("shallow_init_only", {})
        aggressive = shallow_results.get("shallow_init_only_aggressive", {})

        if baseline and init_only:
            baseline_loss = baseline.get("final_loss", float('nan'))
            init_loss = init_only.get("final_loss", float('nan'))
            diff = ((init_loss - baseline_loss) / baseline_loss * 100) if baseline_loss else 0
            lines.append(f"- Init-only vs Runtime: {diff:+.1f}% difference in final loss")

        if aggressive:
            lines.append(f"- Aggressive crashed: {'Yes' if aggressive.get('crashed') else 'No'}")

    lines.extend([
        "",
        "### Deep Configuration (8+2)",
        "",
    ])

    # Deep analysis
    deep_results = {k: v for k, v in results.items() if "deep" in k}
    if deep_results:
        baseline = deep_results.get("deep_runtime_depth", {})
        init_only = deep_results.get("deep_init_only", {})
        aggressive = deep_results.get("deep_init_only_aggressive", {})

        if baseline and init_only:
            baseline_loss = baseline.get("final_loss", float('nan'))
            init_loss = init_only.get("final_loss", float('nan'))
            diff = ((init_loss - baseline_loss) / baseline_loss * 100) if baseline_loss else 0
            lines.append(f"- Init-only vs Runtime: {diff:+.1f}% difference in final loss")

        if aggressive:
            lines.append(f"- Aggressive crashed: {'Yes' if aggressive.get('crashed') else 'No'}")

    lines.extend([
        "",
        "## Conclusions",
        "",
        "_(To be filled after running experiments)_",
        "",
        "### Recommendations",
        "",
        "1. If init_only shows faster convergence without crashing: Consider migration",
        "2. If init_only crashes or shows instability: Keep runtime_depth",
        "3. If init_only_aggressive crashes: init_only with 1/sqrt(2) is safer",
        "",
    ])

    return "\n".join(lines)


def print_summary(results: Dict):
    """Print comparison summary to console."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    headers = ["Config", "Strategy", "Final Loss", "Converged", "Crashed", "Time(s)"]
    print(f"{headers[0]:<12} {headers[1]:<22} {headers[2]:<12} {headers[3]:<10} {headers[4]:<8} {headers[5]:<8}")
    print("-" * 70)

    for key, r in results.items():
        config = r['config_name']
        strategy = r['strategy']
        loss = r['final_loss']
        conv = "Yes" if r['converged'] else "No"
        crash = "Yes" if r['crashed'] else "No"
        elapsed = r['elapsed_time']

        print(f"{config:<12} {strategy:<22} {loss:<12.6f} {conv:<10} {crash:<8} {elapsed:<8.1f}")

    print("=" * 70)

    # Convergence speed comparison
    print("\nConvergence Speed Comparison:")
    for config in ["shallow", "deep"]:
        print(f"\n{config.upper()}:")
        config_results = {k: v for k, v in results.items() if config in k}

        for key, r in sorted(config_results.items()):
            steps = r.get("steps", [])
            if len(steps) >= 2:
                # Compare loss at step 100 vs step 0
                step_0 = next((s for s in steps if s["step"] == 0), None)
                step_100 = next((s for s in steps if s["step"] == 100), None)

                if step_0 and step_100:
                    reduction = (step_0["loss"] - step_100["loss"]) / step_0["loss"] * 100
                    print(f"  {r['strategy']}: {reduction:.1f}% loss reduction by step 100")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Init-Only vs Runtime Scaling Comparison")
    parser.add_argument("--image", type=str, default=r"G:\Experimental_H_model\單圖測試\01.jpg")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/init_only_vs_runtime")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_comparison(
        image_path=args.image,
        num_steps=args.steps,
        image_size=args.size,
        learning_rate=args.lr,
        output_dir=args.output,
        seed=args.seed,
    )
