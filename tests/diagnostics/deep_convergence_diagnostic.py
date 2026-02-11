"""
Deep Convergence Diagnostic - Deep vs Shallow Model Comparison

This script implements the diagnostic plan from GPT/改進深層收斂/:
1. Compare deep vs shallow models (iso-shape, only change depth)
2. Fixed seed/batch/t/noise for reproducibility
3. Single-image overfit 200-500 steps
4. Log 4 key metrics every 10 steps:
   - update_ratio: std(x_out - x_in) / std(x_in) per layer
   - gamma stats: mean/std + (1+gamma)<0 ratio per layer
   - compaction_ratio: std(h_after_compaction) / std(x_in) per layer
   - grad_norm: per layer/block

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-23
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LayerMetrics:
    """Metrics for a single layer at a single step."""
    update_ratio: float = 0.0  # std(x_out - x_in) / std(x_in)
    gamma_mean: float = 0.0
    gamma_std: float = 0.0
    gamma_neg_ratio: float = 0.0  # ratio of (1+gamma) < 0
    alpha_mean: float = 0.0
    compaction_ratio: float = 0.0  # std(h) / std(x_in)
    grad_norm: float = 0.0


@dataclass
class StepMetrics:
    """All metrics for a single training step."""
    step: int
    loss: float
    correlation: float
    patch_block_metrics: List[LayerMetrics] = field(default_factory=list)
    pixel_block_metrics: List[LayerMetrics] = field(default_factory=list)


@dataclass
class DiagnosticResults:
    """Complete diagnostic results for a run."""
    config_name: str
    patch_layers: int
    pixel_layers: int
    hidden_dim: int
    total_params: int
    steps: List[StepMetrics] = field(default_factory=list)
    elapsed_time: float = 0.0
    converged: bool = False
    convergence_step: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "config_name": self.config_name,
            "patch_layers": self.patch_layers,
            "pixel_layers": self.pixel_layers,
            "hidden_dim": self.hidden_dim,
            "total_params": self.total_params,
            "elapsed_time": self.elapsed_time,
            "converged": self.converged,
            "convergence_step": self.convergence_step,
            "steps": [
                {
                    "step": s.step,
                    "loss": s.loss,
                    "correlation": s.correlation,
                    "patch_block_metrics": [asdict(m) for m in s.patch_block_metrics],
                    "pixel_block_metrics": [asdict(m) for m in s.pixel_block_metrics],
                }
                for s in self.steps
            ]
        }


# ============================================================================
# Diagnostic Hooks
# ============================================================================

class DiagnosticHookManager:
    """Manages hooks for capturing intermediate activations."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captured: Dict[str, torch.Tensor] = {}
        self.grad_norms: Dict[str, float] = {}

    def _capture_hook(self, name: str):
        """Hook to capture forward output."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.captured[name] = output.detach()
        return hook

    def _capture_input_hook(self, name: str):
        """Hook to capture forward input."""
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                self.captured[name] = input[0].detach()
            else:
                self.captured[name] = input.detach()
        return hook

    def attach(self):
        """Attach all diagnostic hooks."""
        # Patch blocks
        if hasattr(self.model, 'patch_blocks'):
            for i, block in enumerate(self.model.patch_blocks):
                # Capture input and output for update_ratio
                h = block.register_forward_hook(self._capture_input_hook(f"patch_block_{i}_input"))
                self.handles.append(h)
                h = block.register_forward_hook(self._capture_hook(f"patch_block_{i}_output"))
                self.handles.append(h)

        # Pixel blocks
        if hasattr(self.model, 'pixel_blocks'):
            for i, block in enumerate(self.model.pixel_blocks):
                # Capture input and output
                h = block.register_forward_hook(self._capture_input_hook(f"pixel_block_{i}_input"))
                self.handles.append(h)
                h = block.register_forward_hook(self._capture_hook(f"pixel_block_{i}_output"))
                self.handles.append(h)

                # Capture compaction output
                if hasattr(block, 'compaction'):
                    h = block.compaction.register_forward_hook(
                        self._capture_hook(f"pixel_block_{i}_compaction_output")
                    )
                    self.handles.append(h)

    def detach(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self):
        """Clear captured values."""
        self.captured.clear()
        self.grad_norms.clear()

    def compute_grad_norms(self):
        """Compute gradient norms for each block."""
        # Patch blocks
        if hasattr(self.model, 'patch_blocks'):
            for i, block in enumerate(self.model.patch_blocks):
                total_norm = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                self.grad_norms[f"patch_block_{i}"] = math.sqrt(total_norm)

        # Pixel blocks
        if hasattr(self.model, 'pixel_blocks'):
            for i, block in enumerate(self.model.pixel_blocks):
                total_norm = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                self.grad_norms[f"pixel_block_{i}"] = math.sqrt(total_norm)

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()


# ============================================================================
# AdaLN Parameter Extractor
# ============================================================================

def extract_adaln_params(
    model: nn.Module,
    s_cond: torch.Tensor,
    num_pixel_blocks: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract gamma and alpha from pixel blocks' AdaLN layers."""
    results = []
    with torch.no_grad():
        for i, block in enumerate(model.pixel_blocks):
            if hasattr(block, 'adaln'):
                params = block.adaln(s_cond)
                if len(params) >= 6:
                    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params[:6]
                    results.append((gamma1, alpha1))
    return results


# ============================================================================
# Diagnostic Runner
# ============================================================================

class DeepConvergenceDiagnostic:
    """Main diagnostic class for deep vs shallow comparison."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_path: str,
        config_name: str = "unknown",
        image_size: int = 256,
        learning_rate: float = 1e-4,
        num_steps: int = 300,
        log_interval: int = 10,
        convergence_threshold: float = 0.01,
        seed: int = 42,
    ):
        self.model = model
        self.device = device
        self.image_path = image_path
        self.config_name = config_name
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.convergence_threshold = convergence_threshold
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

    def _create_batch(self, t_value: Optional[float] = None, step: int = 0) -> Dict[str, torch.Tensor]:
        """Create training batch with fixed noise for given t."""
        x_clean = self.image

        if t_value is None:
            # Use step-based seed for full reproducibility
            torch.manual_seed(self.seed + step * 7919)  # prime number for better spread
            t = torch.rand(1, device=self.device) * 0.9 + 0.05
        else:
            t = torch.tensor([t_value], device=self.device)

        # Fixed noise based on step (reproducible)
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

    def _compute_update_ratio(self, x_in: torch.Tensor, x_out: torch.Tensor) -> float:
        """Compute update ratio: std(x_out - x_in) / std(x_in)."""
        delta = x_out - x_in
        x_in_std = x_in.std().item()
        delta_std = delta.std().item()
        if x_in_std < 1e-8:
            return 0.0
        return delta_std / x_in_std

    def _compute_gamma_stats(self, gamma: torch.Tensor) -> Tuple[float, float, float]:
        """Compute gamma statistics: mean, std, and negative ratio."""
        gamma_flat = gamma.flatten().float()
        gamma_mean = gamma_flat.mean().item()
        gamma_std = gamma_flat.std().item()
        # (1 + gamma) < 0 means gamma < -1
        neg_ratio = ((gamma_flat + 1.0) < 0).float().mean().item()
        return gamma_mean, gamma_std, neg_ratio

    def _extract_s_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract s_cond for AdaLN parameter computation."""
        with torch.no_grad():
            x_t = batch["x_t"]
            if x_t.shape[1] == 3:
                x_t = x_t.permute(0, 2, 3, 1)

            x = self.model.patch_embed(x_t)
            t_embed = self.model.time_embed(batch["t"])

            text_embed = batch.get("text_embed")
            text_mask = batch.get("text_mask")
            text_len = 0

            if text_embed is not None:
                T = text_embed.shape[1]
                x = torch.cat([text_embed, x], dim=1)
                text_len = T

            # Create position IDs
            from src.models.layers.rope import create_position_ids_batched
            B = batch["x_t"].shape[0]
            H = W = self.image_size

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

            joint_mask = None
            if text_mask is not None:
                img_mask = torch.ones(B, x.shape[1] - text_len, dtype=torch.bool, device=x.device)
                joint_mask = torch.cat([text_mask, img_mask], dim=1)

            for block in self.model.patch_blocks:
                x = block(x, t_embed=t_embed, rope_fn=rope_fn, position_ids=position_ids, attention_mask=joint_mask)

            semantic_tokens = x[:, text_len:] if text_len > 0 else x

            s_cond = semantic_tokens + t_embed.unsqueeze(1)
            pooled_text_embed = batch.get("pooled_text_embed")
            if pooled_text_embed is not None:
                s_cond = s_cond + pooled_text_embed.unsqueeze(1)

            return s_cond

    def run(self) -> DiagnosticResults:
        """Run the diagnostic test."""
        results = DiagnosticResults(
            config_name=self.config_name,
            patch_layers=self.patch_layers,
            pixel_layers=self.pixel_layers,
            hidden_dim=self.hidden_dim,
            total_params=sum(p.numel() for p in self.model.parameters()),
        )

        print(f"\n{'='*60}")
        print(f"DEEP CONVERGENCE DIAGNOSTIC: {self.config_name}")
        print(f"{'='*60}")
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

        hook_manager = DiagnosticHookManager(self.model)

        with hook_manager:
            for step in range(self.num_steps):
                hook_manager.clear()
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

                # Compute grad norms before optimizer step
                hook_manager.compute_grad_norms()

                self.optimizer.step()

                # Record metrics at log intervals
                if step % self.log_interval == 0:
                    with torch.no_grad():
                        corr = self._compute_correlation(output, batch["v_target"])

                        step_metrics = StepMetrics(
                            step=step,
                            loss=loss.item(),
                            correlation=corr,
                        )

                        # Patch block metrics
                        for i in range(self.patch_layers):
                            input_key = f"patch_block_{i}_input"
                            output_key = f"patch_block_{i}_output"
                            grad_key = f"patch_block_{i}"

                            layer_metrics = LayerMetrics()

                            if input_key in hook_manager.captured and output_key in hook_manager.captured:
                                x_in = hook_manager.captured[input_key]
                                x_out = hook_manager.captured[output_key]
                                layer_metrics.update_ratio = self._compute_update_ratio(x_in, x_out)

                            if grad_key in hook_manager.grad_norms:
                                layer_metrics.grad_norm = hook_manager.grad_norms[grad_key]

                            step_metrics.patch_block_metrics.append(layer_metrics)

                        # Pixel block metrics (with AdaLN stats)
                        s_cond = self._extract_s_cond(batch)
                        adaln_params = extract_adaln_params(self.model, s_cond, self.pixel_layers)

                        for i in range(self.pixel_layers):
                            input_key = f"pixel_block_{i}_input"
                            output_key = f"pixel_block_{i}_output"
                            comp_key = f"pixel_block_{i}_compaction_output"
                            grad_key = f"pixel_block_{i}"

                            layer_metrics = LayerMetrics()

                            if input_key in hook_manager.captured and output_key in hook_manager.captured:
                                x_in = hook_manager.captured[input_key]
                                x_out = hook_manager.captured[output_key]
                                layer_metrics.update_ratio = self._compute_update_ratio(x_in, x_out)

                                # Compaction ratio
                                if comp_key in hook_manager.captured:
                                    h_comp = hook_manager.captured[comp_key]
                                    x_in_std = x_in.std().item()
                                    h_comp_std = h_comp.std().item()
                                    if x_in_std > 1e-8:
                                        layer_metrics.compaction_ratio = h_comp_std / x_in_std

                            # AdaLN gamma/alpha stats
                            if i < len(adaln_params):
                                gamma, alpha = adaln_params[i]
                                gamma_mean, gamma_std, gamma_neg = self._compute_gamma_stats(gamma)
                                layer_metrics.gamma_mean = gamma_mean
                                layer_metrics.gamma_std = gamma_std
                                layer_metrics.gamma_neg_ratio = gamma_neg
                                layer_metrics.alpha_mean = alpha.mean().item()

                            if grad_key in hook_manager.grad_norms:
                                layer_metrics.grad_norm = hook_manager.grad_norms[grad_key]

                            step_metrics.pixel_block_metrics.append(layer_metrics)

                        results.steps.append(step_metrics)

                        # Print progress
                        print(f"Step {step:4d}: loss={loss.item():.6f}, corr={corr:.4f}")

                        # Print detailed pixel block metrics
                        if len(step_metrics.pixel_block_metrics) > 0:
                            print(f"  Pixel blocks: ", end="")
                            for i, m in enumerate(step_metrics.pixel_block_metrics):
                                print(f"[{i}] ur={m.update_ratio:.3f} g={m.gamma_mean:.3f}({m.gamma_neg_ratio:.2%}) ", end="")
                            print()

                # Check convergence
                if loss.item() < self.convergence_threshold:
                    results.converged = True
                    results.convergence_step = step
                    print(f"\nCONVERGED at step {step}! loss={loss.item():.6f}")
                    break

        results.elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC COMPLETE")
        print(f"Elapsed: {results.elapsed_time:.1f}s")
        print(f"Final loss: {results.steps[-1].loss:.6f}")
        print(f"Converged: {'YES' if results.converged else 'NO'}")
        print(f"{'='*60}")

        return results

    def generate_and_save(self, output_dir: str, num_steps: int = 50) -> str:
        """Generate image using full sampling and save comparison.

        Uses Euler method for flow matching:
        - Start from t=0 (pure noise)
        - Step to t=1 (clean image)
        - z_{t+dt} = z_t + dt * v_pred
        """
        import os
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

                if (i + 1) % 10 == 0:
                    print(f"  Step {i+1}/{num_steps}")

            # Final result
            x_pred = z.clamp(-1, 1)
            x_pred_np = ((x_pred[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            pred_img = Image.fromarray(x_pred_np)

            # Original image
            x_clean = self.image
            x_clean_np = ((x_clean[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            orig_img = Image.fromarray(x_clean_np)

            # Save images
            orig_path = f"{output_dir}/{self.config_name}_original.png"
            pred_path = f"{output_dir}/{self.config_name}_sampled.png"

            orig_img.save(orig_path)
            pred_img.save(pred_path)

            # Create comparison grid [orig | sampled]
            grid = Image.new('RGB', (self.image_size * 2, self.image_size))
            grid.paste(orig_img, (0, 0))
            grid.paste(pred_img, (self.image_size, 0))
            grid_path = f"{output_dir}/{self.config_name}_comparison.png"
            grid.save(grid_path)

            print(f"Images saved to {output_dir}/")
            print(f"  - {self.config_name}_original.png")
            print(f"  - {self.config_name}_sampled.png")
            print(f"  - {self.config_name}_comparison.png")

            return grid_path


# ============================================================================
# Configuration Helpers
# ============================================================================

def create_shallow_config():
    """Create shallow config (2 patch + 1 pixel layers)."""
    from src.config import PixelHDMConfig
    return PixelHDMConfig(
        hidden_dim=256,
        pixel_dim=8,
        patch_size=16,
        patch_layers=2,
        pixel_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        mlp_ratio=2.0,
        bottleneck_dim=64,
        text_hidden_size=256,
        repa_enabled=False,
        freq_loss_enabled=False,
        use_flash_attention=False,
        use_gradient_checkpointing=False,
        zero_init_output=False,
    )


def create_deep_config():
    """Create deep config (8 patch + 2 pixel layers) - iso-shape with shallow."""
    from src.config import PixelHDMConfig
    return PixelHDMConfig(
        hidden_dim=256,  # Same as shallow
        pixel_dim=8,     # Same
        patch_size=16,   # Same
        patch_layers=8,  # Deeper: 2 -> 8
        pixel_layers=2,  # Deeper: 1 -> 2
        num_heads=4,     # Same
        num_kv_heads=2,  # Same
        head_dim=64,     # Same
        mlp_ratio=2.0,   # Same
        bottleneck_dim=64,  # Same
        text_hidden_size=256,  # Same
        repa_enabled=False,
        freq_loss_enabled=False,
        use_flash_attention=False,
        use_gradient_checkpointing=False,
        zero_init_output=False,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def run_comparison(
    image_path: str = r"G:\Experimental_H_model\單圖測試\01.jpg",
    num_steps: int = 300,
    image_size: int = 256,
    learning_rate: float = 1e-4,
    output_dir: str = "outputs/deep_convergence_diagnostic",
    seed: int = 42,
):
    """Run deep vs shallow comparison diagnostic."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src.models.pixelhdm import PixelHDM

    all_results = {}

    # Run shallow
    print("\n" + "="*70)
    print("RUNNING SHALLOW MODEL")
    print("="*70)

    shallow_config = create_shallow_config()
    shallow_model = PixelHDM(config=shallow_config).to(device)

    shallow_diagnostic = DeepConvergenceDiagnostic(
        model=shallow_model,
        device=device,
        image_path=image_path,
        config_name="shallow",
        image_size=image_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
    )
    shallow_results = shallow_diagnostic.run()
    all_results["shallow"] = shallow_results.to_dict()

    # Generate and save shallow model output image
    shallow_diagnostic.generate_and_save(output_dir)

    del shallow_model
    torch.cuda.empty_cache()

    # Run deep
    print("\n" + "="*70)
    print("RUNNING DEEP MODEL")
    print("="*70)

    deep_config = create_deep_config()
    deep_model = PixelHDM(config=deep_config).to(device)

    deep_diagnostic = DeepConvergenceDiagnostic(
        model=deep_model,
        device=device,
        image_path=image_path,
        config_name="deep",
        image_size=image_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
    )
    deep_results = deep_diagnostic.run()
    all_results["deep"] = deep_results.to_dict()

    # Generate and save deep model output image
    deep_diagnostic.generate_and_save(output_dir)

    # Create side-by-side comparison of both models
    shallow_pred = Image.open(f"{output_dir}/shallow_sampled.png")
    deep_pred = Image.open(f"{output_dir}/deep_sampled.png")
    orig = Image.open(f"{output_dir}/shallow_original.png")

    combined = Image.new('RGB', (image_size * 3, image_size))
    combined.paste(orig, (0, 0))
    combined.paste(shallow_pred, (image_size, 0))
    combined.paste(deep_pred, (image_size * 2, 0))
    combined.save(f"{output_dir}/final_comparison.png")
    print(f"\nFinal comparison saved: {output_dir}/final_comparison.png")
    print("  [Original | Shallow | Deep]")

    # Save results
    output_file = f"{output_dir}/comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<30} {'Shallow':<15} {'Deep':<15}")
    print("-"*60)
    print(f"{'Patch layers':<30} {shallow_results.patch_layers:<15} {deep_results.patch_layers:<15}")
    print(f"{'Pixel layers':<30} {shallow_results.pixel_layers:<15} {deep_results.pixel_layers:<15}")
    print(f"{'Total params':<30} {shallow_results.total_params/1e6:.2f}M{'':<10} {deep_results.total_params/1e6:.2f}M")
    print(f"{'Final loss':<30} {shallow_results.steps[-1].loss:.6f}{'':<9} {deep_results.steps[-1].loss:.6f}")
    print(f"{'Converged':<30} {'YES' if shallow_results.converged else 'NO':<15} {'YES' if deep_results.converged else 'NO'}")
    print(f"{'Elapsed time':<30} {shallow_results.elapsed_time:.1f}s{'':<10} {deep_results.elapsed_time:.1f}s")
    print("="*70)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Convergence Diagnostic")
    parser.add_argument("--image", type=str, default=r"G:\Experimental_H_model\單圖測試\01.jpg")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/deep_convergence_diagnostic")
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
