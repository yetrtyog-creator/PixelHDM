"""
单图过拟合测试 - 测试模型是否能完全记住一张图片

这是验证模型架构正确性的关键测试：
- 如果模型正确，应该能在 ~1000 步内将 loss 降到接近 0
- 如果 loss 卡住，说明架构或训练流程有问题

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-22
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class OverfitResults:
    """单图过拟合测试结果"""
    steps: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    v_loss: list = field(default_factory=list)
    correlation: list = field(default_factory=list)  # 输出与目标的相关性
    output_std: list = field(default_factory=list)
    target_std: list = field(default_factory=list)

    # 诊断指标
    s_cond_std: list = field(default_factory=list)
    gamma1_mean: list = field(default_factory=list)
    alpha1_mean: list = field(default_factory=list)
    compaction_ratio: list = field(default_factory=list)

    elapsed_time: float = 0.0
    final_loss: float = float('inf')
    converged: bool = False

    def print_summary(self):
        print(f"\n{'='*60}")
        print("SINGLE IMAGE OVERFIT TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total steps: {len(self.loss)}")
        print(f"Elapsed: {self.elapsed_time:.1f}s")
        print(f"Initial loss: {self.loss[0]:.4f}")
        print(f"Final loss: {self.loss[-1]:.6f}")
        print(f"Converged: {'YES' if self.converged else 'NO'}")

        if self.correlation:
            print(f"Initial corr: {self.correlation[0]:.4f}")
            print(f"Final corr: {self.correlation[-1]:.4f}")

        # Print milestones
        print(f"\nMilestones:")
        milestones = [0, len(self.loss)//4, len(self.loss)//2, 3*len(self.loss)//4, len(self.loss)-1]
        for i in milestones:
            if i < len(self.loss):
                corr = self.correlation[i] if i < len(self.correlation) else 0
                print(f"  Step {self.steps[i]:4d}: loss={self.loss[i]:.6f}, corr={corr:.4f}")

        print(f"{'='*60}")


class SingleImageOverfitTest:
    """单图过拟合测试器"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_path: str,
        caption: str = "",
        image_size: int = 512,
        learning_rate: float = 1e-4,
        num_steps: int = 1000,
        log_interval: int = 50,
        convergence_threshold: float = 0.01,
    ):
        self.model = model
        self.device = device
        self.image_path = image_path
        self.caption = caption
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.convergence_threshold = convergence_threshold

        self.config = getattr(model, 'config', None)
        self.hidden_dim = getattr(self.config, 'hidden_dim', 1024) if self.config else 1024

        # 加载图片
        self.image = self._load_image()

        # 创建固定的文本嵌入（随机但固定）
        torch.manual_seed(42)
        self.text_embed = torch.randn(1, 32, self.hidden_dim, device=device)
        self.text_mask = torch.ones(1, 32, dtype=torch.bool, device=device)
        self.pooled_text_embed = torch.randn(1, self.hidden_dim, device=device)

        # 设置优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # 设置损失函数
        from src.training.losses import VLoss
        self.v_loss_fn = VLoss()

    def _load_image(self) -> torch.Tensor:
        """Load and preprocess image"""
        img = Image.open(self.image_path).convert('RGB')

        # Resize to target size
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [-1, 1]
        img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        img_np = (img_np - 0.5) / 0.5  # [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (C, H, W)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

        return img_tensor

    def _create_batch(self, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """创建训练 batch"""
        x_clean = self.image

        # 随机时间步
        if t is None:
            t = torch.rand(1, device=self.device) * 0.9 + 0.05  # [0.05, 0.95]

        # 固定噪声（用于一致性检查）
        torch.manual_seed(int(t.item() * 10000) % 10000)
        noise = torch.randn_like(x_clean)

        # Flow matching: z_t = t * x + (1-t) * noise
        x_t = t.view(1, 1, 1, 1) * x_clean + (1 - t.view(1, 1, 1, 1)) * noise

        # v_target = x - noise
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
        """计算预测和目标的相关性"""
        pred_flat = pred.flatten().float()
        target_flat = target.flatten().float()

        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()

        corr = (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm() + 1e-8
        )
        return corr.item()

    def run(self) -> OverfitResults:
        """Run single image overfit test"""
        results = OverfitResults()

        print(f"Starting single image overfit test...")
        print(f"Image: {self.image_path}")
        print(f"Size: {self.image_size}x{self.image_size}")
        print(f"Steps: {self.num_steps}")
        print(f"LR: {self.learning_rate}")
        print(f"Device: {self.device}")
        print()

        start_time = time.time()

        self.model.train()

        for step in range(self.num_steps):
            # 创建 batch
            batch = self._create_batch()

            # 前向传播
            self.optimizer.zero_grad()

            output = self.model(
                x_t=batch["x_t"],
                t=batch["t"],
                text_embed=batch["text_embed"],
                text_mask=batch["text_mask"],
                pooled_text_embed=batch["pooled_text_embed"],
            )

            # 转换为 BCHW
            if output.shape[-1] == 3:
                output = output.permute(0, 3, 1, 2)

            # 计算损失
            loss = self.v_loss_fn(output, batch["x_clean"], batch["noise"])

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录
            loss_val = loss.item()
            results.steps.append(step)
            results.loss.append(loss_val)
            results.v_loss.append(loss_val)

            # 计算相关性
            with torch.no_grad():
                corr = self._compute_correlation(output, batch["v_target"])
                results.correlation.append(corr)
                results.output_std.append(output.std().item())
                results.target_std.append(batch["v_target"].std().item())

            # 打印进度
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:4d}: loss={loss_val:.6f}, corr={corr:.4f}, "
                      f"out_std={output.std().item():.3f}, elapsed={elapsed:.1f}s")

            # Check convergence
            if loss_val < self.convergence_threshold:
                results.converged = True
                print(f"\nCONVERGED at step {step}! loss={loss_val:.6f}")
                break

        results.elapsed_time = time.time() - start_time
        results.final_loss = results.loss[-1]

        return results

    def generate_and_save(self, output_dir: str = "outputs/overfit_test") -> str:
        """Generate image from trained model and save comparison."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            # Use fixed t=0.5 for generation
            batch = self._create_batch(t=torch.tensor([0.5], device=self.device))

            output = self.model(
                x_t=batch["x_t"],
                t=batch["t"],
                text_embed=batch["text_embed"],
                text_mask=batch["text_mask"],
                pooled_text_embed=batch["pooled_text_embed"],
            )

            # Convert output (v_pred) to image estimate
            # v = x - noise, so x = v + noise, but we need to use flow matching formula
            # z_t = t*x + (1-t)*noise, v = x - noise
            # x_pred = z_t + (1-t)*v = t*x + (1-t)*noise + (1-t)*(x-noise) = t*x + (1-t)*x = x
            # Simplified: x_pred = x_t + (1 - t) * v_pred
            if output.shape[-1] == 3:  # BHWC
                output = output.permute(0, 3, 1, 2)  # BCHW

            t = batch["t"].view(1, 1, 1, 1)
            x_pred = batch["x_t"] + (1 - t) * output

            # Clamp and convert to image
            x_pred = x_pred.clamp(-1, 1)
            x_pred_np = ((x_pred[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            pred_img = Image.fromarray(x_pred_np)

            # Original image
            x_clean = self.image
            x_clean_np = ((x_clean[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            orig_img = Image.fromarray(x_clean_np)

            # v_pred visualization
            v_pred_vis = output[0].cpu().numpy().transpose(1, 2, 0)
            v_pred_vis = ((v_pred_vis - v_pred_vis.min()) / (v_pred_vis.max() - v_pred_vis.min() + 1e-8) * 255).astype(np.uint8)
            v_img = Image.fromarray(v_pred_vis)

            # Save individual images
            orig_img.save(f"{output_dir}/original.png")
            pred_img.save(f"{output_dir}/predicted.png")
            v_img.save(f"{output_dir}/v_pred.png")

            # Create comparison grid
            grid = Image.new('RGB', (self.image_size * 3, self.image_size))
            grid.paste(orig_img, (0, 0))
            grid.paste(pred_img, (self.image_size, 0))
            grid.paste(v_img, (self.image_size * 2, 0))
            grid.save(f"{output_dir}/comparison.png")

            print(f"\nImages saved to {output_dir}/")
            print(f"  - original.png: Ground truth")
            print(f"  - predicted.png: Model reconstruction at t=0.5")
            print(f"  - v_pred.png: Velocity prediction (normalized)")
            print(f"  - comparison.png: Side-by-side [orig | pred | v_pred]")

            return f"{output_dir}/comparison.png"


def run_single_image_overfit_test(
    image_path: str = r"G:\Experimental_H_model\單圖測試\01.jpg",
    num_steps: int = 500,
    image_size: int = 256,
    learning_rate: float = 1e-4,
    use_cuda: bool = True,
):
    """Run single image overfit test"""

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    from src.config import PixelHDMConfig
    from src.models.pixelhdm import PixelHDM

    print("Creating model...")
    config = PixelHDMConfig.for_testing()
    model = PixelHDM(config=config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params/1e6:.1f}M")

    # 运行测试
    tester = SingleImageOverfitTest(
        model=model,
        device=device,
        image_path=image_path,
        image_size=image_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        log_interval=50,
        convergence_threshold=0.01,
    )

    results = tester.run()
    results.print_summary()

    # Save comparison images
    tester.generate_and_save()

    return results, tester


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="单图过拟合测试")
    parser.add_argument("--image", type=str, default=r"G:\Experimental_H_model\單圖測試\01.jpg")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    run_single_image_overfit_test(
        image_path=args.image,
        num_steps=args.steps,
        image_size=args.size,
        learning_rate=args.lr,
        use_cuda=not args.no_cuda,
    )
