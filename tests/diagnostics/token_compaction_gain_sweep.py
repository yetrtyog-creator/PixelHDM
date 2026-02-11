"""
Token Compaction Expand Gain Sweep

Runs scale probes and short training loops to compare expand_gain settings.
This script is designed to be low-cost and reproducible on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import PixelHDMConfig
from src.models import PixelHDM
from src.models.layers.rope import create_position_ids_batched
from src.models.attention.gating import GateProjection


def parse_gains(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(
    config: PixelHDMConfig,
    batch_size: int,
    image_size: int,
    text_len: int,
    device: torch.device,
    gen: torch.Generator,
) -> Dict[str, torch.Tensor]:
    h = w = image_size
    x_clean = torch.randn(batch_size, 3, h, w, generator=gen)
    t = torch.rand(batch_size, generator=gen)
    noise = torch.randn(x_clean.shape, generator=gen)
    x_t = t.view(batch_size, 1, 1, 1) * x_clean + (1 - t.view(batch_size, 1, 1, 1)) * noise

    text_embed = torch.randn(batch_size, text_len, config.hidden_dim, generator=gen)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)

    x_clean = x_clean.to(device)
    t = t.to(device)
    noise = noise.to(device)
    x_t = x_t.to(device)
    text_embed = text_embed.to(device)
    text_mask = text_mask.to(device)

    return {
        "x_t": x_t,
        "x_clean": x_clean,
        "t": t,
        "noise": noise,
        "text_embed": text_embed,
        "text_mask": text_mask,
    }


def compute_s_cond(
    model: PixelHDM,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_t = batch["x_t"]
    if x_t.shape[1] == 3:
        x_t_bhwc = x_t.permute(0, 2, 3, 1)
    else:
        x_t_bhwc = x_t

    x = model.patch_embed(x_t_bhwc)
    t_embed = model.time_embed(batch["t"])

    text_embed = batch.get("text_embed")
    text_mask = batch.get("text_mask")
    if text_embed is not None and model.text_projector is not None:
        if text_embed.shape[-1] != model.hidden_dim:
            text_embed = model.text_projector(text_embed)
        if model.text_processor is not None:
            if text_mask is not None:
                text_mask = text_mask.to(dtype=torch.bool)
            text_embed = model.text_processor(text_embed, text_mask=text_mask)

    x, joint_mask, text_len, text_only_mask = model._create_joint_sequence(
        x, text_embed, text_mask
    )

    b, _, _ = x.shape
    h, w = x_t_bhwc.shape[1], x_t_bhwc.shape[2]
    position_ids = create_position_ids_batched(
        batch_size=b,
        text_len=text_len,
        img_height=h,
        img_width=w,
        patch_size=model.patch_size,
        device=x.device,
        text_mask=text_only_mask,
    )
    rope_fn = model._create_rope_fn(position_ids)

    for block in model.patch_blocks:
        x = block(
            x,
            t_embed=t_embed,
            rope_fn=rope_fn,
            position_ids=position_ids,
            attention_mask=joint_mask,
        )

    if text_len > 0:
        semantic_tokens = x[:, text_len:]
    else:
        semantic_tokens = x
    s_cond = semantic_tokens + t_embed.unsqueeze(1)
    return s_cond, x_t_bhwc


def compute_gate_stats(
    compaction,
    h_mod: torch.Tensor,
) -> Optional[Dict[str, float]]:
    attn = getattr(compaction, "attention", None)
    if attn is None or not hasattr(attn, "gate"):
        return None

    gate = attn.gate
    if not isinstance(gate, GateProjection):
        return None

    b, l, p2, d = h_mod.shape
    x_flat = h_mod.reshape(b, l, p2 * d)
    x_comp = compaction.compress(x_flat)
    x_comp = compaction.norm(x_comp)

    gate_scores = gate.proj(x_comp)
    gate_values = gate.activation(gate_scores)

    stats: Dict[str, float] = {
        "gate_mean": gate_values.mean().item(),
        "gate_std": gate_values.std().item(),
    }

    if gate.activation == torch.sigmoid:
        stats["gate_sat_low"] = (gate_values < 0.05).float().mean().item()
        stats["gate_sat_high"] = (gate_values > 0.95).float().mean().item()

    return stats


def probe_compaction(
    model: PixelHDM,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    with torch.no_grad():
        s_cond, x_t_bhwc = compute_s_cond(model, batch)
        x_pixel = model.pixel_embed(x_t_bhwc)

        b, _, _ = s_cond.shape
        h, w = x_t_bhwc.shape[1], x_t_bhwc.shape[2]
        pixel_position_ids = create_position_ids_batched(
            batch_size=b,
            text_len=0,
            img_height=h,
            img_width=w,
            patch_size=model.patch_size,
            device=x_pixel.device,
        )
        pixel_rope_fn = model._create_rope_fn(pixel_position_ids)

        eps = 1e-8
        blocks: List[Dict[str, float]] = []
        for block in model.pixel_blocks:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = block.adaln(s_cond)

            x_in = x_pixel
            x_in_std = x_in.std().item()

            h_mod = block.adaln.modulate(x_in, gamma1, beta1)
            gate_stats = compute_gate_stats(block.compaction, h_mod)

            h = block.compaction(h_mod, pixel_rope_fn, pixel_position_ids)
            h_std = h.std().item()

            update = (alpha1 * block.residual_scale) * h
            update_std = update.std().item()

            block_stats: Dict[str, float] = {
                "x_in_std": x_in_std,
                "h_std": h_std,
                "compaction_ratio": h_std / (x_in_std + eps),
                "update_std": update_std,
                "update_ratio": update_std / (x_in_std + eps),
                "alpha1_mean": alpha1.mean().item(),
                "alpha1_std": alpha1.std().item(),
            }
            if gate_stats:
                block_stats.update(gate_stats)

            blocks.append(block_stats)

            # Full block update for next layer
            x_pixel = x_in + update
            h2 = block.adaln.modulate(x_pixel, gamma2, beta2)
            h2 = block.mlp(h2)
            x_pixel = x_pixel + (alpha2 * block.residual_scale) * h2

        # Aggregate
        def _mean(key: str) -> Optional[float]:
            vals = [b[key] for b in blocks if key in b]
            if not vals:
                return None
            return sum(vals) / len(vals)

        summary = {
            "compaction_ratio_avg": _mean("compaction_ratio"),
            "update_ratio_avg": _mean("update_ratio"),
            "alpha1_mean_avg": _mean("alpha1_mean"),
            "gate_mean_avg": _mean("gate_mean"),
            "gate_std_avg": _mean("gate_std"),
            "gate_sat_low_avg": _mean("gate_sat_low"),
            "gate_sat_high_avg": _mean("gate_sat_high"),
        }

        return {
            "blocks": blocks,
            "summary": summary,
        }


def compute_grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def reinit_compaction_expand(model: PixelHDM, gain: float) -> None:
    for module in model.modules():
        if module.__class__.__name__ in {"TokenCompaction", "TokenCompactionNoResidual"}:
            nn.init.xavier_uniform_(module.expand.weight, gain=gain)


def run_short_training(
    model: PixelHDM,
    config: PixelHDMConfig,
    batch_size: int,
    image_size: int,
    text_len: int,
    steps: int,
    lr: float,
    device: torch.device,
    data_seed: int,
) -> Dict[str, float]:
    if steps <= 0:
        return {}

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: List[float] = []
    grad_norms: List[float] = []

    gen = torch.Generator()
    gen.manual_seed(data_seed)

    for _ in range(steps):
        batch = make_batch(config, batch_size, image_size, text_len, device, gen)
        opt.zero_grad()
        output = model(
            x_t=batch["x_t"],
            t=batch["t"],
            text_embed=batch["text_embed"],
            text_mask=batch["text_mask"],
        )
        if output.shape[-1] == 3:
            output = output.permute(0, 3, 1, 2)

        v_target = batch["x_clean"] - batch["noise"]
        loss = (output - v_target).pow(2).mean()
        loss.backward()

        grad_norms.append(compute_grad_norm(model.parameters()))
        opt.step()
        losses.append(loss.item())

    loss_start = losses[0]
    loss_end = losses[-1]
    slope = (loss_end - loss_start) / max(1, len(losses) - 1)
    auc = sum(losses) / len(losses)

    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_slope": slope,
        "loss_auc": auc,
        "grad_norm_mean": sum(grad_norms) / len(grad_norms),
        "grad_norm_max": max(grad_norms),
    }


def run_gain_sweep(
    gains: List[float],
    batch_size: int,
    image_size: int,
    text_len: int,
    train_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    reinit_expand: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    data_seed = seed + 999

    for gain in gains:
        set_seed(seed)
        config = PixelHDMConfig.for_testing()
        config.token_compaction_expand_gain = float(gain)

        model = PixelHDM(config=config).to(device)
        if reinit_expand:
            reinit_compaction_expand(model, float(gain))

        gen = torch.Generator()
        gen.manual_seed(data_seed)
        batch = make_batch(config, batch_size, image_size, text_len, device, gen)

        init_probe = probe_compaction(model, batch)
        train_metrics = run_short_training(
            model=model,
            config=config,
            batch_size=batch_size,
            image_size=image_size,
            text_len=text_len,
            steps=train_steps,
            lr=lr,
            device=device,
            data_seed=data_seed,
        )

        results.append({
            "gain": float(gain),
            "init_probe": init_probe,
            "train": train_metrics,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenCompaction expand_gain sweep")
    parser.add_argument("--gains", type=str, default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--text-len", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--reinit-expand",
        action="store_true",
        help="Reinitialize TokenCompaction expand weights after model init.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    gains = parse_gains(args.gains)

    results = run_gain_sweep(
        gains=gains,
        batch_size=args.batch_size,
        image_size=args.image_size,
        text_len=args.text_len,
        train_steps=args.train_steps,
        lr=args.lr,
        device=device,
        seed=args.seed,
        reinit_expand=args.reinit_expand,
    )

    # Print summary table
    print("\nGAIN SWEEP SUMMARY")
    print("=" * 72)
    header = "gain | upd_ratio | comp_ratio | gate_mean | loss_end | slope"
    print(header)
    print("-" * len(header))
    for r in results:
        summ = r["init_probe"]["summary"]
        train = r["train"]
        print(
            f"{r['gain']:>4.2f} | "
            f"{(summ.get('update_ratio_avg') or 0):>8.4f} | "
            f"{(summ.get('compaction_ratio_avg') or 0):>9.4f} | "
            f"{(summ.get('gate_mean_avg') or 0):>8.4f} | "
            f"{(train.get('loss_end') or 0):>7.4f} | "
            f"{(train.get('loss_slope') or 0):>6.4f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
