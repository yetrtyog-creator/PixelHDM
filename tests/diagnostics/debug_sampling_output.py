"""
Debug Sampling Output - Diagnose why sampling produces only noise

This script checks:
1. Model output magnitude during sampling
2. Velocity direction correctness
3. EMA weight loading
4. Pooling sync contribution
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import PixelHDMConfig
from src.models.pixelhdm import PixelHDM
from src.inference.sampler import create_sampler


def load_model_and_check(checkpoint_path: str, device: str = "cuda"):
    """Load model and check weights."""
    print("=" * 60)
    print("STEP 1: Load checkpoint and check weights")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Check for EMA
    if "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            print(f"EMA shadow keys count: {len(ema_state['shadow'])}")
            # Check some weight stats
            for key in list(ema_state['shadow'].keys())[:3]:
                w = ema_state['shadow'][key]
                print(f"  {key}: shape={w.shape}, mean={w.mean():.4f}, std={w.std():.4f}")
        else:
            print("EMA state is direct dict (old format)")

    if "model" in checkpoint:
        print(f"Model keys count: {len(checkpoint['model'])}")

    # Load config
    config_dict = checkpoint.get("config", {})
    if config_dict:
        config = PixelHDMConfig.from_dict(config_dict)
        print(f"\nConfig: hidden_dim={config.hidden_dim}, pixel_dim={config.pixel_dim}")
        print(f"  patch_layers={config.patch_layers}, pixel_layers={config.pixel_layers}")
        print(f"  use_pooling_sync={config.use_pooling_sync}")
    else:
        config = PixelHDMConfig()
        print("Using default config (no config in checkpoint)")

    return config, checkpoint


def test_model_output(checkpoint_path: str, device: str = "cuda"):
    """Test model output during sampling."""
    print("\n" + "=" * 60)
    print("STEP 2: Test model output")
    print("=" * 60)

    config, checkpoint = load_model_and_check(checkpoint_path, device)

    # Create model
    from src.models.pixelhdm import create_pixelhdm_for_t2i
    model = create_pixelhdm_for_t2i(config=config)

    # Load weights (prefer EMA)
    if "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            missing, unexpected = model.load_state_dict(ema_state["shadow"], strict=False)
        else:
            missing, unexpected = model.load_state_dict(ema_state, strict=False)
        print(f"Loaded EMA weights: {len(missing)} missing, {len(unexpected)} unexpected")
    elif "model" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Loaded model weights: {len(missing)} missing, {len(unexpected)} unexpected")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Test forward pass at different timesteps
    B, H, W = 1, 512, 512
    z = torch.randn(B, 3, H, W, device=device, dtype=torch.bfloat16)

    print(f"\nInput noise: shape={z.shape}, mean={z.mean():.4f}, std={z.std():.4f}")

    timesteps = [0.05, 0.25, 0.5, 0.75, 0.95]

    print("\nModel output (velocity) at different timesteps:")
    print("-" * 60)

    for t_val in timesteps:
        t = torch.tensor([t_val], device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            v_pred = model(z, t)

        v_mean = v_pred.mean().item()
        v_std = v_pred.std().item()
        v_min = v_pred.min().item()
        v_max = v_pred.max().item()

        print(f"t={t_val:.2f}: mean={v_mean:+.4f}, std={v_std:.4f}, range=[{v_min:.4f}, {v_max:.4f}]")

    # Expected: v = x - noise, so for pure noise input, v should have similar magnitude to noise
    print(f"\nExpected v_std ~ sqrt(2) * noise_std ≈ {z.std().item() * 1.414:.4f}")
    print("If v_std << expected, model may not be learning correctly")

    return model, config


def test_sampling_step_by_step(model, device: str = "cuda"):
    """Test sampling step by step."""
    print("\n" + "=" * 60)
    print("STEP 3: Test sampling step by step")
    print("=" * 60)

    B, H, W = 1, 512, 512

    # Initial noise
    z = torch.randn(B, 3, H, W, device=device, dtype=torch.bfloat16)

    print(f"Initial z: mean={z.mean():.4f}, std={z.std():.4f}")

    sampler = create_sampler(method="euler", num_steps=10, t_eps=0.05)
    timesteps = sampler._sampler.get_timesteps(10, device, torch.bfloat16)

    print(f"\nTimesteps: {timesteps.tolist()}")
    print("\nSampling progression:")
    print("-" * 60)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t

        t_batch = t.expand(B)

        with torch.no_grad():
            v = model(z, t_batch)

        z_new = z + dt * v

        v_norm = v.norm().item()
        z_change = (z_new - z).norm().item()
        z_new_std = z_new.std().item()

        print(f"Step {i}: t={t.item():.4f}→{t_next.item():.4f}, "
              f"dt={dt.item():.4f}, |v|={v_norm:.2f}, "
              f"Δz={z_change:.4f}, z_std={z_new_std:.4f}")

        z = z_new

    print(f"\nFinal z: mean={z.mean():.4f}, std={z.std():.4f}")

    # Check if final output looks like noise or image
    if z.std() > 0.5:
        print("WARNING: Final output has high variance, likely still noise!")
    else:
        print("Final output has lower variance, might be denoised")

    return z


def check_pooling_sync_influence(model, device: str = "cuda"):
    """Check pooling sync influence."""
    print("\n" + "=" * 60)
    print("STEP 4: Check pooling sync influence")
    print("=" * 60)

    if not hasattr(model, 'pooling_sync') or model.pooling_sync is None:
        print("Pooling sync is disabled")
        return

    ps = model.pooling_sync
    gate_val = torch.sigmoid(ps.gate_param).item()
    print(f"Pooling sync gate: {gate_val:.4f}")
    print(f"Pooling sync rho: {ps.rho}")
    print(f"Effective influence: {gate_val * ps.rho:.4f} (~{gate_val * ps.rho * 100:.1f}% of signal RMS)")

    if gate_val > 0.9:
        print("WARNING: Gate is very high, pooling sync has strong influence!")

    # Test pooling sync directly
    B, H_p, W_p, D = 1, 32, 32, model.hidden_dim
    L = H_p * W_p
    x = torch.randn(B, L, D, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = ps(x, H_p, W_p, text_len=0)

    diff = (out - x).abs().mean().item()
    x_rms = x.pow(2).mean().sqrt().item()
    relative_diff = diff / x_rms

    print(f"\nPooling sync test:")
    print(f"  Input RMS: {x_rms:.4f}")
    print(f"  Output diff (abs mean): {diff:.6f}")
    print(f"  Relative diff: {relative_diff:.6f} (expected ~{gate_val * ps.rho:.6f})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("SAMPLING DEBUG DIAGNOSTIC")
    print("=" * 60)

    model, config = test_model_output(args.checkpoint, args.device)
    test_sampling_step_by_step(model, args.device)
    check_pooling_sync_influence(model, args.device)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
