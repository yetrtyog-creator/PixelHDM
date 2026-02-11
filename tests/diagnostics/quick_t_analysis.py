"""
Quick Timestep Difficulty Analysis

Lightweight analysis using synthetic data.
Shows model prediction difficulty at each timestep.
"""

import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def main():
    # Force CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (will be slow)")

    # checkpoint_completed.pt: 使用 3e-4 初始學習率訓練
    checkpoint_path = project_root / "checkpoints" / "checkpoint_completed.pt"

    print(f"\n1. Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    from src.config import PixelHDMConfig
    config = PixelHDMConfig(**(ckpt["config"] if isinstance(ckpt.get("config"), dict) else {}))

    print(f"2. Creating model (hidden_dim={config.hidden_dim})...")
    from src.models.pixelhdm import PixelHDM
    model = PixelHDM(config)

    # Handle different checkpoint formats (use strict=False for compatibility)
    if "ema" in ckpt and "shadow" in ckpt["ema"]:
        missing, unexpected = model.load_state_dict(ckpt["ema"]["shadow"], strict=False)
        print(f"   Loaded EMA shadow (missing: {len(missing)}, unexpected: {len(unexpected)})")
    elif "ema" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["ema"], strict=False)
        print(f"   Loaded EMA (missing: {len(missing)}, unexpected: {len(unexpected)})")
    elif "model" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"   Loaded model (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        raise KeyError(f"No model weights found. Keys: {list(ckpt.keys())}")

    model = model.to(device).eval()

    # Use smaller resolution for speed
    print(f"\n3. Analyzing at 512x512 resolution...")
    H, W = 512, 512

    # T values
    t_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    results = []
    results_normalized = []

    print("\n" + "=" * 85)
    print(f"{'t':>6} | {'V-Error':>10} | {'Var(v_t)':>10} | {'Norm Err':>10} | {'Rel%':>8} | Bar")
    print("-" * 85)

    with torch.no_grad():
        # Create synthetic "clean" image
        x_clean = torch.randn(1, 3, H, W, device=device) * 0.5

        for t_val in t_values:
            t = torch.tensor([t_val], device=device)

            errors = []
            variances = []
            for _ in range(3):  # 3 noise samples
                noise = torch.randn_like(x_clean)
                z_t = t_val * x_clean + (1 - t_val) * noise
                v_true = x_clean - noise

                v_pred = model(z_t, t)
                if v_pred.shape[-1] == 3:
                    v_pred = v_pred.permute(0, 3, 1, 2)

                # Raw MSE
                err = (v_pred - v_true).pow(2).mean().item()
                errors.append(err)

                # Variance of v_target
                var_v = v_true.var().item()
                variances.append(var_v)

            mean_err = np.mean(errors)
            mean_var = np.mean(variances)
            # Normalized error = MSE / Var(v_target)
            norm_err = mean_err / mean_var if mean_var > 0 else mean_err

            results.append((t_val, mean_err, mean_var))
            results_normalized.append((t_val, norm_err))

    # Display with both raw and normalized
    max_norm = max(r[1] for r in results_normalized)
    min_norm = min(r[1] for r in results_normalized)

    for (t_val, raw_err, var_v), (_, norm_err) in zip(results, results_normalized):
        rel = (norm_err - min_norm) / (max_norm - min_norm) * 100 if max_norm > min_norm else 0
        bar_len = int(rel / 2.5)
        bar = "|" * bar_len
        print(f"{t_val:>6.2f} | {raw_err:>10.4f} | {var_v:>10.4f} | {norm_err:>10.4f} | {rel:>7.1f}% | {bar}")

    # Find peak using NORMALIZED error (the fair comparison)
    peak_t, peak_norm = max(results_normalized, key=lambda x: x[1])
    min_t, min_norm_v = min(results_normalized, key=lambda x: x[1])

    # Also get raw values for reference
    peak_raw = next(r[1] for r in results if r[0] == peak_t)
    min_raw = next(r[1] for r in results if r[0] == min_t)

    print("-" * 85)
    print(f"Peak difficulty (NORMALIZED): t = {peak_t:.2f}")
    print(f"  - Normalized Error: {peak_norm:.4f}")
    print(f"  - Raw V-Error:      {peak_raw:.4f}")
    print(f"Easiest:              t = {min_t:.2f}")
    print(f"  - Normalized Error: {min_norm_v:.4f}")
    print(f"  - Raw V-Error:      {min_raw:.4f}")
    print(f"Ratio (normalized):   {peak_norm / min_norm_v:.2f}x harder")

    # Interval summary using NORMALIZED errors
    print("\n" + "=" * 85)
    print("INTERVAL SUMMARY (Using Normalized Error = MSE / Var(v_target))")
    print("=" * 85)

    intervals = [
        ([0.05, 0.10, 0.15, 0.20, 0.25], "High-freq [0.05-0.25]", 0.18),
        ([0.25, 0.30, 0.35, 0.40, 0.45, 0.50], "Mid-freq  [0.25-0.50]", 0.32),
        ([0.50, 0.55, 0.60, 0.65, 0.70, 0.75], "Mid-low   [0.50-0.75]", 0.32),
        ([0.75, 0.80, 0.85, 0.90, 0.95], "Low-freq  [0.75-0.95]", 0.18),
    ]

    print(f"{'Interval':<25} | {'Avg Raw':>10} | {'Avg Var':>10} | {'Avg Norm':>10} | {'Weight':>8}")
    print("-" * 85)

    interval_norms = []
    for t_range, name, weight in intervals:
        interval_raw = [r[1] for r in results if r[0] in t_range]
        interval_var = [r[2] for r in results if r[0] in t_range]
        interval_norm = [r[1] for r in results_normalized if r[0] in t_range]

        avg_raw = np.mean(interval_raw) if interval_raw else 0
        avg_var = np.mean(interval_var) if interval_var else 0
        avg_norm = np.mean(interval_norm) if interval_norm else 0

        interval_norms.append((name, avg_norm, weight))
        print(f"{name:<25} | {avg_raw:>10.4f} | {avg_var:>10.4f} | {avg_norm:>10.4f} | {weight:>7.0%}")

    # Identify bottleneck using normalized
    bottleneck_interval = max(interval_norms, key=lambda x: x[1])
    print("-" * 85)
    print(f"BOTTLENECK (by normalized error): {bottleneck_interval[0]} = {bottleneck_interval[1]:.4f}")

    # Statistical validation
    print("\n" + "=" * 85)
    print("STATISTICAL VALIDATION")
    print("=" * 85)

    # Check if Var(v_target) varies significantly across t
    all_vars = [r[2] for r in results]
    var_range = max(all_vars) - min(all_vars)
    var_mean = np.mean(all_vars)
    var_cv = np.std(all_vars) / var_mean * 100  # Coefficient of variation

    print(f"Var(v_target) statistics:")
    print(f"  - Min:  {min(all_vars):.4f}")
    print(f"  - Max:  {max(all_vars):.4f}")
    print(f"  - Mean: {var_mean:.4f}")
    print(f"  - CV:   {var_cv:.1f}%")

    if var_cv < 10:
        print(f"\n>> Var(v_target) is STABLE across t (CV < 10%)")
        print(f">> Raw V-Error comparison is VALID (no statistical trap)")
    else:
        print(f"\n>> Var(v_target) varies across t (CV = {var_cv:.1f}%)")
        print(f">> Use NORMALIZED error for fair comparison")

    print(f"\nConclusion: Peak difficulty at t = {peak_t:.2f} is {'CONFIRMED' if var_cv < 10 or peak_t == max(results_normalized, key=lambda x: x[1])[0] else 'NEEDS REVIEW'}")

    # Save plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        t_arr = [r[0] for r in results]
        err_arr = [r[1] for r in results]

        ax.plot(t_arr, err_arr, 'ro-', linewidth=2, markersize=8)
        ax.axvline(x=peak_t, color='darkred', linestyle='--', alpha=0.7,
                   label=f'Peak at t={peak_t:.2f}')
        ax.fill_between([0.25, 0.50], 0, max(err_arr) * 1.1, alpha=0.2, color='red',
                        label='Mid-freq bottleneck')

        ax.set_xlabel('Timestep t', fontsize=12)
        ax.set_ylabel('V-Prediction Error', fontsize=12)
        ax.set_title('Model Difficulty by Timestep (Epoch 10)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        output_path = project_root / "outputs" / "t_difficulty_epoch10.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Could not save plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
