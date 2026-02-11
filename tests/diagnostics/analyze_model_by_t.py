"""
Analyze Model Prediction Quality by Timestep

Analyzes how well the model predicts velocity at different t values.
Does NOT require DINOv3 download - uses prediction consistency instead.

Key insight: At t close to 1, z_t ≈ x_clean, so v_pred should be stable.
At t close to 0.5, z_t is ambiguous, so v_pred variance is higher.

Usage:
    python tests/diagnostics/analyze_model_by_t.py
"""

import sys
from pathlib import Path
import re

import torch
import numpy as np

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

_EPOCH_STEP_PATTERN = re.compile(r"checkpoint_epoch(\d+)_step(\d+)\.pt$")


def _parse_epoch_step(path_like) -> tuple[int, int] | None:
    """Parse checkpoint_epoch{E}_step{S}.pt into (E, S)."""
    name = str(path_like).replace("\\", "/").rsplit("/", 1)[-1]
    match = _EPOCH_STEP_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _pick_latest_epoch_step(paths) -> tuple[Path | None, int]:
    """Pick latest checkpoint by numeric (epoch, step), ignoring invalid names."""
    ignored = 0
    candidates: list[tuple[tuple[int, int], Path]] = []
    for path_like in paths:
        parsed = _parse_epoch_step(path_like)
        if parsed is None:
            ignored += 1
            continue
        normalized = str(path_like).replace("\\", "/")
        candidates.append((parsed, Path(normalized)))
    if not candidates:
        return None, ignored
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1], ignored


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    # Use glob to find latest epoch checkpoint (new format: checkpoint_epoch{N}_step{S}.pt)
    import glob as _glob
    epoch_matches = _glob.glob(str(project_root / "checkpoints" / "checkpoint_epoch*_step*.pt"))
    latest_ckpt, ignored = _pick_latest_epoch_step(epoch_matches)
    if ignored > 0:
        print(f"Ignored {ignored} invalid checkpoint filename(s).")
    checkpoint_path = latest_ckpt or (project_root / "checkpoints" / "checkpoint_epoch_10.pt")
    data_dir = project_root / "data" / "train"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    from src.config import PixelHDMConfig

    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        if isinstance(config_dict, dict):
            config = PixelHDMConfig(**config_dict)
        else:
            config = config_dict
    else:
        config = PixelHDMConfig()

    print(f"Config: hidden_dim={config.hidden_dim}")

    # Create model
    from src.models.pixelhdm import PixelHDM

    print("Creating model...")
    model = PixelHDM(config)

    # Load weights (prefer EMA)
    if "ema_state_dict" in checkpoint:
        print("Loading EMA weights...")
        model.load_state_dict(checkpoint["ema_state_dict"])
    elif "model_state_dict" in checkpoint:
        print("Loading model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    # Load test images
    print(f"\nLooking for images in: {data_dir}")
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Find images
    image_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.webp"]:
        image_files.extend(list(data_dir.glob(f"**/{ext}")))

    if not image_files:
        print("No images found. Using synthetic test images.")
        # Create synthetic test images (gradients, patterns)
        test_images = []
        for i in range(5):
            # Random but structured images
            img = torch.randn(1, 3, 512, 512) * 0.5
            test_images.append(img.to(device))
    else:
        print(f"Found {len(image_files)} images, using first 10")
        test_images = []
        for img_path in image_files[:10]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                test_images.append(img_tensor)
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

    if not test_images:
        print("No test images available!")
        return

    print(f"Using {len(test_images)} test images")

    # T values to test
    t_values = torch.linspace(0.05, 0.95, 19)

    print(f"\nAnalyzing at {len(t_values)} timesteps...")
    print("=" * 70)

    # Metrics to collect at each t:
    # 1. V-prediction error (how close v_pred is to true v = x - noise)
    # 2. Feature variance (how stable the intermediate features are)

    results = {t.item(): {"v_errors": [], "feature_vars": []} for t in t_values}

    with torch.no_grad():
        for img_idx, x_clean in enumerate(test_images):
            for t_val in t_values:
                t = torch.tensor([t_val], device=device)

                # Create multiple noisy versions to test consistency
                errors = []
                for _ in range(3):  # 3 noise samples per t
                    noise = torch.randn_like(x_clean)

                    # z_t = t*x + (1-t)*noise
                    z_t = t_val * x_clean + (1 - t_val) * noise

                    # True velocity: v = x - noise (for flow matching)
                    v_true = x_clean - noise

                    # Model prediction
                    v_pred = model(z_t, t)

                    # Ensure same format (B, C, H, W)
                    if v_pred.shape[-1] == 3:  # (B, H, W, 3) -> (B, 3, H, W)
                        v_pred = v_pred.permute(0, 3, 1, 2)

                    # V-prediction error
                    v_error = (v_pred - v_true).pow(2).mean().item()
                    errors.append(v_error)

                results[t_val.item()]["v_errors"].append(np.mean(errors))

            if (img_idx + 1) % 2 == 0:
                print(f"  Image {img_idx + 1}/{len(test_images)} done")

    # Compute statistics
    print("\n" + "=" * 70)
    print("V-Prediction Error by Timestep (Lower = Better)")
    print("=" * 70)
    print(f"{'t':>6} | {'V-Error Mean':>12} | {'Std':>8} | {'Relative':>10} | {'Bar'}")
    print("-" * 70)

    t_list = []
    error_list = []
    std_list = []

    for t_val in t_values:
        errors = results[t_val.item()]["v_errors"]
        if errors:
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            t_list.append(t_val.item())
            error_list.append(mean_err)
            std_list.append(std_err)

    # Normalize for display
    max_err = max(error_list) if error_list else 1
    min_err = min(error_list) if error_list else 0

    for t_val, mean_err, std_err in zip(t_list, error_list, std_list):
        rel = (mean_err - min_err) / (max_err - min_err) if max_err > min_err else 0
        bar_len = int(rel * 40)
        bar = "█" * bar_len
        print(f"{t_val:>6.2f} | {mean_err:>12.4f} | {std_err:>8.4f} | {rel:>10.2%} | {bar}")

    # Find peak difficulty
    peak_idx = np.argmax(error_list)
    peak_t = t_list[peak_idx]
    peak_err = error_list[peak_idx]

    min_idx = np.argmin(error_list)
    min_t = t_list[min_idx]
    min_err_val = error_list[min_idx]

    print("-" * 70)
    print(f"Hardest timestep: t = {peak_t:.2f} (error = {peak_err:.4f})")
    print(f"Easiest timestep: t = {min_t:.2f} (error = {min_err_val:.4f})")
    print(f"Difficulty ratio: {peak_err / min_err_val:.2f}x")

    # Interval analysis
    print("\n" + "=" * 70)
    print("Interval Analysis (V-Prediction Error)")
    print("=" * 70)

    intervals = [
        (0.05, 0.25, "High-freq  [0.05-0.25]"),
        (0.25, 0.50, "Mid-freq   [0.25-0.50]"),
        (0.50, 0.75, "Mid-low    [0.50-0.75]"),
        (0.75, 0.95, "Low-freq   [0.75-0.95]"),
    ]

    for t_low, t_high, name in intervals:
        interval_errors = [e for t, e in zip(t_list, error_list) if t_low <= t <= t_high]
        if interval_errors:
            interval_mean = np.mean(interval_errors)
            rel = (interval_mean - min_err_val) / (max_err - min_err_val) if max_err > min_err_val else 0
            bar_len = int(rel * 40)
            bar = "█" * bar_len
            print(f"{name}: Error = {interval_mean:.4f} ({rel:.0%} relative) {bar}")

    # Map to estimated REPA
    print("\n" + "=" * 70)
    print("Estimated REPA Contribution (based on V-Error correlation)")
    print("=" * 70)
    print("""
Assumption: V-Error and REPA are correlated
- High V-Error → Model struggles → High REPA contribution
- Low V-Error → Model confident → Low REPA contribution
""")

    # Logit-normal weights
    weights = [
        (0.05, 0.25, 0.18, "High-freq"),
        (0.25, 0.50, 0.32, "Mid-freq"),
        (0.50, 0.75, 0.32, "Mid-low"),
        (0.75, 0.95, 0.18, "Low-freq"),
    ]

    total_weighted = 0
    for t_low, t_high, weight, name in weights:
        interval_errors = [e for t, e in zip(t_list, error_list) if t_low <= t <= t_high]
        if interval_errors:
            interval_mean = np.mean(interval_errors)
            # Normalize to REPA-like scale (rough approximation)
            repa_est = interval_mean / max_err * 0.4  # Scale to ~0-0.4 range
            contribution = weight * repa_est
            total_weighted += contribution
            print(f"{name:.<25} weight={weight:.0%}, est_REPA={repa_est:.3f}, contribution={contribution:.4f}")

    print(f"\nEstimated total REPA (rough): {total_weighted:.3f}")
    print(f"Your actual REPA: ~0.25")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: V-Error vs t
        ax1 = axes[0]
        ax1.fill_between(t_list, np.array(error_list) - np.array(std_list),
                         np.array(error_list) + np.array(std_list), alpha=0.3, color='red')
        ax1.plot(t_list, error_list, 'r-', linewidth=2, marker='o', markersize=4)
        ax1.axvline(x=peak_t, color='darkred', linestyle='--', label=f'Peak at t={peak_t:.2f}')
        ax1.set_xlabel('Timestep t', fontsize=12)
        ax1.set_ylabel('V-Prediction Error (MSE)', fontsize=12)
        ax1.set_title('Model Difficulty by Timestep (Epoch 10)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)

        # Plot 2: Compare with Logit-normal sampling
        ax2 = axes[1]

        # Logit-normal PDF
        t_dense = np.linspace(0.01, 0.99, 200)
        logit_t = np.log(t_dense / (1 - t_dense))
        pdf = np.exp(-0.5 * logit_t ** 2) / (t_dense * (1 - t_dense) * np.sqrt(2 * np.pi))

        ax2.fill_between(t_dense, pdf, alpha=0.3, color='green')
        ax2.plot(t_dense, pdf, 'g-', linewidth=2, label='Sampling Density')
        ax2.set_ylabel('Sampling Density', color='green', fontsize=12)

        # Overlay error (scaled)
        ax2_twin = ax2.twinx()
        error_scaled = np.array(error_list) / max(error_list) * max(pdf)
        ax2_twin.plot(t_list, error_scaled, 'r-', linewidth=2, marker='o', markersize=4, label='V-Error (scaled)')
        ax2_twin.set_ylabel('V-Error (scaled)', color='red', fontsize=12)

        ax2.set_xlabel('Timestep t', fontsize=12)
        ax2.set_title('Sampling Density vs Difficulty', fontsize=14)
        ax2.set_xlim(0, 1)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()

        output_path = project_root / "outputs" / "model_difficulty_by_t_epoch10.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")

        plt.show()

    except ImportError:
        print("\nMatplotlib not available, skipping plot")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print("""
Key Insight:
- V-Error peaks around t=0.3-0.5 (mid-frequency region)
- This correlates with REPA difficulty
- Once model learns this region, both V-Error and REPA should drop
""")


if __name__ == "__main__":
    main()
