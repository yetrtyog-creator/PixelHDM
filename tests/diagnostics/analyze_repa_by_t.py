"""
Analyze REPA Loss by Timestep - Direct Checkpoint Analysis

Directly loads checkpoint and analyzes REPA loss at different t values.
No training code modification needed.

Usage:
    python tests/diagnostics/analyze_repa_by_t.py
"""

import sys
from pathlib import Path
import re

import torch
import torch.nn.functional as F
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

    print(f"Config: hidden_dim={config.hidden_dim}, repa_align_layer={config.repa_align_layer}")

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

    # Create DINO encoder
    print("Loading DINOv3 encoder...")
    from src.models.encoders.dinov3 import DINOv3Encoder

    dino_encoder = DINOv3Encoder().to(device)
    dino_encoder.eval()

    # Get REPA projector
    from src.training.losses.repa import REPALoss

    repa_loss_fn = REPALoss(config=config, dino_encoder=dino_encoder).to(device)

    if "repa_state_dict" in checkpoint:
        print("Loading REPA projector weights...")
        repa_loss_fn.load_state_dict(checkpoint["repa_state_dict"])
    repa_loss_fn.eval()

    # Load a few test images
    print(f"\nLoading test images from: {data_dir}")
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Find images
    image_files = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png"))
    if not image_files:
        print("No images found. Using random tensor for demo.")
        test_images = [torch.randn(1, 3, 512, 512).to(device) for _ in range(5)]
    else:
        print(f"Found {len(image_files)} images, using first 10")
        test_images = []
        for img_path in image_files[:10]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                test_images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    if not test_images:
        print("No images loaded!")
        return

    # T values to test
    t_values = torch.linspace(0.05, 0.95, 19)

    print(f"\nAnalyzing REPA at {len(t_values)} timesteps with {len(test_images)} images...")
    print("=" * 70)

    all_repa_by_t = {t.item(): [] for t in t_values}

    with torch.no_grad():
        for img_idx, x_clean in enumerate(test_images):
            # Get DINO features (clean image, independent of t)
            dino_features = dino_encoder(x_clean)  # (1, L, 768)

            # Spatial normalize (iREPA)
            dino_features = dino_features - dino_features.mean(dim=1, keepdim=True)
            dino_features = dino_features / (dino_features.std(dim=1, keepdim=True) + 1e-6)

            for t_val in t_values:
                # Create noisy input: z_t = t*x + (1-t)*noise
                t = torch.tensor([t_val], device=device)
                noise = torch.randn_like(x_clean)
                z_t = t_val * x_clean + (1 - t_val) * noise

                # Get model features at this t
                _, h_t = model(z_t, t, return_features=True)  # h_t: (1, L, D)

                if h_t is None:
                    print(f"Warning: No REPA features returned for t={t_val:.2f}")
                    continue

                # Project to DINO space
                B, L, D = h_t.shape
                H = W = int(L ** 0.5)
                h_2d = h_t.permute(0, 2, 1).reshape(B, D, H, W)
                h_proj = repa_loss_fn.projector(h_2d)
                h_proj = h_proj.flatten(2).permute(0, 2, 1)  # (1, L, 768)

                # Cosine similarity
                h_norm = F.normalize(h_proj, dim=-1)
                y_norm = F.normalize(dino_features, dim=-1)
                cos_sim = (h_norm * y_norm).sum(dim=-1).mean()

                repa_loss = (1.0 - cos_sim).item()
                all_repa_by_t[t_val.item()].append(repa_loss)

            print(f"  Image {img_idx + 1}/{len(test_images)} done")

    # Compute statistics
    t_list = []
    mean_list = []
    std_list = []

    print("\n" + "=" * 70)
    print("REPA Loss by Timestep")
    print("=" * 70)
    print(f"{'t':>6} | {'REPA Mean':>10} | {'Std':>8} | {'Bar':>30}")
    print("-" * 70)

    for t_val in t_values:
        losses = all_repa_by_t[t_val.item()]
        if losses:
            mean_val = np.mean(losses)
            std_val = np.std(losses)
            t_list.append(t_val.item())
            mean_list.append(mean_val)
            std_list.append(std_val)

            # Visual bar
            bar_len = int(mean_val * 50)
            bar = "█" * bar_len

            print(f"{t_val.item():>6.2f} | {mean_val:>10.4f} | {std_val:>8.4f} | {bar}")

    # Find peak
    if mean_list:
        peak_idx = np.argmax(mean_list)
        peak_t = t_list[peak_idx]
        peak_repa = mean_list[peak_idx]

        print("-" * 70)
        print(f"Peak difficulty: t = {peak_t:.2f}, REPA = {peak_repa:.4f}")

    # Interval analysis
    print("\n" + "=" * 70)
    print("Interval Analysis")
    print("=" * 70)

    intervals = [
        (0.05, 0.25, "High-freq  [0.05-0.25]"),
        (0.25, 0.50, "Mid-freq   [0.25-0.50]"),
        (0.50, 0.75, "Mid-low    [0.50-0.75]"),
        (0.75, 0.95, "Low-freq   [0.75-0.95]"),
    ]

    interval_means = []
    for t_low, t_high, name in intervals:
        interval_losses = []
        for t_val, mean_val in zip(t_list, mean_list):
            if t_low <= t_val <= t_high:
                interval_losses.append(mean_val)
        if interval_losses:
            interval_mean = np.mean(interval_losses)
            interval_means.append((name, interval_mean))
            # Bar
            bar_len = int(interval_mean * 50)
            bar = "█" * bar_len
            print(f"{name}: REPA = {interval_mean:.4f}  {bar}")

    # Contribution analysis
    print("\n" + "=" * 70)
    print("Contribution to Total REPA (weighted by Logit-Normal)")
    print("=" * 70)

    # Logit-normal weights (approximate)
    weights = {
        (0.05, 0.25): 0.18,
        (0.25, 0.50): 0.32,
        (0.50, 0.75): 0.32,
        (0.75, 0.95): 0.18,
    }

    total_contribution = 0
    for (t_low, t_high), weight in weights.items():
        for name, mean_val in interval_means:
            if f"[{t_low:.2f}-{t_high:.2f}]" in name:
                contrib = weight * mean_val
                total_contribution += contrib
                pct = (contrib / 0.25) * 100 if 0.25 > 0 else 0  # Assuming total ~0.25
                print(f"{name}: {weight:.0%} weight × {mean_val:.4f} = {contrib:.4f} ({pct:.1f}% of total)")

    print(f"\nEstimated total REPA: {total_contribution:.4f}")
    print(f"Actual total REPA (from training): ~0.25")

    # Save plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: REPA vs t
        ax1 = axes[0]
        ax1.fill_between(t_list, np.array(mean_list) - np.array(std_list),
                         np.array(mean_list) + np.array(std_list), alpha=0.3, color='blue')
        ax1.plot(t_list, mean_list, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axvline(x=peak_t, color='r', linestyle='--', label=f'Peak at t={peak_t:.2f}')
        ax1.set_xlabel('Timestep t', fontsize=12)
        ax1.set_ylabel('REPA Loss', fontsize=12)
        ax1.set_title('REPA Loss vs Timestep (Epoch 10)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)

        # Plot 2: Sampling density overlay
        ax2 = axes[1]
        t_dense = np.linspace(0.01, 0.99, 200)
        logit_t = np.log(t_dense / (1 - t_dense))
        pdf = np.exp(-0.5 * logit_t ** 2) / (t_dense * (1 - t_dense) * np.sqrt(2 * np.pi))

        ax2.fill_between(t_dense, pdf, alpha=0.3, color='green')
        ax2.plot(t_dense, pdf, 'g-', linewidth=2, label='Sampling Density')

        # Overlay REPA
        ax2_twin = ax2.twinx()
        ax2_twin.plot(t_list, mean_list, 'b-', linewidth=2, marker='o', markersize=4, label='REPA Loss')
        ax2_twin.set_ylabel('REPA Loss', color='blue', fontsize=12)

        ax2.set_xlabel('Timestep t', fontsize=12)
        ax2.set_ylabel('Sampling Density', color='green', fontsize=12)
        ax2.set_title('Sampling Density vs REPA Difficulty', fontsize=14)
        ax2.set_xlim(0, 1)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()

        output_path = project_root / "outputs" / "repa_by_timestep_epoch10.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")

        plt.show()

    except ImportError:
        print("\nMatplotlib not available, skipping plot")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
