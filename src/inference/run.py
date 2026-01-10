"""
PixelHDM-RPEA-DinoV3 - Inference Entry Point

Usage:
    # With trained checkpoint (auto-searches checkpoints/ if not specified)
    python -m src.inference.run --prompt "a beautiful sunset"
    python -m src.inference.run --checkpoint path/to/model.pt --prompt "a cat" --steps 50 --cfg 7.5

    # Test pipeline without trained weights (output will be noise)
    python -m src.inference.run --random-init --prompt "test"

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PixelHDMConfig
from src.models.pixelhdm import create_pixelhdm_for_t2i
from src.inference.pipeline import PixelHDMPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PixelHDM-RPEA-DinoV3 - Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (required unless --random-init is used)",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use randomly initialized model (for testing pipeline without trained weights)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None for random)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        choices=["euler", "heun", "dpm_pp"],
        help="Sampler method",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use model weights instead of EMA",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip environment verification",
    )
    parser.add_argument(
        "--mock-text-encoder",
        action="store_true",
        help="Use mock text encoder (for testing without Qwen3)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (bfloat16 requires Ampere+ GPU)",
    )

    return parser.parse_args()


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def find_checkpoint(checkpoint_path: str | None) -> Path | None:
    """Find checkpoint file, checking multiple locations if not specified"""
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        return None

    # 自動搜尋常見位置
    search_paths = [
        Path("checkpoints/latest.pt"),
        Path("checkpoints/best.pt"),
        Path("outputs/checkpoints/latest.pt"),
        Path("outputs/checkpoints/best.pt"),
    ]

    # 搜尋 checkpoints 目錄下最新的 .pt 文件
    for ckpt_dir in [Path("checkpoints"), Path("outputs/checkpoints")]:
        if ckpt_dir.exists():
            pt_files = list(ckpt_dir.glob("*.pt"))
            if pt_files:
                # 按修改時間排序，取最新的
                latest = max(pt_files, key=lambda p: p.stat().st_mtime)
                search_paths.insert(0, latest)

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load model from checkpoint"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint).__name__}")
    if "model" not in checkpoint and "ema" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing both 'model' and 'ema' keys")

    # Extract config if available
    config_dict = checkpoint.get("config", {})
    if config_dict:
        config = PixelHDMConfig.from_dict(config_dict)
    else:
        config = PixelHDMConfig()

    # Create model
    model = create_pixelhdm_for_t2i(config=config)

    # Load weights (strict=False 允許忽略訓練時的額外組件如 DINO projector)
    # 檢查可用的權重類型
    has_ema = "ema" in checkpoint
    has_model = "model" in checkpoint

    if use_ema and has_ema:
        logger.info("Using EMA weights")
        ema_state = checkpoint["ema"]
        # EMA state_dict 包含 {"shadow": {...}, "decay": ..., ...}
        # shadow 才是實際的模型權重
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            missing, unexpected = model.load_state_dict(ema_state["shadow"], strict=False)
        else:
            # 相容舊格式 (直接保存權重)
            missing, unexpected = model.load_state_dict(ema_state, strict=False)
        if unexpected:
            logger.debug(f"Ignored {len(unexpected)} unexpected keys (training-only components)")
    elif has_model:
        logger.info("Using Model weights (not EMA)")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected:
            logger.debug(f"Ignored {len(unexpected)} unexpected keys")
    elif has_ema:
        # use_ema=False 但只有 EMA 權重可用，回退使用 EMA
        logger.warning("Model weights not found in checkpoint, falling back to EMA weights")
        ema_state = checkpoint["ema"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            missing, unexpected = model.load_state_dict(ema_state["shadow"], strict=False)
        else:
            missing, unexpected = model.load_state_dict(ema_state, strict=False)
        if unexpected:
            logger.debug(f"Ignored {len(unexpected)} unexpected keys")
    else:
        # 最後嘗試：假設整個 checkpoint 就是 state_dict（舊格式）
        # 先檢查是否像是 state_dict 格式
        if any(key in checkpoint for key in ["optimizer", "state", "lr_scheduler"]):
            raise ValueError(
                "Checkpoint contains metadata keys but no 'model' or 'ema' weights. "
                "This checkpoint may be corrupted or incompatible."
            )
        logger.warning("Loading checkpoint as raw state_dict (legacy format)")
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if unexpected:
            logger.debug(f"Ignored {len(unexpected)} unexpected keys")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model, config


def create_random_model(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> tuple:
    """Create a randomly initialized model (for testing)"""
    logger.warning("=" * 60)
    logger.warning("Using RANDOMLY INITIALIZED model (no trained weights)")
    logger.warning("Output will be noise - this is for testing only!")
    logger.warning("=" * 60)

    config = PixelHDMConfig()
    model = create_pixelhdm_for_t2i(config=config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model, config


def main() -> None:
    """Main inference function"""
    args = parse_args()

    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Setup dtype
    dtype = _parse_dtype(args.dtype)
    logger.info(f"Using dtype: {dtype}")

    # Load model
    if args.random_init:
        # 使用隨機初始化模型（僅用於測試流程）
        model, config = create_random_model(device, dtype)
    else:
        # 尋找 checkpoint
        checkpoint_path = find_checkpoint(args.checkpoint)

        if checkpoint_path is None:
            logger.error("=" * 60)
            logger.error("No checkpoint found!")
            logger.error("")
            if args.checkpoint:
                logger.error(f"  Specified path does not exist: {args.checkpoint}")
            else:
                logger.error("  Searched locations:")
                logger.error("    - checkpoints/latest.pt")
                logger.error("    - checkpoints/best.pt")
                logger.error("    - outputs/checkpoints/latest.pt")
                logger.error("    - outputs/checkpoints/best.pt")
            logger.error("")
            logger.error("Options:")
            logger.error("  1. Train a model first:")
            logger.error("     python -m src.training.train --config configs/train_config.yaml")
            logger.error("")
            logger.error("  2. Specify a checkpoint path:")
            logger.error("     python -m src.inference.run --checkpoint /path/to/model.pt --prompt \"...\"")
            logger.error("")
            logger.error("  3. Test pipeline with random weights (output will be noise):")
            logger.error("     python -m src.inference.run --random-init --prompt \"...\"")
            logger.error("=" * 60)
            sys.exit(1)

        try:
            model, config = load_checkpoint(
                str(checkpoint_path),
                device,
                use_ema=not args.no_ema,
                dtype=dtype,
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)

    # Create pipeline
    logger.info("Creating pipeline...")
    pipeline = PixelHDMPipeline(model)

    # Use mock text encoder if requested
    if args.mock_text_encoder:
        pipeline.use_mock_text_encoder(hidden_size=config.hidden_dim)

    # Setup seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Enable deterministic mode for reproducible results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Using seed: {args.seed} (deterministic mode enabled)")

    # Generate
    logger.info("=" * 50)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Size: {args.width}x{args.height}")
    logger.info(f"Steps: {args.steps}, CFG: {args.cfg}")
    logger.info(f"Sampler: {args.sampler}")
    logger.info("=" * 50)

    logger.info("Generating...")
    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        sampler_method=args.sampler,
    )

    # Save output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"_s{args.seed}" if args.seed is not None else ""
    filename = f"pixelhdm_{timestamp}{seed_str}.png"
    output_path = output_dir / filename

    output.images[0].save(output_path)
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
