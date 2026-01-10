"""
PixelHDM-RPEA-DinoV3 - Training Entry Point

Usage:
    python -m src.training.train --config configs/train_config.yaml
    python -m src.training.train --resume checkpoints/step_10000.pt

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PixelHDMConfig, TrainingConfig, DataConfig, Config
from src.models.pixelhdm import create_pixelhdm_for_t2i
from src.training.trainer import Trainer
from src.training.train_utils import find_latest_checkpoint, setup_seed, verify_environment, setup_file_logging
from src.training.train_dataloader import create_dataloader_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PixelHDM Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Training config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-verify", action="store_true", help="Skip environment verification")
    return parser.parse_args()


def _load_text_encoder(model_config: PixelHDMConfig, device: torch.device):
    """Load text encoder if configured.

    Args:
        model_config: Model configuration containing text encoder settings
        device: Device to load the encoder on

    Note:
        text_encoder_frozen 參數從配置中讀取並傳遞給編碼器。
        默認值為 True（凍結），這是推薦的設置。
    """
    if not model_config.text_encoder_name:
        return None
    logger.info(f"Loading text encoder: {model_config.text_encoder_name}")
    logger.info(f"  frozen={model_config.text_encoder_frozen}")
    try:
        from src.models.encoders.text_encoder import Qwen3TextEncoder
        encoder = Qwen3TextEncoder(
            model_name=model_config.text_encoder_name,
            max_length=model_config.text_max_length,
            freeze=model_config.text_encoder_frozen,  # 從配置讀取凍結設置
        )
        encoder = encoder.to(device)
        logger.info(f"Text encoder loaded successfully (device: {device})")
        return encoder
    except Exception as e:
        logger.warning(f"Failed to load text encoder: {e}")
        return None


def _setup_dino_encoder(model, model_config: PixelHDMConfig, trainer: Trainer, device: torch.device) -> None:
    """Setup DINOv3 encoder for REPA Loss."""
    if not model_config.repa_enabled or not hasattr(model, 'dino_encoder'):
        return
    logger.info("Loading DINOv3 encoder for REPA Loss...")
    dino_encoder = model.dino_encoder
    if dino_encoder is not None:
        dino_encoder = dino_encoder.to(device)
        trainer.set_dino_encoder(dino_encoder)
        logger.info("DINOv3 encoder set for REPA Loss")
    else:
        logger.warning("DINOv3 encoder not available, REPA Loss disabled")


def _resolve_resume_path(args, config: Config, output_dir: Path) -> Optional[Path]:
    """Resolve checkpoint path for resume."""
    if args.resume:
        return Path(args.resume)
    if config.resume.enabled and config.resume.checkpoint_path:
        ckpt_path = config.resume.checkpoint_path
        if ckpt_path.lower() == "auto":
            path = find_latest_checkpoint(output_dir)
            if path:
                logger.info(f"Auto-detected latest checkpoint: {path}")
            return path
        return Path(ckpt_path)
    auto_path = output_dir / "latest.pt"
    if auto_path.exists():
        logger.info(f"Found checkpoint, auto-resuming: {auto_path}")
        return auto_path
    return None


def _handle_resume(trainer: Trainer, resume_path: Path, config: Config, training_config: TrainingConfig) -> None:
    """Handle checkpoint resume and LR reset.

    Logic:
        - reset_optimizer=True, reset_scheduler=False:
            Optimizer state (momentum) reset, but scheduler state preserved.
            LR follows scheduler's current cycle (e.g., cycle 8 -> LR ~4.3e-5).
        - reset_optimizer=True, reset_scheduler=True:
            Both reset. Scheduler rebuilt, LR reset to base_lr.
        - reset_optimizer=False, reset_scheduler=False:
            Full resume, both states preserved.
    """
    load_optimizer = not config.resume.reset_optimizer
    load_scheduler = not config.resume.reset_scheduler
    trainer.load_checkpoint(resume_path, load_optimizer=load_optimizer, load_ema=True, load_scheduler=load_scheduler)

    if config.resume.reset_scheduler:
        # Scheduler will be rebuilt - reset LR to base and clear scheduler
        # CRITICAL: Set skip_sync flag to prevent syncing to current step
        # This ensures scheduler starts fresh from cycle 0
        new_lr = training_config.learning_rate
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = new_lr
            if 'initial_lr' in pg:
                pg['initial_lr'] = new_lr
        trainer._lr_scheduler = None
        trainer._scheduler_skip_sync = True  # Skip sync, start from cycle 0
        logger.info(f"LR scheduler will be rebuilt from cycle 0 (restart_epochs={training_config.restart_epochs}), LR reset to: {new_lr:.2e}")
    else:
        # Scheduler state preserved - LR follows scheduler's current cycle
        current_lr = trainer.optimizer.param_groups[0]['lr']
        if hasattr(trainer, '_lr_scheduler') and trainer._lr_scheduler is not None:
            # Log scheduler state for debugging
            sched = trainer._lr_scheduler
            if hasattr(sched, 'cycle'):
                logger.info(f"Scheduler restored: cycle={sched.cycle}, LR={current_lr:.2e}")
            else:
                logger.info(f"Scheduler restored, LR={current_lr:.2e}")
        else:
            logger.info(f"Resumed with LR={current_lr:.2e}")


def _log_config_summary(model_config: PixelHDMConfig, training_config: TrainingConfig, text_encoder, training_desc: str) -> None:
    """Log configuration summary."""
    logger.info("=" * 50)
    logger.info("Config Summary:")
    logger.info(f"  Model: hidden_dim={model_config.hidden_dim}, patch_layers={model_config.patch_layers}")
    logger.info(f"  GQA: Q={model_config.num_heads}, KV={model_config.num_kv_heads}")
    logger.info(f"  Flow Matching: P_mean={model_config.time_p_mean}, P_std={model_config.time_p_std}")
    logger.info(f"  Text Encoder: {'enabled' if text_encoder else 'disabled'}")
    logger.info(f"  REPA: {'enabled' if model_config.repa_enabled else 'disabled'}")
    logger.info(f"  Training: lr={training_config.learning_rate}, {training_desc}")
    logger.info(f"  Gradient: accumulation={training_config.gradient_accumulation_steps}, max_norm={training_config.max_grad_norm}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info("=" * 50)


def _get_training_params(training_config: TrainingConfig) -> Tuple[Optional[int], Optional[int], str]:
    """Get training parameters (num_epochs, num_steps, description)."""
    training_mode = getattr(training_config, 'training_mode', 'epochs')
    if training_mode == "epochs":
        return training_config.num_epochs, None, f"epochs={training_config.num_epochs}"
    return None, training_config.max_steps, f"steps={training_config.max_steps}"


def main() -> None:
    """Main training function."""
    args = parse_args()

    if not args.skip_verify:
        verify_environment()

    setup_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Load config
    logger.info(f"Loading config: {args.config}")
    config = Config.from_yaml(args.config)
    model_config, training_config, data_config = config.model, config.training, config.data
    logger.info(f"Using data_dir: {data_config.data_dir}")

    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device = torch.device("cpu")

    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(log_dir, experiment_name="pixelhdm_train")

    # Create model
    logger.info("Creating model...")
    model = create_pixelhdm_for_t2i(config=model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Create DataLoader
    logger.info("Creating DataLoader...")
    try:
        train_dataloader = create_dataloader_from_config(data_config, model_config, training_config)
        logger.info(f"Training set: {len(train_dataloader.dataset)} images")
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        sys.exit(1)

    # Create text encoder and trainer
    text_encoder = _load_text_encoder(model_config, device)
    trainer = Trainer(model=model, config=model_config, training_config=training_config,
                      dataloader=train_dataloader, device=device, text_encoder=text_encoder)
    _setup_dino_encoder(model, model_config, trainer, device)

    # Resume checkpoint
    resume_path = _resolve_resume_path(args, config, output_dir)
    if resume_path and resume_path.exists():
        logger.info(f"Resuming training: {resume_path}")
        _handle_resume(trainer, resume_path, config, training_config)
    elif resume_path:
        logger.warning(f"Checkpoint not found: {resume_path}")

    # Training parameters
    num_epochs, num_steps, training_desc = _get_training_params(training_config)
    _log_config_summary(model_config, training_config, text_encoder, training_desc)

    # Start training
    logger.info(f"Starting training... (save every {training_config.save_interval} steps)")
    try:
        trainer.train(num_steps=num_steps, num_epochs=num_epochs, log_interval=training_config.log_interval,
                      save_interval=training_config.save_interval, save_path=output_dir)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        trainer.save_checkpoint(output_dir)
    except Exception as e:
        logger.error(f"Training error: {e}")
        trainer.save_checkpoint(output_dir)
        raise
    finally:
        # Cleanup GPU resources
        if device.type == "cuda":
            import gc
            if 'trainer' in locals():  # Check if trainer was created
                del trainer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU resources cleaned up")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
