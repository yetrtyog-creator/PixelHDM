"""Run training with runtime-only perf instrumentation.

This script keeps src/ untouched by installing monkeypatch hooks from tests/
and writing phase-0 compatible step_metrics.jsonl under logs/profiling/<run_id>.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, PixelHDMConfig, TrainingConfig
from src.models.pixelhdm import create_pixelhdm_for_t2i
from src.training.train_dataloader import create_dataloader_from_config
from src.training.train_utils import (
    find_latest_checkpoint,
    setup_file_logging,
    setup_seed,
    verify_environment,
)
from src.training.trainer import Trainer
from src.training.trainer.loop import TrainingLoop
from src.training.trainer.step import StepExecutor
from tests.tester.perf_bottleneck.runtime_instrumentation import (
    RuntimeHookSession,
    StepMetricsCollector,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training runner with runtime-only bottleneck instrumentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--profile-root", type=str, default="logs/profiling")
    parser.add_argument("--timing-mode", type=str, choices=["light", "probe"], default="light")
    parser.add_argument("--profile-stride", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=None)
    parser.add_argument("--log-every-epochs", type=int, default=None)
    parser.add_argument("--gc-interval", type=int, default=100)
    parser.add_argument("--pin-memory", type=parse_bool, default=None)
    parser.add_argument("--use-progress-bar", action="store_true")
    return parser.parse_args()


def _load_text_encoder(model_config: PixelHDMConfig, device: torch.device):
    if not model_config.text_encoder_name:
        return None
    logger.info("Loading text encoder: %s", model_config.text_encoder_name)
    try:
        from src.models.encoders.text_encoder import Qwen3TextEncoder

        encoder = Qwen3TextEncoder(
            model_name=model_config.text_encoder_name,
            max_length=model_config.text_max_length,
            freeze=model_config.text_encoder_frozen,
        )
        encoder = encoder.to(device)
        logger.info("Text encoder loaded on %s", device)
        return encoder
    except Exception as exc:  # pragma: no cover - depends on local environment.
        logger.warning("Failed to load text encoder: %s", exc)
        return None


def _setup_dino_encoder(model, model_config: PixelHDMConfig, trainer: Trainer, device: torch.device) -> None:
    if not model_config.repa_enabled or not hasattr(model, "dino_encoder"):
        return
    dino_encoder = model.dino_encoder
    if dino_encoder is None:
        logger.warning("DINO encoder unavailable; REPA path remains disabled.")
        return
    trainer.set_dino_encoder(dino_encoder.to(device))
    logger.info("DINO encoder attached to trainer.")


def _resolve_resume_path(
    explicit_resume: Optional[str],
    config: Config,
    checkpoint_dir: Path,
) -> Optional[Path]:
    if explicit_resume:
        return Path(explicit_resume)
    if config.resume.enabled and config.resume.checkpoint_path:
        if config.resume.checkpoint_path.lower() == "auto":
            return find_latest_checkpoint(checkpoint_dir)
        return Path(config.resume.checkpoint_path)
    auto_path = checkpoint_dir / "latest.pt"
    if auto_path.exists():
        return auto_path
    return None


def _handle_resume(trainer: Trainer, resume_path: Path, config: Config, training_config: TrainingConfig) -> None:
    load_optimizer = not config.resume.reset_optimizer
    load_scheduler = not config.resume.reset_scheduler
    trainer.load_checkpoint(
        resume_path,
        load_optimizer=load_optimizer,
        load_ema=True,
        load_scheduler=load_scheduler,
    )
    if config.resume.reset_scheduler:
        new_lr = training_config.learning_rate
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = new_lr
            if "initial_lr" in pg:
                pg["initial_lr"] = new_lr
        trainer._lr_scheduler = None
        trainer._scheduler_skip_sync = True
        logger.info("Scheduler reset to cycle 0 with LR=%s", f"{new_lr:.2e}")


def _resolve_train_limits(
    training_config: TrainingConfig,
    max_steps: Optional[int],
    num_epochs: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    if max_steps is not None:
        training_config.training_mode = "steps"
        training_config.max_steps = int(max_steps)
        return int(max_steps), None
    if num_epochs is not None:
        training_config.training_mode = "epochs"
        training_config.num_epochs = int(num_epochs)
        return None, int(num_epochs)
    if getattr(training_config, "training_mode", "epochs") == "steps":
        return int(training_config.max_steps), None
    return None, int(training_config.num_epochs)


def _write_run_metadata(
    metadata_path: Path,
    *,
    run_id: str,
    args: argparse.Namespace,
    config_path: str,
    checkpoint_dir: Path,
    step_metrics_path: Path,
) -> None:
    metadata = {
        "run_id": run_id,
        "config": config_path,
        "timing_mode": args.timing_mode,
        "profile_stride": int(args.profile_stride),
        "seed": int(args.seed),
        "device": args.device,
        "max_steps": args.max_steps,
        "num_epochs": args.num_epochs,
        "gc_interval": int(args.gc_interval),
        "pin_memory_override": args.pin_memory,
        "checkpoint_dir": str(checkpoint_dir),
        "step_metrics_path": str(step_metrics_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.skip_verify:
        verify_environment()
    setup_seed(args.seed)

    run_dir = Path(args.profile_root) / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    step_metrics_path = run_dir / "step_metrics.jsonl"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(log_dir, experiment_name=f"profile_{args.run_id}")
    _write_run_metadata(
        run_dir / "run_metadata.json",
        run_id=args.run_id,
        args=args,
        config_path=args.config,
        checkpoint_dir=checkpoint_dir,
        step_metrics_path=step_metrics_path,
    )

    logger.info("Loading config: %s", args.config)
    config = Config.from_yaml(args.config)
    model_config = config.model
    training_config = config.training
    data_config = config.data

    if args.pin_memory is not None:
        data_config.pin_memory = bool(args.pin_memory)
    if args.log_interval is not None:
        training_config.log_interval = int(args.log_interval)
    if args.save_interval is not None:
        training_config.save_interval = int(args.save_interval)
    if args.save_every_epochs is not None:
        training_config.save_every_epochs = int(args.save_every_epochs)
    if args.log_every_epochs is not None:
        training_config.log_every_epochs = int(args.log_every_epochs)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; fallback to CPU.")
        device = torch.device("cpu")

    logger.info("Creating model...")
    model = create_pixelhdm_for_t2i(config=model_config).to(device)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    logger.info("Creating dataloader...")
    train_dataloader = create_dataloader_from_config(data_config, model_config, training_config)
    logger.info("Training samples: %s", f"{len(train_dataloader.dataset):,}")

    text_encoder = _load_text_encoder(model_config, device)
    trainer = Trainer(
        model=model,
        config=model_config,
        training_config=training_config,
        dataloader=train_dataloader,
        device=device,
        text_encoder=text_encoder,
    )
    _setup_dino_encoder(model, model_config, trainer, device)

    resume_path = _resolve_resume_path(args.resume, config, checkpoint_dir)
    if resume_path is not None and resume_path.exists():
        logger.info("Resuming from %s", resume_path)
        _handle_resume(trainer, resume_path, config, training_config)

    num_steps, num_epochs = _resolve_train_limits(training_config, args.max_steps, args.num_epochs)

    collector = StepMetricsCollector(
        run_id=args.run_id,
        output_path=step_metrics_path,
        timing_mode=args.timing_mode,
        profile_stride=int(args.profile_stride),
    )

    def _callback(step: int, metrics) -> None:
        collector.finalize_step(
            step=int(step),
            metrics=metrics,
            epoch=int(trainer.state.epoch),
            batch_idx=int(trainer.state.batch_idx),
        )

    logger.info("Starting instrumented training run: %s", args.run_id)
    logger.info("Step metrics: %s", step_metrics_path)
    logger.info("Checkpoint dir: %s", checkpoint_dir)

    try:
        with RuntimeHookSession(
            collector=collector,
            loop_cls=TrainingLoop,
            step_executor_cls=StepExecutor,
        ):
            trainer.train(
                num_steps=num_steps,
                num_epochs=num_epochs,
                log_interval=training_config.log_interval,
                save_interval=training_config.save_interval,
                save_every_epochs=training_config.save_every_epochs,
                log_every_epochs=training_config.log_every_epochs,
                gc_interval=int(args.gc_interval),
                save_path=checkpoint_dir,
                callback=_callback,
                use_progress_bar=bool(args.use_progress_bar),
            )
    except KeyboardInterrupt:
        logger.info("Interrupted. Saving checkpoint before exit.")
        trainer.save_checkpoint(checkpoint_dir)
    finally:
        collector.close()
        if device.type == "cuda":
            import gc

            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    logger.info("Instrumented run completed: %s", args.run_id)


if __name__ == "__main__":
    main()

