"""
Quick diagnostic script to check LR schedule parameters.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.training.train_dataloader import create_dataloader_from_config


def main():
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    print(f"Loading config: {config_path}")

    config = Config.from_yaml(str(config_path))
    training_config = config.training
    data_config = config.data
    model_config = config.model

    print("\n" + "="*60)
    print("Training Config:")
    print("="*60)
    print(f"  restart_epochs: {training_config.restart_epochs}")
    print(f"  restart_period: {training_config.restart_period}")
    print(f"  gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"  num_epochs: {training_config.num_epochs}")

    # Stepped cosine restart config
    stepped = training_config.stepped_cosine_restart
    if stepped and stepped.enabled:
        print(f"\n  Stepped Cosine Restart (enabled):")
        print(f"    base_lr: {stepped.base_lr}")
        print(f"    cycle_min_lr: {stepped.cycle_min_lr}")
        print(f"    decay_rate: {stepped.decay_rate}")
        print(f"    global_min_lr: {stepped.global_min_lr}")

    print("\n" + "="*60)
    print("DataLoader Info:")
    print("="*60)

    try:
        dataloader = create_dataloader_from_config(data_config, model_config, training_config)
        steps_per_epoch = len(dataloader)
        dataset_size = len(dataloader.dataset)
        batch_size = training_config.batch_size

        print(f"  Dataset size: {dataset_size} images")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
    except Exception as e:
        print(f"  Error creating dataloader: {e}")
        # Use default values for calculation
        steps_per_epoch = 1000  # placeholder
        print(f"  Using placeholder steps_per_epoch: {steps_per_epoch}")

    print("\n" + "="*60)
    print("T_0 Calculation:")
    print("="*60)

    grad_accum = training_config.gradient_accumulation_steps
    restart_epochs = training_config.restart_epochs
    restart_period = training_config.restart_period

    if restart_period > 0:
        t_0 = restart_period
        print(f"  Using hardcoded restart_period: {t_0}")
    else:
        optimizer_steps_per_epoch = steps_per_epoch // grad_accum
        t_0 = optimizer_steps_per_epoch * restart_epochs

        print(f"  optimizer_steps_per_epoch = {steps_per_epoch} // {grad_accum} = {optimizer_steps_per_epoch}")
        print(f"  T_0 = {optimizer_steps_per_epoch} * {restart_epochs} = {t_0}")

    print(f"\n  >> Cycle length (T_0): {t_0} optimizer steps")

    # Calculate expected LR decay
    if stepped and stepped.enabled:
        print("\n" + "="*60)
        print("Expected LR Schedule:")
        print("="*60)
        base_lr = stepped.base_lr
        cycle_min = stepped.cycle_min_lr
        decay_rate = stepped.decay_rate
        global_min = stepped.global_min_lr

        for cycle in range(10):
            peak = max(global_min, base_lr * (decay_rate ** cycle))
            trough = max(global_min, cycle_min * (decay_rate ** cycle))
            if trough > peak:
                trough = peak
            steps_start = cycle * t_0
            steps_end = (cycle + 1) * t_0
            print(f"  Cycle {cycle}: steps {steps_start:,}-{steps_end:,}, LR {peak:.2e} -> {trough:.2e}")

            if peak == global_min:
                print(f"  (reached global_min_lr, subsequent cycles stay at {global_min:.2e})")
                break


if __name__ == "__main__":
    main()
