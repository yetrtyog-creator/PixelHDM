"""
Training tests configuration.

Tests in this directory are organized by speed:
- Fast: TrainerState, TrainMetrics, Trainer initialization
- Slow: train_step (runs actual forward/backward), train loop, checkpointing

This conftest marks specific test classes that involve actual training operations
as 'slow' for CI optimization.
"""

import pytest


# Test class patterns that involve actual training operations
SLOW_TEST_CLASSES = {
    "TestTrainStep",
    "TestTrainLoop",
    "TestCheckpointing",
    "TestEdgeCases",
    "TestOOMRecovery",
}


def pytest_collection_modifyitems(items):
    """Mark slow test classes in training/."""
    for item in items:
        # Check if the test is in the training directory
        if "tests/training" not in str(item.fspath) and "tests\\training" not in str(item.fspath):
            continue

        # Check if the test belongs to a slow class
        if hasattr(item, "cls") and item.cls is not None:
            class_name = item.cls.__name__
            if class_name in SLOW_TEST_CLASSES:
                item.add_marker(pytest.mark.slow)
