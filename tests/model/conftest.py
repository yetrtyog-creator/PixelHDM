"""
Model tests configuration.

Tests in this directory create real PixelHDM models (not mocks).
While the testing config uses minimal dimensions (hidden_dim=256, patch_layers=2),
the model creation and forward passes are still slower than mock-based tests.

These tests are marked as 'slow' for CI optimization.
Run with: pytest -m slow tests/model/
Or run all tests: pytest -m ""
"""

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in model/ as slow."""
    for item in items:
        # Check if the test is in the model directory
        if "tests/model" in str(item.fspath) or "tests\\model" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
