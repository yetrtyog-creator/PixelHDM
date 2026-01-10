"""
Block tests configuration.

Tests in this directory create real PatchTransformerBlock and PixelTransformerBlock
components. While using minimal configs, these tests involve actual transformer
operations (attention, MLP) which are slower than mock-based tests.

These tests are marked as 'slow' for CI optimization.
Run with: pytest -m slow tests/blocks/
Or run all tests: pytest -m ""
"""

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in blocks/ as slow."""
    for item in items:
        # Check if the test is in the blocks directory
        if "tests/blocks" in str(item.fspath) or "tests\\blocks" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
