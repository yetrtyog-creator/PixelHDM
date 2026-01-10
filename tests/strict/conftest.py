"""
Strict tests configuration.

All tests in this directory are marked as 'slow' by default.
Run with: pytest -m slow tests/strict/
Or run all tests: pytest -m ""
"""

import pytest

# Mark all tests in this directory as slow
def pytest_collection_modifyitems(items):
    """Automatically mark all tests in strict/ as slow."""
    for item in items:
        if "strict" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
