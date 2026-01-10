"""
Aspect Ratio Bucket Generator (Legacy Compatibility)

This module re-exports from the new modular structure.
For new code, import directly from training.bucket.generator.
"""

from .generator import AspectRatioBucket

__all__ = ["AspectRatioBucket"]
