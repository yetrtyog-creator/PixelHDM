"""
Aspect Ratio Bucket Generator Module

Calculates all valid (width, height) combinations for training.
"""

from .core import AspectRatioBucket
from .generation import generate_buckets, ASPECT_RATIOS
from .matching import find_best_bucket

__all__ = [
    "AspectRatioBucket",
    "generate_buckets",
    "find_best_bucket",
    "ASPECT_RATIOS",
]
