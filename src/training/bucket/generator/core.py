"""
Aspect Ratio Bucket Generator (Core)

Calculates all valid (width, height) combinations for training.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from .generation import generate_buckets
from .matching import find_best_bucket

logger = logging.getLogger(__name__)


class AspectRatioBucket:
    """
    Aspect Ratio Bucket Manager

    Pre-computes all valid (width, height) combinations,
    then assigns images to the closest bucket based on aspect ratio.

    Args:
        min_resolution: Minimum resolution (min side length)
        max_resolution: Maximum resolution (max side length)
        patch_size: Patch size (16, matches DINOv3)
        max_aspect_ratio: Maximum aspect ratio
        target_pixels: Target pixel count
        bucket_max_resolution: Maximum resolution constraint for buckets
        follow_max_resolution: Whether to follow max resolution constraint
    """

    def __init__(
        self,
        min_resolution: int = 256,
        max_resolution: int = 1024,
        patch_size: int = 16,
        max_aspect_ratio: float = 2.0,
        target_pixels: Optional[int] = None,
        bucket_max_resolution: Optional[int] = None,
        follow_max_resolution: bool = True,
    ) -> None:
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.patch_size = patch_size
        self.max_aspect_ratio = max_aspect_ratio
        self.target_pixels = target_pixels if target_pixels is not None else (512 * 512)

        self.bucket_max_resolution = bucket_max_resolution if bucket_max_resolution is not None else max_resolution
        self.follow_max_resolution = follow_max_resolution

        if follow_max_resolution:
            self.effective_max_resolution = min(max_resolution, self.bucket_max_resolution)
        else:
            self.effective_max_resolution = max_resolution

        logger.info(
            f"AspectRatioBucket: min={min_resolution}, max={max_resolution}, "
            f"bucket_max={self.bucket_max_resolution}, follow={follow_max_resolution}, "
            f"effective_max={self.effective_max_resolution}"
        )

        self.buckets = generate_buckets(
            min_resolution=min_resolution,
            effective_max_resolution=self.effective_max_resolution,
            patch_size=patch_size,
            max_aspect_ratio=max_aspect_ratio,
        )

        if not self.buckets:
            raise ValueError(
                f"No valid buckets generated with min_resolution={min_resolution}, "
                f"max_resolution={max_resolution}, patch_size={patch_size}. "
                f"Check your configuration - min_resolution may be too large."
            )

        self.bucket_to_id = {bucket: i for i, bucket in enumerate(self.buckets)}

        logger.info(f"Generated {len(self.buckets)} aspect ratio buckets")
        logger.debug(f"Buckets: {self.buckets}")

    def get_bucket_id(self, width: int, height: int) -> int:
        """
        Find the closest bucket ID for given image dimensions.

        Args:
            width: Image width
            height: Image height

        Returns:
            bucket_id: Bucket index
        """
        return find_best_bucket(
            width=width,
            height=height,
            buckets=self.buckets,
            min_resolution=self.min_resolution,
        )

    def get_bucket_resolution(self, bucket_id: int) -> Tuple[int, int]:
        """Get resolution (width, height) for specified bucket."""
        return self.buckets[bucket_id]

    def __len__(self) -> int:
        return len(self.buckets)


__all__ = ["AspectRatioBucket"]
