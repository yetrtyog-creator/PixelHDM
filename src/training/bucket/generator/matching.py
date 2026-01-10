"""
Bucket Matching Logic

Finds the best bucket for a given image.
"""

import logging
import math
from typing import List, Tuple

logger = logging.getLogger(__name__)


def find_best_bucket(
    width: int,
    height: int,
    buckets: List[Tuple[int, int]],
    min_resolution: int,
) -> int:
    """
    Find the closest bucket ID for given image dimensions.

    Strategy:
    1. Strictly prohibit assigning buckets larger than original image pixels (avoid upscaling)
    2. Prioritize aspect ratio matching
    3. Among similar aspect ratios, choose the closest resolution

    Args:
        width: Image width
        height: Image height
        buckets: List of (width, height) tuples
        min_resolution: Minimum resolution

    Returns:
        bucket_id: Bucket index
    """
    if width <= 0 or height <= 0:
        logger.error(
            f"Invalid image dimensions: {width}x{height}. "
            f"Expected positive values. Using default bucket (id=0, "
            f"resolution={buckets[0][0]}x{buckets[0][1]})"
        )
        return 0

    aspect = width / height
    aspect = max(1e-6, min(aspect, 1e6))

    image_pixels = width * height
    min_score = float('inf')
    best_bucket_id = 0
    fallback_bucket_id = 0
    fallback_score = float('inf')

    min_bucket_pixels = buckets[0][0] * buckets[0][1]

    for i, (bw, bh) in enumerate(buckets):
        bucket_aspect = bw / bh
        bucket_pixels = bw * bh

        aspect_diff = abs(math.log(aspect) - math.log(bucket_aspect))

        # Track best bucket at minimum pixel level as fallback
        if bucket_pixels == min_bucket_pixels:
            if aspect_diff < fallback_score:
                fallback_score = aspect_diff
                fallback_bucket_id = i

        # Strictly prohibit larger buckets
        if bucket_pixels > image_pixels:
            continue

        resolution_diff = (image_pixels - bucket_pixels) / image_pixels

        # Combined score: aspect ratio weight 0.7, resolution weight 0.3
        score = aspect_diff * 0.7 + resolution_diff * 0.3

        if score < min_score:
            min_score = score
            best_bucket_id = i

    # Use fallback if no suitable bucket found (image too small)
    if min_score == float('inf'):
        fallback_w, fallback_h = buckets[fallback_bucket_id]
        logger.warning(
            f"Image {width}x{height} ({width*height} pixels) is smaller than all buckets. "
            f"Minimum bucket size is {min_resolution}x{min_resolution}. "
            f"Using fallback bucket (id={fallback_bucket_id}, resolution={fallback_w}x{fallback_h}). "
            f"Consider filtering images with min_resolution >= {min_resolution}."
        )
        return fallback_bucket_id

    return best_bucket_id


__all__ = ["find_best_bucket"]
