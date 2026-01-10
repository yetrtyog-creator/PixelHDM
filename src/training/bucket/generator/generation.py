"""
Bucket Generation Logic

Generates all valid (width, height) bucket combinations.
"""

from typing import List, Set, Tuple

# Common aspect ratios (width/height)
ASPECT_RATIOS = [
    1.0,      # 1:1 square
    4/3,      # 4:3 landscape
    3/4,      # 3:4 portrait
    3/2,      # 3:2 landscape
    2/3,      # 2:3 portrait
    16/9,     # 16:9 landscape
    9/16,     # 9:16 portrait
    2/1,      # 2:1 ultra-wide
    1/2,      # 1:2 ultra-tall
]


def _compute_base_resolutions(
    min_resolution: int,
    max_resolution: int,
    patch_size: int,
) -> List[int]:
    """Compute base resolutions (multiples of patch_size)."""
    base_resolutions = []
    step = patch_size * 4

    res = min_resolution
    while res <= max_resolution:
        aligned_res = (res // patch_size) * patch_size
        if aligned_res >= min_resolution:
            base_resolutions.append(aligned_res)
        res += step

    max_aligned = (max_resolution // patch_size) * patch_size
    if max_aligned not in base_resolutions and max_aligned >= min_resolution:
        base_resolutions.append(max_aligned)

    return sorted(set(base_resolutions))


def _generate_pixel_based_buckets(
    base_resolutions: List[int],
    min_resolution: int,
    max_resolution: int,
    patch_size: int,
    max_aspect_ratio: float,
) -> Set[Tuple[int, int]]:
    """Generate buckets based on pixel count."""
    buckets = set()
    EPSILON = 1e-9

    for base_res in base_resolutions:
        if base_res < min_resolution or base_res > max_resolution:
            continue

        base_pixels = base_res * base_res

        for aspect in ASPECT_RATIOS:
            if aspect > max_aspect_ratio or aspect < 1.0 / max_aspect_ratio:
                continue

            width = int((base_pixels * aspect) ** 0.5)
            height = int((base_pixels / aspect) ** 0.5)

            width = (width // patch_size) * patch_size
            height = (height // patch_size) * patch_size

            if width < min_resolution or width > max_resolution:
                continue
            if height < min_resolution or height > max_resolution:
                continue

            actual_aspect = width / height
            if actual_aspect > max_aspect_ratio + EPSILON:
                continue
            if actual_aspect < 1.0 / max_aspect_ratio - EPSILON:
                continue

            buckets.add((width, height))

    return buckets


def _generate_edge_based_buckets(
    base_resolutions: List[int],
    min_resolution: int,
    max_resolution: int,
    patch_size: int,
    max_aspect_ratio: float,
) -> Set[Tuple[int, int]]:
    """Generate high-resolution non-square buckets based on max edge."""
    buckets = set()
    EPSILON = 1e-9

    high_res_edges = [r for r in base_resolutions if r >= max_resolution // 2]
    if not high_res_edges:
        high_res_edges = base_resolutions[-3:] if len(base_resolutions) >= 3 else base_resolutions

    for max_edge in high_res_edges:
        if max_edge > max_resolution:
            continue

        for aspect in ASPECT_RATIOS:
            if aspect > max_aspect_ratio or aspect < 1.0 / max_aspect_ratio:
                continue
            if aspect == 1.0:
                continue

            if aspect > 1.0:
                width = max_edge
                height = int(max_edge / aspect)
            else:
                height = max_edge
                width = int(max_edge * aspect)

            width = (width // patch_size) * patch_size
            height = (height // patch_size) * patch_size

            if width < min_resolution or height < min_resolution:
                continue

            actual_aspect = width / height
            if actual_aspect > max_aspect_ratio + EPSILON:
                continue
            if actual_aspect < 1.0 / max_aspect_ratio - EPSILON:
                continue

            buckets.add((width, height))

    return buckets


def generate_buckets(
    min_resolution: int,
    effective_max_resolution: int,
    patch_size: int,
    max_aspect_ratio: float,
) -> List[Tuple[int, int]]:
    """
    Generate all valid (width, height) buckets.

    Strategy:
    1. Generate buckets at multiple pixel levels with various aspect ratios
    2. Ensure all dimensions are multiples of patch_size
    3. Ensure dimensions are in [min_resolution, effective_max_resolution]
    4. Additionally generate high-resolution non-square buckets
    """
    base_resolutions = _compute_base_resolutions(
        min_resolution, effective_max_resolution, patch_size
    )

    buckets = set()

    # Method 1: Pixel-based generation
    buckets.update(_generate_pixel_based_buckets(
        base_resolutions, min_resolution, effective_max_resolution,
        patch_size, max_aspect_ratio
    ))

    # Method 2: Edge-based generation for high-res
    buckets.update(_generate_edge_based_buckets(
        base_resolutions, min_resolution, effective_max_resolution,
        patch_size, max_aspect_ratio
    ))

    # Ensure standard square buckets exist
    for res in base_resolutions:
        if min_resolution <= res <= effective_max_resolution:
            buckets.add((res, res))

    # Sort: by total pixels, then by width
    return sorted(buckets, key=lambda x: (x[0] * x[1], x[0]))


__all__ = ["generate_buckets", "ASPECT_RATIOS"]
