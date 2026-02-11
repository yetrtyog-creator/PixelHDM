"""
Shared training constants to avoid package import cycles.
"""

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"
}

__all__ = ["IMAGE_EXTENSIONS"]
