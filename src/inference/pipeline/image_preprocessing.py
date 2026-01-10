"""
Image Preprocessing Utilities

Handles image preprocessing for I2I pipeline.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def preprocess_image(
    image: Image.Image,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Preprocess PIL image for I2I pipeline.

    Args:
        image: Input PIL image
        patch_size: Patch size for alignment

    Returns:
        Tensor (1, H, W, 3) in range [-1, 1]
    """
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Align to patch_size
    w, h = image.size
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size

    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Convert to tensor
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array * 2.0 - 1.0  # Scale to [-1, 1]
    tensor = torch.from_numpy(img_array)

    return tensor.unsqueeze(0)  # (1, H, W, 3)


__all__ = ["preprocess_image"]
