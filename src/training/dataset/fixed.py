"""
PixelHDM-RPEA-DinoV3 Fixed Resolution Dataset

固定解析度的圖像-文本數據集。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

from .base import BaseImageTextDataset, find_images

logger = logging.getLogger(__name__)


class ImageTextDataset(BaseImageTextDataset):
    """
    Image-Text Dataset for fixed resolution training.

    使用 patch_size=16 (與 DINOv3 匹配)

    Directory structure:
        train_dir/
            image1.png
            image1.txt        # Caption
            subfolder/
                image2.jpg
                image2.txt

    Args:
        root_dir: Root directory
        target_resolution: Target resolution (must be multiple of patch_size)
        patch_size: Patch size (default 16, matches DINOv3)
        max_resolution: Maximum resolution
        min_resolution: Minimum resolution
        caption_dropout: Caption dropout rate
        default_caption: Default caption when .txt not found
        use_random_crop: Use random crop
        use_random_flip: Use random horizontal flip
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        target_resolution: int = 256,
        patch_size: int = 16,  # 與 DINOv3 匹配
        max_resolution: int = 1024,
        min_resolution: int = 256,
        caption_dropout: float = 0.1,
        default_caption: str = "",
        use_random_crop: bool = True,
        use_random_flip: bool = True,
    ) -> None:
        super().__init__(
            caption_dropout=caption_dropout,
            default_caption=default_caption,
            use_random_crop=use_random_crop,
            use_random_flip=use_random_flip,
        )

        self.root_dir = Path(root_dir)
        self.target_resolution = target_resolution
        self.patch_size = patch_size
        self.max_resolution = max_resolution
        self.min_resolution = min_resolution

        # Validate target resolution
        if target_resolution % patch_size != 0:
            raise ValueError(
                f"target_resolution ({target_resolution}) must be "
                f"a multiple of patch_size ({patch_size})"
            )

        # Find all images
        self.image_paths = find_images(self.root_dir)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

        logger.info(f"Found {len(self.image_paths)} images in {self.root_dir}")

    def _preprocess_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        Preprocess image:
        1. Skip if too small
        2. Scale down if larger than max_resolution
        3. Crop to target_resolution
        4. Random horizontal flip
        5. Normalize to [-1, 1]

        Returns:
            Tensor [3, H, W] or None if too small
        """
        W, H = image.size

        # Skip if smaller than target_resolution
        if min(W, H) < self.target_resolution:
            return None

        # Scale down if larger than max_resolution
        if max(W, H) > self.max_resolution:
            scale = self.max_resolution / max(W, H)
            new_W = int(W * scale)
            new_H = int(H * scale)
            image = image.resize((new_W, new_H), Image.LANCZOS)
            W, H = new_W, new_H

            if min(W, H) < self.target_resolution:
                return None

        # Crop to target_resolution
        target = self.target_resolution

        if self.use_random_crop:
            left = random.randint(0, max(0, W - target))
            top = random.randint(0, max(0, H - target))
        else:
            left = (W - target) // 2
            top = (H - target) // 2

        image = image.crop((left, top, left + target, top + target))

        # Random horizontal flip
        image = self._apply_random_flip(image)

        # Convert to tensor and normalize
        return self._image_to_tensor(image)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Dict[str, Union[torch.Tensor, str, Path]]:
        """
        Get a single sample.

        Returns:
            Dict with:
                - "image": Tensor [3, H, W], values in [-1, 1]
                - "caption": str
                - "image_path": Path
        """
        MAX_RETRIES = min(len(self), 100)
        if _retry_count >= MAX_RETRIES:
            raise RuntimeError(f"Failed to load valid image after {MAX_RETRIES} retries")

        image_path = self.image_paths[idx]

        # Load image
        image = self._load_image(image_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        # Check minimum resolution
        if min(image.size) < self.min_resolution:
            image.close()
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        # Load caption
        caption = self._load_caption(image_path)

        # Caption dropout
        caption = self._apply_caption_dropout(caption)

        # Preprocess image
        image_tensor = self._preprocess_image(image)

        image.close()
        del image

        if image_tensor is None:
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        return {
            "image": image_tensor,
            "caption": caption,
            "image_path": image_path,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for DataLoader."""
    if not batch:
        raise ValueError("Empty batch")

    # Check shape consistency
    shapes = [item["image"].shape for item in batch]
    if len(set(shapes)) > 1:
        raise ValueError(f"Inconsistent image shapes: {set(shapes)}")

    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "images": images,
        "captions": captions,
        "image_paths": image_paths,
    }


__all__ = [
    "ImageTextDataset",
    "collate_fn",
]
