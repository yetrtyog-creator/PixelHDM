"""
PixelHDM-RPEA-DinoV3 Dataset Base

共享的數據集基類和工具函數。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"
}


class BaseImageTextDataset(Dataset, ABC):
    """
    圖像-文本數據集基類

    提供共享的圖像加載和標題處理功能。

    Args:
        caption_dropout: Caption dropout rate for CFG training
        default_caption: Default caption when .txt not found
        use_random_crop: Use random crop
        use_random_flip: Use random horizontal flip
    """

    def __init__(
        self,
        caption_dropout: float = 0.1,
        default_caption: str = "",
        use_random_crop: bool = True,
        use_random_flip: bool = True,
    ) -> None:
        self.caption_dropout = caption_dropout
        self.default_caption = default_caption
        self.use_random_crop = use_random_crop
        self.use_random_flip = use_random_flip

    def _load_caption(self, image_path: Path) -> str:
        """
        Load caption from corresponding .txt file.

        Args:
            image_path: Path to the image file

        Returns:
            Caption string or default_caption if not found
        """
        caption_path = image_path.with_suffix(".txt")

        if caption_path.exists():
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                return caption if caption else self.default_caption
            except Exception as e:
                logger.warning(f"Failed to read caption {caption_path}: {e}")
                return self.default_caption
        else:
            return self.default_caption

    def _load_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Load and validate image.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image in RGB mode or None if failed

        Raises:
            PermissionError: If access is denied (should not be silently ignored)
            IOError: If there's a critical I/O error
        """
        try:
            with Image.open(image_path) as img:
                img.load()
                image = img.copy()

            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except PermissionError as e:
            # 權限錯誤應該重新拋出，不應該被靜默處理
            logger.error(f"Permission denied for image {image_path}: {e}")
            raise
        except OSError as e:
            # 區分嚴重的 I/O 錯誤 (如磁盤空間不足) 和普通的圖像損壞
            if "No space left" in str(e) or "disk" in str(e).lower():
                logger.error(f"Critical I/O error for image {image_path}: {e}")
                raise
            # 圖像損壞或格式不支持 - 記錄警告並返回 None
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
        except Exception as e:
            # 其他錯誤 (如圖像損壞) - 記錄警告並返回 None
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _apply_random_flip(self, image: Image.Image) -> Image.Image:
        """Apply random horizontal flip if enabled."""
        if self.use_random_flip and random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to normalized tensor.

        Args:
            image: PIL Image

        Returns:
            Tensor of shape [3, H, W] with values in [-1, 1]
        """
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array * 2.0 - 1.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return tensor

    def _apply_caption_dropout(self, caption: str) -> str:
        """Apply caption dropout for CFG training."""
        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            return ""
        return caption

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        pass


def find_images(root_dir: Path) -> List[Path]:
    """
    Recursively find all images in a directory.

    Args:
        root_dir: Root directory to search

    Returns:
        Sorted list of image paths
    """
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(root_dir.rglob(f"*{ext}"))
        images.extend(root_dir.rglob(f"*{ext.upper()}"))

    images = sorted(set(images))
    return images


__all__ = [
    "IMAGE_EXTENSIONS",
    "BaseImageTextDataset",
    "find_images",
]
