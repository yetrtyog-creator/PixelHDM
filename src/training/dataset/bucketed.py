"""
PixelHDM-RPEA-DinoV3 Bucketed Dataset

支援分桶的圖像-文本數據集（多分辨率訓練）。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image

from .base import BaseImageTextDataset

if TYPE_CHECKING:
    from ..bucket import AspectRatioBucket

logger = logging.getLogger(__name__)


class BucketImageTextDataset(BaseImageTextDataset):
    """
    支援分桶的圖像-文本數據集

    每張圖片根據其長寬比被分配到最接近的桶，
    並在該桶的目標解析度下進行處理。

    包含 RAM 優化:
    - 路徑序列化為 torch.Tensor (避免多進程 copy-on-write)
    - bucket_ids 序列化為 torch.Tensor

    Args:
        image_paths: 圖片路徑列表
        bucket_ids: 對應的桶 ID 列表
        bucket_manager: AspectRatioBucket 實例
        caption_dropout: 標題丟棄率
        default_caption: 預設標題
        use_random_crop: 使用隨機裁切
        use_random_flip: 使用隨機水平翻轉
    """

    def __init__(
        self,
        image_paths: List[Path],
        bucket_ids: List[int],
        bucket_manager: "AspectRatioBucket",
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

        # RAM 優化: 將 Python 對象序列化為 torch.Tensor
        # 參考: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        self._serialize_paths(image_paths)
        self._serialize_bucket_ids(bucket_ids)

        self.bucket_manager = bucket_manager
        self._length = len(image_paths)

        if len(image_paths) != len(bucket_ids):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and bucket_ids ({len(bucket_ids)}) "
                "must have the same length"
            )

        logger.info(f"BucketImageTextDataset: {len(image_paths)} images (serialized)")

    def _serialize_paths(self, image_paths: List[Path]) -> None:
        """將 image_paths 序列化為 torch.Tensor"""
        serialized = [pickle.dumps(str(p)) for p in image_paths]
        self._path_addr = np.cumsum([len(x) for x in serialized])
        self._path_addr = torch.from_numpy(self._path_addr.astype(np.int64))
        self._path_data = torch.from_numpy(
            np.frombuffer(b''.join(serialized), dtype=np.uint8).copy()
        )

    def _serialize_bucket_ids(self, bucket_ids: List[int]) -> None:
        """將 bucket_ids 序列化為 torch.Tensor"""
        self._bucket_ids = torch.tensor(bucket_ids, dtype=torch.int32)

    def _get_path(self, idx: int) -> Path:
        """從序列化數據中獲取 image_path"""
        start = 0 if idx == 0 else int(self._path_addr[idx - 1])
        end = int(self._path_addr[idx])
        path_str = pickle.loads(bytes(self._path_data[start:end].numpy()))
        return Path(path_str)

    def _get_bucket_id(self, idx: int) -> int:
        """獲取 bucket_id"""
        return int(self._bucket_ids[idx])

    def _preprocess_image(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int,
    ) -> torch.Tensor:
        """
        根據目標桶解析度預處理圖片

        Args:
            image: PIL Image
            target_width: 桶的目標寬度
            target_height: 桶的目標高度

        Returns:
            Tensor of shape [3, H, W] with values in [-1, 1]
        """
        W, H = image.size

        # 計算縮放比例以覆蓋目標區域
        scale_w = target_width / W
        scale_h = target_height / H
        scale = max(scale_w, scale_h)

        # 縮放圖片
        if abs(scale - 1.0) > 1e-6:
            new_W = int(W * scale)
            new_H = int(H * scale)
            new_W = max(new_W, target_width)
            new_H = max(new_H, target_height)
            image = image.resize((new_W, new_H), Image.LANCZOS)
            W, H = new_W, new_H

        # 確保圖片尺寸足夠
        if W < target_width or H < target_height:
            image = image.resize(
                (max(W, target_width), max(H, target_height)),
                Image.LANCZOS
            )
            W, H = image.size

        # 裁切到目標尺寸
        if self.use_random_crop:
            left = random.randint(0, max(0, W - target_width))
            top = random.randint(0, max(0, H - target_height))
        else:
            left = (W - target_width) // 2
            top = (H - target_height) // 2

        left = max(0, min(left, max(0, W - target_width)))
        top = max(0, min(top, max(0, H - target_height)))

        image = image.crop((left, top, left + target_width, top + target_height))

        # 隨機水平翻轉
        image = self._apply_random_flip(image)

        # 轉換為 tensor 並正規化
        return self._image_to_tensor(image)

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int, _retry_count: int = 0
    ) -> Dict[str, Union[torch.Tensor, str, Path, int, Tuple[int, int]]]:
        """
        Get a single sample with bucket-specific resolution.

        Returns:
            Dict with keys:
                - "image": Tensor [3, H, W], values in [-1, 1]
                - "caption": str
                - "image_path": Path
                - "bucket_id": int
                - "resolution": Tuple[int, int] (width, height)
        """
        MAX_RETRIES = min(len(self), 100)
        if _retry_count >= MAX_RETRIES:
            raise RuntimeError(
                f"Failed to load valid image after {MAX_RETRIES} retries (starting from idx={idx}). "
                f"Possible causes: corrupted images, permission issues, or unsupported formats. "
                f"Check dataset directory and ensure images are valid."
            )

        image_path = self._get_path(idx)
        bucket_id = self._get_bucket_id(idx)
        target_width, target_height = self.bucket_manager.get_bucket_resolution(bucket_id)

        image = self._load_image(image_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        caption = self._load_caption(image_path)

        # Caption dropout
        caption = self._apply_caption_dropout(caption)

        image_tensor = self._preprocess_image(image, target_width, target_height)

        image.close()
        del image

        return {
            "image": image_tensor,
            "caption": caption,
            "image_path": image_path,
            "bucket_id": bucket_id,
            "resolution": (target_width, target_height),
        }


def bucket_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    分桶 DataLoader 的 collate 函數

    注意: 同一批次內的所有圖片應該有相同的解析度

    Raises:
        ValueError: 如果批次為空或圖片尺寸不一致
    """
    if not batch:
        raise ValueError("Empty batch received in bucket_collate_fn")

    # 驗證同一批次的解析度一致
    resolutions = [item["resolution"] for item in batch]
    if len(set(resolutions)) > 1:
        raise ValueError(
            f"Mixed resolutions in bucket batch: {set(resolutions)}. "
            f"BucketSampler should ensure all items in a batch have the same resolution."
        )

    # 驗證實際 tensor 形狀一致
    shapes = [item["image"].shape for item in batch]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"Inconsistent image shapes in bucket batch: {set(shapes)}. "
            f"This indicates a bug in _preprocess_image or bucket assignment."
        )

    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    bucket_ids = [item["bucket_id"] for item in batch]

    return {
        "images": images,
        "captions": captions,
        "image_paths": image_paths,
        "bucket_ids": bucket_ids,
        "resolution": resolutions[0],
    }


__all__ = [
    "BucketImageTextDataset",
    "bucket_collate_fn",
]
