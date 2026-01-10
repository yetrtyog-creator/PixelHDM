"""
Patch Embedding Modules

Contains:
    - PatchEmbedding: Image -> Patch Tokens (with Bottleneck)
    - PixelUnpatchify: Patch-Level -> Pixel-Level features
    - PixelPatchify: Pixel-Level features -> Image space
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class PixelEmbedding(nn.Module):
    """
    1×1 Patchify: Per-pixel linear embedding

    This is the "1×1 Patchify" from the PixelHDM paper.
    Each pixel is treated as a "patch" and independently projected.

    Flow: (B, H, W, C) -> Linear -> (B, H, W, D_pix) -> reshape -> (B, L, p², D_pix)

    This provides a direct path from input image to Pixel-Level processing,
    preserving high-frequency details that might be lost in 16×16 patch embedding.
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        in_channels: int = 3,
        pixel_dim: int = 16,
        patch_size: int = 16,
    ) -> None:
        super().__init__()

        if config is not None:
            in_channels = config.in_channels
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size

        self.in_channels = in_channels
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.p2 = patch_size ** 2

        # Per-pixel linear projection
        self.proj = nn.Linear(in_channels, pixel_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, H, W, C) or (B, C, H, W)

        Returns:
            Pixel tokens (B, L, p², D_pix) aligned with patch structure
        """
        x = self._normalize_input(x)
        B, H, W, C = x.shape

        self._validate_dimensions(H, W)

        # Per-pixel linear projection
        x = self.proj(x)  # (B, H, W, pixel_dim)

        # Reshape to align with patch structure
        x = self._reshape_to_patches(x, B, H, W)

        return x

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input should be 4D, got: {x.dim()}")

        if x.shape[-1] == self.in_channels:
            return x
        return x.permute(0, 2, 3, 1)

    def _validate_dimensions(self, H: int, W: int) -> None:
        assert H % self.patch_size == 0, f"H={H} not divisible by patch_size={self.patch_size}"
        assert W % self.patch_size == 0, f"W={W} not divisible by patch_size={self.patch_size}"

    def _reshape_to_patches(self, x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """Reshape (B, H, W, D) to (B, L, p², D) aligned with patch structure."""
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        L = h_patches * w_patches

        # (B, H, W, D) -> (B, h_p, p, w_p, p, D)
        x = x.reshape(B, h_patches, p, w_patches, p, self.pixel_dim)
        # -> (B, h_p, w_p, p, p, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        # -> (B, L, p², D)
        x = x.reshape(B, L, self.p2, self.pixel_dim)

        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, pixel_dim={self.pixel_dim}, patch_size={self.patch_size}"


class PatchEmbedding(nn.Module):
    """
    Patch Embedding with Bottleneck Design (16×16 Patchify)

    Flow: (B, H, W, 3) -> unfold -> (B, L, p^2, 3) -> Linear -> (B, L, D)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        in_channels: int = 3,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()

        if config is not None:
            patch_size = config.patch_size
            hidden_dim = config.hidden_dim
            in_channels = config.in_channels
            bottleneck_dim = config.bottleneck_dim

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.p2 = patch_size ** 2

        input_dim = in_channels * self.p2
        self.proj_down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.act = nn.SiLU()
        self.proj_up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        B, H, W, C = x.shape

        self._validate_dimensions(H, W)

        x = self._unfold_patches(x, B, H, W)
        x = self._apply_projection(x)

        return x

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input should be 4D, got: {x.dim()}")

        if x.shape[-1] == self.in_channels:
            return x
        return x.permute(0, 2, 3, 1)

    def _validate_dimensions(self, H: int, W: int) -> None:
        assert H % self.patch_size == 0, f"H={H} not divisible by patch_size={self.patch_size}"
        assert W % self.patch_size == 0, f"W={W} not divisible by patch_size={self.patch_size}"

    def _unfold_patches(self, x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        L = h_patches * w_patches

        x = x.reshape(B, h_patches, p, w_patches, p, self.in_channels)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, L, self.p2 * self.in_channels)

        return x

    def _apply_projection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_down(x)
        x = self.act(x)
        x = self.proj_up(x)
        return x

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, hidden_dim={self.hidden_dim}"


class PixelUnpatchify(nn.Module):
    """
    Pixel Unpatchify: Patch-Level features -> Pixel-Level features

    Flow: (B, L, D) -> Linear -> (B, L, p^2 x D_pix) -> reshape -> (B, L, p^2, D_pix)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        pixel_dim: int = 16,
        patch_size: int = 16,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size

        self.hidden_dim = hidden_dim
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.p2 = patch_size ** 2

        output_dim = self.p2 * pixel_dim
        self.proj = nn.Linear(hidden_dim, output_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = self.proj(x)
        x = x.reshape(B, L, self.p2, self.pixel_dim)
        return x

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, pixel_dim={self.pixel_dim}, p^2={self.p2}"


class PixelPatchify(nn.Module):
    """
    Pixel Patchify (Output Projection): Pixel-Level features -> Image space

    Flow: (B, L, p^2, D_pix) -> Linear -> (B, L, p^2, C) -> reshape -> (B, H, W, C)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        pixel_dim: int = 16,
        patch_size: int = 16,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        if config is not None:
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size
            out_channels = config.out_channels

        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.p2 = patch_size ** 2

        self.proj = nn.Linear(pixel_dim, out_channels, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Use Xavier init for proper output range.

        CRITICAL FIX (2026-01-06):
        原本使用 trunc_normal_(std=0.02) 導致輸出範圍過小：
        - 輸出 std ≈ 0.08，目標 v_target std ≈ 1.15
        - 覆蓋率僅 7%，模型無法生成正確範圍的 velocity
        - 導致 V-Loss 卡在 0.73 無法下降

        修復：使用 Xavier 初始化，輸出 std ≈ 1.30，覆蓋率 113%
        """
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        B, L, P2, D = x.shape
        p = self.patch_size

        h_patches, w_patches, H, W = self._compute_dimensions(L, p, image_size)
        assert L == h_patches * w_patches, f"L={L} != h x w={h_patches} x {w_patches}"

        x = self.proj(x)
        x = self._reshape_to_image(x, B, h_patches, w_patches, p, H, W)

        return x

    def _compute_dimensions(
        self, L: int, p: int, image_size: Optional[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        if image_size is None:
            h_patches = w_patches = int(math.sqrt(L))
            H = h_patches * p
            W = w_patches * p
        else:
            H, W = image_size
            h_patches = H // p
            w_patches = W // p

        return h_patches, w_patches, H, W

    def _reshape_to_image(
        self, x: torch.Tensor, B: int, h_patches: int, w_patches: int, p: int, H: int, W: int
    ) -> torch.Tensor:
        x = x.reshape(B, h_patches, w_patches, p, p, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H, W, self.out_channels)
        return x

    def extra_repr(self) -> str:
        return f"pixel_dim={self.pixel_dim}, out_channels={self.out_channels}"


# === Factory Functions ===

def create_pixel_embedding(
    in_channels: int = 3,
    pixel_dim: int = 16,
    patch_size: int = 16,
) -> PixelEmbedding:
    """Create PixelEmbedding (1×1 Patchify) with explicit parameters."""
    return PixelEmbedding(
        config=None,
        in_channels=in_channels,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
    )


def create_pixel_embedding_from_config(config: "PixelHDMConfig") -> PixelEmbedding:
    """Create PixelEmbedding from config."""
    return PixelEmbedding(config=config)


def create_patch_embedding(
    patch_size: int = 16,
    hidden_dim: int = 1024,
    in_channels: int = 3,
    bottleneck_dim: int = 256,
) -> PatchEmbedding:
    """Create PatchEmbedding with explicit parameters."""
    return PatchEmbedding(
        config=None,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        in_channels=in_channels,
        bottleneck_dim=bottleneck_dim,
    )


def create_patch_embedding_from_config(config: "PixelHDMConfig") -> PatchEmbedding:
    """Create PatchEmbedding from config."""
    return PatchEmbedding(config=config)


def create_pixel_unpatchify(
    hidden_dim: int = 1024,
    pixel_dim: int = 16,
    patch_size: int = 16,
) -> PixelUnpatchify:
    """Create PixelUnpatchify with explicit parameters."""
    return PixelUnpatchify(
        config=None,
        hidden_dim=hidden_dim,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
    )


def create_pixel_unpatchify_from_config(config: "PixelHDMConfig") -> PixelUnpatchify:
    """Create PixelUnpatchify from config."""
    return PixelUnpatchify(config=config)


def create_pixel_patchify(
    pixel_dim: int = 16,
    patch_size: int = 16,
    out_channels: int = 3,
) -> PixelPatchify:
    """Create PixelPatchify with explicit parameters."""
    return PixelPatchify(
        config=None,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
        out_channels=out_channels,
    )


def create_pixel_patchify_from_config(config: "PixelHDMConfig") -> PixelPatchify:
    """Create PixelPatchify from config."""
    return PixelPatchify(config=config)


__all__ = [
    "PixelEmbedding",
    "PatchEmbedding",
    "PixelUnpatchify",
    "PixelPatchify",
    "create_pixel_embedding",
    "create_pixel_embedding_from_config",
    "create_patch_embedding",
    "create_patch_embedding_from_config",
    "create_pixel_unpatchify",
    "create_pixel_unpatchify_from_config",
    "create_pixel_patchify",
    "create_pixel_patchify_from_config",
]
