"""
Pipeline Input Validation

Handles resolution validation and input parameter checking.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, List, Union

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig

logger = logging.getLogger(__name__)


# === Default Constants (fallbacks when config unavailable) ===
DEFAULT_PATCH_SIZE = 16
DEFAULT_MAX_TOKENS = 1024


class InputValidator:
    """
    Validates pipeline inputs including resolution and parameters.

    Attributes:
        patch_size: Patch size for token calculation
        max_tokens: Maximum allowed token count
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        patch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize validator with config or explicit parameters.

        Args:
            config: PixelHDMConfig instance (takes precedence)
            patch_size: Override patch size
            max_tokens: Override max tokens
        """
        if config is not None:
            self.patch_size = config.patch_size
            self.max_tokens = config.max_patches
        else:
            self.patch_size = patch_size or DEFAULT_PATCH_SIZE
            self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    def validate_resolution(self, height: int, width: int) -> None:
        """
        Validate image resolution.

        Checks:
        1. Dimensions are divisible by patch_size
        2. Total token count does not exceed max_tokens

        Args:
            height: Image height in pixels
            width: Image width in pixels

        Raises:
            ValueError: If validation fails
        """
        self._check_divisibility(height, width)
        self._check_token_limit(height, width)

    def _check_divisibility(self, height: int, width: int) -> None:
        """Check if dimensions are divisible by patch_size."""
        if height % self.patch_size != 0:
            suggested = (height // self.patch_size) * self.patch_size
            raise ValueError(
                f"圖像高度 {height} 必須是 patch_size ({self.patch_size}) 的倍數。"
                f"建議使用: {suggested} 或 {suggested + self.patch_size}"
            )

        if width % self.patch_size != 0:
            suggested = (width // self.patch_size) * self.patch_size
            raise ValueError(
                f"圖像寬度 {width} 必須是 patch_size ({self.patch_size}) 的倍數。"
                f"建議使用: {suggested} 或 {suggested + self.patch_size}"
            )

    def _check_token_limit(self, height: int, width: int) -> None:
        """Check if token count is within limit."""
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        total_tokens = num_patches_h * num_patches_w

        if total_tokens > self.max_tokens:
            max_side = int((self.max_tokens ** 0.5) * self.patch_size)
            raise ValueError(
                f"Token 總數 {total_tokens} ({num_patches_h}x{num_patches_w}) "
                f"超過最大限制 {self.max_tokens}。\n"
                f"當前解析度: {height}x{width}\n"
                f"建議:\n"
                f"  - 降低解析度 (例如: {max_side}x{max_side} = "
                f"{(max_side // self.patch_size) ** 2} tokens)\n"
                f"  - 或使用分塊生成策略"
            )

    def compute_max_resolution(self) -> int:
        """
        Compute maximum square resolution based on token limit.

        Returns:
            Maximum side length (multiple of patch_size)
        """
        max_patches_per_side = int(self.max_tokens ** 0.5)
        return max_patches_per_side * self.patch_size

    def validate_prompt(
        self,
        prompt: Union[str, List[str]],
    ) -> List[str]:
        """
        Validate and normalize prompt input.

        Args:
            prompt: Single string or list of strings

        Returns:
            Normalized list of prompts
        """
        if isinstance(prompt, str):
            return [prompt]
        return list(prompt)

    def validate_guidance_scale(self, guidance_scale: float) -> float:
        """
        Validate guidance scale.

        Args:
            guidance_scale: CFG guidance scale

        Returns:
            Validated guidance scale
        """
        if guidance_scale < 0:
            logger.warning(
                f"Negative guidance_scale ({guidance_scale}) is unusual. "
                "Using absolute value."
            )
            return abs(guidance_scale)
        return guidance_scale


# === Standalone Functions (for backward compatibility) ===

def validate_resolution(
    height: int,
    width: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    """
    Validate image resolution (standalone function).

    Args:
        height: Image height
        width: Image width
        patch_size: Patch size
        max_tokens: Maximum tokens

    Raises:
        ValueError: If validation fails
    """
    validator = InputValidator(patch_size=patch_size, max_tokens=max_tokens)
    validator.validate_resolution(height, width)


def compute_max_resolution(
    patch_size: int = DEFAULT_PATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> int:
    """
    Compute maximum square resolution (standalone function).

    Args:
        patch_size: Patch size
        max_tokens: Maximum tokens

    Returns:
        Maximum side length
    """
    validator = InputValidator(patch_size=patch_size, max_tokens=max_tokens)
    return validator.compute_max_resolution()


__all__ = [
    # Constants
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_MAX_TOKENS",
    # Classes
    "InputValidator",
    # Functions
    "validate_resolution",
    "compute_max_resolution",
]
