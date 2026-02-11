"""
Pipeline Input Preprocessing

Handles text encoding, latent preparation, and input normalization.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Tuple, Union

import torch

from .validation import InputValidator
from .image_preprocessing import preprocess_image

if TYPE_CHECKING:
    from ...models.encoders.text_encoder import Qwen3TextEncoder

logger = logging.getLogger(__name__)


@dataclass
class GenerationInputs:
    """Preprocessed inputs ready for generation."""
    z_0: torch.Tensor
    text_embed: torch.Tensor
    text_mask: torch.Tensor
    null_text_embed: Optional[torch.Tensor]
    null_text_mask: Optional[torch.Tensor]
    batch_size: int
    height: int
    width: int


class Preprocessor:
    """Handles input preprocessing for the pipeline."""

    def __init__(
        self,
        text_encoder: Optional["Qwen3TextEncoder"],
        validator: InputValidator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.text_encoder = text_encoder
        self.validator = validator
        self.device = device
        self.dtype = dtype
        self._null_text_embed: Optional[torch.Tensor] = None
        self._null_text_mask: Optional[torch.Tensor] = None

    def prepare_inputs(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]],
        height: int,
        width: int,
        num_images_per_prompt: int,
        generator: Optional[torch.Generator],
    ) -> GenerationInputs:
        """Prepare all inputs for generation."""
        prompts = self.validator.validate_prompt(prompt)
        batch_size = len(prompts)
        total_batch = batch_size * num_images_per_prompt

        self.validator.validate_resolution(height, width)

        text_embed, text_mask, null_embed, null_mask = self.encode_prompt(
            prompts, negative_prompt, num_images_per_prompt
        )
        z_0 = self.prepare_latents(total_batch, height, width, generator)

        return GenerationInputs(
            z_0=z_0, text_embed=text_embed, text_mask=text_mask,
            null_text_embed=null_embed, null_text_mask=null_mask,
            batch_size=total_batch, height=height, width=width,
        )

    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[Union[str, List[str]]],
        num_images_per_prompt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode text prompts.

        Returns:
            Tuple of (text_embed, text_mask, null_embed, null_mask)
        """
        if self.text_encoder is None:
            raise RuntimeError("沒有可用的文本編碼器")

        batch_size = len(prompt)
        text_embed, text_mask = self._encode_text(prompt)

        if num_images_per_prompt > 1:
            text_embed = text_embed.repeat_interleave(num_images_per_prompt, dim=0)
            text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        null_embed, null_mask = self._get_negative_embeddings(
            negative_prompt, batch_size, num_images_per_prompt
        )

        text_embed = text_embed.to(device=self.device, dtype=self.dtype)
        text_mask = text_mask.to(device=self.device)
        if null_embed is not None:
            null_embed = null_embed.to(device=self.device, dtype=self.dtype)
            null_mask = null_mask.to(device=self.device)

        return text_embed, text_mask, null_embed, null_mask

    def _encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of texts.

        Returns:
            Tuple of (hidden_states, mask)
        """
        result = self.text_encoder(texts=texts, return_pooled=False)
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
        elif isinstance(result, dict):
            hidden_states = result.get("hidden_states", result.get("last_hidden_state"))
            mask = result.get("attention_mask")
            return hidden_states, mask
        return result, None

    def _get_negative_embeddings(
        self,
        negative_prompt: Optional[Union[str, List[str]]],
        batch_size: int,
        num_images_per_prompt: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get negative embeddings for CFG.

        Returns:
            Tuple of (null_embed, null_mask)
        """
        total_size = batch_size * num_images_per_prompt

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                # Validate and broadcast negative_prompt list
                if len(negative_prompt) == 1:
                    # Broadcast single negative prompt to all prompts
                    negative_prompt = negative_prompt * batch_size
                elif len(negative_prompt) != batch_size:
                    raise ValueError(
                        f"negative_prompt list length ({len(negative_prompt)}) must match "
                        f"prompt batch size ({batch_size}) or be 1 for broadcasting"
                    )
            null_embed, null_mask = self._encode_text(negative_prompt)
            if num_images_per_prompt > 1:
                null_embed = null_embed.repeat_interleave(num_images_per_prompt, dim=0)
                null_mask = null_mask.repeat_interleave(num_images_per_prompt, dim=0)
            return null_embed, null_mask

        return self._get_null_text_embed(total_size)

    def _get_null_text_embed(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get cached null text embedding.

        Returns:
            Tuple of (null_embed, null_mask). null_mask may be None if encoder
            doesn't return attention mask.
        """
        if self._null_text_embed is None:
            if self.text_encoder is None:
                raise RuntimeError("無法獲取空文本嵌入")
            self._null_text_embed, self._null_text_mask = self._encode_text([""])

        null_embed = self._null_text_embed.expand(batch_size, -1, -1)

        # Handle case where encoder doesn't return attention mask
        if self._null_text_mask is not None:
            null_mask = self._null_text_mask.expand(batch_size, -1)
            return null_embed.clone(), null_mask.clone()
        else:
            return null_embed.clone(), None

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Prepare initial noise latents."""
        shape = (batch_size, height, width, 3)
        if generator is not None:
            z_0 = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:
            z_0 = torch.randn(shape, device=self.device, dtype=self.dtype)
        return z_0

    def preprocess_image(self, image, patch_size: int = 16) -> torch.Tensor:
        """Preprocess PIL image for I2I pipeline."""
        return preprocess_image(image, patch_size)

    def clear_cache(self) -> None:
        """Clear cached null text embeddings."""
        self._null_text_embed = None
        self._null_text_mask = None


__all__ = ["GenerationInputs", "Preprocessor"]
