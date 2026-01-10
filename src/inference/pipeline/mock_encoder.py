"""
Mock Text Encoder for Testing

Provides a lightweight text encoder replacement for testing pipelines
without loading large language models.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn


class MockTextEncoder(nn.Module):
    """
    Mock text encoder for testing.

    Outputs deterministic random embeddings for testing pipelines
    without loading large language models.

    Args:
        hidden_size: Embedding dimension
        max_length: Maximum sequence length
        device: Target device
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        max_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self._seed = 42

    def forward(
        self,
        texts: List[str],
        return_pooled: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode texts.

        Args:
            texts: List of text strings
            return_pooled: Whether to return pooled embedding
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            If return_pooled=True: (embeddings, attention_mask, pooled_output)
            If return_pooled=False: (embeddings, attention_mask)
        """
        batch_size = len(texts)
        seq_lens = [min(len(text) // 4 + 1, self.max_length) for text in texts]
        fixed_seq_len = 77  # Fixed length like CLIP

        # Deterministic random generation
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self._seed)

        embeddings = torch.randn(
            batch_size,
            fixed_seq_len,
            self.hidden_size,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Attention mask
        attention_mask = torch.zeros(
            batch_size,
            fixed_seq_len,
            device=self.device,
            dtype=torch.bool,
        )
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :min(seq_len, fixed_seq_len)] = True

        if return_pooled:
            # Pooled output: last token embedding for each sequence
            pooled_output = torch.randn(
                batch_size,
                self.hidden_size,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
            return embeddings, attention_mask, pooled_output

        return embeddings, attention_mask

    def to(self, *args, **kwargs) -> "MockTextEncoder":
        """Move to device or convert dtype."""
        for arg in args:
            if isinstance(arg, torch.device):
                self.device = arg
            elif isinstance(arg, torch.dtype):
                self.dtype = arg

        if "device" in kwargs:
            self.device = kwargs["device"]
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]

        return self


__all__ = [
    "MockTextEncoder",
]
