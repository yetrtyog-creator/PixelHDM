"""
Text Processor Blocks

Modality-internal processing for text tokens before joint text-image attention.
"""

from .core import TextProcessorBlock, TextProcessorStack

__all__ = ["TextProcessorBlock", "TextProcessorStack"]
