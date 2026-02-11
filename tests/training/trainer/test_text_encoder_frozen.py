"""
Trainer text_encoder_frozen behavior tests.

Validates that Trainer respects PixelHDMConfig.text_encoder_frozen for
text encoder training mode and requires_grad settings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import PixelHDMConfig, TrainingConfig
from src.training.trainer.core import Trainer


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _create_trainer(text_encoder_frozen: bool) -> Trainer:
    config = PixelHDMConfig.for_testing()
    config.text_encoder_frozen = text_encoder_frozen
    training_config = TrainingConfig()
    model = DummyModel()
    encoder = DummyTextEncoder()
    return Trainer(
        model=model,
        config=config,
        training_config=training_config,
        dataloader=None,
        device=torch.device("cpu"),
        text_encoder=encoder,
    )


def test_text_encoder_frozen_true_sets_eval_and_no_grad():
    trainer = _create_trainer(text_encoder_frozen=True)
    assert trainer.text_encoder is not None
    assert trainer.text_encoder.training is False
    assert all(not p.requires_grad for p in trainer.text_encoder.parameters())


def test_text_encoder_frozen_false_sets_train_and_grad():
    trainer = _create_trainer(text_encoder_frozen=False)
    assert trainer.text_encoder is not None
    assert trainer.text_encoder.training is True
    assert all(p.requires_grad for p in trainer.text_encoder.parameters())
