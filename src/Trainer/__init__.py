"""Training components for the keystroke synthesis model.

This package provides the model architecture, loss function, training loop,
and associated utilities for training a text-to-keystroke synthesis model.

Modules:
    Trainer: Orchestrates the full training loop with validation and early stopping.
    TextToKeystrokeModelMultiHead: DeBERTa-based model with heteroscedastic regression heads.
    HeteroscedasticKLLoss: Gaussian NLL + KL divergence loss for mean/variance prediction.
"""

from .Trainer import Trainer
from .TextToKeystrokeModelMultiHead import TextToKeystrokeModelMultiHead
from .HeteroscedasticKLLoss import HeteroscedasticKLLoss

__all__ = ['Trainer', 'TextToKeystrokeModelMultiHead', 'HeteroscedasticKLLoss']