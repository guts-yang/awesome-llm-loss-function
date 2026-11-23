"""Importing this package fills the loss registry."""

from . import alignment, distillation, language_modeling, unlearning

__all__ = ["alignment", "distillation", "language_modeling", "unlearning"]
