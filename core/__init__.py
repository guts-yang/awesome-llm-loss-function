"""LLM loss registry, toy dynamics, and the formula sandbox."""

from core.losses import alignment, distillation, language_modeling, unlearning  # noqa: F401
from core.schema import REGISTRY, all_losses, get_loss

__all__ = ["REGISTRY", "all_losses", "get_loss", "alignment", "distillation", "language_modeling", "unlearning"]
