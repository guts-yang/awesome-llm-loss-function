"""Stable scalar helpers shared by the loss plugins and the toy dynamics."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

CLIP_LOW = 1e-6
CLIP_HIGH = 1.0 - 1e-6


def as_float(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def clip_p(probability) -> np.ndarray:
    return np.clip(as_float(probability), CLIP_LOW, CLIP_HIGH)


def sigmoid(logits) -> np.ndarray:
    return expit(as_float(logits))


def softplus(values) -> np.ndarray:
    values = as_float(values)
    positive = np.maximum(values, 0.0)
    return positive + np.log1p(np.exp(-np.abs(values)))


def log_sigmoid(values) -> np.ndarray:
    return -softplus(-as_float(values))


def binary_kl(p, q) -> np.ndarray:
    left = clip_p(p)
    right = clip_p(q)
    return left * np.log(left / right) + (1.0 - left) * np.log((1.0 - left) / (1.0 - right))


def binary_js(p, q) -> np.ndarray:
    midpoint = 0.5 * (clip_p(p) + clip_p(q))
    return 0.5 * binary_kl(p, midpoint) + 0.5 * binary_kl(q, midpoint)
