"""Two-feature toy: forget and retain directions overlap by an angle phi."""

from __future__ import annotations

import numpy as np

from core.evaluate import central_diff
from core.math_utils import clip_p, sigmoid
from core.schema import LossPlugin


def feature_axes(phi_degrees: float):
    phi = np.deg2rad(float(phi_degrees))
    forget = np.array([1.0, 0.0])
    retain = np.array([np.cos(phi), np.sin(phi)])
    return forget, retain


def channel_probabilities(theta, forget_axis, retain_axis):
    theta = np.asarray(theta, dtype=float)
    forget_p = sigmoid(np.asarray(theta @ forget_axis, dtype=float))
    retain_p = sigmoid(np.asarray(theta @ retain_axis, dtype=float))
    return forget_p, retain_p


def parameter_gradient(plugin: LossPlugin, theta, forget_axis, retain_axis, params=None, step: float = 1e-4):
    gradient = np.zeros(2, dtype=float)
    for index in range(2):
        direction = np.zeros(2)
        direction[index] = step
        plus_f, plus_r = channel_probabilities(theta + direction, forget_axis, retain_axis)
        minus_f, minus_r = channel_probabilities(theta - direction, forget_axis, retain_axis)
        upper = float(plugin.loss(plus_f, plus_r, params))
        lower = float(plugin.loss(minus_f, minus_r, params))
        gradient[index] = (upper - lower) / (2.0 * step)
    return gradient


def retain_cross_entropy_gradient(theta, retain_axis, step: float = 1e-4):
    """Gradient of -log p_r, the utility direction used by the orthogonal step."""
    gradient = np.zeros(2, dtype=float)
    for index in range(2):
        direction = np.zeros(2)
        direction[index] = step
        _, plus_r = channel_probabilities(theta + direction, np.array([1.0, 0.0]), retain_axis)
        _, minus_r = channel_probabilities(theta - direction, np.array([1.0, 0.0]), retain_axis)
        upper = float(-np.log(clip_p(plus_r)))
        lower = float(-np.log(clip_p(minus_r)))
        gradient[index] = (upper - lower) / (2.0 * step)
    return gradient


def project_orthogonal(gradient, protect):
    denominator = float(np.dot(protect, protect)) + 1e-12
    return gradient - (float(np.dot(gradient, protect)) / denominator) * protect


def parameter_grid(plugin: LossPlugin, phi_degrees: float, params=None, n: int = 51, limit: float = 2.5):
    axis = np.linspace(-limit, limit, n)
    theta_1, theta_2 = np.meshgrid(axis, axis, indexing="xy")
    forget_axis, retain_axis = feature_axes(phi_degrees)
    flat_theta = np.stack([theta_1.ravel(), theta_2.ravel()], axis=1)
    forget_p, retain_p = channel_probabilities(flat_theta, forget_axis, retain_axis)
    values = plugin.loss(forget_p, retain_p, params).reshape(theta_1.shape)
    return theta_1, theta_2, values


def simulate(
    plugin: LossPlugin,
    params=None,
    phi_degrees: float = 30.0,
    optimizer: str = "sgd",
    steps: int = 40,
    learning_rate: float = 0.15,
    theta0=(1.3, 0.2),
):
    forget_axis, retain_axis = feature_axes(phi_degrees)
    theta = np.asarray(theta0, dtype=float).copy()
    history = [theta.copy()]
    moment = np.zeros(2)
    second = np.zeros(2)
    beta1, beta2 = 0.9, 0.999
    for step_index in range(1, steps + 1):
        gradient = parameter_gradient(plugin, theta, forget_axis, retain_axis, params)
        if optimizer == "projected" or (optimizer == "auto" and plugin.step_mode == "project"):
            protect = retain_cross_entropy_gradient(theta, retain_axis)
            gradient = project_orthogonal(gradient, protect)
            theta = theta - learning_rate * gradient
        elif optimizer == "adam":
            moment = beta1 * moment + (1.0 - beta1) * gradient
            second = beta2 * second + (1.0 - beta2) * gradient * gradient
            moment_hat = moment / (1.0 - beta1**step_index)
            second_hat = second / (1.0 - beta2**step_index)
            theta = theta - learning_rate * moment_hat / (np.sqrt(second_hat) + 1e-8)
        else:
            theta = theta - learning_rate * gradient
        history.append(theta.copy())
    path = np.stack(history)
    forget_p, retain_p = channel_probabilities(path, forget_axis, retain_axis)
    losses = np.asarray(plugin.loss(forget_p, retain_p, params), dtype=float)
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    return {
        "path": path,
        "forget_probability": np.asarray(forget_p, dtype=float),
        "retain_probability": np.asarray(retain_p, dtype=float),
        "loss": losses,
        "arc": arc,
        "thrust": central_diff(losses, arc + 1e-8),
    }
