"""Grid evaluation, logit-gradient thrust, and reference-plane rewiring."""

from __future__ import annotations

import numpy as np

from core.schema import LossPlugin


def call_loss(plugin: LossPlugin, x, y, params=None, mode: str = "native", other: float = 0.4):
    if mode == "native":
        return plugin.loss(x, y, params)
    if not plugin.supports_reference():
        raise KeyError(f"{plugin.id} 没有参考模型项")
    merged = plugin.defaults()
    if params:
        merged.update(params)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_arr, y_arr = np.broadcast_arrays(x_arr, y_arr)
    other_arr = np.full_like(x_arr, float(other))
    if "p_ref_w" in merged:
        merged["p_ref_w"] = y_arr
        return plugin.loss(x_arr, other_arr, merged)
    merged["p_ref"] = y_arr
    return plugin.loss(x_arr, other_arr, merged)


def probability_grid(plugin: LossPlugin, params=None, n: int = 61, low: float = 0.02, high: float = 0.98, mode: str = "native", other: float = 0.4):
    axis = np.linspace(low, high, n)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    values = call_loss(plugin, x_grid, y_grid, params, mode=mode, other=other)
    return x_grid, y_grid, np.asarray(values, dtype=float)


def central_diff(values, axis) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    axis = np.asarray(axis, dtype=float)
    derivative = np.empty_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (axis[2:] - axis[:-2])
    derivative[0] = (values[1] - values[0]) / (axis[1] - axis[0])
    derivative[-1] = (values[-1] - values[-2]) / (axis[-1] - axis[-2])
    return derivative


def logit_slice(plugin: LossPlugin, params=None, y_fixed: float = 0.6, n: int = 240, low: float = 1e-3, high: float = 0.99, mode: str = "native", other: float = 0.4):
    """Return probability, loss, and logit-space thrust |dL/dz| along a 1D slice."""
    probability = np.linspace(low, high, n)
    held = np.full_like(probability, y_fixed)
    if mode == "reference":
        loss = call_loss(plugin, probability, held, params, mode=mode, other=other)
    else:
        loss = call_loss(plugin, probability, held, params, mode="native")
    loss = np.asarray(loss, dtype=float)
    dloss_dp = central_diff(loss, probability)
    dloss_dz = dloss_dp * probability * (1.0 - probability)
    return probability, loss, dloss_dz


def margin_marker(plugin: LossPlugin, params=None, y_fixed: float = 0.6, low: float = 1e-3, high: float = 0.99):
    """Label and probability where a margin parameter changes the 1D curve.

    SimPO's zero-margin point is ``y * exp(gamma / beta)``. SimNPO's logistic
    argument is zero at ``exp(gamma / beta)``. SLiC's hinge kinks at
    ``y * exp(delta)``. The point is omitted when it falls outside the slice.
    """
    merged = plugin.defaults()
    if params:
        merged.update(params)
    if plugin.id == "simpo":
        beta = float(merged.get("beta", 0.0))
        if beta <= 0:
            return None
        gamma = float(merged["gamma"])
        probability = float(y_fixed) * float(np.exp(gamma / beta))
        label = f"γ={gamma:.2f}"
    elif plugin.id == "simnpo":
        beta = float(merged.get("beta", 0.0))
        if beta <= 0:
            return None
        gamma = float(merged["gamma"])
        probability = float(np.exp(gamma / beta))
        label = f"γ={gamma:.2f}"
    elif plugin.id == "slic":
        delta = float(merged["delta"])
        probability = float(y_fixed) * float(np.exp(delta))
        label = f"δ={delta:.2f}"
    else:
        return None
    if not (low < probability < high):
        return None
    return {"label": label, "x": probability}


def numeric_gradient(plugin: LossPlugin, x: float, y: float, params=None, step: float = 1e-4):
    loss_plus = float(plugin.loss(x + step, y, params))
    loss_minus = float(plugin.loss(x - step, y, params))
    return (loss_plus - loss_minus) / (2.0 * step)
