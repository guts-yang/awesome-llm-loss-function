import numpy as np

from core.sandbox import FormulaError, compile_formula, derivative_latex, plugin_from_formula
from core.schema import get_loss
from core.toy_dynamics import feature_axes, project_orthogonal, retain_cross_entropy_gradient, simulate
from viz.plots import _downhill


def test_sandbox_compiles_and_rejects_escape():
    expression, function = compile_formula("log(x)+beta*y")
    assert "log" in str(expression)
    value = function(np.array([0.5]), np.array([0.25]), 2.0, 0.0, 1.0, 0.1)
    assert abs(float(value[0]) - (np.log(0.5) + 0.5)) < 1e-8
    try:
        compile_formula("__import__('os').system('echo no')")
        raise AssertionError("sandbox accepted an import")
    except FormulaError:
        pass


def test_sandbox_derivative_and_plugin_grid():
    latex = derivative_latex("log(x)+beta*y")
    assert "x" in latex
    plugin = plugin_from_formula("-(2/beta)*log(sigmoid(-beta*(log(x)-log(0.6))))")
    values = plugin.loss(np.array([0.05, 0.6, 0.9]), np.array([0.4, 0.4, 0.4]))
    assert np.isfinite(values).all()
    assert values[0] < values[2]


def test_orthogonal_step_removes_the_retain_component():
    gradient = np.array([0.8, -0.3])
    protect = np.array([0.2, 0.7])
    projected = project_orthogonal(gradient, protect)
    assert abs(float(np.dot(projected, protect))) < 1e-8


def test_projected_dynamics_stay_finite_and_protect_retain_direction():
    plugin = get_loss("ga")
    plain = simulate(plugin, optimizer="sgd", steps=15, learning_rate=0.2, phi_degrees=25)
    projected = simulate(plugin, optimizer="projected", steps=15, learning_rate=0.2, phi_degrees=25)
    assert np.isfinite(plain["loss"]).all()
    assert np.isfinite(projected["loss"]).all()
    retain_drop_plain = plain["retain_probability"][0] - plain["retain_probability"][-1]
    retain_drop_projected = projected["retain_probability"][0] - projected["retain_probability"][-1]
    assert retain_drop_projected < retain_drop_plain
    _, retain_axis = feature_axes(25)
    protect = retain_cross_entropy_gradient(np.array([1.0, 0.2]), retain_axis)
    assert np.linalg.norm(protect) > 0


def test_downhill_arrows_point_toward_a_bowl():
    axis = np.linspace(0.0, 1.0, 25)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    values = (x_grid - 0.2) ** 2 + (y_grid - 0.8) ** 2
    x_points, y_points, arrow_x, arrow_y = _downhill(x_grid, y_grid, values, stride=6)
    target_x = 0.2 - x_points
    target_y = 0.8 - y_points
    alignment = arrow_x * target_x + arrow_y * target_y
    assert np.nanmean(alignment) > 0.2
