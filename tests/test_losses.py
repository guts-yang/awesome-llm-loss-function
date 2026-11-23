import numpy as np

from core import all_losses, get_loss
from core.evaluate import logit_slice, margin_marker, numeric_gradient, probability_grid
from core.math_utils import binary_js
from core.schema import REGISTRY, LossPlugin, register_loss


def test_registry_covers_the_four_families():
    families = {plugin.family for plugin in all_losses()}
    assert families == {"language_modeling", "alignment", "distillation", "unlearning"}
    assert len(all_losses()) >= 24


def test_every_loss_is_finite_inside_the_open_square():
    for plugin in all_losses():
        _, _, values = probability_grid(plugin, n=7, low=0.08, high=0.92)
        assert np.isfinite(values).all(), plugin.id


def test_formulas_stay_in_english_latex():
    for plugin in all_losses():
        assert plugin.latex
        assert not any("\u4e00" <= char <= "\u9fff" for char in plugin.latex)
        assert plugin.summary_zh
        assert plugin.use_zh
        assert any("\u4e00" <= char <= "\u9fff" for char in plugin.summary_zh)
        assert any("\u4e00" <= char <= "\u9fff" for char in plugin.use_zh)


def test_gradient_ascent_has_no_floor_and_full_logit_thrust():
    probability, loss, thrust = logit_slice(get_loss("ga"), low=1e-3, high=0.9, n=300)
    assert loss[0] < loss[np.argmin(np.abs(probability - 0.5))] < 0
    index = np.argmin(np.abs(probability - 0.2))
    assert abs(thrust[index] - (1.0 - probability[index])) < 0.03
    low = np.argmin(np.abs(probability - 0.01))
    assert thrust[low] > 0.95


def test_npo_thrust_shuts_off_as_probability_vanishes():
    plugin = get_loss("npo")
    params = plugin.defaults()
    params["lam"] = 0.0
    probability, loss, thrust = logit_slice(plugin, params, n=300)
    reference = params["p_ref"]
    at_ref = np.argmin(np.abs(probability - reference))
    at_low = np.argmin(np.abs(probability - 0.005))
    assert loss[at_low] < loss[at_ref]
    assert loss[at_low] < 1.0
    assert abs(thrust[at_low]) < abs(thrust[np.argmin(np.abs(probability - 0.35))])


def test_satimp_is_quieter_than_gradient_ascent_near_zero():
    sat_p, _, sat_thrust = logit_slice(get_loss("satimp"), n=300)
    ga_p, _, ga_thrust = logit_slice(get_loss("ga"), n=300)
    sat_index = np.argmin(np.abs(sat_p - 0.01))
    ga_index = np.argmin(np.abs(ga_p - 0.01))
    assert abs(sat_thrust[sat_index]) < abs(ga_thrust[ga_index])


def test_jsd_matches_symmetric_js_and_stays_nonnegative():
    plugin = get_loss("jsd")
    params = plugin.defaults()
    student = np.linspace(0.05, 0.95, 7)
    values = plugin.loss(student, 1.0 - student, params)
    assert np.allclose(values, binary_js(params["teacher"], student), atol=1e-6)
    assert np.all(values <= np.log(2) + 1e-6)
    assert abs(float(plugin.loss(params["teacher"], 1.0 - params["teacher"], params))) < 1e-6
    skewed = dict(params)
    skewed["beta"] = 0.1
    skewed_values = plugin.loss(student, 1.0 - student, skewed)
    assert np.all(skewed_values >= -1e-8)


def test_jensen_shannon_toy_stays_bounded():
    plugin = get_loss("jensun")
    values = plugin.loss(np.linspace(0.02, 0.98, 9), np.linspace(0.98, 0.02, 9))
    assert np.all(values >= -1e-8)
    assert np.all(values <= 2 * np.log(2) + 1e-6)
    assert binary_js(0.01, 0.99) <= np.log(2) + 1e-6


def test_cross_entropy_derivative():
    plugin = get_loss("next_token_ce")
    for probability in (0.15, 0.4, 0.8):
        estimated = numeric_gradient(plugin, probability, 0.3)
        assert abs(estimated - (-1.0 / probability)) < 2e-3


def test_margin_marker_sits_on_the_visible_feature():
    simpo = get_loss("simpo")
    simpo_params = simpo.defaults()
    simpo_params["beta"] = 1.0
    simpo_params["gamma"] = 0.0
    simpo_marker = margin_marker(simpo, simpo_params, y_fixed=0.6)
    assert simpo_marker["label"] == "γ=0.00"
    assert abs(simpo_marker["x"] - 0.6) < 1e-9

    slic = get_loss("slic")
    slic_params = slic.defaults()
    slic_params["delta"] = 0.5
    slic_marker = margin_marker(slic, slic_params, y_fixed=0.6)
    assert slic_marker["label"] == "δ=0.50"
    assert abs(slic_marker["x"] - 0.6 * np.exp(0.5)) < 1e-9

    simnpo = get_loss("simnpo")
    simnpo_params = simnpo.defaults()
    simnpo_params["beta"] = 0.2
    simnpo_params["gamma"] = -0.2
    simnpo_marker = margin_marker(simnpo, simnpo_params, y_fixed=0.6)
    assert simnpo_marker["label"] == "γ=-0.20"
    assert abs(simnpo_marker["x"] - np.exp(-1.0)) < 1e-9


def test_margin_marker_is_empty_outside_the_slice():
    simpo = get_loss("simpo")
    simpo_params = simpo.defaults()
    simpo_params["beta"] = 0.1
    simpo_params["gamma"] = 0.5
    assert margin_marker(simpo, simpo_params, y_fixed=0.6) is None

    slic = get_loss("slic")
    slic_params = slic.defaults()
    slic_params["delta"] = 1.0
    assert margin_marker(slic, slic_params, y_fixed=0.6) is None

    assert margin_marker(get_loss("dpo"), y_fixed=0.6) is None
    assert margin_marker(get_loss("simpo"), {"beta": 0.0, "gamma": 0.5}, y_fixed=0.6) is None


def test_class_registration_api():
    @register_loss
    class TemporaryLoss(LossPlugin):
        id = "temp_test_loss"
        name = "Temp"
        name_zh = "临时损失"
        family = "unlearning"
        latex = r"\mathcal{L}=x+y"
        summary_zh = "测试注册。"
        plane = "unlearning"

        def loss(self, x, y, params=None):
            x_arr, y_arr, _ = self.prepare(x, y, params)
            return x_arr + y_arr

    try:
        assert abs(float(get_loss("temp_test_loss").loss(0.2, 0.1)) - 0.3) < 1e-9
    finally:
        REGISTRY.pop("temp_test_loss", None)
