"""Parse a small LaTeX-like formula into a vectorized loss plugin."""

from __future__ import annotations

import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import auto_number, auto_symbol, parse_expr, repeated_decimals

from core.schema import LossPlugin, ParamSpec

_FORBIDDEN = re.compile(r"__|import|lambda|exec|eval|open\s*\(|os\.|sys\.")
_SYMBOLS = ("x", "y", "beta", "gamma", "lam", "eps")
_SAFE_TRANSFORMATIONS = (auto_symbol, repeated_decimals, auto_number)
_SAFE_GLOBALS = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "log": sp.log,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
}


class FormulaError(ValueError):
    pass


def _namespace():
    symbols = {name: sp.Symbol(name, real=True) for name in _SYMBOLS}
    sigmoid = sp.Function("sigmoid")
    softplus = sp.Function("softplus")
    symbols.update({"sigmoid": sigmoid, "softplus": softplus, "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "Abs": sp.Abs})
    return symbols


def compile_formula(formula: str):
    text = formula.strip()
    if not text:
        raise FormulaError("公式是空的")
    if _FORBIDDEN.search(text):
        raise FormulaError("公式包含不允许的语法")
    local = _namespace()
    try:
        expression = parse_expr(
            text,
            local_dict=local,
            global_dict=dict(_SAFE_GLOBALS),
            transformations=_SAFE_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface a single sandbox error.
        raise FormulaError(f"无法解析公式: {exc}") from exc
    unknown = {str(symbol) for symbol in expression.free_symbols} - set(_SYMBOLS)
    if unknown:
        raise FormulaError("只允许符号 x, y, beta, gamma, lam, eps")

    def _sigmoid(values):
        values = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _softplus(values):
        values = np.asarray(values, dtype=float)
        return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))

    modules = [{"sigmoid": _sigmoid, "softplus": _softplus, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "Abs": np.abs}, "numpy"]
    arguments = [local[name] for name in _SYMBOLS]
    function = sp.lambdify(arguments, expression, modules=modules)
    return expression, function


def plugin_from_formula(formula: str, loss_id: str = "sandbox", display_name: str = "自定义损失") -> LossPlugin:
    expression, function = compile_formula(formula)

    class SandboxLoss(LossPlugin):
        id = loss_id
        name = display_name
        name_zh = display_name
        family = "unlearning"
        year = 2026
        latex = sp.latex(expression)
        summary_zh = "由公式实验室编译的自定义损失。"
        intuition_zh = "系统对 x 和 y 做数值求导，并套用同一套三维、二维和一维图。"
        physics_zh = "请自己解释每一项是间隔、温度、下界还是锚。"
        failure_zh = "未通过有限差分和边界检查之前，不要把这张图当成论文公式。"
        paper_title = ""
        paper_url = ""
        implementation_url = ""
        plane = "unlearning"
        params = [
            ParamSpec("beta", "beta", 0.5, 0.0, 4.0, 0.05),
            ParamSpec("gamma", "gamma", 0.0, -2.0, 2.0, 0.05),
            ParamSpec("lam", "lambda", 1.0, 0.0, 4.0, 0.1),
            ParamSpec("eps", "epsilon", 0.1, 0.0, 1.0, 0.01),
        ]
        bounded_below = False
        keywords = ["sandbox"]
        toy_note_zh = "符号限于 x, y, beta, gamma, lam, eps，函数限于 log, exp, sqrt, Abs, sigmoid, softplus。"

        def loss(self, x, y, params=None):
            x_arr, y_arr, merged = self.prepare(x, y, params)
            values = function(x_arr, y_arr, merged["beta"], merged["gamma"], merged["lam"], merged["eps"])
            return np.asarray(values, dtype=float)

    plugin = SandboxLoss()
    plugin.expression = expression
    return plugin


def derivative_latex(formula: str) -> str:
    expression, _ = compile_formula(formula)
    x_symbol = next((symbol for symbol in expression.free_symbols if str(symbol) == "x"), None)
    if x_symbol is None:
        return "0"
    return sp.latex(sp.diff(expression, x_symbol))


def plugin_skeleton(formula: str, loss_id: str = "my_novel_loss") -> str:
    expression, _ = compile_formula(formula)
    latex = sp.latex(expression)
    return f'''from core.schema import LossPlugin, ParamSpec, register_loss
from core.sandbox import compile_formula

_EXPR, _FN = compile_formula(r"{formula}")

@register_loss
class MyNovelLoss(LossPlugin):
    id = "{loss_id}"
    name = "My Novel Loss"
    name_zh = "我的新损失"
    family = "unlearning"
    year = 2026
    latex = r"{latex}"
    summary_zh = "一句话说明它解决的失败模式。"
    intuition_zh = "它在概率平面上把哪一个方向变成坑、哪一个方向变成墙。"
    physics_zh = "写出对 logit 的推力在 p→0 和 p→1 时分别怎样。"
    failure_zh = "写出它仍然会伤到哪一类相邻知识。"
    paper_title = ""
    paper_url = ""
    implementation_url = ""
    plane = "unlearning"
    params = [ParamSpec("beta", "beta", 0.5, 0.0, 4.0, 0.05)]
    bounded_below = False
    keywords = ["custom"]

    def loss(self, x, y, params=None):
        x_arr, y_arr, merged = self.prepare(x, y, params)
        return _FN(x_arr, y_arr, merged["beta"], merged.get("gamma", 0.0), merged.get("lam", 1.0), merged.get("eps", 0.1))
'''
