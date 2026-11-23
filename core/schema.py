"""Plugin registry for LLM loss functions.

A new loss is a subclass of ``LossPlugin`` decorated with ``@register_loss``.
The subclass supplies metadata and a vectorized ``loss`` method. The dashboard,
finite-difference checks, and toy optimizer all read the same object.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

FAMILIES: List[Tuple[str, str]] = [
    ("language_modeling", "语言建模"),
    ("alignment", "偏好对齐"),
    ("distillation", "蒸馏"),
    ("unlearning", "机器遗忘"),
]
FAMILY_LABELS = dict(FAMILIES)

PLANE_AXIS = {
    "lm": ("目标 token 概率 p_target", "对照 token 概率 p_other"),
    "alignment": ("偏好回答概率 p_w", "落选回答概率 p_l"),
    "distillation": ("学生通道概率 x", "学生通道概率 y"),
    "unlearning": ("遗忘目标概率 p_f", "保留目标概率 p_r"),
    "reference": ("当前模型概率 p_theta", "参考模型概率 p_ref"),
}


class ParamSpec:
    def __init__(self, key, label, default, minimum, maximum, step, help_text=""):
        self.key = key
        self.label = label
        self.default = float(default)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.help_text = help_text

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "help": self.help_text,
        }


class LossPlugin:
    id = ""
    name = ""
    name_zh = ""
    family = ""
    year = 0
    latex = ""
    summary_zh = ""
    intuition_zh = ""
    physics_zh = ""
    failure_zh = ""
    paper_title = ""
    paper_url = ""
    implementation_url = ""
    plane = "lm"
    params: List[ParamSpec] = []
    bounded_below = True
    step_mode = "gradient"
    keywords: List[str] = []
    toy_note_zh = ""

    def loss(self, x, y, params=None) -> np.ndarray:
        raise NotImplementedError(self.id)

    def defaults(self) -> Dict[str, float]:
        return {spec.key: spec.default for spec in self.params}

    def prepare(self, x, y, params=None):
        merged = self.defaults()
        if params:
            merged.update(params)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_arr, y_arr = np.broadcast_arrays(x_arr, y_arr)
        return x_arr, y_arr, merged

    def family_zh(self) -> str:
        return FAMILY_LABELS.get(self.family, self.family)

    def axis_labels(self, mode: str = "native") -> Tuple[str, str]:
        if mode == "reference":
            return PLANE_AXIS["reference"]
        return PLANE_AXIS.get(self.plane, PLANE_AXIS["lm"])

    def supports_reference(self) -> bool:
        keys = {spec.key for spec in self.params}
        return "p_ref" in keys or "p_ref_w" in keys

    def to_catalog(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "name_zh": self.name_zh,
            "family": self.family,
            "family_zh": self.family_zh(),
            "year": self.year,
            "latex": self.latex,
            "summary_zh": self.summary_zh,
            "intuition_zh": self.intuition_zh,
            "physics_zh": self.physics_zh,
            "failure_zh": self.failure_zh,
            "paper_title": self.paper_title,
            "paper_url": self.paper_url,
            "implementation_url": self.implementation_url,
            "plane": self.plane,
            "bounded_below": self.bounded_below,
            "step_mode": self.step_mode,
            "keywords": list(self.keywords),
            "toy_note_zh": self.toy_note_zh,
            "params": [spec.as_dict() for spec in self.params],
        }


REGISTRY: Dict[str, LossPlugin] = {}


def register_loss(cls):
    plugin = cls() if isinstance(cls, type) else cls
    if not plugin.id:
        raise ValueError("损失函数必须有 id")
    REGISTRY[plugin.id] = plugin
    return cls


def make_loss(loss_fn, **attributes):
    attrs = dict(attributes)
    class_name = "".join(part.capitalize() for part in attrs["id"].split("_")) + "Loss"

    def loss(self, x, y, params=None):
        x_arr, y_arr, merged = self.prepare(x, y, params)
        values = loss_fn(x_arr, y_arr, merged)
        return np.asarray(values, dtype=float)

    plugin_cls = type(class_name, (LossPlugin,), {"loss": loss, **attrs})
    register_loss(plugin_cls)
    return plugin_cls


def all_losses() -> List[LossPlugin]:
    order = {key: index for index, (key, _) in enumerate(FAMILIES)}
    return sorted(REGISTRY.values(), key=lambda item: (order.get(item.family, 99), item.year, item.name))


def get_loss(loss_id: str) -> LossPlugin:
    return REGISTRY[loss_id]


def losses_in(family: str) -> Iterable[LossPlugin]:
    return [item for item in all_losses() if item.family == family]
