"""Next-token and span language-modeling losses on a two-token toy."""

import numpy as np

from core.math_utils import clip_p
from core.schema import ParamSpec, make_loss


def _log(probability):
    return np.log(clip_p(probability))


def _cross_entropy(x, y, params):
    return -_log(x) + 0.0 * y


def _label_smoothing(x, y, params):
    epsilon = params["epsilon"]
    return -(1.0 - epsilon) * _log(x) - epsilon * _log(y)


def _span_corruption(x, y, params):
    # UL2 changes which tokens are predicted. The scoring rule remains cross-entropy.
    return -_log(x) + 0.0 * y


make_loss(
    _cross_entropy,
    id="next_token_ce",
    name="Next-token CE",
    name_zh="下一 token 交叉熵",
    family="language_modeling",
    year=2003,
    latex=r"\mathcal{L}_{\mathrm{CE}}=-\log p_{\theta}(x_t\mid x_{<t})",
    summary_zh="自回归语言模型的基本损失：提高下一个正确 token 的概率。",
    intuition_zh="交叉熵只奖励目标 token。概率越低，对数损失下降得越陡，所以模型会把质量集中到当前答案上。",
    physics_zh="对 logit z 而言，dL/dz = p_target - 1。目标概率接近 1 时推力消失，接近 0 时推力维持在 1 附近。这是后面所有偏好损失和遗忘损失的基准力。",
    failure_zh="单纯最大似然会过拟合表面形式，也会把相近答案的概率一起抬高。它没有偏好间隔，也没有遗忘下界。",
    paper_title="A Neural Probabilistic Language Model",
    paper_url="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
    implementation_url="https://github.com/huggingface/transformers",
    plane="lm",
    params=[],
    bounded_below=True,
    keywords=["cross-entropy", "language modeling"],
    toy_note_zh="二元玩具里，横轴是目标 token 概率。损失与纵轴无关。",
)

make_loss(
    _label_smoothing,
    id="label_smoothing",
    name="Label smoothing",
    name_zh="标签平滑",
    family="language_modeling",
    year=2016,
    latex=r"\mathcal{L}_{\mathrm{LS}}=-(1-\epsilon)\log p_{\mathrm{target}}-\epsilon\log p_{\mathrm{other}}",
    summary_zh="把一部分概率质量分给非目标 token，避免模型把全部质量压到一个 logit 上。",
    intuition_zh="平滑项是在交叉熵旁边加一个对照 token 的对数惩罚。epsilon 越大，最优点越不愿意把目标概率推到 1。",
    physics_zh="目标通道的 logit 梯度变成 (1-epsilon)(p_target-1) 与对照通道泄漏的叠加。它给最大似然加了一个软上限，后面 Unilogit 的均匀目标用的是同一类机制。",
    failure_zh="平滑过强会抬高不该出现的 token，续写变得发虚。这里的对照项是二元简化，不是对整个词表做均匀平均。",
    paper_title="Rethinking the Inception Architecture for Computer Vision",
    paper_url="https://arxiv.org/abs/1512.00567",
    implementation_url="",
    plane="lm",
    params=[
        ParamSpec("epsilon", "平滑系数 epsilon", 0.1, 0.0, 0.4, 0.01, "分给对照 token 的概率质量。"),
    ],
    bounded_below=True,
    keywords=["label smoothing"],
    toy_note_zh="经典论文来自视觉网络，LLM 训练借用的是同一条平滑规则。图上只保留目标与一个对照 token。",
)

make_loss(
    _span_corruption,
    id="ul2_span",
    name="UL2 span CE",
    name_zh="UL2 span 交叉熵",
    family="language_modeling",
    year=2022,
    latex=r"\mathcal{L}_{\mathrm{UL2}}=-\log p_{\theta}(x_{\mathrm{span}}\mid x_{\setminus\mathrm{span}})",
    summary_zh="UL2 把不同跨度的遮盖、前缀语言建模和普通下一 token 混在同一个交叉熵里。",
    intuition_zh="损失的形状仍是负对数似然。UL2 改变的是被预测的 span 和任务混合比例，不是评分函数本身。",
    physics_zh="因此它的 logit 推力和下一 token 交叉熵相同：目标概率靠近 1 时力消失。差别出现在批量里哪些位置产生这条力。",
    failure_zh="如果把 UL2 理解成一种新的几何约束，会高估它。它解决的是去噪任务覆盖，不解决偏好冲突或遗忘无界。",
    paper_title="UL2: Unifying Language Learning Paradigms",
    paper_url="https://arxiv.org/abs/2205.05131",
    implementation_url="",
    plane="lm",
    params=[],
    bounded_below=True,
    keywords=["UL2", "span corruption"],
    toy_note_zh="曲面上它与下一 token 交叉熵重合。这是故意保留的：两者的差异在数据构造，不在损失形状。",
)
