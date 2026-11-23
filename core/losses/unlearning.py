"""Machine-unlearning losses on a forget/retain probability plane."""

import numpy as np

from core.math_utils import binary_js, binary_kl, clip_p, log_sigmoid, softplus
from core.schema import ParamSpec, make_loss


def _log(probability):
    return np.log(clip_p(probability))


def _ga(x, y, params):
    return _log(x) + 0.0 * y


def _graddiff(x, y, params):
    return _log(x) - _log(y)


def _ga_kl(x, y, params):
    return _log(x) + params["beta"] * binary_kl(y, params["p_ref"])


def _npo(x, y, params):
    ratio = _log(x) - _log(params["p_ref"])
    forget = -(2.0 / params["beta"]) * log_sigmoid(-params["beta"] * ratio)
    retain = params["lam"] * (-_log(y))
    return forget + retain


def _simnpo(x, y, params):
    forget = -(2.0 / params["beta"]) * log_sigmoid(-(params["beta"] * _log(x) - params["gamma"]))
    return forget + params["lam"] * (-_log(y))


def _whp(x, y, params):
    # Alternative-label CE on the normalized pair. y is the replacement completion.
    return -_log(y) + _log(clip_p(x) + clip_p(y))


def _unilogit(x, y, params):
    vocab = params["vocab"]
    uniform_ce = -(1.0 / vocab) * _log(x) - (1.0 - 1.0 / vocab) * np.log(clip_p(1.0 - x))
    retain = params["lam"] * binary_kl(params["p_ref"], y)
    return uniform_ce + retain


def _rmu(x, y, params):
    return (clip_p(x) - params["noise"]) ** 2 + params["alpha"] * (clip_p(y) - params["p_ref"]) ** 2


def _shred(x, y, params):
    demotion = softplus(params["beta"] * (_log(x) - _log(params["p_teacher"])) + params["margin"])
    self_distill = params["lam"] * binary_kl(params["p_teacher"], y)
    return demotion + self_distill


def _jensun(x, y, params):
    return params["lam_f"] * binary_js(x, params["p_safe"]) + params["lam_r"] * binary_js(y, params["p_ref"])


def _blur(x, y, params):
    return _log(x) + params["lam"] * binary_kl(y, params["p_ref"])


def _satimp(x, y, params):
    probability = clip_p(x)
    weight = probability ** params["beta1"] * (1.0 - probability) ** params["beta2"]
    return weight * _log(x) + params["lam"] * (-_log(y))


make_loss(
    _ga,
    id="ga",
    name="Gradient Ascent",
    name_zh="梯度上升",
    family="unlearning",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{GA}}=\log p_{\theta}(y_f\mid x)",
    summary_zh="把遗忘答案的交叉熵翻号。最小化这个损失就是在降低遗忘答案的概率。",
    intuition_zh="p_f 越小，损失越负。曲面向概率 0 的方向无限下降，没有坑底。",
    physics_zh="令 p=σ(z)。对 logit 的导数是 1-p。p 趋向 0 时导数趋向 1，推力停在最大值，不会自己关掉。优化器可以一直把这个 token 往下压，并把纠缠在一起的相邻 logit 带走。",
    failure_zh="遗忘集稍大、学习率稍高，模型就会塌成拒答、复读或胡言。TOFU 一类基准里它是强遗忘、差效用的参照。",
    paper_title="TOFU: A Task of Fictitious Unlearning for LLMs",
    paper_url="https://arxiv.org/abs/2401.06121",
    implementation_url="https://github.com/locuslab/tofu",
    plane="unlearning",
    params=[],
    bounded_below=False,
    keywords=["gradient ascent", "GA", "TOFU"],
    toy_note_zh="纵轴不进入损失。能力保持要靠别的项，或者靠参数平面上的投影。",
)

make_loss(
    _graddiff,
    id="graddiff",
    name="GradDiff",
    name_zh="梯度差",
    family="unlearning",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{GradDiff}}=\log p_{\theta}(y_f\mid x)-\log p_{\theta}(y_r\mid x)",
    summary_zh="遗忘集上做梯度上升，保留集上继续做交叉熵。",
    intuition_zh="损失沿 p_f 下降、沿 p_r 上升的方向变小。它同时要求忘掉一个答案、记住另一个答案。",
    physics_zh="两条力分别是 1-p_f 和 p_r-1 的 logit 梯度。保留项有下界 0 附近的停止点，遗忘项没有。合力和两条任务的特征夹角有关，夹角小的时候保留梯度抵消不了遗忘梯度在相邻方向上的分量。",
    failure_zh="保留集覆盖不到的相邻知识仍会被带走。遗忘项无界，训练久了保留项也会被淹没。",
    paper_title="TOFU: A Task of Fictitious Unlearning for LLMs",
    paper_url="https://arxiv.org/abs/2401.06121",
    implementation_url="https://github.com/locuslab/tofu",
    plane="unlearning",
    params=[],
    bounded_below=False,
    keywords=["GradDiff", "gradient difference"],
    toy_note_zh="",
)

make_loss(
    _ga_kl,
    id="ga_kl",
    name="GA + KL",
    name_zh="梯度上升加 KL",
    family="unlearning",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{GA+KL}}=\log p_{\theta}(y_f)+\beta\, D_{\mathrm{KL}}\!\left(p_{\theta}(\cdot\mid x_r)\Vert p_{\mathrm{ref}}(\cdot\mid x_r)\right)",
    summary_zh="遗忘答案仍做梯度上升，保留分布用 KL 锚在参考模型上。",
    intuition_zh="KL 的最低点在 p_r = p_ref，不像交叉熵那样把保留概率一直推向 1。",
    physics_zh="beta 是弹簧刚度。它限制保留通道离开原模型的距离，但不改变遗忘通道在 p_f→0 时推力趋向 1 的事实。",
    failure_zh="遗忘集和保留集的表示重叠时，KL 锚会和梯度上升对着拉，结果常常是两边都没到位，或者学习率一大仍从遗忘通道崩掉。",
    paper_title="Machine Unlearning of Pre-trained Large Language Models",
    paper_url="https://arxiv.org/abs/2402.15159",
    implementation_url="https://github.com/locuslab/tofu",
    plane="unlearning",
    params=[
        ParamSpec("beta", "KL 权重 beta", 0.5, 0.0, 4.0, 0.05),
        ParamSpec("p_ref", "参考保留概率", 0.7, 0.05, 0.95, 0.01),
    ],
    bounded_below=False,
    keywords=["KL", "gradient ascent"],
    toy_note_zh="KL 写在伯努利保留通道上，对应序列实现里保留提示上的 token KL。",
)

make_loss(
    _npo,
    id="npo",
    name="NPO",
    name_zh="负偏好优化",
    family="unlearning",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{NPO}}=-\frac{2}{\beta}\log\sigma\!\left(-\beta\log\frac{\pi_{\theta}(y_f)}{\pi_{\mathrm{ref}}(y_f)}\right)",
    summary_zh="把遗忘回答当成偏好优化里的落选回答，让遗忘损失在概率变低后自动失去梯度。",
    intuition_zh="p_f 远低于参考模型时，sigmoid 饱和，损失贴着 0。它不再像梯度上升那样越忘越有力。",
    physics_zh="对 logit 的推力正比于 1-σ(-β log(p/p_ref))，再乘上 (1-p)。p 趋向 0 时第一项指数衰减，推力归零。β 越小，开始减速的区间越宽。图上的保留交叉熵是可选锚，权重可以调到 0。",
    failure_zh="参考模型如果已经把遗忘答案写得很深，log(p/p_ref) 要降很多才饱和，早期更新仍接近梯度上升。只调 β 不能解决保留集里没有覆盖到的相邻事实。",
    paper_title="Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning",
    paper_url="https://arxiv.org/abs/2404.05868",
    implementation_url="https://github.com/licong-lin/negative-preference-optimization",
    plane="unlearning",
    params=[
        ParamSpec("beta", "温度 beta", 2.0, 0.1, 5.0, 0.1, "序列实现常用 0.1。单个 token 的对数概率尺度更小，玩具默认放大 beta，饱和才会落在概率轴上。"),
        ParamSpec("p_ref", "参考遗忘概率", 0.6, 0.05, 0.95, 0.01),
        ParamSpec("lam", "保留交叉熵权重", 0.0, 0.0, 2.0, 0.05),
    ],
    bounded_below=True,
    keywords=["NPO", "negative preference"],
    toy_note_zh="公式主体是论文中的遗忘项。保留交叉熵是可选附加项，默认权重为 0。",
)

make_loss(
    _simnpo,
    id="simnpo",
    name="SimNPO",
    name_zh="简单负偏好优化",
    family="unlearning",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{SimNPO}}=-\frac{2}{\beta}\log\sigma\!\left(-\frac{\beta}{|y|}\log\pi_{\theta}(y_f)-\gamma\right)",
    summary_zh="NPO 去掉参考模型，改用长度归一化的对数概率和一个间隔 gamma。",
    intuition_zh="没有 p_ref 可调。概率本身变小之后，logistic 仍然饱和，所以下界还在。",
    physics_zh="gamma 平移饱和开始的位置。beta 控制饱和斜率。二元玩具取回答长度为 1。它和 SimPO 的关系，相当于 NPO 和 DPO 的关系。",
    failure_zh="没有参考模型时，模型可以把所有答案的概率一起压低，只要遗忘答案相对更低。长度估计偏了，长短回答的遗忘强度会不一致。",
    paper_title="Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning",
    paper_url="https://arxiv.org/abs/2410.07163",
    implementation_url="",
    plane="unlearning",
    params=[
        ParamSpec("beta", "温度 beta", 1.5, 0.1, 5.0, 0.1, "二元玩具放大了 beta，用来代替一整段回答的对数概率。"),
        ParamSpec("gamma", "间隔 gamma", 0.0, -1.0, 1.5, 0.05),
        ParamSpec("lam", "保留交叉熵权重", 0.0, 0.0, 2.0, 0.05),
    ],
    bounded_below=True,
    keywords=["SimNPO", "NPO"],
    toy_note_zh="长度归一化取 |y|=1。",
)

make_loss(
    _whp,
    id="whp",
    name="WHP",
    name_zh="替代答案微调",
    family="unlearning",
    year=2023,
    latex=r"\mathcal{L}_{\mathrm{WHP}}=-\log\frac{p_{\mathrm{alt}}}{p_f+p_{\mathrm{alt}}}",
    summary_zh="不把原答案的对数概率往下推，而在替代续写上做普通交叉熵。Who's Harry Potter 用的是这条路线。",
    intuition_zh="损失只关心替代答案在这对概率里的份额。原答案概率下降是归一化的副作用，不是无界的梯度上升。",
    physics_zh="最低点在 p_f 相对 p_alt 很小的方向，损失本身有下界 0。力来自份额而不是 log p_f 的绝对高度，所以不会出现越忘越猛的 logit 推力。",
    failure_zh="替代语料如果和原知识写在相邻句式里，交叉熵会把相邻事实一起改写。没有替代答案时，这条损失无从定义。",
    paper_title="Who's Harry Potter? Approximate Unlearning in LLMs",
    paper_url="https://arxiv.org/abs/2310.02238",
    implementation_url="",
    plane="unlearning",
    params=[],
    bounded_below=True,
    keywords=["Harry Potter", "WHP", "alternative label"],
    toy_note_zh="纵轴是替代续写的概率。真实方法在新答案的 token 上做交叉熵。",
)

make_loss(
    _unilogit,
    id="unilogit",
    name="Unilogit",
    name_zh="均匀目标自蒸馏",
    family="unlearning",
    year=2025,
    latex=r"\mathcal{L}_{\mathrm{Unilogit}}=\mathrm{CE}(p_{\theta}, U)+\lambda\, D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\theta})",
    summary_zh="遗忘位置的目标改成均匀分布，保留位置用参考模型做自蒸馏。不需要外部保留答案。",
    intuition_zh="均匀目标的交叉熵在 p_f = 1/|V| 处最低。原答案既不会被抬到 1，也不会被梯度上升到 0。",
    physics_zh="词表越大，最低点越靠近 0，但坑底仍在。保留项是正向 KL，学生必须覆盖参考模型的模式，不能只把质量堆到一个拒答 token。",
    failure_zh="均匀分布不是一个自然的拒答。模型可以在遗忘提示上变得很平，看起来像遗忘，生成质量却变成高熵噪声。",
    paper_title="Unilogit: Robust Machine Unlearning for LLMs Using Uniform-Target Self-Distillation",
    paper_url="https://arxiv.org/abs/2505.06027",
    implementation_url="",
    plane="unlearning",
    params=[
        ParamSpec("vocab", "词表大小 |V|", 128, 8, 1024, 8),
        ParamSpec("lam", "保留蒸馏权重", 1.0, 0.0, 4.0, 0.1),
        ParamSpec("p_ref", "参考保留概率", 0.7, 0.05, 0.95, 0.01),
    ],
    bounded_below=True,
    keywords=["Unilogit", "uniform", "self-distillation"],
    toy_note_zh="遗忘通道用“目标 token 对其余质量”的二元交叉熵近似全词表均匀目标。",
)

make_loss(
    _rmu,
    id="rmu",
    name="RMU",
    name_zh="表示错向",
    family="unlearning",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{RMU}}=\lVert h_f-c\,u\rVert_2^2+\alpha\lVert h_r-h_{\mathrm{ref}}\rVert_2^2",
    summary_zh="WMDP 的表示错向：遗忘激活被拉向一个随机方向，保留激活被拉回原模型。",
    intuition_zh="概率平面上这是两个二次坑。遗忘通道的坑在噪声目标，保留通道的坑在参考概率。两个方向都有下界。",
    physics_zh="alpha 决定保留弹簧有多硬。二次损失的梯度随误差线性下降，靠近目标就停，不像 log p 在边界上维持满推力。参数平面里，这两个坑分别沿遗忘特征和保留特征打开。",
    failure_zh="随机方向如果落进仍被读出的子空间，模型会改口而不是遗忘。系数 c 过大时，遗忘层的激活幅度异常，下游层会跟着失真。",
    paper_title="The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning",
    paper_url="https://arxiv.org/abs/2403.03218",
    implementation_url="https://github.com/centerforaisafety/wmdp",
    plane="unlearning",
    params=[
        ParamSpec("noise", "遗忘通道噪声目标", 0.15, 0.0, 0.5, 0.01),
        ParamSpec("alpha", "保留权重 alpha", 1.0, 0.0, 5.0, 0.1),
        ParamSpec("p_ref", "参考保留概率", 0.75, 0.05, 0.95, 0.01),
    ],
    bounded_below=True,
    keywords=["RMU", "representation misdirection", "WMDP"],
    toy_note_zh="隐藏状态上的平方距离被压成两个概率通道上的二次项，用来保留“两边都有坑底”这一几何。",
)

make_loss(
    _shred,
    id="shred",
    name="SHRED",
    name_zh="自蒸馏 logit 降权",
    family="unlearning",
    year=2026,
    latex=r"\mathcal{L}_{\mathrm{SHRED}}=\mathrm{softplus}\!\left(\beta\log\frac{p_{\theta}(y_f)}{p_{\mathrm{self}}(y_f)}+m\right)+\lambda\, D_{\mathrm{KL}}(p_{\mathrm{self}}\Vert p_{\theta})",
    summary_zh="没有外部保留集。用冻结的自己当教师，压低遗忘 token 的 logit，同时在其余通道上做自蒸馏。",
    intuition_zh="softplus 在学生已经低于教师一个间隔后梯度消失。自蒸馏把其余通道锚在更新前的模型上。",
    physics_zh="margin 决定要低出多少才停。beta 决定降权有多陡。因为教师是冻结的，损失不会随着自己的概率一起滑向负无穷。",
    failure_zh="教师本身就会输出遗忘答案时，降权只是相对这个教师下移一点，答案仍可能留在 top-k 里。自蒸馏权重太小，其余通道会跟着漂移。",
    paper_title="SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion",
    paper_url="https://arxiv.org/abs/2605.07482",
    implementation_url="",
    plane="unlearning",
    params=[
        ParamSpec("beta", "降权斜率 beta", 2.0, 0.2, 6.0, 0.1),
        ParamSpec("margin", "间隔 m", 0.5, 0.0, 2.0, 0.05),
        ParamSpec("lam", "自蒸馏权重", 1.0, 0.0, 4.0, 0.1),
        ParamSpec("p_teacher", "冻结自教师概率", 0.65, 0.05, 0.95, 0.01),
    ],
    bounded_below=True,
    keywords=["SHRED", "self-distillation", "logit demotion"],
    toy_note_zh="二元软间隔加自蒸馏是对论文机制的玩具，不是逐 token 的完整实现。",
)

make_loss(
    _jensun,
    id="jensun",
    name="JensUn",
    name_zh="Jensen-Shannon 遗忘",
    family="unlearning",
    year=2025,
    latex=r"\mathcal{L}_{\mathrm{JensUn}}=\lambda_F\,\mathrm{JSD}(p_{\theta}\Vert p_{\mathrm{safe}})+\lambda_R\,\mathrm{JSD}(p_{\theta}\Vert p_{\mathrm{ref}})",
    summary_zh="遗忘通道去贴近一个安全目标，保留通道去贴近参考模型。两边都用有上界的 Jensen-Shannon 散度。",
    intuition_zh="JS 散度落在 0 和 log 2 之间。目标达到以后梯度消失，离目标再远也不会像 KL 或 log p 那样把力放到无限大。",
    physics_zh="p_safe 是遗忘通道的坑底，通常是一个很小的概率或拒答 token。p_ref 是保留通道的坑底。两个二次型附近的梯度和 JS 的饱和一起，使更新在结束时自动变小。",
    failure_zh="安全目标如果只是空 token 或拒答，模型学会的可能是换一种说法，而不是丢掉那个事实。JS 对两个分布的中间点对称，早期梯度比梯度上升温和，遗忘速度也更慢。",
    paper_title="Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs",
    paper_url="https://arxiv.org/abs/2509.02820",
    implementation_url="https://github.com/nmndeep/JensUn-Unlearning",
    plane="unlearning",
    params=[
        ParamSpec("lam_f", "遗忘权重", 1.0, 0.0, 4.0, 0.1),
        ParamSpec("lam_r", "保留权重", 1.0, 0.0, 4.0, 0.1),
        ParamSpec("p_safe", "安全目标概率", 0.05, 0.01, 0.4, 0.01),
        ParamSpec("p_ref", "参考保留概率", 0.75, 0.05, 0.95, 0.01),
    ],
    bounded_below=True,
    keywords=["JensUn", "Jensen", "JSD"],
    toy_note_zh="论文在序列上对拒绝目标的 one-hot 和参考分布分别计算 JSD。这里把两个目标放进遗忘、保留两个伯努利通道。",
)

make_loss(
    _blur,
    id="blur",
    name="BLUR",
    name_zh="双层正交遗忘",
    family="unlearning",
    year=2025,
    latex=r"\theta^{\star}(\omega)=\arg\min_{\theta}\mathcal{L}_R(\theta;\omega),\quad \min_{\omega}\mathcal{L}_F(\theta^{\star}(\omega))",
    summary_zh="外层降低遗忘损失，内层先把保留损失修回来。参数平面上用保留梯度的正交补来近似这个双层约束。",
    intuition_zh="概率平面上仍能看到无界的遗忘项和一条保留 KL 弹簧。真正不同的是更新方向：步进前先去掉沿保留梯度的分量。",
    physics_zh="设 g_f 是遗忘损失的参数梯度，g_r 是保留损失的参数梯度。正交步使用 g_f - proj_{g_r} g_f。特征夹角 φ 越小，被去掉的分量越大，遗忘变慢，保留概率的漂移变小。",
    failure_zh="正交投影只保护当前这一步的保留梯度方向。保留损失的梯度和真正要保护的知识不一致时，投影会保护错方向。它也不给遗忘损失增加下界。",
    paper_title="BLUR: A Bi-Level Optimization Approach for LLM Unlearning",
    paper_url="https://arxiv.org/abs/2506.08164",
    implementation_url="",
    plane="unlearning",
    params=[
        ParamSpec("lam", "保留 KL 权重", 1.0, 0.0, 4.0, 0.1),
        ParamSpec("p_ref", "参考保留概率", 0.75, 0.05, 0.95, 0.01),
    ],
    bounded_below=False,
    step_mode="project",
    keywords=["BLUR", "bi-level", "orthogonal"],
    toy_note_zh="概率平面画的是外层遗忘项加保留 KL。参数平面请把优化器设为正交投影，那才是双层约束的玩具。",
)

make_loss(
    _satimp,
    id="satimp",
    name="SatImp",
    name_zh="饱和-重要性重加权",
    family="unlearning",
    year=2025,
    latex=r"w(p)=p^{\beta_1}(1-p)^{\beta_2},\qquad \mathcal{L}_{\mathrm{SatImp}}=w(p_f)\log p_f",
    summary_zh="仍以梯度上升为底，但用当前概率给每个 token 加权。已经很小的概率权重大幅下降，更新自动停手。",
    intuition_zh="β1 越大，低概率 token 越快失去权重。β2 压低已经接近 1 的 token。中间概率的 token 承担大部分遗忘。",
    physics_zh="p^β1 log p 在 β1>0 时随 p→0 趋向 0，而不是趋向负无穷。logit 推力也随之消失。这和 NPO 的饱和是同一类几何，只是权重写在损失外面。",
    failure_zh="β1 过大时，稍一下降权重就没了，遗忘停在半途。权重如果还回传到概率本身，早期梯度会比停梯度的重加权更尖。",
    paper_title="Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning",
    paper_url="https://arxiv.org/abs/2505.11953",
    implementation_url="https://github.com/tmlr-group/SatImp",
    plane="unlearning",
    params=[
        ParamSpec("beta1", "饱和指数 beta1", 2.0, 0.2, 6.0, 0.1, "论文常用 5。过大时曲面会挤在很窄的区间。"),
        ParamSpec("beta2", "重要性指数 beta2", 0.5, 0.0, 2.0, 0.05, "论文仓库推荐可低至 0.1。"),
        ParamSpec("lam", "保留交叉熵权重", 0.0, 0.0, 2.0, 0.05),
    ],
    bounded_below=True,
    keywords=["SatImp", "reweighting"],
    toy_note_zh="权重写进标量损失，数值梯度会穿过权重。这对应没有对权重停梯度的实现。",
)
