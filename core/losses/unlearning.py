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
    summary_zh="把遗忘答案的交叉熵翻号，最小化就是在压低遗忘概率。",
    use_zh="把不该再出现的答案的概率往下压，是最简单的遗忘基线。用来对照别的方法到底忘得够不够。只想先看这个答案还会不会被说出来，并且接受模型可能说崩时，可以先试它。",
    intuition_zh="p_f 越小，损失越负。曲面向概率 0 的方向无限下降，没有坑底。",
    physics_zh="对 logit 的导数是 1−p。p 趋向 0 时导数趋向 1，推力不消失。保留概率不进这项。",
    failure_zh="遗忘集稍大或学习率稍高时，相邻 logit 一起被带走。",
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
    summary_zh="遗忘答案做梯度上升，保留答案继续做交叉熵。",
    use_zh="一边压低要遗忘的答案，一边在保留数据上继续做普通训练。想忘掉某一类问答，同时保住通用能力时用。手头既有遗忘样本，也有一批应该继续答对的样本时选它。",
    intuition_zh="损失沿 p_f 下降、沿 p_r 上升的方向变小。它同时要求忘掉一个答案、记住另一个答案。",
    physics_zh="遗忘通道的 logit 导数是 1−p_f，没有停止点；保留通道是 p_r−1，靠近 1 时停下。",
    failure_zh="保留集没覆盖到的相邻知识仍会被带走。训练久了，无界的遗忘项会淹没保留项。",
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
    summary_zh="遗忘答案做梯度上升，保留分布用 KL 锚在参考模型上。",
    use_zh="遗忘答案继续往下压，保留行为被拉回原来的模型，而不是一味把保留答案推到满分。有参考模型、希望保留集上的说法别漂太远时用。",
    intuition_zh="KL 的最低点在 p_r = p_ref，不像交叉熵那样把保留概率一直推向 1。",
    physics_zh="β 越大，保留通道越被拉回参考概率。遗忘通道在 p_f 趋向 0 时推力仍趋向 1。",
    failure_zh="两条通道的表示重叠时，KL 和梯度上升对着拉，两边都难以到位。",
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
    summary_zh="把遗忘回答当成落选回答，概率降下去之后梯度自己关掉。",
    use_zh="把要遗忘的回答当成偏好学习里的差回答，概率降下去之后推力自己变小。用来在忘掉目标的同时，避免一直推到模型崩溃。有参考模型、希望训练自己收手时用。",
    intuition_zh="p_f 远低于参考模型时，sigmoid 饱和，损失贴着 0。它不再像梯度上升那样越忘越有力。",
    physics_zh="logit 推力随 p 趋向 0 指数衰减到 0。β 越小，开始减速的区间越宽。保留交叉熵默认权重为 0。",
    failure_zh="参考模型把遗忘答案写得很深时，要降很多才饱和，早期更新仍接近满推力。",
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
    summary_zh="NPO 去掉参考模型，改用长度归一化的对数概率和一个间隔 γ。",
    use_zh="和 NPO 一样让遗忘推力自己停下来，但不需要参考模型。没有留下原模型、又希望遗忘损失有底时用。只有遗忘文本、凑不出参考分布时选它。",
    intuition_zh="没有 p_ref 可调。概率本身变小之后，logistic 仍然饱和，所以下界还在。",
    physics_zh="γ 平移饱和开始的位置，β 控制斜率。图上回答长度取 1。",
    failure_zh="没有参考模型时，所有答案的概率可以一起下降，只要遗忘答案相对更低。长度估计偏了，长短回答的遗忘强度不一致。",
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
    summary_zh="在替代答案上做交叉熵，原答案变低只是归一化的副作用。",
    use_zh="不直接踩原答案，而是拿一个替代说法做普通训练，让模型遇到同类问题时改口。版权角色、特定事实需要换成另一种叙述时用。已经准备好替代答案才能用。",
    intuition_zh="损失只关心替代答案在这对概率里的份额。原答案概率下降是归一化的副作用，不是无界的梯度上升。",
    physics_zh="纵轴是替代概率。最低点在原答案相对替代答案很小的方向，损失有下界 0，力来自份额。",
    failure_zh="没有替代答案时这项无从定义。替代句式和原知识相邻时，相邻事实会被一起改写。",
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
    summary_zh="遗忘位置的目标改成均匀分布，保留位置用参考模型做蒸馏。",
    use_zh="把要遗忘的位置训得在词表上差不多一样平，保留位置继续模仿原模型。没有现成的保留答案，也不想把遗忘提示训成一句固定拒答时用。",
    intuition_zh="均匀目标的交叉熵在 p_f = 1/|V| 处最低。原答案既不会被抬到 1，也不会被梯度上升到 0。",
    physics_zh="词表越大，最低点越靠近 0，但坑底仍在。图上用目标 token 对其余质量近似均匀目标。",
    failure_zh="模型可以在遗忘提示上变得很平，生成变成高熵噪声。",
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
    summary_zh="遗忘激活拉向一个随机方向，保留激活拉回原模型。",
    use_zh="不改词的概率，而改中间层表示：遗忘样本被拨向一个随机方向，保留样本被拉回原模型。危险知识、希望模型别再顺着原来的内部方向想下去时会用。改输出词容易误伤相邻知识时可以试它。",
    intuition_zh="概率平面上这是两个二次坑。遗忘通道的坑在噪声目标，保留通道的坑在参考概率。两个方向都有下界。",
    physics_zh="α 决定保留坑有多硬。图上把隐藏状态的平方距离压成两个概率通道的二次项，靠近目标梯度就下降。",
    failure_zh="随机方向仍能被读出时，模型会改口。噪声目标过大时，遗忘通道的幅度会变得异常。",
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
    summary_zh="用冻结的自己当教师，压低遗忘 token，并在其余通道上做自蒸馏。",
    use_zh="没有外部保留集。用冻结的自己当老师，把遗忘词压低一个间隔，其余通道继续模仿更新前的自己。只有遗忘样本、另找保留语料不方便时用。",
    intuition_zh="softplus 在学生已经低于教师一个间隔后梯度消失。自蒸馏把其余通道锚在更新前的模型上。",
    physics_zh="间隔 m 决定要低出多少才停，β 决定降权有多陡。教师冻结，损失不会跟着自己的概率滑向负无穷。",
    failure_zh="教师自己就会说出遗忘答案时，降权只是相对下移。自蒸馏权重太小，其余通道跟着漂移。",
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
    summary_zh="遗忘通道贴近安全目标，保留通道贴近参考模型，两边都是有上界的 JS 散度。",
    use_zh="遗忘通道靠近一个安全目标，保留通道靠近原模型，两边的力都有上限。希望忘得比较稳、又尽量保住原有能力时用。能给出安全目标，也还留着参考模型时选它。",
    intuition_zh="JS 散度落在 0 和 log 2 之间。目标达到以后梯度消失，离目标再远也不会像 KL 或 log p 那样把力放到无限大。",
    physics_zh="p_safe 和 p_ref 分别是两个坑底。散度落在 0 和 log 2 之间，到位以后梯度消失。",
    failure_zh="安全目标只是拒答时，模型可以换一种说法，事实仍在。",
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
    summary_zh="外层降低遗忘损失，内层先把保留损失修回来。步进时去掉沿保留梯度的分量。",
    use_zh="更新前先看保留方向，把遗忘更新里沿着这个方向的分量去掉，让遗忘少带动该留的知识。遗忘和保留缠在一起、直接相加会误伤时用。",
    intuition_zh="概率平面上仍能看到无界的遗忘项和一条保留 KL 弹簧。真正不同的是更新方向：步进前先去掉沿保留梯度的分量。",
    physics_zh="φ 越小，被去掉的分量越大，遗忘变慢，保留概率的漂移变小。概率平面画的是遗忘项加保留 KL。",
    failure_zh="投影只保护这一步的保留梯度。那个方向和要保护的知识不一致时，会保护错方向。遗忘项仍然没有下界。",
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
    summary_zh="在梯度上升上按当前概率加权。概率已经很小的 token，权重下降，更新停手。",
    use_zh="仍是把遗忘答案往下压，但概率已经很小的词权重也变小，更新会自己收手。想要梯度上升那种方向，又希望低概率处别再死命推时用。",
    intuition_zh="β1 越大，低概率 token 越快失去权重。β2 压低已经接近 1 的 token。中间概率的 token 承担大部分遗忘。",
    physics_zh="β1 越大，低概率处的损失越快回到 0，logit 推力随之消失。权重写进损失，梯度会穿过它。",
    failure_zh="β1 过大时，概率稍一下降权重就没了，遗忘停在半途。",
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
