"""Forward KL, reverse KL, Jensen-Shannon, and their difference on a two-class student."""

import numpy as np

from core.math_utils import binary_kl, clip_p
from core.schema import ParamSpec, make_loss


def _student(x, y):
    total = clip_p(x) + clip_p(y)
    return clip_p(x) / total, clip_p(y) / total


def _teacher(params):
    teacher = clip_p(params["teacher"])
    return teacher, 1.0 - teacher


def _forward_kl(x, y, params):
    student_x, student_y = _student(x, y)
    teacher_x, teacher_y = _teacher(params)
    return teacher_x * np.log(teacher_x / student_x) + teacher_y * np.log(teacher_y / student_y)


def _reverse_kl(x, y, params):
    student_x, student_y = _student(x, y)
    teacher_x, teacher_y = _teacher(params)
    return student_x * np.log(student_x / teacher_x) + student_y * np.log(student_y / teacher_y)


def _rkld(x, y, params):
    return _reverse_kl(x, y, params) - _forward_kl(x, y, params)


def _jsd(x, y, params):
    student_x, _ = _student(x, y)
    teacher_x, _ = _teacher(params)
    beta = float(params["beta"])
    mixture = beta * teacher_x + (1.0 - beta) * student_x
    return beta * binary_kl(teacher_x, mixture) + (1.0 - beta) * binary_kl(student_x, mixture)


_TEACHER = [ParamSpec("teacher", "教师在第一类上的概率", 0.8, 0.05, 0.95, 0.01)]
_JS_PARAMS = _TEACHER + [
    ParamSpec(
        "beta",
        "混合系数 beta",
        0.5,
        0.05,
        0.95,
        0.05,
        "0.5 是对称 JS。靠近 0 时梯度像正向 KL，靠近 1 时像反向 KL。",
    )
]

make_loss(
    _forward_kl,
    id="forward_kl",
    name="Forward KL",
    name_zh="正向 KL",
    family="distillation",
    year=2015,
    latex=r"D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert p_{\theta})=\sum_t p_{\mathrm{teacher}}(t)\log\frac{p_{\mathrm{teacher}}(t)}{p_{\theta}(t)}",
    summary_zh="学生要覆盖教师的全部质量。教师有概率的地方，学生不能放空。",
    use_zh="把教师的整份概率交给学生，教师给过概率的地方学生都不能空着。模型压缩、草稿模型，以及遗忘里把保留行为锚回原模型，都会用到它。学生容量够、希望多种说法都继承下来时选它。",
    intuition_zh="等高线只依赖 x:y 的比例。把两个通道同时放大不会改变损失，因为学生分布先被归一化。",
    physics_zh="学生在教师的某个模式上接近 0 时，损失按对数上升，质量被铺开。两个通道先归一化，对角线方向是平的。",
    failure_zh="教师错了，错误模式也会被完整复制下来。",
    paper_title="Distilling the Knowledge in a Neural Network",
    paper_url="https://arxiv.org/abs/1503.02531",
    implementation_url="",
    plane="distillation",
    params=list(_TEACHER),
    bounded_below=True,
    keywords=["forward KL", "distillation"],
    toy_note_zh="二元分布先做归一化，所以曲面对角线方向是平的。",
)

make_loss(
    _reverse_kl,
    id="reverse_kl",
    name="Reverse KL",
    name_zh="反向 KL",
    family="distillation",
    year=2023,
    latex=r"D_{\mathrm{KL}}(p_{\theta}\Vert p_{\mathrm{teacher}})=\sum_t p_{\theta}(t)\log\frac{p_{\theta}(t)}{p_{\mathrm{teacher}}(t)}",
    summary_zh="学生只把自己的质量放在教师也认可的模式上，可以丢掉教师的其他模式。",
    use_zh="学生只把概率放在教师也认可的说法上，可以丢掉教师的次要说法。用来把大模型蒸馏成更会生成、不那么平均的小模型。学生比教师小很多、生成发虚发平时用。",
    intuition_zh="反向 KL 惩罚学生放到教师低概率处的质量。它允许学生比教师更尖。",
    physics_zh="惩罚落在教师低概率处的学生质量上，力集中在学生当前的模式。",
    failure_zh="任务需要教师的次模式时，损失可以很低，覆盖却不完整。",
    paper_title="MiniLLM: Knowledge Distillation of Large Language Models",
    paper_url="https://arxiv.org/abs/2306.08543",
    implementation_url="",
    plane="distillation",
    params=list(_TEACHER),
    bounded_below=True,
    keywords=["reverse KL", "distillation"],
    toy_note_zh="",
)

make_loss(
    _jsd,
    id="jsd",
    name="Jensen-Shannon",
    name_zh="JS 散度",
    family="distillation",
    year=2024,
    latex=r"m=\beta p_{\mathrm{teacher}}+(1-\beta)p_{\theta},\quad \mathrm{JSD}_{\beta}(p_{\mathrm{teacher}}\Vert p_{\theta})=\beta D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert m)+(1-\beta)D_{\mathrm{KL}}(p_{\theta}\Vert m)",
    summary_zh="学生对混合分布 m 做 KL。β=0.5 时损失落在 0 和 log 2 之间。",
    use_zh="用有上限的 JS 散度比较学生和教师，介于全覆盖和只追一个说法之间。GKD 用它做在线蒸馏：学生按自己生成的句子去问教师该怎么改。希望蒸馏比较稳，或学生会走到教师没见过的前缀上时用。",
    intuition_zh="m 同时含有两边的质量，所以一边接近 0 不会把另一边的对数项打到无穷。等高线仍只看学生两个通道的比例。",
    physics_zh="β 靠近 0 时学生要铺开覆盖教师，靠近 1 时可以丢掉教师的次模式。一端接近 0 时力不再继续变陡。",
    failure_zh="损失有上界。覆盖还没完成时，力也可以先饱和。",
    paper_title="On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
    paper_url="https://arxiv.org/abs/2306.13649",
    implementation_url="",
    plane="distillation",
    params=list(_JS_PARAMS),
    bounded_below=True,
    keywords=["Jensen-Shannon", "JSD", "distillation"],
    toy_note_zh="β=0.5 是 Lin 的对称 JS。广义形式来自 GKD，m 是教师和学生的凸组合。序列上的实现还要对每个 token 取平均。",
)

make_loss(
    _rkld,
    id="rkld",
    name="RKLD",
    name_zh="反向减正向 KL",
    family="distillation",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{RKLD}}=D_{\mathrm{KL}}(p_{\theta}\Vert p_{\mathrm{ref}})-D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\theta})",
    summary_zh="用反向 KL 减去正向 KL。学生更尖时差为正，学生为了覆盖而铺开时差被拉回。",
    use_zh="留下把学生推离教师不想要的说法的那股力。遗忘里常把反向项放在要忘掉的分布上，正向项放在要保留的分布上。两种方向需要拆开、而不是简单相加时用。",
    intuition_zh="两个 KL 都非负，它们的差可正可负。学生比教师更尖时，反向项占优；学生为了覆盖而铺开时，正向项把差拉回来。",
    physics_zh="两项放在同一对概率上。差可正可负，负值不是概率误差。",
    failure_zh="差没有下界。图上只保留这一对概率的符号方向。",
    paper_title="MiniLLM: Knowledge Distillation of Large Language Models",
    paper_url="https://arxiv.org/abs/2306.08543",
    implementation_url="",
    plane="distillation",
    params=list(_TEACHER),
    bounded_below=False,
    keywords=["RKLD", "KL"],
    toy_note_zh="这是同一对学生分布上的 RKL-FKL，用来看两种 KL 的方向差。遗忘论文里的实现通常把两项分配到遗忘集和保留集。",
)
