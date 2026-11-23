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
    summary_zh="学生去覆盖教师的全部质量。教师有概率的地方，学生不能放空。",
    intuition_zh="等高线只依赖 x:y 的比例。把两个通道同时放大不会改变损失，因为学生分布先被归一化。",
    physics_zh="正向 KL 是模式覆盖。学生若在教师的某个模式上接近 0，损失以对数速度上升，所以它倾向于把质量铺开。",
    failure_zh="教师自己是错的时候，正向 KL 会把错误模式也完整复制下来。遗忘里如果拿它当保留项，这是优点；拿它当遗忘项，则会继续覆盖不该保留的答案。",
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
    summary_zh="学生只需要把自己的质量放在教师也认可的模式上，可以丢掉教师的其他模式。",
    intuition_zh="反向 KL 惩罚学生放到教师低概率处的质量。它允许学生比教师更尖。",
    physics_zh="这是模式寻求。MiniLLM 用它蒸馏大模型，是为了避免正向 KL 把学生训得又平又散。力集中在学生当前的模式上。",
    failure_zh="学生可以合法地忽略教师的次模式。任务真的需要那些次模式时，反向 KL 会看起来损失很低，覆盖却不完整。",
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
    summary_zh="学生和教师一起对混合分布 m 做 KL。β=0.5 时这是对称的 JS 散度，损失落在 0 和 log 2 之间。",
    intuition_zh="m 同时含有两边的质量，所以一边接近 0 不会把另一边的对数项打到无穷。等高线仍只看学生两个通道的比例。",
    physics_zh="β 靠近 0 时，梯度方向接近正向 KL，学生要铺开去覆盖教师。β 靠近 1 时接近反向 KL，学生可以丢掉教师的次模式。两种情形下力都会饱和，不像 KL 在零概率处继续变陡。",
    failure_zh="β 只决定散度的形状，不解决自回归训练和推理看到的前缀不一样。GKD 还要在学生自己采样的序列上计算这条散度；二元图只保留散度本身。",
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
    summary_zh="用反向 KL 减去正向 KL，保留“把学生推离教师模式”的部分，抵消单纯覆盖带来的力。",
    intuition_zh="两个 KL 都非负，它们的差可正可负。学生比教师更尖时，反向项占优；学生为了覆盖而铺开时，正向项把差拉回来。",
    physics_zh="差值把模式寻求和模式覆盖拆开。遗忘实现里常把反向项用在遗忘分布上，把正向项用在保留分布上；这里先把两个方向放在同一对概率上。",
    failure_zh="差没有下界保证，读曲面时不能把负值理解成概率误差。序列上的 RKLD 还要按 token 求和，二元图只保留符号方向。",
    paper_title="MiniLLM: Knowledge Distillation of Large Language Models",
    paper_url="https://arxiv.org/abs/2306.08543",
    implementation_url="",
    plane="distillation",
    params=list(_TEACHER),
    bounded_below=False,
    keywords=["RKLD", "KL"],
    toy_note_zh="这是同一对学生分布上的 RKL-FKL，用来看两种 KL 的方向差。遗忘论文里的实现通常把两项分配到遗忘集和保留集。",
)
