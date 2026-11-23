"""Preference-optimization and RLHF losses on a chosen/rejected probability plane."""

import numpy as np

from core.math_utils import clip_p, log_sigmoid, sigmoid, softplus
from core.schema import ParamSpec, make_loss


def _log_ratio(probability, reference):
    return np.log(clip_p(probability)) - np.log(clip_p(reference))


def _dpo(x, y, params):
    margin = params["beta"] * (_log_ratio(x, params["p_ref_w"]) - _log_ratio(y, params["p_ref_l"]))
    return -log_sigmoid(margin)


def _ipo(x, y, params):
    gap = _log_ratio(x, params["p_ref_w"]) - _log_ratio(y, params["p_ref_l"])
    return (gap - 1.0 / (2.0 * params["beta"])) ** 2


def _simpo(x, y, params):
    margin = params["beta"] * (np.log(clip_p(x)) - np.log(clip_p(y))) - params["gamma"]
    return -log_sigmoid(margin)


def _kto(x, y, params):
    desirable = np.log(clip_p(x)) - np.log(clip_p(params["p_ref_w"]))
    undesirable = np.log(clip_p(y)) - np.log(clip_p(params["p_ref_l"]))
    beta = params["beta"]
    chosen_term = 1.0 - sigmoid(beta * (desirable - params["z0"]))
    rejected_term = 1.0 - sigmoid(beta * (params["z0"] - undesirable))
    return params["lambda_d"] * chosen_term + params["lambda_u"] * rejected_term


def _orpo(x, y, params):
    odds_gap = np.log(clip_p(x) / (1.0 - clip_p(x))) - np.log(clip_p(y) / (1.0 - clip_p(y)))
    sft = -np.log(clip_p(x))
    return sft + params["lam"] * (-log_sigmoid(odds_gap))


def _slic(x, y, params):
    ranking = np.maximum(0.0, params["delta"] - (np.log(clip_p(x)) - np.log(clip_p(y))))
    return ranking + params["lam"] * (-np.log(clip_p(x)))


def _bradley_terry(x, y, params):
    margin = params["beta"] * (np.log(clip_p(x)) - np.log(clip_p(y)))
    return -log_sigmoid(margin)


def _ppo(x, y, params):
    epsilon = params["epsilon"]
    advantage = params["advantage"]

    def clipped(probability, old, signed_advantage):
        ratio = clip_p(probability) / clip_p(old)
        unclipped = ratio * signed_advantage
        clipped_advantage = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * signed_advantage
        return np.minimum(unclipped, clipped_advantage)

    chosen = clipped(x, params["p_old_w"], advantage)
    rejected = clipped(y, params["p_old_l"], -advantage)
    return -(chosen + rejected)


def _grpo(x, y, params):
    reward_w = params["reward_w"]
    reward_l = params["reward_l"]
    mean = 0.5 * (reward_w + reward_l)
    std = np.sqrt(0.5 * ((reward_w - mean) ** 2 + (reward_l - mean) ** 2)) + 1e-6
    advantage_w = (reward_w - mean) / std
    advantage_l = (reward_l - mean) / std
    policy = -(advantage_w * np.log(clip_p(x)) + advantage_l * np.log(clip_p(y)))
    anchor = 0.5 * params["beta"] * (_log_ratio(x, params["p_ref_w"]) ** 2 + _log_ratio(y, params["p_ref_l"]) ** 2)
    return policy + anchor


_REF = [
    ParamSpec("beta", "温度 beta", 0.1, 0.02, 1.0, 0.02, "越大，偏好间隔被放得越陡。"),
    ParamSpec("p_ref_w", "参考模型偏好概率", 0.55, 0.05, 0.95, 0.01),
    ParamSpec("p_ref_l", "参考模型落选概率", 0.45, 0.05, 0.95, 0.01),
]

make_loss(
    _dpo,
    id="dpo",
    name="DPO",
    name_zh="直接偏好优化",
    family="alignment",
    year=2023,
    latex=r"\mathcal{L}_{\mathrm{DPO}}=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right)",
    summary_zh="用偏好对直接训练策略，隐式奖励是相对参考模型的对数概率差。",
    intuition_zh="损失只看 chosen 与 rejected 的对数比。参考模型把这个比锚定在微调之前的策略上。",
    physics_zh="beta 是这个势阱的温度。参考概率升高时，同样的当前概率得到的奖励变小，等高线跟着平移。没有参考模型时，这条平移就消失了，那是 SimPO 的位置。",
    failure_zh="chosen 和 rejected 很接近时，对数比的梯度很尖，容易把概率质量从相邻的正确回答里挤走。",
    paper_title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    paper_url="https://arxiv.org/abs/2305.18290",
    implementation_url="https://github.com/eric-mitchell/direct-preference-optimization",
    plane="alignment",
    params=list(_REF),
    bounded_below=True,
    keywords=["DPO", "preference"],
    toy_note_zh="",
)

make_loss(
    _ipo,
    id="ipo",
    name="IPO",
    name_zh="身份偏好优化",
    family="alignment",
    year=2023,
    latex=r"\mathcal{L}_{\mathrm{IPO}}=\left(\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}-\frac{1}{2\beta}\right)^2",
    summary_zh="不使用 logistic，而是把对数概率间隔回归到一个固定目标 1/(2β)。",
    intuition_zh="DPO 的 sigmoid 在间隔已经很大时梯度消失。IPO 改成平方损失，间隔不足和间隔过大都会被拉回来。",
    physics_zh="目标间隔是 1/(2β)。β 越小，要求的对数间隔越大，势阱的最低点离对角线越远。",
    failure_zh="平方项两侧都有力。数据噪声会把已经分开的回答再拉近，欠拟合和过拟合表现为间隔偏离目标，而不是概率饱和。",
    paper_title="A General Theoretical Paradigm to Understand Learning from Human Preferences",
    paper_url="https://arxiv.org/abs/2310.12036",
    implementation_url="https://github.com/huggingface/trl",
    plane="alignment",
    params=list(_REF),
    bounded_below=True,
    keywords=["IPO", "preference"],
    toy_note_zh="",
)

make_loss(
    _simpo,
    id="simpo",
    name="SimPO",
    name_zh="简单偏好优化",
    family="alignment",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{SimPO}}=-\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l)-\gamma\right)",
    summary_zh="去掉参考模型，直接用长度归一化的对数概率差，再减去一个间隔 gamma。",
    intuition_zh="等高线只由 p_w 和 p_l 的比值决定。gamma 把决策边界从对角线推开，要求偏好回答高出一个固定余量。",
    physics_zh="与 DPO 相比，少了参考模型造成的平移。gamma 是纯间隔，beta 仍控制斜率。二元玩具里两条回答的长度都取 1。",
    failure_zh="没有参考模型时，策略可以同时压低两条回答的概率，只要偏好那条相对更高。长度归一化不准确时，长回答会被系统性压低。",
    paper_title="SimPO: Simple Preference Optimization with a Reference-Free Reward",
    paper_url="https://arxiv.org/abs/2405.14734",
    implementation_url="https://github.com/princeton-nlp/SimPO",
    plane="alignment",
    params=[
        ParamSpec("beta", "温度 beta", 2.0, 0.2, 5.0, 0.1),
        ParamSpec("gamma", "间隔 gamma", 0.5, 0.0, 2.0, 0.05, "偏好回答需要高出的对数余量。"),
    ],
    bounded_below=True,
    keywords=["SimPO", "preference"],
    toy_note_zh="长度归一化在二元玩具中取 |y|=1。",
)

make_loss(
    _kto,
    id="kto",
    name="KTO",
    name_zh="前景理论对齐",
    family="alignment",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{KTO}}=\lambda_D\left(1-\sigma(\beta(r_\theta(y_+)-z_0))\right)+\lambda_U\left(1-\sigma(\beta(z_0-r_\theta(y_-)))\right)",
    summary_zh="不需要成对偏好。好回答和坏回答各自相对一个 KL 参考点 z0 计算得失。",
    intuition_zh="好回答的损失在奖励超过 z0 后饱和，坏回答的损失在奖励低于 z0 后饱和。两条通道不必来自同一次采样。",
    physics_zh="z0 是势阱的分水岭。beta 决定饱和速度，lambda_d 与 lambda_u 决定两侧的力谁更大。",
    failure_zh="z0 通常要用当前策略相对参考模型的 KL 来估计。把它固定错了，好回答和坏回答会一起被推向同一个方向。",
    paper_title="KTO: Model Alignment as Prospect Theoretic Optimization",
    paper_url="https://arxiv.org/abs/2402.01306",
    implementation_url="https://github.com/ContextualAI/HALOs",
    plane="alignment",
    params=[
        ParamSpec("beta", "温度 beta", 0.1, 0.02, 1.0, 0.02),
        ParamSpec("z0", "参考点 z0", 0.0, -1.0, 1.0, 0.05),
        ParamSpec("lambda_d", "好回答权重", 1.0, 0.2, 2.0, 0.1),
        ParamSpec("lambda_u", "坏回答权重", 1.0, 0.2, 2.0, 0.1),
        ParamSpec("p_ref_w", "参考模型偏好概率", 0.55, 0.05, 0.95, 0.01),
        ParamSpec("p_ref_l", "参考模型落选概率", 0.45, 0.05, 0.95, 0.01),
    ],
    bounded_below=True,
    keywords=["KTO", "preference"],
    toy_note_zh="z0 在真实训练里是批内 KL 估计，这里做成可调常数。",
)

make_loss(
    _orpo,
    id="orpo",
    name="ORPO",
    name_zh="赔率比偏好优化",
    family="alignment",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{ORPO}}=-\log p_\theta(y_w)+\lambda\left(-\log\sigma\!\left(\log\frac{\mathrm{odds}(y_w)}{\mathrm{odds}(y_l)}\right)\right)",
    summary_zh="把监督微调的负对数似然和赔率比惩罚加在一起，不使用参考模型。",
    intuition_zh="第一项把偏好回答的概率往上拉，第二项要求它的赔率 p/(1-p) 高于落选回答。",
    physics_zh="监督项防止 SimPO 那种两条概率一起下降。赔率比在概率靠近 0 或 1 时比对数概率差更敏感。",
    failure_zh="lambda 过大时，赔率项会压过似然项，偏好回答的绝对概率反而下降。",
    paper_title="ORPO: Monolithic Preference Optimization without Reference Model",
    paper_url="https://arxiv.org/abs/2403.07691",
    implementation_url="https://github.com/xfactlab/orpo",
    plane="alignment",
    params=[ParamSpec("lam", "赔率权重 lambda", 1.0, 0.0, 4.0, 0.1)],
    bounded_below=True,
    keywords=["ORPO", "preference"],
    toy_note_zh="",
)

make_loss(
    _slic,
    id="slic",
    name="SLiC-HF",
    name_zh="序列似然校准",
    family="alignment",
    year=2023,
    latex=r"\mathcal{L}_{\mathrm{SLiC}}=\max(0,\delta-\log\pi(y_w)+\log\pi(y_l))-\lambda\log\pi(y_w)",
    summary_zh="用铰链间隔把偏好回答的对数概率抬到落选回答之上，并用似然项保住流畅度。",
    intuition_zh="间隔已经大于 delta 时，排序项的力是 0，只剩监督项。间隔不够时，两条回答受到方向相反的常数力。",
    physics_zh="铰链造成等高线上的折线。delta 决定折线离对角线多远，lambda 决定偏好回答那一侧还有多强的似然拉力。",
    failure_zh="铰链在间隔满足后完全停力，不能继续纠正校准。delta 选得比数据里的自然间隔还大时，模型会过度压低落选回答。",
    paper_title="SLiC-HF: Sequence Likelihood Calibration with Human Feedback",
    paper_url="https://arxiv.org/abs/2305.10425",
    implementation_url="https://github.com/huggingface/trl",
    plane="alignment",
    params=[
        ParamSpec("delta", "间隔 delta", 0.5, 0.0, 2.0, 0.05),
        ParamSpec("lam", "似然权重 lambda", 0.1, 0.0, 1.0, 0.05),
    ],
    bounded_below=True,
    keywords=["SLiC", "preference"],
    toy_note_zh="",
)

make_loss(
    _bradley_terry,
    id="bradley_terry",
    name="Bradley-Terry RM",
    name_zh="Bradley-Terry 奖励模型",
    family="alignment",
    year=2017,
    latex=r"\mathcal{L}_{\mathrm{BT}}=-\log\sigma(\beta\log p_w-\beta\log p_l)",
    summary_zh="奖励模型的成对损失。这里把奖励取成对数概率，用来对照后面的策略损失。",
    intuition_zh="它和 gamma 为 0 的 SimPO 同形。差别只在它训练的是奖励头，不直接更新策略。",
    physics_zh="sigmoid 把任意大的分差压进 (0,1)。分差继续扩大时梯度消失，所以奖励模型可以很有信心，同时不再提供训练信号。",
    failure_zh="奖励如果没有长度和参考模型校正，会偏向高概率的短回答。PPO 再去追逐这个奖励时，偏差会被放大。",
    paper_title="Deep Reinforcement Learning from Human Preferences",
    paper_url="https://arxiv.org/abs/1706.03741",
    implementation_url="https://github.com/huggingface/trl",
    plane="alignment",
    params=[ParamSpec("beta", "奖励缩放 beta", 1.0, 0.2, 4.0, 0.1)],
    bounded_below=True,
    keywords=["Bradley-Terry", "reward model", "RLHF"],
    toy_note_zh="奖励被直接取为对数概率，用来和 DPO 的隐式奖励对照。",
)

make_loss(
    _ppo,
    id="ppo_clip",
    name="PPO clip",
    name_zh="PPO 裁剪目标",
    family="alignment",
    year=2017,
    latex=r"\mathcal{L}_{\mathrm{PPO}}=-\mathbb{E}\left[\min(r_t A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]",
    summary_zh="用新旧策略的概率比裁剪优势，限制一次更新离开旧策略多远。",
    intuition_zh="优势为正时，概率比超过 1+epsilon 就不再加分。优势为负时，概率比低于 1-epsilon 就不再减分。",
    physics_zh="裁剪在概率平面上切出一块平坦的信任区域。epsilon 是这块区域的宽度。偏好回答用正优势，落选回答用负优势。",
    failure_zh="裁剪只限制概率比，不直接限制 KL。旧策略概率估计偏了，信任区域会切错位置。这里也没有单独画 KL 奖励惩罚。",
    paper_title="Proximal Policy Optimization Algorithms",
    paper_url="https://arxiv.org/abs/1707.06347",
    implementation_url="https://github.com/huggingface/trl",
    plane="alignment",
    params=[
        ParamSpec("epsilon", "裁剪宽度 epsilon", 0.2, 0.05, 0.5, 0.05),
        ParamSpec("advantage", "优势 A", 1.0, 0.2, 3.0, 0.1),
        ParamSpec("p_old_w", "旧策略偏好概率", 0.45, 0.05, 0.95, 0.01),
        ParamSpec("p_old_l", "旧策略落选概率", 0.4, 0.05, 0.95, 0.01),
    ],
    bounded_below=False,
    keywords=["PPO", "RLHF"],
    toy_note_zh="二元动作上的裁剪目标。InstructGPT 的完整目标还包含相对参考模型的 KL 惩罚。",
)

make_loss(
    _grpo,
    id="grpo",
    name="GRPO",
    name_zh="组相对策略优化",
    family="alignment",
    year=2024,
    latex=r"\mathcal{L}_{\mathrm{GRPO}}=-\sum_i \hat A_i\log\pi_\theta(y_i)+\frac{\beta}{2}\left(\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)^2",
    summary_zh="同一道题的一组采样互相比较，用组内标准化奖励当优势，不再训练单独的价值网络。",
    intuition_zh="两组奖励一高一低时，优势大约是 +1 和 -1。策略项把高奖励回答的对数概率抬高，把低奖励回答压低，平方项把两者拉回参考模型。",
    physics_zh="组内均值和标准差决定力的方向。beta 控制回到参考模型的弹簧有多硬。二元玩具用一个平方对数比代替序列 KL。",
    failure_zh="组内奖励没有差别时优势为 0，策略项消失。奖励只反映格式或长度时，标准化会把这个假信号放大成稳定的更新方向。",
    paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
    paper_url="https://arxiv.org/abs/2402.03300",
    implementation_url="https://github.com/huggingface/trl",
    plane="alignment",
    params=[
        ParamSpec("beta", "KL 权重 beta", 0.04, 0.0, 0.5, 0.01),
        ParamSpec("reward_w", "偏好回答奖励", 1.0, -1.0, 2.0, 0.1),
        ParamSpec("reward_l", "落选回答奖励", 0.0, -1.0, 2.0, 0.1),
        ParamSpec("p_ref_w", "参考模型偏好概率", 0.5, 0.05, 0.95, 0.01),
        ParamSpec("p_ref_l", "参考模型落选概率", 0.5, 0.05, 0.95, 0.01),
    ],
    bounded_below=False,
    keywords=["GRPO", "RL"],
    toy_note_zh="平方对数比是序列 KL 的二元替身，用来保留“不要离开参考模型太远”这一项。",
)
