# Jensen-Shannon 遗忘 (JensUn)

- 任务族: 机器遗忘
- 年份: 2025
- 下界: 有
- 论文: [Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs](https://arxiv.org/abs/2509.02820)

## 公式

$$
\mathcal{L}_{\mathrm{JensUn}}=\lambda_F\,\mathrm{JSD}(p_{\theta}\Vert p_{\mathrm{safe}})+\lambda_R\,\mathrm{JSD}(p_{\theta}\Vert p_{\mathrm{ref}})
$$

遗忘通道去贴近一个安全目标，保留通道去贴近参考模型。两边都用有上界的 Jensen-Shannon 散度。

## 什么时候用

遗忘通道靠近一个安全目标，保留通道靠近原模型，两边的力都有上限。希望忘得比较稳、又尽量保住原有能力时用。能给出安全目标，也还留着参考模型时选它。

## 直觉

JS 散度落在 0 和 log 2 之间。目标达到以后梯度消失，离目标再远也不会像 KL 或 log p 那样把力放到无限大。

## 物理意义

p_safe 是遗忘通道的坑底，通常是一个很小的概率或拒答 token。p_ref 是保留通道的坑底。两个二次型附近的梯度和 JS 的饱和一起，使更新在结束时自动变小。

## 失效

安全目标如果只是空 token 或拒答，模型学会的可能是换一种说法，而不是丢掉那个事实。JS 对两个分布的中间点对称，早期梯度比梯度上升温和，遗忘速度也更慢。

## 玩具坐标

论文在序列上对拒绝目标的 one-hot 和参考分布分别计算 JSD。这里把两个目标放进遗忘、保留两个伯努利通道。

实现: https://github.com/nmndeep/JensUn-Unlearning
