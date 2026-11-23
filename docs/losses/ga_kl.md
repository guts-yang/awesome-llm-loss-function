# 梯度上升加 KL (GA + KL)

- 任务族: 机器遗忘
- 年份: 2024
- 下界: 无
- 论文: [Machine Unlearning of Pre-trained Large Language Models](https://arxiv.org/abs/2402.15159)

## 公式

$$
\mathcal{L}_{\mathrm{GA+KL}}=\log p_{\theta}(y_f)+\beta\, D_{\mathrm{KL}}\!\left(p_{\theta}(\cdot\mid x_r)\Vert p_{\mathrm{ref}}(\cdot\mid x_r)\right)
$$

遗忘答案仍做梯度上升，保留分布用 KL 锚在参考模型上。

## 什么时候用

遗忘答案继续往下压，保留行为被拉回原来的模型，而不是一味把保留答案推到满分。有参考模型、希望保留集上的说法别漂太远时用。

## 直觉

KL 的最低点在 p_r = p_ref，不像交叉熵那样把保留概率一直推向 1。

## 物理意义

beta 是弹簧刚度。它限制保留通道离开原模型的距离，但不改变遗忘通道在 p_f→0 时推力趋向 1 的事实。

## 失效

遗忘集和保留集的表示重叠时，KL 锚会和梯度上升对着拉，结果常常是两边都没到位，或者学习率一大仍从遗忘通道崩掉。

## 玩具坐标

KL 写在伯努利保留通道上，对应序列实现里保留提示上的 token KL。

实现: https://github.com/locuslab/tofu
