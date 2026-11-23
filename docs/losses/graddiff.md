# 梯度差 (GradDiff)

- 任务族: 机器遗忘
- 年份: 2024
- 下界: 无
- 论文: [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)

## 公式

$$
\mathcal{L}_{\mathrm{GradDiff}}=\log p_{\theta}(y_f\mid x)-\log p_{\theta}(y_r\mid x)
$$

遗忘集上做梯度上升，保留集上继续做交叉熵。

## 直觉

损失沿 p_f 下降、沿 p_r 上升的方向变小。它同时要求忘掉一个答案、记住另一个答案。

## 物理意义

两条力分别是 1-p_f 和 p_r-1 的 logit 梯度。保留项有下界 0 附近的停止点，遗忘项没有。合力和两条任务的特征夹角有关，夹角小的时候保留梯度抵消不了遗忘梯度在相邻方向上的分量。

## 失效

保留集覆盖不到的相邻知识仍会被带走。遗忘项无界，训练久了保留项也会被淹没。

实现: https://github.com/locuslab/tofu
