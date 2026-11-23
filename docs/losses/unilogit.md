# 均匀目标自蒸馏 (Unilogit)

- 任务族: 机器遗忘
- 年份: 2025
- 下界: 有
- 论文: [Unilogit: Robust Machine Unlearning for LLMs Using Uniform-Target Self-Distillation](https://arxiv.org/abs/2505.06027)

## 公式

$$
\mathcal{L}_{\mathrm{Unilogit}}=\mathrm{CE}(p_{\theta}, U)+\lambda\, D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\theta})
$$

遗忘位置的目标改成均匀分布，保留位置用参考模型做自蒸馏。不需要外部保留答案。

## 直觉

均匀目标的交叉熵在 p_f = 1/|V| 处最低。原答案既不会被抬到 1，也不会被梯度上升到 0。

## 物理意义

词表越大，最低点越靠近 0，但坑底仍在。保留项是正向 KL，学生必须覆盖参考模型的模式，不能只把质量堆到一个拒答 token。

## 失效

均匀分布不是一个自然的拒答。模型可以在遗忘提示上变得很平，看起来像遗忘，生成质量却变成高熵噪声。

## 玩具坐标

遗忘通道用“目标 token 对其余质量”的二元交叉熵近似全词表均匀目标。
