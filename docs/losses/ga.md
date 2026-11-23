# 梯度上升 (Gradient Ascent)

- 任务族: 机器遗忘
- 年份: 2024
- 下界: 无
- 论文: [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)

## 公式

$$
\mathcal{L}_{\mathrm{GA}}=\log p_{\theta}(y_f\mid x)
$$

把遗忘答案的交叉熵翻号。最小化这个损失就是在降低遗忘答案的概率。

## 直觉

p_f 越小，损失越负。曲面向概率 0 的方向无限下降，没有坑底。

## 物理意义

令 p=σ(z)。对 logit 的导数是 1-p。p 趋向 0 时导数趋向 1，推力停在最大值，不会自己关掉。优化器可以一直把这个 token 往下压，并把纠缠在一起的相邻 logit 带走。

## 失效

遗忘集稍大、学习率稍高，模型就会塌成拒答、复读或胡言。TOFU 一类基准里它是强遗忘、差效用的参照。

## 玩具坐标

纵轴不进入损失。能力保持要靠别的项，或者靠参数平面上的投影。

实现: https://github.com/locuslab/tofu
