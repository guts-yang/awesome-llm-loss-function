# 表示错向 (RMU)

- 任务族: 机器遗忘
- 年份: 2024
- 下界: 有
- 论文: [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning](https://arxiv.org/abs/2403.03218)

## 公式

$$
\mathcal{L}_{\mathrm{RMU}}=\lVert h_f-c\,u\rVert_2^2+\alpha\lVert h_r-h_{\mathrm{ref}}\rVert_2^2
$$

WMDP 的表示错向：遗忘激活被拉向一个随机方向，保留激活被拉回原模型。

## 直觉

概率平面上这是两个二次坑。遗忘通道的坑在噪声目标，保留通道的坑在参考概率。两个方向都有下界。

## 物理意义

alpha 决定保留弹簧有多硬。二次损失的梯度随误差线性下降，靠近目标就停，不像 log p 在边界上维持满推力。参数平面里，这两个坑分别沿遗忘特征和保留特征打开。

## 失效

随机方向如果落进仍被读出的子空间，模型会改口而不是遗忘。系数 c 过大时，遗忘层的激活幅度异常，下游层会跟着失真。

## 玩具坐标

隐藏状态上的平方距离被压成两个概率通道上的二次项，用来保留“两边都有坑底”这一几何。

实现: https://github.com/centerforaisafety/wmdp
