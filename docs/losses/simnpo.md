# 简单负偏好优化 (SimNPO)

- 任务族: 机器遗忘
- 年份: 2024
- 下界: 有
- 论文: [Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](https://arxiv.org/abs/2410.07163)

## 公式

$$
\mathcal{L}_{\mathrm{SimNPO}}=-\frac{2}{\beta}\log\sigma\!\left(-\frac{\beta}{|y|}\log\pi_{\theta}(y_f)-\gamma\right)
$$

NPO 去掉参考模型，改用长度归一化的对数概率和一个间隔 gamma。

## 什么时候用

和 NPO 一样让遗忘推力自己停下来，但不需要参考模型。没有留下原模型、又希望遗忘损失有底时用。只有遗忘文本、凑不出参考分布时选它。

## 直觉

没有 p_ref 可调。概率本身变小之后，logistic 仍然饱和，所以下界还在。

## 物理意义

gamma 平移饱和开始的位置。beta 控制饱和斜率。二元玩具取回答长度为 1。它和 SimPO 的关系，相当于 NPO 和 DPO 的关系。

## 失效

没有参考模型时，模型可以把所有答案的概率一起压低，只要遗忘答案相对更低。长度估计偏了，长短回答的遗忘强度会不一致。

## 玩具坐标

长度归一化取 |y|=1。
