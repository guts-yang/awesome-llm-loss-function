# 负偏好优化 (NPO)

- 任务族: 机器遗忘
- 年份: 2024
- 下界: 有
- 论文: [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)

## 公式

$$
\mathcal{L}_{\mathrm{NPO}}=-\frac{2}{\beta}\log\sigma\!\left(-\beta\log\frac{\pi_{\theta}(y_f)}{\pi_{\mathrm{ref}}(y_f)}\right)
$$

把遗忘回答当成偏好优化里的落选回答，让遗忘损失在概率变低后自动失去梯度。

## 直觉

p_f 远低于参考模型时，sigmoid 饱和，损失贴着 0。它不再像梯度上升那样越忘越有力。

## 物理意义

对 logit 的推力正比于 1-σ(-β log(p/p_ref))，再乘上 (1-p)。p 趋向 0 时第一项指数衰减，推力归零。β 越小，开始减速的区间越宽。图上的保留交叉熵是可选锚，权重可以调到 0。

## 失效

参考模型如果已经把遗忘答案写得很深，log(p/p_ref) 要降很多才饱和，早期更新仍接近梯度上升。只调 β 不能解决保留集里没有覆盖到的相邻事实。

## 玩具坐标

公式主体是论文中的遗忘项。保留交叉熵是可选附加项，默认权重为 0。

实现: https://github.com/licong-lin/negative-preference-optimization
