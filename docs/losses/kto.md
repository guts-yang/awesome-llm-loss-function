# 前景理论对齐 (KTO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 有
- 论文: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

## 公式

$$
\mathcal{L}_{\mathrm{KTO}}=\lambda_D\left(1-\sigma(\beta(r_\theta(y_+)-z_0))\right)+\lambda_U\left(1-\sigma(\beta(z_0-r_\theta(y_-)))\right)
$$

好回答和坏回答各自相对参考点 z0 计算得失，不需要成对采样。

## 什么时候用

只有“这条好”或“这条差”，凑不成一对时用。点赞、点踩、有用或无用的日志都能直接拿来对齐。数据是单条评价，不是成对比较时选它。

## 直觉

好回答的损失在奖励超过 z0 后饱和，坏回答的损失在奖励低于 z0 后饱和。两条通道不必来自同一次采样。

## 物理意义

z0 是两侧力的分界，图上它是常数。β 决定饱和速度，两侧权重决定哪一边的力更大。

## 失效

z0 定错时，好回答和坏回答会被推向同一个方向。

## 玩具坐标

z0 在真实训练里是批内 KL 估计，这里做成可调常数。

实现: https://github.com/ContextualAI/HALOs
