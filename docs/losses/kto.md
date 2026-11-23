# 前景理论对齐 (KTO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 有
- 论文: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

## 公式

$$
\mathcal{L}_{\mathrm{KTO}}=\lambda_D\left(1-\sigma(\beta(r_\theta(y_+)-z_0))\right)+\lambda_U\left(1-\sigma(\beta(z_0-r_\theta(y_-)))\right)
$$

不需要成对偏好。好回答和坏回答各自相对一个 KL 参考点 z0 计算得失。

## 直觉

好回答的损失在奖励超过 z0 后饱和，坏回答的损失在奖励低于 z0 后饱和。两条通道不必来自同一次采样。

## 物理意义

z0 是势阱的分水岭。beta 决定饱和速度，lambda_d 与 lambda_u 决定两侧的力谁更大。

## 失效

z0 通常要用当前策略相对参考模型的 KL 来估计。把它固定错了，好回答和坏回答会一起被推向同一个方向。

## 玩具坐标

z0 在真实训练里是批内 KL 估计，这里做成可调常数。

实现: https://github.com/ContextualAI/HALOs
