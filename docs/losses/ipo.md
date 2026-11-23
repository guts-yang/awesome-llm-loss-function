# 身份偏好优化 (IPO)

- 任务族: 偏好对齐
- 年份: 2023
- 下界: 有
- 论文: [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)

## 公式

$$
\mathcal{L}_{\mathrm{IPO}}=\left(\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}-\frac{1}{2\beta}\right)^2
$$

不使用 logistic，而是把对数概率间隔回归到一个固定目标 1/(2β)。

## 直觉

DPO 的 sigmoid 在间隔已经很大时梯度消失。IPO 改成平方损失，间隔不足和间隔过大都会被拉回来。

## 物理意义

目标间隔是 1/(2β)。β 越小，要求的对数间隔越大，势阱的最低点离对角线越远。

## 失效

平方项两侧都有力。数据噪声会把已经分开的回答再拉近，欠拟合和过拟合表现为间隔偏离目标，而不是概率饱和。

实现: https://github.com/huggingface/trl
