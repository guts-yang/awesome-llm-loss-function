# 身份偏好优化 (IPO)

- 任务族: 偏好对齐
- 年份: 2023
- 下界: 有
- 论文: [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)

## 公式

$$
\mathcal{L}_{\mathrm{IPO}}=\left(\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}-\frac{1}{2\beta}\right)^2
$$

把相对参考模型的对数概率间隔回归到固定目标 1/(2β)。

## 什么时候用

同样吃偏好对，但把两条回答的差距拉到一个固定距离，而不是越大越好。偏好数据有噪声、差距被拉得过猛时用。有参考模型，又担心好回答和差回答被撕得太开时选它。

## 直觉

DPO 的 sigmoid 在间隔已经很大时梯度消失。IPO 改成平方损失，间隔不足和间隔过大都会被拉回来。

## 物理意义

β 越小，要求的对数间隔越大，最低点离对角线越远。

## 失效

间隔已经够大时平方项仍往回拉，噪声会把分开的回答再拉近。

实现: https://github.com/huggingface/trl
