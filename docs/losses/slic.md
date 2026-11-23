# 序列似然校准 (SLiC-HF)

- 任务族: 偏好对齐
- 年份: 2023
- 下界: 有
- 论文: [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)

## 公式

$$
\mathcal{L}_{\mathrm{SLiC}}=\max(0,\delta-\log\pi(y_w)+\log\pi(y_l))-\lambda\log\pi(y_w)
$$

用铰链间隔把偏好回答的对数概率抬到落选回答之上，并用似然项保住流畅度。

## 直觉

间隔已经大于 delta 时，排序项的力是 0，只剩监督项。间隔不够时，两条回答受到方向相反的常数力。

## 物理意义

铰链造成等高线上的折线。delta 决定折线离对角线多远，lambda 决定偏好回答那一侧还有多强的似然拉力。

## 失效

铰链在间隔满足后完全停力，不能继续纠正校准。delta 选得比数据里的自然间隔还大时，模型会过度压低落选回答。

实现: https://github.com/huggingface/trl
