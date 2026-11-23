# 序列似然校准 (SLiC-HF)

- 任务族: 偏好对齐
- 年份: 2023
- 下界: 有
- 论文: [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)

## 公式

$$
\mathcal{L}_{\mathrm{SLiC}}=\max(0,\delta-\log\pi(y_w)+\log\pi(y_l))-\lambda\log\pi(y_w)
$$

用铰链把偏好回答的对数概率抬到落选回答之上，并用似然项保住它的概率。

## 什么时候用

要求好回答高出差回答一个间隔，间隔够了就不再加力，同时保住句子通顺。做偏好对齐、希望排序到位后就停手时用。没有参考模型，又担心对齐把流畅度训坏时选它。

## 直觉

间隔已经大于 delta 时，排序项的力是 0，只剩监督项。间隔不够时，两条回答受到方向相反的常数力。

## 物理意义

δ 决定铰链折线离对角线多远，λ 决定折线之外还剩多少似然拉力。

## 失效

间隔已经大于 δ 时排序项停力。δ 大于数据里的自然间隔时，落选回答会被过度压低。

实现: https://github.com/huggingface/trl
