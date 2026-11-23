# 组相对策略优化 (GRPO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 无
- 论文: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

## 公式

$$
\mathcal{L}_{\mathrm{GRPO}}=-\sum_i \hat A_i\log\pi_\theta(y_i)+\frac{\beta}{2}\left(\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)^2
$$

用同一组采样的标准化奖励当优势，不再训练单独的价值网络。

## 什么时候用

同一道题采样一组回答，用组内的相对高低更新策略，不必再训价值网络。数学、代码这类能自动判对错的题最常见。一道题能采多条、奖励可以放在一组里比较时用。

## 直觉

两组奖励一高一低时，优势大约是 +1 和 -1。策略项把高奖励回答的对数概率抬高，把低奖励回答压低，平方项把两者拉回参考模型。

## 物理意义

组内均值和标准差决定力的方向，β 把策略拉回参考模型。图上用平方对数比代替序列 KL。

## 失效

组内奖励没有差别时优势为 0，策略项消失。奖励只反映长度或格式时，标准化会把这个方向固定下来。

## 玩具坐标

平方对数比是序列 KL 的二元替身，用来保留“不要离开参考模型太远”这一项。

实现: https://github.com/huggingface/trl
