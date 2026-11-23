# 组相对策略优化 (GRPO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 无
- 论文: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

## 公式

$$
\mathcal{L}_{\mathrm{GRPO}}=-\sum_i \hat A_i\log\pi_\theta(y_i)+\frac{\beta}{2}\left(\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)^2
$$

同一道题的一组采样互相比较，用组内标准化奖励当优势，不再训练单独的价值网络。

## 直觉

两组奖励一高一低时，优势大约是 +1 和 -1。策略项把高奖励回答的对数概率抬高，把低奖励回答压低，平方项把两者拉回参考模型。

## 物理意义

组内均值和标准差决定力的方向。beta 控制回到参考模型的弹簧有多硬。二元玩具用一个平方对数比代替序列 KL。

## 失效

组内奖励没有差别时优势为 0，策略项消失。奖励只反映格式或长度时，标准化会把这个假信号放大成稳定的更新方向。

## 玩具坐标

平方对数比是序列 KL 的二元替身，用来保留“不要离开参考模型太远”这一项。

实现: https://github.com/huggingface/trl
