# Bradley-Terry 奖励模型 (Bradley-Terry RM)

- 任务族: 偏好对齐
- 年份: 2017
- 下界: 有
- 论文: [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)

## 公式

$$
\mathcal{L}_{\mathrm{BT}}=-\log\sigma(\beta\log p_w-\beta\log p_l)
$$

用成对比较训练奖励，奖励直接取成对数概率差。

## 什么时候用

先训练一个打分器，判断人更喜欢一对回答里的哪一条。它不直接改写生成模型，而是给后面的强化学习提供奖励。有大量成对偏好、需要给回答打分或准备跑 PPO 时用。

## 直觉

它和 gamma 为 0 的 SimPO 同形。差别只在它训练的是奖励头，不直接更新策略。

## 物理意义

β 把分差放大。分差继续扩大时 sigmoid 梯度消失，奖励可以很有信心，同时不再提供训练信号。

## 失效

奖励没有长度校正时，会偏向高概率的短回答。

## 玩具坐标

奖励被直接取为对数概率，用来和 DPO 的隐式奖励对照。

实现: https://github.com/huggingface/trl
