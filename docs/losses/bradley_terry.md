# Bradley-Terry 奖励模型 (Bradley-Terry RM)

- 任务族: 偏好对齐
- 年份: 2017
- 下界: 有
- 论文: [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)

## 公式

$$
\mathcal{L}_{\mathrm{BT}}=-\log\sigma(\beta\log p_w-\beta\log p_l)
$$

奖励模型的成对损失。这里把奖励取成对数概率，用来对照后面的策略损失。

## 直觉

它和 gamma 为 0 的 SimPO 同形。差别只在它训练的是奖励头，不直接更新策略。

## 物理意义

sigmoid 把任意大的分差压进 (0,1)。分差继续扩大时梯度消失，所以奖励模型可以很有信心，同时不再提供训练信号。

## 失效

奖励如果没有长度和参考模型校正，会偏向高概率的短回答。PPO 再去追逐这个奖励时，偏差会被放大。

## 玩具坐标

奖励被直接取为对数概率，用来和 DPO 的隐式奖励对照。

实现: https://github.com/huggingface/trl
