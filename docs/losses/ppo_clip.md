# PPO 裁剪目标 (PPO clip)

- 任务族: 偏好对齐
- 年份: 2017
- 下界: 无
- 论文: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## 公式

$$
\mathcal{L}_{\mathrm{PPO}}=-\mathbb{E}\left[\min(r_t A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]
$$

用新旧策略的概率比裁剪优势，限制一次更新离开旧策略多远。

## 什么时候用

模型自己采样回答，用奖励打分，同时限制这一步不要离旧策略太远。这是 RLHF 里奖励模型训好之后的策略更新。需要在线采样、又怕一次更新把模型带飞时用。

## 直觉

优势为正时，概率比超过 1+epsilon 就不再加分。优势为负时，概率比低于 1-epsilon 就不再减分。

## 物理意义

ε 是信任区域的宽度。优势为正时，概率比超过 1+ε 不再加分；优势为负时，低于 1−ε 不再减分。

## 失效

裁剪只限制概率比。旧策略概率估计偏了，信任区域会切错位置。图上没有相对参考模型的 KL 惩罚。

## 玩具坐标

二元动作上的裁剪目标。InstructGPT 的完整目标还包含相对参考模型的 KL 惩罚。

实现: https://github.com/huggingface/trl
