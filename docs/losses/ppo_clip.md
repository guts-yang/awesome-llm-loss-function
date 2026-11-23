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

## 直觉

优势为正时，概率比超过 1+epsilon 就不再加分。优势为负时，概率比低于 1-epsilon 就不再减分。

## 物理意义

裁剪在概率平面上切出一块平坦的信任区域。epsilon 是这块区域的宽度。偏好回答用正优势，落选回答用负优势。

## 失效

裁剪只限制概率比，不直接限制 KL。旧策略概率估计偏了，信任区域会切错位置。这里也没有单独画 KL 奖励惩罚。

## 玩具坐标

二元动作上的裁剪目标。InstructGPT 的完整目标还包含相对参考模型的 KL 惩罚。

实现: https://github.com/huggingface/trl
