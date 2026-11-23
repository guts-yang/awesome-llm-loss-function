# UL2 span 交叉熵 (UL2 span CE)

- 任务族: 语言建模
- 年份: 2022
- 下界: 有
- 论文: [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)

## 公式

$$
\mathcal{L}_{\mathrm{UL2}}=-\log p_{\theta}(x_{\mathrm{span}}\mid x_{\setminus\mathrm{span}})
$$

UL2 把不同跨度的遮盖、前缀语言建模和普通下一 token 混在同一个交叉熵里。

## 直觉

损失的形状仍是负对数似然。UL2 改变的是被预测的 span 和任务混合比例，不是评分函数本身。

## 物理意义

因此它的 logit 推力和下一 token 交叉熵相同：目标概率靠近 1 时力消失。差别出现在批量里哪些位置产生这条力。

## 失效

如果把 UL2 理解成一种新的几何约束，会高估它。它解决的是去噪任务覆盖，不解决偏好冲突或遗忘无界。

## 玩具坐标

曲面上它与下一 token 交叉熵重合。这是故意保留的：两者的差异在数据构造，不在损失形状。
