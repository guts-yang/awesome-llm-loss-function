# 标签平滑 (Label smoothing)

- 任务族: 语言建模
- 年份: 2016
- 下界: 有
- 论文: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

## 公式

$$
\mathcal{L}_{\mathrm{LS}}=-(1-\epsilon)\log p_{\mathrm{target}}-\epsilon\log p_{\mathrm{other}}
$$

把一部分概率质量分给非目标 token，避免模型把全部质量压到一个 logit 上。

## 直觉

平滑项是在交叉熵旁边加一个对照 token 的对数惩罚。epsilon 越大，最优点越不愿意把目标概率推到 1。

## 物理意义

目标通道的 logit 梯度变成 (1-epsilon)(p_target-1) 与对照通道泄漏的叠加。它给最大似然加了一个软上限，后面 Unilogit 的均匀目标用的是同一类机制。

## 失效

平滑过强会抬高不该出现的 token，续写变得发虚。这里的对照项是二元简化，不是对整个词表做均匀平均。

## 玩具坐标

经典论文来自视觉网络，LLM 训练借用的是同一条平滑规则。图上只保留目标与一个对照 token。
