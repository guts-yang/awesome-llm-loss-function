# 标签平滑 (Label smoothing)

- 任务族: 语言建模
- 年份: 2016
- 下界: 有
- 论文: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

## 公式

$$
\mathcal{L}_{\mathrm{LS}}=-(1-\epsilon)\log p_{\mathrm{target}}-\epsilon\log p_{\mathrm{other}}
$$

把一部分概率质量分给非目标 token，避免全部质量压到一个 logit 上。

## 什么时候用

给正确答案以外的词留一点概率，避免模型把信心全部压在唯一一个词上。近义说法不止一种时常用。模型已经把某个词的概率顶满、续写过分笃定时，加上它。

## 直觉

平滑项是在交叉熵旁边加一个对照 token 的对数惩罚。epsilon 越大，最优点越不愿意把目标概率推到 1。

## 物理意义

ε 越大，最优点越不愿意把目标概率推到 1。图上的对照项只代表一个非目标 token。

## 失效

ε 过大时，不该出现的 token 被抬高，续写发虚。

## 玩具坐标

经典论文来自视觉网络，LLM 训练借用的是同一条平滑规则。图上只保留目标与一个对照 token。
