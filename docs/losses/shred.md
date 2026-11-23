# 自蒸馏 logit 降权 (SHRED)

- 任务族: 机器遗忘
- 年份: 2026
- 下界: 有
- 论文: [SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion](https://arxiv.org/abs/2605.07482)

## 公式

$$
\mathcal{L}_{\mathrm{SHRED}}=\mathrm{softplus}\!\left(\beta\log\frac{p_{\theta}(y_f)}{p_{\mathrm{self}}(y_f)}+m\right)+\lambda\, D_{\mathrm{KL}}(p_{\mathrm{self}}\Vert p_{\theta})
$$

没有外部保留集。用冻结的自己当教师，压低遗忘 token 的 logit，同时在其余通道上做自蒸馏。

## 直觉

softplus 在学生已经低于教师一个间隔后梯度消失。自蒸馏把其余通道锚在更新前的模型上。

## 物理意义

margin 决定要低出多少才停。beta 决定降权有多陡。因为教师是冻结的，损失不会随着自己的概率一起滑向负无穷。

## 失效

教师本身就会输出遗忘答案时，降权只是相对这个教师下移一点，答案仍可能留在 top-k 里。自蒸馏权重太小，其余通道会跟着漂移。

## 玩具坐标

二元软间隔加自蒸馏是对论文机制的玩具，不是逐 token 的完整实现。
