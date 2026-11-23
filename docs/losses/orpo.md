# 赔率比偏好优化 (ORPO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 有
- 论文: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

## 公式

$$
\mathcal{L}_{\mathrm{ORPO}}=-\log p_\theta(y_w)+\lambda\left(-\log\sigma\!\left(\log\frac{\mathrm{odds}(y_w)}{\mathrm{odds}(y_l)}\right)\right)
$$

把监督微调的负对数似然和赔率比惩罚加在一起，不使用参考模型。

## 直觉

第一项把偏好回答的概率往上拉，第二项要求它的赔率 p/(1-p) 高于落选回答。

## 物理意义

监督项防止 SimPO 那种两条概率一起下降。赔率比在概率靠近 0 或 1 时比对数概率差更敏感。

## 失效

lambda 过大时，赔率项会压过似然项，偏好回答的绝对概率反而下降。

实现: https://github.com/xfactlab/orpo
