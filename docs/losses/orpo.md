# 赔率比偏好优化 (ORPO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 有
- 论文: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

## 公式

$$
\mathcal{L}_{\mathrm{ORPO}}=-\log p_\theta(y_w)+\lambda\left(-\log\sigma\!\left(\log\frac{\mathrm{odds}(y_w)}{\mathrm{odds}(y_l)}\right)\right)
$$

偏好回答的负对数似然，加上它相对落选回答的赔率比惩罚。

## 什么时候用

监督微调和偏好对齐合成一步：既要会说偏好的那条，又要让它比落选的那条更占优。没有参考模型，又希望正确答案的概率别被一起压低时用。指令答案和偏好来自同一批样本时很合适。

## 直觉

第一项把偏好回答的概率往上拉，第二项要求它的赔率 p/(1-p) 高于落选回答。

## 物理意义

λ 越大，赔率比越压过似然项。赔率在概率靠近 0 或 1 时更敏感。

## 失效

λ 过大时，偏好回答的绝对概率反而下降。

实现: https://github.com/xfactlab/orpo
