# 简单偏好优化 (SimPO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 有
- 论文: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

## 公式

$$
\mathcal{L}_{\mathrm{SimPO}}=-\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l)-\gamma\right)
$$

去掉参考模型，直接用长度归一化的对数概率差，再减去一个间隔 gamma。

## 直觉

等高线只由 p_w 和 p_l 的比值决定。gamma 把决策边界从对角线推开，要求偏好回答高出一个固定余量。

## 物理意义

与 DPO 相比，少了参考模型造成的平移。gamma 是纯间隔，beta 仍控制斜率。二元玩具里两条回答的长度都取 1。

## 失效

没有参考模型时，策略可以同时压低两条回答的概率，只要偏好那条相对更高。长度归一化不准确时，长回答会被系统性压低。

## 玩具坐标

长度归一化在二元玩具中取 |y|=1。

实现: https://github.com/princeton-nlp/SimPO
