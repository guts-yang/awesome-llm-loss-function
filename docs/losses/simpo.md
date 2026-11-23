# 简单偏好优化 (SimPO)

- 任务族: 偏好对齐
- 年份: 2024
- 下界: 有
- 论文: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

## 公式

$$
\mathcal{L}_{\mathrm{SimPO}}=-\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l)-\gamma\right)
$$

去掉参考模型，用长度归一化的对数概率差减去间隔 γ。

## 什么时候用

做偏好对齐，但不保存参考模型。直接比较两条回答自己的平均对数概率，并要求好回答高出一个余量。只有偏好对、没有留下原来的模型时用它。

## 直觉

等高线只由 p_w 和 p_l 的比值决定。gamma 把决策边界从对角线推开，要求偏好回答高出一个固定余量。

## 物理意义

γ 把决策边界从对角线推开，β 控制斜率。图上两条回答的长度都取 1。

## 失效

两条回答的概率可以一起下降，只要偏好的那条相对更高。长度归一化偏了，长回答会被压低。

## 玩具坐标

长度归一化在二元玩具中取 |y|=1。

实现: https://github.com/princeton-nlp/SimPO
