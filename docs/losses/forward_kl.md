# 正向 KL (Forward KL)

- 任务族: 蒸馏
- 年份: 2015
- 下界: 有
- 论文: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

## 公式

$$
D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert p_{\theta})=\sum_t p_{\mathrm{teacher}}(t)\log\frac{p_{\mathrm{teacher}}(t)}{p_{\theta}(t)}
$$

学生去覆盖教师的全部质量。教师有概率的地方，学生不能放空。

## 直觉

等高线只依赖 x:y 的比例。把两个通道同时放大不会改变损失，因为学生分布先被归一化。

## 物理意义

正向 KL 是模式覆盖。学生若在教师的某个模式上接近 0，损失以对数速度上升，所以它倾向于把质量铺开。

## 失效

教师自己是错的时候，正向 KL 会把错误模式也完整复制下来。遗忘里如果拿它当保留项，这是优点；拿它当遗忘项，则会继续覆盖不该保留的答案。

## 玩具坐标

二元分布先做归一化，所以曲面对角线方向是平的。
