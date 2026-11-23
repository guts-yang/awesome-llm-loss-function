# 直接偏好优化 (DPO)

- 任务族: 偏好对齐
- 年份: 2023
- 下界: 有
- 论文: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

## 公式

$$
\mathcal{L}_{\mathrm{DPO}}=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right)
$$

用偏好对直接训练策略，隐式奖励是相对参考模型的对数概率差。

## 直觉

损失只看 chosen 与 rejected 的对数比。参考模型把这个比锚定在微调之前的策略上。

## 物理意义

beta 是这个势阱的温度。参考概率升高时，同样的当前概率得到的奖励变小，等高线跟着平移。没有参考模型时，这条平移就消失了，那是 SimPO 的位置。

## 失效

chosen 和 rejected 很接近时，对数比的梯度很尖，容易把概率质量从相邻的正确回答里挤走。

实现: https://github.com/eric-mitchell/direct-preference-optimization
