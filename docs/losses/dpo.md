# 直接偏好优化 (DPO)

- 任务族: 偏好对齐
- 年份: 2023
- 下界: 有
- 论文: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

## 公式

$$
\mathcal{L}_{\mathrm{DPO}}=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right)
$$

用偏好对训练策略，奖励是相对参考模型的对数概率差。

## 什么时候用

用“这条回答比那条好”直接改模型，不必先训奖励模型再跑强化学习。对话对齐、让回答更符合人类偏好时最常见。已经有成对偏好，也还留着微调前的模型时用它。

## 直觉

损失只看 chosen 与 rejected 的对数比。参考模型把这个比锚定在微调之前的策略上。

## 物理意义

β 把这个差变陡；参考概率升高时，同样的当前概率得到的奖励变小，等高线平移。

## 失效

chosen 和 rejected 太近时，对数比的梯度很尖。

实现: https://github.com/eric-mitchell/direct-preference-optimization
