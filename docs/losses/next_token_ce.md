# 下一 token 交叉熵 (Next-token CE)

- 任务族: 语言建模
- 年份: 2003
- 下界: 有
- 论文: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## 公式

$$
\mathcal{L}_{\mathrm{CE}}=-\log p_{\theta}(x_t\mid x_{<t})
$$

自回归语言模型的基本损失：提高下一个正确 token 的概率。

## 直觉

交叉熵只奖励目标 token。概率越低，对数损失下降得越陡，所以模型会把质量集中到当前答案上。

## 物理意义

对 logit z 而言，dL/dz = p_target - 1。目标概率接近 1 时推力消失，接近 0 时推力维持在 1 附近。这是后面所有偏好损失和遗忘损失的基准力。

## 失效

单纯最大似然会过拟合表面形式，也会把相近答案的概率一起抬高。它没有偏好间隔，也没有遗忘下界。

## 玩具坐标

二元玩具里，横轴是目标 token 概率。损失与纵轴无关。

实现: https://github.com/huggingface/transformers
