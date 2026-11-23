# 下一 token 交叉熵 (Next-token CE)

- 任务族: 语言建模
- 年份: 2003
- 下界: 有
- 论文: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## 公式

$$
\mathcal{L}_{\mathrm{CE}}=-\log p_{\theta}(x_t\mid x_{<t})
$$

提高下一个正确 token 的概率。

## 什么时候用

用来做预训练和监督微调，让模型把下一个词说对。聊天、续写、指令跟随的第一阶段都靠它。手上只有标准答案，还没有偏好对比或遗忘名单时，先用它。

## 直觉

交叉熵只奖励目标 token。概率越低，对数损失下降得越陡，所以模型会把质量集中到当前答案上。

## 物理意义

对 logit 的导数是 p−1。目标概率接近 1 时推力消失，接近 0 时推力停在 1 附近。损失只随目标概率变化。

## 失效

最大似然会把相近答案的概率一起抬高。

## 玩具坐标

二元玩具里，横轴是目标 token 概率。损失与纵轴无关。

实现: https://github.com/huggingface/transformers
