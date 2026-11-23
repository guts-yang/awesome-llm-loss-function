# 反向 KL (Reverse KL)

- 任务族: 蒸馏
- 年份: 2023
- 下界: 有
- 论文: [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)

## 公式

$$
D_{\mathrm{KL}}(p_{\theta}\Vert p_{\mathrm{teacher}})=\sum_t p_{\theta}(t)\log\frac{p_{\theta}(t)}{p_{\mathrm{teacher}}(t)}
$$

学生只需要把自己的质量放在教师也认可的模式上，可以丢掉教师的其他模式。

## 直觉

反向 KL 惩罚学生放到教师低概率处的质量。它允许学生比教师更尖。

## 物理意义

这是模式寻求。MiniLLM 用它蒸馏大模型，是为了避免正向 KL 把学生训得又平又散。力集中在学生当前的模式上。

## 失效

学生可以合法地忽略教师的次模式。任务真的需要那些次模式时，反向 KL 会看起来损失很低，覆盖却不完整。
