# JS 散度 (Jensen-Shannon)

- 任务族: 蒸馏
- 年份: 2024
- 下界: 有
- 论文: [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)

## 公式

$$
m=\beta p_{\mathrm{teacher}}+(1-\beta)p_{\theta},\quad \mathrm{JSD}_{\beta}(p_{\mathrm{teacher}}\Vert p_{\theta})=\beta D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert m)+(1-\beta)D_{\mathrm{KL}}(p_{\theta}\Vert m)
$$

学生和教师一起对混合分布 m 做 KL。β=0.5 时这是对称的 JS 散度，损失落在 0 和 log 2 之间。

## 直觉

m 同时含有两边的质量，所以一边接近 0 不会把另一边的对数项打到无穷。等高线仍只看学生两个通道的比例。

## 物理意义

β 靠近 0 时，梯度方向接近正向 KL，学生要铺开去覆盖教师。β 靠近 1 时接近反向 KL，学生可以丢掉教师的次模式。两种情形下力都会饱和，不像 KL 在零概率处继续变陡。

## 失效

β 只决定散度的形状，不解决自回归训练和推理看到的前缀不一样。GKD 还要在学生自己采样的序列上计算这条散度；二元图只保留散度本身。

## 玩具坐标

β=0.5 是 Lin 的对称 JS。广义形式来自 GKD，m 是教师和学生的凸组合。序列上的实现还要对每个 token 取平均。
