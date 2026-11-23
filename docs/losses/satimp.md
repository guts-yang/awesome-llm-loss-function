# 饱和-重要性重加权 (SatImp)

- 任务族: 机器遗忘
- 年份: 2025
- 下界: 有
- 论文: [Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning](https://arxiv.org/abs/2505.11953)

## 公式

$$
w(p)=p^{\beta_1}(1-p)^{\beta_2},\qquad \mathcal{L}_{\mathrm{SatImp}}=w(p_f)\log p_f
$$

仍以梯度上升为底，但用当前概率给每个 token 加权。已经很小的概率权重大幅下降，更新自动停手。

## 直觉

β1 越大，低概率 token 越快失去权重。β2 压低已经接近 1 的 token。中间概率的 token 承担大部分遗忘。

## 物理意义

p^β1 log p 在 β1>0 时随 p→0 趋向 0，而不是趋向负无穷。logit 推力也随之消失。这和 NPO 的饱和是同一类几何，只是权重写在损失外面。

## 失效

β1 过大时，稍一下降权重就没了，遗忘停在半途。权重如果还回传到概率本身，早期梯度会比停梯度的重加权更尖。

## 玩具坐标

权重写进标量损失，数值梯度会穿过权重。这对应没有对权重停梯度的实现。

实现: https://github.com/tmlr-group/SatImp
