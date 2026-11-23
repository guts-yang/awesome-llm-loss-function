# 损失选择指南

## 续写和指令微调

先看 `next_token_ce`, `label_smoothing`, `ul2_span`。

先看负对数似然。标签平滑给它一个软上限，UL2 只改变哪些 token 产生这条力。

## 有偏好对，也有参考模型

先看 `dpo`, `ipo`, `bradley_terry`。

DPO 用 logistic 间隔，IPO 把间隔回归到固定目标。奖励模型是同一条 Bradley-Terry 比较，但不直接更新策略。

## 没有参考模型的偏好对齐

先看 `simpo`, `orpo`, `slic`。

SimPO 靠间隔 gamma，ORPO 加回监督似然，SLiC 用铰链。三者都不再相对 π_ref 平移。

## 只有好或坏的单边反馈

先看 `kto`。

KTO 不需要成对采样，好回答和坏回答相对 z0 分别饱和。

## 在线强化学习

先看 `ppo_clip`, `grpo`。

PPO 裁剪新旧概率比。GRPO 用组内奖励代替价值网络，并用参考模型把策略拉回来。

## 把教师分布交给学生

先看 `forward_kl`, `reverse_kl`, `jsd`, `rkld`。

正向 KL 覆盖全部模式，反向 KL 只守住学生自己的模式。JS 取两者的混合并且有上界，RKLD 看两者的差。

## 强行压低遗忘答案，先不管崩塌

先看 `ga`, `graddiff`, `ga_kl`。

梯度上升没有下界。GradDiff 加上保留交叉熵，GA+KL 把保留分布锚在参考模型。

## 遗忘损失要自己停下来

先看 `npo`, `simnpo`, `satimp`, `jensun`。

NPO 和 SimNPO 靠 logistic 饱和，SatImp 靠概率权重，JensUn 靠有界的 JS 散度。

## 没有可用的保留集

先看 `simnpo`, `whp`, `unilogit`, `shred`。

SimNPO 不需要参考模型。WHP 改训替代答案，Unilogit 指向均匀分布，SHRED 用冻结的自己做教师。

## 在表示层而不是 token 概率上动手

先看 `rmu`。

RMU 把遗忘激活拉向随机方向，把保留激活拉回原模型。概率平面上对应两个二次坑。

## 遗忘梯度和保留梯度缠在一起

先看 `blur`, `graddiff`。

先在参数平面把特征夹角 φ 调小。BLUR 的正交步会去掉沿保留梯度的分量，GradDiff 只是把两股力相加。
