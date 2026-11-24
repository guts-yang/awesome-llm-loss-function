# 大语言模型损失函数

这一页按任务族阅读。交互看板只留当前损失的三句；这里把同一条损失的底层逻辑、物理意义和使用场景放在一起。公式保持英文 LaTeX。

先按手头的数据和想保住的东西挑一条，再往下读它的公式、底层逻辑、物理意义和使用场景。

## 先看哪一条

- **续写和指令微调**：先看下一 token 交叉熵（`next_token_ce`）、标签平滑（`label_smoothing`）、UL2 span 交叉熵（`ul2_span`）。先看负对数似然。标签平滑给它一个软上限，UL2 只改变哪些 token 产生这条力。
- **有偏好对，也有参考模型**：先看直接偏好优化（`dpo`）、身份偏好优化（`ipo`）、Bradley-Terry 奖励模型（`bradley_terry`）。DPO 用 logistic 间隔，IPO 把间隔回归到固定目标。奖励模型是同一条 Bradley-Terry 比较，但不直接更新策略。
- **没有参考模型的偏好对齐**：先看简单偏好优化（`simpo`）、赔率比偏好优化（`orpo`）、序列似然校准（`slic`）。SimPO 靠间隔 gamma，ORPO 加回监督似然，SLiC 用铰链。三者都不再相对 π_ref 平移。
- **只有好或坏的单边反馈**：先看前景理论对齐（`kto`）。KTO 不需要成对采样，好回答和坏回答相对 z0 分别饱和。
- **在线强化学习**：先看PPO 裁剪目标（`ppo_clip`）、组相对策略优化（`grpo`）。PPO 裁剪新旧概率比。GRPO 用组内奖励代替价值网络，并用参考模型把策略拉回来。
- **把教师分布交给学生**：先看正向 KL（`forward_kl`）、反向 KL（`reverse_kl`）、JS 散度（`jsd`）、反向减正向 KL（`rkld`）。正向 KL 覆盖全部模式，反向 KL 只守住学生自己的模式。JS 取两者的混合并且有上界，RKLD 看两者的差。
- **强行压低遗忘答案，先不管崩塌**：先看梯度上升（`ga`）、梯度差（`graddiff`）、梯度上升加 KL（`ga_kl`）。梯度上升没有下界。GradDiff 加上保留交叉熵，GA+KL 把保留分布锚在参考模型。
- **遗忘损失要自己停下来**：先看负偏好优化（`npo`）、简单负偏好优化（`simnpo`）、饱和-重要性重加权（`satimp`）、Jensen-Shannon 遗忘（`jensun`）。NPO 和 SimNPO 靠 logistic 饱和，SatImp 靠概率权重，JensUn 靠有界的 JS 散度。
- **没有可用的保留集**：先看简单负偏好优化（`simnpo`）、替代答案微调（`whp`）、均匀目标自蒸馏（`unilogit`）、自蒸馏 logit 降权（`shred`）。SimNPO 不需要参考模型。WHP 改训替代答案，Unilogit 指向均匀分布，SHRED 用冻结的自己做教师。
- **在表示层而不是 token 概率上动手**：先看表示错向（`rmu`）。RMU 把遗忘激活拉向随机方向，把保留激活拉回原模型。概率平面上对应两个二次坑。
- **遗忘梯度和保留梯度缠在一起**：先看双层正交遗忘（`blur`）、梯度差（`graddiff`）。先在参数平面把特征夹角 φ 调小。BLUR 的正交步会去掉沿保留梯度的分量，GradDiff 只是把两股力相加。

## 语言建模

这一族看目标 token。横轴是下一个正确 token 的概率，损失推动的是这一个概率。

### 下一 token 交叉熵

Next-token CE · 2003 · 条目 `next_token_ce` · 有下界

$$\mathcal{L}_{\mathrm{CE}}=-\log p_{\theta}(x_t\mid x_{<t})$$

**底层逻辑**

提高下一个正确 token 的概率。

交叉熵只奖励目标 token。概率越低，对数损失下降得越陡，所以模型会把质量集中到当前答案上。

**物理意义**

对 logit 的导数是 p−1。目标概率接近 1 时推力消失，接近 0 时推力停在 1 附近。损失只随目标概率变化。

二元玩具里，横轴是目标 token 概率。损失与纵轴无关。

最大似然会把相近答案的概率一起抬高。

**使用场景**

用来做预训练和监督微调，让模型把下一个词说对。聊天、续写、指令跟随的第一阶段都靠它。手上只有标准答案，还没有偏好对比或遗忘名单时，先用它。

论文：[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
实现：https://github.com/huggingface/transformers

### 标签平滑

Label smoothing · 2016 · 条目 `label_smoothing` · 有下界

$$\mathcal{L}_{\mathrm{LS}}=-(1-\epsilon)\log p_{\mathrm{target}}-\epsilon\log p_{\mathrm{other}}$$

**底层逻辑**

把一部分概率质量分给非目标 token，避免全部质量压到一个 logit 上。

平滑项是在交叉熵旁边加一个对照 token 的对数惩罚。epsilon 越大，最优点越不愿意把目标概率推到 1。

**物理意义**

ε 越大，最优点越不愿意把目标概率推到 1。图上的对照项只代表一个非目标 token。

经典论文来自视觉网络，LLM 训练借用的是同一条平滑规则。图上只保留目标与一个对照 token。

ε 过大时，不该出现的 token 被抬高，续写发虚。

**使用场景**

给正确答案以外的词留一点概率，避免模型把信心全部压在唯一一个词上。近义说法不止一种时常用。模型已经把某个词的概率顶满、续写过分笃定时，加上它。

论文：[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

### UL2 span 交叉熵

UL2 span CE · 2022 · 条目 `ul2_span` · 有下界

$$\mathcal{L}_{\mathrm{UL2}}=-\log p_{\theta}(x_{\mathrm{span}}\mid x_{\setminus\mathrm{span}})$$

**底层逻辑**

用交叉熵预测被遮住的 span。改变的是哪些位置产生这条力。

损失的形状仍是负对数似然。UL2 改变的是被预测的 span 和任务混合比例，不是评分函数本身。

**物理意义**

目标概率靠近 1 时推力消失。损失形状不随 span 改变，变的是哪些 token 进入这项。

曲面上它与下一 token 交叉熵重合。这是故意保留的：两者的差异在数据构造，不在损失形状。

把 span 混合当成新的几何约束会高估它。它只改变哪些位置被评分。

**使用场景**

同一套交叉熵，轮流预测被遮住的一段、前缀后面的词，或下一个词。用来做通用预训练，让一个模型既能补全也能续写。语料没有标注、希望覆盖多种填空和生成任务时用。

论文：[UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)

## 偏好对齐

这一族看偏好回答和落选回答。横轴是偏好回答的概率，纵轴是落选回答的概率。

### Bradley-Terry 奖励模型

Bradley-Terry RM · 2017 · 条目 `bradley_terry` · 有下界

$$\mathcal{L}_{\mathrm{BT}}=-\log\sigma(\beta\log p_w-\beta\log p_l)$$

**底层逻辑**

用成对比较训练奖励，奖励直接取成对数概率差。

它和 gamma 为 0 的 SimPO 同形。差别只在它训练的是奖励头，不直接更新策略。

**物理意义**

β 把分差放大。分差继续扩大时 sigmoid 梯度消失，奖励可以很有信心，同时不再提供训练信号。

奖励被直接取为对数概率，用来和 DPO 的隐式奖励对照。

奖励没有长度校正时，会偏向高概率的短回答。

**使用场景**

先训练一个打分器，判断人更喜欢一对回答里的哪一条。它不直接改写生成模型，而是给后面的强化学习提供奖励。有大量成对偏好、需要给回答打分或准备跑 PPO 时用。

论文：[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
实现：https://github.com/huggingface/trl

### PPO 裁剪目标

PPO clip · 2017 · 条目 `ppo_clip` · 无下界

$$\mathcal{L}_{\mathrm{PPO}}=-\mathbb{E}\left[\min(r_t A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]$$

**底层逻辑**

用新旧策略的概率比裁剪优势，限制一次更新离开旧策略多远。

优势为正时，概率比超过 1+epsilon 就不再加分。优势为负时，概率比低于 1-epsilon 就不再减分。

**物理意义**

ε 是信任区域的宽度。优势为正时，概率比超过 1+ε 不再加分；优势为负时，低于 1−ε 不再减分。

二元动作上的裁剪目标。InstructGPT 的完整目标还包含相对参考模型的 KL 惩罚。

裁剪只限制概率比。旧策略概率估计偏了，信任区域会切错位置。图上没有相对参考模型的 KL 惩罚。

**使用场景**

模型自己采样回答，用奖励打分，同时限制这一步不要离旧策略太远。这是 RLHF 里奖励模型训好之后的策略更新。需要在线采样、又怕一次更新把模型带飞时用。

论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
实现：https://github.com/huggingface/trl

### 直接偏好优化

DPO · 2023 · 条目 `dpo` · 有下界

$$\mathcal{L}_{\mathrm{DPO}}=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right)$$

**底层逻辑**

用偏好对训练策略，奖励是相对参考模型的对数概率差。

损失只看 chosen 与 rejected 的对数比。参考模型把这个比锚定在微调之前的策略上。

**物理意义**

β 把这个差变陡；参考概率升高时，同样的当前概率得到的奖励变小，等高线平移。

chosen 和 rejected 太近时，对数比的梯度很尖。

**使用场景**

用“这条回答比那条好”直接改模型，不必先训奖励模型再跑强化学习。对话对齐、让回答更符合人类偏好时最常见。已经有成对偏好，也还留着微调前的模型时用它。

论文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
实现：https://github.com/eric-mitchell/direct-preference-optimization

### 身份偏好优化

IPO · 2023 · 条目 `ipo` · 有下界

$$\mathcal{L}_{\mathrm{IPO}}=\left(\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}-\frac{1}{2\beta}\right)^2$$

**底层逻辑**

把相对参考模型的对数概率间隔回归到固定目标 1/(2β)。

DPO 的 sigmoid 在间隔已经很大时梯度消失。IPO 改成平方损失，间隔不足和间隔过大都会被拉回来。

**物理意义**

β 越小，要求的对数间隔越大，最低点离对角线越远。

间隔已经够大时平方项仍往回拉，噪声会把分开的回答再拉近。

**使用场景**

同样吃偏好对，但把两条回答的差距拉到一个固定距离，而不是越大越好。偏好数据有噪声、差距被拉得过猛时用。有参考模型，又担心好回答和差回答被撕得太开时选它。

论文：[A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)
实现：https://github.com/huggingface/trl

### 序列似然校准

SLiC-HF · 2023 · 条目 `slic` · 有下界

$$\mathcal{L}_{\mathrm{SLiC}}=\max(0,\delta-\log\pi(y_w)+\log\pi(y_l))-\lambda\log\pi(y_w)$$

**底层逻辑**

用铰链把偏好回答的对数概率抬到落选回答之上，并用似然项保住它的概率。

间隔已经大于 delta 时，排序项的力是 0，只剩监督项。间隔不够时，两条回答受到方向相反的常数力。

**物理意义**

δ 决定铰链折线离对角线多远，λ 决定折线之外还剩多少似然拉力。

间隔已经大于 δ 时排序项停力。δ 大于数据里的自然间隔时，落选回答会被过度压低。

**使用场景**

要求好回答高出差回答一个间隔，间隔够了就不再加力，同时保住句子通顺。做偏好对齐、希望排序到位后就停手时用。没有参考模型，又担心对齐把流畅度训坏时选它。

论文：[SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)
实现：https://github.com/huggingface/trl

### 组相对策略优化

GRPO · 2024 · 条目 `grpo` · 无下界

$$\mathcal{L}_{\mathrm{GRPO}}=-\sum_i \hat A_i\log\pi_\theta(y_i)+\frac{\beta}{2}\left(\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)^2$$

**底层逻辑**

用同一组采样的标准化奖励当优势，不再训练单独的价值网络。

两组奖励一高一低时，优势大约是 +1 和 -1。策略项把高奖励回答的对数概率抬高，把低奖励回答压低，平方项把两者拉回参考模型。

**物理意义**

组内均值和标准差决定力的方向，β 把策略拉回参考模型。图上用平方对数比代替序列 KL。

平方对数比是序列 KL 的二元替身，用来保留“不要离开参考模型太远”这一项。

组内奖励没有差别时优势为 0，策略项消失。奖励只反映长度或格式时，标准化会把这个方向固定下来。

**使用场景**

同一道题采样一组回答，用组内的相对高低更新策略，不必再训价值网络。数学、代码这类能自动判对错的题最常见。一道题能采多条、奖励可以放在一组里比较时用。

论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
实现：https://github.com/huggingface/trl

### 前景理论对齐

KTO · 2024 · 条目 `kto` · 有下界

$$\mathcal{L}_{\mathrm{KTO}}=\lambda_D\left(1-\sigma(\beta(r_\theta(y_+)-z_0))\right)+\lambda_U\left(1-\sigma(\beta(z_0-r_\theta(y_-)))\right)$$

**底层逻辑**

好回答和坏回答各自相对参考点 z0 计算得失，不需要成对采样。

好回答的损失在奖励超过 z0 后饱和，坏回答的损失在奖励低于 z0 后饱和。两条通道不必来自同一次采样。

**物理意义**

z0 是两侧力的分界，图上它是常数。β 决定饱和速度，两侧权重决定哪一边的力更大。

z0 在真实训练里是批内 KL 估计，这里做成可调常数。

z0 定错时，好回答和坏回答会被推向同一个方向。

**使用场景**

只有“这条好”或“这条差”，凑不成一对时用。点赞、点踩、有用或无用的日志都能直接拿来对齐。数据是单条评价，不是成对比较时选它。

论文：[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
实现：https://github.com/ContextualAI/HALOs

### 赔率比偏好优化

ORPO · 2024 · 条目 `orpo` · 有下界

$$\mathcal{L}_{\mathrm{ORPO}}=-\log p_\theta(y_w)+\lambda\left(-\log\sigma\!\left(\log\frac{\mathrm{odds}(y_w)}{\mathrm{odds}(y_l)}\right)\right)$$

**底层逻辑**

偏好回答的负对数似然，加上它相对落选回答的赔率比惩罚。

第一项把偏好回答的概率往上拉，第二项要求它的赔率 p/(1-p) 高于落选回答。

**物理意义**

λ 越大，赔率比越压过似然项。赔率在概率靠近 0 或 1 时更敏感。

λ 过大时，偏好回答的绝对概率反而下降。

**使用场景**

监督微调和偏好对齐合成一步：既要会说偏好的那条，又要让它比落选的那条更占优。没有参考模型，又希望正确答案的概率别被一起压低时用。指令答案和偏好来自同一批样本时很合适。

论文：[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
实现：https://github.com/xfactlab/orpo

### 简单偏好优化

SimPO · 2024 · 条目 `simpo` · 有下界

$$\mathcal{L}_{\mathrm{SimPO}}=-\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l)-\gamma\right)$$

**底层逻辑**

去掉参考模型，用长度归一化的对数概率差减去间隔 γ。

等高线只由 p_w 和 p_l 的比值决定。gamma 把决策边界从对角线推开，要求偏好回答高出一个固定余量。

**物理意义**

γ 把决策边界从对角线推开，β 控制斜率。图上两条回答的长度都取 1。

长度归一化在二元玩具中取 |y|=1。

两条回答的概率可以一起下降，只要偏好的那条相对更高。长度归一化偏了，长回答会被压低。

**使用场景**

做偏好对齐，但不保存参考模型。直接比较两条回答自己的平均对数概率，并要求好回答高出一个余量。只有偏好对、没有留下原来的模型时用它。

论文：[SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
实现：https://github.com/princeton-nlp/SimPO

## 蒸馏

这一族看学生的两个通道。损失比较的是学生分布和教师分布，两个轴都是学生在归一化前的质量。

### 正向 KL

Forward KL · 2015 · 条目 `forward_kl` · 有下界

$$D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert p_{\theta})=\sum_t p_{\mathrm{teacher}}(t)\log\frac{p_{\mathrm{teacher}}(t)}{p_{\theta}(t)}$$

**底层逻辑**

学生要覆盖教师的全部质量。教师有概率的地方，学生不能放空。

等高线只依赖 x:y 的比例。把两个通道同时放大不会改变损失，因为学生分布先被归一化。

**物理意义**

学生在教师的某个模式上接近 0 时，损失按对数上升，质量被铺开。两个通道先归一化，对角线方向是平的。

二元分布先做归一化，所以曲面对角线方向是平的。

教师错了，错误模式也会被完整复制下来。

**使用场景**

把教师的整份概率交给学生，教师给过概率的地方学生都不能空着。模型压缩、草稿模型，以及遗忘里把保留行为锚回原模型，都会用到它。学生容量够、希望多种说法都继承下来时选它。

论文：[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

### 反向 KL

Reverse KL · 2023 · 条目 `reverse_kl` · 有下界

$$D_{\mathrm{KL}}(p_{\theta}\Vert p_{\mathrm{teacher}})=\sum_t p_{\theta}(t)\log\frac{p_{\theta}(t)}{p_{\mathrm{teacher}}(t)}$$

**底层逻辑**

学生只把自己的质量放在教师也认可的模式上，可以丢掉教师的其他模式。

反向 KL 惩罚学生放到教师低概率处的质量。它允许学生比教师更尖。

**物理意义**

惩罚落在教师低概率处的学生质量上，力集中在学生当前的模式。

任务需要教师的次模式时，损失可以很低，覆盖却不完整。

**使用场景**

学生只把概率放在教师也认可的说法上，可以丢掉教师的次要说法。用来把大模型蒸馏成更会生成、不那么平均的小模型。学生比教师小很多、生成发虚发平时用。

论文：[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)

### JS 散度

Jensen-Shannon · 2024 · 条目 `jsd` · 有下界

$$m=\beta p_{\mathrm{teacher}}+(1-\beta)p_{\theta},\quad \mathrm{JSD}_{\beta}(p_{\mathrm{teacher}}\Vert p_{\theta})=\beta D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert m)+(1-\beta)D_{\mathrm{KL}}(p_{\theta}\Vert m)$$

**底层逻辑**

学生对混合分布 m 做 KL。β=0.5 时损失落在 0 和 log 2 之间。

m 同时含有两边的质量，所以一边接近 0 不会把另一边的对数项打到无穷。等高线仍只看学生两个通道的比例。

**物理意义**

β 靠近 0 时学生要铺开覆盖教师，靠近 1 时可以丢掉教师的次模式。一端接近 0 时力不再继续变陡。

β=0.5 是 Lin 的对称 JS。广义形式来自 GKD，m 是教师和学生的凸组合。序列上的实现还要对每个 token 取平均。

损失有上界。覆盖还没完成时，力也可以先饱和。

**使用场景**

用有上限的 JS 散度比较学生和教师，介于全覆盖和只追一个说法之间。GKD 用它做在线蒸馏：学生按自己生成的句子去问教师该怎么改。希望蒸馏比较稳，或学生会走到教师没见过的前缀上时用。

论文：[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)

### 反向减正向 KL

RKLD · 2024 · 条目 `rkld` · 无下界

$$\mathcal{L}_{\mathrm{RKLD}}=D_{\mathrm{KL}}(p_{\theta}\Vert p_{\mathrm{ref}})-D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\theta})$$

**底层逻辑**

用反向 KL 减去正向 KL。学生更尖时差为正，学生为了覆盖而铺开时差被拉回。

两个 KL 都非负，它们的差可正可负。学生比教师更尖时，反向项占优；学生为了覆盖而铺开时，正向项把差拉回来。

**物理意义**

两项放在同一对概率上。差可正可负，负值不是概率误差。

这是同一对学生分布上的 RKL-FKL，用来看两种 KL 的方向差。遗忘论文里的实现通常把两项分配到遗忘集和保留集。

差没有下界。图上只保留这一对概率的符号方向。

**使用场景**

留下把学生推离教师不想要的说法的那股力。遗忘里常把反向项放在要忘掉的分布上，正向项放在要保留的分布上。两种方向需要拆开、而不是简单相加时用。

论文：[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)

## 机器遗忘

这一族看遗忘目标和保留目标。横轴是要压低的概率，纵轴是要保住的概率。

### 替代答案微调

WHP · 2023 · 条目 `whp` · 有下界

$$\mathcal{L}_{\mathrm{WHP}}=-\log\frac{p_{\mathrm{alt}}}{p_f+p_{\mathrm{alt}}}$$

**底层逻辑**

在替代答案上做交叉熵，原答案变低只是归一化的副作用。

损失只关心替代答案在这对概率里的份额。原答案概率下降是归一化的副作用，不是无界的梯度上升。

**物理意义**

纵轴是替代概率。最低点在原答案相对替代答案很小的方向，损失有下界 0，力来自份额。

纵轴是替代续写的概率。真实方法在新答案的 token 上做交叉熵。

没有替代答案时这项无从定义。替代句式和原知识相邻时，相邻事实会被一起改写。

**使用场景**

不直接踩原答案，而是拿一个替代说法做普通训练，让模型遇到同类问题时改口。版权角色、特定事实需要换成另一种叙述时用。已经准备好替代答案才能用。

论文：[Who's Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/abs/2310.02238)

### 梯度上升加 KL

GA + KL · 2024 · 条目 `ga_kl` · 无下界

$$\mathcal{L}_{\mathrm{GA+KL}}=\log p_{\theta}(y_f)+\beta\, D_{\mathrm{KL}}\!\left(p_{\theta}(\cdot\mid x_r)\Vert p_{\mathrm{ref}}(\cdot\mid x_r)\right)$$

**底层逻辑**

遗忘答案做梯度上升，保留分布用 KL 锚在参考模型上。

KL 的最低点在 p_r = p_ref，不像交叉熵那样把保留概率一直推向 1。

**物理意义**

β 越大，保留通道越被拉回参考概率。遗忘通道在 p_f 趋向 0 时推力仍趋向 1。

KL 写在伯努利保留通道上，对应序列实现里保留提示上的 token KL。

两条通道的表示重叠时，KL 和梯度上升对着拉，两边都难以到位。

**使用场景**

遗忘答案继续往下压，保留行为被拉回原来的模型，而不是一味把保留答案推到满分。有参考模型、希望保留集上的说法别漂太远时用。

论文：[Machine Unlearning of Pre-trained Large Language Models](https://arxiv.org/abs/2402.15159)
实现：https://github.com/locuslab/tofu

### 梯度差

GradDiff · 2024 · 条目 `graddiff` · 无下界

$$\mathcal{L}_{\mathrm{GradDiff}}=\log p_{\theta}(y_f\mid x)-\log p_{\theta}(y_r\mid x)$$

**底层逻辑**

遗忘答案做梯度上升，保留答案继续做交叉熵。

损失沿 p_f 下降、沿 p_r 上升的方向变小。它同时要求忘掉一个答案、记住另一个答案。

**物理意义**

遗忘通道的 logit 导数是 1−p_f，没有停止点；保留通道是 p_r−1，靠近 1 时停下。

保留集没覆盖到的相邻知识仍会被带走。训练久了，无界的遗忘项会淹没保留项。

**使用场景**

一边压低要遗忘的答案，一边在保留数据上继续做普通训练。想忘掉某一类问答，同时保住通用能力时用。手头既有遗忘样本，也有一批应该继续答对的样本时选它。

论文：[TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)
实现：https://github.com/locuslab/tofu

### 梯度上升

Gradient Ascent · 2024 · 条目 `ga` · 无下界

$$\mathcal{L}_{\mathrm{GA}}=\log p_{\theta}(y_f\mid x)$$

**底层逻辑**

把遗忘答案的交叉熵翻号，最小化就是在压低遗忘概率。

p_f 越小，损失越负。曲面向概率 0 的方向无限下降，没有坑底。

**物理意义**

对 logit 的导数是 1−p。p 趋向 0 时导数趋向 1，推力不消失。保留概率不进这项。

纵轴不进入损失。能力保持要靠别的项，或者靠参数平面上的投影。

遗忘集稍大或学习率稍高时，相邻 logit 一起被带走。

**使用场景**

把不该再出现的答案的概率往下压，是最简单的遗忘基线。用来对照别的方法到底忘得够不够。只想先看这个答案还会不会被说出来，并且接受模型可能说崩时，可以先试它。

论文：[TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)
实现：https://github.com/locuslab/tofu

### 负偏好优化

NPO · 2024 · 条目 `npo` · 有下界

$$\mathcal{L}_{\mathrm{NPO}}=-\frac{2}{\beta}\log\sigma\!\left(-\beta\log\frac{\pi_{\theta}(y_f)}{\pi_{\mathrm{ref}}(y_f)}\right)$$

**底层逻辑**

把遗忘回答当成落选回答，概率降下去之后梯度自己关掉。

p_f 远低于参考模型时，sigmoid 饱和，损失贴着 0。它不再像梯度上升那样越忘越有力。

**物理意义**

logit 推力随 p 趋向 0 指数衰减到 0。β 越小，开始减速的区间越宽。保留交叉熵默认权重为 0。

公式主体是论文中的遗忘项。保留交叉熵是可选附加项，默认权重为 0。

参考模型把遗忘答案写得很深时，要降很多才饱和，早期更新仍接近满推力。

**使用场景**

把要遗忘的回答当成偏好学习里的差回答，概率降下去之后推力自己变小。用来在忘掉目标的同时，避免一直推到模型崩溃。有参考模型、希望训练自己收手时用。

论文：[Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)
实现：https://github.com/licong-lin/negative-preference-optimization

### 表示错向

RMU · 2024 · 条目 `rmu` · 有下界

$$\mathcal{L}_{\mathrm{RMU}}=\lVert h_f-c\,u\rVert_2^2+\alpha\lVert h_r-h_{\mathrm{ref}}\rVert_2^2$$

**底层逻辑**

遗忘激活拉向一个随机方向，保留激活拉回原模型。

概率平面上这是两个二次坑。遗忘通道的坑在噪声目标，保留通道的坑在参考概率。两个方向都有下界。

**物理意义**

α 决定保留坑有多硬。图上把隐藏状态的平方距离压成两个概率通道的二次项，靠近目标梯度就下降。

隐藏状态上的平方距离被压成两个概率通道上的二次项，用来保留“两边都有坑底”这一几何。

随机方向仍能被读出时，模型会改口。噪声目标过大时，遗忘通道的幅度会变得异常。

**使用场景**

不改词的概率，而改中间层表示：遗忘样本被拨向一个随机方向，保留样本被拉回原模型。危险知识、希望模型别再顺着原来的内部方向想下去时会用。改输出词容易误伤相邻知识时可以试它。

论文：[The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning](https://arxiv.org/abs/2403.03218)
实现：https://github.com/centerforaisafety/wmdp

### 简单负偏好优化

SimNPO · 2024 · 条目 `simnpo` · 有下界

$$\mathcal{L}_{\mathrm{SimNPO}}=-\frac{2}{\beta}\log\sigma\!\left(-\frac{\beta}{|y|}\log\pi_{\theta}(y_f)-\gamma\right)$$

**底层逻辑**

NPO 去掉参考模型，改用长度归一化的对数概率和一个间隔 γ。

没有 p_ref 可调。概率本身变小之后，logistic 仍然饱和，所以下界还在。

**物理意义**

γ 平移饱和开始的位置，β 控制斜率。图上回答长度取 1。

长度归一化取 |y|=1。

没有参考模型时，所有答案的概率可以一起下降，只要遗忘答案相对更低。长度估计偏了，长短回答的遗忘强度不一致。

**使用场景**

和 NPO 一样让遗忘推力自己停下来，但不需要参考模型。没有留下原模型、又希望遗忘损失有底时用。只有遗忘文本、凑不出参考分布时选它。

论文：[Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](https://arxiv.org/abs/2410.07163)

### 双层正交遗忘

BLUR · 2025 · 条目 `blur` · 无下界

$$\theta^{\star}(\omega)=\arg\min_{\theta}\mathcal{L}_R(\theta;\omega),\quad \min_{\omega}\mathcal{L}_F(\theta^{\star}(\omega))$$

**底层逻辑**

外层降低遗忘损失，内层先把保留损失修回来。步进时去掉沿保留梯度的分量。

概率平面上仍能看到无界的遗忘项和一条保留 KL 弹簧。真正不同的是更新方向：步进前先去掉沿保留梯度的分量。

**物理意义**

φ 越小，被去掉的分量越大，遗忘变慢，保留概率的漂移变小。概率平面画的是遗忘项加保留 KL。

概率平面画的是外层遗忘项加保留 KL。参数平面请把优化器设为正交投影，那才是双层约束的玩具。

投影只保护这一步的保留梯度。那个方向和要保护的知识不一致时，会保护错方向。遗忘项仍然没有下界。

**使用场景**

更新前先看保留方向，把遗忘更新里沿着这个方向的分量去掉，让遗忘少带动该留的知识。遗忘和保留缠在一起、直接相加会误伤时用。

论文：[BLUR: A Bi-Level Optimization Approach for LLM Unlearning](https://arxiv.org/abs/2506.08164)

### Jensen-Shannon 遗忘

JensUn · 2025 · 条目 `jensun` · 有下界

$$\mathcal{L}_{\mathrm{JensUn}}=\lambda_F\,\mathrm{JSD}(p_{\theta}\Vert p_{\mathrm{safe}})+\lambda_R\,\mathrm{JSD}(p_{\theta}\Vert p_{\mathrm{ref}})$$

**底层逻辑**

遗忘通道贴近安全目标，保留通道贴近参考模型，两边都是有上界的 JS 散度。

JS 散度落在 0 和 log 2 之间。目标达到以后梯度消失，离目标再远也不会像 KL 或 log p 那样把力放到无限大。

**物理意义**

p_safe 和 p_ref 分别是两个坑底。散度落在 0 和 log 2 之间，到位以后梯度消失。

论文在序列上对拒绝目标的 one-hot 和参考分布分别计算 JSD。这里把两个目标放进遗忘、保留两个伯努利通道。

安全目标只是拒答时，模型可以换一种说法，事实仍在。

**使用场景**

遗忘通道靠近一个安全目标，保留通道靠近原模型，两边的力都有上限。希望忘得比较稳、又尽量保住原有能力时用。能给出安全目标，也还留着参考模型时选它。

论文：[Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs](https://arxiv.org/abs/2509.02820)
实现：https://github.com/nmndeep/JensUn-Unlearning

### 饱和-重要性重加权

SatImp · 2025 · 条目 `satimp` · 有下界

$$w(p)=p^{\beta_1}(1-p)^{\beta_2},\qquad \mathcal{L}_{\mathrm{SatImp}}=w(p_f)\log p_f$$

**底层逻辑**

在梯度上升上按当前概率加权。概率已经很小的 token，权重下降，更新停手。

β1 越大，低概率 token 越快失去权重。β2 压低已经接近 1 的 token。中间概率的 token 承担大部分遗忘。

**物理意义**

β1 越大，低概率处的损失越快回到 0，logit 推力随之消失。权重写进损失，梯度会穿过它。

权重写进标量损失，数值梯度会穿过权重。这对应没有对权重停梯度的实现。

β1 过大时，概率稍一下降权重就没了，遗忘停在半途。

**使用场景**

仍是把遗忘答案往下压，但概率已经很小的词权重也变小，更新会自己收手。想要梯度上升那种方向，又希望低概率处别再死命推时用。

论文：[Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning](https://arxiv.org/abs/2505.11953)
实现：https://github.com/tmlr-group/SatImp

### 均匀目标自蒸馏

Unilogit · 2025 · 条目 `unilogit` · 有下界

$$\mathcal{L}_{\mathrm{Unilogit}}=\mathrm{CE}(p_{\theta}, U)+\lambda\, D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\theta})$$

**底层逻辑**

遗忘位置的目标改成均匀分布，保留位置用参考模型做蒸馏。

均匀目标的交叉熵在 p_f = 1/|V| 处最低。原答案既不会被抬到 1，也不会被梯度上升到 0。

**物理意义**

词表越大，最低点越靠近 0，但坑底仍在。图上用目标 token 对其余质量近似均匀目标。

遗忘通道用“目标 token 对其余质量”的二元交叉熵近似全词表均匀目标。

模型可以在遗忘提示上变得很平，生成变成高熵噪声。

**使用场景**

把要遗忘的位置训得在词表上差不多一样平，保留位置继续模仿原模型。没有现成的保留答案，也不想把遗忘提示训成一句固定拒答时用。

论文：[Unilogit: Robust Machine Unlearning for LLMs Using Uniform-Target Self-Distillation](https://arxiv.org/abs/2505.06027)

### 自蒸馏 logit 降权

SHRED · 2026 · 条目 `shred` · 有下界

$$\mathcal{L}_{\mathrm{SHRED}}=\mathrm{softplus}\!\left(\beta\log\frac{p_{\theta}(y_f)}{p_{\mathrm{self}}(y_f)}+m\right)+\lambda\, D_{\mathrm{KL}}(p_{\mathrm{self}}\Vert p_{\theta})$$

**底层逻辑**

用冻结的自己当教师，压低遗忘 token，并在其余通道上做自蒸馏。

softplus 在学生已经低于教师一个间隔后梯度消失。自蒸馏把其余通道锚在更新前的模型上。

**物理意义**

间隔 m 决定要低出多少才停，β 决定降权有多陡。教师冻结，损失不会跟着自己的概率滑向负无穷。

二元软间隔加自蒸馏是对论文机制的玩具，不是逐 token 的完整实现。

教师自己就会说出遗忘答案时，降权只是相对下移。自蒸馏权重太小，其余通道跟着漂移。

**使用场景**

没有外部保留集。用冻结的自己当老师，把遗忘词压低一个间隔，其余通道继续模仿更新前的自己。只有遗忘样本、另找保留语料不方便时用。

论文：[SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion](https://arxiv.org/abs/2605.07482)
