# Awesome LLM Loss Function

Interactive LLM Loss Atlas。用中文讲解大语言模型的损失函数，用英文 LaTeX 写公式，并把每条损失从三维势能面投影到二维下降方向和一维 logit 推力。

关键词：`awesome llm loss function`、`llm loss functions`、`dpo npo simpo`、`llm unlearning loss`、`loss landscape`

远程仓库：[guts-yang/awesome-llm-loss-function](https://github.com/guts-yang/awesome-llm-loss-function)

## 它在看什么

语言建模、偏好对齐、蒸馏和机器遗忘被放进同一套概率坐标。

- 语言建模：横轴是目标 token 概率。
- 偏好对齐：横轴是偏好回答，纵轴是落选回答。
- 蒸馏：两个轴是学生分布归一化前的两个通道。
- 机器遗忘：横轴是遗忘目标，纵轴是保留目标。

有参考模型的损失可以改到 `p_θ` 对 `p_ref` 的平面，用来看 DPO 相对 SimPO、NPO 相对 SimNPO 多出来的那一次平移。

参数平面另有一个二维玩具。遗忘特征固定在第一条轴，保留特征与它夹角为 φ。φ 越小，两件事越缠在一起。SGD、Adam 和正交投影可以沿同一条轨迹比较。

## 目录

| 任务族 | 交互损失 |
| --- | --- |
| 语言建模 | Next-token CE、Label smoothing、UL2 span CE |
| 偏好对齐 | DPO、IPO、SimPO、KTO、ORPO、SLiC-HF、Bradley-Terry、PPO clip、GRPO |
| 蒸馏 | Forward KL、Reverse KL、RKLD |
| 机器遗忘 | GA、GradDiff、GA+KL、NPO、SimNPO、WHP、Unilogit、RMU、SHRED、JensUn、BLUR、SatImp |

选择指南在看板的「损失目录」，也写在 [`docs/selection-guide.md`](docs/selection-guide.md)。单条说明在 [`docs/losses/`](docs/losses/)。

概率平面上的曲面是二元玩具。序列长度、词表和隐藏状态被压成两个概率，为的是保留下界、饱和和锚定这些几何性质。每条损失自己的 `toy_note_zh` 写明了哪一部分是简化。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/sync_catalog.py
streamlit run app.py
```

论文索引默认读取同级目录的 `awesome-llm-unlearning/README.md`。对齐和蒸馏的经典论文写在 `scripts/sync_catalog.py` 的 curated 列表中。

```bash
pytest
```

本地计算只用 NumPy 和 SymPy，不需要 GPU。

## 扩展

新损失是一个 `LossPlugin`。最小接口是向量化的 `loss(x, y, params)`，三维、二维、一维图和有限差分会共用它。公式实验室可以把表达式导出成同样的插件骨架。

约定写在 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 参考

条目按任务、公式、论文和实现分开来写，这个组织方式参考了 [stabgan/awesome-loss-functions](https://github.com/stabgan/awesome-loss-functions)。本仓库没有复制它的正文或代码，范围也只限 LLM。

遗忘论文的元数据来自 [awesome-llm-unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)。
