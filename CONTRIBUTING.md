# 贡献

新增一条 LLM 损失时：

1. 在 `core/losses/` 里用 `@register_loss` 写一个 `LossPlugin` 子类，或者调用 `make_loss`。
2. `latex` 使用英文符号。机制说明写中文，并写清下界、概率质量的去向，以及和保留梯度的夹角。
3. 概率平面上的 `loss(x, y, params)` 必须能接受 NumPy 数组。
4. 如果方法的关键差别在优化方向而不是标量损失，把 `step_mode` 设为 `"project"`，并在 `toy_note_zh` 里说明。
5. 运行 `pytest` 和 `python scripts/sync_catalog.py`。

不要把视觉检测、分割、GAN 或扩散损失放进这个仓库。条目组织参考了 [awesome-loss-functions](https://github.com/stabgan/awesome-loss-functions)，不要复制它的正文。
