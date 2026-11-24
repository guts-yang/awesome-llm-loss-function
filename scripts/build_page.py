"""Build the Chinese reading page from the loss plugins.

Markdown in docs/page.md is the source. docs/index.html is the generated view.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import all_losses, get_loss  # noqa: E402
from core.guide import GUIDE  # noqa: E402
from core.schema import FAMILIES  # noqa: E402

DOCS = ROOT / "docs"
PAGE_MD = DOCS / "page.md"
PAGE_HTML = DOCS / "index.html"

FAMILY_INTRO = {
    "language_modeling": "这一族看目标 token。横轴是下一个正确 token 的概率，损失推动的是这一个概率。",
    "alignment": "这一族看偏好回答和落选回答。横轴是偏好回答的概率，纵轴是落选回答的概率。",
    "distillation": "这一族看学生的两个通道。损失比较的是学生分布和教师分布，两个轴都是学生在归一化前的质量。",
    "unlearning": "这一族看遗忘目标和保留目标。横轴是要压低的概率，纵轴是要保住的概率。",
}


def renderer() -> Path:
    candidates = []
    skill_dir = os.environ.get("CLAUDE_SKILL_DIR")
    if skill_dir:
        candidates.append(Path(skill_dir) / "scripts" / "render_html.py")
    candidates.append(Path.home() / ".cursor/skills/render-html/scripts/render_html.py")
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("render_html.py 不在技能目录里，阅读页无法生成。")


def guide_lines() -> list[str]:
    lines = [
        "先按手头的数据和想保住的东西挑一条，再往下读它的公式、底层逻辑、物理意义和使用场景。",
        "",
        "## 先看哪一条",
        "",
    ]
    for item in GUIDE:
        names = "、".join(f"{get_loss(loss_id).name_zh}（`{loss_id}`）" for loss_id in item["losses"])
        lines.append(f"- **{item['goal']}**：先看{names}。{item['reason']}")
    lines.append("")
    return lines


def loss_lines(plugin) -> list[str]:
    bound = "有下界" if plugin.bounded_below else "无下界"
    lines = [
        f"### {plugin.name_zh}",
        "",
        f"{plugin.name} · {plugin.year} · 条目 `{plugin.id}` · {bound}",
        "",
        f"$${plugin.latex.strip()}$$",
        "",
        "**底层逻辑**",
        "",
        plugin.summary_zh.strip(),
        "",
        plugin.intuition_zh.strip(),
        "",
        "**物理意义**",
        "",
        plugin.physics_zh.strip(),
        "",
    ]
    if plugin.toy_note_zh.strip():
        lines.extend([plugin.toy_note_zh.strip(), ""])
    lines.extend([plugin.failure_zh.strip(), "", "**使用场景**", "", plugin.use_zh.strip(), ""])
    if plugin.paper_url:
        lines.append(f"论文：[{plugin.paper_title}]({plugin.paper_url})")
    else:
        lines.append("论文：待补")
    if plugin.implementation_url:
        lines.append(f"实现：{plugin.implementation_url}")
    lines.append("")
    return lines


def page_markdown() -> str:
    grouped: dict[str, list] = {key: [] for key, _ in FAMILIES}
    for plugin in all_losses():
        grouped.setdefault(plugin.family, []).append(plugin)
    lines = [
        "# 大语言模型损失函数",
        "",
        "这一页按任务族阅读。交互看板只留当前损失的三句；这里把同一条损失的底层逻辑、物理意义和使用场景放在一起。公式保持英文 LaTeX。",
        "",
    ]
    lines.extend(guide_lines())
    for key, label in FAMILIES:
        lines.extend([f"## {label}", "", FAMILY_INTRO[key], ""])
        for plugin in grouped.get(key, []):
            lines.extend(loss_lines(plugin))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    PAGE_MD.write_text(page_markdown(), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(renderer()),
            str(PAGE_MD),
            "--template",
            "academic",
            "--out",
            str(PAGE_HTML),
            "--title",
            "大语言模型损失函数",
            "--eyebrow",
            "按模块阅读",
            "--lang",
            "zh-CN",
        ],
        check=True,
    )
    print(f"wrote {PAGE_MD.relative_to(ROOT)} and {PAGE_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
