"""Build the offline paper index and the loss catalog.

The unlearning entries come from the sibling awesome-llm-unlearning README.
Alignment and distillation entries that are outside that list are curated here
by title and URL. Nothing is copied from the awesome-loss-functions prose.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import all_losses  # noqa: E402
from core.guide import GUIDE  # noqa: E402

ENTRY_RE = re.compile(r"^- \[(?P<title>.+?)\]\((?P<url>.+?)\)\s*$")
URL_RE = re.compile(r"https?://[^\s)]+")
TAG_RULES = [
    ("gradient-ascent", re.compile(r"gradient ascent|\bGA\b", re.I)),
    ("npo", re.compile(r"negative preference|\bNPO\b|SimNPO", re.I)),
    ("dpo", re.compile(r"\bDPO\b|direct preference", re.I)),
    ("simpo", re.compile(r"\bSimPO\b", re.I)),
    ("kto", re.compile(r"\bKTO\b", re.I)),
    ("orpo", re.compile(r"\bORPO\b", re.I)),
    ("grpo", re.compile(r"\bGRPO\b", re.I)),
    ("ipo", re.compile(r"\bIPO\b", re.I)),
    ("rmu", re.compile(r"\bRMU\b|representation misdirection", re.I)),
    ("kl", re.compile(r"\bKL\b|Kullback|divergence", re.I)),
    ("distillation", re.compile(r"distill", re.I)),
    ("jensen", re.compile(r"Jensen|JensUn|\bJSD\b", re.I)),
    ("blur", re.compile(r"\bBLUR\b|bi-level", re.I)),
    ("shred", re.compile(r"\bSHRED\b|logit demotion", re.I)),
    ("unilogit", re.compile(r"Unilogit|uniform-target", re.I)),
    ("harry-potter", re.compile(r"Harry Potter", re.I)),
    ("preference", re.compile(r"preference optimization|\bRLHF\b", re.I)),
    ("reweight", re.compile(r"reweight|SatImp|saturation", re.I)),
]


class Paper(BaseModel):
    title: str
    url: str
    authors: str = ""
    date: str = ""
    venue: str = ""
    code: str = ""
    code_url: str = ""
    year: int = 0
    section: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str = "awesome-llm-unlearning"


CURATED = [
    Paper(title="A Neural Probabilistic Language Model", url="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf", authors="Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin", year=2003, venue="JMLR", tags=["language-modeling"], source="curated"),
    Paper(title="Rethinking the Inception Architecture for Computer Vision", url="https://arxiv.org/abs/1512.00567", authors="Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, Zbigniew Wojna", year=2016, venue="CVPR", tags=["label-smoothing"], source="curated"),
    Paper(title="UL2: Unifying Language Learning Paradigms", url="https://arxiv.org/abs/2205.05131", authors="Yi Tay et al.", year=2022, tags=["language-modeling"], source="curated"),
    Paper(title="Proximal Policy Optimization Algorithms", url="https://arxiv.org/abs/1707.06347", authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov", year=2017, tags=["ppo", "rl"], source="curated"),
    Paper(title="Training language models to follow instructions with human feedback", url="https://arxiv.org/abs/2203.02155", authors="Long Ouyang et al.", year=2022, tags=["rlhf", "preference"], source="curated"),
    Paper(title="Deep Reinforcement Learning from Human Preferences", url="https://arxiv.org/abs/1706.03741", authors="Paul F. Christiano et al.", year=2017, tags=["reward-model", "preference"], source="curated"),
    Paper(title="SLiC-HF: Sequence Likelihood Calibration with Human Feedback", url="https://arxiv.org/abs/2305.10425", authors="Yao Zhao et al.", year=2023, tags=["slic", "preference"], source="curated"),
    Paper(title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model", url="https://arxiv.org/abs/2305.18290", authors="Rafael Rafailov et al.", year=2023, tags=["dpo", "preference"], source="curated"),
    Paper(title="A General Theoretical Paradigm to Understand Learning from Human Preferences", url="https://arxiv.org/abs/2310.12036", authors="Mohammad Gheshlaghi Azar et al.", year=2023, tags=["ipo", "preference"], source="curated"),
    Paper(title="KTO: Model Alignment as Prospect Theoretic Optimization", url="https://arxiv.org/abs/2402.01306", authors="Kawin Ethayarajh et al.", year=2024, tags=["kto", "preference"], source="curated"),
    Paper(title="ORPO: Monolithic Preference Optimization without Reference Model", url="https://arxiv.org/abs/2403.07691", authors="Jiwoo Hong, Noah Lee, James Thorne", year=2024, tags=["orpo", "preference"], source="curated"),
    Paper(title="SimPO: Simple Preference Optimization with a Reference-Free Reward", url="https://arxiv.org/abs/2405.14734", authors="Yu Meng, Mengzhou Xia, Danqi Chen", year=2024, tags=["simpo", "preference"], source="curated"),
    Paper(title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", url="https://arxiv.org/abs/2402.03300", authors="Zhihong Shao et al.", year=2024, tags=["grpo", "rl"], source="curated"),
    Paper(title="Distilling the Knowledge in a Neural Network", url="https://arxiv.org/abs/1503.02531", authors="Geoffrey Hinton, Oriol Vinyals, Jeff Dean", year=2015, tags=["distillation", "kl"], source="curated"),
    Paper(title="MiniLLM: Knowledge Distillation of Large Language Models", url="https://arxiv.org/abs/2306.08543", authors="Yuxian Gu, Li Dong, Furu Wei, Minlie Huang", year=2023, tags=["distillation", "reverse-kl"], source="curated"),
]


def _code_url(raw: str) -> str:
    if not raw or raw.strip() == "-":
        return ""
    for url in URL_RE.findall(raw):
        if "shields.io" not in url:
            return url
    return ""


def _tags(title: str) -> list[str]:
    found = []
    for name, pattern in TAG_RULES:
        if pattern.search(title):
            found.append(name)
    return found


def parse_unlearning_readme(text: str) -> list[Paper]:
    papers: list[Paper] = []
    current: dict | None = None
    section = ""
    year = 0

    def flush():
        nonlocal current
        if current and current.get("title") and current.get("url"):
            if current["authors"] or current["date"]:
                current["tags"] = _tags(current["title"])
                papers.append(Paper(**current))
        current = None

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue
        year_match = re.match(r"^### (\d{4})\s*$", line)
        if year_match:
            flush()
            year = int(year_match.group(1))
            continue
        entry = ENTRY_RE.match(line)
        if entry:
            flush()
            current = {
                "title": entry.group("title").strip(),
                "url": entry.group("url").strip(),
                "authors": "",
                "date": "",
                "venue": "",
                "code": "",
                "code_url": "",
                "year": year,
                "section": section,
                "source": "awesome-llm-unlearning",
            }
            continue
        if current is None or not line.startswith("  - "):
            continue
        if ": " not in line:
            continue
        label, value = line[4:].split(": ", 1)
        key = {"Author(s)": "authors", "Date": "date", "Venue": "venue", "Code": "code"}.get(label)
        if key:
            current[key] = value.strip()
            if key == "code":
                current["code_url"] = _code_url(value)
    flush()
    return papers


def merge_curated(papers: list[Paper]) -> list[Paper]:
    seen = {paper.url for paper in papers}
    merged = list(papers)
    for paper in CURATED:
        if paper.url not in seen:
            merged.append(paper)
            seen.add(paper.url)
    return merged


def write_loss_docs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for plugin in all_losses():
        lines = [
            f"# {plugin.name_zh} ({plugin.name})",
            "",
            f"- 任务族: {plugin.family_zh()}",
            f"- 年份: {plugin.year}",
            f"- 下界: {'有' if plugin.bounded_below else '无'}",
            f"- 论文: [{plugin.paper_title}]({plugin.paper_url})" if plugin.paper_url else "- 论文: 待补",
            "",
            "## 公式",
            "",
            "$$",
            plugin.latex,
            "$$",
            "",
            plugin.summary_zh,
            "",
            "## 直觉",
            "",
            plugin.intuition_zh,
            "",
            "## 物理意义",
            "",
            plugin.physics_zh,
            "",
            "## 失效",
            "",
            plugin.failure_zh,
            "",
        ]
        if plugin.toy_note_zh:
            lines.extend(["## 玩具坐标", "", plugin.toy_note_zh, ""])
        if plugin.implementation_url:
            lines.extend([f"实现: {plugin.implementation_url}", ""])
        (output / f"{plugin.id}.md").write_text("\n".join(lines), encoding="utf-8")

    guide_lines = ["# 损失选择指南", ""]
    for item in GUIDE:
        names = ", ".join(f"`{loss_id}`" for loss_id in item["losses"])
        guide_lines.extend([f"## {item['goal']}", "", f"先看 {names}。", "", item["reason"], ""])
    (output.parent / "selection-guide.md").write_text("\n".join(guide_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=Path, default=ROOT.parent / "awesome-llm-unlearning" / "README.md")
    args = parser.parse_args()
    text = args.readme.read_text(encoding="utf-8")
    papers = merge_curated(parse_unlearning_readme(text))
    payload = {
        "source_readme": str(args.readme),
        "count": len(papers),
        "papers": [paper.model_dump() for paper in papers],
    }
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "paper_catalog.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    loss_payload = {"count": len(all_losses()), "losses": [plugin.to_catalog() for plugin in all_losses()]}
    (data_dir / "loss_catalog.json").write_text(json.dumps(loss_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_loss_docs(ROOT / "docs" / "losses")
    print(f"papers={payload['count']} losses={loss_payload['count']}")


if __name__ == "__main__":
    main()
