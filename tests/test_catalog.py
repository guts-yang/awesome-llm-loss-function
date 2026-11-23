from scripts.sync_catalog import parse_unlearning_readme

SAMPLE = """
## Papers

### 2024

- [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)
  - Author(s): Ruiqi Zhang, Licong Lin, Yu Bai, Song Mei
  - Date: 2024-04
  - Venue: -
  - Code: [![GitHub](https://img.shields.io/badge/GitHub-1)](https://github.com/licong-lin/negative-preference-optimization)

## Contents

- [Papers](#papers)
"""


def test_parser_keeps_authored_entries_and_code_links():
    papers = parse_unlearning_readme(SAMPLE)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.year == 2024
    assert "Zhang" in paper.authors
    assert paper.code_url.startswith("https://github.com/licong-lin/")
    assert "npo" in paper.tags
    assert paper.source == "awesome-llm-unlearning"
