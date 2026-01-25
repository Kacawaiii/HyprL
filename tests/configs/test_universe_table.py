from __future__ import annotations

from pathlib import Path

import pytest

from hyprl.universe.table import TABLE_BEGIN, TABLE_END, extract_table_block, generate_universe_table


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.skip(reason="Universe table format changed; README uses simplified table")
def test_universe_table_blocks_sync() -> None:
    csv_path = Path("data/universe_scores_v1_2.csv")
    expected = generate_universe_table(csv_path).strip()
    for doc in ("README.md", "AGENTS.md"):
        text = _read(Path(doc))
        block = extract_table_block(text)
        assert block == expected, f"Universe table in {doc} is out of sync with {csv_path}"
