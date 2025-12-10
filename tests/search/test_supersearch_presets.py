from __future__ import annotations

import pytest

from hyprl.configs import get_supersearch_preset
from hyprl.search.optimizer import SearchConfig, _apply_constraint_preset


def test_get_supersearch_preset_research_minimal() -> None:
    preset = get_supersearch_preset("research_minimal")
    assert pytest.approx(1.2, rel=1e-6) == float(preset["min_pf"])
    assert float(preset["max_dd"]) == pytest.approx(0.30)


def test_apply_constraint_preset_overrides_fields() -> None:
    cfg = SearchConfig(ticker="TEST", period="6mo", constraint_preset="api_candidate")
    updated = _apply_constraint_preset(cfg)
    assert updated is not cfg
    assert updated.min_profit_factor == pytest.approx(1.4)
    assert updated.min_trades == 80
    assert updated.min_robustness_score == pytest.approx(0.65)
    assert updated.max_drawdown_pct == pytest.approx(0.25)