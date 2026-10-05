import json

import pytest

from scripts.trading_lab.trader_agent.config import Authorization, instant
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.service import TraderService
from scripts.trading_lab.trader_agent.synthetic import Clock, SyntheticData, SyntheticRunner


@pytest.fixture
def grant(tmp_path):
    models = {role: {"cli": "codex" if role == "analyst_gpt" else "claude",
              "model": "gpt-synthetic-test" if role == "analyst_gpt" else "sonnet" if role == "reviewer" else "opus",
              "tools": ["WebSearch", "WebFetch"], "calls_per_day": 1, "retries_per_day": 1, "timeout_minutes": 1}
              for role in ("analyst_claude", "analyst_gpt", "reviewer")}
    payload = {"mode": "PAPER_SHADOW_ONLY", "granted_at": "2026-01-01T00:00:00Z", "not_after": "2027-02-01T00:00:00Z",
        "external_models": models,
        "data_sources": {"yahoo_chart": {"host": "query1.finance.yahoo.com", "max_requests_per_day": 80},
                         "coinbase_exchange_public": {"host": "api.exchange.coinbase.com", "max_requests_per_day": 20},
                         "gdelt_doc_api": {"host": "api.gdeltproject.org", "max_requests_per_day": 80, "min_spacing_seconds": 6}},
        "universe": {"stocks": ["AAPL", "MSFT"], "sector_etfs": ["XLK"], "crypto": ["BTC-USD", "ETH-USD"],
                     "benchmarks_not_predicted": ["SPY", "QQQ"]},
        "budgets": {"max_runs_per_day": 1, "max_llm_calls_per_day": 6, "paper_capital_usd": 100000},
        "cadence": {"runs_per_us_trading_day": 1}}
    path = tmp_path / "synthetic-grant.json"
    path.write_text(json.dumps(payload))
    return Authorization.load(path)


@pytest.fixture
def clock():
    return Clock(instant("2026-10-06T12:00:00Z"))


@pytest.fixture
def ledger(tmp_path, grant, clock):
    return Ledger(tmp_path / "private", grant, clock=clock)


@pytest.fixture
def service(ledger, clock):
    return TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
