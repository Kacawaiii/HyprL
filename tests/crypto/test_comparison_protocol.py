"""The preregistered comparison protocol: committed artifact, hash, and its own invariants."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from scripts.trading_lab import comparison_protocol as cp
from scripts.trading_lab import research_protection as rp
from scripts.trading_lab.sources.canonical import sha256_canonical

REPO = Path(__file__).resolve().parents[2]
UTC = timezone.utc
ARTIFACT = REPO / "docs/artifacts/comparison_protocol_v1.json"
DOC = REPO / "docs/COMPARISON_PROTOCOL_V1.md"


def committed():
    return json.loads(ARTIFACT.read_text())


def test_committed_artifact_is_the_generated_protocol_and_hash_is_canonical():
    data = committed()
    assert data == json.loads(json.dumps(cp.build()))
    body = {k: v for k, v in data.items() if k != "protocol_hash"}
    assert sha256_canonical(body) == data["protocol_hash"]
    assert data["results_exist"] is False and data["status"] == "PREREGISTERED_BEFORE_ANY_RESULT"
    assert data["no_action"] == {"capture": False, "training": False, "trading": False, "requests_sent": 0,
                                 "protected_intervals": "closed and untouched"}
    assert data["protocol_hash"] in DOC.read_text()


def test_every_expected_decision_is_clear_of_every_protected_interval():
    for kind, decisions, products in (("crypto", cp.crypto_decisions(), ("BTC-USD", "ETH-USD")),
                                      ("equity", cp.equity_decisions(), ("AAPL", "MSFT", "NVDA", "QQQ"))):
        assert decisions and decisions == sorted(set(decisions))
        for product in products:
            for T in decisions:
                flags = rp.protection_flags(product, T)
                assert flags == {"inside": [], "event_window_touches": []}, (product, T)
        assert decisions[0] >= cp.EVAL_START and decisions[-1] < cp.EVAL_END
    # label windows end inside the series, features start inside it
    assert cp.CRYPTO_SERIES_START >= rp.crypto_interval().end_exclusive
    assert datetime.fromisoformat(cp.EQUITY_SERIES_START).replace(tzinfo=UTC) >= rp.equity_interval().end_exclusive


def test_warmups_fit_before_the_evaluation_period():
    crypto = rp.crypto_ranges(cp.CRYPTO_SERIES_START, cp.EVAL_END)["runs"][0]
    assert datetime.fromisoformat(crypto["first_decision_at"].replace("Z", "+00:00")) <= cp.EVAL_START
    equity = rp.equity_ranges(cp.EQUITY_SERIES_START, cp.EQUITY_SERIES_END)["runs"][0]
    assert datetime.fromisoformat(equity["first_decision_at"].replace("Z", "+00:00")) <= cp.EVAL_START
    assert cp.EVAL_START == cp.CAPTURE_START + timedelta(days=30)  # the event warm-up


def test_split_purges_label_overlap_and_counts_match_the_data():
    counts = cp.counts()
    crypto, equity = counts["crypto_per_product"], counts["equity_per_instrument"]
    assert crypto["train"] + crypto["purged"] + crypto["test"] == len(cp.crypto_decisions())
    assert equity["train"] + equity["purged"] + equity["test"] == len(cp.equity_decisions())
    assert crypto["purged"] == 3 and equity["purged"] == 5  # label horizon 4 bars / 5 sessions minus the grid step
    assert crypto["expected_to_meet_minimum"] is True
    # stated, not hidden: at this period the equity family cannot reach its minimum sample
    assert equity["expected_to_meet_minimum"] is False


def test_budgets_follow_the_cadences_and_periods():
    b = cp.budgets()
    days = 153
    assert (cp.CAPTURE_END - cp.CAPTURE_START).days == days == b["capture_window"]["days"]
    assert b["fomc"]["feed_polls"] == days * 1440 and b["edgar"]["listings"] == 3 * days * 144
    assert b["fomc"]["statement_pages_nominal"] == 16 * 5
    assert b["fomc"]["total_request_ceiling"] == b["fomc"]["feed_polls"] + 16 * 5 * 6
    assert b["prices"]["coinbase"]["requests"] == 2 * 10 and b["prices"]["coinbase"]["bars_per_product"] == 2976
    assert b["edgar"]["total_request_ceiling"] == b["edgar"]["listings"] + 1


def test_both_variants_share_everything_but_the_event_columns():
    data = committed()
    columns = data["pairing"]["features"]
    for product, events in columns["events"].items():
        assert events and len(events) == len(set(events))
    assert columns["events"]["QQQ"] == columns["events"]["BTC-USD"]  # FOMC only: no 8-K population, no SEC issuer
    assert set(columns["events"]["AAPL"]) > set(columns["events"]["QQQ"])
    reused = data["pairing"]["frozen_reused"]
    from scripts.trading_lab.economic_backtest import ExecutionSpec
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1
    assert reused["signal_spec_hash"] == SIGNAL_SPEC_V1.spec_hash
    assert reused["risk_spec_hash"] == RISK_SPEC_V1.risk_spec_hash
    assert reused["execution_spec_hash"] == ExecutionSpec().execution_spec_hash
