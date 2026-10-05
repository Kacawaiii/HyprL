"""Protected intervals per product and admissible decisions, derived from the contracts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.trading_lab import research_protection as rp
from scripts.trading_lab.equity_calendar import USEquityRegularCalendar
from scripts.trading_lab.equity_research import EQUITY_CONFIRMATORY_HOLDOUT_V1, EQUITY_RESEARCH_SPEC_V1
from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[2]
HOUR = timedelta(hours=1)


def test_protection_table_comes_from_the_contracts():
    table = rp.protection_table()
    assert sorted(table) == ["AAPL", "BTC-USD", "ETH-USD", "MSFT", "NVDA", "QQQ"]
    for product in ("BTC-USD", "ETH-USD"):
        (crypto,) = table[product]
        assert crypto["identity_hash"] == PROTECTED_WINDOW_V1.holdout_hash
        assert (crypto["start"], crypto["end_exclusive"]) == ("2026-09-01T00:00:00Z", "2026-12-01T00:00:00Z")
        assert crypto["last_protected_bar_open"] == PROTECTED_WINDOW_V1.end == "2026-11-30T23:00:00Z"
    for product in ("AAPL", "MSFT", "NVDA", "QQQ"):
        (equity,) = table[product]
        assert equity["identity_hash"] == EQUITY_CONFIRMATORY_HOLDOUT_V1.spec_hash
        assert equity["research_spec_hash"] == EQUITY_RESEARCH_SPEC_V1.spec_hash
        # the id says 2027q1, the contract starts 2026-12-01 and its last protected date is 2027-02-28
        assert (equity["start"], equity["end_exclusive"]) == ("2026-12-01T00:00:00Z", "2027-03-01T00:00:00Z")


def test_equity_interval_agrees_with_the_contract_membership_test():
    holdout = EQUITY_CONFIRMATORY_HOLDOUT_V1
    interval = rp.equity_interval()
    day = interval.start - timedelta(days=3)
    while day < interval.end_exclusive + timedelta(days=3):
        assert holdout.covers(day.date().isoformat()) == (interval.start <= day < interval.end_exclusive)
        day += timedelta(days=1)


def test_the_two_intervals_adjoin_without_overlap():
    assert rp.crypto_interval().end_exclusive == rp.equity_interval().start


def test_warmups_and_horizons_are_measured_not_restated():
    assert rp.crypto_price_warmup() == 25  # EMA 26 seed is the slowest feature
    assert rp.equity_price_warmup() == 20  # return over 20 sessions
    ranges = rp.crypto_ranges(datetime(2025, 8, 1, tzinfo=UTC), datetime(2027, 8, 1, tzinfo=UTC))
    assert ranges["label_horizon_bars"] == SIGNAL_SPEC_V1.prediction_horizon == 4
    assert rp.equity_ranges("2024-08-01", "2027-07-31")["label_horizon_sessions"] == 5


def test_crypto_ranges_match_a_brute_force_check_of_every_bar_read():
    start, end = datetime(2025, 8, 1, tzinfo=UTC), datetime(2027, 8, 1, tzinfo=UTC)
    ranges = rp.crypto_ranges(start, end)
    interval = rp.crypto_interval()
    expected, t = [], start
    while t < end:
        first, last = t - 25 * HOUR, t + 4 * HOUR  # bars behind the features, bars of the label window
        if first >= start and last < end and not interval.touches(first, last):
            expected.append(t)
        t += HOUR
    got = []
    for run in ranges["runs"]:
        t, last = (datetime.fromisoformat(run[k].replace("Z", "+00:00")) for k in ("first_bar_open", "last_bar_open"))
        while t <= last:
            got.append(t)
            t += HOUR
    assert got == expected
    assert [(r["first_decision_at"], r["last_decision_at"]) for r in ranges["runs"]] == [
        ("2025-08-02T02:00:00Z", "2026-08-31T20:00:00Z"), ("2026-12-02T02:00:00Z", "2027-07-31T20:00:00Z")]
    assert ranges["event_window_clean_from"] == "2026-12-31T00:00:00Z"


def test_equity_ranges_match_a_brute_force_check_of_every_session_read():
    ranges = rp.equity_ranges("2024-08-01", "2027-07-31")
    sessions = USEquityRegularCalendar().sessions_between(datetime(2024, 8, 1, tzinfo=UTC), datetime(2027, 7, 31, tzinfo=UTC))
    protected = [EQUITY_CONFIRMATORY_HOLDOUT_V1.covers(s.session_date) for s in sessions]
    expected = {s.close_at for i, s in enumerate(sessions)
                if i >= 20 and i + 5 < len(sessions) and not any(protected[i - 20:i + 6])}
    listed = set()
    for run in ranges["runs"]:
        listed |= {s.close_at for s in sessions
                   if run["first_decision_at"] <= s.close_at.isoformat().replace("+00:00", "Z") <= run["last_decision_at"]}
    assert listed == expected
    assert [(r["first_decision_at"], r["last_decision_at"]) for r in ranges["runs"]] == [
        ("2024-08-29T20:00:00Z", "2026-11-20T21:00:00Z"), ("2027-03-30T20:00:00Z", "2027-07-23T20:00:00Z")]
    assert ranges["event_window_clean_from"] == "2027-03-31T00:00:00Z"


def test_docs_cite_the_equity_holdout_as_the_contract_states_it():
    for path in sorted((REPO / "docs").glob("*.md")):
        text = path.read_text()
        assert "2027-Q1" not in text, path.name  # the id is equity_confirmatory_2027q1; the range is Dec 2026 - Feb 2027
    text = (REPO / "docs/EVENT_FEATURES_V1.md").read_text()
    assert "2026-12-01" in text and "2027-02-28" in text
