"""The crypto core must not acquire a pandas dependency by adding equities.

Phase 6D introduces a calendar that needs ``pandas_market_calendars``, which
needs pandas. That is fine for equities and unacceptable for everything else:
the shadow runtime, the paper portfolio and the whole Phase 1-5 core install
run without it today, and a stray module-scope import would end that quietly --
quietly because the library *is* installed on this machine, so nothing would
fail here. It would fail on a machine that installed only the core extra.

So, like the ML contract, the probe runs in a fresh interpreter with pandas
actively blocked. "Installed and unused" and "not required" are different
claims, and only the second one is worth anything.

The same probe checks the other half: with pandas blocked, asking for the
equity calendar must produce a clear error naming the missing extra, and never
fall back to the crypto calendar.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Every module the crypto core, the API and the paper runtime load. If any of
# them reaches pandas at import time, this list is where it shows up.
CORE_MODULES = (
    "instruments",
    "instrument_registry",
    "identity",
    "trading_calendar",
    "market_providers",
    "market_bar",
    "coinbase_candles",
    "market_data_store",
    "market_snapshots",
    "market_series",
    "market_dataset",
    "credentials",
    "equity_market",
    "massive_provider",
    "massive_http_transport",
    "equity_corpus",
    "capture_us_equity_corpus",
    "verify_us_equity_corpus",
    "paper_portfolio",
    "paper_engine",
    "protected_holdout",
    "app_api.service",
    "app_api.server",
)

_BLOCKER = '''
import sys

class _Blocked:
    """Refuse pandas and the calendar library without uninstalling them."""

    # numpy is deliberately absent: the core genuinely depends on it, and
    # blocking it would test a claim this project has never made.
    BANNED = {"pandas", "pandas_market_calendars", "exchange_calendars"}

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BANNED:
            raise ImportError(f"blocked for this probe: {name}")
        return None

sys.meta_path.insert(0, _Blocked())
'''


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-"], input=_BLOCKER + script,
                          capture_output=True, text=True, cwd=str(REPO_ROOT),
                          timeout=300)


def test_the_blocker_itself_actually_blocks() -> None:
    """A probe that cannot fail proves nothing."""
    result = _run("import pandas\n")
    assert result.returncode != 0
    assert "blocked for this probe" in result.stderr
    assert _run("import pandas_market_calendars\n").returncode != 0


def test_the_core_imports_with_pandas_blocked() -> None:
    script = "\n".join(
        f"import scripts.trading_lab.{name}" for name in CORE_MODULES
    ) + '''
import sys
leaked = sorted(n for n in sys.modules
                if n.split(".")[0] in {"pandas", "pandas_market_calendars"})
assert not leaked, leaked
print("CORE-OK")
'''
    result = _run(script)
    assert result.returncode == 0, result.stderr
    assert "CORE-OK" in result.stdout


def test_the_registry_still_lists_the_equities_without_pandas() -> None:
    """Describing a market must not require the library that schedules it.

    The instrument metadata, the venue and the calendar *name* are plain data.
    Only the schedule itself needs the extra, so browsing the catalogue works
    on a core install and merely cannot answer session questions.
    """
    result = _run('''
from scripts.trading_lab.instrument_registry import CATALOGUE_V1, EQUITY_INSTRUMENTS_V1
assert len(EQUITY_INSTRUMENTS_V1) == 4
assert "xnas:AAPL" in CATALOGUE_V1.ids()
spec = CATALOGUE_V1.resolve("xnas:QQQ")
assert spec.asset_class == "ETF"
assert spec.trading_calendar == "US_EQUITY_REGULAR"
print("REGISTRY-OK")
''')
    assert result.returncode == 0, result.stderr
    assert "REGISTRY-OK" in result.stdout


def test_the_equity_provider_is_usable_without_pandas() -> None:
    """Bars, splits and dividends are Decimal arithmetic, not dataframes."""
    result = _run('''
from decimal import Decimal
from scripts.trading_lab.equity_market import (
    ADJUSTMENT_RAW, StockSplit, apply_splits, build_equity_bar)
from scripts.trading_lab.instruments import InstrumentId

aapl = InstrumentId(venue="xnas", symbol="AAPL")
bar = build_equity_bar(
    instrument=aapl, timeframe="30m", provider_id="massive-stocks-historical-v1",
    adjustment_policy=ADJUSTMENT_RAW, bar_open_at="2026-06-09T14:30:00Z",
    bar_close_at="2026-06-09T15:00:00Z", open="400.00", high="400.00",
    low="400.00", close="400.00", volume="1000", session_date="2026-06-09")
split = StockSplit(instrument_id=aapl, effective_date="2026-06-10",
                   ratio_numerator=4, ratio_denominator=1)
assert apply_splits([bar], [split])[0].close == Decimal("100")
print("PROVIDER-OK")
''')
    assert result.returncode == 0, result.stderr
    assert "PROVIDER-OK" in result.stdout


def test_the_capture_contract_declares_itself_without_pandas() -> None:
    """The corpus identity is verifiable on a core install. The grid is not.

    A useful split, and not an accident. The calendar's *spec* -- which
    library, which version, which calendar, which session type -- is pure
    metadata, so `corpus_spec_hash` can be recomputed by anyone auditing what
    was requested, without installing anything. The calendar's *schedule* is
    the part that needs the extra, so `sessions()` and `expected_bar_opens()`
    refuse rather than returning an empty grid that would make every bar look
    unexpected and every gap look real.
    """
    result = _run("""
from scripts.trading_lab.equity_corpus import CORPUS_SPEC_V1

assert CORPUS_SPEC_V1.instruments == (
    "xnas:AAPL", "xnas:MSFT", "xnas:NVDA", "xnas:QQQ")
assert CORPUS_SPEC_V1.adjustment_policy == "SPLIT_ADJUSTED"
assert CORPUS_SPEC_V1.timeframe == "30m"
assert CORPUS_SPEC_V1.session_type == "REGULAR"
assert CORPUS_SPEC_V1.requested_start == "2024-08-01"
assert CORPUS_SPEC_V1.requested_end == "2026-07-31"
assert CORPUS_SPEC_V1.calendar_id == "US_EQUITY_REGULAR"

# The identity is computable: it is metadata about the rules, not the rules.
payload = CORPUS_SPEC_V1.canonical()
assert payload["calendar_spec_hash"] == (
    "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314")
assert payload["calendar_dependency_version"] == "5.4.0"
assert CORPUS_SPEC_V1.corpus_spec_hash == (
    "93cfdb1a749bfa1de5c69c5dced2908413cfb8bba7f3f663fd9c747151b1b5ed")

# The schedule is not. An empty grid would make every captured bar look
# unexpected and every absent bar look like a real gap.
for attempt in (CORPUS_SPEC_V1.expected_bar_opens, CORPUS_SPEC_V1.sessions):
    try:
        attempt()
    except Exception as error:
        assert "equities" in str(error), str(error)
    else:
        raise AssertionError(f"{attempt} produced a schedule without a calendar")
print("CORPUS-OK")
""")
    assert result.returncode == 0, result.stderr
    assert "CORPUS-OK" in result.stdout


def test_the_missing_extra_produces_a_clear_error_not_a_wrong_calendar() -> None:
    """The failure mode this whole separation exists to prevent.

    Falling back to the crypto calendar would annualise a 30-minute equity
    series by 8760 instead of about 3263 and inflate every Sharpe ratio by
    roughly 1.6x. So the missing extra must raise, and the message must name
    the extra rather than leaving someone to guess at a pandas ImportError.
    """
    result = _run('''
from scripts.trading_lab.trading_calendar import (
    CRYPTO_247_CALENDAR, TradingCalendarError, calendar_available, get_calendar,
    known_calendars)

assert "US_EQUITY_REGULAR" in known_calendars()
assert calendar_available("CRYPTO_24_7") is True
assert calendar_available("US_EQUITY_REGULAR") is False

try:
    get_calendar("US_EQUITY_REGULAR")
except Exception as error:
    message = str(error)
    assert "equities" in message, message
    assert "pandas_market_calendars" in message, message
else:
    raise AssertionError("a missing extra must raise, not return a calendar")

# And emphatically not this.
assert CRYPTO_247_CALENDAR.annualization_periods("1h") == 8760
print("EXTRA-OK")
''')
    assert result.returncode == 0, result.stderr
    assert "EXTRA-OK" in result.stdout


def test_the_api_serves_the_crypto_endpoints_without_pandas() -> None:
    """A core install must still run the app, equities described but sessionless."""
    result = _run('''
from scripts.trading_lab.app_api.service import AppService

service = AppService.__new__(AppService)
universe = AppService._portfolio_instruments(service)
assert universe == ["coinbase:BTC-USD", "coinbase:ETH-USD"], universe
print("API-OK")
''')
    assert result.returncode == 0, result.stderr
    assert "API-OK" in result.stdout


def test_no_core_module_imports_pandas_at_module_scope() -> None:
    """A static check to catch the import before a probe has to.

    The runtime probes above are the real guarantee; this one points at the
    offending line instead of at a stack trace ten imports deep.
    """
    package = REPO_ROOT / "scripts" / "trading_lab"
    allowed = {"equity_calendar.py"}
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if path.name in allowed:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if "pandas" not in stripped:
                continue
            # An import nested inside a function is loaded on demand, which is
            # the whole point; only column-zero imports run at import time.
            if line[:1] not in {" ", "\t"}:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders, offenders
