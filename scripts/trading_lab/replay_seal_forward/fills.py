"""Fill clocks and full trade costs, shared by the replay and the forward scorer.

Stocks enter at the actual session open + 30 s (daylight saving moves it with the exchange).
Crypto enters at the decision time + 15 min: after the 10 min forward dispatch window, so no
fill precedes a stored decision. Trade P&L charges both legs of the SPY-relative position,
borrow on the short leg, excludes hard-to-borrow shorts and drops corporate-action windows.
"""
from datetime import timedelta

from scripts.trading_lab.replay_bench import marketdata as md
from scripts.trading_lab.replay_bench.grant import ReplayError, instant

STOCK_ENTRY_DELAY = timedelta(seconds=30)
CRYPTO_ENTRY_OFFSET = timedelta(minutes=15)
DISPATCH_WINDOW = timedelta(minutes=10)
BORROW_RATE = 0.03
HEDGE = 'SPY'


def stock_entry(day):
    return md.sessions(day, day)[0].open_at + STOCK_ENTRY_DELAY


def crypto_entry(T):
    return T + CRYPTO_ENTRY_OFFSET


def entry(day, T, crypto):
    """The designed fill; refused unless strictly after the decision and its dispatch window."""
    at = crypto_entry(T) if crypto else stock_entry(day)
    if at <= T + DISPATCH_WINDOW:
        raise ReplayError('ENTRY_NOT_AFTER_DECISION', day)
    return at


def hedge_cost(costs):
    """Round trip of the SPY leg: same fee, the >$1B liquidity tier."""
    return 2 * (costs['fee_per_side'] + costs['spread_slippage_per_side']['dollar_volume_gt_1B'])


def borrow_cost(entry_day, exit_day):
    """3 %/yr on the short leg, calendar days held, at least one day."""
    days = max(1, (instant(exit_day + 'T00:00:00Z') - instant(entry_day + 'T00:00:00Z')).days)
    return BORROW_RATE * days / 365


def hard_to_borrow(flags):
    """Alpaca asset flags say so explicitly. Missing flags (crypto, absent assets) are not invented."""
    return bool(flags) and (flags.get('shortable') is False or flags.get('easy_to_borrow') is False)


def adjustment_ratios(raw_rows, adjusted_rows):
    raw = {r['t'][:10]: r['c'] for r in raw_rows}
    return {d: raw[d] / r['c'] for r in adjusted_rows for d in [r['t'][:10]] if d in raw and r['c']}


def corporate_action(ratios, first, last):
    """True when the raw/adjusted ratio moves inside (first, last]: a split or dividend took effect."""
    days = sorted(d for d in ratios if first <= d <= last)
    if not days or days[0] != first or days[-1] != last:
        return None
    values = [ratios[d] for d in days]
    return max(values) / min(values) - 1 > 1e-6
