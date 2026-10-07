"""Buy-and-hold comparisons from already granted public OHLC providers."""
from datetime import timedelta

from .alpaca_paper import FIRST_EXECUTION, decimal
from .config import TraderError, instant, iso
from .data import PublicData, calendar_session, protected
from .ledger import Ledger


def benchmarks(paper, runtime, grant, *, data=None):
    at = paper.clock()
    accounts = [e for e in paper.events(event='account') if instant(e['at']) >= FIRST_EXECUTION]
    if not accounts:
        return {s: {'state': 'PENDING', 'reason': 'PAPER_NOT_STARTED'} for s in ('SPY', 'QQQ', 'BTC')}
    first_day = min(e['at'][:10] for e in accounts)
    first_anchor = instant(first_day + 'T00:00:00Z')
    ledger = Ledger(runtime, grant, clock=paper.clock,
                    budget_root=paper.root.parent / 'trader-agent-budget')
    data = data or PublicData(ledger)
    output = {}
    with ledger.owner():
        for display, asset in (('SPY', 'SPY'), ('QQQ', 'QQQ'), ('BTC', 'BTC-USD')):
            try:
                grant.check(at)
                history = [e for e in paper.events(event='public_benchmark') if e['asset'] == display]
                if protected(asset, first_anchor, at):
                    raise TraderError('PROTECTED_BENCHMARK')
                if asset == 'BTC-USD':
                    anchor = at.replace(hour=13, minute=30, second=0, microsecond=0)
                    price, source = data.crypto(asset, anchor, anchor + timedelta(minutes=1), minute=True)
                    baseline_at = first_anchor.replace(hour=13, minute=30)
                    if history:
                        initial = history[0]['base_price']
                    else:
                        initial, _ = data.crypto(asset, baseline_at, baseline_at + timedelta(minutes=1), minute=True)
                else:
                    bars, source = data.yahoo(asset, first_anchor - timedelta(days=1), at)
                    closes = [b for b in bars if calendar_session(b['bar_open_at'].date().isoformat())
                              and calendar_session(b['bar_open_at'].date().isoformat()).close_at <= at]
                    if not closes:
                        raise TraderError('MISSING_BENCHMARK_PRICE')
                    price = sorted(closes, key=lambda b: b['bar_open_at'])[-1]['close']
                    session = calendar_session(first_day)
                    if session is None:
                        raise TraderError('MISSING_BENCHMARK_SESSION')
                    baseline_at = session.open_at
                    first = [b for b in bars if b['bar_open_at'] == baseline_at]
                    if not first:
                        raise TraderError('MISSING_BENCHMARK_ENTRY')
                    initial = first[0]['open']
                price = decimal(price)
                if price <= 0:
                    raise TraderError('INVALID_BENCHMARK_PRICE')
                base = decimal(history[0]['base_price']) if history else decimal(initial)
                day_base = decimal(history[-1]['price']) if history else price
                output[display] = {'state': 'AVAILABLE', 'price': str(price), 'return_since_paper_start': str(price / base - 1),
                                   'return_since_previous_report': str(price / day_base - 1), 'base_at': iso(baseline_at)}
                paper.append('public_benchmark', None, asset=display, price=str(price), base_price=str(base), base_at=iso(baseline_at), source_digest=source['digest'])
            except TraderError as error:
                output[display] = {'state': 'PENDING', 'reason': error.code}
    return output
