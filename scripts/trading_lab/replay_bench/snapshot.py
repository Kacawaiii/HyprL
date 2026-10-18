"""InformationSnapshot(T) for the replay: what a live run at T could have seen, nothing else.

The context shown to the model holds no acquisition metadata (receipt times are after T and would reveal the replay
date); acquisition digests live in the manifest, hashed with the context.
"""
from datetime import timedelta
import random
import re
import statistics

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.trader_agent.config import AMENDED_CRYPTO
from scripts.trading_lab.trader_agent.data import price_summary
from scripts.trading_lab.trader_agent.news_queries import QUERIES

from .grant import ReplayError, instant, iso
from . import marketdata as md

STOCKS = ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD', 'INTC', 'JPM', 'GS', 'XOM', 'CVX',
          'LLY', 'UNH', 'PFE', 'WMT', 'COST', 'NKE', 'BA', 'CAT', 'GM', 'NFLX', 'DIS')
ETFS = ('XLK', 'XLF', 'XLE', 'XLI', 'XLV')
CRYPTO = tuple(AMENDED_CRYPTO)
BENCHMARKS = ('SPY', 'QQQ')
UNIVERSE = STOCKS + ETFS + CRYPTO
CLUSTERS = {**{s: 'tech' for s in ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD', 'INTC',
                                   'NFLX', 'XLK')},
            **{s: 'financials' for s in ('JPM', 'GS', 'XLF')}, **{s: 'energy' for s in ('XOM', 'CVX', 'XLE')},
            **{s: 'health' for s in ('LLY', 'UNH', 'PFE', 'XLV')},
            **{s: 'consumer' for s in ('WMT', 'COST', 'NKE', 'DIS', 'GM')},
            **{s: 'industrials' for s in ('BA', 'CAT', 'XLI')}, **{s: 'crypto' for s in CRYPTO}}
EXCLUDED_NEWS_SYMBOLS = {'BTCUSD', 'ETHUSD'}
NOT_SMALL_MID = set(UNIVERSE) | set(BENCHMARKS) | {'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'USO', 'UNG', 'SMH', 'SOXX',
                                                   'ARKK', 'VOO', 'IVV', 'VTI', 'EEM', 'EFA', 'HYG', 'LQD', 'XBI'}
TICKER = re.compile(r'^[A-Z]{1,5}$')
LIMITATIONS_A = ['Alpaca/Benzinga created_at is the publication clock; items revised before T are excluded',
                 'Raw OHLC; corporate actions may distort returns', 'Probabilities are uncalibrated analyst judgments',
                 'No web access in this run; the headline digest is the only news input']


def alpaca_news_symbol(asset):
    return asset.replace('-USD', 'USD')


def headline(item):
    return {'news_id': item['id'], 'title': item['headline'][:500], 'source': item['source'],
            'published_at': iso(instant(item['created_at'])), 'publication_clock': 'ALPACA_CREATED_AT',
            'url': item['url']}


def visible(item, T):
    """Causal and admissible: created and last updated strictly before T, never tagged BTC/ETH."""
    return (instant(item['created_at']) < T and instant(item['updated_at'] or item['created_at']) < T
            and item.get('url', '').startswith('https://') and not md.PROTECTED.intersection(item['symbols']))


def summary(asset, bars, T, crypto):
    rows = md.crypto_rows(bars) if crypto else md.session_rows(bars)
    rows = [r for r in rows if r['close_at'] <= T]
    try:
        return price_summary(rows, asset, T, crypto)
    except Exception as error:   # TraderError codes: MISSING_PRICES, PRICE_GAP_OR_STALE, ...
        return {'asset': asset, 'state': getattr(error, 'code', 'PRICE_INVALID')}


def context_a(T, bars, news_items):
    """The live v1 context shape (trader-context-v1 fields), built only from data available before T."""
    prices, exclusions = {}, []
    for asset in UNIVERSE + BENCHMARKS:
        prices[asset] = summary(asset, bars.get(asset, []), T, asset in CRYPTO)
        if prices[asset]['state'] != 'AVAILABLE':
            exclusions.append({'asset': asset, 'reason': prices[asset]['state']})
    eligible = [a for a in UNIVERSE if prices[a]['state'] == 'AVAILABLE']
    if not eligible or prices['SPY']['state'] != 'AVAILABLE':
        raise ReplayError('MISSING_PRICES', iso(T))
    window = [n for n in news_items if visible(n, T) and instant(n['created_at']) >= T - timedelta(hours=72)]
    window.sort(key=lambda n: (n['created_at'], n['id']), reverse=True)
    headlines, used = {}, set()
    for asset in eligible:
        symbol = alpaca_news_symbol(asset)
        items = [headline(n) for n in window if symbol in n['symbols']][:8]
        headlines[asset] = {'state': 'AVAILABLE', 'items': items}
        used.update(i['news_id'] for i in items)
    macro = [headline(n) for n in window if {'SPY', 'QQQ'}.intersection(n['symbols'])][:15]
    headlines['macro'] = {'state': 'AVAILABLE', 'items': macro}
    used.update(i['news_id'] for i in macro)
    context = {'schema': 'replay-context-v1', 'session': T.date().isoformat(), 'decision_time': iso(T),
               'price_cutoff': iso(T), 'universe': eligible, 'prices': prices, 'headlines': headlines,
               'archives': {'fomc': {'state': 'NOT_CONFIGURED', 'items': []},
                            'edgar': {'state': 'NOT_CONFIGURED', 'items': []}},
               'exclusions': exclusions, 'synthetic': False, 'limitations': LIMITATIONS_A}
    manifest = {'news': sorted([n['id'], n['created_at'], n['updated_at']] for n in window if n['id'] in used)}
    return context, manifest


def negative_score(text, keywords):
    text = text.lower()
    return sum(text.count(k) for k in keywords)


def context_c(T, window_start, news_items, bars, keywords, *, max_candidates=40, per_symbol=3):
    """Liquid small/mid caps with company-specific news since the previous close."""
    by_symbol = {}
    for n in news_items:
        if not visible(n, T) or instant(n['created_at']) < window_start or not 1 <= len(n['symbols']) <= 2:
            continue
        for symbol in n['symbols']:
            if TICKER.match(symbol) and symbol not in NOT_SMALL_MID and not symbol.startswith('XL'):
                by_symbol.setdefault(symbol, []).append(n)
    candidates, rejected = [], {}
    for symbol, items in by_symbol.items():
        rows = [r for r in md.session_rows(bars.get(symbol, [])) if r['close_at'] <= T]
        stats = small_mid(symbol, rows, T)
        if isinstance(stats, str):
            rejected[stats] = rejected.get(stats, 0) + 1
            continue
        scored = sorted(items, key=lambda n: (negative_score(n['headline'] + ' ' + (n['summary'] or ''), keywords),
                                              n['created_at']), reverse=True)
        score = sum(negative_score(n['headline'] + ' ' + (n['summary'] or ''), keywords) for n in items)
        candidates.append((score, max(n['created_at'] for n in items), symbol, stats, scored[:per_symbol]))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    shown = candidates[:max_candidates]
    context = {'schema': 'replay-news-context-v1', 'session': T.date().isoformat(), 'decision_time': iso(T),
               'news_window_start': iso(window_start),
               'candidates': [{'symbol': symbol, 'prices': stats,
                               'headlines': [{'news_id': n['id'], 'title': n['headline'][:500],
                                              'summary': (n['summary'] or '')[:400],
                                              'published_at': iso(instant(n['created_at'])), 'url': n['url']}
                                             for n in items]}
                              for _, _, symbol, stats, items in sorted(shown, key=lambda c: c[2])],
               'limitations': ['Small/mid is a liquidity proxy (20-session median dollar volume 10M..1B USD); no market cap',
                               'Benzinga created_at is the publication clock; items revised before T are excluded',
                               'No web access; the headlines are the only news input']}
    manifest = {'news': sorted([n['id'], n['created_at'], n['updated_at']] for _, _, _, _, items in shown for n in items),
                'eligible_candidates': len(candidates), 'shown': len(shown), 'rejected': rejected,
                'news_symbols_seen': len(by_symbol)}
    return context, manifest


def small_mid(symbol, rows, T):
    rows = sorted(rows, key=lambda r: r['bar_open_at'])
    if len(rows) < 21:
        return 'HISTORY_SHORT'
    expected = md.previous_close(T)
    if rows[-1]['close_at'] != expected:
        return 'STALE'
    last = rows[-21:]
    closes = [float(r['close']) for r in last]
    if closes[-1] < 5:
        return 'PRICE_LT_5'
    dollar = statistics.median(float(r['close']) * float(r['volume']) for r in last[1:])
    if not 10e6 < dollar <= 1e9:
        return 'DOLLAR_VOLUME_OUT_OF_BAND'
    daily = [closes[i] / closes[i - 1] - 1 for i in range(1, 21)]
    return {'recent_closes': closes, 'returns': {str(n): closes[-1] / closes[-1 - n] - 1 for n in (1, 5, 20)},
            'vol_20d': statistics.stdev(daily), 'median_dollar_volume_20d': dollar,
            'last_price_at': iso(last[-1]['close_at'])}


def validate(context, T):
    """Fail closed if anything shown is not strictly available at T."""
    if context['decision_time'] != iso(T):
        raise ReplayError('SNAPSHOT_TIME_MISMATCH')
    if context['schema'] == 'replay-context-v1':
        for asset, price in context['prices'].items():
            if price['state'] == 'AVAILABLE' and instant(price['last_price_at']) > T:
                raise ReplayError('FUTURE_PRICE', asset)
        for group in context['headlines'].values():
            for item in group['items']:
                if instant(item['published_at']) >= T:
                    raise ReplayError('FUTURE_NEWS', item['news_id'])
    else:
        for candidate in context['candidates']:
            if instant(candidate['prices']['last_price_at']) > T:
                raise ReplayError('FUTURE_PRICE', candidate['symbol'])
            for item in candidate['headlines']:
                if not instant(context['news_window_start']) <= instant(item['published_at']) < T:
                    raise ReplayError('FUTURE_NEWS', item['news_id'])
    return context


def seal(context, manifest, T):
    validate(context, T)
    context_hash = sha256_canonical(context)
    return {'context': context, 'manifest': manifest, 'context_hash': context_hash,
            'snapshot_hash': sha256_canonical({'context_hash': context_hash, 'manifest': manifest})}


# ---- anonymized robustness variant (Glasserman & Lin) ----

ALIASES = {'GOOGL': ['Alphabet', 'Google', 'YouTube', 'Waymo'], 'META': ['Meta Platforms', 'Meta', 'Facebook', 'Instagram',
           'WhatsApp'], 'AMZN': ['Amazon.com', 'Amazon', 'AWS'], 'MSFT': ['Microsoft', 'Azure'], 'AAPL': ['Apple', 'iPhone'],
           'NVDA': ['Nvidia', 'NVIDIA'], 'TSLA': ['Tesla', 'Musk'], 'AVGO': ['Broadcom'], 'AMD': ['Advanced Micro Devices', 'AMD'],
           'INTC': ['Intel'], 'JPM': ['JPMorgan Chase', 'JPMorgan', 'JP Morgan', 'Dimon'], 'GS': ['Goldman Sachs', 'Goldman'],
           'XOM': ['Exxon Mobil', 'ExxonMobil', 'Exxon'], 'CVX': ['Chevron'], 'LLY': ['Eli Lilly', 'Lilly', 'Mounjaro', 'Zepbound'],
           'UNH': ['UnitedHealth Group', 'UnitedHealth', 'Optum'], 'PFE': ['Pfizer'], 'WMT': ['Walmart'], 'COST': ['Costco'],
           'NKE': ['Nike'], 'BA': ['Boeing'], 'CAT': ['Caterpillar'], 'GM': ['General Motors', 'GM'], 'NFLX': ['Netflix'],
           'DIS': ['Walt Disney', 'Disney'], 'SOL-USD': ['Solana', 'SOL'], 'AVAX-USD': ['Avalanche', 'AVAX'],
           'LINK-USD': ['Chainlink', 'LINK'], 'DOGE-USD': ['Dogecoin', 'DOGE'], 'LTC-USD': ['Litecoin', 'LTC']}


def anonymize(sealed, seed):
    """Stable per-day codes for every asset, names masked in headlines, URLs opaque, price levels rescaled."""
    context = sealed['context']
    rng = random.Random(seed)
    stocks, etfs, coins = list(STOCKS), list(ETFS), list(CRYPTO)
    for group in (stocks, etfs, coins):
        rng.shuffle(group)
    codes = {**{a: f'STOCK_{i + 1:02d}' for i, a in enumerate(stocks)},
             **{a: f'SECTOR_ETF_{i + 1}' for i, a in enumerate(etfs)},
             **{a: f'COIN_{i + 1}' for i, a in enumerate(coins)}}
    patterns = []
    for asset, code in codes.items():
        names = ALIASES.get(asset, []) + [asset, md.crypto_pair(asset), alpaca_news_symbol(asset)]
        for q in [QUERIES.get(asset, '').strip('"')]:
            if q and asset not in ETFS:
                names.append(q)
        for name in sorted(set(names), key=len, reverse=True):
            patterns.append((re.compile(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', re.I), code))

    def mask(text):
        for pattern, code in patterns:
            text = pattern.sub(code, text)
        return text

    prices = {}
    for asset, price in context['prices'].items():
        name = codes.get(asset, asset)
        price = dict(price, asset=name)
        if price['state'] == 'AVAILABLE' and asset in codes:
            factor = rng.uniform(0.5, 2.0)
            price['recent_closes'] = [round(c * factor, 6) for c in price['recent_closes']]
        prices[name] = price
    headlines = {}
    for asset, group in context['headlines'].items():
        headlines[codes.get(asset, asset)] = {'state': group['state'], 'items': [
            dict(item, title=mask(item['title']), url=f"https://news.invalid/{item['news_id']}", source='newswire')
            for item in group['items']]}
    anon = dict(context, universe=[codes[a] for a in context['universe']], prices=prices, headlines=headlines,
                exclusions=[dict(e, asset=codes.get(e['asset'], e['asset'])) for e in context['exclusions']],
                limitations=context['limitations'] + ['Identities are masked: tickers, names and price levels are coded'])
    return anon, {code: asset for asset, code in codes.items()}
