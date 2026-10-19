"""Anonymized control: identities, calendar dates and price levels are coded.

Fixes the parent mask: headline `symbols` are coded, news ids are renumbered (vendor ids are
sequential and date the story), every timestamp becomes a relative day number and UTC time, SPY/QQQ levels are rescaled, executives and products tied
to a ticker are masked, and ticker masks are case-sensitive and word-bounded so that ordinary
words such as "cost", "cat" or "link" survive.
"""
import random
import re

from scripts.trading_lab.trader_agent.news_queries import QUERIES
from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn
from scripts.trading_lab.replay_bench.grant import instant

PEOPLE_AND_PRODUCTS = {
    'AAPL': ['Tim Cook', 'iPhone', 'iPad', 'MacBook', 'Vision Pro', 'App Store', 'Apple Intelligence'],
    'MSFT': ['Satya Nadella', 'Nadella', 'Copilot', 'Xbox', 'LinkedIn', 'Activision'],
    'NVDA': ['Jensen Huang', 'GeForce', 'Blackwell', 'CUDA', 'DGX', 'Rubin GPU'],
    'AMZN': ['Andy Jassy', 'Jassy', 'Bezos', 'Alexa', 'Prime Video', 'Whole Foods', 'Kuiper'],
    'GOOGL': ['Sundar Pichai', 'Pichai', 'Gemini', 'Android', 'DeepMind'],
    'META': ['Mark Zuckerberg', 'Zuckerberg', 'Llama', 'Reality Labs', 'Oculus', 'Quest headset'],
    'TSLA': ['Elon Musk', 'Cybertruck', 'Model Y', 'Model 3', 'Robotaxi', 'Optimus', 'Autopilot', 'FSD', 'Gigafactory'],
    'AVGO': ['Hock Tan', 'VMware'],
    'AMD': ['Lisa Su', 'Ryzen', 'EPYC', 'Radeon', 'Xilinx'],
    'INTC': ['Lip-Bu Tan', 'Gelsinger', 'Xeon', 'Gaudi', 'Intel Foundry', 'Mobileye'],
    'JPM': ['Jamie Dimon', 'Chase Bank'],
    'GS': ['David Solomon', 'Marcus by Goldman'],
    'XOM': ['Darren Woods', 'Pioneer Natural'],
    'CVX': ['Mike Wirth', 'Hess'],
    'LLY': ['Dave Ricks', 'David Ricks', 'orforglipron', 'tirzepatide', 'donanemab', 'Kisunla'],
    'UNH': ['Stephen Hemsley', 'Andrew Witty', 'UnitedHealthcare', 'Change Healthcare'],
    'PFE': ['Albert Bourla', 'Bourla', 'Paxlovid', 'Comirnaty', 'Seagen'],
    'WMT': ['Doug McMillon', 'McMillon', 'John Furner', "Sam's Club"],
    'COST': ['Ron Vachris', 'Kirkland Signature'],
    'NKE': ['Elliott Hill', 'Air Jordan', 'Converse'],
    'BA': ['Kelly Ortberg', 'Ortberg', '737 MAX', '787 Dreamliner', 'Dreamliner', 'Starliner', 'Spirit AeroSystems'],
    'CAT': ['Jim Umpleby', 'Joe Creed'],
    'GM': ['Mary Barra', 'Chevrolet', 'Cadillac', 'Chevy', 'GMC'],
    'NFLX': ['Ted Sarandos', 'Greg Peters'],
    'DIS': ['Bob Iger', 'Iger', 'Disney+', 'ESPN', 'Hulu', 'Pixar', 'Marvel Studios', 'Lucasfilm'],
    'SOL-USD': ['Anatoly Yakovenko'],
    'DOGE-USD': ['dogecoin'],
}
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2}))?$')
NEUTRAL_LIMITATIONS = ['Identities, calendar dates and price levels are coded consistently within this request',
                       'Raw prices; corporate actions may distort returns',
                       'No web access; the coded headlines are the only news input']


def mask_patterns(codes):
    """Case-sensitive, word-bounded; longer names first so 'Meta Platforms' wins over 'Meta'."""
    names = []
    for asset, code in codes.items():
        tickers = {asset, md.crypto_pair(asset), sn.alpaca_news_symbol(asset), asset.removesuffix('-USD')}
        values = {v for v in sn.ALIASES.get(asset, []) + PEOPLE_AND_PRODUCTS.get(asset, [])
                  if v.upper() not in tickers}
        query = QUERIES.get(asset, '').strip('"')
        if query and asset not in sn.ETFS and query.upper() not in tickers:
            values.add(query)
        names += [(v, code, re.I) for v in values if v]
        names += [(v, code, 0) for v in tickers if v]
    names.sort(key=lambda item: (-len(item[0]), item[0]))
    return [(re.compile(r'(?<![\w])\$?' + re.escape(name) + r'(?![\w])', flags), code)
            for name, code, flags in names]


def relative_date(value, day):
    at = instant(value if 'T' in value else value + 'T00:00:00Z')
    label = f'DAY{(at.date() - day.date()).days:+d}'
    return label + (at.strftime(' %H:%M:%S UTC') if 'T' in value else '')


def shift(value, day):
    if isinstance(value, dict):
        return {k: shift(v, day) for k, v in value.items()}
    if isinstance(value, list):
        return [shift(v, day) for v in value]
    if isinstance(value, str) and DATE.match(value):
        return relative_date(value, day)
    return value


def anonymize(sealed, seed):
    context = sealed['context']
    rng = random.Random(seed)
    stocks, etfs, coins = list(sn.STOCKS), list(sn.ETFS), list(sn.CRYPTO)
    for group in (stocks, etfs, coins):
        rng.shuffle(group)
    codes = {**{a: f'STOCK_{i + 1:02d}' for i, a in enumerate(stocks)},
             **{a: f'SECTOR_ETF_{i + 1}' for i, a in enumerate(etfs)},
             **{a: f'COIN_{i + 1}' for i, a in enumerate(coins)}}
    patterns = mask_patterns(codes)

    day = instant(context['decision_time']).replace(hour=0, minute=0, second=0, microsecond=0)

    def mask(text):
        for pattern, code in patterns:
            text = pattern.sub(code, text)
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', lambda m: relative_date(m[0], day), text)
        # Month/day and quarter/year labels can also reveal the real calendar.
        text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|'
                      r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:,?\s+\d{4})?\b',
                      'CALENDAR_DATE', text, flags=re.I)
        text = re.sub(r'\b(?:Q[1-4]\s*)?(?:19|20)\d{2}\b', 'CALENDAR_PERIOD', text)
        text = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', 'CALENDAR_DATE', text)
        return text

    others, news_ids = {}, {}

    def symbol(s):
        if s in codes or s.replace('USD', '-USD') in codes:
            return codes.get(s) or codes[s.replace('USD', '-USD')]
        if s in sn.BENCHMARKS:
            return s
        return others.setdefault(s, f'TICKER_{len(others) + 1:03d}')

    prices = {}
    for asset, price in sorted(context['prices'].items()):
        name = codes.get(asset, asset)
        price = dict(price, asset=name)
        if price.get('state') == 'AVAILABLE' and (asset in codes or asset in sn.BENCHMARKS):
            factor = rng.uniform(0.5, 2.0)
            price['recent_closes'] = [round(c * factor, 6) for c in price['recent_closes']]
        prices[name] = price
    items = sorted({(i['published_at'], i['news_id']) for g in context['headlines'].values() for i in g['items']})
    for _, news_id in items:
        news_ids.setdefault(news_id, len(news_ids) + 1)
    headlines = {}
    for asset, group in sorted(context['headlines'].items()):
        headlines[codes.get(asset, asset)] = {'state': group['state'], 'items': [
            {'news_id': news_ids[i['news_id']], 'title': mask(i['title']), 'source': 'newswire',
             'symbols': [symbol(s) for s in i.get('symbols', [])], 'published_at': i['published_at'],
             'url': f"https://news.invalid/{news_ids[i['news_id']]}"} for i in group['items']]}
    anon = {k: v for k, v in context.items() if k not in ('provenance', 'limitations')}
    anon.update(schema='research-context-v2', universe=[codes[a] for a in context['universe']], prices=prices, headlines=headlines,
                exclusions=[dict({k: v for k, v in e.items() if k not in ('asset', 'symbol')},
                                 asset=symbol(e.get('asset', e.get('symbol')))) for e in context['exclusions']],
                limitations=NEUTRAL_LIMITATIONS)
    return shift(anon, day), {code: asset for asset, code in codes.items()}
