"""Small synthetic headlines and real-shaped provider bars; no captured fixtures."""
from datetime import timedelta

from .core import iso
from .sources import item


def seed(store):
    at = store.clock()
    examples = [
        ('synthetic:wire', 'synthetic_wire', 'micron', 'Micron raises memory guidance after earnings', 'Synthetic memory demand story.', ['MU']),
        ('synthetic:editor', 'synthetic_editor', 'micron-followup', 'Micron raises memory guidance after earnings', 'Independent synthetic reporting.', []),
        ('synthetic:retail', 'synthetic_channel', 'micron-video', 'Micron raises memory guidance after earnings', 'Synthetic retail narrative.', []),
        ('synthetic:wire', 'synthetic_wire', 'brazil', 'Brazil election rally: risk premium in focus', 'Synthetic election result context.', []),
        ('synthetic:editor', 'synthetic_editor', 'euro', 'ECB digital euro regulation proposal', 'Synthetic regulation proposal; expectations unknown.', []),
        ('synthetic:retail', 'synthetic_channel', 'crypto', 'Bitcoin could rally on crypto regulation rumour', 'Unconfirmed synthetic narrative.', []),
    ]
    for source, publisher, slug, headline, summary, symbols in examples:
        story = item(source, publisher, 'https://example.invalid/' + slug, headline, summary,
                     iso(at - timedelta(hours=2)), at - timedelta(hours=1), symbols=symbols,
                     retail=source == 'synthetic:retail')
        store.append('news', story['id'], story)
    for symbol in ('MU', 'SPY', 'QQQ', 'EWZ', 'BTC/USD', 'ETH/USD'):
        daily = []
        for day in range(45, 0, -1):
            start = (at - timedelta(days=day)).replace(hour=0, minute=0, second=0, microsecond=0)
            close = 100 + (45 - day) * .1
            daily.append({'t': iso(start), 'o': close, 'h': close + 2, 'l': close - 2,
                          'c': close, 'v': 1000 if day > 1 else 3000, 'n': 100, 'vw': close})
        store.append('daily_bars', symbol, {'symbol': symbol, 'provider': 'synthetic', 'bars': daily, 'received_at': iso(at)})
    bars = []
    for minute in range(180, 0, -1):
        opened = (at - timedelta(minutes=minute)).replace(second=0, microsecond=0)
        close = 104 + (180 - minute) * .01
        bars.append({'t': iso(opened), 'o': close, 'h': close + .1, 'l': close - .1, 'c': close, 'v': 10, 'n': 2, 'vw': close})
    store.append('minute_bars', 'MU', {'symbol': 'MU', 'provider': 'synthetic', 'bars': bars, 'received_at': iso(at)})
