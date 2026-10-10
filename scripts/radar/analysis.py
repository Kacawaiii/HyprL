"""Conservative event clustering and causal, observed price measurements."""
from datetime import timedelta
from statistics import mean
import math
import re

from .core import digest, instant, iso
from .entities import Dictionary, fold, transmission
from .registry import REGIME

STOP = set('the a an of to in for and on with as after de la le les des du un une et en sur pour au aux'.split())
RUMOUR = re.compile(r'\b(rumou?r|reportedly|unconfirmed|may|could|might|rumeur|pourrait|serait|speculat\w*)\b', re.I)


def tokens(value):
    return set(re.findall(r'\w+', fold(value))) - STOP


def independent_publisher(story):
    headline = fold(story['headline'] + ' ' + story['summary'])
    for source in ('reuters', 'benzinga', 'bloomberg', 'associated press', 'coindesk', 'cointelegraph'):
        if re.search(r'\b' + source + r'\b', headline):
            return source.replace(' ', '_')
    host = story['publisher'].removeprefix('www.')
    for needle, publisher in [('cnbc', 'cnbc'), ('yahoo', 'yahoo'), ('marketwatch', 'dowjones'),
                              ('wsj', 'dowjones'), ('coindesk', 'coindesk'), ('cointelegraph', 'cointelegraph'),
                              ('ecb.europa.eu', 'ecb')]:
        if needle in host:
            return publisher
    return host


def cluster(stories, *, previous=(), dictionary=None, at=None, crypto_first=False):
    dictionary = dictionary or Dictionary()
    events = []
    urls, hashes, word_index, entity_index = {}, {}, {}, {}
    # Stable order anchors IDs to the earliest time actually observed, not backdated publication.
    for story in sorted(stories, key=lambda s: (instant(s['received_at']), s['id'])):
        if at and instant(story['received_at']) > at:
            continue
        if at and story['published_at'] and instant(story['published_at']) < at - timedelta(days=3):
            # A newly fetched archive entry is not a new market event.
            continue
        entities = dictionary.map(story)
        words = tokens(story['headline'])
        match = None
        candidates = set()
        if story['url'] in urls:
            candidates.add(urls[story['url']])
        if story['headline_hash'] in hashes:
            candidates.add(hashes[story['headline_hash']])
        word_candidates = set().union(*(word_index.get(w, set()) for w in words))
        entity_candidates = set().union(*(entity_index.get(s, set()) for s in entities['symbols'] + entities['countries']))
        candidates |= word_candidates & entity_candidates
        match_index = None
        for index in sorted(candidates):
            event = events[index]
            gap = abs((instant(story['received_at']) - instant(event['first_received_at'])).total_seconds())
            if gap > 72 * 3600:
                continue
            same_url = any(s['url'] == story['url'] for s in event['stories'])
            same_headline = any(s['headline_hash'] == story['headline_hash'] for s in event['stories'])
            event_words = tokens(event['headline'])
            similarity = len(words & event_words) / max(1, len(words | event_words))
            shared_entities = bool(set(entities['symbols']) & set(event['entities']['symbols']) or
                                   set(entities['countries']) & set(event['entities']['countries']))
            # Shared ticker alone cannot merge different events about the same company.
            if same_url or same_headline or (shared_entities and similarity >= .5):
                match = event
                match_index = index
                break
        if not match:
            match = {'id': digest([story['url'], story['headline_hash']]), 'headline': story['headline'],
                     'first_received_at': story['received_at'], 'stories': [],
                     'entities': {'symbols': [], 'countries': [], 'themes': []}}
            events.append(match)
            match_index = len(events) - 1
        match['stories'].append(story)
        for key in entities:
            match['entities'][key] = sorted(set(match['entities'][key]) | set(entities[key]))
        urls[story['url']], hashes[story['headline_hash']] = match_index, match_index
        for word in words:
            word_index.setdefault(word, set()).add(match_index)
        for entity in entities['symbols'] + entities['countries']:
            entity_index.setdefault(entity, set()).add(match_index)
    old = {e['id']: e for e in previous}
    weights = {'earnings': 20, 'guidance': 25, 'm&a': 25, 'regulation': 25, 'macro': 30,
               'geopolitics': 30, 'election': 25}
    for event in events:
        stories = event['stories']
        editorial = [s for s in stories if not s['retail']]
        lead_candidates = [s for s in editorial if s['primary']] or editorial or stories
        event['headline'] = max(lead_candidates, key=lambda s: (instant(s['received_at']), s['id']))['headline']
        publishers = sorted({independent_publisher(s) for s in editorial})
        primary = any(s['primary'] for s in editorial)
        rumours = any(RUMOUR.search(s['headline'] + ' ' + s['summary']) for s in stories)
        event['evidence'] = {'score': 90 if primary and not rumours else 65 if len(publishers) >= 2 and not rumours else 25,
                             'status': 'rumour' if rumours else 'primary_statement' if primary else 'corroborated_reporting' if len(publishers) >= 2 else 'single_source',
                             'publishers': publishers, 'primary': primary,
                             'independence': 'editorial_group_after_attribution; syndication_without_attribution_remains_uncertain'}
        past = old.get(event['id'])
        old_hashes = {(s['headline_hash'], s['summary_hash'], independent_publisher(s)) for s in past['stories']} if past else set()
        novel = not past or any((s['headline_hash'], s['summary_hash'], independent_publisher(s)) not in old_hashes for s in stories)
        event['novelty'] = 'new' if not past else 'updated' if novel else 'repeat'
        entities = event['entities']
        breadth = 15 if entities['countries'] or set(entities['themes']) & {'macro', 'geopolitics'} else 8 if len(entities['symbols']) >= 3 else 3
        size = 10 if set(entities['symbols']) & {'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'GOOGL', 'META', 'BTC/USD', 'ETH/USD'} else 4
        event['importance'] = min(100, max([weights.get(t, 5) for t in entities['themes']] or [5]) +
                                   min(20, len(publishers) * 5) + (20 if novel else 0) + breadth + size)
        retail_channels = {s['publisher'] for s in stories if s['retail']}
        event['retail_hype'] = {'score': min(100, len(retail_channels) * 20), 'channels': len(retail_channels),
                                'posts': len({s['url'] for s in stories if s['retail']}),
                                'meaning': 'observed channel breadth; no audience/view or TikTok data'}
        publication_times = [s['published_at'] for s in stories if s['published_at']]
        event['published_at'] = min(publication_times, key=instant) if publication_times else None
        event['transmission_hypotheses'] = transmission(entities)
        event['conviction'] = None  # only separate qualitative scenario conviction may be produced by the LLM
        event['expectations'] = 'Consensus inconnu; un titre de presse ne prouve pas une surprise.'
    return sorted(events, key=lambda e: (not ('crypto' in e['entities']['themes']) if crypto_first else False,
                                         -e['importance'], -e['evidence']['score'], e['id']))


def valid_bars(rows, at, *, daily=True):
    """Retain finite, positive OHLC, closed intervals only; never future bars."""
    result = {}
    duration = timedelta(days=1) if daily else timedelta(minutes=1)
    for bar in rows:
        try:
            opened = instant(bar['t'])
            values = [float(bar[k]) for k in ('o', 'h', 'l', 'c')]
            volume = float(bar.get('v', 0))
            if (opened + duration > at or any(not math.isfinite(v) or v <= 0 for v in values)
                    or not math.isfinite(volume) or volume < 0
                    or values[1] < max(values[0], values[2], values[3]) or values[2] > min(values[0], values[1], values[3])):
                continue
            result[iso(opened)] = {**bar, 'o': values[0], 'h': values[1], 'l': values[2], 'c': values[3], 'v': volume,
                                   'closed_at': iso(opened + duration)}
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
    return [result[k] for k in sorted(result, key=instant)]


def bar_map(store, kind, at):
    result = {}
    for payload in store.rows(kind, since=at - timedelta(days=7)):
        if instant(payload['received_at']) <= at:
            # A partial candle captured yesterday never becomes a final candle
            # merely because wall time advanced. Completion must precede receipt.
            valid = valid_bars(payload['bars'], instant(payload['received_at']), daily=kind != 'minute_bars')
            result.setdefault(payload['symbol'], []).extend(valid)
    return {symbol: valid_bars(rows, at, daily=kind != 'minute_bars') for symbol, rows in result.items()}


def atr(rows):
    if len(rows) < 21:
        return None
    selected = rows[-21:]
    return mean(max(b['h'] - b['l'], abs(b['h'] - a['c']), abs(b['l'] - a['c'])) for a, b in zip(selected, selected[1:]))


def daily_observations(store, at):
    """Latest provider daily observations, including explicitly partial days."""
    result = {}
    for payload in store.rows('daily_bars', since=at - timedelta(days=7)):
        received = instant(payload['received_at'])
        if received > at:
            continue
        for raw in payload['bars']:
            try:
                opened = instant(raw['t'])
            except (ValueError, KeyError, TypeError):
                continue
            if opened > received:
                continue
            rows = valid_bars([raw], received + timedelta(days=1))
            if not rows:
                continue
            bar = rows[0]
            bar['complete_at_receipt'] = instant(bar['closed_at']) <= received
            bar['received_at'] = payload['received_at']
            symbol = payload['symbol']
            if symbol not in result or (instant(bar['t']), instant(bar['received_at'])) >= (instant(result[symbol]['t']), instant(result[symbol]['received_at'])):
                result[symbol] = bar
    return result


def anomalies(daily, observations=None):
    result = []
    for symbol, rows in daily.items():
        last = (observations or {}).get(symbol) or (rows[-1] if rows else None)
        if not last:
            continue
        prior = [b for b in rows if b['t'] < last['t']]
        if len(prior) < 21:
            continue
        norms = prior[-20:]
        norm_volume = mean(b['v'] for b in norms)
        norm_range = atr(prior)
        move = (last['c'] - prior[-1]['c']) / norm_range if norm_range else None
        ratio = last['v'] / norm_volume if norm_volume else None
        if (move is not None and abs(move) >= 2) or (ratio is not None and ratio >= 2):
            result.append({'symbol': symbol, 'price': last['c'], 'move_atr': round(move, 4) if move is not None else None,
                           'volume_vs_20d': round(ratio, 4) if ratio is not None else None,
                           'closed_at': last['closed_at'] if last.get('complete_at_receipt', True) else None,
                           'received_at': last.get('received_at'),
                           'partial_day': not last.get('complete_at_receipt', True),
                           'basis': 'observed daily bar vs completed prior days; partial-day volume not extrapolated; IEX stock volume is partial-market'})
    return sorted(result, key=lambda r: -abs(r['move_atr'] or 0))


def priced_in(event, daily, minute, at):
    if not event['published_at']:
        return {'status': 'UNKNOWN_PUBLICATION_TIME', 'measurements': []}
    publication = instant(event['published_at'])
    results = []
    for symbol in event['entities']['symbols']:
        history = [b for b in daily.get(symbol, []) if instant(b['closed_at']) <= publication]
        norm = atr(history)
        intraday = minute.get(symbol, [])
        before = [b for b in intraday if instant(b['closed_at']) <= publication]
        after = [b for b in intraday if publication < instant(b['closed_at']) <= at]
        if not norm or not before or not after:
            continue
        baseline, latest = before[-1], after[-1]
        # No stale pre-weekend/pre-market quote may masquerade as an event-time baseline.
        if publication - instant(baseline['closed_at']) > timedelta(minutes=15):
            continue
        results.append({'symbol': symbol, 'baseline': baseline['c'], 'price': latest['c'],
                        'baseline_at': baseline['closed_at'], 'price_at': latest['closed_at'],
                        'atr_20d_before_publication': norm,
                        'move_atr': round((latest['c'] - baseline['c']) / norm, 4),
                        'return_pct': round(100 * (latest['c'] / baseline['c'] - 1), 4),
                        'interpretation': 'observed movement, not proof of causation or remaining upside'})
    return {'status': 'MEASURED' if results else 'INSUFFICIENT_EVENT_TIME_BARS', 'measurements': results}


def regime(daily, at, observations=None):
    observations = observations or {}
    result = {}
    for label, (provider, symbol, instrument) in REGIME.items():
        rows = daily.get(symbol, [])
        result[label] = {'provider': provider, 'symbol': symbol, 'instrument': instrument,
                         'status': 'MISSING', 'last': None, 'at': None, 'returns_pct': {'1d': None, '5d': None, '1m': None}}
        observation = observations.get(symbol)
        if observation and instant(observation['at']) > at:
            observation = None
        if not rows and not observation:
            continue
        last = {'c': observation['price'], 'closed_at': observation['at']} if observation else rows[-1]
        result[label].update(status='OBSERVED', last=last['c'], at=last['closed_at'], return_anchors={})
        # Calendar horizons, last completed close on/before each anchor; weekends visible.
        for horizon, days in [('1d', 1), ('5d', 5), ('1m', 30)]:
            anchor = instant(last['closed_at']) - timedelta(days=days)
            previous = [b for b in rows if instant(b['closed_at']) <= anchor]
            if previous:
                result[label]['returns_pct'][horizon] = round(100 * (last['c'] / previous[-1]['c'] - 1), 4)
                result[label]['return_anchors'][horizon] = previous[-1]['closed_at']
        if at - instant(last['closed_at']) > timedelta(days=4):
            result[label]['status'] = 'STALE'
    return result


def paper_followup(store, events, daily, minute, at):
    """Prospective observation journal only. No orders, accounts or position sizes."""
    opened = {r['key'] for r in store.rows('paper_watch')}
    for event in events:
        scenario = event.get('scenario')
        if not scenario:
            continue
        for exposure in scenario['exposures']:
            symbol, direction = exposure['symbol'], exposure['direction']
            key = digest([event['id'], symbol, direction])
            rows = minute.get(symbol) or daily.get(symbol, [])
            if key in opened or not rows or at - instant(rows[-1]['closed_at']) > timedelta(days=4):
                continue
            row = {'key': key, 'event_id': event['id'], 'symbol': symbol, 'direction': direction,
                   'recorded_at': iso(at), 'baseline': rows[-1]['c'], 'baseline_at': rows[-1]['closed_at'],
                   'mechanism': exposure['mechanism'], 'invalidation': scenario['invalidation'],
                   'horizons_calendar_days': [1, 5], 'kind': 'observation_only'}
            store.append('paper_watch', key, row)
            opened.add(key)
    labels = []
    for watch in store.rows('paper_watch'):
        for days in watch['horizons_calendar_days']:
            key = digest([watch['key'], days])
            if store.latest('paper_label', key):
                continue
            target = instant(watch['recorded_at']) + timedelta(days=days)
            if target > at:
                continue
            rows = minute.get(watch['symbol']) or daily.get(watch['symbol'], [])
            eligible = [b for b in rows if target <= instant(b['closed_at']) <= at]
            if not eligible:
                continue
            observed = eligible[0]
            row = {'watch_key': watch['key'], 'horizon_calendar_days': days, 'target_at': iso(target),
                   'observed_at': observed['closed_at'], 'received_at': iso(at), 'price': observed['c'],
                   'return_pct': round(100 * (observed['c'] / watch['baseline'] - 1), 4),
                   'causality': 'unproven; no costs, PnL or execution simulation'}
            store.append('paper_label', key, row)
            labels.append(row)
    return labels
