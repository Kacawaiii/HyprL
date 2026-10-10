"""Separate collector and radar runs. Never imports a trader or broker module."""
from datetime import timedelta
from urllib.parse import quote, urlencode
import argparse
import json
import math
import time

from .analysis import anomalies, bar_map, cluster, daily_observations, link_anomalies, paper_followup, priced_in, regime
from .core import Authorization, Client, RadarError, Store, instant, iso, write_private
from .digest import render, summarize
from .entities import Dictionary
from .registry import REGIME
from .sources import Collector
from .official import OfficialAuthorization, OfficialCollector
from .telegram import TelegramDelivery


def yahoo(collector, symbol):
    at = collector.store.clock()
    params = {'period1': int((at - timedelta(days=70)).timestamp()), 'period2': int(at.timestamp()), 'interval': '1d'}
    payload, received = collector.client.json('https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol, safe='') + '?' + urlencode(params), 'market', 'market:' + symbol)
    result = payload['chart']['result'][0]
    if result['meta']['symbol'] != symbol:
        raise RadarError('REGIME_SYMBOL_MISMATCH')
    quote_rows = result['indicators']['quote'][0]
    bars = []
    for i, stamp in enumerate(result['timestamp']):
        row = {'t': iso(instant('1970-01-01T00:00:00Z') + timedelta(seconds=stamp))}
        row.update({key: quote_rows[field][i] for key, field in [('o', 'open'), ('h', 'high'), ('l', 'low'), ('c', 'close'), ('v', 'volume')]})
        if all(row[k] is not None for k in ('o', 'h', 'l', 'c')):
            row['v'] = row['v'] or 0
            bars.append(row)
    collector.store.append('daily_bars', symbol, {'symbol': symbol, 'provider': 'yahoo', 'bars': bars, 'received_at': iso(received)})
    price, price_time = result['meta'].get('regularMarketPrice'), result['meta'].get('regularMarketTime')
    if isinstance(price, (int, float)) and math.isfinite(price) and price > 0 and isinstance(price_time, int):
        observed_at = instant('1970-01-01T00:00:00Z') + timedelta(seconds=price_time)
        if observed_at <= received:
            collector.store.append('price_observation', symbol, {'symbol': symbol, 'price': price, 'at': iso(observed_at), 'received_at': iso(received), 'provider': 'yahoo'})
    return len(bars)


def radar(store, grant=None, *, slot='manual', live=False, client=None, channels=None, llm_runner=None, publish_book=None, official_client=None, cached=False):
    with store.owner():
        key = store.clock().date().isoformat() + '-' + slot
        existing = store.latest('report', key)
        if existing:
            if publish_book and existing['live']:
                write_private(publish_book, existing['markdown'])
            return existing
        collector = None
        official_health = []
        if live and not grant:
            raise RadarError('AUTHORIZATION_REQUIRED')
        if live and not cached:
            collector = Collector(client, channels=channels)
            collector.collect(sleep=time.sleep)
            if official_client:
                official_health = OfficialCollector(official_client, sleep=time.sleep).collect()
            for provider, symbol, _ in REGIME.values():
                if provider == 'yahoo':
                    collector.attempt('market:' + symbol, lambda symbol=symbol: yahoo(collector, symbol))
        at = store.clock()
        previous = store.latest('last_events', 'radar') or {'events': []}
        dictionary = Dictionary()
        events = cluster(store.rows('news', since=at - timedelta(days=3)), previous=previous['events'], dictionary=dictionary, at=at, crypto_first=at.weekday() >= 5)
        if collector:
            # Acquire event-time bars only for ranked opportunities, with daily
            # lookback for all assets retained separately. Unknown stays unknown.
            selected = {symbol: at - timedelta(days=2) for symbol in ('SPY', 'QQQ', 'BTC/USD', 'ETH/USD')}
            for event in events[:8]:
                if not event['published_at']:
                    continue
                start = instant(event['published_at']) - timedelta(hours=1)
                if at - start > timedelta(days=4):
                    continue
                for symbol in event['entities']['symbols'][:4]:
                    if symbol in dictionary.allowed:
                        selected[symbol] = min(start, selected.get(symbol, start))
            # Existing paper observations also need fresh event-time prices.
            for watch in store.rows('paper_watch'):
                if watch['symbol'] in dictionary.allowed and at - instant(watch['recorded_at']) <= timedelta(days=7):
                    selected.setdefault(watch['symbol'], at - timedelta(days=2))
            for symbol, start in list(selected.items())[:32]:
                collector.attempt('alpaca:minute:' + symbol, lambda symbol=symbol, start=start:
                                  collector.bars([symbol], crypto='/' in symbol, minute=True, start=iso(start)))
            at = store.clock()
        daily, minute = bar_map(store, 'daily_bars', at), bar_map(store, 'minute_bars', at)
        for event in events:
            event['priced_in'] = priced_in(event, daily, minute, at)
        observations = {p['symbol']: p for p in store.rows('price_observation', since=at - timedelta(days=7)) if instant(p['received_at']) <= at}
        for symbol in ('SPY', 'QQQ', 'BTC/USD', 'ETH/USD'):
            if minute.get(symbol):
                last = minute[symbol][-1]
                observations[symbol] = {'price': last['c'], 'at': last['closed_at']}
        panel = regime(daily, at, observations)
        llm = {'status': 'OFFLINE_NOT_CALLED', 'calls': 0}
        if live:
            try:
                kwargs = {'runner': llm_runner} if llm_runner else {}
                llm = summarize(store, grant, events, panel, **kwargs)
            except RadarError as error:
                llm = {'status': str(error), 'calls': None, 'budget_reservations': store.counts().get('llm', 0)}
        # Only post-scenario observation baselines may enter the watch journal.
        labels = paper_followup(store, events, daily, minute, store.clock())
        limitations = ['Prix: cours clôturés; volumes actions IEX partiels; variations calendaires.',
                       'GDELT fournit une découverte, pas une preuve de publication.',
                       'Scénarios conditionnels; causalité et consensus non établis; aucune donnée TikTok.',
                       'Univers actions: snapshot du dépôt; cryptos candidates, couverture selon réponse Alpaca.']
        if not live:
            limitations.append('HORS LIGNE: aucune source vérifiée en direct; aucune synthèse LLM.')
        missing = [s for s in dictionary.coverage()['names_unmapped']]
        if missing:
            limitations.append('Identités de noms non vérifiées: ' + ', '.join(missing))
        health = (collector.health + official_health) if collector else []
        if live and cached:
            health = list({h['source']: h for h in store.rows('health')}.values())
            limitations.append('Relecture des captures privées: aucune collecte HTTP pendant ce radar; horodatages conservés.')
        if any(h['status'] != 'LIVE' for h in health):
            limitations.append('Des sources sont bloquées ou mortes; détails et codes dans le JSON.')
        cursor = store.latest('news_cursor', 'alpaca')
        if cursor and cursor['backlog']:
            limitations.append('Flux Alpaca en retard: pagination reprise au prochain passage, sans filtre de symboles.')
        if any(v['status'] in ('MISSING', 'STALE') for v in panel.values()):
            limitations.append('Régime incomplet ou périmé: valeurs manquantes explicitement inconnues.')
        observed_daily = daily_observations(store, at)
        anomaly_rows = anomalies(daily, observed_daily)
        anomaly_groups = link_anomalies(anomaly_rows, events, at)
        coverage = {**dictionary.coverage(),
                    'news_items_received_3d': sum(len(e['stories']) for e in events),
                    'observed_stock_etf_symbols': sorted(s for s in observed_daily if s in dictionary.stocks),
                    'observed_crypto_symbols': sorted(s for s in observed_daily if '/' in s),
                    'priced_in_events': sum(e['priced_in']['status'] == 'MEASURED' for e in events)}
        any_live = any(h['status'] == 'LIVE' for h in health)
        complete = (live and any_live and llm['status'] == 'OK'
                    and all(v['status'] == 'OBSERVED' for v in panel.values())
                    and not any(h['status'] == 'BLOCKED' for h in health))
        status = 'READY' if complete else 'PARTIAL' if live and any_live else 'BLOCKED' if live else 'SYNTHETIC'
        report = {'schema': 'news-radar-v3', 'status': status, 'date': at.date().isoformat(), 'slot': slot, 'cutoff': iso(at),
                  'live': live, 'sources': health, 'coverage': coverage, 'events': events,
                  'regime': panel, 'anomalies': anomaly_rows, 'anomaly_groups': anomaly_groups,
                  'llm': llm, 'budgets_used': store.counts(),
                  'paper_watch_count': len(store.rows('paper_watch')), 'paper_new_labels': len(labels), 'limitations': limitations}
        report['markdown'] = render(report)
        store.append('report', key, report)
        store.append('last_events', 'radar', {'events': events})
        if publish_book:
            # Dedicated task exception: the only file written in Claude book.
            write_private(publish_book, report['markdown'])
        return report


def deliver_alerts(store, grant, reports):
    at = store.clock()
    prior = store.latest('last_events', 'radar') or {'events': []}
    events = cluster(store.rows('news', since=at-timedelta(days=3)), previous=prior['events'], at=at)
    daily, minute = bar_map(store, 'daily_bars', at), bar_map(store, 'minute_bars', at)
    for event in events:
        event['priced_in'] = priced_in(event, daily, minute, at)
    try:
        return TelegramDelivery(store, grant).alerts(events, reports / 'radar-latest.md')
    except RadarError as error:
        return [{'status': 'BLOCKED', 'reason': str(error)}]


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Independent private news radar; no trades')
    parser.add_argument('action', choices=['run', 'collect', 'official', 'status', 'demo'])
    parser.add_argument('--store', type=Path, default=Path.home() / 'private/radar-v1')
    parser.add_argument('--reports', type=Path, default=Path.home() / 'reports/radar')
    parser.add_argument('--authorization', type=Path, default=Path.home() / 'authorizations/news-radar-v1.json')
    parser.add_argument('--credentials', type=Path, default=Path.home() / 'private/alpaca/momentum.env')
    parser.add_argument('--channels', type=Path, help='Private JSON list of handle/channel_id records')
    parser.add_argument('--slot', choices=['morning', 'evening', 'manual'], default='manual')
    parser.add_argument('--publish-book', action='store_true')
    parser.add_argument('--official-authorization', type=Path, default=Path.home() / 'authorizations/news-radar-sec-fed-telegram-v1.json')
    parser.add_argument('--telegram', action='store_true', help='Deliver this run or fresh collection alerts to the granted operator chat')
    parser.add_argument('--cached', action='store_true', help='Build a radar from prior receipts without refreshing HTTP sources')
    args = parser.parse_args()
    try:
        store = Store(args.store)
        if args.action == 'status':
            print(json.dumps({'counts_today': store.counts(), 'health': store.rows('health', limit=50)}))
            return 0
        if args.action == 'demo':
            from .synthetic import seed
            if store.rows('news') or store.rows('report'):
                raise RadarError('DEMO_REQUIRES_EMPTY_STORE')
            seed(store)
            report = radar(store, slot=args.slot)
        else:
            official_client = None
            if args.official_authorization.exists() or args.action == 'official' or args.telegram:
                official_grant = OfficialAuthorization(args.official_authorization)
                official_grant.check(store.clock())
                official_client = Client(store, official_grant)
            if args.action == 'official':
                with store.owner():
                    health = OfficialCollector(official_client, sleep=time.sleep).collect()
                    delivery = deliver_alerts(store, official_grant, args.reports)
                print(json.dumps({'sources_live': sum(h['status'] == 'LIVE' for h in health), 'counts_today': store.counts(), 'delivery': delivery}))
                return 0
            grant = Authorization(args.authorization)
            grant.check(store.clock())
            client = Client(store, grant, credentials=args.credentials)
            channels = json.loads(args.channels.read_text()) if args.channels else None
            if args.action == 'collect':
                with store.owner():
                    at = store.clock()
                    health = Collector(client, channels=channels).collect(markets=False, gdelt=at.hour % 2 == 0, sleep=time.sleep)
                    delivery = deliver_alerts(store, official_grant, args.reports) if args.telegram else []
                print(json.dumps({'sources_live': sum(h['status'] == 'LIVE' for h in health), 'counts_today': store.counts(), 'delivery': delivery}))
                return 0
            report = radar(store, grant, slot=args.slot, live=True, client=client, channels=channels,
                           publish_book=Path.home() / 'claude-book/radar-latest.md' if args.publish_book else None,
                           official_client=official_client, cached=args.cached)
        name = f"radar-{report['date']}-{args.slot}"
        write_private(args.reports / (name + '.md'), report['markdown'])
        write_private(args.reports / (name + '.json'), json.dumps({k: v for k, v in report.items() if k != 'markdown'}, ensure_ascii=False, indent=2))
        write_private(args.reports / 'radar-latest.md', report['markdown'])
        delivery = None
        alerts = []
        if args.telegram and args.action != 'demo':
            try:
                sender = TelegramDelivery(store, official_grant)
                delivery = sender.digest(report, args.reports / (name + '.md'))
                alerts = sender.alerts(report['events'], args.reports / (name + '.md'))
            except RadarError as error:
                delivery = {'status': 'BLOCKED', 'reason': str(error)}
        print(json.dumps({'status': report['status'], 'date': report['date'], 'slot': args.slot, 'events': len(report['events']), 'sources_live': sum(h['status'] == 'LIVE' for h in report['sources']), 'delivery': delivery, 'alerts': alerts}))
        return 0
    except RadarError as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
