"""Declared method: strict cutoffs, headline-only news, and causal universes."""
from datetime import timedelta
import re
import statistics

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from . import spec, quality
from scripts.trading_lab.trader_agent.config import TraderError

RISKS = ['vendor corrections of historical bars', "headline text is the vendor's latest version, last edited before T",
         'universe survivorship for names without Alpaca history', 'news vendor coverage gaps',
         'asset borrow flags are the collection-time values, not point-in-time']


def declaration(root):
    import json
    from pathlib import Path
    values = [json.loads((Path(root) / name).read_text()) for name in ('input_proof.json', 'duel_input_proof.json')]
    if values[0] != values[1] or sha256_canonical(values[0]) != spec.spec()['declared_method_sha256']:
        raise ReplayError('METHOD_DECLARATION_CHANGED')
    return values[0]


def paired_label_exclusions(root):
    """Explicitly retain the operator-decided 658 keys, bound to committed evidence."""
    import json
    from pathlib import Path
    evidence = json.loads((spec.ROOT / 'docs/artifacts' /
                          spec.spec()['unblock_2_amendment']['missing_labels_source_artifact']).read_text())['labels']
    source = Path(root) / 'duel-v5' / 'missing-label-retry.json'
    if not source.is_file() or spec.file_hash(source) != evidence['retry_record_sha256']:
        raise ReplayError('PAIRED_LABEL_EXCLUSION_SOURCE_CHANGED')
    rows = json.loads(source.read_text())['missing']
    keys = sorted((r['group'], r['day'], r['asset'], r['h']) for r in rows)
    if len(keys) != evidence['remaining_excluded'] or sha256_canonical(keys) != evidence['excluded_keys_sha256']:
        raise ReplayError('PAIRED_LABEL_EXCLUSION_KEYS_CHANGED')
    return rows


def news_at(rows, T):
    out = []
    for row in rows:
        if quality.news_reason(row):
            continue
        created = instant(row['created_at'])
        updated = instant(row['updated_at'])
        if updated < created:
            continue  # Operator 2026-10-19: uniform data-quality exclusion.
        # Operator decision 2026-10-18: both clocks strictly before T, no edit tolerance.
        if created >= T or updated >= T or md.PROTECTED.intersection(row['symbols']):
            continue
        # Only creation-time fields: no summary, article, author, URL or later-edited
        # content enters selection or the model. The URL is an opaque citation id.
        out.append({'id': row['id'], 'headline': row['headline'], 'source': row['source'],
                    'symbols': sorted(row['symbols']), 'created_at': iso(created)})
    return sorted(out, key=lambda r: (r['created_at'], r['id']), reverse=True)


def rows_at(bars, T, *, crypto=False):
    visible = []
    for bar in bars:
        try:
            at = instant(bar['t'])
        except (KeyError, TypeError, ValueError, AttributeError, ReplayError):
            raise ReplayError('INVALID_BAR') from None
        if at < T:
            quality.validate_bar(bar)
            visible.append(bar)
    bars = visible
    rows = md.crypto_rows(bars) if crypto else md.session_rows(bars)
    rows = [r for r in rows if r['close_at'] < T]
    if len({r['bar_open_at'] for r in rows}) != len(rows):
        raise ReplayError('DUPLICATE_PRICE_SESSION')
    return sorted(rows, key=lambda r: r['bar_open_at'])


def headline(row):
    return {'news_id': row['id'], 'title': row['headline'], 'source': row['source'],
            'symbols': row['symbols'], 'published_at': row['created_at'],
            'url': 'https://news.invalid/' + str(row['id'])}


def small_mid(rows, T):
    if len(rows) < 200:
        return 'HISTORY_LT_200_SESSIONS'
    dollar = statistics.mean(float(r['close']) * float(r['volume']) for r in rows[-50:])
    if dollar < 10e6:
        return 'DOLLAR_VOLUME_50D_LT_10M'
    from scripts.trading_lab.trader_agent.data import price_summary
    try:
        stats = price_summary(rows, 'candidate', T, False)
    except TraderError as error:
        return quality.require_data_error(error)
    return dict(stats, mean_dollar_volume_50d=dollar, history_sessions=len(rows))


def build(archive, T, method_hash):
    from collections import Counter
    day_errors = []
    try:
        archive.require_assets()
    except ReplayError as error:
        day_errors.append(quality.require_data_error(error))
    start = md.previous_close(T)
    earliest = min(T - timedelta(hours=72), start)
    try:
        archive.require_news(earliest, T)
    except ReplayError as error:
        day_errors.append(quality.require_data_error(error))
    visible = [n for n in news_at(archive.news.values(), T) if instant(n['created_at']) >= earliest]
    news_errors = {str(n['id']): quality.news_reason(n) for n in archive.news.values() if quality.news_reason(n)}
    news_errors.update(getattr(archive, 'invalid_news', {}))
    invalid_news = sorted(news_errors)
    sec_policy = spec.spec()['unblock_2_amendment']['sec_policy']
    filings_unavailable = sec_policy['state'] == 'UNAVAILABLE'
    prices, eligible, excluded = {}, [], []
    items = {}
    def bind(kind, identity, value):
        key = kind + ':' + str(identity)
        items[key] = sha256_canonical(value)
    # Include items consulted during ranking and exclusions, not only the winners.
    for n in visible:
        bind('news', n['id'], n)
    for asset in sn.UNIVERSE + sn.BENCHMARKS:
        from scripts.trading_lab.trader_agent.data import price_summary
        try:
            if asset in archive.ambiguous_assets:
                raise ReplayError('CONTRADICTORY_ASSET_METADATA')
            if asset in getattr(archive, 'invalid_bars', {}):
                raise ReplayError(archive.invalid_bars[asset])
            if asset not in sn.CRYPTO:
                reason = quality.asset_reason(archive.assets.get(asset))
                if reason:
                    raise ReplayError(reason)
            bars = archive.bars.get(asset, [])
            rows = rows_at(bars, T, crypto=asset in sn.CRYPTO)
            if not any(instant(b['t']) < instant('2026-07-01T00:00:00Z') for b in bars):
                raise ReplayError('FIXED_UNIVERSE_PRE_JULY_HISTORY_MISSING', asset)
            prices[asset] = price_summary(rows, asset, T, asset in sn.CRYPTO)
        except (ReplayError, TraderError) as error:
            reason = quality.require_data_error(error)
            excluded.append({'symbol': asset, 'reason': reason})
            if asset in sn.BENCHMARKS:
                day_errors.append(reason)
            continue
        for row in rows[-21:]:
            bind('bar', asset + ':' + iso(row['bar_open_at']), {k: iso(v) if hasattr(v, 'tzinfo') else v for k, v in row.items()})
        if asset in sn.UNIVERSE:
            eligible.append(asset)
    window = [n for n in visible if instant(n['created_at']) >= T - timedelta(hours=72)]
    headlines = {}
    for asset in eligible:
        selected = [n for n in window if sn.alpaca_news_symbol(asset) in n['symbols']][:8]
        headlines[asset] = {'state': 'AVAILABLE', 'items': [headline(n) for n in selected]}
        for n in selected:
            bind('news', n['id'], n)
    macro = [n for n in window if {'SPY', 'QQQ'}.intersection(n['symbols'])][:15]
    headlines['macro'] = {'state': 'AVAILABLE', 'items': [headline(n) for n in macro]}
    for n in macro:
        bind('news', n['id'], n)
    ab = {'schema': 'replay-context-v1', 'session': T.date().isoformat(), 'decision_time': iso(T),
          'price_cutoff': iso(T), 'universe': eligible, 'prices': prices, 'headlines': headlines,
          'archives': {'fomc': {'state': 'NOT_CONFIGURED', 'items': []},
                       'edgar': {'state': 'UNAVAILABLE' if filings_unavailable else 'AVAILABLE', 'items': []}},
          'exclusions': excluded, 'synthetic': False, 'limitations': RISKS,
          'provenance': spec.PROVENANCE}
    by_symbol = {}
    for n in visible:
        if instant(n['created_at']) < start or not 1 <= len(n['symbols']) <= 2:
            continue
        for symbol in n['symbols']:
            if sn.TICKER.fullmatch(symbol) and symbol not in sn.NOT_SMALL_MID:
                by_symbol.setdefault(symbol, []).append(n)
    candidates, rejections = [], {}
    keywords = spec.spec()['variants']['c_news_small_mid']['negative_keywords']
    for symbol, news in sorted(by_symbol.items()):
        if symbol in archive.ambiguous_assets:
            rejections[symbol] = 'CONTRADICTORY_ASSET_METADATA'
            continue
        asset = archive.assets.get(symbol)
        reason = quality.asset_reason(asset)
        if reason:
            rejections[symbol] = reason
            continue
        bind('asset', symbol, asset)
        if asset['class'] != 'us_equity' or re.search(r'\bETF\b', asset.get('name', ''), re.I):
            rejections[symbol] = 'NOT_COMPANY_EQUITY'
            continue
        try:
            if symbol in getattr(archive, 'invalid_bars', {}):
                raise ReplayError(archive.invalid_bars[symbol])
            archive.require_bar_history(symbol, T)
            rows = rows_at(archive.bars.get(symbol, []), T)
        except ReplayError as error:
            rejections[symbol] = quality.require_data_error(error)
            continue
        for r in rows:
            bind('universe-bar', symbol + ':' + iso(r['bar_open_at']),
                 {k: iso(v) if hasattr(v, 'tzinfo') else v for k, v in r.items()})
        stats = small_mid(rows, T)
        if isinstance(stats, str):
            rejections[symbol] = stats
            continue
        # Only headline bytes affect rank. Later-edited summaries cannot leak.
        news = sorted(news, key=lambda n: (sn.negative_score(n['headline'], keywords), n['created_at'], n['id']), reverse=True)
        score = sum(sn.negative_score(n['headline'], keywords) for n in news)
        candidates.append((score, max(n['created_at'] for n in news), symbol, stats, news[:3]))
    candidates.sort(key=lambda v: (v[0], v[1], v[2]), reverse=True)
    shown = []
    for _, _, symbol, stats, news in sorted(candidates[:40], key=lambda v: v[2]):
        filings = [] if filings_unavailable else [f for f in archive.filings.get(symbol, {}).values()
                                                  if instant(f['accepted_at']) < T]
        for f in filings:
            bind('filing', symbol + ':' + f['accession'], f)
        for n in news:
            bind('news', n['id'], n)
        shown.append({'symbol': symbol, 'prices': dict(stats, asset=symbol),
                      'headlines': [headline(n) for n in news], 'filings': filings,
                      'filings_state': 'AVAILABLE' if not filings_unavailable and symbol in archive.filing_symbols else 'UNAVAILABLE'})
    cd = {'schema': 'replay-news-context-v1', 'session': T.date().isoformat(), 'decision_time': iso(T),
          'news_window_start': iso(start), 'candidates': shown, 'limitations': RISKS, 'provenance': spec.PROVENANCE}
    # Scoring-only borrow flags (never shown to a model): hard-to-borrow shorts are excluded.
    borrow = {s: {k: archive.assets[s].get(k) for k in ('shortable', 'easy_to_borrow')}
              for s in list(sn.STOCKS + sn.ETFS + sn.BENCHMARKS) + [c['symbol'] for c in shown] if s in archive.assets}
    manifest = {'schema': 'replay-per-day-input-manifest-v2', 'decision_time': iso(T), 'spec_hash': spec.SPEC_HASH,
                'method_hash': method_hash, 'provenance': spec.PROVENANCE, 'residual_risks': RISKS,
                'items': [{'id': k, 'sha256': v} for k, v in sorted(items.items())],
                'rejections': rejections, 'candidate_count': len(candidates), 'shown_count': len(shown),
                'borrow_flags': borrow}
    manifest['news_data_quality'] = {'rule': 'missing, invalid or contradictory required news fields excluded uniformly',
                                    'excluded_count': len(invalid_news),
                                    'excluded_ids_sha256': sha256_canonical(sorted(invalid_news))}
    manifest['sec_availability'] = sec_policy
    manifest['asset_metadata_exclusions'] = {'rule': 'exclude contradictory asset metadata uniformly',
        'excluded_count': len(archive.ambiguous_assets),
        'excluded_symbols_sha256': sha256_canonical(sorted(archive.ambiguous_assets))}
    if day_errors:
        ab.update(universe=[], prices={}, headlines={})
        cd['candidates'] = []
    records = ([dict(scope='news', identity=k, reason=v) for k, v in sorted(news_errors.items())] +
               [dict(scope='fixed', identity=r['symbol'], reason=r['reason']) for r in excluded] +
               [dict(scope='candidate', identity=k, reason=v) for k, v in sorted(rejections.items())] +
               [dict(scope='day', identity=iso(T), reason=r) for r in sorted(set(day_errors))])
    manifest['data_quality'] = {'rule': 'uniform shared exclusions before model calls; no repair or retry',
        'counts_by_reason': dict(Counter(r['reason'] for r in records)),
        'excluded_day': bool(day_errors), 'records': records, 'records_sha256': sha256_canonical(records)}
    return {g: sn.seal(c, dict(manifest, group=g), T) for g, c in (('ab', ab), ('cd', cd))}


def primary_filter(views, context):
    kept, evidence = set(), {}
    for c in context['candidates']:
        accepted = {f['accession'] for f in c['filings'] if f['form'] == '8-K' and
                    instant(context['decision_time']) - timedelta(hours=72) <= instant(f['accepted_at']) < instant(context['decision_time'])}
        titles = {h['news_id']: h['title'] for h in c['headlines']}
        view = next(v for v in views if v['symbol'] == c['symbol'])
        exact = sorted({a for i in view['news_ids'] for a in re.findall(r'\b\d{10}-\d{2}-\d{6}\b', titles[i])} & accepted)
        evidence[c['symbol']] = exact
        if exact:
            kept.add(c['symbol'])
    return kept, evidence
