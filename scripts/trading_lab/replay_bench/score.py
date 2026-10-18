"""Score v3 only: per-view labels, baselines on the same eligible rows, and daily marked P&L."""
from collections import Counter, defaultdict
from datetime import timedelta
import json
import random

from . import experiment, labels as lb, marketdata as md, metrics, run, snapshot as sn, spec as binding
from .grant import ReplayError, instant, iso, now
from .storage import atomic_json

RANDOM_SEED = 20261016


def period_days(first, last):
    days = []
    day = instant(first + 'T00:00:00Z')
    while day.date().isoformat() <= last:
        days.append(day.date().isoformat())
        day += timedelta(days=1)
    return days


def allocate_pnl(trades, capital_divisor):
    """Equal weight within each entry cohort; a multiday cohort uses 1/max_horizon initial capital."""
    counts = Counter(day for day, _, _ in trades)
    pnl = defaultdict(float)
    for day, _, path in trades:
        weight = 1 / (capital_divisor * counts[day])
        for marked_day, change in path.items():
            pnl[marked_day] += weight * change
    return dict(pnl)


def primary_context(out, record):
    return run.read_json(out / 'snapshots' / f"{record['snapshot_hash']}.json.gz")['context']


def score(root, period, transport, spec):
    # Causality and lineage are checked before even fetching outcome prices.
    experiment.input_proof(root)
    groups = experiment.GROUP_MODELS if period == 'dev' else ('ab', 'cd')
    planned = {g: experiment.planned(root, period, g)[0] for g in groups}
    if not all(planned.values()):
        raise ReplayError('NO_POST_CUTOFF_DATES')
    if not all(experiment.trial_complete(root, period, g) for g in groups):
        raise ReplayError('PERIOD_DECISIONS_INCOMPLETE')
    if period == 'test':
        experiment.ready_test(root)
        experiment.claim_test(root)
    first, last = spec['periods'][period]
    dates = period_days(first, last)
    if instant(last + 'T23:59:59Z') > now():
        raise ReplayError('PERIOD_NOT_ENDED')
    end = iso(instant(last + 'T23:59:59Z'))
    # Dev labels never read the test period. Near-boundary horizons are explicitly censored.
    universe = run.Universe(transport, end=end)
    anchors = lb.MinuteAnchors(transport)
    cells = {}
    labels, counts = [], Counter()
    trades = defaultdict(list)
    paired = defaultdict(dict)
    sec = md.SecFilings(transport)
    costs = spec['costs']
    for group, days in planned.items():
        meta = experiment.trial_meta(period, group)
        out = experiment.trial_out(root, meta)
        for day in days:
            record = run.checked_record(out / group / f'{day}.json.gz', meta, out)
            if not record['views']:
                counts[group + (':no_candidates' if group == 'cd' and record.get('candidates') == 0
                                 else ':failed_model_day')] += 1
                continue
            required = {'ab': run.VARIANTS_AB, 'anon': ('b_v1_no_reviewer',), 'opus': ('a_v1_as_live',),
                        'cd': ('c_news_small_mid',)}[group]
            for variant in required:
                if variant not in record['views']:
                    counts[group + ':missing_variant_day'] += 1
            T = run.decision_time(day)
            if group != 'cd':
                if day not in cells:
                    cells[day] = lb.label_universe_day(day, universe.bars, anchors, costs, last_day=last)
                views_by_variant = record['views']
                if group == 'anon':
                    views_by_variant = {'audit_anonymized': record['views'].get('b_v1_no_reviewer', [])}
                if group == 'opus':
                    views_by_variant = {'audit_opus_a': record['views'].get('a_v1_as_live', [])}
                for variant, views in views_by_variant.items():
                    for view in views:
                        asset, horizon = view['asset'], view['horizon']
                        key = variant + ':' + horizon
                        counts[key + ':' + view['view']] += 1
                        crypto = asset in sn.CRYPTO
                        if crypto and horizon != '1d':
                            counts[key + ':crypto_5d_not_declared_for_scoring'] += view['view'] != 'ABSTAIN'
                            continue
                        h = 1 if horizon == '1d' else 5
                        following = md.sessions(day, (T + timedelta(days=14)).date().isoformat())
                        if following[h - 1].session_date > last or (crypto and (T + timedelta(days=1)).date().isoformat() > last):
                            counts[key + ':boundary_censored'] += 1
                            continue
                        cell = cells[day].get(asset)
                        if not cell or horizon not in cell:
                            counts[key + ':missing_fill_or_history'] += 1
                            continue
                        rows = (md.crypto_rows if crypto else md.session_rows)(universe.bars.get(asset, []))
                        spy_rows = md.session_rows(universe.bars['SPY'])
                        signs = {variant: None}
                        if group == 'ab':
                            rng = random.Random(f'{RANDOM_SEED}-{day}-{asset}-{horizon}')
                            signs.update(always_up=1, random=rng.choice((-1, 1)),
                                         momentum20=1 if cell['momentum_up'] else -1, spy=1)
                        for name, sign in signs.items():
                            if sign is None and view['view'] == 'ABSTAIN':
                                continue
                            signed = sign if sign is not None else (1 if view['view'] == 'UP' else -1)
                            result = lb.outcome(signed, cell[horizon]['rel'], cell['cost'], cell['stats']['vol_20d'], h)
                            result.update(p=view['p_outperform'] if sign is None else None,
                                          cluster=cell['cluster'], direction='UP' if signed > 0 else 'DOWN')
                            if crypto:
                                if name == 'spy':
                                    continue
                                try:
                                    path = lb.crypto_path(day, asset, anchors, signed, cell['cost'])
                                except ReplayError:
                                    counts[key + ':missing_crypto_mark'] += 1
                                    continue
                            elif h == 1:
                                path = {day: result['net']}
                                if name == 'spy':
                                    raw = cells[day]['_spy']['1d']
                                    if raw is None:
                                        continue
                                    result = lb.outcome(1, raw, 2 * (costs['fee_per_side'] + .0002),
                                                        lb.daily_stats(spy_rows, T)['vol_20d'], 1)
                                    result.update(p=None, cluster='index', direction='UP')
                                    path = {day: result['net']}
                            else:
                                _, path = lb.marked_path(day, asset, rows, spy_rows, h, signed, cell['cost'])
                                if name == 'spy':
                                    # Absolute SPY daily path, same five sessions.
                                    a = {r['bar_open_at'].date().isoformat(): r for r in spy_rows}
                                    entry, previous = a[day]['open'], a[day]['open']
                                    path = {}
                                    for i, s in enumerate(following[:h]):
                                        close = a[s.session_date]['close']
                                        path[s.session_date] = (close - previous) / entry - (.0012 if i in (0, h - 1) else 0)
                                        previous = close
                                    result = lb.outcome(1, previous / entry - 1, .0024,
                                                        lb.daily_stats(spy_rows, T)['vol_20d'], h)
                                    result.update(p=None, cluster='index', direction='UP')
                            label = {'group': group, 'variant': name, 'session': day, 'asset': asset, 'horizon': horizon,
                                     'snapshot_hash': record['snapshot_hash'], 'result': result, 'daily_pnl': path}
                            labels.append(label)
                            series = key if sign is None else f'baseline:{name}:{variant}:{horizon}'
                            trades[series].append((day, result, path))
                            if group in ('ab', 'opus', 'anon') and sign is None:
                                paired[(day, asset, horizon)][name] = view
            else:
                context = primary_context(out, record)
                views = record['views'].get('c_news_small_mid', [])
                kept, evidence = run.apply_primary_filter(record, sec, context)
                kept_symbols = {v['symbol'] for v in kept}
                symbols = [c['symbol'] for c in context['candidates']]
                bars, _ = md.stock_bars(transport, sorted(set(symbols + ['SPY'])),
                                        iso(T - timedelta(days=45)), end) if symbols else ({}, [])
                spy_rows = md.session_rows(bars.get('SPY', []))
                for view in views:
                    counts['c:' + view['view']] += 1
                    h = view['holding_sessions']
                    following = md.sessions(day, (T + timedelta(days=21)).date().isoformat())
                    if following[h - 1].session_date > last:
                        counts['c:boundary_censored'] += 1
                        continue
                    names = {'c_news_small_mid': None}
                    if view['symbol'] in kept_symbols:
                        names['d_news_primary_only'] = None
                    if view['view'] != 'ABSTAIN' and view['symbol'] not in kept_symbols:
                        counts['d:primary_source_not_proven'] += 1
                    rng = random.Random(f'{RANDOM_SEED}-{day}-{view["symbol"]}-cd')
                    names.update(always_up=1, random=rng.choice((-1, 1)), momentum20=0, spy=1)
                    for name, sign in names.items():
                        if sign is None and view['view'] == 'ABSTAIN':
                            continue
                        result = lb.label_news_view(view, day, bars, costs, spy_rows, signed=sign)
                        if result['state'] != 'LABELED':
                            counts['c:missing_outcome'] += 1
                            continue
                        if name == 'momentum20':
                            sign = 1 if result['momentum_up'] else -1
                            result = lb.label_news_view(view, day, bars, costs, spy_rows, signed=sign)
                        signed = sign if sign is not None else (1 if view['view'] == 'UP' else -1)
                        _, path = lb.marked_path(day, view['symbol'], md.session_rows(bars[view['symbol']]),
                                                 spy_rows, h, signed, result['cost'])
                        if name == 'spy':
                            a = {r['bar_open_at'].date().isoformat(): r for r in spy_rows}
                            entry, previous = a[day]['open'], a[day]['open']
                            path = {}
                            for i, s in enumerate(following[:h]):
                                close = a[s.session_date]['close']
                                path[s.session_date] = (close - previous) / entry - (.0012 if i in (0, h - 1) else 0)
                                previous = close
                            result = lb.outcome(1, previous / entry - 1, .0024,
                                                lb.daily_stats(spy_rows, T)['vol_20d'], h)
                        result.update(p=view['p_outperform'] if sign is None else None,
                                      cluster='smallmid', direction='UP' if signed > 0 else 'DOWN')
                        labels.append({'group': group, 'variant': name, 'session': day, 'asset': view['symbol'],
                                       'horizon': h, 'snapshot_hash': record['snapshot_hash'], 'result': result,
                                       'daily_pnl': path, 'primary_evidence': evidence if name.startswith('d_') else []})
                        trades[name if sign is None else 'baseline:' + name + ':c'].append((day, result, path))
    summary = {}
    for key, rows in trades.items():
        divisor = 10 if 'small_mid' in key or 'primary_only' in key or key.endswith(':c') else (5 if key.endswith(':5d') else 1)
        pnl = allocate_pnl(rows, divisor)
        summary[key] = metrics.summarize([(d, r) for d, r, _ in rows], dates, pnl_by_day=pnl, annualization=365)
        summary[key]['capital_divisor'] = divisor
    agreement = {}
    for name in ('audit_opus_a', 'audit_anonymized'):
        target = 'a_v1_as_live' if name == 'audit_opus_a' else 'b_v1_no_reviewer'
        pairs = [v for v in paired.values() if name in v and target in v]
        agreement[name] = {'n_pairs': len(pairs), 'agree': sum(v[name]['view'] == v[target]['view'] for v in pairs)}
    missing = {k: v for k, v in counts.items() if v and ('missing_' in k or 'failed_model' in k)}
    result = {'status': 'BLOCKED' if missing else 'SCORED', 'missing_coverage': missing,
              'provenance': binding.PROVENANCE, 'period': period, 'spec_hash': binding.SPEC_HASH,
              'code_hash': run.code_hash(), 'summary': summary, 'counts': dict(counts),
              'paired_audits': agreement, 'labels': labels,
              'limitations': ['Sparse crypto fills or midnight marks remain unscored and counted',
                              'Boundary-crossing horizons are censored and counted',
                              'Wilson intervals ignore clustering and are descriptive',
                              'SPY-relative strategies charge costs on the asset leg only; SPY baseline is absolute',
                              'Daily P&L uses fixed initial capital, equal weights in entry cohorts, calendar days x sqrt(365)']}
    atomic_json(root / 'v3/results' / f'{period}.json', result)
    if missing:
        raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
    return result
