"""Private collection plans; the prediction archive and outcome archive stay separate."""
from datetime import timedelta
import json
from pathlib import Path

from scripts.trading_lab.replay_bench import marketdata as md, run as parent, snapshot as sn
from scripts.trading_lab.replay_bench.grant import Budget, ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import experiment as ex, fills, inputs, spec
from .archive import Archive
from .collect import BARS_FROM, Collector, equities, news_symbols, news_rows, sec_or_unavailable
from .score import Outcomes


def candidate_plan(root):
    archive = Archive(root, Path(root) / 'archive_index.json')
    method = sha256_canonical(inputs.declaration(root))
    plans = {}
    for period in ('dev', 'test'):
        for day in ex.planned(period, 'ab'):
            bundle = inputs.build(archive, parent.decision_time(day), method)['cd']
            plans[day] = [c['symbol'] for c in bundle['context']['candidates']]
    return plans


def shown_candidates(root):
    return set(s for names in candidate_plan(root).values() for s in names)


def historical_outcomes(root, collector, *, include_news=True):
    plans = candidate_plan(root) if include_news else {
        d: [] for p in ('dev', 'test') for d in ex.planned(p, 'ab')}
    collector.scoring_symbols = set(sn.STOCKS + sn.ETFS + sn.BENCHMARKS) | set(s for names in plans.values() for s in names)
    collector.scoring_symbols.update(md.crypto_pair(s) for s in sn.CRYPTO)
    counts, missing, base_rows = {}, [], []
    for period in ('dev', 'test'):
        first, last = spec.spec()['periods'][period]
        outcomes = Outcomes(collector, first, last, spec.spec()['costs'])
        count = 0
        for day in ex.planned(period, 'ab'):
            for group, assets, horizons in (('ab', sn.UNIVERSE, (1, 5)), ('cd', plans[day], range(2, 11))):
                for asset in assets:
                    for h in horizons:
                        try:
                            result = outcomes.get(group, day, asset, h)
                        except ReplayError as error:
                            if error.code not in ('INCOMPLETE_LABEL_COVERAGE', 'MISSING_CRYPTO_MARK',
                                                  'MISSING_SYNCHRONIZED_SPY_MARK', 'INCOMPLETE_MARK_PATH'):
                                raise
                            missing.append({'period': period, 'day': day, 'group': group, 'asset': asset,
                                            'h': h, 'error': error.code})
                            continue
                        if result is not None:
                            count += 1
                            if period == 'dev' and group == 'ab' and h == 1:
                                base_rows.append({'session': day, 'asset': asset, 'outperform': result['rel'] > 0})
        counts[period] = count
    value = {'n': len(base_rows), 'successes': sum(r['outperform'] for r in base_rows),
             'outcomes_sha256': sha256_canonical(base_rows), 'spec_hash': spec.SPEC_HASH,
             'panel': 'a_v1_as_live:1d'}
    target = Path(root) / spec.STATE_DIR / 'dev-label-base-rate.json'
    if target.exists() and json.loads(target.read_text()) != value:
        raise ReplayError('DEV_BASE_RATE_CHANGED')
    atomic_json(target, value)
    atomic_json(Path(root) / spec.STATE_DIR / 'outcome-collection.json', {'counts': counts, 'missing': missing})
    return {'labels': counts, 'missing_labels': len(missing), 'dev_base_rate': value}


def retry_missing_labels(root, collector, plan_file):
    """One bounded recheck of a pre-existing missing-label census, before inference."""
    if spec.spec()['revision'] >= 10:
        raise ReplayError('MISSING_LABEL_RETRY_CLOSED')
    import hashlib
    target = Path(root) / spec.STATE_DIR / 'missing-label-retry.json'
    intent = target.with_suffix('.intent')
    if target.exists() or intent.exists():
        raise ReplayError('MISSING_LABEL_RETRY_ALREADY_ATTEMPTED')
    payload = Path(plan_file).read_bytes()
    plan = json.loads(payload)['missing']
    keys = [(r['period'], r['group'], r['day'], r['asset'], r['h']) for r in plan]
    if not keys or len(set(keys)) != len(keys):
        raise ReplayError('MISSING_LABEL_PLAN_INVALID')
    atomic_json(intent, {'plan_sha256': hashlib.sha256(payload).hexdigest(), 'labels': len(plan), 'at': iso(now())})
    collector.scoring_symbols = {md.crypto_pair(r['asset']) if r['asset'] in sn.CRYPTO else r['asset'] for r in plan}
    collector.scoring_symbols.update(sn.STOCKS + sn.ETFS + sn.BENCHMARKS)
    collector.scoring_symbols.update(md.crypto_pair(s) for s in sn.CRYPTO)
    collector.retry_missing = True
    remaining, recovered = [], 0
    for period in ('dev', 'test'):
        outcomes = Outcomes(collector, *spec.spec()['periods'][period], spec.spec()['costs'])
        for row in [r for r in plan if r['period'] == period]:
            try:
                result = outcomes.get(row['group'], row['day'], row['asset'], row['h'])
            except ReplayError as error:
                if error.code not in ('INCOMPLETE_LABEL_COVERAGE', 'MISSING_CRYPTO_MARK',
                                      'MISSING_SYNCHRONIZED_SPY_MARK', 'INCOMPLETE_MARK_PATH'):
                    raise
                remaining.append(dict(row, error=error.code))
            else:
                if result is None:
                    remaining.append(dict(row, error='LABEL_UNAVAILABLE'))
                else:
                    recovered += 1
    collector.retry_missing = False
    value = {'spec_hash': spec.SPEC_HASH, 'plan_sha256': hashlib.sha256(payload).hexdigest(),
             'attempted_labels': len(plan), 'recovered_labels': recovered, 'missing': remaining,
             'refreshed_requests': len(collector.retried_keys), 'paired': True}
    atomic_json(target, value)
    # Recompute the entire fixed panel and its baseline from captured responses;
    # never migrate the previous protocol's base rate or outcomes.
    value['census'] = historical_outcomes(root, collector, include_news=False)
    atomic_json(target, value)
    return {'attempted_labels': len(plan), 'recovered_labels': recovered, 'remaining_labels': len(remaining),
            'refreshed_requests': len(collector.retried_keys), 'census': value['census']}


def live(root, grants, alpaca_env, agent, *, clock=now, collector_factory=Collector, dry_run=False,
         decision_at=None):
    from . import forward
    at = clock()
    day = at.astimezone(md.NEW_YORK).date().isoformat()
    intraday = decision_at is not None
    if intraday and day != spec.spec()['night_fixes_amendment']['intraday']['session']:
        raise ReplayError('NOT_INTRADAY_SESSION')
    T = decision_at or (at + timedelta(minutes=15) if dry_run else forward.decision_time(day))
    if not md.sessions(day, day) or day < spec.spec()['forward']['first_session']:
        raise ReplayError('NOT_FORWARD_SESSION')
    if not dry_run and not intraday and not T.replace(hour=T.hour - 5, minute=0) <= at < T:
        raise ReplayError('OUTSIDE_LIVE_COLLECTION_WINDOW')
    directory = (Path(root) / 'forward-drill' / at.strftime('%Y%m%dT%H%M%S') if dry_run else
                 Path(root) / 'live-archive' / day)
    if intraday:
        directory = directory / 'intraday'
    if not dry_run and (forward.directory(root, day) / 'snapshot.json.gz').exists():
        raise ReplayError('LIVE_SNAPSHOT_ALREADY_SEALED')
    collector = collector_factory(grants, Budget(Path(root) / 'collector'), directory / 'cache',
                                  alpaca_env=alpaca_env, sec_user_agent=agent, clock=clock,
                                  refresh_news=True, before=T)
    index = directory / 'archive_index.json'
    earliest = min(T - timedelta(hours=72), md.previous_close(T))
    try:
        rows = news_rows(collector, iso(earliest), iso(T))
        names = equities(collector, news_symbols(rows, md.previous_close(T), T))
        # Daily inputs only need completed sessions. Including the current SIP
        # interval requests real-time entitlement and can return 403 for a
        # historical-data subscription, before a single history is archived.
        midnight = T.replace(hour=0, minute=0, second=0, microsecond=0)
        end = iso(max(midnight, md.previous_close(T) + timedelta(seconds=1)))
        md.stock_bars(collector, sorted(set(sn.STOCKS + sn.ETFS + sn.BENCHMARKS) | names), BARS_FROM, end)
        md.crypto_bars(collector, list(sn.CRYPTO), BARS_FROM, iso(midnight))
        collector.save(index)
        archive = Archive(root, index, before=T)
        bundles = inputs.build(archive, T, sha256_canonical(inputs.declaration(root)))
        from .quality import require_pipeline_health
        require_pipeline_health(bundles['ab']['manifest'])
        shown = [c['symbol'] for c in bundles['cd']['context']['candidates']]
        if shown:
            sec_or_unavailable(collector, shown, root)
        collector.save(index)
    finally:
        collector.save(index)
    return {'session': day, 'received_before': iso(T), 'requests': collector.requests,
            'objects': len(json.loads(index.read_text())['objects']), 'shown_candidates': len(shown),
            'fixed_present': len(bundles['ab']['context']['universe']),
            'dry_run': dry_run, 'model_calls': 0, 'publications': 0}


def live_outcomes(root, grants, alpaca_env, agent, *, clock=now, collector_factory=Collector):
    from . import forward
    at = clock()
    directory = Path(root) / 'live-archive'
    collector = collector_factory(grants, Budget(Path(root) / 'collector'), directory / 'outcome-cache' / 'cache',
                                  alpaca_env=alpaca_env, sec_user_agent=agent, clock=clock)
    index = directory / 'outcome-cache' / 'archive_index.json'
    last = at.date().isoformat()
    labels = 0
    from collections import Counter
    exclusions = Counter()
    try:
        for file in sorted((Path(root) / forward.STATE_DIR).glob('????-??-??/snapshot.json.gz')):
            day = file.parent.name
            sealed = parent.read_json(file)
            plans = set()
            for family in ex.FAMILIES:
                decision = file.parent / (family + '.json')
                if not decision.exists():
                    continue
                value = json.loads(decision.read_text())['output']
                plans.update(('ab', r['asset'], int(r['horizon'][:-1])) for r in value['analyst']['views'])
                plans.update(('cd', r['asset'], int(r['horizon'][:-1])) for r in value['news']['forecasts'])
                plans.update(('cd', r['symbol'], r['holding_sessions']) for r in value['news']['views'])
            if not plans:
                continue
            collector.scoring_symbols.update(md.crypto_pair(a) if a in sn.CRYPTO else a for _, a, _ in plans)
            collector.scoring_symbols.add('SPY')
            factory, clock_for = Outcomes, forward.decision_time
            if sealed.get('mode') == 'INTRADAY':
                from .intraday import Outcomes as IntradayOutcomes, clock_record
                factory = IntradayOutcomes
                T = instant(clock_record(root, day)['decision_time'])
                clock_for = lambda unused: T
            outcomes = factory(collector, day, last, spec.spec()['costs'], decision_time=clock_for, live=True)
            following = md.sessions(day, (instant(day + 'T00:00:00Z') + timedelta(days=21)).date().isoformat())
            for group, asset, h in sorted(plans):
                crypto = asset in sn.CRYPTO
                mature = (fills.crypto_entry(forward.decision_time(day)) + timedelta(days=1) if crypto
                          else following[h - 1].close_at)
                if crypto and h != 1 or mature >= at:
                    continue
                try:
                    result = outcomes.get(group, day, asset, h)
                    labels += result is not None
                    if result is None:
                        reason = outcomes.exclusion_reasons.get((group, day, asset, h), 'censored_or_missing_label')
                        exclusions[reason] += 1
                except ReplayError as error:
                    if error.code not in ('INCOMPLETE_LABEL_COVERAGE', 'MISSING_CRYPTO_MARK',
                                          'MISSING_SYNCHRONIZED_SPY_MARK', 'INCOMPLETE_MARK_PATH'):
                        raise
    finally:
        collector.save(index)
    return {'labels': labels, 'exclusions_by_reason': dict(exclusions), 'requests': collector.requests, 'index_hash': sha256_canonical(json.loads(index.read_text()))}
