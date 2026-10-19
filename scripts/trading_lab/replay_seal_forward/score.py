"""One shared outcome panel; both models scored, including all abstaining forecasts."""
from collections import Counter, defaultdict
from datetime import timedelta
import json
from pathlib import Path

from scripts.trading_lab.replay_bench import labels as lb, marketdata as md, run as parent, snapshot as sn
from scripts.trading_lab.replay_bench.score import allocate_pnl, period_days
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import commitment, experiment as ex, fills, metrics, run, schemas, spec, stats


def spy_mark(at, spy_rows, anchors):
    sessions = md.sessions(at.date().isoformat(), at.date().isoformat())
    if sessions and sessions[0].open_at <= at < sessions[0].close_at:
        trades = anchors.stock(at.date().isoformat(), at.strftime('%H:%M:%S'), ['SPY'])
        if not trades.get('SPY'):
            raise ReplayError('MISSING_SYNCHRONIZED_SPY_MARK')
        return trades['SPY']['p']
    # SPY is held at its last actual session close while the stock market is shut.
    eligible = [r for r in spy_rows if r['close_at'] <= at]
    if not eligible:
        raise ReplayError('MISSING_SYNCHRONIZED_SPY_MARK')
    return max(eligible, key=lambda r: r['close_at'])['close']


def crypto_outcome(day, asset, anchors, spy_rows, start):
    """Entry at the decision time + fixed offset (never a fixed UTC clock), exit 24 h later."""
    middle = (start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    times = (start, middle, start + timedelta(days=1))
    bars = [anchors.crypto(at).get(asset) for at in times]
    if not all(bars):
        raise ReplayError('MISSING_CRYPTO_MARK')
    spy = [spy_mark(at, spy_rows, anchors) for at in times]
    values = [b['o'] / bars[0]['o'] - s / spy[0] for b, s in zip(bars, spy)]
    return values[-1], {day: values[1], times[-1].date().isoformat(): values[2] - values[1]}


class Outcomes:
    def __init__(self, transport, first, last, costs, *, decision_time=None):
        self.transport, self.last, self.costs = transport, last, costs
        self.decision_time = decision_time or parent.decision_time
        from types import SimpleNamespace
        end = last + 'T23:59:59Z'
        bars, _ = md.stock_bars(transport, list(sn.STOCKS + sn.ETFS + sn.BENCHMARKS), parent.DATA_START, end)
        crypto, _ = md.crypto_bars(transport, list(sn.CRYPTO), parent.DATA_START, end)
        bars.update(crypto)
        self.universe = SimpleNamespace(bars=bars)
        self.anchors = lb.MinuteAnchors(transport)
        self.cells, self.cache, self.news_bars, self.ratios = {}, {}, {}, {}

    def _ratios(self, symbol, raw):
        """Raw/adjusted close ratios; None when adjusted evidence is not in the sealed cache."""
        if symbol not in self.ratios:
            params = {'symbols': symbol, 'timeframe': '1Day', 'start': parent.DATA_START, 'end': self.last + 'T23:59:59Z',
                      'feed': 'sip', 'adjustment': 'all', 'limit': 10000}
            try:
                rows = [b for body, _ in md.paginate(self.transport, '/v2/stocks/bars', params, 'bars')
                        for b in (body.get('bars') or {}).get(symbol, [])]
                self.ratios[symbol] = fills.adjustment_ratios(raw, rows)
            except ReplayError as error:
                if error.code not in ('OFFLINE_CACHE_MISS', 'PREDICTION_CACHE_NOT_SEALED'):
                    raise
                self.ratios[symbol] = None
        return self.ratios[symbol]

    def corporate_action(self, symbols, bars, first, last):
        """True if any leg had a split/dividend effective inside the window; None if unchecked."""
        flags = []
        for symbol in symbols:
            ratios = self._ratios(symbol, bars.get(symbol, []))
            flags.append(None if ratios is None else fills.corporate_action(ratios, first, last))
        if any(flags):
            return True
        return None if None in flags else False

    def get(self, group, day, asset, h):
        key = (group, day, asset, h)
        T = self.decision_time(day)
        crypto = asset in sn.CRYPTO
        # Refuse any designed fill at or before the decision (and its dispatch window).
        start = fills.entry(day, T, crypto)
        if key in self.cache:
            return self.cache[key]
        following = md.sessions(day, (T + timedelta(days=21)).date().isoformat())
        if (crypto and h != 1) or following[h - 1].session_date > self.last or (
                crypto and (start + timedelta(days=1)).date().isoformat() > self.last):
            self.cache[key] = None
            return None
        exit_day = following[h - 1].session_date
        if group == 'ab':
            if day not in self.cells:
                self.cells[day] = lb.label_universe_day(day, self.universe.bars, self.anchors, self.costs, last_day=self.last,
                                                        decision_at=T, crypto_entry=fills.crypto_entry(T))
            cell = self.cells[day].get(asset)
            horizon = str(h) + 'd'
            if not cell or horizon not in cell:
                raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
            spy_rows = md.session_rows(self.universe.bars['SPY'])
            rel = cell[horizon]['rel']
            action = False
            if crypto:
                rel, path = crypto_outcome(day, asset, self.anchors, spy_rows, start)
                exit_day = (start + timedelta(days=1)).date().isoformat()
            elif h == 1:
                path = {day: rel}
            else:
                prices = self.entry_prices(day, start, [asset, 'SPY'])
                _, path = lb.marked_path(day, asset, md.session_rows(self.universe.bars[asset]), spy_rows, h, 1, 0,
                                         entry_prices=prices)
                action = self.corporate_action([asset, 'SPY'], self.universe.bars, day, exit_day)
            result = {'rel': rel, 'cost': cell['cost'], 'vol': cell['stats']['vol_20d'], 'cluster': cell['cluster']}
        else:
            if asset not in self.news_bars:
                self.news_bars[asset], _ = md.stock_bars(self.transport, sorted([asset, 'SPY']),
                    iso(instant(spec.spec()['periods']['dev'][0] + 'T12:00:00Z') - timedelta(days=45)),
                    self.last + 'T23:59:59Z')
            bars = self.news_bars[asset]
            spy_rows = md.session_rows(bars.get('SPY', []))
            prices = self.entry_prices(day, start, [asset, 'SPY'])
            label = lb.label_news_view({'symbol': asset, 'holding_sessions': h, 'view': 'UP'}, day, bars, self.costs,
                                      spy_rows, decision_at=T, entry_prices=prices, last_day=self.last)
            if label['state'] != 'LABELED':
                raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
            rows = md.session_rows(bars[asset])
            stats = lb.daily_stats(rows, T)
            _, path = lb.marked_path(day, asset, rows, spy_rows, h, 1, 0, entry_prices=prices)
            action = self.corporate_action([asset, 'SPY'], bars, day, exit_day)
            result = {'rel': label['rel'], 'cost': label['cost'], 'vol': stats['vol_20d'], 'cluster': 'smallmid'}
        if abs(sum(path.values()) - result['rel']) > 1e-9:
            raise ReplayError('PNL_OUTCOME_MISMATCH')
        result.update(gross_path=path, entry_at=iso(start), entry_day=day, exit_day=exit_day, corporate_action=action)
        result['available_at'] = iso(start + timedelta(days=1)) if crypto else iso(following[h - 1].close_at)
        self.cache[key] = result
        return result

    def entry_prices(self, day, start, symbols):
        fixed = list(sn.STOCKS + sn.ETFS + ('SPY',))
        requested = fixed if set(symbols) <= set(fixed) else symbols
        trades = self.anchors.stock(day, start.strftime('%H:%M:%S'), requested)
        if any(not trades.get(s) for s in symbols):
            raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
        return {s: trades[s]['p'] for s in symbols}


def signed_outcome(outcome, view, h, costs=None, flags=None, hedge_flags=None):
    """Net of both legs' costs and borrow on the short leg; excluded windows return (None, reason).

    The SPY-relative trade is long one leg and short the other: UP shorts SPY, DOWN shorts the
    asset. Hard-to-borrow shorts (explicit Alpaca flags) and corporate-action windows are dropped.
    """
    sign = 1 if view == 'UP' else -1
    if outcome.get('corporate_action'):
        return None, 'CORPORATE_ACTION_WINDOW'
    short_flags = flags if sign < 0 else hedge_flags
    if fills.hard_to_borrow(short_flags):
        return None, 'HARD_TO_BORROW'
    entry = outcome['cost'] + (fills.hedge_cost(costs) if costs else 0)
    borrow = fills.borrow_cost(outcome['entry_day'], outcome['exit_day']) if costs and 'entry_day' in outcome else 0
    result = lb.outcome(sign, outcome['rel'], entry + borrow, outcome['vol'], h)
    result.update(cluster=outcome['cluster'], direction=view, p=None, hedge_and_borrow=entry + borrow - outcome['cost'])
    path = {d: sign * v for d, v in outcome['gross_path'].items()}
    days = sorted(path)
    path[days[0]] -= entry / 2
    path[days[-1]] -= entry / 2 + borrow
    checked = (costs is not None and outcome.get('corporate_action') is not None and short_flags is not None and
               all(short_flags.get(k) is True for k in ('shortable', 'easy_to_borrow')))
    return (result, path), ('NET_FULL_COSTS' if checked else 'GROSS_DIAGNOSTIC')


def baseline_rates(rows):
    by_panel = defaultdict(list)
    for row in rows:
        by_panel[row['panel']].append(row)
    return {panel: {'successes': sum(r['outperform'] for r in values), 'n': len(values),
                    'p': (sum(r['outperform'] for r in values) + 1) / (len(values) + 2)}
            for panel, values in by_panel.items()}


def base_probabilities(rows, period, dev_rates):
    if period == 'test':
        panel = rows[0]['panel'] if rows else None
        if rows and panel not in dev_rates:
            raise ReplayError('DEV_BASE_RATE_MISSING')
        return [dev_rates[panel]['p']] * len(rows)
    # Expanding dev baseline excludes every label on the current decision day.
    result = {}
    for day in sorted({r['session'] for r in rows}):
        values = [r for r in rows if instant(r['available_at']) < parent.decision_time(day)]
        result[day] = (sum(r['outperform'] for r in values) + 1) / (len(values) + 2)
    return [result[r['session']] for r in rows]


def record_key(family, period, group, day):
    return ':'.join((family, period, group, day))


def within_test_cut(outcome, period, family):
    return period != 'test' or instant(outcome['available_at']).date().isoformat() <= ex.test_cut()[family]


def frozen_primary_rate(observed):
    rate = json.loads(ex.FREEZE.read_text())['frozen_dev_base_rate']
    if (rate.get('n', 0) <= 0 or not 0 <= rate['successes'] <= rate['n'] or observed is None or
            any(observed[k] != rate[k] for k in ('n', 'successes'))):
        raise ReplayError('DEV_BASE_RATE_DRIFT')
    return dict(rate, p=(rate['successes'] + 1) / (rate['n'] + 2))


def score(root, period, transport, *, outcome_factory=Outcomes, clock=now):
    root = Path(root)
    proof = ex.input_proof(root)
    ex.check_freeze(root, test=period == 'test')
    if not all(ex.complete(root, period, f) for f in ex.FAMILIES):
        raise ReplayError('BOTH_PERIODS_DECISIONS_INCOMPLETE')
    dev_hash = ex.ready_test(root) if period == 'test' else None
    records, hashes = {}, {}
    for family in ex.FAMILIES:
        for group in (ex.GROUPS if period == 'dev' else ('ab', 'cd')):
            meta = ex.meta(root, period, group, family)
            for day in ex.planned_for(period, group, family):
                record = run.checked(ex.record_path(root, period, group, family, day), meta)
                records[(family, group, day)] = record
                hashes[family + ':' + group + ':' + day] = record['decision_hash']
    # External commitment: every response hash was published (ledger + Discord) before scoring.
    at = clock()
    public = commitment.published_hashes(root, 'replay-night', at)
    for (family, group, day), record in records.items():
        if public.get(record_key(family, period, group, day)) != record['decision_hash']:
            raise ReplayError('DECISION_HASH_NOT_PUBLISHED', record_key(family, period, group, day))
        for call in record['calls'].values():
            if 'transcript' in call:
                value = json.loads((root / spec.STATE_DIR / 'transcripts' / (call['transcript'] + '.json')).read_text())
                digest = sha256_canonical({k: value[k] for k in ('raw', 'err', 'returncode')})
                if public.get('response:' + call['transcript']) != digest:
                    raise ReplayError('RESPONSE_HASH_NOT_PUBLISHED')
    if period == 'test':
        for family in ex.FAMILIES:
            marker = json.loads((root / spec.STATE_DIR / ('test_started_' + family + '.json')).read_text())
            calls = [instant(c['called_at']) for (f, _, _), r in records.items() if f == family for c in r['calls'].values()]
            commitment.require(root, 'replay-test-start', family, {'test_start_marker': sha256_canonical(marker)},
                               min(calls) if calls else at)
    binding = {'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(), 'decision_hashes': hashes,
               'input_proof_hash': sha256_canonical(proof), 'dev_result_hash': dev_hash}
    path = ex.result_path(root, period)
    if path.is_file():
        old = json.loads(path.read_text())
        if old.get('status') == 'SCORED':
            if any(old.get(k) != v for k, v in binding.items()):
                raise ReplayError('SCORED_RESULT_BINDING_MISMATCH')
            return old
    first, last = spec.spec()['periods'][period]
    if instant(last + 'T23:59:59Z') > now():
        raise ReplayError('PERIOD_NOT_ENDED')
    outcomes = outcome_factory(ex.offline(root, transport, outcomes=True), first, last, spec.spec()['costs'])
    costs = spec.spec()['costs']
    dates = period_days(first, last)
    forecasts = {f: [] for f in ex.FAMILIES}
    trades = {f: defaultdict(list) for f in ex.FAMILIES}
    decisions = {f: defaultdict(list) for f in ex.FAMILIES}
    counts = {f: Counter() for f in ex.FAMILIES}
    basis = {f: defaultdict(set) for f in ex.FAMILIES}
    for (family, group, day), record in sorted(records.items()):
        other = records.get((ex.FAMILIES[1] if family == ex.FAMILIES[0] else ex.FAMILIES[0], group, day))
        if other is None:
            # A probe-flagged model's test stops at its cut; the other model's later days stay unpaired.
            counts[family][group + ':unpaired_after_probe_cut'] += 1
        elif (record['snapshot_hash'] != other['snapshot_hash'] or
                record['shown_context_hash'] != other['shown_context_hash']):
            raise ReplayError('PAIRED_SNAPSHOT_MISMATCH')
        stage = 'news' if group == 'cd' else 'analyst'
        if other is not None and record['calls'].get(stage, {}).get('prompt_sha256') != other['calls'].get(stage, {}).get('prompt_sha256'):
            raise ReplayError('PAIRED_PROMPT_MISMATCH')
        for name, errors in record.get('errors', {}).items():
            # Validation retries are counted, never dropped silently.
            counts[family]['failures:' + name] += len(errors)
        sealed = run.shared_snapshot(root, period, 'ab' if group == 'anon' else group, day, ex.offline(root, transport))
        if sealed['snapshot_hash'] != record['snapshot_hash']:
            raise ReplayError('SNAPSHOT_HASH_MISMATCH')
        flags = sealed['manifest'].get('borrow_flags', {})
        context = sealed['context']
        assets = [c['symbol'] for c in context['candidates']] if group == 'cd' else context['universe']
        horizons = ('2d', '10d') if group == 'cd' else ('1d', '5d')
        variants = run.CD if group == 'cd' else ((run.AB[1],) if group == 'anon' else run.AB)
        if not assets:
            counts[family][group + ':no_candidates'] += 1
        for variant in variants:
            name = 'audit_anonymized' if group == 'anon' else variant
            rows = record['forecasts'][variant]
            schemas.coverage([(r['asset'], r['horizon']) for r in rows], [(a, h) for a in assets for h in horizons])
            for row in rows:
                panel = name + ':' + row['horizon']
                outcome = outcomes.get('cd' if group == 'cd' else 'ab', day, row['asset'], int(row['horizon'][:-1]))
                if outcome is None:
                    counts[family][panel + ':censored_or_crypto5d'] += 1
                    continue
                if not within_test_cut(outcome, period, family) or outcome.get('corporate_action') is True:
                    counts[family][panel + ':excluded_cut_or_corporate_action'] += 1
                    continue
                forecasts[family].append(dict(panel=panel, session=day, asset=row['asset'], horizon=row['horizon'],
                                              p=row['p_outperform'], outperform=outcome['rel'] > 0,
                                              cluster=outcome['cluster'], available_at=outcome['available_at']))
            ds = record['decisions'][variant]
            schemas.coverage([(v['symbol'] if group == 'cd' else v['asset'],
                               'trade' if group == 'cd' else v['horizon']) for v in ds],
                             [(a, h) for a in assets for h in (('trade',) if group == 'cd' else horizons)])
            for v in ds:
                h = v['holding_sessions'] if group == 'cd' else int(v['horizon'][:-1])
                asset = v['symbol'] if group == 'cd' else v['asset']
                panel = name + (':trades' if group == 'cd' else ':' + v['horizon'])
                decisions[family][panel].append(v['view'])
                if v['view'] == 'ABSTAIN':
                    continue
                outcome = outcomes.get('cd' if group == 'cd' else 'ab', day, asset, h)
                if outcome is None:
                    counts[family][panel + ':censored_trade'] += 1
                    continue
                if not within_test_cut(outcome, period, family):
                    counts[family][panel + ':excluded_probe_cut_boundary'] += 1
                    continue
                signed, label = signed_outcome(outcome, v['view'], h, costs, flags.get(asset), flags.get('SPY'))
                if signed is None:
                    counts[family][panel + ':excluded_' + label.lower()] += 1
                    continue
                basis[family][panel].add(label)
                result, pnl = signed
                trades[family][panel].append((day, result, pnl))
    dev_rates = json.loads(ex.result_path(root, 'dev').read_text())['base_rates'] if period == 'test' else {}
    declared = {v + ':' + h for v in run.AB for h in ('1d', '5d')}
    declared |= {v + ':' + h for v in run.CD for h in ('2d', '10d', 'trades')}
    if period == 'dev':
        declared |= {'audit_anonymized:1d', 'audit_anonymized:5d'}
    summaries = {}
    for family in ex.FAMILIES:
        summary = {}
        for panel in sorted(declared | set(decisions[family]) | {r['panel'] for r in forecasts[family]}):
            ts = trades[family][panel]
            divisor = 10 if panel.endswith(':trades') else (5 if panel.endswith(':5d') else 1)
            summary[panel] = metrics.trading([(d, r) for d, r, _ in ts], dates, allocate_pnl(ts, divisor), decisions[family][panel])
            summary[panel]['trade_cost_basis'] = ('GROSS_DIAGNOSTIC' if not basis[family][panel] or
                                                  'GROSS_DIAGNOSTIC' in basis[family][panel] else 'NET_FULL_COSTS')
            rows = [r for r in forecasts[family] if r['panel'] == panel]
            summary[panel].update(metrics.proper(rows, base_probabilities(rows, period, dev_rates)))
        summaries[family] = summary
    # Named panels only for formal comparison; anonymized data remain a dev diagnostic. Unpaired
    # rows (after a probe cut) are excluded from paired tests and counted above.
    named = {f: [r for r in forecasts[f] if not r['panel'].startswith('audit_')] for f in ex.FAMILIES}
    keyed = {f: {(r['panel'], r['session'], r['asset'], r['horizon']) for r in named[f]} for f in ex.FAMILIES}
    common = keyed['opus'] & keyed['gpt']
    paired = metrics.paired(*[[r for r in named[f] if (r['panel'], r['session'], r['asset'], r['horizon']) in common]
                              for f in ex.FAMILIES], ex.planned(period, 'ab'))
    rates = baseline_rates(forecasts['opus']) if period == 'dev' else dev_rates
    rates[stats.PRIMARY['panel']] = frozen_primary_rate(rates.get(stats.PRIMARY['panel']))
    primary = stats.primary_endpoint(named['opus'], named['gpt'], rates.get(stats.PRIMARY['panel'], {}).get('p'))
    if period == 'dev':
        primary['base_rate_note'] = 'dev uses its own in-sample rate: diagnostic only, not the frozen test baseline'
    failures = {f: dict(ex.failure_summary(root)[f],
                       failed_or_retried_stages=sum(v for k, v in counts[f].items() if k.startswith('failures:')),
                       missing_planned_days=0) for f in ex.FAMILIES}
    result = dict(binding, status='SCORED', period=period, provenance=spec.PROVENANCE, evidence_class='EXPLORATORY',
                  evidence_note=spec.spec()['evidence']['replay'],
                  residual_risks=spec.spec()['residual_risks'], models=summaries, forecasts=forecasts,
                  base_rates=rates, primary_endpoint=primary, verdict_wording=stats.wording(primary),
                  failures=failures, test_cut=ex.test_cut(),
                  counts={f: dict(c) for f, c in counts.items()}, paired_tests=paired,
                  head_to_head={p: {f: summaries[f].get(p) for f in ex.FAMILIES}
                                for p in sorted(set(summaries['opus']) | set(summaries['gpt']))},
                  limitations=spec.spec()['statistics']['limitations'])
    atomic_json(path, result)
    write_table(path.with_suffix('.md'), result)
    return result


def write_table(path, result):
    primary = result.get('primary_endpoint', {})
    text = ['# Replay duel — ' + result['period'] + ' — EXPLORATORY', '',
            'EXPLORATORY: ' + spec.spec()['evidence']['replay'], '', spec.PROVENANCE, '', *spec.spec()['residual_risks'], '',
            '## Primary endpoint (declared in the freeze)', '',
            f"{primary.get('panel')} {primary.get('metric')}: state {primary.get('state')}, base rate {primary.get('base_rate')}, "
            f"pairs {primary.get('n_pairs')}, sessions {primary.get('n_sessions')}", '',
            'Brier skill: ' + json.dumps(primary.get('brier_skill')) + '; 95% CI: ' + json.dumps(primary.get('ci95')), '',
            result.get('verdict_wording', ''), '',
            '## Failures (never dropped silently)', '', json.dumps(result.get('failures'), sort_keys=True), '',
            '## Full table', '',
            '| Variant / horizon | Model | Trades / decisions | Hits / scored | Abstain | Hit rate | Brier | Log loss | Brier base | Log loss base | R | Sharpe | Drawdown | ESS | Trade basis |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(value):
        return '—' if value is None else f'{value:.5g}'
    for panel, models in result['head_to_head'].items():
        for family, row in models.items():
            text.append('| ' + ' | '.join([panel, family, f"{row['n_trade_decisions']} / {row['n_decisions']}", f"{row.get('hits', 0)} / {row['n_scored_trades']}"] +
                [fmt(row.get(k)) for k in ('abstention_rate', 'hit_rate', 'brier', 'log_loss', 'brier_base_rate',
                 'log_loss_base_rate', 'expectancy_R', 'daily_pnl_sharpe', 'max_drawdown', 'probability_ess_cluster_day_kish')] +
                [row.get('trade_cost_basis', 'GROSS_DIAGNOSTIC')]) + ' |')
    text.extend(['', 'Constant 0.5 baseline on every nonempty probability panel: Brier 0.25; log loss ln(2).', '', 'Paired tests (negative Opus minus GPT loss favors Opus):', '',
                 '| Panel / loss | Pairs | Days | Blocks | Mean difference | p | Holm p |',
                 '|---|---:|---:|---:|---:|---:|---:|'])
    for panel, row in result['paired_tests'].items():
        text.append('| ' + ' | '.join([panel] + [fmt(row.get(k)) for k in
            ('n_pairs', 'n_days', 'n_blocks', 'mean_opus_minus_gpt', 'p_two_sided', 'p_holm')]) + ' |')
    text.extend(['', *result['limitations'], ''])
    Path(path).write_text('\n'.join(text))
