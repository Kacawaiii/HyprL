"""Replay bench: causality of snapshots, budgets, protected guard, probe rule, labels and metrics (synthetic only)."""
from datetime import datetime, timedelta, timezone
import json
import subprocess

import pytest

from scripts.trading_lab.replay_bench import grant as gr, labels as lb, marketdata as md, metrics, models, probe
from scripts.trading_lab.replay_bench import run, snapshot as sn, spec
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso

UTC = timezone.utc
T = instant('2026-07-15T12:00:00Z')


def daily_bars(day_count=40, end='2026-07-14', price=100.0, volume=1e6, step=0.001):
    """Alpaca 1Day shape: t at 04:00Z of each session date, raw o/h/l/c/v/vw/n."""
    sessions = [s.session_date for s in md.sessions('2026-04-01', end)][-day_count:]
    bars = []
    for i, day in enumerate(sessions):
        close = price * (1 + step) ** i
        bars.append({'t': day + 'T04:00:00Z', 'o': close * 0.999, 'h': close * 1.01, 'l': close * 0.99, 'c': close,
                     'v': volume, 'vw': close, 'n': 1000})
    return bars


def crypto_daily(day_count=40, end='2026-07-14'):
    last = datetime.fromisoformat(end).replace(tzinfo=UTC)
    return [{'t': iso(last - timedelta(days=day_count - 1 - i)), 'o': 10, 'h': 11, 'l': 9, 'c': 10 + i * 0.01, 'v': 5,
             'vw': 10, 'n': 20} for i in range(day_count)]


def item(i, created, symbols, headline='Company update', updated=None, url=None):
    return {'id': i, 'headline': headline, 'summary': '', 'source': 'benzinga', 'symbols': symbols,
            'url': url or f'https://www.benzinga.com/news/26/07/{i}/x', 'created_at': created,
            'updated_at': updated or created, 'author': 'Benzinga Newsdesk'}


def universe_bars():
    bars = {a: daily_bars() for a in sn.STOCKS + sn.ETFS + sn.BENCHMARKS}
    # A bar for the decision session itself exists in the store (t=04:00Z < T) but closes after T.
    bars['AAPL'] = bars['AAPL'] + [{'t': '2026-07-15T04:00:00Z', 'o': 1, 'h': 999, 'l': 1, 'c': 999, 'v': 1, 'vw': 1, 'n': 1}]
    bars.update({a: crypto_daily() for a in sn.CRYPTO})
    return bars


def test_spec_is_pinned_and_binds_prompts():
    value = spec.spec()
    assert value['provenance'] == spec.PROVENANCE == 'REPLAY_POST_CUTOFF'
    assert value['variant_count_declared'] == len(value['variants']) == 4
    assert value['models']['tools'] == []


def test_context_a_sees_only_information_before_T():
    news = [item(1, '2026-07-15T11:59:59Z', ['NVDA'], 'Before T'),
            item(2, '2026-07-15T12:00:00Z', ['NVDA'], 'At T'),
            item(3, '2026-07-15T13:00:00Z', ['NVDA'], 'After T'),
            item(4, '2026-07-14T10:00:00Z', ['NVDA'], 'Revised after T', updated='2026-07-15T12:30:00Z'),
            item(5, '2026-07-14T10:00:00Z', ['SOLUSD', 'BTCUSD'], 'Tagged bitcoin'),
            item(6, '2026-07-11T10:00:00Z', ['NVDA'], 'Older than 72h'),
            item(7, '2026-07-14T09:00:00Z', ['SPY'], 'Macro item')]
    context, manifest = sn.context_a(T, universe_bars(), news)
    titles = [i['title'] for i in context['headlines']['NVDA']['items']]
    assert titles == ['Before T']
    assert context['headlines']['SOL-USD']['items'] == []
    assert [i['title'] for i in context['headlines']['macro']['items']] == ['Macro item']
    # The session that closes after T is invisible: last price is the previous close, not the 999 print.
    assert context['prices']['AAPL']['last_price_at'] == '2026-07-14T20:00:00Z'
    assert 999 not in context['prices']['AAPL']['recent_closes']
    assert context['prices']['SOL-USD']['state'] == 'AVAILABLE'
    assert 'received_at' not in json.dumps(context)
    sealed = sn.seal(context, manifest, T)
    assert len(sealed['snapshot_hash']) == 64 and sealed['context_hash'] != sealed['snapshot_hash']


def test_validate_rejects_any_future_item():
    context, manifest = sn.context_a(T, universe_bars(), [item(1, '2026-07-15T11:00:00Z', ['NVDA'])])
    context['headlines']['NVDA']['items'][0]['published_at'] = '2026-07-15T12:00:00Z'
    with pytest.raises(ReplayError) as error:
        sn.seal(context, manifest, T)
    assert error.value.code == 'FUTURE_NEWS'
    context, manifest = sn.context_a(T, universe_bars(), [])
    context['prices']['AAPL']['last_price_at'] = '2026-07-15T20:00:00Z'
    with pytest.raises(ReplayError):
        sn.validate(context, T)


def test_context_c_liquidity_band_price_floor_and_negative_ranking():
    start = md.previous_close(T)
    assert start == instant('2026-07-14T20:00:00Z')
    news = [item(1, '2026-07-14T21:00:00Z', ['SMLL'], 'SMLL cuts guidance after weak quarter'),
            item(2, '2026-07-14T22:00:00Z', ['MIDC'], 'MIDC announces new product'),
            item(3, '2026-07-14T22:00:00Z', ['PENY'], 'PENY misses'),
            item(4, '2026-07-14T22:00:00Z', ['HUGE'], 'HUGE misses'),
            item(5, '2026-07-14T19:00:00Z', ['SMLL'], 'Before the window'),
            item(6, '2026-07-14T23:00:00Z', ['AAPL'], 'Live universe name'),
            item(7, '2026-07-14T23:00:00Z', ['A', 'B', 'C'], 'Sector roundup')]
    bars = {'SMLL': daily_bars(price=20, volume=2e6), 'MIDC': daily_bars(price=50, volume=4e6),
            'PENY': daily_bars(price=2, volume=1e8), 'HUGE': daily_bars(price=500, volume=1e7)}
    keywords = spec.spec()['variants']['c_news_small_mid']['negative_keywords']
    context, manifest = sn.context_c(T, start, news, bars, keywords)
    shown = {c['symbol']: c for c in context['candidates']}
    assert set(shown) == {'SMLL', 'MIDC'}
    assert [h['news_id'] for h in shown['SMLL']['headlines']] == [1]
    assert manifest['rejected'] == {'PRICE_LT_5': 1, 'DOLLAR_VOLUME_OUT_OF_BAND': 1}
    sn.seal(context, manifest, T)


def test_anonymize_masks_names_and_keeps_returns():
    news = [item(1, '2026-07-15T10:00:00Z', ['NVDA'], 'Nvidia and Microsoft expand deal; NVDA up'),
            item(2, '2026-07-15T10:00:00Z', ['SPY'], 'Tesla, Apple lead futures')]
    sealed = sn.seal(*sn.context_a(T, universe_bars(), news), T)
    anon, mapping = sn.anonymize(sealed, seed=7)
    text = json.dumps(anon)
    for word in ('Nvidia', 'Microsoft', 'NVDA', 'Tesla', 'Apple', 'benzinga.com'):
        assert word not in text
    code = next(c for c, a in mapping.items() if a == 'NVDA')
    original, masked = sealed['context']['prices']['NVDA'], anon['prices'][code]
    assert masked['returns'] == original['returns'] and masked['recent_closes'] != original['recent_closes']
    assert sorted(mapping.values()) == sorted(sn.UNIVERSE)


class FakeGrant:
    payload = {'scope': {o: {'max_requests_per_day': 2} for o in gr.PATHS},
               'models': {'claude_cli_subscription': {'opus_calls_per_day': 2, 'sonnet_calls_per_day': 2}}}

    def check(self):
        pass

    origin_limit = gr.Grant.origin_limit
    model_limit = gr.Grant.model_limit


def test_transport_paths_budget_cache_and_protected_guard(tmp_path):
    calls = []
    getter = lambda url, headers: calls.append(url) or b'{"news": [], "next_page_token": null}'
    t = gr.Transport(FakeGrant(), gr.Budget(tmp_path), tmp_path / 'cache', getter=getter, sleep=lambda s: None)
    t._alpaca = {'APCA_API_KEY_ID': 'k', 'APCA_API_SECRET_KEY': 's'}
    with pytest.raises(ReplayError):
        t.get(md.ALPACA, '/v2/orders')
    t.get(md.ALPACA, '/v1beta1/news', {'start': 'a'})
    _, meta = t.get(md.ALPACA, '/v1beta1/news', {'start': 'a'})
    assert meta['cached'] and len(calls) == 1
    t.get(md.ALPACA, '/v1beta1/news', {'start': 'b'})
    with pytest.raises(ReplayError) as error:
        t.get(md.ALPACA, '/v1beta1/news', {'start': 'c'})
    assert error.value.code == 'BUDGET_EXHAUSTED'
    with pytest.raises(ReplayError) as error:
        md.crypto_bars(t, ['BTC-USD'], '2026-09-02T00:00:00Z', '2026-09-03T00:00:00Z')
    assert error.value.code == 'PROTECTED_REQUEST'
    md.guard(['BTC/USD'], '2026-07-01T00:00:00Z', '2026-08-01T00:00:00Z')


def test_claude_runs_without_tools_and_stops_the_day_on_usage_limit(tmp_path):
    seen = []

    def runner(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 1, stdout='', stderr="You've hit your limit")
    claude = models.Claude(FakeGrant(), gr.Budget(tmp_path), tmp_path / 't', runner=runner, clock=lambda: instant('2026-10-19T01:00:00Z'))
    with pytest.raises(ReplayError) as error:
        claude.call('opus', 'x', {'type': 'object'}, tag='t')
    assert error.value.code == 'MODEL_USAGE_LIMIT'
    assert seen[0][seen[0].index('--tools') + 1] == '' and seen[0][seen[0].index('--model') + 1] == 'opus'
    with pytest.raises(ReplayError):
        claude.call('opus', 'x', {'type': 'object'}, tag='t')
    assert len(seen) == 1


def test_a_429_result_event_is_a_usage_limit(tmp_path):
    event = {'type': 'result', 'subtype': 'success', 'is_error': True, 'api_error_status': 429,
             'result': "You've hit your session limit · resets 12am (UTC)", 'permission_denials': []}
    runner = lambda args, **kwargs: subprocess.CompletedProcess(args, 1, stdout=json.dumps(event), stderr='')
    claude = models.Claude(FakeGrant(), gr.Budget(tmp_path), tmp_path / 't', runner=runner, clock=lambda: instant('2026-10-19T01:00:00Z'))
    with pytest.raises(ReplayError) as error:
        claude.call('opus', 'x', {'type': 'object'}, tag='t')
    assert error.value.code == 'MODEL_USAGE_LIMIT'
    assert gr.Budget(tmp_path).stopped(instant('2026-10-19T01:00:00Z')) == 'MODEL_USAGE_LIMIT'


def test_probe_scoring_and_floor_rule():
    event = {'answer_key_all_of_any': [['world labs'], ['8\\.2']]}
    assert probe.correct(event, 'World Labs for $8.2B') and not probe.correct(event, 'World Labs')
    rows = [{'family': 'opus', 'id': f'{m}{i}', 'month': m, 'knows': True, 'correct': i < k}
            for m, k in (('2026-05', 3), ('2026-06', 2), ('2026-07', 1)) for i in range(6)]
    result = probe.summarize(rows, {'known_min_correct': 2, 'events_per_month': 6})['opus']
    assert result['known_months'] == ['2026-05', '2026-06'] and result['replay_floor'] == '2026-07-01'
    rows = [dict(r, correct=False) for r in rows]
    assert probe.summarize(rows, {'known_min_correct': 2, 'events_per_month': 6})['opus']['replay_floor'] is None


def test_edgar_acceptance_is_read_as_the_later_instant():
    assert md.acceptance_upper_bound('2026-07-14T16:05:00.000Z') == instant('2026-07-14T20:05:00Z')


def test_outcome_costs_hits_and_r():
    costs = spec.spec()['costs']
    assert lb.tier(2e9, costs) == 0.0002 and lb.tier(5e8, costs) == 0.0005 and lb.tier(2e7, costs) == 0.0015
    result = lb.outcome(-1, -0.01, 0.0024, 0.02, 1)
    assert result['hit'] and round(result['net'], 4) == 0.0076 and round(result['R'], 2) == 0.38
    assert not lb.outcome(1, 0.001, 0.0024, 0.02, 1)['hit']


def test_metrics_counts_wilson_and_ess():
    days = ['d1', 'd2', 'd3']
    rows = [('d1', {'net': .01, 'gross': .012, 'hit': True, 'hit_gross': True, 'R': .5, 'p': .6, 'outperform': True,
                    'cluster': 'tech', 'direction': 'UP'}),
            ('d1', {'net': .01, 'gross': .012, 'hit': True, 'hit_gross': True, 'R': .5, 'p': .6, 'outperform': True,
                    'cluster': 'tech', 'direction': 'UP'}),
            ('d2', {'net': -.02, 'gross': -.018, 'hit': False, 'hit_gross': False, 'R': -1, 'p': .4, 'outperform': True,
                    'cluster': 'energy', 'direction': 'DOWN'})]
    out = metrics.summarize(rows, days)
    assert out['hits'] == 2 and out['n_views'] == 3 and out['ess_cluster_day_direction'] == 2
    assert out['brier'] == round(((.6 - 1) ** 2 * 2 + (.4 - 1) ** 2) / 3, 4)
    assert out['max_drawdown'] == round(-.02 / 1.01, 5)
    assert metrics.wilson(0, 0) is None


def test_decide_ab_records_both_variants_with_hashes(tmp_path, monkeypatch):
    news = [item(1, '2026-07-15T10:00:00Z', ['XLE'], 'Oil jumps')]
    universe = type('U', (), {'bars': universe_bars(), 'news': news})()
    context, _ = sn.context_a(T, universe.bars, news)
    url = context['headlines']['XLE']['items'][0]['url']

    def view(asset, horizon):
        directional = asset == 'XLE' and horizon == '1d'
        return {'asset': asset, 'horizon': horizon, 'view': 'UP' if directional else 'ABSTAIN',
                'p_outperform': .56 if directional else .5, 'confidence_reason': 'r',
                'catalysts': [{'url': url, 'published_at': '2026-07-15T10:00:00Z', 'fact': 'f'}] if directional else [],
                'priced_in_assessment': '', 'second_order': '', 'counter_thesis': '', 'falsifier': '', 'event_risk': []}

    class Fake:
        def __init__(self):
            self.calls = []

        def call(self, family, text, schema, *, tag):
            self.calls.append((family, tag))
            assert 'POINT-IN-TIME REPLAY' in text
            if '-analyst' in tag:
                views = [view(a, h) for a in context['universe'] for h in ('1d', '5d')]
                if len(self.calls) == 1:   # first attempt cites a URL outside the snapshot: retried once
                    views[0] = dict(view(context['universe'][0], '1d'), view='UP', p_outperform=.6, catalysts=[
                        {'url': 'https://example.com/later', 'published_at': '2026-07-15T09:00:00Z', 'fact': 'x'}])
                return {'regime': [], 'views': views}, {'family': family}
            return {'verdicts': [{'analyst': 'analyst_claude', 'asset': a, 'horizon': h,
                                  'verdict': 'DOWNGRADE' if (a, h) == ('XLE', '1d') else 'KEEP',
                                  'reason_code': 'priced_in' if (a, h) == ('XLE', '1d') else 'abstained',
                                  'adjusted_p': .5 if (a, h) == ('XLE', '1d') else None, 'note': ''}
                                 for a in context['universe'] for h in ('1d', '5d')]}, {'family': family}
    fake = Fake()
    record = run.decide_ab('2026-07-15', universe, fake, tmp_path, {'trial_id': 't', 'spec_hash': spec.SPEC_HASH})
    assert record['provenance'] == 'REPLAY_POST_CUTOFF' and len(record['snapshot_hash']) == 64
    assert record['analyst']['errors'] == ['CITATION_NOT_IN_SNAPSHOT'] and len(fake.calls) == 3
    b = {(v['asset'], v['horizon']): v for v in record['views']['b_v1_no_reviewer']}
    a = {(v['asset'], v['horizon']): v for v in record['views']['a_v1_as_live']}
    assert b[('XLE', '1d')]['view'] == 'UP' and a[('XLE', '1d')]['view'] == 'ABSTAIN'
    assert run.decide_ab('2026-07-15', universe, fake, tmp_path, {}) == record and len(fake.calls) == 3


def test_citations_match_exact_time_and_url_pairs():
    slug = 'https://www.benzinga.com/news/26/06/1/openais-jalapeno-chip-and-the-stock-is-down-20-this-month'
    context = {'headlines': {'AVGO': {'items': [{'url': slug, 'published_at': '2026-06-29T12:33:05Z'}]},
                             'macro': {'items': [{'url': 'https://www.benzinga.com/quote/SPY', 'published_at': '2026-06-30T10:10:15Z'},
                                                 {'url': 'https://www.benzinga.com/quote/SPY', 'published_at': '2026-06-29T14:31:53Z'}]}}}
    view = lambda url, at: {'views': [{'catalysts': [{'url': url, 'published_at': at}]}]}
    assert run.citations_ok(view('https://www.benzinga.com/quote/SPY', '2026-06-29T14:31:53Z'), context) == 0
    for url, at in ((slug.replace('jalapeno', 'jalape-o'), '2026-06-29T12:33:05Z'), (slug, '2026-06-29T12:33:06Z'), ('https://www.reuters.com/later', '2026-06-29T12:33:05Z')):
        with pytest.raises(Exception) as error:
            run.citations_ok(view(url, at), context)
        assert error.value.code == 'CITATION_NOT_IN_SNAPSHOT'
