import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench import marketdata as md
from scripts.trading_lab.replay_bench.grant import Budget, ReplayError, instant
from scripts.trading_lab.replay_seal_forward import collect, collect_plan, experiment as ex, forward, models, probe4, spec
from scripts.trading_lab.sources.canonical import sha256_canonical

AT = instant('2026-10-20T01:00:00Z')


@pytest.fixture
def grants(tmp_path, monkeypatch):
    collection = {'id': collect.COLLECTION_GRANT, 'expires': '2026-11-30'}
    bench = {'authorization': 'replay-bench-v1', 'granted_at': '2026-10-01T00:00:00Z',
             'not_after': '2026-11-30T23:59:59Z', 'scope': {collect.SEC_WWW: {'max_requests_per_day': 10},
                                                        md.ALPACA: {'max_requests_per_day': 10}}}
    api = {'id': 'replay-bench-api-v1', 'expires': '2026-11-30', 'claude_api': {
        'key_file': str(tmp_path/'api.env'), 'replay_cap_usd': 60, 'forward_duel_cap_usd_per_month': 10}}
    paths = {}
    for name, value in [('collection', collection), ('bench', bench), ('api', api)]:
        paths[name] = tmp_path/(name + '.json')
        paths[name].write_text(json.dumps(value))
    config = spec.spec()
    config['data_collection']['collection_grant_sha256'] = sha256_canonical(collection)
    config['data_collection']['api_grant_sha256'] = sha256_canonical(api)
    config['authorization']['payload_sha256'] = sha256_canonical(bench)
    monkeypatch.setattr(spec, 'spec', lambda: config)
    return collect.Grants(paths['collection'], paths['bench'], clock=lambda: AT), paths


def test_collector_indexes_cache_and_counts_retries_and_refuses_forbidden_requests(tmp_path, grants):
    authority, paths = grants
    keys = tmp_path/'keys.env'
    keys.write_text('APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\n')
    calls, sleeps = [], []
    def getter(url, headers):
        calls.append(url)
        if len(calls) == 1:
            raise ReplayError('HTTP_429')
        return b'{"news":[],"next_page_token":null}'
    collector = collect.Collector(authority, Budget(tmp_path/'budget'), tmp_path/'cache', alpaca_env=keys,
                                  getter=getter, sleep=sleeps.append, clock=lambda: AT)
    params = {'start': '2026-07-01T00:00:00Z', 'end': '2026-07-02T00:00:00Z'}
    collector.get(md.ALPACA, '/v1beta1/news', params)
    collector.get(md.ALPACA, '/v1beta1/news', params)
    assert len(calls) == collector.budget.used('data.alpaca.markets', AT) == 2
    assert 5 in sleeps
    index = tmp_path/'archive_index.json'
    assert collector.save(index) == 1
    from scripts.trading_lab.replay_seal_forward.archive import Archive
    assert len(Archive(tmp_path, index).objects) == 1
    for origin, path, p in [(collect.PAPER, '/v2/orders', {}), (collect.PAPER, '/v2/account', {}),
                            (collect.PAPER, '/v2/assets', {'status': 'all'})]:
        with pytest.raises(ReplayError):
            collector.get(origin, path, p)
    with pytest.raises(ReplayError, match='PROTECTED_REQUEST'):
        md.crypto_bars(collector, ['BTC-USD'], '2026-10-01T00:00:00Z', '2026-10-02T00:00:00Z')
    with pytest.raises(ReplayError, match='SCORING_SYMBOL_NOT_FORECAST'):
        md.stock_trades(collector, ['UNPLANNED'], AT)
    paths['collection'].write_text('{}')
    with pytest.raises(ReplayError, match='GRANT_CHANGED'):
        collector.get(md.ALPACA, '/v1beta1/news', params)


def test_collector_stops_4xx_and_cannot_record_a_receipt_at_T(tmp_path, grants):
    authority, _ = grants
    def forbidden(url, headers):
        raise ReplayError('HTTP_403')
    collector = collect.Collector(authority, Budget(tmp_path/'budget'), tmp_path/'cache',
                                  sec_user_agent='synthetic', getter=forbidden, sleep=lambda _: None, clock=lambda: AT)
    with pytest.raises(ReplayError, match='HTTP_403'):
        collector.get(collect.SEC_WWW, '/files/company_tickers.json')
    assert collector.budget.used('www.sec.gov', AT) == 1
    collector.before = AT
    with pytest.raises(ReplayError, match='DEADLINE_PASSED'):
        collector.get(collect.SEC_WWW, '/files/company_tickers.json')
    assert not list((tmp_path/'cache').rglob('*.gz'))


def test_150_nightly_cap_inside_opus_daily_cap_and_forward_reservation(tmp_path):
    budget = models.DuelBudget(tmp_path)
    for _ in range(149):
        budget.reserve_model('opus', 150, AT)
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        budget.reserve_model('opus', 150, AT)
    fw = forward.ForwardBudget(tmp_path)
    fw.reserve_model('opus', 150, AT)
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        budget.reserve_probe('opus', 150, AT)
    for _ in range(150):
        budget.reserve_model('gpt', 150, AT)
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        budget.reserve_model('gpt', 150, AT)
    budget.stop_model('gpt', AT)
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        forward.ForwardBudget(tmp_path).reserve_model('gpt', 150, AT)


def test_credit_error_is_a_durable_hard_stop_and_optional_api_stays_off(tmp_path, monkeypatch):
    monkeypatch.delenv('REPLAY_OPUS_BILLING', raising=False)
    assert models.api_billing(tmp_path) is None
    with pytest.raises(ReplayError, match='API_CREDIT_EXHAUSTED'):
        models.decode('Credit balance is too low', '', 1, 'opus', {})
    budget = models.DuelBudget(tmp_path)
    budget.credit_stop(AT)
    with pytest.raises(ReplayError, match='API_CREDIT_EXHAUSTED'):
        budget.reserve_model('opus', 150, instant('2026-10-21T01:00:00Z'))
    with pytest.raises(ReplayError, match='API_CREDIT_EXHAUSTED'):
        forward.ForwardBudget(tmp_path).reserve_model('opus', 150, instant('2026-10-21T13:05:00Z'))


def test_api_unknown_spend_blocks_and_scopes_enforce_dollar_caps(tmp_path, grants):
    _, paths = grants
    billing = models.ApiBilling(paths['api'], tmp_path/'billing.json', clock=lambda: AT)
    assert billing.begin('replay', 'one') == 60
    with pytest.raises(ReplayError, match='API_SPEND_UNKNOWN'):
        billing.check('forward')
    billing.record('replay', 'one', '{"total_cost_usd":60}')
    with pytest.raises(ReplayError, match='API_DOLLAR_CAP_REACHED'):
        billing.check('replay')
    assert billing.begin('forward', 'two') == 10
    with pytest.raises(ReplayError, match='API_SPEND_UNKNOWN'):
        billing.record('forward', 'two', '{}')
    with pytest.raises(ReplayError, match='API_SPEND_UNKNOWN'):
        billing.check('forward')


def prereg():
    closes = {f'S{i}': {'2026-09-30': 100, '2026-10-08': 110 if i%2 else 90,
                        '2026-10-16': 100} for i in range(20)}
    questions = probe4.questions(closes)
    return {'questions': questions, 'criteria': probe4.rule(questions), 'rule': probe4.rule(questions)['text']}


def test_probe4_is_balanced_flag_rule_cuts_only_flagged_model(tmp_path, monkeypatch):
    value = prereg()
    always = {q['id']: 'up' for q in value['questions']}
    correct = {q['id']: 'up' if q['up'] else 'down' for q in value['questions']}
    assert probe4.grade(value, always)['always_up'] == len(always)//2
    assert not probe4.grade(value, always)['flagged']
    flagged = probe4.grade(value, correct)
    assert flagged['flagged'] and flagged['test_end'] == '2026-09-30'
    freeze = tmp_path/'freeze.json'
    freeze.write_text(json.dumps({'test_end': {'opus': '2026-09-30', 'gpt': '2026-10-09'}}))
    monkeypatch.setattr(ex, 'FREEZE', freeze)
    assert max(ex.planned_for('test', 'ab', 'opus')) == '2026-09-30'
    assert max(ex.planned_for('test', 'ab', 'gpt')) == '2026-10-09'
    from scripts.trading_lab.replay_seal_forward.score import within_test_cut
    assert within_test_cut({'available_at': '2026-09-30T20:00:00Z'}, 'test', 'opus')
    assert not within_test_cut({'available_at': '2026-10-01T20:00:00Z'}, 'test', 'opus')
    assert within_test_cut({'available_at': '2026-10-01T20:00:00Z'}, 'test', 'gpt')
    assert probe4.grade(value, always)['test_end'] == '2026-10-09'
    bad = dict(value, criteria=dict(value['criteria'], pooled_threshold=0))
    with pytest.raises(ReplayError, match='PROBE4_RULE_CHANGED'):
        probe4.grade(bad, correct)


def test_probe4_requires_committed_digest_and_never_retries_unknown_call(tmp_path, monkeypatch):
    target = tmp_path/probe4.PREREGISTRATION
    target.write_text(json.dumps(prereg()))
    public = dict(preregistration_sha256='0'*64)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: json.dumps(public).encode())
    with pytest.raises(ReplayError, match='PREREGISTRATION_HASH_MISMATCH'):
        probe4.committed_preregistration(tmp_path)
    value = prereg()
    import hashlib
    value['prompt_sha256'] = hashlib.sha256(probe4.prompt(value['questions']).encode()).hexdigest()
    monkeypatch.setattr(probe4, 'committed_preregistration', lambda _: (value, {}))
    attempts = []
    def timeout(args, **kwargs):
        attempts.append(args)
        raise subprocess.TimeoutExpired(args, 1)
    budget = models.DuelBudget(tmp_path/'budget')
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        probe4.ask(tmp_path, 'opus', budget, runner=timeout, clock=lambda: AT)
    with pytest.raises(ReplayError, match='ALREADY_ASKED'):
        probe4.ask(tmp_path, 'opus', budget, runner=timeout, clock=lambda: AT)
    assert len(attempts) == 1


def test_live_collector_real_shapes_feed_a_snapshot_with_receipts_before_T(tmp_path, grants, monkeypatch):
    from datetime import timedelta
    from urllib.parse import parse_qs, urlsplit
    from scripts.trading_lab.replay_bench import snapshot as sn
    from scripts.trading_lab.replay_bench.grant import iso
    from scripts.trading_lab.replay_seal_forward import inputs
    authority, _ = grants
    T = forward.decision_time('2026-10-20')
    at = T - timedelta(hours=1)
    monkeypatch.setattr(inputs, 'declaration', lambda _: {'synthetic': True})
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c'*64)
    key = tmp_path/'keys.env'
    key.write_text('APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\n')
    requests = []
    def getter(url, headers):
        requests.append(url)
        parsed = urlsplit(url)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.path == '/v1beta1/news':
            body = {'news': [{'id': 1, 'headline': 'SMLL lowers guidance', 'source': 'synthetic', 'symbols': ['SMLL'],
                             'created_at': iso(at-timedelta(hours=1)), 'updated_at': iso(at-timedelta(hours=1))}]}
        elif parsed.path == '/v2/assets':
            body = [{'id': 'synthetic-' + s, 'symbol': s, 'class': 'us_equity', 'name': s, 'status': 'active', 'shortable': True,
                     'easy_to_borrow': True} for s in sn.STOCKS + sn.ETFS + sn.BENCHMARKS + ('SMLL',)] if params['status'] == 'active' else []
        elif parsed.path == '/files/company_tickers.json':
            body = {'0': {'ticker': 'SMLL', 'cik_str': 1}}
        elif parsed.path.startswith('/submissions/'):
            body = {'cik': '0000000001', 'tickers': ['SMLL'], 'filings': {'recent': {'form': [], 'acceptanceDateTime': [], 'accessionNumber': []}}}
        else:
            if 'crypto' in parsed.path:
                times = []
                d = instant('2025-01-01T00:00:00Z')
                while d < at:
                    times.append(d)
                    d += timedelta(days=1)
            else:
                times = [s.open_at for s in md.sessions('2025-01-01', '2026-10-20')]
            rows = [{'t': iso(d), 'o': 100, 'h': 102, 'l': 99, 'c': 101, 'v': 1e6} for d in times]
            body = {'bars': {s: rows for s in params['symbols'].split(',')}}
        return json.dumps(body).encode()
    def factory(*a, **kw):
        return collect.Collector(*a, **kw, getter=getter, sleep=lambda _: None)
    result = collect_plan.live(tmp_path, authority, key, 'synthetic', clock=lambda: at, collector_factory=factory)
    assert result['shown_candidates'] == 1 and result['session'] == '2026-10-20'
    value = forward.snapshot(tmp_path, T)
    assert value['contexts']['cd']['candidates'][0]['symbol'] == 'SMLL'
    assert value['archive']['objects']
    count = len(requests)
    with pytest.raises(ReplayError, match='OUTSIDE_LIVE_COLLECTION_WINDOW'):
        collect_plan.live(tmp_path, authority, key, 'synthetic', clock=lambda: T, collector_factory=factory)
    assert len(requests) == count
