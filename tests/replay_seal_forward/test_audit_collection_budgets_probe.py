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
    keys = tmp_path/'keys.env'
    keys.write_text('APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\n')
    collector = collect.Collector(authority, Budget(tmp_path/'budget'), tmp_path/'cache',
                                  alpaca_env=keys, getter=forbidden, sleep=lambda _: None, clock=lambda: AT)
    with pytest.raises(ReplayError, match='HTTP_403'):
        collector.get(md.ALPACA, '/v1beta1/news')
    assert collector.budget.used('data.alpaca.markets', AT) == 1
    collector.before = AT
    with pytest.raises(ReplayError, match='DEADLINE_PASSED'):
        collector.get(md.ALPACA, '/v1beta1/news')
    assert not list((tmp_path/'cache').rglob('*.gz'))


def test_sec_operator_unavailable_never_dispatches_and_is_global(tmp_path, grants):
    authority, _ = grants
    calls = []
    def forbidden(url, headers):
        calls.append(url)
        raise ReplayError('HTTP_403')
    collector = collect.Collector(authority, Budget(tmp_path/'collector'), tmp_path/'cache',
        sec_user_agent='synthetic', getter=forbidden, clock=lambda: AT)
    assert collect.sec_or_unavailable(collector, ['SMLL'], tmp_path) == 0
    assert collect.sec_or_unavailable(collector, ['OTHER'], tmp_path) == 0
    assert len(calls) == 0
    with pytest.raises(ReplayError, match='SEC_COLLECTION_DISABLED'):
        collector.get(collect.SEC_WWW, '/files/company_tickers.json')
    index = tmp_path/'archive_index.json'
    collector.save(index)
    assert json.loads(index.read_text())['sec_availability']['state'] == 'UNAVAILABLE'


def test_incomplete_price_response_is_refreshed_once_and_preserved(tmp_path, grants):
    authority, _ = grants
    keys = tmp_path/'keys.env'
    keys.write_text('APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\n')
    calls = []
    def getter(url, headers):
        calls.append(url)
        return b'{"bars":{"SMLL":[]},"next_page_token":null}'
    collector = collect.Collector(authority, Budget(tmp_path/'collector'), tmp_path/'cache', alpaca_env=keys,
        getter=getter, sleep=lambda _: None, clock=lambda: AT)
    params = {'start':'2026-07-01T00:00:00Z', 'end':'2026-07-02T00:00:00Z', 'symbols':'SMLL',
              'timeframe':'1Day', 'feed':'sip', 'adjustment':'raw'}
    collector.get(md.ALPACA, '/v2/stocks/bars', params)
    original = next((tmp_path/'cache').rglob('*.gz')).read_bytes()
    collector.retry_missing = True
    for _ in range(3):
        collector.get(md.ALPACA, '/v2/stocks/bars', params)
    assert len(calls) == 2
    assert next((tmp_path/'missing-label-originals').rglob('*.gz')).read_bytes() == original


def test_missing_label_retry_is_closed_without_reading_or_dispatching(tmp_path):
    with pytest.raises(ReplayError, match='MISSING_LABEL_RETRY_CLOSED'):
        collect_plan.retry_missing_labels(tmp_path, SimpleNamespace(), tmp_path/'no-new-plan.json')
    assert not list(tmp_path.iterdir())


def test_400_nightly_cap_inside_opus_daily_cap_and_forward_reservation(tmp_path):
    budget = models.DuelBudget(tmp_path)
    for _ in range(399):
        budget.reserve_model('opus', 400, AT)
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        budget.reserve_model('opus', 400, AT)
    fw = forward.ForwardBudget(tmp_path)
    fw.reserve_model('opus', 400, AT)
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        budget.reserve_probe('opus', 400, AT)
    for _ in range(400):
        budget.reserve_model('gpt', 400, AT)
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        budget.reserve_model('gpt', 400, AT)
    budget.stop_model('gpt', AT)
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        forward.ForwardBudget(tmp_path).reserve_model('gpt', 400, AT)


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


def test_failed_probe_envelopes_are_rejected_even_with_an_answer():
    raw = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': True,
                      'result': 'Not logged in. Please run /login', 'modelUsage': {}})
    with pytest.raises(ReplayError, match='MODEL_FAILED'):
        models.decode(raw, '', 1, 'opus', {})
    events = [{'type': 'thread.started'}, {'type': 'item.completed', 'item': {'type': 'error', 'message': 'synthetic CLI failure'}},
              {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{}'}}, {'type': 'turn.completed'}]
    with pytest.raises(ReplayError, match='MODEL_FAILED'):
        models.decode('\n'.join(json.dumps(e) for e in events), '', 0, 'gpt', {})


def test_gpt_launch_disables_code_mode_and_local_warning_sources(tmp_path):
    args = models.argv('gpt', {}, tmp_path/'schema.json')
    assert 'suppress_unstable_features_warning=true' in args
    for name in ('shell_tool', 'unified_exec', 'code_mode', 'code_mode_only', 'code_mode_host', 'apps', 'plugins'):
        index = args.index(name)
        assert args[index-1] == '--disable'
    assert '--ignore-user-config' in args and '--ignore-rules' in args and '--no-daemon' in args


def test_tainted_quota_envelope_still_stops_later_dispatches(tmp_path):
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        events = [{'type': 'thread.started'}, {'type': 'item.completed',
                  'item': {'type': 'error', 'message': 'usage limit reached'}},
                  {'type': 'item.completed', 'item': {'type': 'command_execution'}}, {'type': 'turn.completed'}]
        return subprocess.CompletedProcess(args, 0, '\n'.join(json.dumps(e) for e in events), '')
    budget = models.DuelBudget(tmp_path)
    client = models.Models(SimpleNamespace(check=lambda: None, model_limit=lambda _: 150), budget,
                           tmp_path/'transcripts', clock=lambda: AT, window=lambda _: 900, runner=runner)
    with pytest.raises(ReplayError, match='TAINTED_RUN'):
        client.call('gpt', 'synthetic', {}, tag='first')
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        client.call('gpt', 'synthetic', {}, tag='second')
    assert len(calls) == 1


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


def test_historical_probe_rule_stays_fixed_and_current_test_cuts_are_uniform(tmp_path, monkeypatch):
    value = prereg()
    always = {q['id']: 'up' for q in value['questions']}
    correct = {q['id']: 'up' if q['up'] else 'down' for q in value['questions']}
    assert probe4.grade(value, always)['always_up'] == len(always)//2
    assert not probe4.grade(value, always)['flagged']
    flagged = probe4.grade(value, correct)
    assert flagged['flagged'] and flagged['test_end'] == '2026-09-30'
    freeze = tmp_path/'freeze.json'
    freeze.write_text(json.dumps({'test_end': {'opus': '2026-09-30', 'gpt': '2026-09-30'}}))
    monkeypatch.setattr(ex, 'FREEZE', freeze)
    assert max(ex.planned_for('test', 'ab', 'opus')) == '2026-09-30'
    assert max(ex.planned_for('test', 'ab', 'gpt')) == '2026-09-30'
    from scripts.trading_lab.replay_seal_forward.score import within_test_cut
    assert within_test_cut({'available_at': '2026-09-30T20:00:00Z'}, 'test', 'opus')
    assert not within_test_cut({'available_at': '2026-10-01T20:00:00Z'}, 'test', 'opus')
    assert not within_test_cut({'available_at': '2026-10-01T20:00:00Z'}, 'test', 'gpt')
    assert probe4.grade(value, always)['test_end'] == '2026-09-30'
    bad = dict(value, criteria=dict(value['criteria'], pooled_threshold=0))
    with pytest.raises(ReplayError, match='PROBE4_RULE_CHANGED'):
        probe4.grade(bad, correct)


def test_probe4_keeps_committed_digest_and_closed_calls_never_dispatch(tmp_path, monkeypatch):
    target = tmp_path/probe4.PREREGISTRATION
    target.write_text(json.dumps(prereg()))
    public = dict(preregistration_sha256='0'*64)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: json.dumps(public).encode())
    with pytest.raises(ReplayError, match='PREREGISTRATION_HASH_MISMATCH'):
        probe4.committed_preregistration(tmp_path)
    attempts = []
    def runner(*args, **kwargs):
        attempts.append(args)
        pytest.fail('closed probe must never dispatch')
    budget = models.DuelBudget(tmp_path/'budget')
    for family in ('opus', 'gpt'):
        for retry in (False, True):
            with pytest.raises(ReplayError, match='PROBE4_CLOSED_NOT_RUN'):
                probe4.ask(tmp_path, family, budget, runner=runner, authorized_retry=retry)
    assert attempts == [] and not budget.path.exists()


def test_closed_probe_evidence_hashes_opaque_envelopes_without_decoding(tmp_path, monkeypatch):
    import hashlib
    public = {'status': 'NOT RUN', 'families': {}, 'test_end': {f:'2026-09-30' for f in ex.FAMILIES}}
    for family in ex.FAMILIES:
        public['families'][family] = {}
        for prefix, key in (('', 'original_response_sha256'), ('retry-', 'retry_response_sha256')):
            file = tmp_path/f'probe4-{prefix}answers-{family}.response'
            file.write_bytes(b'opaque envelope, not JSON')
            public['families'][family][key] = hashlib.sha256(file.read_bytes()).hexdigest()
    artifacts = tmp_path/'docs/artifacts'
    artifacts.mkdir(parents=True)
    config = spec.spec()
    policy = config['probe']['probe4']
    policy['closure_sha256'] = sha256_canonical(public)
    (artifacts/policy['closure_artifact']).write_text(json.dumps(public))
    monkeypatch.setattr(spec, 'ROOT', tmp_path)
    monkeypatch.setattr(spec, 'spec', lambda: config)
    assert probe4.evidence(tmp_path) == public
    (tmp_path/'probe4-answers-gpt.response').write_bytes(b'changed')
    with pytest.raises(ReplayError, match='PROBE4_ENVELOPE_HASH_MISMATCH'):
        probe4.evidence(tmp_path)


def test_real_model_calls_require_systemd_before_reserving_or_dispatching(tmp_path, monkeypatch):
    monkeypatch.delenv('INVOCATION_ID', raising=False)
    budget = models.DuelBudget(tmp_path)
    with pytest.raises(ReplayError, match='PROBE4_CLOSED_NOT_RUN'):
        probe4.ask(tmp_path, 'opus', budget)
    client = models.Models(SimpleNamespace(), budget, tmp_path/'transcripts')
    with pytest.raises(ReplayError, match='MODEL_SYSTEMD_REQUIRED'):
        client.call('gpt', 'synthetic', {}, tag='no-shell')
    assert not budget.path.exists()


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
            body = [{'id': 'synthetic-' + s, 'symbol': s, 'class': 'us_equity', 'name': s, 'exchange': 'NASDAQ', 'status': 'active', 'shortable': True,
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


@pytest.mark.parametrize('scope', ['replay', 'forward'])
@pytest.mark.parametrize('tool', [None, 'command_execution', 'web_search', 'mcp_tool_call'])
def test_gpt_cli_diagnostics_require_an_event_stream_without_tools(tmp_path, scope, tool):
    events = [{'type': 'thread.started'},
              {'type': 'item.completed', 'item': {'type': 'error', 'message':
               'Under-development features enabled: skip_host_skill_discovery. Under-development features are incomplete and may behave unpredictably. To suppress this warning, set `suppress_unstable_features_warning = true` in config.toml.'}},
              {'type': 'item.completed', 'item': {'type': 'error', 'message':
               'Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.'}},
              {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"ok":true}'}},
              {'type': 'turn.completed'}]
    if tool:
        events.insert(2, {'type': 'item.completed', 'item': {'type': tool}})
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, '\n'.join(json.dumps(e) for e in events), '')
    budget = (forward.ForwardBudget if scope == 'forward' else models.DuelBudget)(tmp_path)
    client = models.Models(SimpleNamespace(check=lambda: None, model_limit=lambda _: 150), budget,
                           tmp_path/'transcripts', clock=lambda: AT, window=lambda _: 900,
                           runner=runner, scope=scope)
    if tool:
        with pytest.raises(ReplayError, match='TAINTED_RUN'):
            client.call('gpt', 'synthetic', {}, tag='diagnostics')
    else:
        output, _ = client.call('gpt', 'synthetic', {}, tag='diagnostics')
        assert output == {'ok': True}
