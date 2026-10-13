"""Synthetic health probes, read-only broker calls and durable retry/audit boundaries."""
import json
from datetime import timedelta

import pytest

from scripts.trading_lab.trader_agent.config import TraderError, instant


def test_failure_continues_chain_alerts_and_reserves_second_rehearsal(tmp_path, clock):
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState, rehearse
    state = MonitorState(tmp_path / 'monitor')
    calls, alerts = [], []
    def bad():
        calls.append('yahoo')
        raise TraderError('SOURCE_UNAVAILABLE')
    def good():
        calls.append('coinbase')
        return {'assets_checked': 5}
    clock.at = instant('2026-10-12T10:45:00Z')
    result = rehearse(state, [('yahoo', bad), ('coinbase', good)], clock=clock,
                       notify=lambda key, text: alerts.append(text) or 'DELIVERED', reports=tmp_path / 'reports')
    assert result['state'] == 'BLOCKED'
    assert calls == ['yahoo', 'coinbase'] and 'yahoo' in alerts[0]
    assert result['retry_at'] == '2026-10-12T11:15:00Z'
    assert (tmp_path / 'reports/rehearsal-2026-10-12.md').exists()
    assert rehearse(state, [], clock=clock, notify=None, reports=tmp_path / 'reports')['state'] == 'ALREADY_ATTEMPTED'
    clock.at += timedelta(minutes=30)
    again = rehearse(state, [('yahoo', good)], clock=clock, retry=True,
                     notify=lambda *args: 'DELIVERED', reports=tmp_path / 'reports')
    assert again['state'] == 'GREEN' and again['retry_at'] is None
    assert 'SOURCE_UNAVAILABLE' in (tmp_path / 'reports/rehearsal-2026-10-12.md').read_text()


@pytest.mark.parametrize('code,expected', [('SOURCE_HTTP_429', 'GREEN'),
    ('SOURCE_UNAVAILABLE', 'GREEN'), ('SOURCE_BUDGET', 'BLOCKED'),
    ('AUTHORIZATION_EXPIRED_OR_NOT_STARTED', 'BLOCKED'), ('NEWS_QUERY_UNMAPPED', 'BLOCKED')])
def test_gdelt_loss_alerts_as_partial_context_but_controls_still_block(tmp_path, clock, code, expected):
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState, rehearse
    state = MonitorState(tmp_path / 'monitor')
    def unavailable():
        raise TraderError(code)
    messages = []
    result = rehearse(state, [('gdelt', unavailable), ('executor', lambda: {'production_writes': 0})],
        clock=clock, notify=lambda key, text: messages.append(text) or 'DELIVERED', reports=tmp_path / 'reports')
    assert result['state'] == expected
    assert result['steps'][0]['state'] == ('DEGRADED' if expected == 'GREEN' else 'BLOCKED')
    assert result['alerts'][0]['delivery'] == 'DELIVERED'
    if expected == 'GREEN':
        assert result['context_state'] == 'DEGRADED' and result['retry_at'] is None
        assert 'PARTIAL_CONTEXT' in messages[0]
        assert 'DEGRADED' in (tmp_path / ('reports/rehearsal-' + clock().date().isoformat() + '.md')).read_text()


def test_gdelt_degradation_cannot_hide_failed_alert_delivery(tmp_path, clock):
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState, rehearse
    def unavailable():
        raise TraderError('SOURCE_HTTP_429')
    result = rehearse(MonitorState(tmp_path / 'monitor'), [('gdelt', unavailable)],
        clock=clock, notify=lambda *a: 'FAILED', reports=tmp_path / 'reports')
    assert result['state'] == 'BLOCKED'
    assert result['steps'][-1]['error'] == 'TELEGRAM_DELIVERY_FAILED'


@pytest.mark.parametrize('payload,state,code', [({}, 'GREEN', 'NEWS_EMPTY_RESPONSE'),
    ({'articles': None}, 'BLOCKED', 'NEWS_SHAPE_INVALID'),
    ({'error': 'Synthetic invalid query'}, 'BLOCKED', 'NEWS_SHAPE_INVALID')])
def test_gdelt_native_empty_response_degrades_but_unknown_shapes_block(live_checks, monkeypatch, payload, state, code):
    from urllib.request import OpenerDirector
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, monitor, clock, _, _ = live_checks
    checks.configuration()
    original = OpenerDirector.open
    def native_shape(self, request, **kwargs):
        response = original(self, request, **kwargs)
        if 'api.gdeltproject.org' in request.full_url:
            response.body = json.dumps(payload).encode()
        return response
    monkeypatch.setattr(OpenerDirector, 'open', native_shape)
    result = rehearse(monitor, [('gdelt', checks.gdelt)], clock=clock,
                      notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['state'] == state and result['steps'][0]['error'] == code
    assert result['alerts'][0]['delivery'] == 'DELIVERED'


def test_retry_reuses_successful_probes_and_never_spends_trader_model_calls(ledger, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts.trading_lab.trader_agent.rehearsal import LiveChecks, MonitorState
    from tests.trader_agent.test_gpt_protocol import cli
    state = MonitorState(tmp_path / 'monitor')
    checks = LiveChecks(SimpleNamespace(), state, clock=ledger.clock)
    checks.ready, checks.grant, checks.ledger = True, ledger.grant, ledger
    calls = cli(monkeypatch, json.dumps({'subtype': 'success', 'structured_output': {'regime': ['PREFLIGHT_OK'], 'views': []}}))
    assert checks.model('analyst_claude')['model']['model'] == 'opus'
    assert checks.model('analyst_claude', retry=True)['cached']
    assert len(calls) == 1 and ledger.counts() == {'rehearsal_analyst_claude': 1}
    ledger.reserve('rehearsal_analyst_claude')
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        ledger.reserve('rehearsal_analyst_claude')
    assert calls[0]['timeout'] <= 120


def test_gpt_failed_preflight_has_exactly_one_separate_retry(ledger, monkeypatch):
    from scripts.trading_lab.trader_agent.preflight import preflight
    from scripts.trading_lab.trader_agent.runners import ModelRunner
    from tests.trader_agent.test_gpt_protocol import cli, stream, WEB_COMPLETED
    cli(monkeypatch, 'bad-json')
    assert preflight(ledger, ModelRunner(ledger, clock=ledger.clock))['state'] == 'BLOCKED'
    calls = cli(monkeypatch, stream({'regime': ['PREFLIGHT_OK'], 'views': []}, WEB_COMPLETED))
    assert preflight(ledger, ModelRunner(ledger, clock=ledger.clock), retry=True)['state'] == 'GREEN'
    assert preflight(ledger, ModelRunner(ledger, clock=ledger.clock), retry=True)['state'] == 'ALREADY_ATTEMPTED'
    assert len(calls) == 1 and ledger.counts() == {'gpt_preflight': 2}


def test_exception_bodies_and_telegram_tokens_never_enter_report(tmp_path, clock):
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState, rehearse
    def secret_failure():
        raise OSError('https://secret.invalid/botTOKEN/private-account-body')
    result = rehearse(MonitorState(tmp_path / 'monitor'), [('executor', secret_failure)], clock=clock,
                       notify=lambda *args: (_ for _ in ()).throw(OSError('TOKEN')), reports=tmp_path / 'reports')
    text = json.dumps(result) + (tmp_path / 'reports/rehearsal-2026-10-06.md').read_text()
    assert 'TOKEN' not in text and 'private-account-body' not in text
    assert result['steps'][0]['error'] == 'UNEXPECTED_OSERROR'
    assert result['alerts'][0]['delivery'] == 'FAILED'


def test_audit_alerts_expiry_missing_labels_and_overdue_lots_without_submissions(environment):
    from scripts.trading_lab.trader_agent.post_audit import inspect_audit
    from scripts.trading_lab.trader_agent.rehearsal import SnapshotPaperLedger, readonly_executor, get_only_client
    from tests.trader_agent.test_alpaca_paper import issue
    executor, broker, grant, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:30:30Z')
    executor.run('execute', accounts=['ia_actions'])
    key = broker.posts()[0][3]['client_order_id']
    broker.fill('ia_actions', key)
    original_events = executor.ledger.events()
    snapshot = SnapshotPaperLedger(executor.ledger.root, grant, clock)
    def client_factory(grant, name, *, clock):
        return get_only_client(grant, name, clock=clock, transport=broker.transport(name))
    reader = readonly_executor(snapshot, executor.research, clock=clock, client_factory=client_factory)
    clock.at = instant('2026-10-12T21:40:00Z')
    with snapshot.owner():
        result = inspect_audit(reader, label_at=clock())
    assert result['state'] == 'BLOCKED'
    assert {'PAPER_LOT_PAST_HORIZON', 'LABELS_MISSING'} <= set(result['errors'])
    assert executor.ledger.events() == original_events and len(broker.posts()) == 1
    broker.orders[('ia_actions', key)]['status'] = 'expired'
    with snapshot.owner():
        result = inspect_audit(reader)
    assert 'PAPER_ORDER_EXPIRED' in result['errors']


# Existing execution fixtures are fully synthetic; importing does not call the network.
from tests.trader_agent.test_execution_fill_fix import environment


@pytest.mark.parametrize('day,entry,exit_at', [
    ('2026-10-12', '13:40:30', '20:05:00'),
    ('2026-11-02', '14:40:30', '21:05:00'),
    ('2026-11-27', '14:40:30', '18:05:00'),
])
def test_audit_entry_checkpoint_follows_dst_and_half_days(grant, service, day, entry, exit_at):
    from scripts.trading_lab.trader_agent.post_audit import checkpoints
    due = checkpoints(grant, service.store, [], instant(day + 'T23:00:00Z'))
    assert due['session-entry:' + day] == instant(day + 'T' + entry + 'Z')


def test_weekend_holiday_populations_follow_approved_crypto_calendar(grant):
    from dataclasses import replace
    from scripts.trading_lab.trader_agent.rehearsal import population
    assert population(grant, instant('2026-10-12T10:45:00Z')) == 'weekday'
    assert population(grant, instant('2026-10-17T10:45:00Z')) == 'closed'
    assert population(grant, instant('2026-12-25T10:45:00Z')) == 'closed'


def test_rendered_early_and_final_rehearsals_and_ten_minute_audits(tmp_path):
    from scripts.trading_lab.trader_agent.rehearsal_schedule import render
    units = render(tmp_path, '/python', '/grant', tmp_path / 'runtime', '/paper', '/data', path='/bin')
    assert len(units) == 9
    assert '*-*-* 07:30:00 UTC' in units['hyprl-trader-rehearsal-early.timer']
    assert '"--early"' in units['hyprl-trader-rehearsal-early.service']
    assert '*-*-* 10:45:00 UTC' in units['hyprl-trader-rehearsal.timer']
    assert '*-*-* 11:15:00 UTC' in units['hyprl-trader-rehearsal-retry.timer']
    assert 'OnFailure=hyprl-trader-rehearsal-failure.service' in units['hyprl-trader-rehearsal.service']
    assert '*-*-* *:01/10:30 UTC' in units['hyprl-trader-post-audit.timer']
    assert 'Persistent=false' in units['hyprl-trader-rehearsal.timer']
    assert 'paper-execute' not in ''.join(units.values())


def test_early_failure_does_not_consume_or_replace_the_final_gate(tmp_path, clock):
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState, rehearse
    state = MonitorState(tmp_path / 'monitor')
    clock.at = instant('2026-10-14T07:30:00Z')
    def unavailable():
        raise TraderError('SOURCE_UNAVAILABLE')
    early = rehearse(state, [('yahoo', unavailable)], clock=clock, early=True,
                     notify=lambda *a: 'DELIVERED', reports=tmp_path / 'reports')
    assert early['state'] == 'BLOCKED' and early['slot'] == 'early' and early['retry_at'] is None
    assert rehearse(state, [], clock=clock, early=True, reports=tmp_path / 'reports')['state'] == 'ALREADY_ATTEMPTED'
    clock.at = instant('2026-10-14T10:45:00Z')
    final = rehearse(state, [('yahoo', lambda: {'fresh': True})], clock=clock,
                     reports=tmp_path / 'reports')
    assert final['state'] == 'GREEN' and final['slot'] == 'primary'
    report = (tmp_path / 'reports/rehearsal-2026-10-14.md').read_text()
    assert 'early: BLOCKED' in report and 'primary: GREEN' in report
    assert report.startswith('STATUS: DONE')


def test_rendered_model_units_resolve_both_clis_without_a_login_shell(tmp_path):
    import os
    import shlex
    import subprocess
    from scripts.trading_lab.trader_agent.schedule import render
    from scripts.trading_lab.trader_agent.rehearsal_schedule import render as monitoring
    cli_dir = tmp_path / 'model-bin'
    cli_dir.mkdir()
    for name in ('claude', 'codex'):
        executable = cli_dir / name
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o700)
    transient = str(tmp_path / '.codex/tmp/arg0/expired')
    path = transient + os.pathsep + str(cli_dir) + ':/usr/bin:/bin'
    units = {**render(tmp_path, '/python', '/grant', tmp_path, path=path, include_pm=True),
             **monitoring(tmp_path, '/python', '/grant', tmp_path, '/paper', '/data', path=path, include_pm=True)}
    for name, body in units.items():
        if not name.endswith('.service'):
            continue
        setting = next(line.removeprefix('Environment=') for line in body.splitlines()
                       if line.startswith('Environment=') and 'PATH=' in line)
        unit_path = shlex.split(setting)[0].removeprefix('PATH=')
        assert transient not in unit_path
        for cli in ('claude', 'codex'):
            subprocess.run([cli], env={'PATH': unit_path}, check=True, timeout=5)


@pytest.mark.parametrize('method', ['POST', 'DELETE', 'PATCH', 'PUT'])
def test_readonly_broker_boundary_refuses_all_mutations_and_claude_account(environment, method):
    from scripts.trading_lab.trader_agent.rehearsal import get_only_client
    executor, broker, grant, clock = environment
    client = get_only_client(grant, 'ia_actions', clock=clock, transport=broker.transport('ia_actions'))
    client.start()
    before = len(broker.calls)
    with pytest.raises(TraderError, match='REHEARSAL_GET_ONLY'):
        client.request(method, '/v2/orders')
    with pytest.raises(TraderError, match='REHEARSAL_AI_ACCOUNTS_ONLY'):
        get_only_client(grant, 'momentum', clock=clock)
    assert len(broker.calls) == before


def test_reviewer_rehearsal_uses_the_real_verdict_schema(ledger, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts.trading_lab.trader_agent.rehearsal import LiveChecks, MonitorState
    from tests.trader_agent.test_gpt_protocol import cli
    checks = LiveChecks(SimpleNamespace(), MonitorState(tmp_path / 'monitor'), clock=ledger.clock)
    checks.ready, checks.grant, checks.ledger = True, ledger.grant, ledger
    calls = cli(monkeypatch, json.dumps({'subtype': 'success', 'structured_output': {'verdicts': []}}))
    assert checks.model('reviewer')['model']['model'] == 'sonnet'
    assert ledger.counts() == {'rehearsal_reviewer': 1}
    assert 'verdicts' in calls[0]['prompt'] and 'market data' in calls[0]['prompt']


@pytest.fixture
def live_checks(environment, tmp_path, monkeypatch):
    """Real coordinator/providers/validators/stores, synthetic HTTP and inference processes."""
    from types import SimpleNamespace
    from urllib.parse import urlparse, parse_qs, unquote
    from urllib.request import OpenerDirector
    from scripts.radar.core import Store
    from scripts.radar.sources import item
    from scripts.trading_lab.research.store import ResearchStore
    from scripts.trading_lab.trader_agent.config import iso
    from scripts.trading_lab.trader_agent.data import equity_calendar
    from scripts.trading_lab.trader_agent.rehearsal import LiveChecks, MonitorState
    from tests.trader_agent.test_gpt_protocol import stream, WEB_COMPLETED
    executor, broker, paper_grant, clock = environment
    clock.at = instant('2026-10-12T10:45:00Z')
    with executor.ledger.owner():
        pass
    runtime = tmp_path / 'runtime'
    ResearchStore(runtime / 'evidence').close()
    data = tmp_path / 'data-grant.json'
    data.write_text(json.dumps({'authorization': 'trader-alpaca-data-v1', 'extends': 'trader-alpaca-paper-v1',
        'granted_at': '2026-01-01T00:00:00Z', 'not_after': '2027-01-01T00:00:00Z',
        'data_endpoint': {'origin': 'https://data.alpaca.markets', 'methods': ['GET'], 'credentials': 'ia_actions keys'}}))
    radar = Store(tmp_path / 'radar', clock=clock)
    with radar.owner():
        pass
    stamp = clock() - timedelta(minutes=5)
    record = item('rss:synthetic', 'example.invalid', 'https://example.invalid/synthetic',
                  'Synthetic market update', 'Synthetic summary', iso(stamp), stamp, symbols=['AAPL'])
    radar.append('news', record['id'], record, at=stamp)
    veille = tmp_path / 'veille.md'
    veille.write_text('2026-10-11T00:00:00Z\nSynthetic context\n')
    import os
    os.utime(veille, (stamp.timestamp(), stamp.timestamp()))
    reports = tmp_path / 'reports'
    (reports / 'radar').mkdir(parents=True)
    (reports / 'radar/report.json').write_text(json.dumps({'schema': 'news-radar-report-v1', 'events': [],
                                                       'status': 'COMPLETE', 'cutoff': iso(stamp)}))
    args = SimpleNamespace(authorization=str(tmp_path / 'synthetic-grant.json'),
        paper_authorization=str(tmp_path / 'paper-grant.json'), data_authorization=str(data),
        runtime=str(runtime), budget_root=str(tmp_path / 'budget'), paper_runtime=str(executor.ledger.root),
        radar_root=str(radar.root), veille=str(veille), fomc_store=None, edgar_store=None, reports=str(reports))
    state = MonitorState(tmp_path / 'monitor')
    checks = LiveChecks(args, state, clock=clock)
    calls = []
    faults = {}
    monkeypatch.setattr('scripts.trading_lab.trader_agent.rehearsal.protected', lambda *args: None)
    monkeypatch.setattr('scripts.trading_lab.trader_agent.alpaca_data.protected', lambda *args: None)
    for name in ('ia_actions', 'ia_crypto'):
        from pathlib import Path
        credentials = Path(paper_grant.payload['accounts'][name]['credentials'])
        credentials.write_text(credentials.read_text().replace('synthetic-key', 'synthetic-' + name))
    class Response:
        def __init__(self, body):
            self.body = json.dumps(body).encode()
        def read(self, size):
            return self.body[:size]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def http(self, request, **kwargs):
        parsed = urlparse(request.full_url)
        host, path, query = parsed.hostname, unquote(parsed.path), parse_qs(parsed.query)
        calls.append((host, request.get_method(), path))
        if host in faults:
            raise OSError('Synthetic provider unavailable')
        assert request.get_method() == 'GET'
        if host == 'query1.finance.yahoo.com':
            asset = path.split('/')[-1]
            sessions = equity_calendar(2026).sessions_between(clock() - timedelta(days=70), clock())
            return Response({'chart': {'error': None, 'result': [{'meta': {'symbol': asset, 'currency': 'USD',
                'exchangeTimezoneName': 'America/New_York', 'exchangeName': 'NMS',
                'instrumentType': 'ETF' if asset in ['XLK', 'SPY'] else 'EQUITY'},
                'timestamp': [int(s.open_at.timestamp()) for s in sessions], 'indicators': {'quote': [{
                    'open': [100] * len(sessions), 'close': [101] * len(sessions),
                    'high': [102] * len(sessions), 'low': [99] * len(sessions), 'volume': [1000] * len(sessions)}]}}]}})
        if host == 'api.exchange.coinbase.com':
            start = instant(query['start'][0])
            end = instant(query['end'][0])
            days = int((end - start).total_seconds() // 86400)
            return Response([[int((start + timedelta(days=i)).timestamp()), 99, 102, 100, 101, 1000] for i in range(days)])
        if host == 'api.gdeltproject.org':
            return Response({'articles': [{'url': 'https://example.invalid/news', 'title': 'Synthetic fresh news',
                                          'seendate': stamp.strftime('%Y%m%dT%H%M%SZ'), 'domain': 'example.invalid'}]})
        if host == 'data.alpaca.markets':
            symbols = query['symbols'][0].split(',')
            crypto = 'crypto' in path
            quote_at = stamp if crypto else instant('2026-10-09T19:59:59Z')
            if crypto:
                quote_at = clock() - timedelta(seconds=1)
            if 'quotes' in path:
                return Response({'quotes': {s: {'bp': 100, 'ap': 100.01, 't': iso(quote_at)} for s in symbols}})
            return Response({'trades': {s: {'p': 100, 't': iso(quote_at)} for s in symbols}})
        if host == 'paper-api.alpaca.markets':
            name = request.get_header('Apca-api-key-id').removeprefix('synthetic-')
            if path == '/v2/account':
                account = dict(broker.accounts[name], buying_power='100000')
                return Response(account)
            if path == '/v2/clock':
                return Response({'timestamp': iso(clock()), 'is_open': False,
                                 'next_open': '2026-10-12T13:30:00Z', 'next_close': '2026-10-12T20:00:00Z'})
            if path == '/v2/calendar':
                return Response([{'date': '2026-10-12', 'open': '09:30', 'close': '16:00'}])
            if path.startswith('/v2/assets/'):
                return Response({'class': 'crypto' if '/USD' in path else 'us_equity', 'status': 'active',
                                 'tradable': True, 'shortable': '/USD' not in path})
            if path in ('/v2/positions', '/v2/orders'):
                return Response([])
        raise AssertionError('Unexpected synthetic URL: ' + host + path)
    monkeypatch.setattr(OpenerDirector, 'open', http)
    monkeypatch.setattr('subprocess.run', lambda *args, **kw: SimpleNamespace(stdout='synthetic-cli'))
    class Process:
        returncode = 0
        def __init__(self, args, **kw):
            self.args, self.out = args, kw['stdout']
        def communicate(self, prompt, **kw):
            reviewer = 'sonnet' in self.args
            output = {'verdicts': []} if reviewer else {'regime': ['PREFLIGHT_OK'], 'views': []}
            if self.args[0] == 'codex':
                self.out.write(stream(output, WEB_COMPLETED))
            else:
                self.out.write(json.dumps({'subtype': 'success', 'structured_output': output}))
            self.out.flush()
    monkeypatch.setattr('subprocess.Popen', Process)
    return checks, state, clock, calls, faults


def test_full_chain_uses_real_configuration_with_only_gets_and_zero_decisions(live_checks):
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, calls, _ = live_checks
    result = rehearse(state, checks.steps(), clock=clock, notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['state'] == 'GREEN', [s for s in result['steps'] if s['state'] != 'GREEN']
    assert len(result['steps']) == 14
    assert all(method == 'GET' for _, method, _ in calls)
    assert checks.research.records('prediction') == []
    assert checks.ledger.counts() == {'yahoo_chart': 3, 'coinbase_exchange_public': 2, 'gdelt_doc_api': 1,
                                      'gpt_preflight': 1, 'rehearsal_analyst_claude': 1, 'rehearsal_reviewer': 1}
    assert (state.root / 'cockpit-probe/analysis.json').exists()


@pytest.mark.parametrize('status', [200, 429])
def test_recent_radar_gdelt_receipt_is_reused_without_a_live_request(live_checks, status):
    from scripts.radar.core import Store
    from scripts.trading_lab.trader_agent.config import iso
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, calls, _ = live_checks
    at = clock() - timedelta(minutes=30)
    radar = Store(checks.args.radar_root, clock=clock)
    radar.append('http', 'gdelt:macro', {'url': 'https://api.gdeltproject.org/api/v2/doc/doc',
        'status': status, 'received_at': iso(at), 'body_hash': 'a' * 64}, at=at)
    radar.append('health', 'gdelt:macro', {'source': 'gdelt:macro', 'status': 'LIVE' if status == 200 else 'BLOCKED',
        'items': 3, 'checked_at': iso(at)}, at=at)
    checks.configuration()
    result = rehearse(state, [('gdelt', checks.gdelt)], clock=clock, reports=checks.args.reports,
                      notify=lambda *a: 'DELIVERED')
    assert result['state'] == 'GREEN'
    step = result['steps'][0]
    assert step['state'] == ('GREEN' if status == 200 else 'DEGRADED')
    assert step['evidence']['reused_radar_receipt'] is True
    assert not any(host == 'api.gdeltproject.org' for host, _, _ in calls)


def test_radar_gdelt_backoff_outlives_receipt_freshness_and_does_not_spend_budget(live_checks):
    from scripts.radar.core import Store
    from scripts.trading_lab.trader_agent.config import iso
    checks, _, clock, calls, _ = live_checks
    radar = Store(checks.args.radar_root, clock=clock)
    radar.append('cooldown', 'gdelt', {'until': iso(clock() + timedelta(hours=8)), 'status': 429, 'failures': 4},
                 at=clock() - timedelta(hours=8))
    checks.configuration()
    with pytest.raises(TraderError, match='SOURCE_COOLDOWN') as error:
        checks.gdelt()
    assert error.value.evidence['failures'] == 4
    assert not any(host == 'api.gdeltproject.org' for host, _, _ in calls)
    assert 'gdelt_doc_api' not in checks.ledger.counts()


def test_future_radar_gdelt_evidence_cannot_suppress_the_live_probe(live_checks):
    from scripts.radar.core import Store
    from scripts.trading_lab.trader_agent.config import iso
    checks, _, clock, calls, _ = live_checks
    radar = Store(checks.args.radar_root, clock=clock)
    future = clock() + timedelta(minutes=1)
    radar.append('http', 'gdelt:macro', {'url': 'https://api.gdeltproject.org/api/v2/doc/doc',
        'status': 200, 'received_at': iso(future), 'body_hash': 'a' * 64}, at=future)
    radar.append('cooldown', 'gdelt', {'until': iso(future + timedelta(hours=1))}, at=future)
    checks.configuration()
    assert checks.gdelt()['articles'] == 1
    assert sum(host == 'api.gdeltproject.org' for host, _, _ in calls) == 1


@pytest.mark.parametrize('retry_after', ['10800', 'Mon, 12 Oct 2026 14:00:00 GMT'])
def test_gdelt_429_headers_persist_backoff_across_restart(live_checks, monkeypatch, retry_after):
    from urllib.error import HTTPError
    from urllib.request import OpenerDirector
    from scripts.trading_lab.trader_agent.rehearsal import LiveChecks
    checks, state, clock, _, _ = live_checks
    checks.configuration()
    requests = []
    def limited(self, request, **kw):
        requests.append(request.full_url)
        raise HTTPError(request.full_url, 429, 'synthetic', {'Retry-After': retry_after}, None)
    monkeypatch.setattr(OpenerDirector, 'open', limited)
    with pytest.raises(TraderError, match='SOURCE_HTTP_429'):
        checks.gdelt()
    restarted = LiveChecks(checks.args, state, clock=clock)
    restarted.configuration()
    with pytest.raises(TraderError, match='SOURCE_COOLDOWN') as error:
        restarted.gdelt()
    assert instant(error.value.evidence['retry_at']) >= instant('2026-10-12T13:45:00Z')
    assert len(requests) == 1


@pytest.mark.parametrize('host,step', [('query1.finance.yahoo.com', 'yahoo'),
    ('api.exchange.coinbase.com', 'coinbase'), ('api.gdeltproject.org', 'gdelt'),
    ('data.alpaca.markets', 'alpaca_data'), ('paper-api.alpaca.markets', 'executor')])
def test_each_source_failure_is_attributed_and_other_steps_still_run(live_checks, host, step):
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, _, faults = live_checks
    faults[host] = True
    result = rehearse(state, checks.steps(), clock=clock, notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['state'] == ('GREEN' if step == 'gdelt' else 'BLOCKED')
    assert next(s for s in result['steps'] if s['step'] == step)['state'] == ('DEGRADED' if step == 'gdelt' else 'BLOCKED')
    assert result['steps'][-1]['step'] == 'cockpit_export'
    assert result['retry_at'] == (None if step == 'gdelt' else '2026-10-12T11:15:00Z')


def test_retry_fresh_prices_and_successful_models_do_not_refund_or_repeat_calls(live_checks):
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, calls, faults = live_checks
    faults['data.alpaca.markets'] = True
    assert rehearse(state, checks.steps(), clock=clock, notify=lambda *a: 'DELIVERED', reports=checks.args.reports)['state'] == 'BLOCKED'
    faults.clear()
    first = dict(checks.ledger.counts())
    clock.at += timedelta(minutes=30)
    result = rehearse(state, checks.steps(retry=True), clock=clock, retry=True,
                       notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['state'] == 'GREEN', [s for s in result['steps'] if s['state'] != 'GREEN']
    second = checks.ledger.counts()
    assert second == {**first, 'gdelt_doc_api': 2}
    assert next(s for s in result['steps'] if s['step'] == 'yahoo')['evidence']['cache_hits'] == 3


@pytest.mark.parametrize('which', ['configuration', 'radar_veille', 'archives', 'scorer_labels', 'cockpit_export'])
def test_non_network_chain_failures_reach_report_and_alert(live_checks, which):
    from pathlib import Path
    import fcntl
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, _, _ = live_checks
    lock = None
    if which == 'configuration':
        grant = json.loads(Path(checks.args.authorization).read_text())
        grant['not_after'] = '2026-10-11T00:00:00Z'
        Path(checks.args.authorization).write_text(json.dumps(grant))
    elif which == 'radar_veille':
        checks.args.radar_root = str(state.root / 'missing-radar')
    elif which == 'archives':
        checks.args.fomc_store = str(state.root / 'missing-archive')
    elif which == 'scorer_labels':
        # Invalid synthetic prediction evidence reaches the actual scorer, not a mocked internal method.
        # The append-only API rejects unsafe inputs; use an invalid store schema instead.
        import sqlite3
        path = Path(checks.args.runtime) / 'evidence/research.sqlite'
        with sqlite3.connect(path) as db:
            db.execute("UPDATE metadata SET schema='synthetic-corrupt-schema'")
        which = 'configuration'  # integrity admission fails before scoring can read the store
    else:
        (Path(checks.args.reports) / 'radar/report.json').write_text('bad-json')
    try:
        result = rehearse(state, checks.steps(), clock=clock, notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    finally:
        if lock:
            lock.close()
    assert result['state'] == 'BLOCKED'
    assert any(s['step'] == which and s['state'] == 'BLOCKED' for s in result['steps'])
    assert any(a['step'] == which and a['delivery'] == 'DELIVERED' for a in result['alerts'])


def test_audit_missing_eligible_entry_is_not_a_successful_empty_ledger(environment):
    from scripts.trading_lab.trader_agent.rehearsal import SnapshotPaperLedger, readonly_executor, get_only_client
    from scripts.trading_lab.trader_agent.post_audit import inspect_audit
    from tests.trader_agent.test_alpaca_paper import issue
    executor, broker, grant, clock = environment
    issue(executor, horizons=('1d',))
    with executor.ledger.owner():
        pass
    clock.at += timedelta(minutes=10)
    snapshot = SnapshotPaperLedger(executor.ledger.root, grant, clock)
    def factory(grant, name, *, clock):
        return get_only_client(grant, name, clock=clock, transport=broker.transport(name))
    with snapshot.owner():
        result = inspect_audit(readonly_executor(snapshot, executor.research, clock=clock, client_factory=factory))
    assert result['state'] == 'BLOCKED' and 'PAPER_ELIGIBLE_ENTRY_NOT_DISPATCHED' in result['errors']
    assert broker.posts() == []


def test_audit_does_not_hold_execution_lock_during_broker_reads(environment):
    from scripts.trading_lab.trader_agent.rehearsal import SnapshotPaperLedger
    executor, _, grant, clock = environment
    with executor.ledger.owner():
        pass
    snapshot = SnapshotPaperLedger(executor.ledger.root, grant, clock)
    with snapshot.owner():
        with executor.ledger.owner():
            executor.ledger.append('alert', 'ia_actions', code='SYNTHETIC_CHANGE_DURING_HTTP')
        assert not snapshot.unchanged()


def test_audit_busy_writer_defers_quietly_without_marking_checkpoint_done(live_checks):
    import fcntl
    from pathlib import Path
    from scripts.trading_lab.trader_agent.post_audit import audit
    checks, state, clock, calls, _ = live_checks
    clock.at = instant('2026-10-12T21:40:00Z')
    with (Path(checks.args.paper_runtime) / 'paper-owner.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = len(calls)
        result = audit(checks.args, state, clock=clock, force=True,
                       notify=lambda *a: pytest.fail('busy audit alerted'))
    assert result['state'] == 'PAPER_EXECUTION_IN_PROGRESS' and len(calls) == before
    with state.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0] == 0


@pytest.mark.parametrize('action', ['paper-audit', 'paper-execute', 'paper-exit'])
def test_cli_dry_run_paper_checks_preserve_journal_and_release_owner_for_http(live_checks, capsys, monkeypatch, action):
    import fcntl
    from pathlib import Path
    from urllib.request import OpenerDirector
    from scripts.trading_lab.trader_agent.cli import main
    from scripts.trading_lab.research.store import ResearchStore
    checks, _, _, calls, _ = live_checks
    shared = Path.home() / '.local/share/hyprl/trader-alpaca-paper'
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.symlink_to(checks.args.paper_runtime, target_is_directory=True)
    original = OpenerDirector.open
    def writer_can_acquire_during_http(self, request, **kwargs):
        with (shared / 'paper-owner.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return original(self, request, **kwargs)
    monkeypatch.setattr(OpenerDirector, 'open', writer_can_acquire_during_http)
    paper = ResearchStore(shared / 'paper-evidence', read_only=True)
    try:
        before = paper.records('replay-summary')
    finally:
        paper.close()
    assert main([action, '--authorization', checks.args.authorization,
                 '--paper-authorization', checks.args.paper_authorization,
                 '--runtime', checks.args.runtime, '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    paper = ResearchStore(shared / 'paper-evidence', read_only=True)
    try:
        assert paper.records('replay-summary') == before
    finally:
        paper.close()
    assert result['production_writes'] == 0 and result['broker_methods'] == ['GET']


def test_cli_dry_run_audit_skips_a_busy_writer_without_http(live_checks, capsys):
    import fcntl
    from pathlib import Path
    from scripts.trading_lab.trader_agent.cli import main
    checks, _, _, calls, _ = live_checks
    shared = Path.home() / '.local/share/hyprl/trader-alpaca-paper'
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.symlink_to(checks.args.paper_runtime, target_is_directory=True)
    before = len(calls)
    with (shared / 'paper-owner.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert main(['paper-audit', '--authorization', checks.args.authorization,
                     '--paper-authorization', checks.args.paper_authorization,
                     '--runtime', checks.args.runtime, '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['state'] == 'PAPER_EXECUTION_IN_PROGRESS' and result['production_writes'] == 0
    assert len(calls) == before


def test_scorer_failure_after_admission_is_reported(live_checks):
    from pathlib import Path
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, _, _ = live_checks
    checks.configuration()
    path = Path(checks.args.runtime) / 'evidence/research.sqlite'
    path.rename(path.with_suffix('.unavailable'))
    result = rehearse(state, [('scorer_labels', checks.scoring)], clock=clock,
                       notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['state'] == 'BLOCKED' and result['alerts'][0]['step'] == 'scorer_labels'


def test_stale_quotes_fail_while_older_positive_trades_match_production_contract(live_checks, monkeypatch):
    from urllib.request import OpenerDirector
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, _, _ = live_checks
    checks.configuration()
    original = OpenerDirector.open
    def old_trades(self, request, **kw):
        result = original(self, request, **kw)
        if 'trades' in request.full_url:
            body = json.loads(result.body)
            for row in body.get('trades', {}).values():
                row['t'] = '2026-10-09T19:59:00Z'
            result.body = json.dumps(body).encode()
        return result
    monkeypatch.setattr(OpenerDirector, 'open', old_trades)
    assert checks.alpaca_data()['trades_older_than_60s'] == 5
    def stale_quotes(self, request, **kw):
        result = original(self, request, **kw)
        if '/crypto/' in request.full_url and 'quotes' in request.full_url:
            body = json.loads(result.body)
            for row in body['quotes'].values():
                row['t'] = '2026-10-09T19:59:00Z'
            result.body = json.dumps(body).encode()
        return result
    monkeypatch.setattr(OpenerDirector, 'open', stale_quotes)
    result = rehearse(state, [('alpaca_data', checks.alpaca_data)], clock=clock,
                       notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['steps'][0]['error'] == 'REHEARSAL_DATA_STALE'


@pytest.mark.parametrize('before_open', [True, False])
def test_iex_closed_book_quotes_do_not_gate_next_regular_day_order_but_do_gate_open_session(live_checks, monkeypatch, before_open):
    from urllib.request import OpenerDirector
    checks, _, clock, _, _ = live_checks
    checks.configuration()
    if not before_open:
        clock.at = instant('2026-10-12T15:15:00Z')
    original = OpenerDirector.open
    def closing_book(self, request, **kw):
        response = original(self, request, **kw)
        if 'data.alpaca.markets/v2/stocks/' in request.full_url:
            body = json.loads(response.body)
            for row in body.get('quotes', {}).values():
                row.update(ap=0, bs=100, **{'as': 0})
                row['t'] = '2026-10-09T20:00:03Z' if before_open else '2026-10-12T15:14:59Z'
            for row in body.get('trades', {}).values():
                row['t'] = '2026-10-09T19:59:59Z' if before_open else '2026-10-12T15:14:59Z'
            response.body = json.dumps(body).encode()
        return response
    monkeypatch.setattr(OpenerDirector, 'open', closing_book)
    if before_open:
        evidence = checks.alpaca_data()
        assert evidence['equity_quote_use'] == 'INDICATIVE_CLOSED_BOOK'
        assert evidence['non_executable_equity_quotes'] == 3
    else:
        with pytest.raises(TraderError, match='REHEARSAL_DATA_PRICE_INVALID'):
            checks.alpaca_data()


@pytest.mark.parametrize('price,timestamp,code,field', [
    (0, '2026-10-09T19:59:59Z', 'REHEARSAL_DATA_PRICE_INVALID', 'trade.p'),
    (100, '2026-10-08T19:59:59Z', 'REHEARSAL_DATA_STALE', 'trade.t'),
    (100, '2026-10-12T10:46:00Z', 'DATA_TRADE_INVALID', 'trade.t')])
def test_closed_session_trade_failures_still_block_with_asset_and_field(live_checks, monkeypatch, price, timestamp, code, field):
    from urllib.request import OpenerDirector
    checks, _, _, _, _ = live_checks
    checks.configuration()
    original = OpenerDirector.open
    def invalid(self, request, **kw):
        response = original(self, request, **kw)
        if 'data.alpaca.markets/v2/stocks/trades/' in request.full_url:
            body = json.loads(response.body)
            body['trades']['AAPL'].update(p=price, t=timestamp)
            response.body = json.dumps(body).encode()
        return response
    monkeypatch.setattr(OpenerDirector, 'open', invalid)
    with pytest.raises(TraderError, match=code) as error:
        checks.alpaca_data()
    assert error.value.evidence == {'failures': [{'asset': 'AAPL', 'field': field, 'code': code}]}


def test_approved_crypto_rehearsal_population_and_holiday_checkpoint(weekend_grant, service):
    from scripts.trading_lab.trader_agent.rehearsal import population
    from scripts.trading_lab.trader_agent.post_audit import checkpoints
    for day in ('2026-10-17', '2026-12-25'):
        at = instant(day + 'T23:00:00Z')
        assert population(weekend_grant, at) == 'weekend_crypto'
        assert checkpoints(weekend_grant, service.store, [], at)['session-entry:' + day] == instant(day + 'T13:40:00Z')


from tests.trader_agent.test_weekend_crypto import weekend_grant
from tests.trader_agent.test_execution_v2 import amended


def test_new_checkpoints_do_not_repeat_known_order_alerts_but_new_orders_do(tmp_path):
    from scripts.trading_lab.trader_agent.post_audit import alert_issues
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    state = MonitorState(tmp_path / 'monitor')
    messages = []
    send = lambda key, text: messages.append(text) or 'DELIVERED'
    old = {'code': 'PAPER_ORDER_EXPIRED', 'identity': 'a' * 64}
    new = {'code': 'PAPER_ORDER_EXPIRED', 'identity': 'b' * 64}
    assert alert_issues(state, [old], send)['delivery'] == 'DELIVERED'
    assert alert_issues(state, [old], send)['delivery'] == 'ALREADY_NOTIFIED'
    assert alert_issues(state, [old, new], send)['new_issues'] == 1
    assert len(messages) == 2
    labels = [{'code': 'LABELS_MISSING', 'identity': str(i)} for i in range(200)]
    assert alert_issues(state, labels, send)['new_issues'] == 200
    assert len(messages) == 3


def test_acknowledged_history_stays_visible_but_a_new_issue_blocks(tmp_path, clock):
    from scripts.trading_lab.trader_agent.post_audit import acknowledge_issues, apply_acknowledgements, alert_issues
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    state = MonitorState(tmp_path / 'monitor')
    old = {'code': 'LABELS_MISSING', 'identity': 'a' * 64}
    new = {'code': 'LABELS_MISSING', 'identity': 'b' * 64}
    acknowledge_issues(state, [old], reason='EXACT_NATIVE_MINUTE_ANCHOR_UNAVAILABLE', at=clock(), unlabelable=True)
    # A restart and a repeated acknowledgement cannot replace the original reason.
    state = MonitorState(state.root)
    acknowledge_issues(state, [old], reason='REPEATED_ACKNOWLEDGEMENT', at=clock())
    result = {'state': 'BLOCKED', 'errors': ['LABELS_MISSING'], 'issues': [old], 'labels_missing': 1}
    apply_acknowledgements(state, result)
    assert result['state'] == 'GREEN' and result['errors'] == []
    assert result['labels_missing'] == 1 and result['labels_unlabelable'] == 1
    assert result['acknowledged_issues'][0]['reason'] == 'EXACT_NATIVE_MINUTE_ANCHOR_UNAVAILABLE'
    assert alert_issues(state, [old], lambda *a: pytest.fail('known issue re-alerted'))['new_issues'] == 0
    result = {'state': 'BLOCKED', 'errors': ['LABELS_MISSING'], 'issues': [old, new]}
    apply_acknowledgements(state, result)
    assert result['state'] == 'BLOCKED' and result['issues'] == [new]
    assert alert_issues(state, result['issues'], lambda *a: 'DELIVERED')['new_issues'] == 1
    with pytest.raises(TraderError, match='AUDIT_ACKNOWLEDGEMENT_REFUSED'):
        acknowledge_issues(state, [{'code': 'PAPER_LOT_PAST_HORIZON', 'identity': 'c' * 64}],
                           reason='STILL_OPEN', at=clock())


def test_not_activated_pm_disposition_persists_once_and_future_missing_run_alerts(tmp_path, clock):
    from scripts.trading_lab.sources.canonical import sha256_canonical
    from scripts.trading_lab.trader_agent.post_audit import acknowledge_issues, apply_acknowledgements, alert_issues
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    state = MonitorState(tmp_path / 'monitor')
    old = {'code': 'MISSING_PM_RUN', 'identity': sha256_canonical(['MISSING_PM_RUN', '2026-10-13'])}
    tomorrow = {'code': 'MISSING_PM_RUN', 'identity': sha256_canonical(['MISSING_PM_RUN', '2026-10-14'])}
    reason = 'ACTIVATION_HELD_UNTIL_GREEN_PREFLIGHT'
    acknowledge_issues(state, [old], reason=reason, at=clock(), not_activated=True, decision_sha256='a' * 64)
    recorded_at = clock().isoformat().replace('+00:00', 'Z')
    state = MonitorState(state.root)
    clock.at += timedelta(minutes=1)
    acknowledge_issues(state, [old], reason='REPEATED_ACKNOWLEDGEMENT', at=clock(),
                       not_activated=True, decision_sha256='b' * 64)
    result = {'state': 'BLOCKED', 'errors': ['MISSING_PM_RUN'], 'issues': [old]}
    apply_acknowledgements(state, result)
    assert result['state'] == 'GREEN' and result['observed_errors'] == ['MISSING_PM_RUN']
    assert result['acknowledged_issues'] == [{**old, 'reason': reason, 'disposition': 'NOT_ACTIVATED',
                                            'at': recorded_at, 'decision_sha256': 'a' * 64}]
    assert alert_issues(state, [old], lambda *a: pytest.fail('disposed PM slot re-alerted'))['new_issues'] == 0
    result = {'state': 'BLOCKED', 'errors': ['MISSING_PM_RUN'], 'issues': [old, tomorrow]}
    apply_acknowledgements(state, result)
    assert result['state'] == 'BLOCKED' and result['issues'] == [tomorrow]
    messages = []
    assert alert_issues(state, result['issues'], lambda *a: messages.append(a) or 'DELIVERED')['new_issues'] == 1
    assert len(messages) == 1


@pytest.mark.parametrize('code,options', [
    ('MISSING_PM_RUN', {}),
    ('MISSING_PM_RUN', {'decision_sha256': 'a' * 64}),
    ('MISSING_PM_RUN', {'not_activated': True}),
    ('MISSING_PM_RUN', {'not_activated': True, 'decision_sha256': 'invalid'}),
    ('MISSING_PM_RUN', {'not_activated': True, 'decision_sha256': 'a' * 64, 'unlabelable': True}),
    ('LABELS_MISSING', {'not_activated': True, 'decision_sha256': 'a' * 64}),
    ('PAPER_LOT_PAST_HORIZON', {'not_activated': True, 'decision_sha256': 'a' * 64}),
])
def test_not_activated_requires_exact_pm_issue_and_decision_hash(tmp_path, clock, code, options):
    from scripts.trading_lab.trader_agent.post_audit import acknowledge_issues, apply_acknowledgements
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    state = MonitorState(tmp_path / 'monitor')
    issue = {'code': code, 'identity': 'a' * 64}
    with pytest.raises(TraderError, match='AUDIT_ACKNOWLEDGEMENT_REFUSED'):
        acknowledge_issues(state, [issue], reason='OPERATOR_DECISION', at=clock(), **options)
    result = {'state': 'BLOCKED', 'errors': [code], 'issues': [issue]}
    apply_acknowledgements(state, result)
    assert result['state'] == 'BLOCKED' and result['acknowledged_issues'] == []


def test_not_activated_cli_records_without_opening_production_runtime(tmp_path, capsys):
    from scripts.trading_lab.trader_agent.post_audit import main, apply_acknowledgements
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    issue = {'code': 'MISSING_PM_RUN', 'identity': 'a' * 64}
    assert main(['--authorization', str(tmp_path / 'absent-grant'),
                 '--paper-authorization', str(tmp_path / 'absent-paper-grant'),
                 '--data-authorization', str(tmp_path / 'absent-data-grant'),
                 '--runtime', str(tmp_path / 'absent-runtime'), '--monitor-runtime', str(tmp_path / 'monitor'),
                 '--acknowledge-issue', 'MISSING_PM_RUN:' + issue['identity'],
                 '--acknowledge-reason', 'ACTIVATION_HELD_UNTIL_GREEN_PREFLIGHT',
                 '--not-activated', '--decision-sha256', 'b' * 64]) == 0
    assert json.loads(capsys.readouterr().out) == {'state': 'NOT_ACTIVATED', 'issues': 1, 'production_writes': 0}
    assert not (tmp_path / 'absent-runtime').exists()
    result = {'state': 'BLOCKED', 'errors': ['MISSING_PM_RUN'], 'issues': [issue]}
    apply_acknowledgements(MonitorState(tmp_path / 'monitor'), result)
    assert result['state'] == 'GREEN' and result['acknowledged_issues'][0]['decision_sha256'] == 'b' * 64


@pytest.mark.parametrize('stamp', ['2026-10-14T11:59:00Z', '2026-10-14T13:30:30Z',
    '2026-10-14T16:00:30Z', '2026-10-14T16:30:30Z', '2026-10-14T19:55:30Z',
    '2026-11-02T14:31:30Z', '2026-11-27T17:55:30Z'])
def test_audit_defers_even_forced_checks_before_touching_execution_or_http(tmp_path, clock, stamp):
    from types import SimpleNamespace
    from scripts.trading_lab.trader_agent.post_audit import audit
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    clock.at = instant(stamp)
    state = MonitorState(tmp_path / 'monitor')
    result = audit(SimpleNamespace(), state, clock=clock, force=True,
                   client_factory=lambda *a, **kw: pytest.fail('broker request in execution window'))
    assert result['state'] == 'PAPER_EXECUTION_WINDOW'
    with state.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0] == 0


def test_newly_appended_backdated_radar_receipts_cannot_prove_freshness(live_checks):
    from scripts.radar.core import Store
    from scripts.radar.sources import item
    from scripts.trading_lab.trader_agent.config import iso
    from scripts.trading_lab.trader_agent.rehearsal import rehearse
    checks, state, clock, _, _ = live_checks
    clock.at += timedelta(hours=3)
    checks.configuration()
    store = Store(checks.args.radar_root, clock=clock)
    old = clock() - timedelta(hours=3)
    record = item('rss:synthetic', 'example.invalid', 'https://example.invalid/backdated',
                  'Synthetic older receipt', 'Synthetic summary', iso(old), old, symbols=['AAPL'])
    store.append('news', record['id'], record, at=clock() - timedelta(seconds=1))
    result = rehearse(state, [('radar_veille', checks.news)], clock=clock,
                       notify=lambda *a: 'DELIVERED', reports=checks.args.reports)
    assert result['steps'][0]['error'] == 'REHEARSAL_RADAR_STALE_OR_EMPTY'
