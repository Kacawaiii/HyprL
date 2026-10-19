from datetime import timedelta
import gzip
import hashlib
import json
from urllib.parse import urlencode

import pytest

from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_seal_forward import archive, inputs, experiment as ex, spec

T = instant('2026-07-15T12:00:00Z')


def add(root, objects, origin, path, params, body, received='2026-10-18T18:00:00Z'):
    url = origin + path + ('?' + urlencode(sorted(params.items())) if params else '')
    key = hashlib.sha256(url.encode()).hexdigest()
    file = root / 'cache' / key[:2] / (key + '.json.gz')
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(gzip.compress(json.dumps({'body': body, 'received_at': received,
                                              'body_sha256': sha256_canonical(body)}).encode()))
    objects.append({'origin': origin, 'path': path, 'params': params, 'cache_key': key,
                    'sha256': archive.digest(file)})
    return file


def fixture(root, T=T):
    objects = []
    start = T - timedelta(days=551)
    sessions = md.sessions(start.date().isoformat(), T.date().isoformat())
    stock = [{'t': iso(s.open_at.replace(hour=4, minute=0)), 'o': 100, 'h': 102, 'l': 99, 'c': 101, 'v': 1000000}
             for s in sessions]
    bars = {s: stock for s in sn.STOCKS + sn.ETFS + sn.BENCHMARKS + ('SMLL',)}
    add(root, objects, md.ALPACA, '/v2/stocks/bars', {'start': iso(start), 'end': iso(T), 'symbols': ','.join(sorted(bars)),
        'timeframe': '1Day', 'feed': 'sip', 'adjustment': 'raw'}, {'bars': bars, 'next_page_token': None})
    daily = []
    at = T.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=300)
    while at < T:
        daily.append({'t': iso(at), 'o': 100, 'h': 102, 'l': 99, 'c': 101, 'v': 1000000})
        at += timedelta(days=1)
    add(root, objects, md.ALPACA, '/v1beta3/crypto/us/bars', {'start': iso(start), 'end': iso(T),
        'timeframe': '1Day', 'symbols': ','.join(md.crypto_pair(s) for s in sn.CRYPTO)},
        {'bars': {md.crypto_pair(s): daily for s in sn.CRYPTO}, 'next_page_token': None})
    n = {'id': 1, 'headline': 'SMLL cuts guidance 0000000001-26-000001', 'summary': 'FUTURE RESULT NEVER SHOWN',
         'content': 'later edited body', 'source': 'benzinga', 'symbols': ['SMLL'],
         'created_at': iso(T - timedelta(hours=1)), 'updated_at': iso(T - timedelta(minutes=30)),
         'url': 'https://example.invalid/future-story'}
    add(root, objects, md.ALPACA, '/v1beta1/news', {'start': iso(T-timedelta(hours=72)), 'end': iso(T)},
        {'news': [n], 'next_page_token': None})
    for status in ('active', 'inactive'):
        add(root, objects, 'https://paper-api.alpaca.markets', '/v2/assets', {'status': status, 'asset_class': 'us_equity'},
            [{'id': 'synthetic-asset', 'class': 'us_equity', 'exchange': 'NASDAQ', 'symbol': 'SMLL',
              'name': 'Synthetic Company', 'status': 'inactive', 'tradable': False}] if status == 'inactive' else [])
    add(root, objects, 'https://data.sec.gov', '/submissions/CIK0000000001.json', {},
        {'cik': '0000000001', 'tickers': ['SMLL'], 'filings': {'recent': {'form': ['8-K', '8-K'],
         'acceptanceDateTime': ['2026-07-14T09:00:00Z', '2026-07-15T08:00:00Z'],
         'accessionNumber': ['0000000001-26-000001', '0000000001-26-000002']}}})
    index = root / 'archive_index.json'
    index.write_text(json.dumps({'schema': 'replay-vendor-request-index-v1', 'objects': objects}))
    return archive.Archive(root, index), index


def test_strict_news_updated_before_T_and_creation_fields_only():
    # Audit item 4: updated_at < T strictly replaces the former 24 h edit tolerance.
    row = {'id': 1, 'headline': 'headline', 'source': 'benzinga', 'symbols': ['SMLL'],
           'created_at': iso(T - timedelta(hours=1)), 'updated_at': iso(T - timedelta(microseconds=1)),
           'summary': 'hidden', 'content': 'hidden', 'url': 'https://example.invalid/hidden'}
    assert inputs.news_at([row], T) == [{k: row[k] for k in ('id', 'headline', 'source', 'symbols', 'created_at')}]
    assert inputs.news_at([dict(row, updated_at=iso(T))], T) == []
    assert inputs.news_at([dict(row, updated_at=iso(T + timedelta(minutes=5)))], T) == []
    assert inputs.news_at([dict(row, created_at=iso(T), updated_at=iso(T))], T) == []
    with pytest.raises(ReplayError, match='NEWS_EDIT_CLOCK_MISSING'):
        inputs.news_at([dict(row, updated_at=None)], T)
    assert inputs.news_at([dict(row, updated_at=iso(T-timedelta(hours=2)))], T) == []


def test_raw_timestamps_completed_intervals_and_200_50_boundaries(tmp_path):
    data, _ = fixture(tmp_path)
    rows = inputs.rows_at(data.bars['SMLL'], T)
    assert all(r['close_at'] < T for r in rows)
    assert inputs.small_mid(rows[-199:], T) == 'HISTORY_LT_200_SESSIONS'
    exact = [dict(r, close=100.0, volume=100000) for r in rows[-200:]]
    assert inputs.small_mid(exact, T)['mean_dollar_volume_50d'] == 10e6
    exact[-1]['volume'] -= 1
    assert inputs.small_mid(exact, T) == 'DOLLAR_VOLUME_50D_LT_10M'
    future = dict(data.bars['SMLL'][-1], t=iso(T))
    assert inputs.rows_at([future], T) == []


def test_bundles_exclude_later_bodies_acceptance_at_T_and_keep_inactive(tmp_path):
    data, _ = fixture(tmp_path)
    bundles = inputs.build(data, T, 'a'*64)
    ctx = bundles['cd']['context']
    assert [c['symbol'] for c in ctx['candidates']] == ['SMLL']
    assert [f['accession'] for f in ctx['candidates'][0]['filings']] == ['0000000001-26-000001']
    assert 'FUTURE RESULT' not in json.dumps(bundles)
    assert 'future-story' not in json.dumps(bundles)
    assert 'summary' not in ctx['candidates'][0]['headlines'][0]
    assert ctx['provenance'] == 'APPROXIMATE_POINT_IN_TIME'
    assert bundles['cd']['manifest']['residual_risks'] == inputs.RISKS
    kept, evidence = inputs.primary_filter([{'symbol': 'SMLL', 'news_ids': [1]}], ctx)
    assert kept == {'SMLL'} and evidence['SMLL'] == ['0000000001-26-000001']
    ctx['candidates'][0]['headlines'][0]['title'] = 'unrelated story'
    assert not inputs.primary_filter([{'symbol': 'SMLL', 'news_ids': [1]}], ctx)[0]


def test_index_hash_pagination_raw_feed_and_receipts_fail_closed(tmp_path):
    data, index = fixture(tmp_path)
    with pytest.raises(ReplayError, match='LIVE_ARCHIVE_RECEIVED_AFTER_T'):
        archive.Archive(tmp_path, index, before=T)
    data.asset_statuses.remove('inactive')
    with pytest.raises(ReplayError, match='ACTIVE_AND_INACTIVE'):
        inputs.build(data, T, 'a'*64)
    data.news_windows[0][1]['next_page_token'] = 'missing'
    with pytest.raises(ReplayError, match='NEWS_ARCHIVE_COVERAGE_MISSING'):
        data.require_news(T - timedelta(hours=72), T)
    value = json.loads(index.read_text()); value['objects'][0]['params']['adjustment'] = 'all'
    index.write_text(json.dumps(value))
    with pytest.raises(ReplayError, match='ARCHIVE_REQUEST_HASH_MISMATCH'):
        archive.Archive(tmp_path, index)


def test_live_receipt_equal_to_T_is_not_before_the_decision(tmp_path):
    objects = []
    add(tmp_path, objects, md.ALPACA, '/v1beta1/news', {'start': iso(T-timedelta(hours=72)), 'end': iso(T)},
        {'news': [], 'next_page_token': None}, received=iso(T))
    index = tmp_path/'archive_index.json'
    index.write_text(json.dumps({'schema': 'replay-vendor-request-index-v1', 'objects': objects}))
    with pytest.raises(ReplayError, match='LIVE_ARCHIVE_RECEIVED_AFTER_T'):
        archive.Archive(tmp_path, index, before=T)


def test_missing_primary_archive_abstains_without_deleting_candidate(tmp_path):
    data, _ = fixture(tmp_path)
    data.filings.clear(); data.filing_symbols.clear()
    ctx = inputs.build(data, T, 'a'*64)['cd']['context']
    assert ctx['candidates'][0]['filings_state'] == 'UNAVAILABLE'
    assert not inputs.primary_filter([{'symbol': 'SMLL', 'news_ids': [1]}], ctx)[0]


def test_vendor_asset_UUIDs_are_identities_and_symbol_ambiguity_stays_rejected(tmp_path):
    data, index = fixture(tmp_path)
    objects = json.loads(index.read_text())['objects']
    row = dict(data.assets['SMLL'], id='second-synthetic-uuid')
    # Same response receipt, different UUID, same metadata: valid vendor shape.
    original = dict(data.assets['SMLL'])
    objects = [o for o in objects if not (o['path'] == '/v2/assets' and o['params']['status'] == 'inactive')]
    add(tmp_path, objects, 'https://paper-api.alpaca.markets', '/v2/assets',
        {'status': 'inactive', 'asset_class': 'us_equity'}, [original, row])
    index.write_text(json.dumps({'schema': 'replay-vendor-request-index-v1', 'objects': objects}))
    result = archive.Archive(tmp_path, index)
    assert result.assets['SMLL']['class'] == 'us_equity'
    assert not result.ambiguous_assets
    row = dict(row, id='third-synthetic-uuid', shortable=False)
    objects = [o for o in objects if not (o['path'] == '/v2/assets' and o['params']['status'] == 'inactive')]
    add(tmp_path, objects, 'https://paper-api.alpaca.markets', '/v2/assets',
        {'status': 'inactive', 'asset_class': 'us_equity'}, [original, row])
    index.write_text(json.dumps({'schema': 'replay-vendor-request-index-v1', 'objects': objects}))
    result = archive.Archive(tmp_path, index)
    assert result.ambiguous_assets == {'SMLL'}
    with pytest.raises(ReplayError, match='CANDIDATE_ASSET_IDENTITY_AMBIGUOUS'):
        inputs.build(result, T, 'a'*64)


def test_no_partial_manifest_on_missing_archive_and_no_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, 'declaration', lambda root: {'synthetic': True})
    with pytest.raises(ReplayError, match='VENDOR_REQUEST_INDEX_MISSING'):
        ex.build_inputs(tmp_path)
    assert not (tmp_path / spec.STATE_DIR / 'input-manifest.json').exists()
    data, _ = fixture(tmp_path)
    monkeypatch.setattr(ex, 'planned', lambda p, g: ['2026-07-15'])
    monkeypatch.setattr(ex, 'INPUT_SEAL', tmp_path / 'public-seal.json')
    result = ex.build_inputs(tmp_path)
    assert result['snapshot_count'] == 4
    proof = ex.input_proof(tmp_path)
    assert proof['method_hash'] == sha256_canonical({'synthetic': True})
    with pytest.raises(ReplayError, match='ALREADY_BUILT'):
        ex.build_inputs(tmp_path)
    file = next((tmp_path / spec.STATE_DIR / 'snapshots').rglob('*.json.gz'))
    file.write_bytes(file.read_bytes()+b'changed')
    with pytest.raises(ReplayError, match='INPUT_BUNDLE_HASH_MISMATCH'):
        ex.input_proof(tmp_path)


def test_long_weekend_requires_news_since_actual_previous_close(tmp_path):
    at = instant('2026-07-06T12:00:00Z')
    data, _ = fixture(tmp_path, at)
    assert md.previous_close(at) < at - timedelta(hours=72)
    with pytest.raises(ReplayError, match='NEWS_ARCHIVE_COVERAGE_MISSING'):
        inputs.build(data, at, 'a'*64)


def test_live_snapshot_freezes_real_receipt_cutoff_and_refuses_tampering(tmp_path, monkeypatch):
    from scripts.trading_lab.replay_seal_forward import forward
    from scripts.trading_lab.replay_bench import run as parent
    at = instant('2026-10-20T13:05:00Z')
    data, index = fixture(tmp_path, at)
    target = tmp_path/'live-archive/2026-10-20/archive_index.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(index.read_bytes())
    import shutil
    shutil.copytree(tmp_path/'cache', target.parent/'cache')
    monkeypatch.setattr(inputs, 'declaration', lambda root: {'synthetic': True})
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c'*64)
    first = forward.snapshot(tmp_path, at)
    assert first['provenance'] == 'LIVE_FORWARD'
    assert all(c['provenance'] == 'LIVE_FORWARD' for c in first['contexts'].values())
    assert forward.snapshot(tmp_path, at+timedelta(minutes=1)) == first
    path = forward.directory(tmp_path, '2026-10-20')/'snapshot.json.gz'
    value = parent.read_json(path); value['contexts']['ab']['universe'] = []
    parent.write_json(path, value)
    with pytest.raises(ReplayError, match='FORWARD_SNAPSHOT_HASH_MISMATCH'):
        forward.snapshot(tmp_path, at)


def test_declared_replay_end_to_end_keeps_all_abstention_forecasts(tmp_path, monkeypatch, ledger):
    from types import SimpleNamespace
    from scripts.trading_lab.replay_seal_forward import run, score, schemas
    import hashlib
    data, _ = fixture(tmp_path)
    monkeypatch.setattr(inputs, 'declaration', lambda root: {'synthetic': True})
    monkeypatch.setattr(ex, 'planned', lambda p, g: ['2026-07-15'])
    monkeypatch.setattr(ex, 'INPUT_SEAL', tmp_path/'public-seal.json')
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **kw: None)
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c'*64)
    freeze = tmp_path/'synthetic-freeze.json'
    freeze.write_text(json.dumps({'frozen_dev_base_rate': {'n': len(sn.UNIVERSE), 'successes': len(sn.UNIVERSE)}}))
    monkeypatch.setattr(ex, 'FREEZE', freeze)
    ex.build_inputs(tmp_path)
    class Client:
        def call(self, family, text, schema, *, tag):
            context = json.loads(text.split('\nCONTEXT_DATA:\n')[1].split('\nANALYST_DATA:\n')[0])
            if 'verdicts' in schema['properties']:
                original = json.loads(text.split('\nANALYST_DATA:\n')[1])
                output = {'verdicts': [dict(asset=v['asset'], horizon=v['horizon'], verdict='REJECT',
                    p_outperform=.6, note='synthetic') for v in original['views']]}
            elif 'forecasts' in schema['properties']:
                output = {'views': [dict(symbol=c['symbol'], view='ABSTAIN', p_outperform=.7, holding_sessions=3,
                    event_code='guidance_cut', news_ids=[1], reason='', counter_thesis='', falsifier='') for c in context['candidates']],
                    'forecasts': [dict(asset=c['symbol'], horizon=h, p_outperform=.7) for c in context['candidates'] for h in ('2d','10d')]}
            else:
                output = {'regime': ['synthetic'], 'views': [dict(asset=a, horizon=h, view='ABSTAIN', p_outperform=.7,
                    confidence_reason='synthetic', catalysts=[], priced_in_assessment='', second_order='', counter_thesis='',
                    falsifier='', event_risk=[]) for a in context['universe'] for h in ('1d','5d')]}
            return output, {'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
    client = Client()
    for family in ex.FAMILIES:
        for group in ex.GROUPS:
            record = ex.run_one(tmp_path, 'dev', group, '2026-07-15', family, SimpleNamespace(), client)
            assert record['provenance'] == 'APPROXIMATE_POINT_IN_TIME'
            assert record['residual_risks'] == inputs.RISKS
    class Outcomes:
        def __init__(self, *args, **kwargs):
            pass
        def get(self, group, day, asset, h):
            return {'rel': .02, 'cost': .003, 'vol': .01, 'cluster': 'synthetic',
                    'gross_path': {day: .02}, 'available_at': iso(T+timedelta(days=h, hours=8))}
    # Audit item 3: nothing is scored before its response hashes are public.
    with pytest.raises(ReplayError, match='DECISION_HASH_NOT_PUBLISHED'):
        score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=Outcomes)
    receipt = ex.publish_pending(tmp_path, ledger, '2026-10-19')
    assert len(receipt['line']['hashes']) == 6 and ex.publish_pending(tmp_path, ledger, '2026-10-19') is None
    result = score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=Outcomes)
    assert result['status'] == 'SCORED' and result['provenance'] == 'APPROXIMATE_POINT_IN_TIME'
    # Audit items 6, 7 and 10: EXPLORATORY label, declared primary endpoint, failure counts.
    assert result['evidence_class'] == 'EXPLORATORY'
    assert result['primary_endpoint']['panel'] == 'a_v1_as_live:1d' and result['primary_endpoint']['verdict'] == 'NO_WINNER'
    assert all(r['failed_or_retried_stages'] == r['missing_planned_days'] == r['failed_attempts'] == 0 for r in result['failures'].values())
    table = (ex.result_path(tmp_path, 'dev').with_suffix('.md')).read_text()
    assert 'EXPLORATORY' in table and 'No winner' in table and 'Failures' in table
    for family in ex.FAMILIES:
        assert result['models'][family]['b_v1_no_reviewer:1d']['n_forecasts'] == len(sn.UNIVERSE)
        assert result['models'][family]['b_v1_no_reviewer:1d']['abstention_rate'] == 1
