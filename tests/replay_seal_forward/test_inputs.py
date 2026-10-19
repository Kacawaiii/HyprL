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
              'name': 'Synthetic Company', 'status': 'inactive', 'tradable': False}] if status == 'inactive' else
            [dict(id='synthetic-' + symbol, symbol=symbol, name='Synthetic Company',
                  exchange='NASDAQ', status='active', **{'class':'us_equity'})
             for symbol in sn.STOCKS + sn.ETFS + sn.BENCHMARKS])
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
    assert inputs.news_at([dict(row, updated_at=None)], T) == []
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


def test_bundles_withhold_all_cached_filings_and_bodies_and_keep_inactive(tmp_path):
    data, _ = fixture(tmp_path)
    bundles = inputs.build(data, T, 'a'*64)
    ctx = bundles['cd']['context']
    assert [c['symbol'] for c in ctx['candidates']] == ['SMLL']
    assert ctx['candidates'][0]['filings'] == []
    assert ctx['candidates'][0]['filings_state'] == 'UNAVAILABLE'
    assert 'FUTURE RESULT' not in json.dumps(bundles)
    assert 'future-story' not in json.dumps(bundles)
    assert 'summary' not in ctx['candidates'][0]['headlines'][0]
    assert ctx['provenance'] == 'APPROXIMATE_POINT_IN_TIME'
    assert bundles['cd']['manifest']['residual_risks'] == inputs.RISKS
    kept, evidence = inputs.primary_filter([{'symbol': 'SMLL', 'news_ids': [1]}], ctx)
    assert kept == set() and evidence['SMLL'] == []
    ctx['candidates'][0]['headlines'][0]['title'] = 'unrelated story'
    assert not inputs.primary_filter([{'symbol': 'SMLL', 'news_ids': [1]}], ctx)[0]


def test_index_hash_pagination_raw_feed_and_receipts_fail_closed(tmp_path):
    data, index = fixture(tmp_path)
    with pytest.raises(ReplayError, match='LIVE_ARCHIVE_RECEIVED_AFTER_T'):
        archive.Archive(tmp_path, index, before=T)
    data.asset_statuses.remove('inactive')
    bundles = inputs.build(data, T, 'a'*64)
    assert bundles['ab']['manifest']['data_quality']['excluded_day']
    assert bundles['ab']['context']['universe'] == []
    assert bundles['cd']['context']['candidates'] == []
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


def test_uniform_news_and_sec_exclusions_apply_before_selection(tmp_path):
    data, _ = fixture(tmp_path)
    original = dict(data.news[1])
    data.news[2] = dict(original, id=2, updated_at=iso(T-timedelta(hours=2)))
    data.index['sec_availability'] = {'state': 'UNAVAILABLE', 'reason': 'HTTP_403'}
    bundles = inputs.build(data, T, 'a'*64)
    for bundle in bundles.values():
        assert bundle['manifest']['news_data_quality']['excluded_count'] == 1
    candidate = bundles['cd']['context']['candidates'][0]
    assert [h['news_id'] for h in candidate['headlines']] == [1]
    assert candidate['filings_state'] == 'UNAVAILABLE' and candidate['filings'] == []
    assert bundles['ab']['context']['archives']['edgar']['state'] == 'UNAVAILABLE'


def test_same_receipt_news_body_conflicts_remain_rejected(tmp_path):
    data, index = fixture(tmp_path)
    objects = json.loads(index.read_text())['objects']
    # Bodies are not held in memory, but their conflicts must still fail closed.
    original = dict(data.news[1], content='conflicting body')
    add(tmp_path, objects, md.ALPACA, '/v1beta1/news', {'start': iso(T-timedelta(hours=73)), 'end': iso(T)},
        {'news': [original], 'next_page_token': None})
    index.write_text(json.dumps({'schema': 'replay-vendor-request-index-v1', 'objects': objects}))
    loaded = archive.Archive(tmp_path, index)
    assert inputs.build(loaded, T, 'a'*64)['ab']['manifest']['data_quality']['counts_by_reason']['CONTRADICTORY_NEWS'] == 1


def test_disk_bar_index_preserves_latest_full_rows_and_rejects_conflicts(tmp_path):
    data, index = fixture(tmp_path)
    objects = json.loads(index.read_text())['objects']
    original = data.bars['SMLL'][-1]
    params = {'start':iso(T-timedelta(days=500)), 'end':iso(T), 'symbols':'SMLL',
              'timeframe':'1Day', 'feed':'sip', 'adjustment':'raw'}
    fresh = dict(original, c=105, n=42, vw=103.5)
    add(tmp_path, objects, md.ALPACA, '/v2/stocks/bars', params, {'bars':{'SMLL':[fresh]}},
        received='2026-10-18T20:00:00Z')
    add(tmp_path, objects, md.ALPACA, '/v2/stocks/bars', dict(params,start=iso(T-timedelta(days=501))),
        {'bars':{'SMLL':[dict(original,c=99)]}}, received='2026-10-18T17:00:00Z')
    index.write_text(json.dumps({'schema':'replay-vendor-request-index-v1','objects':objects}))
    loaded = archive.Archive(tmp_path,index)
    assert loaded.bars['SMLL'][-1] == fresh
    assert len(loaded.bars['SMLL']) == len(data.bars['SMLL'])
    add(tmp_path, objects, md.ALPACA, '/v2/stocks/bars', dict(params,start=iso(T-timedelta(days=502))),
        {'bars':{'SMLL':[dict(fresh,c=106)]}}, received='2026-10-18T20:00:00Z')
    index.write_text(json.dumps({'schema':'replay-vendor-request-index-v1','objects':objects}))
    loaded = archive.Archive(tmp_path, index)
    assert loaded.invalid_bars['SMLL'] == 'CONTRADICTORY_BARS'


def test_vendor_asset_UUID_duplicates_agree_and_contradictions_exclude_uniformly(tmp_path):
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
    bundles = inputs.build(result, T, 'a'*64)
    assert bundles['cd']['context']['candidates'] == []
    for bundle in bundles.values():
        assert bundle['manifest']['rejections']['SMLL'] == 'CONTRADICTORY_ASSET_METADATA'
        assert bundle['manifest']['asset_metadata_exclusions']['excluded_count'] == 1


def test_missing_archive_seals_excluded_days_and_admitted_inputs_cannot_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, 'declaration', lambda root: {'synthetic': True})
    monkeypatch.setattr(inputs, 'paired_label_exclusions', lambda root: [])
    result = ex.build_inputs(tmp_path / 'missing')
    assert result['status'] == 'BUILT'
    proof = ex.input_proof(tmp_path / 'missing')
    assert proof['data_quality']['excluded_days'] == 64
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
    bundles = inputs.build(data, at, 'a'*64)
    assert bundles['ab']['manifest']['data_quality']['excluded_day']
    assert bundles['cd']['context']['candidates'] == []


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


@pytest.mark.parametrize('paired_missing', [False, True])
def test_declared_replay_end_to_end_keeps_all_abstention_forecasts(tmp_path, monkeypatch, ledger, paired_missing):
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
    expected = len(sn.UNIVERSE) - int(paired_missing)
    freeze.write_text(json.dumps({'frozen_dev_base_rate': {'n': expected, 'successes': expected}}))
    monkeypatch.setattr(ex, 'FREEZE', freeze)
    coverage = tmp_path/spec.STATE_DIR/'outcome-collection.json'
    coverage.parent.mkdir(parents=True)
    coverage.write_text(json.dumps({'missing': [dict(group='ab', day='2026-07-15', asset=sn.UNIVERSE[0], h=1)]
                                   if paired_missing else []}))
    monkeypatch.setattr(inputs, 'paired_label_exclusions', lambda root: json.loads(coverage.read_text())['missing'])
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
        def __init__(self, transport, *args, **kwargs):
            self.excluded = transport.excluded_labels
        def get(self, group, day, asset, h):
            if (group, day, asset, h) in self.excluded:
                return None
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
        assert result['models'][family]['b_v1_no_reviewer:1d']['n_forecasts'] == expected
        assert result['models'][family]['b_v1_no_reviewer:1d']['abstention_rate'] == 1
        if paired_missing:
            assert result['counts'][family]['a_v1_as_live:1d:paired_missing_label'] == 1
            assert result['counts'][family]['b_v1_no_reviewer:1d:paired_missing_label'] == 1


def test_missing_asset_metadata_excludes_uniformly_and_is_not_a_contradiction(tmp_path):
    data, _ = fixture(tmp_path)
    del data.assets['SMLL']
    bundles = inputs.build(data, T, 'a'*64)
    assert bundles['cd']['context']['candidates'] == []
    for bundle in bundles.values():
        assert bundle['manifest']['rejections']['SMLL'] == 'CANDIDATE_ASSET_METADATA_MISSING'
        assert bundle['manifest']['data_quality']['counts_by_reason']['CANDIDATE_ASSET_METADATA_MISSING'] == 1


def test_asset_contradiction_across_receipts_excludes_the_symbol(tmp_path):
    data, index = fixture(tmp_path)
    objects = json.loads(index.read_text())['objects']
    row = dict(data.assets['SMLL'], shortable=False)
    objects = [o for o in objects if not (o['path'] == '/v2/assets' and o['params']['status'] == 'active')]
    add(tmp_path, objects, 'https://paper-api.alpaca.markets', '/v2/assets',
        {'status': 'active', 'asset_class': 'us_equity'}, [row], received='2026-10-19T00:00:00Z')
    index.write_text(json.dumps({'schema':'replay-vendor-request-index-v1','objects':objects}))
    result = archive.Archive(tmp_path, index)
    assert result.ambiguous_assets == {'SMLL'}
    assert inputs.build(result, T, 'a'*64)['cd']['context']['candidates'] == []


def test_paired_label_exclusion_source_is_required_and_digest_bound(tmp_path, monkeypatch):
    rows = [{'group':'ab','day':'2026-07-01','asset':'AAPL','h':1}]
    source = tmp_path/'duel-v5/missing-label-retry.json'
    source.parent.mkdir()
    source.write_text(json.dumps({'missing':rows}))
    artifacts = tmp_path/'docs/artifacts'
    artifacts.mkdir(parents=True)
    labels = {'retry_record_sha256':archive.digest(source), 'remaining_excluded':1,
              'excluded_keys_sha256':sha256_canonical([('ab','2026-07-01','AAPL',1)])}
    (artifacts/'replay_unblock_evidence_v1.json').write_text(json.dumps({'labels':labels}))
    config = spec.spec()
    monkeypatch.setattr(spec, 'spec', lambda: config)
    monkeypatch.setattr(spec, 'ROOT', tmp_path)
    assert inputs.paired_label_exclusions(tmp_path) == rows
    source.write_text(json.dumps({'missing':[]}))
    with pytest.raises(ReplayError, match='PAIRED_LABEL_EXCLUSION_SOURCE_CHANGED'):
        inputs.paired_label_exclusions(tmp_path)


@pytest.mark.parametrize('change,reason', [
    ({'updated_at': None}, 'NEWS_CLOCK_MISSING'),
    ({'created_at': 'invalid'}, 'NEWS_CLOCK_INVALID'),
    ({'updated_at': '2026-07-01T00:00:00Z'}, 'NEWS_CLOCK_CONTRADICTORY'),
    ({'symbols': None}, 'NEWS_FIELDS_INVALID'),
])
def test_bad_required_news_data_is_excluded_and_counted_in_both_bundles(tmp_path, change, reason):
    data, _ = fixture(tmp_path)
    data.news[1].update(change)
    bundles = inputs.build(data, T, 'a'*64)
    for bundle in bundles.values():
        assert bundle['manifest']['data_quality']['counts_by_reason'][reason] == 1
    assert bundles['cd']['context']['candidates'] == []


@pytest.mark.parametrize('failure,reason', [
    ('missing_history', 'CANDIDATE_HISTORY_COVERAGE_MISSING'),
    ('invalid_bar', 'INVALID_BAR'),
    ('invalid_metadata', 'INVALID_ASSET_METADATA'),
])
def test_candidate_required_data_excludes_before_ranking(tmp_path, failure, reason):
    data, _ = fixture(tmp_path)
    if failure == 'missing_history':
        data.requests = [r for r in data.requests if r[1] != '/v2/stocks/bars']
    elif failure == 'invalid_bar':
        data.invalid_bars['SMLL'] = 'INVALID_BAR'
    else:
        data.assets['SMLL']['class'] = None
    bundles = inputs.build(data, T, 'a'*64)
    assert bundles['cd']['context']['candidates'] == []
    assert bundles['ab']['context']['universe']
    assert all(b['manifest']['rejections']['SMLL'] == reason for b in bundles.values())


def test_missing_benchmark_excludes_whole_day_but_integrity_failure_still_blocks(tmp_path):
    data, _ = fixture(tmp_path)
    del data.assets['SPY']
    bundles = inputs.build(data, T, 'a'*64)
    assert bundles['ab']['context']['universe'] == []
    assert bundles['cd']['context']['candidates'] == []
    assert all(b['manifest']['data_quality']['excluded_day'] for b in bundles.values())
    def corrupt(*args):
        raise ReplayError('ARCHIVE_OBJECT_HASH_MISMATCH')
    data.require_news = corrupt
    with pytest.raises(ReplayError, match='ARCHIVE_OBJECT_HASH_MISMATCH'):
        inputs.build(data, T, 'a'*64)


def test_missing_live_index_produces_a_shared_excluded_day(tmp_path, monkeypatch):
    from scripts.trading_lab.replay_seal_forward import forward
    monkeypatch.setattr(inputs, 'declaration', lambda root: {'synthetic': True})
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c'*64)
    value = forward.snapshot(tmp_path, forward.decision_time('2026-10-20'))
    assert value['contexts']['ab']['universe'] == []
    assert value['contexts']['cd']['candidates'] == []
    for manifest in value['manifests'].values():
        assert manifest['data_quality']['excluded_day']
        assert manifest['data_quality']['counts_by_reason']['LIVE_VENDOR_REQUEST_INDEX_MISSING'] == 1
