"""Synthetic decision chains, cost-aware comparisons and offline refresh boundaries."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from scripts.radar.cockpit_analysis import build_analysis, source, book_trades, news_stats
from scripts.radar.cockpit_refresh import embargo, render_units, radar_runtime, refresh
from scripts.radar.cockpit_export import ExportError, load_trader
from scripts.trading_lab.app_api.radar import RadarViews
from tests.radar.test_cockpit_export import RADAR, TRADER, FakeStore


def intent(symbol, tag, at='2026-10-10T10:00:00Z'):
    return {'at': at, 'action': 'buy_intent', 'symbol': symbol, 'limit': 100, 'risk_usd': 50,
            'journal': {'tag': tag, 'event': 'Synthetic earnings surprise', 'sources': [
                {'url': 'https://www.sec.gov/Archives/edgar/data/0000000001/a.htm', 'primary': True}],
                'expectations': 'Synthetic consensus EPS 1.00', 'expectations_at': '2026-10-09T18:00:00Z',
                'scenario': 'Margin expansion', 'plan': 'Wait for 101 breakout', 'invalidation': 'Margin below 10%'}}


def close(symbol, result=None):
    return {'at': '2026-10-10T20:00:00Z', 'action': 'closed', 'symbol': symbol, 'result': result or {}}


def test_news_comparison_excludes_missing_costs_and_open_marks():
    journal = [intent('AAA', 'news'), close('AAA', {'gross_pnl': 105, 'costs': 5}),
               intent('BBB', 'news'), close('BBB', {'gross_pnl': 80}),
               intent('CCC', 'technical'), close('CCC', {'pnl_after_costs': -50}),
               intent('DDD', 'news'), {'at': '2026-10-10T21:00:00Z', 'action': 'note', 'symbol': 'DDD', 'unrealized_pl': 90}]
    snapshot = build_analysis(RADAR, TRADER, journal=journal)
    news, technical, _ = snapshot['news']['groups']
    assert (news['count'], news['closed'], news['scored']) == (3, 2, 1)
    assert news['hit_rate'] == 1 and news['pnl_after_costs'] == 100 and news['average_r'] == 2
    assert news['state'] == 'TOO_EARLY'
    assert technical['hit_rate'] == 0 and technical['average_r'] == -1
    d = snapshot['book_trades'][0]['decision']
    assert d['verification'] == 'OFFICIEL' and d['expectation_at'] == '2026-10-09T18:00:00Z'
    assert d['entry_condition'] == 'Wait for 101 breakout' and d['invalidation'] == 'Margin below 10%'
    assert snapshot['news']['context_comparison'] == 'UNAVAILABLE'
    assert snapshot['news']['by_tag'][0]['group'] == 'news'


def test_ambiguous_lots_and_unknown_tags_are_not_guessed():
    trades = book_trades([intent('AAA', 'unfamiliar'), intent('AAA', 'news', '2026-10-10T11:00:00Z'), close('AAA', {'pnl_after_costs': 100})])
    assert all(t['state'] == 'INTENT' and t['decision']['result'] is None for t in trades)
    assert trades[0]['group'] == 'unclassified'
    assert news_stats(trades)[0]['scored'] == 0


def test_feed_and_primary_provenance_are_distinct_and_links_sanitized():
    assert source({'url': 'https://www.sec.gov/Archives/a.htm'})['verification'] == 'OFFICIEL'
    assert source({'url': 'https://www.sec.gov/Archives/a.htm', 'primary': False})['verification'] == 'FIL'
    assert source({'url': 'https://sec.gov.evil.invalid/a', 'publisher': 'SEC'})['verification'] == 'FIL'
    assert source({'url': 'https://news.example.invalid/a?token=SECRET', 'publisher': 'Benzinga'}) == {
        'url': 'https://news.example.invalid/a', 'publisher': 'Benzinga', 'published_at': None, 'verification': 'FIL'}
    assert source({'publisher': 'Fed'})['verification'] == 'NON VERIFIE'
    assert source({'url': 'https://credentials:password@example.invalid/a'})['url'] is None
    snapshot = build_analysis(RADAR, TRADER)
    assert all(o['decision']['verification'] == 'FIL' for o in snapshot['overlays'] if o['kind'] == 'event')
    assert 'RAW ARTICLE TEXT' not in json.dumps(snapshot)


def test_join_results_to_exact_run_analyst_asset_horizon():
    snapshot = build_analysis(RADAR, TRADER)
    decisions = [o for o in snapshot['overlays'] if o['kind'] == 'decision']
    results = [o for o in decisions if o['decision']['result']]
    assert len(results) == 1 and results[0]['analyst'] == 'analyst_claude'
    assert results[0]['at'] == '2026-10-09T12:00:00Z'
    assert results[0]['decision']['result']['net_return'] == .004
    assert all(o['decision']['result'] is None for o in snapshot['overlays'] if o['kind'] == 'event')


def test_missing_fields_and_secret_refusal():
    snapshot = build_analysis(RADAR)
    d = next(o['decision'] for o in snapshot['overlays'] if o['kind'] == 'event')
    assert d['entry_condition'] is None and d['expectation_at'] is None and d['result'] is None
    poisoned = intent('AAA', 'news')
    poisoned['journal']['plan'] = 'api_key=SECRET'
    with pytest.raises(ExportError):
        build_analysis(RADAR, journal=[poisoned])


@pytest.mark.parametrize('clock,blocked', [('11:41:59', False), ('11:42:00', True), ('11:45:00', True),
                                         ('12:29:59', True), ('12:30:00', False)])
def test_refresh_window_is_utc_and_protects_its_full_runtime(clock, blocked):
    at = datetime.fromisoformat('2026-10-11T' + clock + '+00:00')
    assert embargo(at) is blocked
    if blocked:
        assert refresh({}, at=at) == {'state': 'SKIPPED_TRADER_WINDOW'}


def test_timer_is_bounded_and_avoids_blackout(tmp_path):
    service, timer = render_units(tmp_path, '/usr/bin/python3', tmp_path / 'config.json')
    assert 'KillMode=process' in service and 'MemoryMax=384M' in service and 'TimeoutStartSec=3min' in service
    assert '*:35:00 UTC' in timer and 'Persistent=false' in timer
    assert 'claude-book status' not in service and '--live' not in service


def test_radar_budget_health_read_does_not_initialize_or_modify_store(tmp_path):
    db_path = tmp_path / 'radar.sqlite'
    with sqlite3.connect(db_path) as db:
        db.executescript('CREATE TABLE dispatch(day,group_name); CREATE TABLE evidence(seq,kind,body);')
        db.executemany('INSERT INTO dispatch VALUES (?,?)', [('2026-10-11', 'rss'), ('2026-10-11', 'rss'), ('2026-10-10', 'rss')])
        db.execute('INSERT INTO evidence VALUES (1,?,?)', ('health', json.dumps({'source': 'rss:example', 'status': 'BLOCKED', 'reason': 'OWNER_BUSY'})))
    before = db_path.read_bytes()
    counts, health = radar_runtime(tmp_path, '2026-10-11')
    assert counts == {'rss': 2} and health[0]['reason'] == 'OWNER_BUSY'
    assert db_path.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError):
        radar_runtime(tmp_path / 'missing', '2026-10-11')
    assert not (tmp_path / 'missing').exists()


def test_analysis_endpoint_is_fixed_schema_and_read_only(tmp_path):
    payload = build_analysis(RADAR)
    (tmp_path / 'analysis.json').write_text(json.dumps(payload))
    assert RadarViews(tmp_path).dispatch('/api/v1/radar/analysis', {}) == payload


def test_context_prices_keep_real_dates_and_exclude_synthetic():
    store = FakeStore({'replay-summary': [
        {'sequence': 1, 'payload': {'schema': 'trader-context-evidence-v1', 'context': {'synthetic': True, 'prices': {}}}},
        {'sequence': 2, 'payload': {'schema': 'trader-context-evidence-v1', 'context': {'synthetic': False, 'prices': {
            'AAA': {'state': 'AVAILABLE', 'recent_closes': [90, 100], 'last_price_at': '2026-10-09T20:00:00Z'}}}}}]})
    trader = load_trader(store)
    assert len(trader['contexts']) == 1
    assert build_analysis({}, trader)['prices']['AAA'] == [{'at': '2026-10-09T20:00:00Z', 'price': 100, 'basis': 'reference_close'}]


def test_offline_refresh_preserves_cached_account_dates_and_history(tmp_path):
    from scripts.trading_lab.research.store import ResearchStore
    from tests.radar.test_cockpit_export import BOOK_STATUS, PAPER_REPORT, JOURNAL
    from scripts.radar.cockpit_export import build_paper
    runtime = tmp_path / 'runtime'
    store = ResearchStore(runtime / 'evidence')
    store.close()
    radar = tmp_path / 'radar.json'
    radar.write_text(json.dumps(RADAR))
    journal = tmp_path / 'journal.jsonl'
    journal.write_text('\n'.join(json.dumps(r) for r in JOURNAL))
    out = tmp_path / 'snapshots'
    out.mkdir()
    paper = build_paper(PAPER_REPORT, BOOK_STATUS, JOURNAL, generated_at='2026-10-10T17:10:00Z')
    (out / 'paper.json').write_text(json.dumps(paper))
    history = out / 'equity-history.jsonl'
    history.write_text('synthetic existing equity point\n')
    config = {'out': str(out), 'radar': str(radar), 'trader_root': str(runtime), 'book_journal': str(journal)}
    at = datetime(2026, 10, 11, 1, 35, tzinfo=timezone.utc)
    before = (runtime / 'evidence/research.sqlite').read_bytes()
    assert refresh(config, at=at, read_units=False)['state'] == 'EXPORTED'
    after = json.loads((out / 'paper.json').read_text())
    assert after['generated_at'] == '2026-10-11T01:35:00Z'
    assert all(a['observed_at'] == '2026-10-10T17:10:00Z' for a in after['accounts'])
    assert history.read_text() == 'synthetic existing equity point\n'
    assert (runtime / 'evidence/research.sqlite').read_bytes() == before
    assert (out / 'analysis.json').is_file()
    assert refresh(config, at=datetime(2026, 10, 11, 2, 35, tzinfo=timezone.utc), read_units=False)['state'] == 'EXPORTED'
    assert all(a['observed_at'] == '2026-10-10T17:10:00Z' for a in json.loads((out / 'paper.json').read_text())['accounts'])
