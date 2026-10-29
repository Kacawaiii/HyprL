"""Synthetic daily source boundaries; never issue a real HTTP request."""
from datetime import date, timedelta
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import pytest
from scripts.agent_v2_refresh.calendar import fomc_events
from scripts.agent_v2_refresh import daily
from scripts.trading_lab.agent_v2_quality.contracts import Blocked, instant, iso
from scripts.trading_lab.agent_v2_quality.ledger import Journal
from scripts.trading_lab.agent_v2_quality.monday_staging import prepare
from scripts.trading_lab.agent_v2_quality.producer import atomic_json, read_local
from scripts.trading_lab.agent_v2_quality.synthetic import bundle

FOMC = ('2026 FOMC Meetings January 27-28 March 17-18 April 28-29 June 16-17 '
        'July 28-29 September 15-16 October 27-28 December 8-9 2025 FOMC Meetings')


def test_fomc_daily_horizon_and_missing_schedule_fail_closed():
    assert fomc_events(FOMC, date(2026,10,29), date(2026,11,5)) == []
    rows = fomc_events(FOMC, date(2026,10,26), date(2026,11,3))
    assert [r['at'] for r in rows] == ['2026-10-27T00:00:00Z','2026-10-28T00:00:00Z']
    with pytest.raises(RuntimeError, match='COMPLETE_SCHEDULE'):
        fomc_events('2026 FOMC Meetings October 27-28',date(2026,10,29),date(2026,11,5))
    with pytest.raises(RuntimeError, match='YEAR_COVERAGE'):
        fomc_events(FOMC,date(2026,12,31),date(2027,1,6))


def setup(tmp_path, monkeypatch):
    settings, boundary = prepare(tmp_path/'staging')
    boundary.clock.at = instant('2026-10-29T10:15:00Z')
    data = bundle(boundary.clock())
    old = read_local(settings['seed_registry'])
    data['registry'].update(edges=old['edges'], sources=old['sources'])
    for row in data['registry']['instruments']: row['provenance'] = {'synthetic':True}
    atomic_json(settings['seed_registry'], data['registry'])
    original = read_local(settings['base_config'])
    atomic_json(original['hunter'], data['snapshot']['hunter'])
    boundary.data = data
    folder = tmp_path/'reports'; folder.mkdir()
    radar = read_local(settings['radar_source'])
    radar['cutoff'] = '2026-10-28T19:30:00Z'
    for event in radar['events']:
        event['stories'] = [event['stories'][0]]
        event['stories'][0].update(published_at='2026-10-28T18:00:00Z', received_at='2026-10-28T18:01:00Z')
    atomic_json(folder/'radar-current.json',radar)
    runtime = tmp_path/'runtime'
    journal = Journal.initialize(runtime,'SHADOW_FORWARD','a'*40,instant('2026-10-28T09:00:00Z'))
    with journal.owned():
        journal.append([('http_reservation',{'group':'sec','url':'https://data.sec.gov/submissions/CIK0000000001.json'})],instant('2026-10-28T09:01:00Z'))
    journal.close()
    monkeypatch.setattr(daily,'Grant',lambda path, clock: boundary.grant(path))
    send_original = boundary.send
    def send(url,headers):
        parsed = urlsplit(url)
        if parsed.hostname == 'www.federalreserve.gov':
            boundary.requests.append(url)
            return 200, ('<html>'+FOMC+'</html>').encode()
        if parsed.hostname == 'www.bls.gov':
            boundary.requests.append(url)
            return 200, ('<table>'+''.join('<td id="d'+(date(2026,10,29)+timedelta(days=i)).strftime('%m%d')+'"></td>' for i in range(8))+'</table>').encode()
        if parsed.hostname == 'api.nasdaq.com':
            boundary.requests.append(url)
            day = date.fromisoformat(parse_qs(parsed.query)['date'][0])
            return 200,json.dumps({'status':{'rCode':200},'data':{'asOf':day.strftime('%a, %b %d, %Y'),
                'rows':[dict(symbol=s,marketCap='$2,000,000,000',time='time-not-supplied') for s in ('ACME','HUNT')] if day==date(2026,10,29) else []}}).encode()
        return send_original(url,headers)
    return dict(runtime=str(runtime),engine_sha='a'*40,output=str(tmp_path/'daily'),
                base_config=settings['base_config'],seed_registry=settings['seed_registry'],
                calendar_grant=settings['calendar_grant'],radar_reports=str(folder),pinned_files={}),boundary,send


def test_thursday_refresh_preserves_frozen_binding_and_budgets(tmp_path,monkeypatch):
    settings,boundary,send = setup(tmp_path,monkeypatch)
    result=daily.refresh(settings,clock=boundary.clock,send=send,pause=boundary.pause)
    assert result['status']=='REFRESHED' and result['model_calls']==result['broker_orders']==0
    config=read_local(Path(settings['output'])/'current-config.json')
    assert read_local(config['calendar'])['coverage_end']=='2026-11-05T21:00:00Z'
    assert read_local(config['registry'])['edges']['equity_momentum']['enabled'] is False
    journal=Journal(settings['runtime'],read_only=True)
    try:
        assert journal.binding['code_sha']=='a'*40
        assert not journal.records('model_reservation') and not journal.records('cycle_start')
        assert len(journal.records('http_reservation'))>1
    finally:
        journal.close()
    requests=len(boundary.requests)
    with pytest.raises(FileExistsError):
        daily.refresh(settings,clock=boundary.clock,send=send,pause=boundary.pause)
    assert len(boundary.requests)==requests


def test_calendar_failure_keeps_previous_generation_and_never_retries(tmp_path,monkeypatch):
    settings,boundary,send = setup(tmp_path,monkeypatch)
    root=Path(settings['output']);root.mkdir()
    atomic_json(root/'current-config.json',{'previous':'unchanged'})
    with pytest.raises(RuntimeError,match='CALENDAR_HTTP_403'):
        daily.refresh(settings,clock=boundary.clock,send=lambda *a:(403,b''),pause=boundary.pause)
    assert read_local(root/'current-config.json')=={'previous':'unchanged'}
    with pytest.raises(FileExistsError):
        daily.refresh(settings,clock=boundary.clock,send=lambda *a:pytest.fail('retry'),pause=boundary.pause)


def test_closed_or_late_day_never_acquires(tmp_path):
    clock=lambda:instant('2026-10-31T10:15:00Z')
    assert daily.refresh({},clock=clock)['status']=='SKIPPED_NYSE_CLOSED'
    with pytest.raises(Blocked,match='WINDOW_CLOSED'):
        daily.refresh({},clock=lambda:instant('2026-10-29T12:30:00Z'))
