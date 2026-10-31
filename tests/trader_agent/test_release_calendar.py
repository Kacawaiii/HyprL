"""Literal UTC expectations for the US fall clock change, with no market access."""
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from scripts.trading_lab.trader_agent.config import TraderError, instant, iso
from scripts.trading_lab.trader_agent.data import calendar_session, label_window
from scripts.trading_lab.trader_agent.alpaca_paper import after_hours_cutoff, order_terms
from scripts.trading_lab.trader_agent.release_window import rollout_day
from scripts.trading_lab.trader_agent.slots import start_at
from scripts.trading_lab.trader_agent.schedule import render
from scripts.trading_lab.trader_agent.rehearsal_schedule import render as render_rehearsal


@pytest.mark.parametrize('day,opening,closing,ny_hour', [
    ('2026-10-30', '13:30', '20:00', 8),
    ('2026-11-02', '14:30', '21:00', 7),
    ('2026-11-03', '14:30', '21:00', 7),
    ('2026-11-04', '14:30', '21:00', 7)])
def test_am_session_and_label_anchors_across_est(day, opening, closing, ny_hour):
    session = calendar_session(day)
    assert iso(session.open_at) == f'{day}T{opening}:00Z'
    assert iso(session.close_at) == f'{day}T{closing}:00Z'
    run = start_at(day, 'am')
    assert iso(run) == f'{day}T12:00:00Z'
    assert run.astimezone(ZoneInfo('America/New_York')).hour == ny_hour
    assert label_window('AAPL', day, '1d', []) == (session.open_at, session.close_at)
    crypto_entry, crypto_exit = label_window('SOL-USD', day, '1d', ['SOL-USD'])
    assert iso(crypto_entry) == f'{day}T13:30:00Z'
    assert crypto_exit == crypto_entry + timedelta(days=1)


def test_november_second_long_labels_and_after_hours_cross_utc_midnight():
    first, last = label_window('AAPL', '2026-11-02', '5d', [])
    assert (iso(first), iso(last)) == ('2026-11-02T14:30:00Z', '2026-11-06T21:00:00Z')
    first, last = label_window('AAPL', '2026-11-02', '1d', [], entry_at='close')
    assert (iso(first), iso(last)) == ('2026-11-02T21:00:00Z', '2026-11-03T21:00:00Z')
    session = calendar_session('2026-11-02')
    assert iso(after_hours_cutoff(session)) == '2026-11-03T00:30:00Z'


def test_rendered_timers_keep_utc_runs_and_new_york_execution(tmp_path):
    units = render(tmp_path, '/synthetic/python', '/synthetic/grant', tmp_path,
                   paper_authorization='/synthetic/paper')
    assert 'OnCalendar=*-*-* 12:00:00 UTC' in units['hyprl-trader-run.timer']
    assert 'OnCalendar=*-*-* 21:30:00 UTC' in units['hyprl-trader-label.timer']
    assert '09:30:30 America/New_York' in units['hyprl-trader-paper-open.timer']
    assert '15:55:00 America/New_York' in units['hyprl-trader-paper-exit-ia-actions.timer']
    early = render_rehearsal(tmp_path, '/synthetic/python', '/synthetic/grant', tmp_path,
                             '/synthetic/paper', '/synthetic/data', path='/usr/bin')
    assert 'OnCalendar=*-*-* 07:35:00 UTC' in early['hyprl-trader-rehearsal-early.timer']
    assert 'OnCalendar=*-*-* 10:45:00 UTC' in early['hyprl-trader-rehearsal.timer']
    assert 'OnCalendar=*-*-* 11:15:00 UTC' in early['hyprl-trader-rehearsal-retry.timer']


@pytest.mark.parametrize('purpose,clock,allowed', [
    ('entry', '14:30:29', False), ('entry', '14:30:30', True),
    ('entry', '14:34:59', True), ('entry', '14:35:00', False),
    ('entry', '13:30:30', False), ('exit', '20:55:00', True),
    ('exit', '20:54:59', False), ('exit', '21:00:00', False)])
def test_est_entry_and_exit_boundaries(purpose, clock, allowed):
    session = calendar_session('2026-11-02')
    at = instant(f'2026-11-02T{clock}Z')
    if allowed:
        assert order_terms('ia_actions', purpose, at, session) == {'type': 'market', 'time_in_force': 'day'}
    else:
        with pytest.raises(TraderError, match='PAPER_.*_WINDOW'):
            order_terms('ia_actions', purpose, at, session)


def test_est_post_audit_guards_and_checkpoints(grant, tmp_path):
    from scripts.trading_lab.research.store import ResearchStore
    from scripts.trading_lab.trader_agent.post_audit import checkpoints, execution_window
    for clock, blocked in [('14:28:00', True), ('14:32:59', True), ('14:33:00', False),
                            ('20:53:00', True), ('20:57:59', True), ('20:58:00', False)]:
        assert execution_window(instant('2026-11-02T' + clock + 'Z')) == blocked
    research = ResearchStore(tmp_path / 'evidence')
    try:
        due = checkpoints(grant, research, [], instant('2026-11-02T22:00:00Z'))
        assert iso(due['session-entry:2026-11-02']) == '2026-11-02T14:40:30Z'
        assert iso(due['label-deadline:2026-11-02']) == '2026-11-02T21:40:00Z'
    finally:
        research.close()


@pytest.mark.parametrize('day', ['2026-11-03', '2026-11-04'])
def test_rollout_is_fresh_utc_day_before_early_rehearsal(day):
    # 00:00 UTC is 19:00 EST on the prior local date; the bank uses UTC.
    assert rollout_day(instant(day + 'T00:00:00Z')) == day
    assert rollout_day(instant(day + 'T07:24:59Z')) == day
    assert rollout_day(instant(day + 'T07:29:59Z'), finishing=True) == day
    for clock, finishing in [('07:25:00', False), ('07:30:00', True), ('07:35:00', False)]:
        with pytest.raises(TraderError, match='RELEASE_OUTSIDE_ROLLOUT_WINDOW'):
            rollout_day(instant(day + 'T' + clock + 'Z'), finishing=finishing)


@pytest.mark.parametrize('day', ['2026-11-01', '2026-11-02', '2026-11-05'])
def test_rollout_refuses_unreviewed_dates(day):
    with pytest.raises(TraderError, match='RELEASE_OUTSIDE_ROLLOUT_WINDOW'):
        rollout_day(instant(day + 'T00:00:00Z'))
