"""No systemd mutation or notification leaves these tests."""
from datetime import timedelta
import hashlib
from types import SimpleNamespace
import pytest
from scripts.trading_lab.trader_agent.config import instant
from scripts.trading_lab.trader_agent.release_guard import STEPS, early_green, guard


def attempt(day='2026-10-30'):
    return dict(state='GREEN', slot='early', run_slot='am', at=day+'T07:35:00Z',
                finished_at=day+'T07:40:00Z', steps=[dict(step=s, state='GREEN') for s in STEPS])


@pytest.mark.parametrize('fault', ['missing', 'stale', 'running', 'partial', 'late', 'undelivered', 'good'])
def test_only_complete_current_rehearsal_can_retain_release(fault):
    row = attempt()
    if fault == 'missing': row = None
    elif fault == 'stale': row['at'] = '2026-10-29T07:35:00Z'
    elif fault == 'running': row['state'] = 'RUNNING'
    elif fault == 'partial': row['steps'].pop()
    elif fault == 'late': row['finished_at'] = '2026-10-30T08:00:00Z'
    elif fault == 'undelivered': row['alerts'] = [dict(delivery='FAILED')]
    assert early_green(row, '2026-10-30', '2026-10-29T23:00:00Z', instant('2026-10-30T07:50:00Z')) == (fault == 'good')


@pytest.mark.parametrize('green,rollback_rc', [(True, 0), (False, 0), (False, 1)])
@pytest.mark.parametrize('day', ['2026-10-30', '2026-11-02', '2026-11-03', '2026-11-04'])
def test_guard_runs_bound_rollback_on_missing_monitor_and_reports_outcome(tmp_path, monkeypatch, green, rollback_rc, day):
    import scripts.trading_lab.trader_agent.release_guard as module
    calls = []
    script = tmp_path / 'handoff.sh'
    script.write_text('#!/bin/bash\nexit 0\n')
    config = dict(day=day, deployed_at=day+'T00:00:00Z', release='/synthetic/release',
                  monitor=str(tmp_path / 'missing.sqlite'), rollback_script=str(script),
                  rollback_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), proof='/synthetic/proof', notify='/synthetic/notify')
    if green: monkeypatch.setattr(module, 'read_attempt', lambda *a: attempt(day))
    def run(args, **kwargs):
        calls.append(args)
        if args[0] == 'systemctl':
            return SimpleNamespace(returncode=0, stdout='WorkingDirectory=/synthetic/release\nResult=success\nExecMainStatus=0\n')
        if args[0] == 'bash': return SimpleNamespace(returncode=rollback_rc, stdout='')
        assert args == ['/synthetic/notify', 'alertes']
        assert kwargs['input'].startswith('Trader release guard: ')
        return SimpleNamespace(returncode=0, stdout='discord\n')
    result = guard(config, at=instant(day+'T07:50:00Z'), run=run)
    assert result['notification'] == 'DELIVERED'
    assert any(c[0] == 'bash' for c in calls) == (not green)
    assert result['outcome'] == ('RELEASE_RETAINED_EARLY_GREEN' if green else 'ROLLED_BACK_EARLY_NOT_GREEN' if rollback_rc == 0 else 'ROLLBACK_FAILED_TIMERS_REQUIRE_REVIEW')
