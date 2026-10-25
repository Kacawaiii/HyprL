"""Sentinel keeps AM, historical evidence and position-management failures visible."""
import hashlib

import pytest

from scripts.trading_lab.trader_agent.config import TraderError, instant
from scripts.trading_lab.trader_agent.sentinel_policy import expected_alert, expected_unit, patch_sentinel


@pytest.mark.parametrize('name', ['run-pm', 'recover-pm', 'rehearsal-pm',
    'rehearsal-pm-retry', 'rehearsal-pm-failure'])
def test_only_exact_pm_services_retire(name):
    unit = 'hyprl-trader-' + name + '.service'
    assert expected_unit(unit, instant('2026-10-25T23:59:59Z'))
    assert not expected_unit(unit, instant('2026-10-26T00:00:00Z'))


@pytest.mark.parametrize('name', ['run-pm', 'recover-pm', 'rehearsal-pm', 'rehearsal-pm-retry'])
def test_only_exact_pm_timers_retire(name):
    assert not expected_unit('hyprl-trader-' + name + '.timer', instant('2026-10-26T00:00:00Z'))


@pytest.mark.parametrize('name', ['run', 'recover', 'rehearsal', 'rehearsal-early', 'rehearsal-retry',
    'rehearsal-failure', 'health', 'label', 'paper-open', 'paper-audit', 'post-audit',
    'paper-exit-ia-actions', 'paper-exit-ia-crypto', 'some-other-pm'])
def test_am_and_position_management_units_remain_expected(name):
    for suffix in ('.service', '.timer'):
        assert expected_unit('hyprl-trader-' + name + suffix, instant('2026-10-26T18:00:00Z'))


@pytest.mark.parametrize('code', ['MISSING_PM_RUN', 'FAILED_PM_RUN'])
def test_only_post_retirement_pm_alerts_are_ignored(code):
    at = instant('2026-10-26T18:00:00Z')
    assert not expected_alert({'code': code, 'at': '2026-10-26T00:00:00Z'}, at)
    assert expected_alert({'code': code, 'at': '2026-10-25T23:59:59Z'}, at)
    assert expected_alert({'code': code, 'at': '2026-10-26T00:00:00Z'}, instant('2026-10-25T23:59:59Z'))
    for invalid in (None, 1, '', 'corrupt', '2026-10-26T00:00:00'):
        assert expected_alert({'code': code, 'at': invalid}, at)
    assert expected_alert({'code': code}, at)


@pytest.mark.parametrize('code', ['MISSING_DAILY_RUN', 'FAILED_DAILY_RUN', 'MISSING_LABEL_JOB',
    'PAPER_LOT_PAST_HORIZON', 'PAPER_RECONCILIATION_MISMATCH', 'PAPER_ORDER_UNRESOLVED',
    'LABELS_MISSING', 'RECOVERY_UNEXPECTED_FAILURE', 'PM_PAPER_PAPER_HTTP_500'])
def test_all_other_alerts_remain_actionable(code):
    assert expected_alert({'code': code, 'at': '2026-10-26T18:00:00Z'}, instant('2026-10-26T18:05:00Z'))


# Synthetic input with the installed script's four public control-flow anchors.
SOURCE = '''issues = []
for unit in units:
    if unit: issues.append(("unit:" + unit, unit))
for u, n in repeated:
    if n >= 3: issues.append(("repeat:" + u, u))
for a in alerts:
    if a.get("code") not in noisy: issues.append(("alert:" + a["code"], a["code"]))
'''


@pytest.mark.parametrize('at,expected', [
    ('2026-10-25T23:59:59Z', ['unit:hyprl-trader-run-pm.service', 'unit:hyprl-trader-run.service',
        'repeat:hyprl-trader-rehearsal-pm-retry.service', 'repeat:hyprl-trader-post-audit.service',
        'alert:MISSING_PM_RUN', 'alert:FAILED_PM_RUN', 'alert:MISSING_DAILY_RUN']),
    ('2026-10-26T18:00:00Z', ['unit:hyprl-trader-run.service', 'repeat:hyprl-trader-post-audit.service',
        'alert:FAILED_PM_RUN', 'alert:MISSING_DAILY_RUN']),
])
def test_staged_patch_filters_failed_units_repeated_units_and_new_alerts(tmp_path, at, expected):
    patched = patch_sentinel(SOURCE, tmp_path, hashlib.sha256(SOURCE.encode()).hexdigest())
    namespace = {'now': instant(at), 'units': ['hyprl-trader-run-pm.service', 'hyprl-trader-run.service'],
        'repeated': [('hyprl-trader-rehearsal-pm-retry.service', 3), ('hyprl-trader-post-audit.service', 3)],
        'alerts': [{'code': 'MISSING_PM_RUN', 'at': '2026-10-26T17:40:00Z'},
                   {'code': 'FAILED_PM_RUN', 'at': '2026-10-25T17:40:00Z'},
                   {'code': 'MISSING_DAILY_RUN', 'at': '2026-10-26T13:35:00Z'}], 'noisy': set()}
    exec(compile(patched, '<synthetic-sentinel>', 'exec'), namespace)
    assert [key for key, _ in namespace['issues']] == expected


def test_stage_refuses_changed_source_or_missing_anchor(tmp_path):
    with pytest.raises(TraderError, match='SENTINEL_SOURCE_CHANGED'):
        patch_sentinel(SOURCE, tmp_path, 'a' * 64)
    changed = SOURCE.replace('issues = []', 'issues=[]')
    with pytest.raises(TraderError, match='SENTINEL_PATCH_ANCHOR_MISMATCH'):
        patch_sentinel(changed, tmp_path, hashlib.sha256(changed.encode()).hexdigest())
