import json
from types import SimpleNamespace
import pytest

from scripts.trading_lab.agent_v2.contracts import Blocked, iso
from scripts.trading_lab.agent_v2.models import Claude
from scripts.trading_lab.agent_v2.spec import PREREG_HASH, spec


def grant(tmp_path, clock, cap=1):
    path=tmp_path/'synthetic-v2-grant.json'
    path.write_text(json.dumps({'schema':'agent-v2-shadow-authorization-v1','operator_signed':True,
        'mode':'SHADOW_FORWARD','preregistration_hash':PREREG_HASH,'operator_decision_hash':spec()['operator_decision_hash'],
        'granted_at':'2026-01-01T00:00:00Z','not_after':'2027-01-01T00:00:00Z','model':'sonnet','claude_calls_per_day':cap}))
    return path


def test_budget_reserved_before_failure_and_no_retry(ledger, clock, tmp_path, monkeypatch):
    invoked=[]
    def failure(argv, **kwargs):
        invoked.append(argv)
        assert len(ledger.records('model_reservation')) == 1
        assert argv[argv.index('--tools')+1] == ''
        assert '--strict-mcp-config' in argv and '--no-session-persistence' in argv
        return SimpleNamespace(returncode=1, stdout='', stderr='synthetic failure')
    monkeypatch.setattr('subprocess.run', failure)
    runner=Claude(ledger,grant(tmp_path,clock),clock=clock)
    with pytest.raises(Blocked, match='MODEL_FAILED'):
        runner.call('news', {'sources':[]})
    with pytest.raises(Blocked, match='MODEL_BUDGET'):
        runner.call('news', {'sources':[]})
    assert len(invoked) == 1


def test_usage_limit_halts_remaining_day(service, ledger, snapshot, clock, tmp_path, monkeypatch):
    invoked=[]
    def quota(argv, **kwargs):
        invoked.append(argv)
        return SimpleNamespace(returncode=1, stdout='', stderr='usage limit reached')
    monkeypatch.setattr('subprocess.run', quota)
    service.runner=Claude(ledger,grant(tmp_path,clock,8),clock=clock)
    with pytest.raises(Blocked,match='MODEL_USAGE_LIMIT'):
        service.decide(snapshot)
    with pytest.raises(Blocked,match='MODEL_USAGE_LIMIT'):
        service.runner.call('news',{'sources':[]})
    assert len(invoked) == 1
    assert len(ledger.records('model_reservation')) == 1
