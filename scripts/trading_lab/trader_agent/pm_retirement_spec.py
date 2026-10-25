"""Forward-only PM lifecycle; original PM bindings still govern historical lots."""
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError, instant, strict_json

DECISION_NAME = 'trader-operator-decisions-2026-10-25.json'
DECISION_HASH = '569b16014a1f33dff820a84cbb1a2105ddde01501733a652611112a756125730'
SPEC_HASH = '911289a7423b82a3e48965d15e034bc915ca1c414dd33add48db50b362cf5fcc'
EFFECTIVE_FROM = '2026-10-26T00:00:00Z'
ROOT = Path(__file__).resolve().parents[3] / 'docs/artifacts'
PM_SERVICES = frozenset('hyprl-trader-' + suffix + '.service' for suffix in (
    'run-pm', 'recover-pm', 'rehearsal-pm', 'rehearsal-pm-retry', 'rehearsal-pm-failure'))
PM_TIMERS = frozenset('hyprl-trader-' + suffix + '.timer' for suffix in (
    'run-pm', 'recover-pm', 'rehearsal-pm', 'rehearsal-pm-retry'))


def check_decision(path):
    if sha256_canonical(strict_json(Path(path).read_text())) != DECISION_HASH:
        raise TraderError('PM_RETIREMENT_OPERATOR_DECISION_HASH_MISMATCH')


def spec():
    from .slots import SPEC_HASH as previous
    value = strict_json((ROOT / 'trader_pm_slot_spec_v3.json').read_text())
    if value.pop('canonical_sha256') != SPEC_HASH or sha256_canonical(value) != SPEC_HASH:
        raise TraderError('PM_RETIREMENT_HASH_MISMATCH')
    check_decision(ROOT / DECISION_NAME)
    if (value['previous_spec_hash'] != previous or value['operator_decision_hash'] != DECISION_HASH
            or value['effective_from'] != EFFECTIVE_FROM):
        raise TraderError('PM_RETIREMENT_BINDING_MISMATCH')
    return value


def active(at):
    return at >= instant(spec()['effective_from'])


def expected(grant, at):
    return (not active(at) and bool(grant.pm_amendment)
            and at >= instant(grant.pm_amendment['effective_from']))
