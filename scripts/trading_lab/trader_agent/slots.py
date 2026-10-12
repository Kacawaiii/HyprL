"""Pinned additive pm contract; legacy am grant and execution bindings stay frozen."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError, instant, strict_json

DECISION_NAME = 'trader-operator-decisions-2026-10-12.json'
DECISION_HASH = '530c2d6a5d6e9e3b16cda77ad3060ffb8b827a7c269eec17ce4439495d8439a3'
AMENDMENT_NAME = 'trader-agent-pm-slot-amendment-v1.json'
AMENDMENT_HASH = '0f0ae85b2becb7e95c5237250215c6c5ec106cef5d260a92c479210ba9f6f61e'
SPEC_HASH = '3866c2cfffac99991312044c088ff478ede0c8e53e2d0d43e09e2fffaa2e6c30'
PREREG_HASH = 'eef176adfdd071c3bc1edb7f73439bf9e4b5c414afa8d24279e7599d884c75bb'
ROOT = Path(__file__).resolve().parents[3] / 'docs/artifacts'


def pinned(name, expected):
    value = strict_json((ROOT / name).read_text())
    if value.pop('canonical_sha256') != expected or sha256_canonical(value) != expected:
        raise TraderError('PM_REGISTRATION_HASH_MISMATCH')
    return value


def spec():
    from .paper_spec import SPEC_HASH as base
    value = pinned('trader_pm_slot_spec_v1.json', SPEC_HASH)
    if value['base_execution_spec_hash'] != base or value['operator_decision_hash'] != DECISION_HASH:
        raise TraderError('PM_REGISTRATION_BINDING_MISMATCH')
    return value


def preregistration():
    value = pinned('trader_agent_preregistration_v8.json', PREREG_HASH)
    if value['pm_slot_spec_hash'] != SPEC_HASH or value['amendment_hash'] != AMENDMENT_HASH:
        raise TraderError('PM_REGISTRATION_BINDING_MISMATCH')
    spec()
    return PREREG_HASH


def load_amendment(grant, folder):
    path = folder / AMENDMENT_NAME
    if not path.exists():
        return grant
    value = strict_json(path.read_text())
    if value.pop('canonical_sha256') != AMENDMENT_HASH or sha256_canonical(value) != AMENDMENT_HASH:
        raise TraderError('PM_AMENDMENT_HASH_MISMATCH')
    decision = strict_json((folder / DECISION_NAME).read_text())
    if (sha256_canonical(decision) != DECISION_HASH or value['operator_decision_hash'] != DECISION_HASH
            or value['base_grant_hash'] != (grant.base_identity or grant.identity)
            or value['extends'] != 'trader-agent-v1' or value['per_slot'] != spec()['budget']
            or instant(value['not_after']) > instant(grant.payload['not_after'])):
        raise TraderError('PM_AMENDMENT_BINDING_MISMATCH')
    preregistration()
    # Separate capability identity: changing it never implicitly rebinds old stores.
    return replace(grant, pm_amendment=value, pm_identity=AMENDMENT_HASH)


def slot_of(payload):
    return payload.get('slot', 'pm' if ':pm' in payload.get('run_id', '') else 'am')


def run_id(day, synthetic, slot='am'):
    return 'trader:' + day + (':synthetic' if synthetic else ':real') + (':pm' if slot == 'pm' else '')


def start_at(day, slot):
    if slot not in {'am', 'pm'}:
        raise TraderError('INVALID_SLOT')
    return instant(day + ('T16:00:00Z' if slot == 'pm' else 'T12:00:00Z'))


def label_window(asset, day, horizon, crypto_assets, *, anchor=None):
    from .data import calendar_session, equity_calendar
    n = 1 if horizon == '1d' else 5
    entry = anchor or instant(day + 'T17:30:00Z')
    if asset in crypto_assets:
        return entry, entry + timedelta(days=n)
    session = calendar_session(day)
    if not session:
        raise TraderError('HOLIDAY')
    entry = entry if anchor else min(entry, session.close_at - timedelta(minutes=30))
    sessions = equity_calendar(entry.year).sessions_between(session.open_at, entry + timedelta(days=30))
    return entry, sessions[n].close_at


def deadline(day, crypto_only=False):
    from .data import calendar_session
    session = calendar_session(day)
    anchor = instant(day + 'T17:30:00Z')
    return min(anchor, session.close_at - timedelta(minutes=30)) if session and not crypto_only else anchor


def collision_policy(grant, at):
    return bool(grant.pm_amendment and instant(grant.pm_amendment['effective_from']) <= at)


def registration_target(paper_grant):
    return {'pm_spec_hash': SPEC_HASH, 'pm_preregistration_hash': preregistration(),
            'amendment_hash': paper_grant.parent.pm_identity, 'operator_decision_hash': DECISION_HASH,
            'paper_grant_hash': paper_grant.identity, 'effective_grant_hash': paper_grant.parent.identity}


def recovery_deadline(day, crypto_only=False):
    from .data import calendar_session
    session = calendar_session(day)
    return session.close_at - timedelta(minutes=15) if session and not crypto_only else instant(day + 'T23:30:00Z')


def recovery_anchor(day, at, crypto_only=False):
    candidate = (at + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0)
    return min(candidate, recovery_deadline(day, crypto_only))


def variant(crypto_only=False, recovery=False):
    return ('pm_weekend_crypto' if crypto_only else 'pm_slot') + ('_recovery_v1' if recovery else '_v1')
