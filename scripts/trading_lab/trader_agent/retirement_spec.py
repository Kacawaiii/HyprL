"""Forward-only GPT retirement; frozen grants and historical bindings stay intact."""
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError, instant, strict_json

DECISION_NAME = 'trader-operator-decisions-2026-10-16.json'
DECISION_HASH = '5136ce360c9672b685e783448d31d7c87e2c8b925343b58e978fc5f3dd6a947c'
SPEC_HASH = 'c9fa4f18638612cfc42861fa775d15e007a1ec221ea69f07aa216f0cd58f928f'
PREREG_HASH = '584d96095e6dac777d2bf67dde604017880aeaee843c2390b9352b9c8c1a8b21'
EFFECTIVE_FROM = '2026-10-16T00:00:00Z'
POPULATION = 'gpt_retired_v1'
RETIRED = frozenset({'analyst_gpt', 'reviewer_gpt', 'consensus', 'gpt_preflight'})
ROOT = Path(__file__).resolve().parents[3] / 'docs/artifacts'


def pinned(name, expected):
    value = strict_json((ROOT / name).read_text())
    if value.pop('canonical_sha256') != expected or sha256_canonical(value) != expected:
        raise TraderError('GPT_RETIREMENT_HASH_MISMATCH')
    return value


def spec():
    from .paper_spec import SPEC_HASH as previous
    value = pinned('trader_alpaca_execution_spec_v5.json', SPEC_HASH)
    if (value['previous_spec_hash'] != previous or value['operator_decision_hash'] != DECISION_HASH
            or value['effective_from'] != EFFECTIVE_FROM):
        raise TraderError('GPT_RETIREMENT_BINDING_MISMATCH')
    return value


def preregistration():
    value = pinned('trader_agent_preregistration_v10.json', PREREG_HASH)
    if (value['execution_spec_hash'] != SPEC_HASH or value['operator_decision_hash'] != DECISION_HASH
            or value['effective_from'] != EFFECTIVE_FROM):
        raise TraderError('GPT_RETIREMENT_BINDING_MISMATCH')
    spec()
    return PREREG_HASH


def active(at):
    return at >= instant(EFFECTIVE_FROM)


def check_dispatch(kind, at):
    if active(at) and kind.removeprefix('pm_').removeprefix('news_') in RETIRED:
        raise TraderError('GPT_RETIRED')


def roles(at):
    return ('analyst_claude',) if active(at) else ('analyst_claude', 'analyst_gpt')


def retired_model():
    return {'model': 'RETIRED', 'cli_version': 'RETIRED', 'reported_version': 'RETIRED'}


def lifecycle():
    preregistration()
    return {role: {'state': 'RETIRED' if role in RETIRED else 'ACTIVE',
                   'effective_from': EFFECTIVE_FROM, 'operator_decision_hash': DECISION_HASH}
            for role in ('analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt', 'consensus')}


def registration_target(ledger):
    return {'execution_spec_hash': SPEC_HASH, 'preregistration_hash': preregistration(),
            'operator_decision_hash': DECISION_HASH, 'paper_grant_hash': ledger.grant.identity,
            'effective_grant_hash': ledger.grant.parent.identity,
            'prior_binding_hash': sha256_canonical(ledger.events(event='binding')[-1])}
