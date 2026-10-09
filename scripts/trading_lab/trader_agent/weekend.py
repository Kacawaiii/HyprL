"""Separate, pinned closed-day crypto population; original paper binding stays frozen."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError, instant

VARIANT = 'weekend_crypto_v1'
ROOT = Path(__file__).resolve().parents[3] / 'docs/artifacts'
SPEC_HASH = '421c3f7d7f71e4fdde59eec1216d72edcf5b720ea64938a4d52edb6c4451d998'
PREREG_HASH = 'd38d91c8c822ddfaa478b102c22099aa00a750ec1957c83bb006ccd38df1d89c'


def pinned(name, expected):
    value = json.loads((ROOT / name).read_text())
    recorded = value.pop('canonical_sha256')
    if recorded != expected or sha256_canonical(value) != expected:
        raise TraderError('WEEKEND_REGISTRATION_HASH_MISMATCH')
    return value


def preregistration():
    from .service import PREREG_HASH as weekday_hash, skills
    from .paper_spec import SPEC_HASH as execution_hash
    value = pinned('trader_agent_preregistration_v3.json', PREREG_HASH)
    spec = pinned('trader_weekend_crypto_spec_v1.json', SPEC_HASH)
    if (value['previous_preregistration_hash'] != weekday_hash or value['spec_hash'] != SPEC_HASH
            or value['skill_hashes'] != skills()[1]
            or value['crypto_amendment_hash'] != spec['crypto_amendment_hash']
            or value['operator_decision_hash'] != spec['operator_decision_hash']
            or any(v['execution_spec_hash'] != execution_hash for v in (value, spec))):
        raise TraderError('WEEKEND_REGISTRATION_BINDING_MISMATCH')
    return PREREG_HASH


def check(grant, at):
    grant.check_weekend(at)
    preregistration()
    if at < instant(pinned('trader_weekend_crypto_spec_v1.json', SPEC_HASH)['effective_from']):
        raise TraderError('WEEKEND_CRYPTO_NOT_STARTED')


def variants():
    preregistration()
    return pinned('trader_agent_preregistration_v3.json', PREREG_HASH)['variants']


def registration_target(paper_grant):
    from .config import WEEKEND_DECISION_HASH
    from .paper_spec import SPEC_HASH as execution_hash
    return {'variant': VARIANT, 'weekend_spec_hash': SPEC_HASH, 'preregistration_hash': preregistration(),
            'operator_decision_hash': WEEKEND_DECISION_HASH, 'execution_spec_hash': execution_hash,
            'grant_hash': paper_grant.identity, 'effective_grant_hash': paper_grant.parent.identity,
            'crypto_amendment_hash': paper_grant.parent.amendment_identity}
