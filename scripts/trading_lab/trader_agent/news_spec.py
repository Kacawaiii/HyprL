"""Additive news/safety contracts; existing execution stores are never migrated."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError

DECISION_NAME = 'trader-operator-decisions-2026-10-11.json'
DECISION_HASH = '9e51bea823f5d29d4eafd183e79fa9668b33c152131988e7350cdf58554221e9'
SPEC_HASH = 'e677c1874c2d549e80bc8dcdf48f8302593407b0560f2c29ca803c5d66d89a0a'
PREREG_HASH = 'e34839a11adbb3ff7aae52f311c190a2200dc6df5c51233422619ed060909b53'
ARTIFACTS = Path(__file__).resolve().parents[3] / 'docs/artifacts'


def bound_artifact(name, expected):
    value = json.loads((ARTIFACTS / name).read_text())
    if value.pop('canonical_sha256') != expected or sha256_canonical(value) != expected:
        raise TraderError('NEWS_SPEC_BINDING_MISMATCH')
    return value


def spec():
    return bound_artifact('trader_news_ab_spec_v1.json', SPEC_HASH)


def preregistration():
    value = bound_artifact('trader_agent_preregistration_v5.json', PREREG_HASH)
    if value['news_spec_hash'] != SPEC_HASH or value['operator_decision_hash'] != DECISION_HASH:
        raise TraderError('NEWS_SPEC_BINDING_MISMATCH')
    spec()
    return PREREG_HASH


def registration_target(grant):
    return {'news_spec_hash': SPEC_HASH, 'news_preregistration_hash': preregistration(),
            'news_operator_decision_hash': DECISION_HASH, 'grant_hash': grant.identity}
