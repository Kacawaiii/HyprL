"""Additive news/safety contracts; existing execution stores are never migrated."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError

DECISION_NAME = 'trader-operator-decisions-2026-10-11.json'
DECISION_HASH = '101b2cabfe50ab9e00f86c0db3f50ff7b31083a5a0007734dbf62f4c8c466004'
SPEC_HASH = 'dab1476046f1be8d8f8ceeb9ab8757558a83b27c9c6844a9817f8eb8a1635b16'
PREREG_HASH = 'eb39435cafa050c497699223c2de80c5d0d6cce44673eb9b55ca4b489b911ec1'
ARTIFACTS = Path(__file__).resolve().parents[3] / 'docs/artifacts'


def bound_artifact(name, expected):
    value = json.loads((ARTIFACTS / name).read_text())
    if value.pop('canonical_sha256') != expected or sha256_canonical(value) != expected:
        raise TraderError('NEWS_SPEC_BINDING_MISMATCH')
    return value


def spec():
    return bound_artifact('trader_news_ab_spec_v2.json', SPEC_HASH)


def preregistration():
    value = bound_artifact('trader_agent_preregistration_v6.json', PREREG_HASH)
    if value['news_spec_hash'] != SPEC_HASH or value['operator_decision_hash'] != DECISION_HASH:
        raise TraderError('NEWS_SPEC_BINDING_MISMATCH')
    spec()
    return PREREG_HASH


def registration_target(grant):
    return {'news_spec_hash': SPEC_HASH, 'news_preregistration_hash': preregistration(),
            'news_operator_decision_hash': DECISION_HASH, 'grant_hash': grant.identity}
