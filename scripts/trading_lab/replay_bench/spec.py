"""Pinned replay spec: the artifact, its canonical hash and the prompt/skill files it binds."""
import hashlib
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

from .grant import ReplayError

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SKILLS = ROOT / 'scripts/trading_lab/trader_agent/skills'
SPEC_FILE = ROOT / 'docs/artifacts/replay_bench_spec_v3.json'
SPEC_HASH = '80a8a4dff0e07baebac5378972d60347a51c42ca0a1ba196d07ed167b129ecaa'
# Revision 1 served the main probe only; it stays on disk and is never loaded for a decision.
SUPERSEDED = {1: 'f96cbe1fad63cc68f22b190a2a5eacca4d86dc1bf59afc256aa802d80811980f',
              2: '95c2ed012a9954a6c4b08309ab32ff1e87217ee2af466c62854d8751b0f3cc9f'}
PROVENANCE = 'REPLAY_POST_CUTOFF'
# Supplementary probe (declared after the main probe, before any replay decision): 12 pre-cutoff control events
# (2025-12, 2026-03) and a forced-guess pass over all 48 events. It can only make the floor more conservative.
CONTROL_SET_SHA256 = 'e981166a35f71abe5696e6230467b49f94c043d7c93ce25f90fc7e7bb94dcacb'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec():
    value = json.loads(SPEC_FILE.read_text())
    expected = value.pop('canonical_sha256')
    if expected != SPEC_HASH or sha256_canonical(value) != expected:
        raise ReplayError('SPEC_HASH_MISMATCH')
    if value['supersedes']['canonical_sha256'] != SUPERSEDED[2]:
        raise ReplayError('SPEC_LINEAGE_MISMATCH')
    variants = value['variants']
    for name, digest in variants['a_v1_as_live']['skills'].items():
        if file_hash(SKILLS / name) != digest:
            raise ReplayError('SKILL_HASH_MISMATCH', name)
    for name, digest in variants['c_news_small_mid']['skill'].items():
        if file_hash(HERE / 'prompts' / name) != digest:
            raise ReplayError('SKILL_HASH_MISMATCH', name)
    for name, digest in variants['a_v1_as_live']['preamble'].items():
        if file_hash(HERE / 'prompts' / name) != digest:
            raise ReplayError('PREAMBLE_HASH_MISMATCH', name)
    return value


def text(name):
    if name in ('TRADER_SKILL_V2.md', 'REVIEWER_SKILL_V2.md'):
        return (SKILLS / name).read_text()
    return (HERE / 'prompts' / name).read_text()
