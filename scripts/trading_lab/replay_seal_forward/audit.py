"""Inherited probe evidence plus the preregistered October 1-16 probe 4 and its test cuts."""
import os
from pathlib import Path

from scripts.trading_lab.replay_duel.audit import evidence as inherited
from scripts.trading_lab.sources.canonical import sha256_canonical


def evidence(root):
    from . import probe4
    value = inherited(root)
    directory = Path(os.environ.get('REPLAY_PROBE_ROOT', str(Path(root) / 'memory-probe')))
    fourth = probe4.evidence(directory)
    return dict(value, probe4=fourth,
                test_end={f: fourth['families'][f]['test_end'] for f in ('opus', 'gpt')},
                evidence_hash=sha256_canonical([value['evidence_hash'], fourth]),
                limitations=value['limitations'] + ' Probe 4 (2026-10-01..16) cuts a flagged model at 2026-09-30.')
