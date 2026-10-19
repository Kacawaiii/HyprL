"""Inherited probe evidence and the opaque NOT RUN closure of probe 4."""
import os
from pathlib import Path

from scripts.trading_lab.replay_duel.audit import evidence as inherited
from scripts.trading_lab.sources.canonical import sha256_canonical


def evidence(root):
    from . import probe4
    value = inherited(root)
    directory = Path(os.environ.get('REPLAY_PROBE_ROOT', str(Path(root) / 'memory-probe')))
    fourth = probe4.evidence(directory)
    return dict(value, probe4=fourth, test_end=fourth['test_end'],
                evidence_hash=sha256_canonical([value['evidence_hash'], fourth]),
                limitations=value['limitations'] + ' Probe 4 closed NOT RUN, no answers read or graded; both test windows cut to 2026-09-30.')
