"""A real hash-only ledger in a temporary Git repository with a bare remote, and a fake notifier."""
import os
from pathlib import Path
import subprocess

import pytest

from scripts.trading_lab.replay_seal_forward import commitment


def git(*args, cwd=None):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def ledger(tmp_path):
    remote, work = tmp_path / 'ledger-remote.git', tmp_path / 'ledger-work'
    git('init', '-q', '--bare', str(remote))
    git('init', '-q', '-b', commitment.BRANCH, str(work))
    git('config', 'user.email', 'ledger@example.invalid', cwd=work)
    git('config', 'user.name', 'synthetic ledger', cwd=work)
    (work / 'README.md').write_text('hash-only ledger\n')
    git('add', 'README.md', cwd=work)
    git('commit', '-q', '-m', 'genesis', cwd=work)
    git('remote', 'add', 'origin', str(remote), cwd=work)
    git('push', '-q', 'origin', commitment.BRANCH, cwd=work)
    notify = tmp_path / 'notify'
    log = tmp_path / 'notify.log'
    notify.write_text('#!/bin/sh\ncat >> ' + str(log) + '\necho "--" >> ' + str(log) + '\necho discord\n')
    os.chmod(notify, 0o700)
    value = commitment.Ledger(tmp_path, work, notify)
    value.remote, value.log = remote, log
    return value
