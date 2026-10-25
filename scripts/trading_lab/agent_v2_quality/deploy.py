"""Reviewable user units and grant-bound activation, never an automatic installer."""
from pathlib import Path
import os
import sys

from scripts.trading_lab.agent_v2.deploy import deploy_window, quote
from .contracts import Blocked, digest, instant, utc
from .spec import PREREG_HASH, SPEC_HASH


def activate(journal, grant, proof, *, clock=utc):
    at = clock()
    deploy_window(at)
    authorization = grant.check()
    required = {'schema': 'agent-v2-quality-deploy-proof-v2', 'sha': journal.binding['code_sha'],
                'spec_hash': SPEC_HASH, 'preregistration_hash': PREREG_HASH,
                'authorization_sha256': digest(authorization), 'ci_conclusion': 'success',
                'local_checks': 'PASS', 'single_owner_reviewed': True, 'replay_waived_by_grant': True}
    if any(proof.get(k) != v for k, v in required.items()):
        raise Blocked('DEPLOY_EVIDENCE_REQUIRED')
    if not proof.get('ci_url', '').startswith('https://github.com/') or not proof.get('producer_snapshot_hash'):
        raise Blocked('DEPLOY_EVIDENCE_REQUIRED')
    receipt = next((r for r in reversed(journal.records('producer_snapshot'))
                    if r['body']['hash'] == proof['producer_snapshot_hash']), None)
    if receipt is None or not 0 <= (at - instant(receipt['at'])).total_seconds() <= 86400:
        raise Blocked('RECENT_PRODUCER_PROOF_REQUIRED')
    if journal.binding['mode'] != 'SHADOW_FORWARD':
        raise Blocked('MODE_MISMATCH')
    with journal.owned():
        if journal.records('activation'):
            raise Blocked('ALREADY_ACTIVATED')
        journal.append([('activation', {'proof': proof, 'proof_hash': digest(proof),
                                      'authorization_sha256': digest(authorization), 'broker_orders': 0})], at)
    return {'status': 'ACTIVATED_SHADOW_ONLY', 'broker_orders': 0}


def render(repo, python, runtime, config):
    output = {}
    for action, schedule in (
            ('cycle', 'Mon..Fri *-*-* 12:45:00 UTC'),
            ('watch', 'Mon..Fri *-*-* 09..16:*:15 America/New_York'),
            ('weekly', 'Fri *-*-* 21:15:00 UTC'),
            ('drill', '*-*-* 22:45:00 UTC')):
        stem = 'hyprl-agent-v2-quality-' + action
        argv = [python, '-m', 'scripts.trading_lab.agent_v2_quality.cli', action,
                '--runtime', runtime, '--config', config]
        output[stem + '.service'] = ('[Unit]\nDescription=HyprL v2 quality shadow ' + action + '\n'
            '[Service]\nType=oneshot\nEnvironment=' + quote('PATH=' + os.environ['PATH']) +
            '\nWorkingDirectory=' + str(repo) + '\nExecStart=' +
            ' '.join(quote(a) for a in argv) + '\nUMask=0077\nNice=15\nMemoryMax=350M\nCPUQuota=35%\n'
            'NoNewPrivileges=yes\nTimeoutStartSec=15min\nKillMode=control-group\n')
        output[stem + '.timer'] = ('[Unit]\nDescription=HyprL v2 quality shadow ' + action + '\n'
            '[Timer]\nOnCalendar=' + schedule + '\nPersistent=false\nAccuracySec=1s\n'
            '[Install]\nWantedBy=timers.target\n')
    return output


def write_units(directory, repo, runtime, config):
    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name, value in render(repo, sys.executable, runtime, config).items():
        (directory / name).write_text(value)
    return {'status': 'RENDERED_DISABLED', 'units': 8, 'broker_orders': 0}
