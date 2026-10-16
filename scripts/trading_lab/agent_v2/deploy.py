"""Render isolated user units; activation requires exact-SHA CI and v2 input/model grants."""
from datetime import time
from pathlib import Path
import argparse
import subprocess
import sys

from .contracts import Blocked, instant, strict_json, utc
from .ledger import Ledger
from .models import Claude
from .spec import PREREG_HASH


def deploy_window(at):
    t = at.time().replace(tzinfo=None)
    if any(start <= t < end for start, end in ((time(11,30), time(13,50)), (time(15), time(16,30)),
                                                (time(19,45), time(20,10)))):
        raise Blocked('PROTECTED_DEPLOY_WINDOW')


def quote(value):
    text = str(value)
    if any(c in text for c in ('\n', '\r', '%')):
        raise Blocked('INVALID_UNIT_PATH')
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def render(repo, python, runtime, authorization):
    output = {}
    for action, calendar, input_name in (
            ('decide', 'Mon..Fri *-*-* 09:40:00 America/New_York\nOnCalendar=Mon..Fri *-*-* 12:40:00 America/New_York\nOnCalendar=Mon..Fri *-*-* 15:40:00 America/New_York', 'snapshot.json'),
            ('watch', 'Mon..Fri *-*-* 09..16:*:30 America/New_York', 'quotes.json'),
            ('reconcile', 'Mon..Fri *-*-* 16:05:00 America/New_York', None)):
        stem = 'hyprl-agent-v2-shadow-' + action
        args = [python, '-m', 'scripts.trading_lab.agent_v2.cli', action, '--runtime', runtime]
        if input_name:
            args += ['--inbox', Path(runtime) / 'inbox' / input_name, '--authorization', authorization]
        output[stem + '.service'] = ('[Unit]\nDescription=HyprL isolated v2 shadow ' + action + '\n'
            '[Service]\nType=oneshot\nWorkingDirectory=' + str(repo) + '\nExecStart=' + ' '.join(quote(v) for v in args) +
            '\nUMask=0077\nNice=15\nMemoryMax=350M\nCPUQuota=40%\nNoNewPrivileges=yes\nTimeoutStartSec=7min\n'
            'KillMode=control-group\n')
        output[stem + '.timer'] = ('[Unit]\nDescription=HyprL isolated v2 shadow ' + action + '\n'
            '[Timer]\nOnCalendar=' + calendar + '\nAccuracySec=1s\nPersistent=false\n'
            '[Install]\nWantedBy=timers.target\n')
    return output


def check_activation(ledger, authorization, evidence, at):
    deploy_window(at)
    Claude(ledger, authorization, clock=lambda: at).grant()
    proof = strict_json(Path(evidence).read_text())
    if (proof.get('schema') != 'agent-v2-deploy-proof-v1' or proof.get('preregistration_hash') != PREREG_HASH
            or proof.get('sha') != ledger.binding['code_sha'] or proof.get('ci_conclusion') != 'success'
            or proof.get('local_checks') != 'PASS' or proof.get('integrated_fast_forward') is not True
            or not proof.get('replay_report_sha256') or not proof.get('input_grant_sha256')
            or proof.get('forward_inputs_ready') is not True):
        raise Blocked('DEPLOY_EVIDENCE_REQUIRED')
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='render only; never modifies an existing user unit')
    args = parser.parse_args(argv)
    try:
        texts = render(Path(__file__).resolve().parents[3], sys.executable, args.runtime.resolve(), args.authorization.resolve())
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        for name, body in texts.items():
            (args.output / name).write_text(body)
        print('RENDERED_DISABLED: activation needs verified CI, integration, replay report and dedicated v2 grants')
        return 0
    except (Blocked, OSError) as error:
        print('BLOCKED: ' + str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
