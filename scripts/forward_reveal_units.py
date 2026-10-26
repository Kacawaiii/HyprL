"""Stage only the two new research reveal user units; never install or start them."""
import argparse
from pathlib import Path
import re


def units(checkout, environment, python, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Expected a full commit SHA')
    for value in (checkout, environment, python):
        if not Path(value).is_absolute() or re.search(r'[\s%"\\]', str(value)):
            raise ValueError('Unsafe unit path')
    service = '\n'.join([
        '[Unit]', 'Description=HyprL research forward post-close reveals', '', '[Service]',
        'Type=exec', 'WorkingDirectory=' + str(checkout), 'EnvironmentFile=' + str(environment),
        'ExecStart=' + str(python) + ' -m scripts.forward_reveal --expected-commit ' + commit + ' --publish',
        'UMask=0077', 'NoNewPrivileges=yes', 'Nice=15', 'CPUQuota=50%', 'MemoryMax=300M',
        'RuntimeMaxSec=900', 'TimeoutStopSec=20', 'KillMode=control-group', 'Restart=no', '',
    ])
    timer = '\n'.join([
        '[Unit]', 'Description=HyprL research post-close reveal schedule', '', '[Timer]',
        'OnCalendar=Mon..Fri *-*-* 16:15:00 America/New_York', 'AccuracySec=1s', 'Persistent=true',
        'Unit=hyprl-research-forward-reveal.service', '', '[Install]', 'WantedBy=timers.target', '',
    ])
    return {'hyprl-research-forward-reveal.service': service, 'hyprl-research-forward-reveal.timer': timer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout', 'environment', 'python', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    rendered = units(args.checkout, args.environment, args.python, args.expected_commit)
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Stage outside the source checkout')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in rendered.items():
        (output / name).write_text(content)


if __name__ == '__main__':
    main()
