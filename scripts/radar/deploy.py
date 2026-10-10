"""Install only the radar's user units, after validating its live grant."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

from .core import Authorization, ROOT, RadarError, now


NAMES = ['hyprl-radar-collect', 'hyprl-radar-morning', 'hyprl-radar-evening']


def unit_texts(python, repo, *, claude=None):
    # systemd command parsing differs from a shell: quote arguments explicitly.
    def quote_arg(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    command = quote_arg(python) + ' -m scripts.radar.service '
    claude = Path(claude or 'claude').parent
    env_path = str(claude) + ':' + str(Path(python).parent) + ':%h/bin:/usr/local/bin:/usr/bin:/bin'
    result = {}
    schedules = {'collect': '*-*-* *:05:00 UTC',
                 'morning': 'Mon..Fri *-*-* 12:20:00 UTC\nOnCalendar=Sat,Sun *-*-* 11:30:00 UTC',
                 'evening': 'Mon..Fri *-*-* 21:20:00 UTC\nOnCalendar=Sat,Sun *-*-* 19:30:00 UTC'}
    for action, schedule in schedules.items():
        name = 'hyprl-radar-' + action
        arguments = 'collect' if action == 'collect' else 'run --slot ' + action + ' --publish-book'
        result[name + '.service'] = f'''[Unit]
Description=HyprL independent news radar ({action})

[Service]
Type=oneshot
WorkingDirectory={str(repo).replace('%', '%%')}
Environment="PATH={env_path}"
ExecStart={command}{arguments}
UMask=0077
Nice=15
MemoryMax=512M
TimeoutStartSec=20min
KillMode=process
NoNewPrivileges=true
'''
        result[name + '.timer'] = f'''[Unit]
Description=Schedule HyprL independent radar ({action})

[Timer]
OnCalendar={schedule}
AccuracySec=1s
RandomizedDelaySec=0
Persistent=false
Unit={name}.service

[Install]
WantedBy=timers.target
'''
    return result


def main():
    parser = argparse.ArgumentParser(description='Install radar user timers only; no trader changes')
    parser.add_argument('--authorization', type=Path, default=Path.home() / 'authorizations/news-radar-v1.json')
    parser.add_argument('--output', type=Path, help='Render units only into this directory, without installation')
    args = parser.parse_args()
    try:
        texts = unit_texts(sys.executable, ROOT, claude=shutil.which('claude'))
        if not args.output:
            grant = Authorization(args.authorization)
            grant.check(now())
        folder = args.output or Path.home() / '.config/systemd/user'
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in texts.items():
            (folder / name).write_text(content)
        subprocess.run(['systemd-analyze', '--user', 'verify', *[str(folder / (n + '.service')) for n in NAMES]], check=True, capture_output=True)
        if not args.output:
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, capture_output=True)
            subprocess.run(['systemctl', '--user', 'enable', '--now', *[n + '.timer' for n in NAMES]], check=True, capture_output=True)
        print('RENDERED' if args.output else 'RADAR_TIMERS_ENABLED')
        return 0
    except RadarError as error:
        print('BLOCKED: ' + str(error))
        return 2
    except (OSError, subprocess.CalledProcessError):
        print('BLOCKED: USER_SYSTEMD_UNAVAILABLE_OR_INVALID_UNIT')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
