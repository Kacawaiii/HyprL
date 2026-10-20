"""Install only the replay/forward research user units, from a clean pinned commit."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def refresh_timer_text(text):
    if '[Timer]' not in text or 'OnCalendar=*-*-* 00:20:00 UTC' not in text:
        raise SystemExit('Unrecognized research login refresh timer')
    schedule = 'OnCalendar=*-*-* 04:00:00 UTC'
    return text if schedule in text.splitlines() else text.replace('[Timer]\n', '[Timer]\n' + schedule + '\n', 1)


def units(root, environment, cli_directory, commit):
    commands = {'hyprl-replay-bench': 'night', 'hyprl-research-forward': 'forward',
                'hyprl-research-forward-score': 'forward-score', 'hyprl-research-forward-weekly': 'weekly',
                'hyprl-research-forward-publish': 'forward-publish',
                'hyprl-research-forward-drill': 'forward-dry-run',
                'hyprl-research-collect-live': 'live', 'hyprl-research-collect-outcomes': 'live-outcomes'}
    calendars = {'night': '*-*-* *:00/15:00 UTC', 'forward': 'Mon..Fri *-*-* 09:05:00 America/New_York',
                 'forward-score': 'Mon..Fri *-*-* 16:45:00 America/New_York',
                 'weekly': 'Fri *-*-* 17:15:00 America/New_York',
                 'forward-publish': 'Mon..Fri *-*-* 09:20:00 America/New_York',
                 'forward-dry-run': 'Mon..Fri *-*-* 22:45:00 UTC',
                 'live': 'Mon..Fri *-*-* 04..08:00/5:00 America/New_York\nOnCalendar=Mon..Fri *-*-* 09:00,03:00 America/New_York',
                 'live-outcomes': 'Mon..Fri *-*-* 16:35:00 America/New_York'}
    values = {}
    for name, command in commands.items():
        module = 'collect' if command in ('live', 'live-outcomes', 'forward-dry-run') else 'cli'
        service = '\n'.join(['[Unit]', 'Description=HyprL research seal/forward ' + command,
            *(['After=hyprl-drill-nightly.service'] if command == 'forward-dry-run' else []), '', '[Service]',
            'Type=exec', 'WorkingDirectory=' + str(root), 'EnvironmentFile=' + str(environment),
            'Environment=PATH=' + str(cli_directory) + ':/usr/bin:/bin',
            *(['Environment=REPLAY_NIGHT_REPORT=' + str(Path.home() / 'reports' / 'codex-replay-night-fixes.md')]
              if command == 'night' else []),
            'ExecStart=' + sys.executable + ' -m scripts.trading_lab.replay_seal_forward.' + module + ' ' + command + ' --expected-commit ' + commit,
            'UMask=0077', 'NoNewPrivileges=yes', 'Nice=15', 'CPUQuota=50%',
            'MemoryMax=' + ('1100M' if command == 'night' else '700M'),
            'RuntimeMaxSec=' + ('infinity' if command == 'night' else '900'), 'TimeoutStopSec=20',
            'KillMode=control-group', 'Restart=no', 'SuccessExitStatus=2', ''])
        timer = '\n'.join(['[Unit]', 'Description=HyprL research ' + command, '', '[Timer]',
            'OnCalendar=' + calendars[command], 'AccuracySec=1s', 'Persistent=false', 'Unit=' + name + '.service',
            '', '[Install]', 'WantedBy=timers.target', ''])
        values[name + '.service'], values[name + '.timer'] = service, timer
    return values


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--claude', type=Path, required=True)
    parser.add_argument('--codex', type=Path, required=True)
    parser.add_argument('--backup-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if not args.environment.is_file() or args.environment.stat().st_mode & 0o077:
        raise SystemExit('Research environment must be a private file (600)')
    for cli in (args.claude, args.codex):
        if not cli.is_file() or not os.access(cli, os.X_OK):
            raise SystemExit('Subscription CLI missing')
    if args.claude.absolute().parent != args.codex.absolute().parent:
        raise SystemExit('CLIs must share one directory')
    if subprocess.check_output([str(args.codex), '--version'], text=True).strip() != 'codex-cli 0.160.0':
        raise SystemExit('Unverified Codex CLI')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Commit and verify source before installing')
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    for value in (root, args.environment.resolve(), args.claude.absolute().parent, Path(sys.executable)):
        if any(c in str(value) for c in '\n\r%" '):
            raise SystemExit('Unsafe systemd path')
    target = Path.home() / '.config/systemd/user'
    rendered = units(root, args.environment.resolve(), args.claude.absolute().parent, sha)
    refresh = target / 'hyprl-claude-token-refresh.timer'
    if refresh.exists():
        if refresh.is_symlink():
            raise SystemExit('Refusing a symlink refresh timer')
        rendered[refresh.name] = refresh_timer_text(refresh.read_text())
    backup = args.backup_dir.resolve()
    if backup.exists() or backup == root or root in backup.parents:
        raise SystemExit('Backup must be a new directory outside Git')
    existing = []
    for name in rendered:
        file = target / name
        if file.is_symlink():
            raise SystemExit('Refusing a symlink unit')
        if file.exists():
            text = file.read_text()
            if name == 'hyprl-replay-bench.service':
                expected = any('-m scripts.trading_lab.' + protocol + '.cli night --expected-commit ' in text
                               for protocol in ('replay_bench', 'replay_duel', 'replay_seal_forward'))
            elif name.endswith('.service'):
                command = name.removesuffix('.service').removeprefix('hyprl-research-')
                command = {'forward-weekly': 'weekly', 'collect-live': 'live', 'collect-outcomes': 'live-outcomes',
                           'forward-drill': 'forward-dry-run'}.get(command, command)
                module = 'collect' if command in ('live', 'live-outcomes', 'forward-dry-run') else 'cli'
                expected = '-m scripts.trading_lab.replay_seal_forward.' + module + ' ' + command + ' --expected-commit ' in text
            elif name == refresh.name:
                expected = '[Timer]' in text and 'OnCalendar=*-*-* 00:20:00 UTC' in text
            else:
                expected = 'Unit=' + name.removesuffix('.timer') + '.service' in text
            if not expected:
                raise SystemExit('Refusing unrelated unit')
            existing.append(file)
        if name.endswith('.service'):
            state = subprocess.check_output(['systemctl', '--user', 'show', name, '-p', 'ActiveState', '--value'], text=True).strip()
            if state not in ('', 'inactive', 'failed'):
                raise SystemExit('Research service active')
    os.umask(0o077)
    with tempfile.TemporaryDirectory(prefix='seal-forward-', dir=target) as stage:
        files = []
        for name, text in rendered.items():
            file = Path(stage) / name
            file.write_text(text)
            files.append(file)
        subprocess.run(['systemd-analyze', '--user', 'verify', *map(str, files)], check=True)
        backup.mkdir(parents=True, mode=0o700)
        for file in existing:
            shutil.copy2(file, backup / file.name)
        for file in files:
            os.replace(file, target / file.name)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    for name in rendered:
        if name.endswith('.timer'):
            subprocess.run(['systemctl', '--user', 'enable', '--now', name], check=True)
    print('Installed only research timers at ' + sha)


if __name__ == '__main__':
    main()
