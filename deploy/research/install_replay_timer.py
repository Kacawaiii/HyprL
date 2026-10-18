"""Render/install only this research user timer; never touch trader units or user bin tools."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--claude', type=Path, required=True)
    parser.add_argument('--replace', action='store_true', help='Replace this research timer after inspection and backup')
    parser.add_argument('--backup-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.environment.is_file() or args.environment.stat().st_mode & 0o077:
        raise SystemExit('A private research environment file is required (mode 600)')
    if not args.claude.is_file() or not os.access(args.claude, os.X_OK):
        raise SystemExit('Claude CLI executable is missing')
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Commit and verify the source before installing the timer')
    replacements = {'WORKTREE': root, 'ENVIRONMENT': args.environment.resolve(), 'PYTHON': Path(sys.executable),
                    'CLI_PATH': args.claude.absolute().parent, 'COMMIT': sha}
    text = (root / 'deploy/research/replay-bench.service.in').read_text()
    for name, value in replacements.items():
        if any(c in str(value) for c in '\n\r%"') or ' ' in str(value):
            raise SystemExit('Unsafe systemd path')
        text = text.replace('@' + name + '@', str(value))
    target = Path.home() / '.config/systemd/user'
    target.mkdir(parents=True, exist_ok=True)
    service = target / 'hyprl-replay-bench.service'
    timer = target / 'hyprl-replay-bench.timer'
    existing = service.exists() or timer.exists()
    if existing:
        if not args.replace or not args.backup_dir:
            raise SystemExit('Research unit already exists; inspect before replacing it with --replace --backup-dir')
        if not service.is_file() or not timer.is_file() or service.is_symlink() or timer.is_symlink():
            raise SystemExit('Both ordinary research unit files are required for replacement')
        if ('-m scripts.trading_lab.replay_bench.cli night --expected-commit ' not in service.read_text() or
                'Unit=hyprl-replay-bench.service' not in timer.read_text()):
            raise SystemExit('Existing units are not the expected research replay units')
        state = subprocess.check_output(['systemctl', '--user', 'show', service.name,
                                         '-p', 'ActiveState', '--value'], text=True).strip()
        if state not in ('inactive', 'failed'):
            raise SystemExit('Research service is running; do not replace its units')
        backup = args.backup_dir.expanduser().resolve()
        if backup == root or root in backup.parents or backup.exists():
            raise SystemExit('A new backup directory outside Git is required')
    elif args.replace:
        raise SystemExit('No research timer exists to replace')
    os.umask(0o077)
    with tempfile.TemporaryDirectory(prefix='replay-units-', dir=target) as stage:
        staged_service, staged_timer = Path(stage) / service.name, Path(stage) / timer.name
        staged_service.write_text(text)
        staged_timer.write_bytes((root / 'deploy/research/replay-bench.timer').read_bytes())
        subprocess.run(['systemd-analyze', '--user', 'verify', str(staged_service), str(staged_timer)], check=True)
        if existing:
            backup.mkdir(parents=True, mode=0o700)
            for file in (service, timer):
                shutil.copy2(file, backup / file.name)
        os.replace(staged_service, service)
        os.replace(staged_timer, timer)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', timer.name], check=True)
    print('Installed research timer at source SHA ' + sha)


if __name__ == '__main__':
    main()
