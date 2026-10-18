"""Replace only the existing replay research timer, with backup and exact source binding."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--environment', required=True, type=Path)
    p.add_argument('--claude', required=True, type=Path)
    p.add_argument('--codex', required=True, type=Path)
    p.add_argument('--backup-dir', required=True, type=Path)
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if not args.environment.is_file() or args.environment.stat().st_mode & 0o077:
        raise SystemExit('Research environment must be a private file (600)')
    for cli in (args.claude, args.codex):
        if not cli.is_file() or not os.access(cli, os.X_OK):
            raise SystemExit('Both subscription CLIs must be executable')
    if args.claude.absolute().parent != args.codex.absolute().parent:
        raise SystemExit('Research CLIs must share the same executable directory')
    version = subprocess.check_output([str(args.codex), '--version'], text=True).strip()
    if version != 'codex-cli 0.160.0':
        raise SystemExit('Unverified Codex CLI version; revise and test the protocol first')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Commit and verify source before installing the timer')
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    target = Path.home() / '.config/systemd/user'
    service, timer = target / 'hyprl-replay-bench.service', target / 'hyprl-replay-bench.timer'
    if not all(f.is_file() and not f.is_symlink() for f in (service, timer)):
        raise SystemExit('Expected two ordinary replay research unit files')
    if not any('-m scripts.trading_lab.' + name + '.cli night --expected-commit ' in service.read_text()
               for name in ('replay_bench', 'replay_duel')) or 'Unit=hyprl-replay-bench.service' not in timer.read_text():
        raise SystemExit('Refusing to replace unrelated units')
    state = subprocess.check_output(['systemctl', '--user', 'show', service.name, '-p', 'ActiveState', '--value'], text=True).strip()
    if state not in ('inactive', 'failed'):
        raise SystemExit('Research replay is active; do not replace its units')
    backup = args.backup_dir.expanduser().resolve()
    if backup.exists() or backup == root or root in backup.parents:
        raise SystemExit('Backup must be a new directory outside Git')
    text = (root / 'deploy/research/replay-duel.service.in').read_text()
    for key, value in {'WORKTREE': root, 'ENVIRONMENT': args.environment.resolve(), 'PYTHON': Path(sys.executable),
                       'CLI_PATH': args.claude.absolute().parent, 'COMMIT': sha}.items():
        if any(c in str(value) for c in '\n\r%" '):
            raise SystemExit('Unsafe systemd path')
        text = text.replace('@' + key + '@', str(value))
    os.umask(0o077)
    with tempfile.TemporaryDirectory(prefix='replay-duel-', dir=target) as stage:
        staged_service, staged_timer = Path(stage) / service.name, Path(stage) / timer.name
        staged_service.write_text(text)
        staged_timer.write_bytes((root / 'deploy/research/replay-bench.timer').read_bytes())
        subprocess.run(['systemd-analyze', '--user', 'verify', str(staged_service), str(staged_timer)], check=True)
        backup.mkdir(parents=True, mode=0o700)
        for file in (service, timer):
            shutil.copy2(file, backup / file.name)
        os.replace(staged_service, service)
        os.replace(staged_timer, timer)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', timer.name], check=True)
    print('Installed research replay duel at source SHA ' + sha)


if __name__ == '__main__':
    main()
