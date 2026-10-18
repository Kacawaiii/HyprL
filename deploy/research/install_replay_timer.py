"""Render/install only this research user timer; never touch trader units or user bin tools."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--claude', type=Path, required=True)
    args = parser.parse_args(argv)
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
    for file in (service, timer):
        if file.exists():
            raise SystemExit('Research unit already exists; inspect before replacing it')
    os.umask(0o077)
    service.write_text(text)
    timer.write_bytes((root / 'deploy/research/replay-bench.timer').read_bytes())
    subprocess.run(['systemd-analyze', '--user', 'verify', str(service), str(timer)], check=True)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', timer.name], check=True)
    print('Installed research timer at source SHA ' + sha)


if __name__ == '__main__':
    main()
