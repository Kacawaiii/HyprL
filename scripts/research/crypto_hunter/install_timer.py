"""Install only the hunter's own user units; never modify existing book/trader units."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


MARKER = "# Managed by crypto_hunter.install_timer\n"
NAME = "hyprl-crypto-hunter"


def quote(value: Path | str) -> str:
    text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError("newlines forbidden in systemd arguments")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def render(args) -> tuple[str, str]:
    working_directory = str(args.worktree).replace("%", "%%")
    if "\n" in working_directory or "\r" in working_directory:
        raise ValueError("newlines forbidden in WorkingDirectory")
    cmd = [args.python, "-m", "scripts.research.crypto_hunter.scan"]
    for name in ("approved", "grant", "credentials", "state", "output"):
        cmd += ["--" + name, getattr(args, name)]
    if args.assets:
        cmd += ["--assets", args.assets]
    service = (MARKER + "[Unit]\nDescription=Read-only weekly crypto hunter report\n\n"
               "[Service]\nType=oneshot\nKillMode=process\nUMask=0077\nNice=10\n"
               "Environment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\n"
               f"WorkingDirectory={working_directory}\nExecStart={' '.join(quote(x) for x in cmd)}\n"
               "TimeoutStartSec=600\n")
    timer = (MARKER + "[Unit]\nDescription=Crypto hunter Saturday 10:00 UTC\n\n"
             "[Timer]\nOnCalendar=Sat *-*-* 10:00:00 UTC\nPersistent=true\nAccuracySec=1s\n"
             f"Unit={NAME}.service\n\n[Install]\nWantedBy=timers.target\n")
    return service, timer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "worktree", "approved", "grant", "credentials", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.worktree.resolve() != Path.cwd().resolve() or not args.python.is_file():
        raise ValueError("run installer from its own worktree with an existing interpreter")
    unitdir = Path.home() / ".config/systemd/user"
    unitdir.mkdir(parents=True, exist_ok=True)
    rendered = render(args)
    paths = [unitdir / f"{NAME}.{suffix}" for suffix in ("service", "timer")]
    # Check both before any mutation, so an unrelated existing unit remains untouched.
    for path in paths:
        if path.exists() and not path.read_text().startswith(MARKER):
            raise ValueError("unmanaged hunter unit exists; refusing overwrite")
    for path, body in zip(paths, rendered):
        path.write_text(body)
        path.chmod(0o600)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", NAME + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "is-active", "--quiet", NAME + ".timer"], check=True)
    print("hunter user timer enabled: Saturday 10:00 UTC")


if __name__ == "__main__":
    main()
