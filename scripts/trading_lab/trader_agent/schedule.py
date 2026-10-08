"""Render/install only these agent-owned systemd user units; never trigger a run."""
import argparse
from pathlib import Path
import subprocess
import sys

from scripts.trading_lab.research.store import ResearchStore

from .config import Authorization, TraderError, now, private_root
from .service import preregistration, skills


CALENDARS = {"run": "Mon..Fri *-*-* 12:00:00 UTC", "label": "Mon..Fri *-*-* 21:30:00 UTC",
             "recover": "Mon..Fri *-*-* 08..19:00/10:00 America/New_York",
             "health": "*-*-* *:05:00 UTC"}


def unit_quote(value):
    if any(c in str(value) for c in ('\n', '\r', '%')):
        raise TraderError("INVALID_UNIT_PATH")
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def render(repo, python, authorization, runtime, *, fomc=None, edgar=None, path=None, paper_authorization=None, paper_quotes=None,
           data_authorization=None):
    unit_quote(repo)  # Validate path; WorkingDirectory takes an unquoted whole path value.
    output = {}
    for action, calendar in CALENDARS.items():
        argv = [python, "-m", "scripts.trading_lab.trader_agent.cli", action,
                "--authorization", authorization, "--runtime", runtime]
        if action in {"run", "recover"}:
            for flag, root in (("--fomc-store", fomc), ("--edgar-store", edgar)):
                if root:
                    argv.extend((flag, root))
        if action == 'recover' and paper_authorization:
            argv.extend(('--paper-authorization', paper_authorization))
            if data_authorization:
                argv.extend(('--data-authorization', data_authorization))
            if paper_quotes:
                argv.extend(('--paper-quotes', paper_quotes))
        output[f"hyprl-trader-{action}.service"] = (
            "[Unit]\nDescription=HyprL paper trader " + action + "\n"
            "[Service]\nType=oneshot\nWorkingDirectory=" + str(repo) + "\n"
            "Environment=" + unit_quote("PATH=" + path) + "\n"
            "ExecStart=" + " ".join(unit_quote(arg) for arg in argv) + "\n"
            "UMask=0077\nNice=10\nMemoryMax=900M\nCPUQuota=75%\nNoNewPrivileges=yes\n"
            "TimeoutStartSec=90min\nTimeoutStopSec=20s\nKillMode=control-group\n"
            "StandardOutput=append:" + str(runtime / (action + ".log")) + "\n"
            "StandardError=append:" + str(runtime / (action + ".error.log")) + "\n")
        output[f"hyprl-trader-{action}.timer"] = (
            "[Unit]\nDescription=HyprL paper trader " + action + " schedule\n"
            "[Timer]\nOnCalendar=" + calendar + "\nAccuracySec=1s\nRandomizedDelaySec=0\nPersistent=false\n"
            "[Install]\nWantedBy=timers.target\n")
    if paper_authorization:
        from .paper_spec import execution_spec
        execution_spec()
        def command(action, account=None):
            argv = [python, '-m', 'scripts.trading_lab.trader_agent.cli', 'paper-' + action,
                    '--authorization', authorization, '--paper-authorization', paper_authorization, '--runtime', runtime]
            if account:
                argv.extend(('--paper-account', account))
            if paper_quotes:
                argv.extend(('--paper-quotes', paper_quotes))
            if data_authorization:
                argv.extend(('--data-authorization', data_authorization))
            return ' '.join(unit_quote(a) for a in argv)
        for action, follow in (('run', 'execute'), ('label', 'report')):
            name = f'hyprl-trader-{action}.service'
            hook = 'ExecStartPost=' if action == 'run' else 'ExecStopPost='
            output[name] = output[name].replace('UMask=0077\n', hook + command(follow) + '\nUMask=0077\n')
        for account, calendars in (
                ('ia_actions', ['Mon..Fri *-*-* 15:40:00 America/New_York', 'Mon..Fri *-*-* 12:40:00 America/New_York',
                                'Mon..Fri *-*-* 19:30:00 America/New_York']),
                ('ia_crypto', ['*-*-* *:30:00 UTC'])):
            stem = 'hyprl-trader-paper-exit-' + account.replace('_', '-')
            output[stem + '.service'] = (
                '[Unit]\nDescription=HyprL paper exits ' + account + '\n'
                '[Service]\nType=oneshot\nWorkingDirectory=' + str(repo) + '\n'
                'ExecStart=' + command('exit', account) + '\n'
                'UMask=0077\nNice=10\nMemoryMax=300M\nCPUQuota=50%\nNoNewPrivileges=yes\n'
                'TimeoutStartSec=10min\nTimeoutStopSec=20s\nKillMode=control-group\n'
                'StandardOutput=append:' + str(runtime / (stem + '.log')) + '\n'
                'StandardError=append:' + str(runtime / (stem + '.error.log')) + '\n')
            output[stem + '.timer'] = (
                '[Unit]\nDescription=HyprL paper exit schedule ' + account + '\n[Timer]\n' +
                ''.join('OnCalendar=' + c + '\n' for c in calendars) +
                'AccuracySec=1s\nRandomizedDelaySec=0\nPersistent=false\n[Install]\nWantedBy=timers.target\n')
    return output


def main(argv=None):
    import os
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--fomc-store")
    parser.add_argument("--edgar-store")
    parser.add_argument('--paper-authorization')
    parser.add_argument('--paper-quotes')
    parser.add_argument('--data-authorization')
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    grant = Authorization.load(args.authorization)
    grant.check(now())
    if args.paper_authorization:
        from .alpaca_paper import PaperAuthorization
        paper_grant = PaperAuthorization.load(args.paper_authorization, grant)
        paper_grant.check(now())
        if args.data_authorization:
            from .alpaca_data import DataAuthorization
            DataAuthorization.load(args.data_authorization, paper_grant).check(now())
    elif args.data_authorization:
        parser.error('--data-authorization requires --paper-authorization')
    preregistration()
    skills()
    root = private_root(args.runtime)
    repo = Path(__file__).resolve().parents[3]
    units = render(repo, sys.executable, Path(args.authorization).resolve(), root,
                   fomc=args.fomc_store, edgar=args.edgar_store, path=os.environ["PATH"],
                   paper_authorization=args.paper_authorization, paper_quotes=args.paper_quotes,
                   data_authorization=args.data_authorization)
    if not args.install:
        for name, body in units.items():
            print(name + "\n" + body)
        return
    # Initialize an empty ledger without inference or acquisition.
    ResearchStore(root / "evidence").close()
    target = Path.home() / ".config/systemd/user"
    target.mkdir(parents=True, exist_ok=True)
    for name, body in units.items():
        (target / name).write_text(body)
    subprocess.run(['systemd-analyze', '--user', 'verify', *[str(target / name) for name in units]], check=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", *[n for n in units if n.endswith('.timer')]], check=True)
    subprocess.run(["systemctl", "--user", "list-timers", "hyprl-trader-*", "--no-pager"], check=True)


if __name__ == "__main__":
    main()
