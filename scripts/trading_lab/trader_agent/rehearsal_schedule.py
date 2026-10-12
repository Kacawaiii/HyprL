"""Render only agent-owned rehearsal and read-only audit units; installation never triggers a rehearsal."""
import argparse
from pathlib import Path
import subprocess
import sys

from .rehearsal import arguments
from .schedule import unit_quote


def render(repo, python, authorization, runtime, paper_authorization, data_authorization, *, path,
           fomc=None, edgar=None, monitor_runtime=None, reports=None):
    unit_quote(repo)
    common = ['--authorization', str(authorization), '--runtime', str(runtime),
              '--paper-authorization', str(paper_authorization), '--data-authorization', str(data_authorization)]
    for flag, value in (('--fomc-store', fomc), ('--edgar-store', edgar),
                        ('--monitor-runtime', monitor_runtime), ('--reports', reports)):
        if value:
            common.extend((flag, str(value)))
    units = {}
    for name, module, extra in (
            ('rehearsal', 'rehearsal', []), ('rehearsal-retry', 'rehearsal', ['--retry']),
            ('rehearsal-failure', 'rehearsal', ['--failure-hook']), ('post-audit', 'post_audit', [])):
        argv = [str(python), '-m', 'scripts.trading_lab.trader_agent.' + module, *common, *extra]
        failure = 'OnFailure=hyprl-trader-rehearsal-failure.service\n' if name in {'rehearsal', 'rehearsal-retry'} else ''
        units['hyprl-trader-' + name + '.service'] = (
            '[Unit]\nDescription=HyprL trader ' + name + '\n' + failure +
            '[Service]\nType=oneshot\nWorkingDirectory=' + str(repo) + '\nEnvironment=' + unit_quote('PATH=' + path) + '\n' +
            'Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\nExecStart=' + ' '.join(unit_quote(a) for a in argv) + '\n' +
            'UMask=0077\nNice=10\nMemoryMax=700M\nCPUQuota=50%\nNoNewPrivileges=yes\n' +
            ('TimeoutStartSec=10min\n' if module == 'rehearsal' else 'TimeoutStartSec=2min\n') +
            'TimeoutStopSec=20s\nKillMode=control-group\n' +
            'StandardOutput=append:' + str(Path(runtime) / (name + '.log')) + '\n' +
            'StandardError=append:' + str(Path(runtime) / (name + '.error.log')) + '\n')
    for name, calendar in (('rehearsal', '*-*-* 10:45:00 UTC'),
                           ('rehearsal-retry', '*-*-* 11:15:00 UTC'),
                           ('post-audit', '*-*-* *:*:30 UTC')):
        units['hyprl-trader-' + name + '.timer'] = (
            '[Unit]\nDescription=HyprL trader ' + name + ' schedule\n[Timer]\nOnCalendar=' + calendar +
            '\nAccuracySec=1s\nRandomizedDelaySec=0\nPersistent=false\n[Install]\nWantedBy=timers.target\n')
    return units


def main(argv=None):
    import os
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args(argv)
    units = render(Path(__file__).resolve().parents[3], sys.executable, args.authorization, args.runtime,
                   args.paper_authorization, args.data_authorization, path=os.environ['PATH'],
                   fomc=args.fomc_store, edgar=args.edgar_store, monitor_runtime=args.monitor_runtime, reports=args.reports)
    if not args.install:
        for name, body in units.items():
            print(name + '\n' + body)
        return
    target = Path.home() / '.config/systemd/user'
    target.mkdir(parents=True, exist_ok=True)
    for name, body in units.items():
        (target / name).write_text(body)
    subprocess.run(['systemd-analyze', '--user', 'verify', *[str(target / n) for n in units]], check=True)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', *[n for n in units if n.endswith('.timer')]], check=True)


if __name__ == '__main__':
    main()
