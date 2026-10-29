"""Operator-installed, one-shot rollback guard; no model, source or broker calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from .config import instant

STEPS = {'configuration', 'yahoo', 'coinbase', 'gdelt', 'radar_veille', 'archives',
         'analyst_claude', 'reviewer', 'alpaca_data', 'executor', 'order_types',
         'scorer_labels', 'cockpit_export'}


def early_green(attempt, day, deployed_at, at):
    try:
        steps = attempt['steps']
        return (attempt['state'] == 'GREEN' and attempt['slot'] == 'early'
                and attempt.get('run_slot') == 'am'
                and attempt['at'][:10] == day
                and instant(deployed_at) <= instant(attempt['at'])
                <= instant(attempt['finished_at']) <= at
                and len(steps) == len(STEPS) and {s['step'] for s in steps} == STEPS
                and all(s['state'] in {'GREEN', 'DEGRADED'} for s in steps)
                and all(a['delivery'] == 'DELIVERED' for a in attempt.get('alerts', [])))
    except (KeyError, TypeError, ValueError):
        return False


def read_attempt(monitor, day):
    with sqlite3.connect(Path(monitor).resolve().as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute('SELECT body FROM attempts WHERE day=? AND slot=?', (day, 'early')).fetchone()
    return json.loads(row[0]) if row else None


def guard(config, *, at=None, run=subprocess.run):
    at = at or datetime.now(timezone.utc)
    def command(args, **kwargs):
        return run(args, capture_output=True, text=True, timeout=120, check=False, **kwargs)
    outcome = 'GUARD_FAILED'
    try:
        if at.date().isoformat() != config['day'] or not '07:50' <= at.strftime('%H:%M') < '11:00':
            outcome = 'GUARD_OUTSIDE_WINDOW'
        else:
            try:
                attempt = read_attempt(config['monitor'], config['day'])
                unit = command(['systemctl', '--user', 'show', 'hyprl-trader-rehearsal-early.service',
                                '-p', 'WorkingDirectory', '-p', 'Result', '-p', 'ExecMainStatus'])
                props = dict(line.split('=', 1) for line in unit.stdout.splitlines() if '=' in line)
                green = (unit.returncode == 0 and props.get('WorkingDirectory') == config['release']
                         and props.get('Result') == 'success' and props.get('ExecMainStatus') == '0'
                         and early_green(attempt, config['day'], config['deployed_at'], at))
            except (OSError, sqlite3.Error, ValueError, TypeError):
                green = False
            if green:
                outcome = 'RELEASE_RETAINED_EARLY_GREEN'
            else:
                script = Path(config['rollback_script'])
                if hashlib.sha256(script.read_bytes()).hexdigest() != config['rollback_sha256']:
                    raise ValueError('rollback binding')
                # A stalled read-only rehearsal must not prevent a safe directory rollback.
                command(['systemctl', '--user', 'stop', 'hyprl-trader-rehearsal-early.service'])
                result = command(['bash', str(script), 'rollback', config['proof']])
                outcome = 'ROLLED_BACK_EARLY_NOT_GREEN' if result.returncode == 0 else 'ROLLBACK_FAILED_TIMERS_REQUIRE_REVIEW'
    except (OSError, ValueError, subprocess.SubprocessError):
        outcome = 'GUARD_FAILED'
    try:
        notice = command([config['notify'], 'alertes'], input='Trader release guard: ' + outcome + '\n')
        delivered = notice.returncode == 0 and notice.stdout.strip() in {'discord', 'telegram'}
    except (OSError, subprocess.SubprocessError):
        delivered = False
    return {'outcome': outcome, 'notification': 'DELIVERED' if delivered else 'FAILED'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    result = guard(json.loads(args.config.read_text()))
    print(json.dumps(result))
    return 0 if result['outcome'] in {'RELEASE_RETAINED_EARLY_GREEN', 'ROLLED_BACK_EARLY_NOT_GREEN'} and result['notification'] == 'DELIVERED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
