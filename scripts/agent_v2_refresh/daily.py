"""Daily data-only sidecar. The activated engine and its journal binding stay frozen."""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.trading_lab.agent_v2.inputs import sessions
from scripts.trading_lab.agent_v2_quality.contracts import Blocked, digest, instant, iso, utc
from scripts.trading_lab.agent_v2_quality.ledger import Journal
from scripts.trading_lab.agent_v2_quality.producer import Client, Grant, Producer, atomic_json, calendar, radar_headlines, read_local, transport
from scripts.trading_lab.agent_v2_quality.registry_refresh import build
from scripts.trading_lab.agent_v2_quality.service import book, code_plans
from .calendar import acquire


class DailyReservations:
    """Builder duplicate detection is daily; Client still uses the full shared budget journal."""
    def __init__(self, journal, day):
        self.journal, self.day = journal, day

    def records(self, kind):
        if kind != 'http_reservation':
            raise Blocked('DAILY_RESERVATION_VIEW_ONLY')
        return [r for r in self.journal.records(kind) if r['at'][:10] == self.day]


def latest_radar(folder):
    best, key = None, None
    for path in Path(folder).glob('radar-*.json'):
        report = read_local(path, maximum=64_000_000)
        if report.get('live') and report.get('schema', '').startswith('news-radar-'):
            candidate = (report.get('cutoff') or '', bool(report.get('review')))
            if key is None or candidate > key:
                best, key = path, candidate
    if best is None:
        raise Blocked('REAL_RADAR_REQUIRED')
    return best


def refresh(settings, *, clock=utc, send=transport, pause=time.sleep, repair_today=False):
    at = clock()
    day = at.date().isoformat()
    if sessions(day, day).empty:
        return {'status': 'SKIPPED_NYSE_CLOSED', 'day': day}
    deadline = at.replace(hour=12, minute=25, second=0, microsecond=0)
    if at >= deadline and not repair_today:
        raise Blocked('REFRESH_WINDOW_CLOSED')
    def guarded_send(url, headers):
        if clock().date().isoformat() != day or not repair_today and clock() >= deadline:
            raise Blocked('REFRESH_WINDOW_CLOSED')
        return send(url, headers)
    root = Path(settings['output'])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root/'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root/day
        output.mkdir(mode=0o700, exist_ok=False)  # durable one-pass reservation, even on failure
        result = {'status': 'BLOCKED', 'day': day, 'repair_today': repair_today, 'broker_orders': 0,
                  'model_calls': 0, 'started_at': iso(at)}
        journal = None
        try:
            for path, expected in settings['pinned_files'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise Blocked('PINNED_SOURCE_CHANGED')
            journal = Journal(settings['runtime'])
            if journal.binding['mode'] != 'SHADOW_FORWARD' or journal.binding['code_sha'] != settings['engine_sha']:
                raise Blocked('ACTIVE_ENGINE_BINDING_REQUIRED')
            original = read_local(settings['base_config'])
            grant = Grant(original['authorization'], clock=clock)
            grant.check()
            source = latest_radar(settings['radar_reports'])
            radar_headlines(read_local(source, maximum=64_000_000), ['SPY'], clock())
            result['stage'] = 'calendar'
            acquire(output/'calendar-refresh', settings['calendar_grant'], day,
                    clock=clock, send=guarded_send, pause=pause)
            client = Client(journal, grant, original['credentials'], original['sec_authorization'],
                            clock=clock, send=guarded_send, pause=pause)
            result['stage'] = 'registry'
            config, review = build(output/'inputs', output/'calendar-refresh', source, original,
                                   settings['seed_registry'], DailyReservations(journal, day), client, clock=clock)
            result['stage'] = 'offline_planner'
            bundle = Producer(client, clock=clock).snapshot(*(config[k] for k in ('registry','calendar','hunter','radar')))
            plans = code_plans(bundle, book(journal), clock())
            if any(p['playbook'] != 'hunter_runner' for p in plans):
                raise Blocked('HUNTER_ONLY_REQUIRED')
            # Normal refresh must also admit the real upcoming cycle time without altering receipts.
            if not repair_today:
                target = at.replace(hour=12, minute=45, second=0, microsecond=0)
                symbols = [r['symbol'] for r in bundle['registry']['instruments']]
                calendar(bundle['calendar'], symbols, target)
                radar_headlines(read_local(config['radar']), symbols, target)
            result.update(status='REFRESHED', at=iso(clock()), plans=len(plans),
                          snapshot_hash=digest(bundle), review=review,
                          source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          note='Current-receipt offline planner preflight; no missed decision cycle is rerun.')
            atomic_json(output/'planner-proof.json', result)
            # One atomic pointer publishes a complete generation. Original Tuesday inputs stay intact.
            atomic_json(root/'current-config.json', config)
        except Exception as error:
            result['reason'] = str(error) if isinstance(error, Blocked) else type(error).__name__
            raise
        finally:
            atomic_json(output/'result.json', result)
            if journal is not None:
                journal.close()
        return result


def render(repo, interpreter, settings):
    command = f'{interpreter} -m scripts.agent_v2_refresh.daily --settings {settings}'
    prefix = 'hyprl-agent-v2-input-refresh'
    return {
        prefix+'.service': '[Unit]\nDescription=Refresh granted v2 shadow inputs daily\nOnFailure='+prefix+'-failure.service\n[Service]\nType=oneshot\nWorkingDirectory='+str(repo)+'\nExecStart='+command+'\nUMask=0077\nNice=15\nMemoryMax=350M\nCPUQuota=35%\nNoNewPrivileges=yes\nTimeoutStartSec=60min\n',
        prefix+'.timer': '[Unit]\nDescription=Refresh inputs before the NYSE shadow cycle\n[Timer]\nOnCalendar=Mon..Fri *-*-* 10:15:00 UTC\nPersistent=false\nAccuracySec=30s\n[Install]\nWantedBy=timers.target\n',
        prefix+'-failure.service': '[Unit]\nDescription=Report v2 input refresh failure\n[Service]\nType=oneshot\nWorkingDirectory='+str(repo)+'\nExecStart='+command+' --notify-failure\nUMask=0077\nNoNewPrivileges=yes\nTimeoutStartSec=60s\n',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--repair-today', action='store_true')
    parser.add_argument('--notify-failure', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.notify_failure:
        result = subprocess.run([str(Path.home()/'bin/hyprl-notify'), 'alertes'],
                                input='Agent v2: daily input refresh BLOCKED; inspect private result.\n',
                                text=True, capture_output=True, timeout=45)
        return 0 if result.returncode == 0 and result.stdout.strip() in {'discord','telegram'} else 1
    try:
        settings = read_local(args.settings)
        repo = Path(__file__).resolve().parents[2]
        sha = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if sha != settings['sidecar_sha'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
            raise Blocked('PINNED_CLEAN_SIDECAR_REQUIRED')
        result = refresh(settings, repair_today=args.repair_today)
        print(json.dumps({k:v for k,v in result.items() if k != 'review'}))
        return 0
    except Exception as error:
        print(json.dumps({'status':'BLOCKED', 'reason':str(error) if isinstance(error, Blocked) else type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
