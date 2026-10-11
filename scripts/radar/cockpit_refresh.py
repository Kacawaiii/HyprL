"""Hourly offline export. Reads existing files and user-unit status; never collects data.

The new cockpit timer is the only unit this installer changes. Trader, radar and
Claude book units are observed with `show`, never started, stopped or rewritten.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess

from . import cockpit_export as export
from .cockpit_analysis import build_analysis

UNIT_NAME = 'hyprl-cockpit-refresh'
CODE = re.compile(r'^[A-Z][A-Z0-9_:-]{0,100}$')


def user_bus_env():
    env = os.environ.copy()
    runtime = Path('/run/user') / str(os.getuid())
    if runtime.is_dir() and runtime.stat().st_uid == os.getuid():
        env.setdefault('XDG_RUNTIME_DIR', str(runtime))
        env.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=' + str(runtime / 'bus'))
    return env


def embargo(at):
    # Three-minute execution ceiling also keeps manual starts clear of 11:45Z.
    return '11:42' <= at.astimezone(timezone.utc).strftime('%H:%M') < '12:30'


def unit_status(unit_dir=None):
    folder = unit_dir or Path.home() / '.config/systemd/user'
    names = sorted({p.name for pattern in ('hyprl-radar-*.service', 'hyprl-radar-*.timer',
                   'hyprl-trader-*.service', 'hyprl-trader-*.timer', 'claude-book-*.timer',
                   'hyprl-cockpit-refresh.*') for p in folder.glob(pattern)})
    out = []
    for name in names:
        try:
            p = subprocess.run(['systemctl', '--user', 'show', name, '--no-pager',
                '--property=Id,ActiveState,SubState,Result,ExecMainStatus,ExecMainStartTimestamp,LastTriggerUSec,NextElapseUSecRealtime'],
                capture_output=True, text=True, timeout=3, check=False, env=user_bus_env())
            props = dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)
            out.append({'unit': name, 'state': props.get('ActiveState') if p.returncode == 0 else 'UNKNOWN',
                        'result': props.get('Result'), 'exit_status': props.get('ExecMainStatus'),
                        'last_run': props.get('LastTriggerUSec') or props.get('ExecMainStartTimestamp') or None,
                        'next_run': props.get('NextElapseUSecRealtime') or None})
        except (OSError, subprocess.TimeoutExpired):
            out.append({'unit': name, 'state': 'UNKNOWN', 'result': 'STATUS_UNAVAILABLE', 'last_run': None, 'next_run': None})
    return out


def radar_runtime(root, day):
    """Open the active radar ledger with SQLite mode=ro, without Store initialization."""
    with sqlite3.connect((Path(root) / 'radar.sqlite').resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
        counts = dict(db.execute('SELECT group_name,count(*) FROM dispatch WHERE day=? GROUP BY group_name', (day,)))
        rows = db.execute("SELECT body FROM evidence WHERE kind='health' ORDER BY seq DESC LIMIT 300").fetchall()
    latest = {}
    for (body,) in rows:
        s = json.loads(body)
        latest.setdefault(s['source'], s)
    return counts, list(latest.values())


def trader_health(root):
    root = Path(root)
    out = {'paused': (root / 'PAUSED').exists(), 'health': None, 'last_label': None, 'failures': []}
    for key, name in (('health', 'health.json'), ('last_label', 'last-label.json')):
        path = root / name
        if path.is_file() and path.stat().st_size <= 65536:
            try:
                value = export.read_json(path)
                out[key] = {k: value.get(k) for k in ('at', 'state', 'budget_counts')}
            except ValueError:
                pass
    path = root / 'alerts.jsonl'
    if path.is_file():
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 65536))
            if size > 65536:
                stream.readline()
            for line in stream.read().splitlines()[-50:]:
                try:
                    row = json.loads(line)
                    code = row.get('code') or ''
                    if CODE.fullmatch(code):
                        out['failures'].append({'at': export.text(row.get('at'), 40), 'code': code})
                except ValueError:
                    continue
    return out


def latest_radar(path):
    path = Path(path)
    if path.is_file():
        return export.read_json(path)
    reports = []
    for p in path.glob('radar-*.json'):
        try:
            r = export.read_json(p)
            if r.get('live') and r.get('schema', '').startswith('news-radar-'):
                reports.append(r)
        except (ValueError, OSError):
            continue
    if not reports:
        raise export.ExportError('NO_REAL_RADAR_REPORT')
    return max(reports, key=lambda r: (r.get('cutoff') or '', bool(r.get('review'))))


def refresh(config, *, at=None, read_units=True):
    at = at or datetime.now(timezone.utc)
    if embargo(at):
        return {'state': 'SKIPPED_TRADER_WINDOW'}
    out = Path(config['out']).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if out == repo or repo in out.parents:
        raise export.ExportError('PRIVATE_OUTPUT_INSIDE_GIT')
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (out / 'refresh.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'state': 'OWNER_BUSY'}
        stamp = at.strftime('%Y-%m-%dT%H:%M:%SZ')
        radar = latest_radar(config['radar'])
        from scripts.trading_lab.research.store import ResearchStore
        store = ResearchStore(Path(config['trader_root']) / 'evidence', read_only=True)
        try:
            trader = export.load_trader(store, export.load_scorecard(store))
        finally:
            store.close()
        journal = export.read_jsonl(config['book_journal'])
        paper_path = out / 'paper.json'
        paper = export.read_json(paper_path) if paper_path.is_file() else None
        # A cached broker mark keeps its observation date. Exporting it again does
        # not create another point in the equity history or claim a fresh price.
        if paper:
            from .cockpit_analysis import journal_chain
            index = export._engine_index(journal)
            for account in paper['accounts']:
                account.setdefault('observed_at', paper['generated_at'])
                if account['account'] == 'claude_book':
                    account['journal'] = export._journal_rows(journal)
                    for p in account['positions']:
                        intent = index.get(export.norm_symbol(p['symbol']))
                        p['decision'] = intent.get('decision') if intent else journal_chain({})
            report_path = config.get('paper_report')
            if report_path and Path(report_path).is_file():
                report = export.read_json(report_path)
                if report.get('at') and (not paper.get('report_at') or report['at'] >= paper['report_at']):
                    accounts = export.build_ai_accounts(report)
                    for account in accounts:
                        account['observed_at'] = report['at']
                    paper['accounts'] = accounts + [a for a in paper['accounts'] if a['account'] == 'claude_book']
                    paper['report_at'] = report['at']
            paper['generated_at'] = stamp
            export.assert_clean(paper)
        health = trader_health(config['trader_root'])
        for run in trader['runs'][-30:]:
            if run.get('status') not in ('COMPLETE', 'RUNNING'):
                health['failures'].append({'at': run['at'], 'code': run.get('error') if CODE.fullmatch(run.get('error') or '') else run.get('status')})
        analysis = build_analysis(radar, trader, paper, journal, health=health,
                                  units=unit_status() if read_units else [], generated_at=stamp)
        if config.get('radar_root'):
            try:
                counts, source_health = radar_runtime(config['radar_root'], stamp[:10])
                known_sources = {s['source']: s for s in radar.get('sources', [])}
                known_sources.update({s['source']: s for s in source_health})
                analysis['system'].update({'budget_date': stamp[:10], 'budgets_used': counts, 'sources': [
                    {k: s.get(k) for k in ('source', 'status', 'reason', 'checked_at', 'items')}
                    for s in known_sources.values()]})
            except (sqlite3.Error, OSError, ValueError):
                analysis['limitations'].append('Santé du ledger radar indisponible; dernier rapport conservé avec sa date.')
        export.assert_clean(analysis)
        home = export.build_home(radar, trader, paper, generated_at=stamp)
        # Validate all outputs before replacing any published file.
        for name, snapshot in [('radar-home.json', home), ('analysis.json', analysis), ('paper.json', paper)]:
            if snapshot:
                export.write_snapshot(out, name, snapshot)
        return {'state': 'EXPORTED', 'events': len(home['events']), 'overlays': len(analysis['overlays'])}


def render_units(repo, interpreter, config):
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    service = ('[Unit]\nDescription=Read-only hourly cockpit snapshots\n\n[Service]\nType=oneshot\n'
               f'WorkingDirectory={str(repo).replace("%", "%%")}\nExecStart={quote(interpreter)} -m scripts.radar.cockpit_refresh --config {quote(config)}\n'
               'KillMode=process\nMemoryMax=384M\nCPUQuota=40%\nNice=15\nUMask=0077\n'
               'NoNewPrivileges=true\nTimeoutStartSec=3min\n')
    timer = ('[Unit]\nDescription=Hourly cockpit refresh outside trader window\n\n[Timer]\n'
             'OnCalendar=*-*-* *:35:00 UTC\nAccuracySec=30s\nPersistent=false\n'
             f'Unit={UNIT_NAME}.service\n\n[Install]\nWantedBy=timers.target\n')
    return service, timer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Private JSON input paths; no credentials')
    parser.add_argument('--install-timer', action='store_true')
    args = parser.parse_args(argv)
    if args.install_timer:
        import sys
        folder = Path.home() / '.config/systemd/user'
        folder.mkdir(parents=True, exist_ok=True)
        service, timer = render_units(Path(__file__).resolve().parents[2], sys.executable, args.config.resolve())
        for suffix, value in (('service', service), ('timer', timer)):
            (folder / (UNIT_NAME + '.' + suffix)).write_text(value)
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, env=user_bus_env())
        subprocess.run(['systemctl', '--user', 'enable', '--now', UNIT_NAME + '.timer'], check=True, env=user_bus_env())
    else:
        print(json.dumps(refresh(export.read_json(args.config))))


if __name__ == '__main__':
    main()
