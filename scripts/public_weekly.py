"""Member weekly presentation, outside the frozen research execution tree.

Keeps the existing score admission, correction and at-most-once receipt format.
Only new prose is rendered with public names, before message splitting/hashing.
"""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.public_agent_names import PUBLIC_METHOD, public_agent_label, public_agent_text
from scripts.trading_lab.replay_bench.grant import Grant, ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json, owned
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_seal_forward import cli, commitment, experiment as ex, forward, spec, stats


def render_weekly(value, week, *, correction=False):
    primary = value.get('primary_endpoint', {})
    lines = [('CORRECTION — ' if correction else '') +
             'LIVE_FORWARD — duel hebdomadaire ' + week + ' (replay EXPLORATORY)',
             'Méthode : ' + PUBLIC_METHOD + '.',
             'Échéances arrêtées au ' + value.get('outcomes_as_of', value['scored_at']) + '.',
             'Critère principal : a:1d Brier skill vs taux de base (' + value.get('base_rate_source', '?') + ').',
             f"Skill : {json.dumps(primary.get('brier_skill'))} ; IC 95 % : {json.dumps(primary.get('ci95'))} ; état {primary.get('state')}.",
             value.get('verdict_wording', stats.wording(primary)),
             'Tableau complet, scores cumulés aux horizons arrivés à échéance :']
    for mode, table in value.get('tables_by_mode', {'REGULAR_0905': value['full_table']}).items():
        lines.append(mode + (' — hors critère principal' if mode == 'INTRADAY' else ' — entrée régulière'))
        for panel, models in table.items():
            for family, row in models.items():
                lines.append(panel + ' ' + public_agent_label(family) + ' : ' + json.dumps(row, sort_keys=True))
    for family in ex.FAMILIES:
        label = public_agent_label(family)
        trades = [r for r in value.get('trades', []) if r['family'] == family]
        taken = [r for r in trades if r['hit'] is not None]
        lines.append(f"{label}: décisions mûres={len(trades)}, abstentions/exclusions={len(trades)-len(taken)}, "
                     f"trades positifs (base de coûts indiquée dans le tableau)={sum(r['hit'] for r in taken)}/{len(taken)}")
        failed = value.get('failures', {}).get(family, {})
        lines.append(f"{label}: jours en échec={failed.get('count', 0)} " +
                     ', '.join(f"{r['session']} {r['error']}" for r in failed.get('failed_sessions', [])))
    lines.append('Recherche sans ordres. Effectifs dépendants ; « gagnant » seulement si l’IC apparié exclut 0.')
    return public_agent_text('\n'.join(lines))


def weekly(root, *, notify, clock=now, runner=subprocess.run, correction_week=None, score_run=None):
    at = clock()
    ny = at.astimezone(forward.md.NEW_YORK)
    if correction_week is None and (ny.weekday() != 4 or ny.hour < 17):
        raise ReplayError('OUTSIDE_WEEKLY_REPORT_WINDOW')
    week = correction_week or ny.strftime('%G-W%V')
    base = Path(root) / forward.STATE_DIR
    marker = base / ('weekly-' + week + '.json')
    original = None
    if correction_week:
        try:
            datetime.strptime(week + '-1', '%G-W%V-%u')
        except ValueError:
            raise ReplayError('WEEKLY_CORRECTION_INVALID') from None
        if not marker.is_file():
            raise ReplayError('WEEKLY_ORIGINAL_MISSING')
        original = json.loads(marker.read_text())
        if original.get('state') != 'DELIVERED':
            raise ReplayError('WEEKLY_ORIGINAL_UNCONFIRMED')
        marker = base / ('weekly-' + week + '-correction-' + original['text_hash'] + '.json')
    if marker.exists():
        raise ReplayError('WEEKLY_DELIVERY_ALREADY_ATTEMPTED')
    if score_run is not None and (len(score_run) != 64 or set(score_run) - commitment.HEX):
        raise ReplayError('WEEKLY_SCORE_HASH_INVALID')
    source = base / 'score-runs' / (score_run + '.json') if score_run else base / 'scores.json'
    value = json.loads(source.read_text())
    if score_run and sha256_canonical(value) != score_run:
        raise ReplayError('WEEKLY_SCORE_HASH_INVALID')
    if not at - timedelta(hours=2) <= instant(value['scored_at']) <= at:
        raise ReplayError('WEEKLY_SCORES_STALE')
    if correction_week and datetime.fromisoformat(value['through_session']).strftime('%G-W%V') != week:
        raise ReplayError('WEEKLY_CORRECTION_PERIOD_MISMATCH')
    key = lambda r: (r['session'], r['group'], r['asset'], r['horizon'], r['snapshot_hash'])
    panels = {f: {key(r) for r in value['rows'] if r['family'] == f} for f in ex.FAMILIES}
    text = render_weekly(value, week, correction=bool(original))
    binding = {'score_sha256': sha256_canonical(value)}
    if original:
        binding['corrects_text_hash'] = original['text_hash']
    atomic_json(marker, dict(binding, state='INTENT', at=iso(at), text_hash=sha256_canonical(text)))
    messages = [text[i:i+1800] for i in range(0, len(text), 1800)]
    for message in messages:
        result = runner([str(notify), 'brouillons'], input=message, capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise ReplayError('WEEKLY_DELIVERY_UNKNOWN')
        if (result.stdout or '').strip() != 'discord':
            raise ReplayError('WEEKLY_DISCORD_DELIVERY_UNCONFIRMED')
    atomic_json(marker, dict(binding, state='DELIVERED', at=iso(at), text_hash=sha256_canonical(text), messages=len(messages)))
    return {'status': 'DELIVERED', 'week': week, 'paired_rows': len(panels['opus'] & panels['gpt']),
            'correction': bool(original)}


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required='REPLAY_DATA_ROOT' not in os.environ)
    parser.add_argument('--authorization', default=os.environ.get('REPLAY_AUTHORIZATION'), required='REPLAY_AUTHORIZATION' not in os.environ)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--correction-week')
    parser.add_argument('--score-run')
    args = parser.parse_args(argv)
    root = None
    try:
        root = Path(args.data_root).expanduser().resolve()
        if root == spec.ROOT or spec.ROOT in root.parents:
            root = None
            raise ReplayError('DATA_ROOT_MUST_BE_OUTSIDE_GIT')
        grant = Grant(args.authorization)
        if sha256_canonical(grant.payload) != spec.spec()['authorization']['payload_sha256']:
            raise ReplayError('GRANT_HASH_MISMATCH')
        with owned(root):
            ex.committed()
            if ex.head() != args.expected_commit:
                raise ReplayError('INSTALLED_SHA_MISMATCH')
            notify = os.environ.get('REPLAY_NOTIFY')
            if not notify:
                raise ReplayError('NOTIFY_EXECUTABLE_MISSING')
            result = weekly(root, notify=notify, correction_week=args.correction_week, score_run=args.score_run)
            atomic_json(root / spec.STATE_DIR / 'weekly-status.json', result)
        cli.progress_report(root, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ReplayError, OSError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'BLOCKED', 'error': error.code if isinstance(error, ReplayError)
                  else 'INVALID_OR_MISSING_INPUT'}
        if root is not None:
            result['failures'] = ex.failure_summary(root)
            atomic_json(root / spec.STATE_DIR / 'weekly-status.json', result)
            cli.progress_report(root, result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
