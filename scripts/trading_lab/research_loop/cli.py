"""Manual Saturday runner. All outputs stay private; no network or scheduling."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import ingest, stats
from .core import TrialLedger, audit, digest, evaluate_trial
from .spec import ROOT, SPEC, SPEC_HASH, REGISTRY_HASH


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    path.chmod(0o600)


def forecasts_report(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['strategy'] + ':' + r['cohort']].append(r)
    result = {}
    family = max(SPEC['family_size_floor'], len(groups))
    target = stats.required_decisions(family=family)
    for cohort, members in groups.items():
        hits = sum((r['p'] > .5) == bool(r['y']) for r in members)
        p = stats.binomial_tail(hits, len(members))
        brier = stats.brier_null([r['p'] for r in members], [r['y'] for r in members], draws=SPEC['draws'])
        sessions = len({r['session'] for r in members})
        result[cohort] = {'n': len(members), 'sessions': sessions, 'hits_forecast_direction': hits,
                          'p_coin_iid_diagnostic': p, 'p_coin_adjusted': min(1, p * family),
                          'brier_coin_null': brier, 'family': family,
                          'further_independent_decisions': max(0, target - sessions),
                          'random_entry': {'status': 'BLOCKED', 'draws': 0,
                                           'reason': 'MISSING_COMPLETE_POINT_IN_TIME_UNIVERSE_AND_MATCHED_PRICES'}}
        rates = {r.get('p0') for r in members}
        if len(rates) == 1 and None not in rates:
            result[cohort]['brier_frozen_base_null'] = stats.brier_null(
                [r['p'] for r in members], [r['y'] for r in members], p0=rates.pop(), draws=SPEC['draws'])
    return result


def run(args):
    out = Path(args.output).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('PRIVATE_OUTPUT_MUST_BE_OUTSIDE_REPOSITORY')
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    # A new immutable run directory prevents accidental overwriting of old analysis.
    run_dir = out / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir.mkdir(mode=0o700)
    predictions = ingest.v1_predictions(args.v1)
    paper, paper_issues = ingest.v1_paper(args.paper, predictions)
    replay, replay_meta = ingest.replay(args.replay)
    forward, forward_meta = ingest.forward(args.forward)
    book = ingest.book_inventory(args.book)
    v2_rows, v2 = ingest.v2_closed(args.v2)
    audits, quarantined = [], []
    for row in predictions + paper + v2_rows:
        try:
            audits.append(audit(row))
        except ValueError as error:
            quarantined.append({'decision_id': row['decision_id'], 'reason': str(error)})
    dump(run_dir / 'postmortem-private.json', audits)
    dump(run_dir / 'normalized-private.json', predictions + paper + v2_rows)
    closed = [r for r in audits if r['closed']]
    valid_closed_ids = {(r['strategy'], r['decision_id']) for r in closed}
    paper = [r for r in paper if (r['strategy'], r['decision_id']) in valid_closed_ids]
    counts = Counter(r['strategy'] for r in closed)
    errors = Counter(e for r in closed for e in r['error_classes'])
    # Existing July/Aug replay and October records NEVER become development rows.
    # Optional normalized development input must carry original at-T evidence.
    development = []
    if args.development:
        development = json.loads(Path(args.development).read_text())
        if any(not '2025-01-01' <= r['entry_at'][:10] <= r['exit_at'][:10] <= '2026-06-30' for r in development):
            raise ValueError('DEVELOPMENT_INPUT_CONTAINS_OTHER_PERIOD')
    ledger = TrialLedger(out / 'trials.sqlite')
    try:
        for path in args.prior_trial_ledger:
            ledger.include_prior_count(path)
        trials = [evaluate_trial(development, h, ledger) for h in ['H01', 'H02', 'H03']]
        attempts = ledger.trial_count()
        history = ledger.prior_references()
    finally:
        ledger.db.close()
    dump(run_dir / 'trials.json', trials)
    shadow_forecasts = [{'strategy': p['strategy'], 'cohort': p['target'], 'session': p['entry_at'][:10],
                         'p': p['p'], 'y': p['outcome']['y']} for p in predictions
                        if (p['strategy'], p['decision_id']) in valid_closed_ids]
    forecast_results = forecasts_report(replay + forward + shadow_forecasts)
    dump(run_dir / 'forecast-luck-private.json', forecast_results)
    dump(run_dir / 'forecast-postmortem-private.json', [dict(r, error_classes=['unidentified_requires_at_T_evidence'])
                                                      for r in replay + forward])
    # Broker evidence uses all unique CLOSED lots; fees are not invented.
    hits = sum(r['outcome']['gross_return'] > 0 for r in paper)
    paper_p = stats.binomial_tail(hits, len(paper)) if paper else None
    sessions = len({r['entry_at'][:10] for r in paper})
    target = stats.required_decisions(family=SPEC['family_size_floor'])
    paper_cohorts = defaultdict(list)
    for row in paper:
        paper_cohorts[row['cohort']].append(row)
    luck = {'v1': {'n_closed_broker_lots': len(paper), 'sessions': sessions, 'gross_hits': hits,
                   'all_cohorts': {name: {'n': len(members), 'gross_hits': sum(r['outcome']['gross_return'] > 0 for r in members)}
                                   for name, members in paper_cohorts.items()},
                   'p_coin_gross_iid_diagnostic': paper_p,
                   'p_coin_adjusted': min(1, paper_p * SPEC['family_size_floor']) if paper_p is not None else None,
                   'further_independent_decisions': max(0, target - sessions),
                   'cost_basis': 'GROSS_BEFORE_BROKER_FEES',
                   'random_entry': {'status': 'BLOCKED', 'draws': 0, 'reason': 'NO_MATCHED_FULL_UNIVERSE'}},
            'agent_v2_shadow': {'status': v2['status']},
            'momentum': {'status': 'BLOCKED_NO_OFFLINE_CLOSED_FILL_AND_UNIVERSE_EXPORT'},
            'hunter': {'status': 'BLOCKED_NO_OFFLINE_CLOSED_FILL_AND_UNIVERSE_EXPORT'},
            'claude_book': {'status': book['status']}}
    for model in ['gpt', 'opus']:
        # Preselected regular AB 1d is descriptive; every panel remains in the full file.
        primary = {k: v for k, v in forecast_results.items()
                   if k.startswith('duel:' + model + ':REGULAR_0905:') and k.endswith(':1d')}
        luck['duel_' + model] = {'status': 'Brier_NULL_DIAGNOSTIC_ONLY', 'primary_panels': primary,
                                 'random_entry': {'status': 'BLOCKED', 'draws': 0}}
    blockers = ['NO_COMPLETE_MATCHED_POINT_IN_TIME_UNIVERSES_FOR_10000_RANDOM_ENTRIES',
                'NO_OFFLINE_BOOK_FILL_EXPORT', 'WEEKLY_SCHEDULER_NOT_INSTALLED_OPERATOR_OWNED']
    if not development:
        blockers.append('NO_CAUSAL_2025_JAN_TO_2026_JUN_DEVELOPMENT_EXPORT')
    if any(t['status'] == 'BLOCKED' for t in trials) and development:
        blockers.append('DEVELOPMENT_TRIAL_EVIDENCE_INCOMPLETE')
    summary = {'status': 'BLOCKED' if blockers else 'RESEARCH_ONLY_NO_PROMOTION',
               'spec_hash': SPEC_HASH, 'registry_hash': REGISTRY_HASH, 'closed_by_strategy': dict(counts),
               'postmortem_errors': dict(errors), 'postmortem_digest': digest(audits),
               'paper_issues': paper_issues, 'replay': replay_meta, 'forward': forward_meta,
               'quarantined': quarantined,
               'book': book, 'v2': v2, 'trials': trials, 'trial_attempts_all_runs': attempts,
               'prior_trial_references': history,
               'luck': luck, 'forecast_luck_digest': digest(forecast_results),
               'promotion': 'NONE', 'validation': SPEC['validation_status'],
               'blocked': blockers}
    dump(run_dir / 'summary.json', summary)
    now = datetime.now(timezone.utc)
    week = now.strftime('%G-W%V')
    timing = 'samedi' if now.weekday() == 5 else 'hors cycle samedi'
    blocked_trials = sum(t['status'] == 'BLOCKED' for t in trials)
    lines = [f'Boucle de recherche v1 — {week}, {timing} — BLOQUÉE pour conclure.',
             f"{len(closed)} décisions/labels clos audités ; {len(paper)} lots broker clos uniques (brut avant frais).",
             f'8 hypothèses figées ; {attempts} tentatives cumulées ; 3 réservations ce passage, {blocked_trials} bloquées.' + (' Aucun export causal janv. 2025–juin 2026.' if not development else ''),
             'Juil.–sept. déjà touchés ; aucune promotion. Holdout BTC/ETH exclu.',
             'Hasard = p sous H0, pas probabilité que la stratégie soit due au hasard.',
             'Tirages de titres à dates/durées identiques : non calculables sans univers complet à T. N est un plan indicatif, pas une garantie.',
             f"v1 : p pile-ou-face brut = {100*paper_p:.1f}% (dépendance non corrigée) ; encore {max(0,target-sessions)} sessions indépendantes pour le plan +5 points." if paper_p is not None else 'v1 : hasard non chiffrable, aucun lot clos prouvé.']
    for label, key in [('v2 shadow','agent_v2_shadow'),('momentum','momentum'),('hunter','hunter'),('book Claude','claude_book')]:
        lines.append(f'{label} : hasard non chiffrable ; encore {target} décisions indépendantes selon le plan 95% / puissance 80% ; données manquantes.')
    for model in ['gpt', 'opus']:
        panels = luck['duel_' + model]['primary_panels']
        if len(panels) == 1:
            v = next(iter(panels.values()))
            lines.append(f"duel {model} : p Brier/pile-ou-face = {100*v['brier_coin_null']['p_null']:.1f}% (diagnostic iid, {v['sessions']} sessions) ; encore {v['further_independent_decisions']} sessions indépendantes (+5 points, taux de succès).")
        else:
            lines.append(f'duel {model} : hasard non chiffrable dans un panel primaire unique ; voir tous les panels privés.')
    lines.append('Le book reste séparé de tous les baselines IA. Aucun ordre, appel source ou service modifié.')
    (run_dir / 'discord-fr.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': summary['status'], 'run': str(run_dir), 'closed': len(closed),
                      'paper_lots': len(paper), 'trial_attempts': attempts, 'promotions': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['v1', 'paper', 'v2', 'book', 'replay', 'forward', 'output']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--development', help='Private normalized JSON list; no fetching or inference of missing facts')
    parser.add_argument('--prior-trial-ledger', action='append', default=[],
                        help='Read-only, hash-verified prior counts; no prior result reuse or migration')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
