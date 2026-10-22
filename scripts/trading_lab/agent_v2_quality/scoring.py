"""Paired, cost-inclusive candidate outcomes, including candidates the AI rejected."""
import math

from .contracts import ARMS, Blocked, digest, iso, utc


def scorecard(journal):
    completed = {r['body']['plan_id']: r['body'] for r in journal.records('evaluation_close')}
    decisions = {p['plan_id']: p for r in journal.records('decision') for p in r['body']['plans']}
    fills = {r['body']['plan_id']: r['body'] for r in journal.records('evaluation_open')}
    result = {}
    for arm in ARMS[1:]:
        pairs = []
        for identity, outcome in completed.items():
            decision = decisions[identity]
            if not decision['layer_valid'] or arm not in decision.get('scored_layers', ARMS):
                continue
            entry = fills[identity]
            qty = entry['qty']
            selected = math.floor(qty * decision['factors'][arm])
            # Return dollars on exactly the baseline's allocated risk, with unused size held as cash.
            baseline = outcome['unit_pnl'] * qty
            adjusted = outcome['unit_pnl'] * selected
            pairs.append({'plan_id': identity, 'closed_at': outcome['closed_at'], 'baseline': baseline,
                          'arm': adjusted, 'delta': adjusted - baseline,
                          'r_delta': (adjusted - baseline) / entry['initial_risk'],
                          'vetoed': selected == 0, 'loser': baseline < 0,
                          'cluster_day': decision['cluster'] + ':' + decision['planned_at'][:10]})
        pairs.sort(key=lambda p: (p['closed_at'], p['plan_id']))
        first = pairs[:40]
        delta = sum(p['delta'] for p in pairs)
        result[arm] = {'decisions': len(pairs), 'baseline_pnl': sum(p['baseline'] for p in pairs),
                       'layer_pnl': sum(p['arm'] for p in pairs), 'delta_net': delta,
                       'delta_r': sum(p['r_delta'] for p in pairs),
                       'cluster_days': len({p['cluster_day'] for p in pairs}),
                       'vetoes_avoiding_losers': sum(p['vetoed'] and p['loser'] for p in pairs),
                       'vetoes_missing_winners': sum(p['vetoed'] and not p['loser'] for p in pairs),
                       'first_40_delta_net': sum(p['delta'] for p in first) if len(first) == 40 else None,
                       'evidence': 'FORWARD_SHADOW' if journal.binding['mode'] == 'SHADOW_FORWARD' else 'SYNTHETIC',
                       'state': 'DISABLED' if any(r['body']['layer'] == arm for r in journal.records('layer_disabled'))
                       else 'COLLECTING' if len(pairs) < 40 else 'POSITIVE_OBSERVED' if delta > 0 else 'DISABLE_REQUIRED'}
    audits = [r['body'] for r in journal.records('fact_audit')]
    return {'schema': 'agent-v2-quality-scorecard-v2', 'mode': journal.binding['mode'],
            'closed_baseline_candidates': len(completed), 'layers': result,
            'extraction_accuracy': {'adjudicated': len(audits),
                                    'confirmed': sum(a['confirmed'] for a in audits),
                                    'state': 'ADJUDICATED_SUBSET_ONLY' if audits else 'UNMEASURED'},
            'limitations': 'paired simulated quotes; costs included; overlapping cluster-days; 40 is a governance rule, not significance',
            'broker_orders': 0}


def disable_underperformers(journal, at):
    card = scorecard(journal)
    events = []
    for layer in ('extraction', 'veto', 'sizing'):
        row = card['layers'][layer]
        # Permanent, prospective removal; retained diagnostic arms continue to be scored.
        if row['state'] != 'DISABLED' and row['decisions'] >= 40 and row['delta_net'] <= 0:
            events.append(('layer_disabled', {'layer': layer, 'decisions': row['decisions'],
                                            'delta_net': row['delta_net'], 'scorecard_hash': digest(card)}))
    if events:
        journal.append(events, at)
    return [r['body']['layer'] for r in journal.records('layer_disabled')]


def audit_fact(journal, value, *, clock=utc):
    facts = {digest(f): f for r in journal.records('decision') for f in r['body']['facts']['facts']}
    if (set(value) != {'fact_hash', 'confirmed', 'reason', 'operator_reviewed'}
            or value['fact_hash'] not in facts or type(value['confirmed']) is not bool
            or value['operator_reviewed'] is not True or not isinstance(value['reason'], str)
            or not 1 <= len(value['reason']) <= 2000):
        raise Blocked('FACT_ADJUDICATION_REQUIRED')
    with journal.owned():
        if any(r['body']['fact_hash'] == value['fact_hash'] for r in journal.records('fact_audit')):
            raise Blocked('FACT_ALREADY_ADJUDICATED')
        journal.append([('fact_audit', value)], clock())


def weekly_text(journal, *, clock=utc):
    card = scorecard(journal)
    lines = [f"Agent v2 — {card['mode']} — {iso(clock())[:10]} — ordres broker: 0"]
    for layer, row in card['layers'].items():
        lines.append(f"{layer}: n={row['decisions']}, baseline={row['baseline_pnl']:.2f}, "
                     f"couche={row['layer_pnl']:.2f}, delta net={row['delta_net']:.2f}, {row['state']}")
    lines.append('Extraction: ' + card['extraction_accuracy']['state'] + '. Simulations, pas une preuve de rentabilité.')
    return '\n'.join(lines) + '\n'
