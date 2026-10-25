"""Read-only adapters for existing result exports. Never call a runtime CLI.

Paths are supplied by the operator; no account credentials are read. Broker
acknowledgements in the book are NOT treated as fills. Protected instruments are
screened before outcomes are inspected or included in any digest.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from .core import digest, protected


def read_db(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    db.execute('PRAGMA query_only=ON')
    db.execute('BEGIN')
    return db


def v1_predictions(path):
    db = read_db(path)
    try:
        # SQL exclusion precedes decoding of payloads or labels.
        allowed = "upper(replace(replace(json_extract(payload,'$.product'),'-',''),'/','')) NOT IN ('BTCUSD','ETHUSD')"
        predictions = db.execute("SELECT identity,payload FROM records WHERE kind='prediction' AND "
                                 "json_extract(payload,'$.synthetic')=0 AND " + allowed).fetchall()
        labels = {r[0]: json.loads(r[1]) for r in db.execute(
            "SELECT json_extract(payload,'$.prediction_hash'),payload FROM records WHERE kind='label' AND " + allowed)}
    finally:
        db.close()
    rows = []
    for identity, body in predictions:
        p = json.loads(body)
        definition, view = p['signal']['label_definition'], p['signal']['view']
        label = labels.get(identity)
        weight = p['proposed_position']['weight']
        action = view['view']
        outcome = None
        if label:
            value = label['value']
            outcome = {'available_at': label['available_at'], 'net_contribution': abs(weight) * value['net_unit_pnl'],
                       'net_unit_return': value['net_unit_pnl'], 'raw_return': value['raw_return'],
                       'opportunity_net_return': value['raw_return'] - value['cost_roundtrip'],
                       'y': int(value['spy_relative_return'] > 0), 'cost_basis': 'NET_FULL_COSTS',
                       'kind': 'RETROSPECTIVE_SHADOW_LABEL_NOT_BROKER_FILL'}
        facts = {key: {'value': value, 'available_at': p['decision_at'], 'observed_at': p['decision_at']}
                 for key, value in {'p_outperform': view['p_outperform'], 'planned_horizon': definition['horizon'],
                                    'intended_weight': weight, 'review_verdict': view.get('verdict'),
                                    'snapshot_hash': p['snapshot_hash'], 'features_hash': p['features_hash']}.items()}
        rows.append({'decision_id': identity, 'decision_hash': identity, 'source_hash': digest(p),
                     'strategy': 'v1-shadow:' + p['model_id'] + ':' + definition['horizon'],
                     'symbol': p['product'], 'decision_at': p['decision_at'], 'entry_at': definition['entry_at'],
                     'exit_at': definition['exit_at'], 'weight': weight, 'action': action,
                     'p': view['p_outperform'], 'facts': facts, 'outcome': outcome})
    return rows


def v1_paper(path, predictions):
    """Latest cumulative CLOSED per lot, joined to its FILLED entry and decision."""
    db = read_db(path)
    try:
        records = db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.event')='trade_outcome' "
                             "AND json_extract(payload,'$.account')='ia_actions' ORDER BY sequence").fetchall()
    finally:
        db.close()
    entries, exits = {}, {}
    for (body,) in records:
        event = json.loads(body)
        if event['purpose'] == 'entry' and event['state'] == 'FILLED':
            entries[event['lot_id']] = event
        elif event['purpose'] == 'exit':
            exits[event['lot_id']] = event
    by_hash = {p['decision_hash']: p for p in predictions}
    rows, issues = [], Counter()
    for lot, exit_event in exits.items():
        if exit_event['state'] != 'CLOSED':
            issues['NOT_FULLY_CLOSED_LOT'] += 1
            continue
        entry = entries.get(lot)
        original = by_hash.get(exit_event['prediction_hash'])
        if not entry or not original:
            issues['MISSING_ENTRY_OR_DECISION'] += 1
            continue
        rows.append({**original, 'decision_id': digest({'lot': lot}),
                     'source_hash': digest({'entry': entry, 'exit': exit_event}),
                     'strategy': 'v1-paper', 'entry_at': entry['entry_at'], 'exit_at': exit_event['exit_at'],
                     'cohort': ':'.join([exit_event.get('variant', 'unknown'), exit_event.get('slot', 'am'),
                                         exit_event.get('horizon', 'unknown')]),
                     'action': 'UP', 'outcome': {'available_at': exit_event['at'],
                     'gross_return': exit_event['raw_return'],
                     'gross_pnl_before_fees': exit_event['lot_pnl_before_broker_fees'],
                     'cost_basis': 'GROSS_BEFORE_BROKER_FEES', 'net_contribution': None,
                     'kind': 'BROKER_CLOSED_LOT'}})
    return rows, dict(issues)


def forward(path):
    data = json.loads(Path(path).read_text())
    rows, excluded = [], 0
    # Rows are forecasts, including abstentions; not executed trades.
    for r in data['rows']:
        start = r['entry_at']
        if protected(r['asset'], start, '2026-11-30T23:59:59Z'):
            excluded += 1
            continue
        rows.append({'strategy': 'duel:' + r['family'],
                     'cohort': ':'.join([r['mode'], r['group'], r['horizon']]),
                     'session': r['session'], 'symbol': r['asset'], 'p': r['p'],
                     'y': int(r['outperform']), 'decision_hash': r['decision_hash'],
                     'source_hash': digest(r), 'kind': 'FORECAST_NOT_TRADE',
                     'p0': data['primary_endpoint']['base_rate'] if r['group'] == 'ab' and r['horizon'] == '1d' else None})
    trades = []
    for r in data['trades']:
        if r['asset'].upper().replace('/', '').replace('-', '') in {'BTCUSD', 'ETHUSD'}:
            continue
        trades.append({**r, 'error_classes': ['unidentified_requires_at_T_evidence'],
                       'coverage_status': 'EXPORT_LACKS_ENTRY_EXIT_AND_FULL_AT_T_FEATURES'})
    return rows, {'digest': digest(rows), 'rows': len(rows), 'protected_excluded': excluded,
                  'trade_records': trades, 'trade_records_digest': digest(trades),
                  'base_rate': data['primary_endpoint']['base_rate'],
                  'scored_at': data['scored_at'], 'outcomes_as_of': data['outcomes_as_of']}


def replay(path):
    """July/August export, never September test answers or frozen raw stores."""
    data = json.loads(Path(path).read_text())
    rows = []
    for family, forecasts in data['forecasts'].items():
        for r in forecasts:
            if not '2026-07-01' <= r['session'] <= '2026-08-31':
                raise ValueError('REPLAY_EXPORT_OUTSIDE_TOUCHED_JULY_AUGUST')
            rows.append({'strategy': 'replay:' + family, 'cohort': r['panel'] + ':' + r['horizon'],
                         'session': r['session'], 'symbol': r['asset'], 'p': r['p'],
                         'y': int(r['outperform']), 'source_hash': digest(r), 'kind': 'FORECAST_NOT_TRADE'})
    return rows, {'digest': digest(rows), 'rows': len(rows), 'status': data['status'],
                  'role': 'TOUCHED_VALIDATION_LESSON_GENERATION_ONLY',
                  'original_trial_counts': data['counts'], 'failures': data['failures']}


def book_inventory(path):
    counts, intents = Counter(), []
    protected_excluded = 0
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        symbol = r.get('symbol', '')
        # Do not inspect notes/equity, which may aggregate protected positions.
        if r.get('action') not in {'buy_intent', 'submitted', 'close_intent', 'closed', 'momentum_rebalance'}:
            continue
        if symbol.upper().replace('/', '').replace('-', '') in {'BTCUSD', 'ETHUSD'}:
            protected_excluded += 1
            continue
        counts[r['action']] += 1
        if r['action'] == 'buy_intent':
            intents.append({'at': r['at'], 'symbol': symbol,
                            'journal_hash': digest(r.get('journal', {})),
                            'playbook': r.get('journal', {}).get('playbook', 'unknown')})
    return {'events': dict(counts), 'intent_digest': digest(intents), 'intents': intents,
            'protected_excluded': protected_excluded, 'proven_closed_fills': 0,
            'status': 'BLOCKED_LOCAL_JOURNAL_HAS_NO_EXECUTED_FILL_HISTORY'}


def v2_closed(path):
    """Verify the v2 event chain without instantiating or mutating its runtime."""
    db = read_db(path)
    try:
        events = db.execute('SELECT seq,at,kind,body,previous,hash FROM events ORDER BY seq').fetchall()
    finally:
        db.close()
    previous = '0' * 64
    entries, approved, rows, counts = {}, {}, [], Counter()
    capital = None
    for index, (seq, at, kind, raw, prev, hashed) in enumerate(events, 1):
        body = json.loads(raw)
        event = {'seq': seq, 'at': at, 'kind': kind, 'body': body, 'previous': prev}
        expected = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if seq != index or previous != prev or hashed != expected:
            raise ValueError('V2_SOURCE_CHAIN_CORRUPT')
        previous = hashed
        counts[kind] += 1
        if seq == 1:
            if kind != 'binding' or body['mode'] != 'SHADOW_FORWARD':
                raise ValueError('V2_NOT_FORWARD_SHADOW')
            capital = body['config']['initial_equity']
        if kind == 'approved':
            approved[body['plan_id']] = (at, body, hashed)
        elif kind == 'entry_fill':
            entries[body['plan_id']] = (at, body, hashed)
        elif kind == 'exit_fill':
            plan_id = body['plan_id']
            if plan_id not in entries or plan_id not in approved:
                raise ValueError('V2_MISSING_APPROVAL_OR_FILL')
            _, entry, entry_hash = entries.pop(plan_id)
            decision_at, plan, plan_hash = approved[plan_id]
            if protected(body['symbol'], entry['filled_at'], body['closed_at']):
                continue
            facts = {key: {'value': value, 'available_at': decision_at, 'observed_at': decision_at}
                     for key, value in {'playbook': plan['playbook'], 'layer': plan['layer'],
                                        'snapshot_hash': plan['snapshot_hash'],
                                        'planned_holding_sessions': plan['holding_days']}.items()}
            rows.append({'decision_id': plan_id, 'decision_hash': plan_hash,
                         'source_hash': digest({'approval': plan_hash, 'entry': entry_hash, 'exit': hashed}),
                         'strategy': 'agent_v2_shadow:' + plan['layer'] + ':' + plan['playbook'],
                         'symbol': body['symbol'], 'decision_at': decision_at, 'entry_at': entry['filled_at'],
                         'exit_at': body['closed_at'], 'action': 'UP',
                         'weight': entry['qty'] * entry['entry'] / capital, 'facts': facts,
                         'outcome': {'available_at': at, 'net_contribution': body['pnl'] / capital,
                                     'net_unit_return': body['pnl'] / (entry['qty'] * entry['entry']),
                                     'r': body['r'], 'cost_basis': 'NET_FULL_COSTS',
                                     'kind': 'V2_SHADOW_FILL_NOT_BROKER'}})
    return rows, {'events': dict(counts), 'ledger_hash': previous, 'closed': len(rows),
                  'status': 'NO_DECISIONS' if not counts['plan'] else 'SHADOW_PLANS_PRESENT'}
