"""Point-in-time post-mortems, counted filter trials and append-only provenance."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from . import stats
from .spec import REGISTRY, SPEC, SPEC_HASH, REGISTRY_HASH


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def instant(value):
    at = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if at.tzinfo is None:
        raise ValueError('NAIVE_CLOCK')
    return at.astimezone(timezone.utc)


def protected(symbol, entry, exit_at):
    return (symbol.upper().replace('/', '').replace('-', '') in {'BTCUSD', 'ETHUSD'}
            and entry[:10] <= SPEC['protected']['end'] and exit_at[:10] >= SPEC['protected']['start'])


def audit(row):
    """Keep recorded decision identity; never infer a cause just from a loss."""
    if protected(row['symbol'], row['entry_at'], row['exit_at']):
        raise ValueError('PROTECTED_HOLDOUT')
    at = instant(row['decision_at'])
    if not at <= instant(row['entry_at']) < instant(row['exit_at']):
        raise ValueError('INVALID_DECISION_CLOCK')
    known, unavailable = {}, []
    for field, fact in row.get('facts', {}).items():
        if instant(fact['available_at']) <= at and instant(fact['observed_at']) <= at:
            known[field] = fact['value']
        else:
            unavailable.append(field)
    errors = []
    # A documented rule violation is identifiable before the outcome.
    if 'risk_fraction' in known and 'risk_limit' in known and known['risk_fraction'] > known['risk_limit']:
        errors.append('bad_sizing')
    if known.get('scope_eligible') is False:
        errors.append('scope_mismatch')
    outcome = row.get('outcome')
    closed = bool(outcome and instant(outcome['available_at']) >= instant(row['exit_at']))
    if outcome and (not closed or instant(outcome['available_at']) > datetime.now(timezone.utc)):
        raise ValueError('PREMATURE_OUTCOME')
    if outcome and any(isinstance(v, (float, int)) and not math.isfinite(v) for v in outcome.values()):
        raise ValueError('NONFINITE_OUTCOME')
    if closed and row['action'] == 'ABSTAIN' and outcome.get('opportunity_net_return', 0) > 0:
        errors.append('positive_abstention_counterfactual')
    if not errors:
        errors.append('unidentified_requires_evidence')
    return {'decision_id': row['decision_id'], 'strategy': row['strategy'], 'session': row['entry_at'][:10],
            'decision_at': row['decision_at'], 'entry_at': row['entry_at'], 'exit_at': row['exit_at'],
            'source_hash': row['source_hash'], 'decision_hash': row['decision_hash'],
            'known_at_T': known, 'not_available_at_T': sorted(unavailable),
            'original_action': row['action'], 'closed': closed, 'outcome': outcome,
            'error_classes': errors, 'causal_claim': False}


class TrialLedger:
    """Reservations precede data evaluation; failed attempts remain counted."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, body TEXT NOT NULL,
                                               previous TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON events
              BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON events
              BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE UNIQUE INDEX IF NOT EXISTS validation_once
              ON events(json_extract(body,'$.rule_hash'))
              WHERE json_extract(body,'$.event')='VALIDATION_RESERVED';
        ''')
        if not self.db.execute('SELECT 1 FROM events LIMIT 1').fetchone():
            self.append({'event': 'BINDING', 'spec_hash': SPEC_HASH, 'registry_hash': REGISTRY_HASH,
                         'implementation_hash': implementation_hash()})
        self.verify()

    def verify(self):
        previous = '0' * 64
        for expected_seq, (seq, body, prev, hashed) in enumerate(self.db.execute('SELECT * FROM events ORDER BY seq'), 1):
            value = json.loads(body)
            if seq != expected_seq or prev != previous or digest({'seq': seq, 'body': value, 'previous': prev}) != hashed:
                raise ValueError('TRIAL_CHAIN_CORRUPT')
            if seq == 1 and value != {'event': 'BINDING', 'spec_hash': SPEC_HASH,
                                     'registry_hash': REGISTRY_HASH, 'implementation_hash': implementation_hash()}:
                raise ValueError('OLD_TRIAL_BINDING_REJECTED')
            previous = hashed

    def append(self, body):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            last = self.db.execute('SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            seq, previous = (last[0] + 1, last[1]) if last else (1, '0' * 64)
            event = {'seq': seq, 'body': body, 'previous': previous}
            self.db.execute('INSERT INTO events VALUES(?,?,?,?)',
                            (seq, json.dumps(body, sort_keys=True), previous, digest(event)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return seq

    def reservations(self):
        return [json.loads(body) for (body,) in self.db.execute('SELECT body FROM events ORDER BY seq')
                if json.loads(body)['event'] == 'RESERVED']

    def reserve(self, hypothesis, data_hash):
        if hypothesis not in REGISTRY['first_trials']:
            raise ValueError('RULE_NOT_IMPLEMENTED')
        return self.append({'event': 'RESERVED', 'hypothesis': hypothesis, 'variant': 1,
                            'data_hash': data_hash, 'spec_hash': SPEC_HASH, 'registry_hash': REGISTRY_HASH,
                            'at': datetime.now(timezone.utc).isoformat()})

    def reserve_validation(self, rule_hash, data_hash, *, acknowledge_touched=False):
        if not acknowledge_touched:
            raise ValueError('VALIDATION_ALREADY_TOUCHED')
        previous = [json.loads(r[0]) for r in self.db.execute('SELECT body FROM events ORDER BY seq')]
        if any(r.get('event') == 'VALIDATION_RESERVED' and r['rule_hash'] == rule_hash for r in previous):
            raise ValueError('VALIDATION_SINGLE_USE_EXHAUSTED')
        return self.append({'event': 'VALIDATION_RESERVED', 'rule_hash': rule_hash,
                            'data_hash': data_hash, 'role': 'EXPLORATORY_TOUCHED_VALIDATION'})


def implementation_hash():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))})


def keep(hypothesis, row, known):
    at = instant(row['decision_at'])
    if hypothesis == 'H01':
        scheduled = instant(known['earnings_at'])
        return not 0 < (scheduled - instant(row['entry_at'])).total_seconds() <= 86400
    if hypothesis == 'H02':
        published = instant(known['document_published_at'])
        available = instant(known['document_available_at'])
        return known['document_form'] in {'8-K', '10-Q'} and 0 <= (at - published).total_seconds() <= 86400 and available <= at
    if hypothesis == 'H03':
        return 300e6 <= known['market_cap_usd'] <= 10e9 and 2 <= known['planned_holding_days'] <= 10
    raise ValueError('RULE_NOT_IMPLEMENTED')


REQUIRED = {'H01': {'earnings_at'}, 'H02': {'document_form', 'document_published_at', 'document_available_at'},
            'H03': {'market_cap_usd', 'planned_holding_days'}}


def evaluate_trial(rows, hypothesis, ledger):
    # Screen metadata before hashing/reading any prohibited outcome values.
    reject = None
    if any(protected(r['symbol'], r['entry_at'], r['exit_at']) for r in rows):
        reject = 'PROTECTED_HOLDOUT'
    elif len({r['decision_id'] for r in rows}) != len(rows):
        reject = 'DUPLICATE_DECISION_IDENTITIES'
    if reject:
        reservation = ledger.reserve(hypothesis, digest({'rejected': reject}))
        result = {'hypothesis': hypothesis, 'reservation': reservation, 'status': 'BLOCKED',
                  'reason': reject, 'metric': None, 'stress': None, 'passes_development': False,
                  'promotion': 'NONE', 'family_size': max(SPEC['family_size_floor'], len(ledger.reservations()))}
        ledger.append({'event': 'RESULT', 'result': result})
        return result
    data_hash = digest(rows)
    reservation = ledger.reserve(hypothesis, data_hash)
    family = max(SPEC['family_size_floor'], len(ledger.reservations()))
    samples, stress, reasons = [], [], Counter()
    eligible = 0
    for row in rows:
        # No outcomes from validation, forward, or a crossing horizon enter the metric.
        lo, hi = SPEC['development']
        if not (lo <= row['entry_at'][:10] <= hi and row['exit_at'][:10] <= hi):
            reasons['OUTSIDE_DEVELOPMENT'] += 1
            continue
        checked = audit(row)
        if not checked['closed']:
            reasons['MISSING_CLOSED_OUTCOME'] += 1
            continue
        eligible += 1
        if REQUIRED[hypothesis] - set(checked['known_at_T']):
            reasons['MISSING_POINT_IN_TIME_FEATURES'] += 1
            continue
        outcome = row['outcome']
        if outcome.get('net_contribution') is None or outcome.get('cost_basis') != 'NET_FULL_COSTS':
            reasons['MISSING_COMPARABLE_NET_CONTRIBUTION'] += 1
            continue
        retained = keep(hypothesis, row, checked['known_at_T'])
        delta = 0.0 if retained else -outcome['net_contribution']
        # Stress only the candidate's retained entries, so skipping costs alone cannot pass.
        extra = SPEC['cost_stress_roundtrip'] * abs(row['weight']) if retained and row['action'] != 'ABSTAIN' else 0.0
        samples.append((checked['session'], delta))
        stress.append((checked['session'], delta - extra))
    blocking = not eligible or any(k != 'OUTSIDE_DEVELOPMENT' for k in reasons)
    result = {'hypothesis': hypothesis, 'reservation': reservation, 'eligible_dev': eligible,
              'exclusions': dict(reasons), 'data_hash': data_hash, 'family_size': family,
              'status': 'BLOCKED' if blocking else 'TESTED',
              'reason': 'NO_CAUSAL_DEVELOPMENT_SAMPLE' if not eligible else 'INCOMPLETE_DEVELOPMENT_EVIDENCE' if blocking else None,
            'metric': stats.paired_test(samples, family=family) if not blocking else None,
              'stress': stats.paired_test(stress, family=family) if not blocking else None}
    metric, stressed = result['metric'], result['stress']
    result['passes_development'] = bool(metric and metric['sessions'] >= SPEC['min_dev_sessions']
                                       and metric['mean_delta'] > 0 and metric['p_adjusted'] <= SPEC['alpha']
                                       and stressed['mean_delta'] > 0 and stressed['p_adjusted'] <= SPEC['alpha'])
    result['promotion'] = 'NOT_PROMOTED_REQUIRES_NEW_FORWARD_SHADOW_AB'
    ledger.append({'event': 'RESULT', 'result': result})
    return result


def promotion_gate(pairs, *, preregistered, validation_receipt, family=8):
    """Audit a proposed forward result; this function cannot change a strategy."""
    if (not preregistered or not pairs or not isinstance(validation_receipt, dict)
            or validation_receipt.get('role') != 'EXPLORATORY_TOUCHED_VALIDATION'
            or not validation_receipt.get('completed') or not validation_receipt.get('rule_hash')):
        return False
    if len({p['decision_id'] for p in pairs}) != len(pairs):
        return False
    if any(not p.get('mature') or not p.get('risk_ok') or p.get('mode') != 'SHADOW_AB'
           or p.get('candidate_net') is None or p.get('current_net') is None
           or p.get('candidate_stressed_net') is None
           or p.get('rule_hash') != validation_receipt['rule_hash']
           or not p.get('registered_at') or instant(p['registered_at']) >= instant(p['decision_at']) for p in pairs):
        return False
    result = stats.paired_test([(p['session'], p['candidate_net'] - p['current_net']) for p in pairs], family=family)
    stressed = stats.paired_test([(p['session'], p['candidate_stressed_net'] - p['current_net']) for p in pairs], family=family)
    return (result['sessions'] >= SPEC['min_shadow_pairs'] and result['mean_delta'] > 0
            and result['p_adjusted'] <= SPEC['alpha'] and stressed['mean_delta'] > 0
            and stressed['p_adjusted'] <= SPEC['alpha'])
