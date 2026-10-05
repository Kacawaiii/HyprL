"""Private SQLite evidence, append-only with bounded writes and verified reads."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

from scripts.trading_lab.platform.contracts import PredictionRecord, LabelRecord, ExperimentManifest, enrich_prediction, timestamp
from scripts.trading_lab.research.monitoring import MonitoringReference
from scripts.trading_lab.research.contracts import number
from scripts.trading_lab.research.contracts import (
    Hypothesis, Trial, PredictionEvidence, ExecutionObservation, InferenceObservation, DecisionObservation)
from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical

SCHEMA = 'research-observability-store-v1'
CLASSES = {'hypothesis': Hypothesis, 'trial': Trial, 'prediction': PredictionRecord, 'label': LabelRecord,
           'inputs': PredictionEvidence, 'execution': ExecutionObservation, 'inference': InferenceObservation, 'decision': DecisionObservation, 'reference': MonitoringReference, 'experiment': ExperimentManifest}
MAX_RECORDS = 100000
MAX_BYTES = 256 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024


class IntegrityError(ValueError):
    """Evidence is not bound to its recorded identity."""


def now():
    return datetime.now(timezone.utc).isoformat()


class ResearchStore:
    def __init__(self, root, *, read_only=False):
        self._reader = None
        self._reader_lock = threading.RLock()
        self.root = Path(root)
        self.path = self.root / 'research.sqlite'
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError('research store unavailable')
        else:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self.connect() as db:
                db.executescript('''
                    CREATE TABLE IF NOT EXISTS metadata (schema TEXT NOT NULL, record_count INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE IF NOT EXISTS records (
                        sequence INTEGER PRIMARY KEY, kind TEXT NOT NULL, identity TEXT NOT NULL UNIQUE,
                        object_id TEXT NOT NULL, parent_id TEXT NOT NULL, recorded_at TEXT NOT NULL,
                        payload TEXT NOT NULL, previous_hash TEXT NOT NULL, chain_hash TEXT NOT NULL);
                    CREATE INDEX IF NOT EXISTS records_kind ON records(kind,sequence);
                    CREATE INDEX IF NOT EXISTS records_parent ON records(kind,parent_id,sequence);
                    CREATE INDEX IF NOT EXISTS records_object ON records(kind,object_id,sequence);
                    CREATE TRIGGER IF NOT EXISTS no_record_update BEFORE UPDATE ON records
                        BEGIN SELECT RAISE(ABORT,'evidence is append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS no_record_delete BEFORE DELETE ON records
                        BEGIN SELECT RAISE(ABORT,'evidence is append-only'); END;
                ''')
                if not db.execute('SELECT 1 FROM metadata').fetchone():
                    db.execute('INSERT INTO metadata(schema) VALUES(?)', (SCHEMA,))
        with self.connect() as db:
            rows = db.execute('SELECT schema FROM metadata').fetchall()
            if len(rows) != 1 or rows[0][0] != SCHEMA:
                raise IntegrityError('research store schema mismatch; no implicit migration')

    @contextmanager
    def connect(self):
        if self.read_only:
            from scripts.trading_lab.sources.store import ReadOnlyDatabase
            def admit(db):
                rows = db.execute('SELECT schema FROM metadata').fetchall()
                if len(rows) != 1 or rows[0][0] != SCHEMA:
                    raise IntegrityError('research store schema mismatch; no implicit migration')
            # SQLite mode=ro alone still writes a WAL index in an archive.
            # This proven reader never opens the original with SQLite.
            # Refresh only when the source files change; a monitoring read may
            # perform thousands of queries over the same verified archive.
            with self._reader_lock:
                if self._reader is None:
                    self._reader = ReadOnlyDatabase(self.path, validator=admit)
                else:
                    self._reader.refresh()
                self._reader.conn.row_factory = sqlite3.Row
                yield self._reader.conn
            return
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        try:
            if not self.read_only:
                db.execute('PRAGMA journal_mode=WAL')
                db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def close(self):
        with self._reader_lock:
            if self._reader is not None:
                self._reader.close()
                self._reader = None

    def __del__(self):
        self.close()

    @staticmethod
    def _chain(row):
        return sha256_canonical({k: row[k] for k in ('sequence', 'kind', 'identity', 'object_id',
                                                   'parent_id', 'recorded_at', 'previous_hash')})

    def _decode(self, row):
        try:
            payload = json.loads(row['payload'])
            if sha256_canonical(payload) != row['identity'] or self._chain(row) != row['chain_hash']:
                raise ValueError('hash mismatch')
            if row['kind'] in CLASSES:
                record = CLASSES[row['kind']].from_dict(payload)
                if record.identity != row['identity']:
                    raise ValueError('noncanonical contract')
            return {'sequence': row['sequence'], 'identity': row['identity'], 'recorded_at': row['recorded_at'],
                    'chain_hash': row['chain_hash'], 'payload': payload}
        except (KeyError, TypeError, ValueError):
            raise IntegrityError('research evidence integrity failure') from None

    def _append(self, db, kind, payload, object_id, parent_id, recorded_at):
        if self.read_only:
            raise ValueError('research store is read-only')
        encoded = canonical_bytes(payload)
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError('research record exceeds size budget')
        identity = sha256_canonical(payload)
        existing = db.execute('SELECT * FROM records WHERE identity=?', (identity,)).fetchone()
        if existing:
            self._decode(existing)
            if (existing['kind'], existing['object_id'], existing['parent_id']) != (kind, object_id, parent_id):
                raise ValueError('identity already registered under another binding')
            return identity
        count, size = db.execute('SELECT record_count,bytes FROM metadata').fetchone()
        if count >= MAX_RECORDS or size + len(encoded) > MAX_BYTES:
            raise ValueError('persistent research budget exhausted')
        previous = db.execute('SELECT sequence,chain_hash FROM records ORDER BY sequence DESC LIMIT 1').fetchone()
        row = {'sequence': previous['sequence'] + 1 if previous else 1, 'kind': kind, 'identity': identity,
               'object_id': object_id, 'parent_id': parent_id, 'recorded_at': timestamp(recorded_at),
               'previous_hash': previous['chain_hash'] if previous else '0' * 64}
        db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)',
                   (row['sequence'], kind, identity, object_id, parent_id, row['recorded_at'], encoded.decode(),
                    row['previous_hash'], self._chain(row)))
        db.execute('UPDATE metadata SET record_count=record_count+1,bytes=bytes+?', (len(encoded),))
        return identity

    def append(self, kind, record, *, object_id=None, parent_id='', recorded_at=None):
        if kind in CLASSES and not isinstance(record, CLASSES[kind]):
            raise ValueError('wrong evidence contract')
        payload = record.to_dict() if hasattr(record, 'to_dict') else record
        if kind not in {*CLASSES, 'reference', 'experiment', 'replay-summary'}:
            raise ValueError('unsupported evidence kind')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            id_field = {'inference': 'observation_id', 'decision': 'observation_id', 'execution': 'observation_id',
                        'reference': 'reference_id', 'inputs': 'prediction_id', 'prediction': 'prediction_id',
                        'label': 'label_id', 'hypothesis': 'hypothesis_id', 'trial': 'trial_id', 'experiment': 'experiment_id'}.get(kind)
            identifier = object_id or (payload.get(id_field) if id_field else None) or sha256_canonical(payload)
            if kind in {'reference', 'inputs', 'inference', 'decision', 'execution', 'replay-summary'}:
                old = db.execute('SELECT * FROM records WHERE kind=? AND object_id=?', (kind, identifier)).fetchone()
                if old and self._decode(old)['identity'] != sha256_canonical(payload):
                    raise ValueError('observation identity immutable; use a new version or identifier')
            return self._append(db, kind, payload, identifier, parent_id,
                                recorded_at or getattr(record, 'recorded_at', None) or now())

    def get(self, identity, *, kind=None):
        with self.connect() as db:
            row = db.execute('SELECT * FROM records WHERE identity=?', (identity,)).fetchone()
            if row is None or (kind and row['kind'] != kind):
                raise KeyError('research resource not found')
            return self._decode(row)

    def records(self, kind, *, parent_id=None, object_id=None, as_of=None, after=0, limit=10000):
        if type(limit) is not int or not 1 <= limit <= 10000 or type(after) is not int or after < 0:
            raise ValueError('invalid research page bounds')
        clauses, args = ['kind=?', 'sequence>?'], [kind, after]
        for key, value in (('parent_id', parent_id), ('object_id', object_id)):
            if value is not None:
                clauses.append(key + '=?')
                args.append(value)
        if as_of is not None:
            clauses.append('recorded_at<=?')
            args.append(timestamp(as_of))
        with self.connect() as db:
            rows = db.execute('SELECT * FROM records WHERE ' + ' AND '.join(clauses) + ' ORDER BY sequence LIMIT ?',
                              (*args, limit)).fetchall()
            return [self._decode(row) for row in rows]

    def verify(self):
        with self.connect() as db:
            previous, count = '0' * 64, 0
            for row in db.execute('SELECT * FROM records ORDER BY sequence'):
                self._decode(row)
                count += 1
                if row['sequence'] != count or row['previous_hash'] != previous:
                    raise IntegrityError('research chain broken')
                previous = row['chain_hash']
            recorded_count, size = db.execute('SELECT record_count,bytes FROM metadata').fetchone()
            actual_size = db.execute('SELECT coalesce(sum(length(CAST(payload AS BLOB))),0) FROM records').fetchone()[0]
            if recorded_count != count or size != actual_size:
                raise IntegrityError('research store budget counters mismatch')
            return {'schema': SCHEMA, 'records': count, 'head_hash': previous, 'verified': True}

    def register(self, hypothesis, *, recorded_at=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute("SELECT * FROM records WHERE kind='hypothesis' AND object_id=?", (hypothesis.hypothesis_id,)).fetchone()
            if old and self._decode(old)['identity'] != hypothesis.identity:
                raise ValueError('hypothesis immutable; create a new version and identifier')
            return self._append(db, 'hypothesis', hypothesis.to_dict(), hypothesis.hypothesis_id, '', recorded_at or now())

    def start_trial(self, hypothesis, prepared, *, trial_id, recorded_at=None):
        if prepared.status != 'PREPARED' or prepared.synthetic is not True or hypothesis.synthetic is not True:
            raise ValueError('only prepared synthetic experiments can be started')
        if prepared.to_dict()['decision_criteria'] != hypothesis.to_dict()['decision_criteria']:
            raise ValueError('trial must use frozen hypothesis criteria')
        if hypothesis.scope != 'EXPLORATORY':
            raise ValueError('protocol comparison cannot run in this slice')
        for key in ('splits', 'baselines', 'costs'):
            if prepared.to_dict()[key] != hypothesis.to_dict()[key]:
                raise ValueError('trial configuration differs from frozen hypothesis')
        if prepared.dataset_hash != hypothesis.population.get('dataset_hash') or prepared.budgets['wall_seconds'] > hypothesis.budgets['wall_seconds']:
            raise ValueError('trial dataset or resource budget differs from hypothesis')
        if hypothesis.population.get('rows', 0) > hypothesis.budgets['max_rows']:
            raise ValueError('trial population exceeds row budget')
        at = recorded_at or now()
        trial = Trial(trial_id=trial_id, hypothesis_hash=hypothesis.identity, experiment_hash=prepared.identity,
                      criteria_hash=hypothesis.criteria_hash, state='PREPARED', outcome='PENDING', recorded_at=at,
                      evidence={'prepared_hash': prepared.identity, 'synthetic': True, 'criteria_frozen_before_result': True})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            registered = db.execute("SELECT * FROM records WHERE kind='hypothesis' AND identity=?", (hypothesis.identity,)).fetchone()
            if not registered:
                raise ValueError('register hypothesis before reserving trials')
            self._decode(registered)
            if db.execute("SELECT 1 FROM records WHERE kind='trial' AND object_id=?", (trial_id,)).fetchone():
                raise ValueError('trial id already reserved')
            count = db.execute("SELECT count(DISTINCT object_id) FROM records WHERE kind='trial' AND parent_id=?", (hypothesis.identity,)).fetchone()[0]
            if count >= hypothesis.budgets['max_trials']:
                raise ValueError('hypothesis trial budget exhausted')
            self._append(db, 'experiment', prepared.to_dict(), prepared.experiment_id, hypothesis.identity, at)
            return self._append(db, 'trial', trial.to_dict(), trial_id, hypothesis.identity, at)

    def observe_trial(self, trial_id, *, state, outcome, evidence, recorded_at=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            history = [self._decode(r) for r in db.execute(
                "SELECT * FROM records WHERE kind='trial' AND object_id=? ORDER BY sequence", (trial_id,))]
            if not history:
                raise KeyError('trial not reserved')
            original, previous = Trial.from_dict(history[0]['payload']), Trial.from_dict(history[-1]['payload'])
            if previous.state in {'COMPLETE', 'FAILED', 'ABANDONED'}:
                raise ValueError('terminal trial immutable; reserve a new trial')
            transitions = {'PREPARED': {'RUNNING', 'COMPLETE', 'FAILED', 'ABANDONED', 'BLOCKED'},
                           'RUNNING': {'COMPLETE', 'FAILED', 'ABANDONED', 'BLOCKED'}, 'BLOCKED': {'RUNNING', 'ABANDONED'}}
            if state not in transitions[previous.state]:
                raise ValueError('invalid trial transition')
            at = timestamp(recorded_at or now())
            if at < previous.recorded_at:
                raise ValueError('trial observations cannot go backwards')
            observation = Trial(trial_id=trial_id, hypothesis_hash=original.hypothesis_hash,
                experiment_hash=original.experiment_hash, criteria_hash=original.criteria_hash,
                state=state, outcome=outcome, recorded_at=at, evidence=evidence)
            return self._append(db, 'trial', observation.to_dict(), trial_id, original.hypothesis_hash, at)

    def issue(self, prediction, evidence):
        if (evidence.prediction_id, evidence.prediction_hash) != (prediction.prediction_id, prediction.identity):
            raise ValueError('inputs do not bind prediction')
        if evidence.recorded_at < prediction.decision_at:
            raise ValueError('prediction cannot be recorded before decision')
        if evidence.features is not None and sha256_canonical(evidence.to_dict()['features']) != prediction.features_hash:
            raise ValueError('actual feature values do not match prediction')
        if sha256_canonical(evidence.to_dict()['snapshot']) != prediction.snapshot_hash:
            raise ValueError('actual snapshot does not match prediction')
        if evidence.snapshot.get('schema') == 'information-snapshot-v1':
            from scripts.trading_lab.platform.contracts import InformationSnapshot
            snapshot = InformationSnapshot.from_dict(evidence.snapshot)
            if snapshot.as_of != prediction.decision_at or prediction.product not in snapshot.products:
                raise ValueError('snapshot decision binding mismatch')
            available_events = {e['event_id'] for e in snapshot.events}
            if not set(prediction.event_ids) <= available_events:
                raise ValueError('used event absent from snapshot')
        if prediction.uncertainty is not None and not prediction.uncertainty.get('method'):
            raise ValueError('uncertainty requires an identified method')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute("SELECT * FROM records WHERE kind='prediction' AND object_id=?", (prediction.prediction_id,)).fetchone()
            if existing and self._decode(existing)['identity'] != prediction.identity:
                raise ValueError('issued prediction immutable')
            old_inputs = db.execute("SELECT * FROM records WHERE kind='inputs' AND parent_id=?", (prediction.identity,)).fetchone()
            if old_inputs and self._decode(old_inputs)['identity'] != evidence.identity:
                raise ValueError('issued input evidence immutable')
            self._append(db, 'prediction', prediction.to_dict(), prediction.prediction_id, '', evidence.recorded_at)
            self._append(db, 'inputs', evidence.to_dict(), prediction.prediction_id, prediction.identity, evidence.recorded_at)
        return prediction.identity

    def append_label(self, label):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM records WHERE kind='prediction' AND identity=?", (label.prediction_hash,)).fetchone()
            if row is None:
                raise KeyError('prediction not issued')
            item = self._decode(row)
            prediction = PredictionRecord.from_dict(item['payload'])
            enrich_prediction(prediction, (label,), as_of=label.recorded_at)
            if label.recorded_at < item['recorded_at']:
                raise ValueError('label arrival cannot precede issued prediction')
            if label.target == 'forward_return':
                number(label.value)
            for old in db.execute("SELECT * FROM records WHERE kind='label' AND object_id=?", (label.label_id,)):
                previous = self._decode(old)
                if previous['payload']['prediction_hash'] != label.prediction_hash:
                    raise ValueError('label identity cannot bind another prediction')
                if previous['payload']['version'] == label.version and previous['identity'] != label.identity:
                    raise ValueError('label version immutable; append a new correction version')
                if previous['payload']['recorded_at'] > label.recorded_at:
                    raise ValueError('label arrival clock cannot go backwards')
            return self._append(db, 'label', label.to_dict(), label.label_id, label.prediction_hash, label.recorded_at)

    def append_execution(self, observation):
        item = self.get(observation.prediction_hash, kind='prediction')
        prediction = PredictionRecord.from_dict(item['payload'])
        if (observation.prediction_id != prediction.prediction_id or observation.available_at < prediction.decision_at
                or observation.recorded_at < item['recorded_at']):
            raise ValueError('execution does not bind an issued decision')
        return self.append('execution', observation, object_id=observation.observation_id, parent_id=prediction.identity)

    def append_decision(self, observation):
        item = self.get(observation.prediction_hash, kind='prediction')
        prediction = PredictionRecord.from_dict(item['payload'])
        if (observation.prediction_id != prediction.prediction_id or observation.available_at < prediction.decision_at
                or observation.recorded_at < item['recorded_at']):
            raise ValueError('decision does not bind an issued prediction')
        return self.append('decision', observation, object_id=observation.observation_id, parent_id=prediction.identity)

    def prediction_view(self, identity, *, as_of):
        at = timestamp(as_of)
        item = self.get(identity, kind='prediction')
        prediction = PredictionRecord.from_dict(item['payload'])
        if item['recorded_at'] > at or prediction.decision_at > at:
            raise KeyError('prediction not recorded at this instant')
        label_rows = self.records('label', parent_id=identity, as_of=at)
        labels = tuple(LabelRecord.from_dict(r['payload']) for r in label_rows)
        view = enrich_prediction(prediction, labels, as_of=at)
        label_sequence = {r['identity']: r['sequence'] for r in label_rows}
        view['labels'].sort(key=lambda label: (label['recorded_at'], label_sequence[label['identity']]))
        executions = [r for r in self.records('execution', parent_id=identity, as_of=at) if r['payload']['available_at'] <= at]
        inputs = self.records('inputs', parent_id=identity, as_of=at)
        decisions = [r for r in self.records('decision', parent_id=identity, as_of=at) if r['payload']['available_at'] <= at]
        return {**view, 'decisions': decisions, 'recorded_at': item['recorded_at'], 'inputs': inputs[0]['payload'] if inputs else None,
                'executions': executions, 'execution_state': executions[-1]['payload']['state'] if executions else 'PENDING',
                'limitations': ['read-time enrichment; original prediction identity unchanged']}
