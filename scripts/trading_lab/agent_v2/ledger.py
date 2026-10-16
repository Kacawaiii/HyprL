"""Private, append-only SQLite events, atomic batches, immutable binding and hash chain."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import sqlite3
import shutil

from .contracts import Blocked, canonical, digest, instant, iso, strict_json, utc
from .spec import PREREG_HASH, SPEC_HASH, configuration


def implementation_hash():
    folder = Path(__file__).resolve().parent
    return digest({str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(folder.rglob('*')) if p.suffix in ('.py', '.txt')})


class Ledger:
    def __init__(self, root, *, read_only=False):
        self.root = Path(root).resolve()
        self.read_only = read_only
        path = self.root / 'shadow.sqlite'
        if read_only:
            self.db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)
            self.db.execute('PRAGMA query_only=ON')
        else:
            if not path.is_file():
                raise Blocked('RUNTIME_NOT_INITIALIZED')
            self.db = sqlite3.connect(path, timeout=5)
        self.verify()

    @classmethod
    def initialize(cls, root, mode, code_sha, at=None):
        at = at or utc()
        config = configuration()
        if mode not in ('SYNTHETIC', 'SHADOW_FORWARD') or len(code_sha) != 40 or any(c not in '0123456789abcdef' for c in code_sha):
            raise Blocked('INVALID_BINDING')
        root = Path(root).resolve()
        # A caller must choose a NEW directory: prevents binding or modifying any v1 runtime.
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(root, 0o700)
        path = root / 'shadow.sqlite'
        db = sqlite3.connect(path)
        os.chmod(path, 0o600)
        try:
            db.executescript('''
                PRAGMA synchronous=FULL;
                CREATE TABLE events(seq INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL,
                                    body TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL UNIQUE);
                CREATE TRIGGER no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
                CREATE TRIGGER no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
            ''')
            body = {'spec_hash': SPEC_HASH, 'preregistration_hash': PREREG_HASH, 'config': config,
                    'config_hash': digest(config), 'mode': mode, 'code_sha': code_sha,
                    'implementation_hash': implementation_hash()}
            row = {'seq': 1, 'at': iso(at), 'kind': 'binding', 'body': body, 'previous': '0' * 64}
            db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                       (1, row['at'], 'binding', canonical(body), row['previous'], digest(row)))
            db.commit()
        finally:
            db.close()
        return cls(root)

    def close(self):
        self.db.close()

    def records(self, kind=None):
        sql = 'SELECT seq,at,kind,body,previous,hash FROM events'
        args = ()
        if kind:
            sql += ' WHERE kind=?'
            args = (kind,)
        return [{'seq': seq, 'at': at, 'kind': k, 'body': strict_json(body), 'previous': previous, 'hash': h}
                for seq, at, k, body, previous, h in self.db.execute(sql + ' ORDER BY seq', args)]

    def verify(self):
        rows = self.records()
        previous = '0' * 64
        last_at = None
        for seq, row in enumerate(rows, 1):
            if (row['seq'] != seq or row['previous'] != previous
                    or digest({k: v for k, v in row.items() if k != 'hash'}) != row['hash']
                    or last_at is not None and instant(row['at']) < instant(last_at)):
                raise Blocked('LEDGER_INTEGRITY')
            previous, last_at = row['hash'], row['at']
        config = configuration()
        if not rows or rows[0]['kind'] != 'binding':
            raise Blocked('BINDING_MISSING')
        self.binding = rows[0]['body']
        if (self.binding['spec_hash'] != SPEC_HASH or self.binding['preregistration_hash'] != PREREG_HASH
                or self.binding['implementation_hash'] != implementation_hash()
                or self.binding['config'] != config or self.binding['config_hash'] != digest(config)):
            raise Blocked('FROZEN_CONFIG_MISMATCH')
        return previous

    @contextmanager
    def owned(self):
        if self.read_only:
            raise Blocked('READ_ONLY')
        with (self.root / 'writer.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Blocked('WRITER_BUSY') from None
            try:
                self.verify()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def append(self, events, at):
        if self.read_only:
            raise Blocked('READ_ONLY')
        if (self.root / 'shadow.sqlite').stat().st_size > 512 * 1024 * 1024 or shutil.disk_usage(self.root).free < 256 * 1024 * 1024:
            raise Blocked('STORAGE_BUDGET')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            previous_row = self.db.execute('SELECT seq,hash,at FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            seq, previous, last_at = previous_row
            if at < instant(last_at):
                raise Blocked('CLOCK_REGRESSION')
            for kind, body in events:
                if kind == 'binding':
                    raise Blocked('REBIND_FORBIDDEN')
                seq += 1
                row = {'seq': seq, 'at': iso(at), 'kind': kind, 'body': body, 'previous': previous}
                h = digest(row)
                self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                                (seq, row['at'], kind, canonical(body), previous, h))
                previous = h
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def state(self):
        state = {'cash': self.binding['config']['initial_equity'], 'positions': {}, 'pending': {},
                 'marks': [], 'closed': [], 'vetoes': [], 'blocked': []}
        for row in self.records():
            body, kind = row['body'], row['kind']
            if kind == 'approved':
                state['pending'][body['plan_id']] = body
            elif kind in ('expired', 'fill_veto'):
                state['pending'].pop(body['plan_id'], None)
            elif kind == 'entry_fill':
                state['pending'].pop(body['plan_id'], None)
                state['cash'] -= body['qty'] * body['entry'] + body['entry_fee']
                state['positions'][body['symbol']] = body
            elif kind == 'exit_fill':
                pos = state['positions'].pop(body['symbol'])
                state['cash'] += body['qty'] * body['exit'] - body['exit_fee']
                state['closed'].append({**pos, **body})
            elif kind == 'trail':
                pos = state['positions'][body['symbol']]
                if body['stop'] < pos['stop']:
                    raise Blocked('STOP_WIDENED')
                pos.update(body)
            elif kind == 'mark':
                state['marks'].append({'at': row['at'], **body})
            elif kind == 'veto':
                state['vetoes'].append(body)
            elif kind == 'blocked':
                state['blocked'].append({'at': row['at'], **body})
        return state
