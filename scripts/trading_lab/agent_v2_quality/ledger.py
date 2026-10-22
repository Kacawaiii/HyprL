"""Own append-only store. No migration, rebinding, or writes to revision one."""
from pathlib import Path
import os
import sqlite3

from scripts.trading_lab.agent_v2.ledger import Ledger
from .contracts import Blocked, canonical, digest, instant, iso, utc
from .spec import SPEC_HASH, PREREG_HASH, configuration, implementation_hash


class Journal(Ledger):
    @classmethod
    def initialize(cls, root, mode, code_sha, at=None):
        at = at or utc()
        if mode not in ('SYNTHETIC', 'SHADOW_FORWARD') or len(code_sha) != 40 or any(
                c not in '0123456789abcdef' for c in code_sha):
            raise Blocked('INVALID_BINDING')
        root = Path(root).resolve()
        config = configuration()
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        with sqlite3.connect(root / 'shadow.sqlite') as db:
            db.executescript('''
                PRAGMA synchronous=FULL;
                CREATE TABLE events(seq INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL,
                                    body TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL UNIQUE);
                CREATE TRIGGER no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
                CREATE TRIGGER no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
            ''')
            binding = {'spec_hash': SPEC_HASH, 'preregistration_hash': PREREG_HASH,
                       'mode': mode, 'code_sha': code_sha, 'implementation_hash': implementation_hash(),
                       'config': config, 'config_hash': digest(config)}
            row = {'seq': 1, 'at': iso(at), 'kind': 'binding', 'body': binding, 'previous': '0' * 64}
            db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                       (1, row['at'], 'binding', canonical(binding), row['previous'], digest(row)))
        os.chmod(root / 'shadow.sqlite', 0o600)
        return cls(root)

    def verify(self):
        rows = self.records()
        previous, last_at = '0' * 64, None
        for seq, row in enumerate(rows, 1):
            if (row['seq'] != seq or row['previous'] != previous
                    or digest({k: v for k, v in row.items() if k != 'hash'}) != row['hash']
                    or last_at is not None and instant(row['at']) < instant(last_at)):
                raise Blocked('LEDGER_INTEGRITY')
            previous, last_at = row['hash'], row['at']
        if not rows or rows[0]['kind'] != 'binding':
            raise Blocked('BINDING_MISSING')
        self.binding = rows[0]['body']
        config = configuration()
        if (self.binding.get('spec_hash') != SPEC_HASH or self.binding.get('preregistration_hash') != PREREG_HASH
                or self.binding.get('implementation_hash') != implementation_hash()
                or self.binding.get('config') != config or self.binding.get('config_hash') != digest(config)):
            raise Blocked('FROZEN_CONFIG_MISMATCH')
        return previous
