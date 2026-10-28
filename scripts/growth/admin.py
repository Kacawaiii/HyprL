"""Operator-only local rights/export tool. Stop the referral service before erasure.

Export writes private JSON to an explicitly chosen mode-0600 file, never stdout.
No Discord, Stripe or other network calls.
"""
import argparse
import json
import os
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--member', required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--export-to')
    operation.add_argument('--erase', action='store_true')
    args = parser.parse_args()
    if not args.member.isdigit():
        parser.error('numeric Discord member required')
    db = sqlite3.connect('file:' + os.path.abspath(args.ledger) + '?mode=' + ('rw' if args.erase else 'ro'), uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    if args.erase:
        db.execute('PRAGMA secure_delete=ON')
    try:
        if args.erase:
            with db:
                db.execute('DELETE FROM sponsors WHERE member=?', (args.member,))
                db.execute('DELETE FROM arrivals WHERE member=?', (args.member,))
            # Remove deleted rows from the SQLite file; no raw record remains in freelist pages.
            db.execute('VACUUM')
            print('Requested referral data erased; revoke its Discord invitation separately.')
        else:
            personal = {}
            for table, field in [('sponsors', 'member'), ('arrivals', 'member'), ('rewards', 'sponsor'), ('credits', 'sponsor')]:
                personal[table] = [dict(r) for r in db.execute(f'SELECT * FROM {table} WHERE {field}=?', (args.member,))]
            # Never export other members referred by this person.
            fd = os.open(args.export_to, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(personal, stream, ensure_ascii=False, indent=2)
            print('Individual export written privately; no record printed.')
    finally:
        db.close()


if __name__ == '__main__':
    main()
