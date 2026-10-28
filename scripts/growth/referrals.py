"""Private, durable referral ledger; Discord and billing remain adapters."""

import calendar
import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

WEEK = 7 * 86400


def expires(at):
    date = datetime.fromtimestamp(at, timezone.utc)
    return date.replace(year=date.year + 2, day=min(date.day, calendar.monthrange(date.year + 2, date.month)[1])).timestamp()


def attributed(before, after):
    """No guessing: exactly one use on exactly one known invite, across ALL invites."""
    if before is None:
        return None
    if any(code not in before and uses for code, uses in after.items()):
        return None
    if any(after.get(code, 0) < uses for code, uses in before.items()):
        return None
    changes = [(code, uses - before.get(code, 0)) for code, uses in after.items() if uses > before.get(code, 0)]
    return changes[0][0] if len(changes) == 1 and changes[0][1] == 1 else None


class Ledger:
    def __init__(self, file, alias_key):
        self.file, self.alias_key = Path(file), alias_key
        self.file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.file.parent.stat().st_mode & 0o077:
            raise ValueError("private ledger directory must be mode 0700")
        if not self.file.exists():
            fd = os.open(self.file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        if self.file.is_symlink() or self.file.stat().st_mode & 0o077:
            raise ValueError("private ledger file must be mode 0600, no symlink")
        self.db = sqlite3.connect(self.file)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS sponsors (
            member TEXT PRIMARY KEY, code TEXT UNIQUE NOT NULL, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS arrivals (
            member TEXT PRIMARY KEY, sponsor TEXT REFERENCES sponsors(member) ON DELETE SET NULL,
            joined REAL NOT NULL, status TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS rewards (
            sponsor TEXT REFERENCES sponsors(member) ON DELETE CASCADE,
            threshold INTEGER NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            created REAL NOT NULL, PRIMARY KEY(sponsor, threshold));
          CREATE TABLE IF NOT EXISTS posts (week TEXT PRIMARY KEY, state TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS credits (
            sponsor TEXT PRIMARY KEY REFERENCES sponsors(member) ON DELETE CASCADE,
            months INTEGER NOT NULL CHECK(months=1), state TEXT NOT NULL DEFAULT 'pending_stripe');
        """)

    def close(self):
        self.db.close()

    def invite(self, member):
        row = self.db.execute("SELECT code FROM sponsors WHERE member=?", (str(member),)).fetchone()
        return row[0] if row else None

    def register(self, member, code, now):
        with self.db:
            self.db.execute("INSERT INTO sponsors VALUES (?,?,?)", (str(member), code, now))

    def join(self, member, code, created, joined, *, bot=False):
        sponsor = self.db.execute("SELECT member FROM sponsors WHERE code=?", (code,)).fetchone()
        owner = sponsor[0] if sponsor else None
        eligible = owner is not None and owner != str(member) and not bot and joined - created >= WEEK
        # Keep every observed arrival, even unattributed: a return cannot acquire a new sponsor.
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO arrivals VALUES (?,?,?,?)", (str(member), owner, joined, "pending" if eligible else "rejected"))

    def leave(self, member, at):
        with self.db:
            self.db.execute("UPDATE arrivals SET status='rejected' WHERE member=? AND status='pending' AND joined+?>?", (str(member), WEEK, at))
            # At the exact boundary, a departure satisfies seven complete days.
            self.db.execute("UPDATE arrivals SET status='valid' WHERE member=? AND status='pending' AND joined+?<=?", (str(member), WEEK, at))

    def pending(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM arrivals WHERE status='pending'")]

    def mature(self, now, present):
        """present maps member ID to current joined_at, obtained from Discord, not a stale cache."""
        with self.db:
            for row in self.pending():
                if now < row['joined'] + WEEK:
                    continue
                current = present.get(row['member'])
                valid = current is not None and current == row['joined']
                self.db.execute("UPDATE arrivals SET status=? WHERE member=?", ('valid' if valid else 'rejected', row['member']))
            for r in self.db.execute("SELECT sponsor, COUNT(*) AS n FROM arrivals WHERE status='valid' AND sponsor IS NOT NULL GROUP BY sponsor").fetchall():
                for threshold, kind in [(3, 'premium_month_credit'), (10, 'fondateur_place')]:
                    if r['n'] >= threshold:
                        self.db.execute("INSERT OR IGNORE INTO rewards(sponsor,threshold,kind,created) VALUES (?,?,?,?)", (r['sponsor'], threshold, kind, now))
                        if threshold == 3:
                            self.db.execute('INSERT OR IGNORE INTO credits(sponsor,months) VALUES (?,1)', (r['sponsor'],))

    def rewards(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM rewards ORDER BY sponsor,threshold")]

    def mark_role(self, sponsor, threshold):
        with self.db:
            self.db.execute("UPDATE rewards SET state='role_granted' WHERE sponsor=? AND threshold=?", (str(sponsor), threshold))

    def alias(self, member):
        return 'Parrain-' + hmac.new(self.alias_key.encode(), str(member).encode(), hashlib.sha256).hexdigest()[:10]

    def counts(self, member):
        return {r[0]: r[1] for r in self.db.execute("SELECT status,COUNT(*) FROM arrivals WHERE sponsor=? GROUP BY status", (str(member),))}

    def leaderboard(self):
        rows = self.db.execute("SELECT sponsor,COUNT(*) AS n FROM arrivals WHERE status='valid' AND sponsor IS NOT NULL GROUP BY sponsor ORDER BY n DESC,sponsor LIMIT 10").fetchall()
        return [(self.alias(r['sponsor']), r['n']) for r in rows]

    def reserve_post(self, week):
        with self.db:
            return self.db.execute("INSERT OR IGNORE INTO posts VALUES (?, 'reserved')", (week,)).rowcount == 1

    def mark_post(self, week):
        with self.db:
            self.db.execute("UPDATE posts SET state='sent' WHERE week=?", (week,))

    def purge(self, now):
        with self.db:
            for row in self.db.execute('SELECT member,created FROM sponsors').fetchall():
                if expires(row['created']) <= now:
                    self.db.execute('DELETE FROM sponsors WHERE member=?', (row['member'],))
            for row in self.db.execute('SELECT member,joined FROM arrivals').fetchall():
                if expires(row['joined']) <= now:
                    self.db.execute('DELETE FROM arrivals WHERE member=?', (row['member'],))

    def erase(self, member):
        with self.db:
            self.db.execute('DELETE FROM sponsors WHERE member=?', (str(member),))
            self.db.execute('DELETE FROM arrivals WHERE member=?', (str(member),))
