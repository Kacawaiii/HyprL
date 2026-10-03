"""Server-attested causal availability (CAUSAL_AVAILABILITY_V3, extracted unchanged from the FOMC slice).

A transaction X becomes available at max(V.observed_at + bound, the observed_at of the observations X
itself concerns, the availability of every earlier transaction), where V is the first clock-verified,
non-late response whose TRANSPORT_INVOKED comes after X; until such a V exists X is unresolved, and so is
every later transaction. A read at T sees the longest prefix of resolved transactions available at or
before T, and is resolved only when the next transaction is resolved with availability after T.

The rule needs only RESPONSE rows carrying `attempt`, `verdict`, `late_evidence` and `observed_at`, and
the rows of `concerned_kinds` carrying `observed_at`; a source's own timestamps (a declared release time,
an EDGAR acceptanceDateTime) never enter it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from scripts.trading_lab.sources.httpclock import parse_iso


@dataclass(frozen=True)
class Avail:
    seq: int
    resolved: bool
    avail: datetime | None


def verified(resp) -> bool:
    """CLOCK_VERIFIED for every rule: LATE_EVIDENCE is treated as CLOCK_UNVERIFIED."""
    return resp.body["verdict"] == "CLOCK_VERIFIED" and not resp.body["late_evidence"]


def observed_at(resp) -> datetime | None:
    value = resp.body.get("observed_at")
    return parse_iso(value) if value else None


def availability(store, horizon: int, *, bound: timedelta, concerned_kinds: tuple[str, ...] = ()) -> list[Avail]:
    store = store.view(horizon)
    responses = store.rows("RESPONSE")
    refs = [r for r in responses if verified(r)]  # already in commit order
    concerned: dict[int, datetime] = {}
    for r in responses:
        if verified(r):
            concerned[r.seq] = max(concerned.get(r.seq, observed_at(r)), observed_at(r))
    for kind in concerned_kinds:
        for row in store.rows(kind):
            if row.body.get("observed_at"):
                t = parse_iso(row.body["observed_at"])
                concerned[row.seq] = max(concerned.get(row.seq, t), t)
    out, pointer, previous = [], 0, None
    for seq, _kind, _wall in store.txns():
        while pointer < len(refs) and refs[pointer].body["attempt"] <= seq:
            pointer += 1
        if pointer == len(refs) or (out and not out[-1].resolved):
            out.append(Avail(seq, False, None))
            continue
        value = observed_at(refs[pointer]) + bound
        for other in (concerned.get(seq), previous):
            if other is not None and other > value:
                value = other
        previous = value
        out.append(Avail(seq, True, value))
    return out


def prefix(table: list[Avail], T: datetime) -> tuple[bool, int]:
    """(resolved, P): P is the longest prefix with resolved avail <= T; the read is resolved only if the
    next transaction is resolved (its avail > T)."""
    end, beyond = 0, None
    for entry in table:
        if entry.resolved and entry.avail <= T:
            end = entry.seq
        else:
            beyond = entry
            break
    return beyond is not None and beyond.resolved, end
