"""Reviewed November rollout dates; all cutoffs use UTC, including the bank day."""
from datetime import timezone

from .config import TraderError
from .data import calendar_session

ROLLOUT_DAYS = ('2026-11-03', '2026-11-04')


def rollout_day(at, *, finishing=False):
    if at.utcoffset() is None:
        raise TraderError('NAIVE_TIME')
    at = at.astimezone(timezone.utc)
    day = at.date().isoformat()
    cutoff = 450 if finishing else 445
    if day not in ROLLOUT_DAYS or not calendar_session(day) or at.hour * 60 + at.minute >= cutoff:
        raise TraderError('RELEASE_OUTSIDE_ROLLOUT_WINDOW')
    return day
