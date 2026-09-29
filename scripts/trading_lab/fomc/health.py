"""FOMC source health per surface (source_health). Every check's result is committed in the same
transaction as the outcome that determines it, resolved by failure_classification_v1.precedence, so
it is durable before it can be visible (design missingness_policy.durable_requirement) and replay
re-derives it. Snapshots expose, only under FOMC_RESOLVED, the latest check of each surface within
P(T); a surface without a check within P(T) is SOURCE_NOT_CHECKED. RATE_LIMITED is an optional
non-failure diagnostic that affects no decision (rate_limited_state): it is not written in V1.
"""

from __future__ import annotations

from scripts.trading_lab.fomc import spec

SURFACES = ("discovery_feed", "primary_statement")
NO_PROVIDER = "NO_PROVIDER_HEALTH_STATE"

# failure_classification_v1.precedence, first applicable entry wins
PRECEDENCE = ("RAW_CORRUPTION", "LOCAL_PERSISTENCE_FAILURE", "SOURCE_UNAVAILABLE", "PARSER_FAILED", "CLOCK_UNVERIFIED",
              "NO_FAILURE")


def surface_of(kind: str) -> str:
    """The fetched surface of a logical fetch (logical_fetch_result)."""
    return "discovery_feed" if kind == "FEED_POLL" else "primary_statement"


def for_attempt_outcome(outcome: str) -> tuple[str, str] | None:
    """(result_state, reason) of a non-response attempt outcome; None when the mapping gives no
    source-health result (cancellation after grant)."""
    if outcome == "SOURCE_UNAVAILABLE":
        return "SOURCE_UNAVAILABLE", "SOURCE_UNAVAILABLE"
    if outcome == "PARSER_FAILED":  # decoded body over the frozen size bound
        return "PARSER_FAILED", "PARSER_FAILED"
    if outcome == "LOCAL_PERSISTENCE_FAILED":
        return NO_PROVIDER, "LOCAL_PERSISTENCE_FAILURE"
    if outcome == "INTERRUPTED":
        return NO_PROVIDER, "INTERRUPTED"
    if outcome == "CANCELLED_AFTER_INVOKE":
        return None
    raise ValueError(f"no source-health mapping for attempt outcome {outcome}")


def for_record(verified: bool, outcome: str, cycle_result: str | None) -> tuple[str | None, str]:
    """(result_state, reason) of a processed per-response record: the first applicable precedence
    entry. A successful check records no failure state (result_state None); a LIVE cycle concluded
    EVENTS_OBSERVED_ZERO records that state."""
    if outcome == "CORRUPTION_FAIL_CLOSED":
        return NO_PROVIDER, "RAW_CORRUPTION"
    if outcome == "PARSER_FAILED":
        return "PARSER_FAILED", "PARSER_FAILED"
    if outcome == "INTERNAL_PROCESSING_ERROR":
        return NO_PROVIDER, "INTERNAL_PROCESSING_ERROR"
    if not verified:  # CLOCK_UNVERIFIED or LATE_EVIDENCE
        return NO_PROVIDER, "CLOCK_UNVERIFIED"
    if cycle_result == "EVENTS_OBSERVED_ZERO":
        return "EVENTS_OBSERVED_ZERO", "NO_FAILURE"
    return None, "NO_FAILURE"


DIAGNOSTIC_REASON = "raw digest mismatch detected after the terminal outcome"
POISON_REASON = "two DEAD processing runs"


def surface_of_record(resp_body: dict) -> str:
    return "discovery_feed" if resp_body["surface"] == "feed" else "primary_statement"


def for_integrity_diagnostic() -> tuple[str, str]:
    """A digest mismatch found after the record's terminal outcome: the outcome is kept, the check
    of that surface is NO_PROVIDER_HEALTH_STATE (RAW_CORRUPTION), nothing is refetched."""
    return NO_PROVIDER, "RAW_CORRUPTION"


def row(surface: str, check_at: str | None, result: tuple, **provenance) -> tuple[str, str, dict]:
    state, reason = result
    return ("SOURCE_HEALTH", surface, {"provider_id": spec.PROVIDER_ID, "surface": surface, "check_at": check_at,
                                       "result_state": state, "reason": reason, **provenance})


def exposed(store, P: int) -> dict:
    """The latest check of each surface within P(T) (read under FOMC_RESOLVED only)."""
    out = {}
    for surface in SURFACES:
        rows = store.rows("SOURCE_HEALTH", key=surface, upto=P)
        if not rows:
            out[surface] = {"result_state": "SOURCE_NOT_CHECKED"}
            continue
        body = rows[-1].body
        out[surface] = {"result_state": body["result_state"], "reason": body["reason"], "check": rows[-1].seq,
                        "check_at": body["check_at"], "outcome": body["outcome"],
                        "record": body.get("record"), "attempt": body["attempt"]}
    return out
