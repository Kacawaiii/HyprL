"""The confirmatory holdout guard, tested adversarially.

A holdout that leaks once is spent forever, so these tests try to get protected
data through rather than merely confirming the happy path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import json

import pytest


HOUR = timedelta(hours=1)


@pytest.fixture
def holdout():
    return importlib.import_module("scripts.trading_lab.protected_holdout")


@pytest.fixture
def window(holdout):
    return holdout.PROTECTED_WINDOW_V1


def test_the_window_is_read_from_the_committed_phase_4d_contract(holdout, window):
    """Restating the dates here would let the two definitions drift apart."""
    v2 = importlib.import_module("scripts.trading_lab.real_benchmark_v2")
    registered = v2.CONFIRMATORY_HOLDOUT_V2
    assert window.holdout_id == registered["holdout_id"]
    assert list(window.products) == registered["products"] == ["BTC-USD", "ETH-USD"]
    assert window.start == registered["range_start"] == "2026-09-01T00:00:00Z"
    assert window.end == registered["range_end"] == "2026-11-30T23:00:00Z"
    assert window.single_use is True
    assert window.observed is False
    assert len(window.holdout_hash) == 64


@pytest.mark.parametrize("opening,protected", [
    ("2026-08-31T22:00:00+00:00", False),
    ("2026-08-31T23:00:00+00:00", False),   # the last hour before the boundary
    ("2026-09-01T00:00:00+00:00", True),    # the boundary itself is protected
    ("2026-10-15T12:00:00+00:00", True),
    ("2026-11-30T23:00:00+00:00", True),    # the last protected opening
    ("2026-12-01T00:00:00+00:00", False),   # the first hour after
])
def test_the_boundary_is_exact_to_the_hour(holdout, opening, protected):
    for product in ("BTC-USD", "ETH-USD"):
        assert holdout.PROTECTED_WINDOW_V1.covers(product, opening) is protected
        if protected:
            with pytest.raises(holdout.ProtectedHoldoutError):
                holdout.require_unprotected_bar(product, opening)
        else:
            holdout.require_unprotected_bar(product, opening)


def test_an_unprotected_product_is_never_blocked(holdout):
    holdout.require_unprotected_bar("SOL-USD", "2026-10-01T00:00:00+00:00")
    state = holdout.embargo_state("SOL-USD", now="2026-10-01T00:00:00+00:00")
    assert state["protected_product"] is False and state["embargoed"] is False


def test_a_request_overlapping_the_window_is_refused_before_it_is_sent(holdout):
    """The first refusal must happen before any byte is requested."""
    with pytest.raises(holdout.ProtectedHoldoutError, match="overlaps"):
        holdout.require_unprotected_request(
            "BTC-USD", start="2026-08-31T20:00:00+00:00",
            end="2026-09-01T04:00:00+00:00")
    with pytest.raises(holdout.ProtectedHoldoutError, match="overlaps"):
        holdout.require_unprotected_request(
            "BTC-USD", start="2026-09-01T00:00:00+00:00",
            end="2026-09-01T00:00:00+00:00")
    with pytest.raises(holdout.ProtectedHoldoutError, match="overlaps"):
        holdout.require_unprotected_request(
            "ETH-USD", start="2026-01-01T00:00:00+00:00",
            end="2027-01-01T00:00:00+00:00")     # a range that swallows the window
    # a request that stops one hour short is fine
    holdout.require_unprotected_request(
        "BTC-USD", start="2026-08-30T00:00:00+00:00",
        end="2026-08-31T23:00:00+00:00")


def test_a_timezone_offset_cannot_smuggle_a_protected_bar_through(holdout):
    """02:00+02:00 IS 00:00Z on the first of September."""
    assert holdout.PROTECTED_WINDOW_V1.covers("BTC-USD", "2026-09-01T02:00:00+02:00")
    with pytest.raises(holdout.ProtectedHoldoutError):
        holdout.require_unprotected_bar("BTC-USD", "2026-09-01T02:00:00+02:00")
    # and the hour before, expressed in another zone, is still allowed
    holdout.require_unprotected_bar("BTC-USD", "2026-09-01T01:00:00+02:00")


def test_a_naive_timestamp_is_refused_rather_than_assumed_to_be_utc(holdout):
    with pytest.raises(holdout.ProtectedHoldoutError, match="timezone"):
        holdout.PROTECTED_WINDOW_V1.covers("BTC-USD", "2026-09-01T00:00:00")
    with pytest.raises(holdout.ProtectedHoldoutError, match="ISO-8601"):
        holdout.PROTECTED_WINDOW_V1.covers("BTC-USD", "not-a-date")


def test_the_embargo_switches_on_by_itself_at_the_boundary(holdout):
    """No human has to remember to stop paper trading on the 31st."""
    before = holdout.embargo_state("BTC-USD", now="2026-08-31T23:59:59+00:00")
    at = holdout.embargo_state("BTC-USD", now="2026-09-01T00:00:00+00:00")
    during = holdout.embargo_state("BTC-USD", now="2026-10-20T09:00:00+00:00")
    after = holdout.embargo_state("BTC-USD", now="2026-12-01T00:00:00+00:00")
    assert before["embargoed"] is False and "allowed until" in before["reason"]
    assert at["embargoed"] is True and "confirmatory research holdout" in at["reason"]
    assert during["embargoed"] is True
    assert after["embargoed"] is False and after["window_elapsed"] is True
    holdout.require_tradeable_now("BTC-USD", now="2026-08-31T23:59:59+00:00")
    with pytest.raises(holdout.ProtectedHoldoutError):
        holdout.require_tradeable_now("BTC-USD", now="2026-09-01T00:00:00+00:00")


def test_a_clock_that_jumps_over_the_boundary_still_lands_embargoed(holdout):
    for now in ("2026-09-01T00:00:00+00:00", "2026-09-14T00:00:00+00:00",
                "2026-11-30T23:59:59+00:00"):
        with pytest.raises(holdout.ProtectedHoldoutError):
            holdout.require_tradeable_now("ETH-USD", now=now)


def test_the_embargo_lasts_until_the_final_protected_bar_has_closed(holdout, window):
    """`end` is the last protected OPENING, so the window outlives it by one bar.

    Lifting at 23:00 on the last day would hand the session that final candle
    an hour later, which is exactly the leak the window exists to prevent.
    """
    assert window.closes_at.isoformat() == "2026-12-01T00:00:00+00:00"
    for still_closed in ("2026-11-30T23:00:00+00:00", "2026-11-30T23:59:59+00:00"):
        assert holdout.embargo_state("BTC-USD", now=still_closed)["embargoed"] is True
        with pytest.raises(holdout.ProtectedHoldoutError):
            holdout.require_tradeable_now("BTC-USD", now=still_closed)
    reopened = holdout.embargo_state("BTC-USD", now="2026-12-01T00:00:00+00:00")
    assert reopened["embargoed"] is False and reopened["window_elapsed"] is True


def test_the_embargo_state_never_carries_a_protected_price(holdout):
    """The status a UI renders must describe the boundary, not the data."""
    payload = holdout.embargo_state("BTC-USD", now="2026-10-01T00:00:00+00:00")
    text = json.dumps(payload)
    for forbidden in ("open", "high", "low", "close", "volume", "price"):
        assert f'"{forbidden}"' not in text
    assert set(payload) == {
        "product", "protected_product", "embargoed", "window_active",
        "window_elapsed", "start", "end", "closes_at", "holdout_id",
        "holdout_hash", "observed", "reason"}
