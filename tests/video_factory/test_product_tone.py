"""Synthetic member-v2 evidence, preserving the deployed source shapes."""

import pytest

from scripts.video_factory import factory, market
from scripts.video_factory.content import evergreen
from tests.video_factory.test_factory import dump, observed


def brief_snapshot():
    s = {
        "url": "https://news.example.test/evenement",
        "sha256": "d" * 64,
        "published_at": "2025-07-07T06:00:00Z",
        "available_at": "2025-07-07T06:01:00Z",
    }
    return {
        "schema_version": 2,
        "provenance": "observed",
        "generated_at": "2025-07-07T20:46:00Z",
        "briefs": [
            {
                "date": "2025-07-07",
                "published_at": "2025-07-07T06:30:00Z",
                "facts": [
                    {
                        "kind": "fed",
                        "symbol": None,
                        "summary": "La Fed publie son compte rendu.",
                        "source": s,
                    }
                ],
            }
        ],
        "calendar": [
            {
                "date": "2025-07-08",
                "kind": "earnings",
                "symbol": "EXM",
                "precision": "date",
                "scheduled_at": None,
                "evidence": {
                    "kind": "observed_day_marker",
                    "sha256": "e" * 64,
                    "dataset_sha256": "f" * 64,
                    "received_at": "2025-07-07T06:00:00Z",
                },
                "source": {**s, "url": "https://calendar.example.test/exm"},
            }
        ],
    }


def test_market_uses_dated_brief_and_tomorrows_sourced_calendar(tmp_path):
    result = market.market_point(
        dump(tmp_path / "snapshot", brief_snapshot()),
        "2025-07-07",
        market.instant("2025-07-07T21:05:00Z"),
    )
    assert result["title"] == "Le point marché du jour"
    assert result["captions"][0] == "Dans le brief du jour : La Fed publie son compte rendu."
    assert result["captions"][1] == "Demain, 2025-07-08 : résultats de EXM, horaire non confirmé."
    assert result["source_labels"][0] == "news.example.test · 07/07 à 02 h 00 New York"
    assert len(result["sources"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        "old_brief",
        "synthetic",
        "future",
        "stale",
        "no_facts",
        "no_calendar",
        "digest_only",
        "repeated_ai",
    ],
)
def test_unavailable_daily_evidence_uses_no_invented_market_point(tmp_path, change):
    value = brief_snapshot()
    if change == "old_brief":
        value["briefs"][0]["date"] = "2025-07-04"
    if change == "synthetic":
        value["provenance"] = "synthetic"
    if change == "future":
        value["generated_at"] = "2025-07-08T20:46:00Z"
    if change == "stale":
        value["generated_at"] = "2025-07-05T20:46:00Z"
    if change == "no_facts":
        value["briefs"][0]["facts"] = []
    if change == "no_calendar":
        value["calendar"] = []
    if change == "digest_only":
        value["calendar"][0]["source"] = None
        value["calendar"][0]["evidence"] = {
            "kind": "observed_day_marker",
            "sha256": "e" * 64,
            "dataset_sha256": "f" * 64,
            "received_at": "2025-07-07T06:00:00Z",
        }
    if change == "repeated_ai":
        value["briefs"][0]["facts"][0]["summary"] = "IA et intelligence artificielle."
    assert (
        market.market_point(
            dump(tmp_path / "snapshot", value), "2025-07-07", market.instant("2025-07-07T21:05:00Z")
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "future_source",
        "noncausal_source",
        "unavailable_brief",
        "calendar_date",
        "duplicate_brief",
        "private_url",
    ],
)
def test_contradictory_market_evidence_fails_closed(tmp_path, change):
    value = brief_snapshot()
    if change == "future_source":
        value["briefs"][0]["facts"][0]["source"]["available_at"] = "2025-07-08T06:00:00Z"
    if change == "noncausal_source":
        value["briefs"][0]["facts"][0]["source"]["published_at"] = "2025-07-07T06:02:00Z"
    if change == "unavailable_brief":
        value["briefs"][0]["published_at"] = "2025-07-07T22:00:00Z"
    if change == "calendar_date":
        value["calendar"][0].update(precision="time", scheduled_at="2025-07-09T13:00:00Z")
    if change == "duplicate_brief":
        value["briefs"].append(value["briefs"][0])
    if change == "private_url":
        value["briefs"][0]["facts"][0]["source"]["url"] = "https://user:secret@example.test/private"
    with pytest.raises(ValueError):
        market.market_point(
            dump(tmp_path / "snapshot", value), "2025-07-07", market.instant("2025-07-07T21:05:00Z")
        )


def test_macro_calendar_preserves_its_known_time(tmp_path):
    value = brief_snapshot()
    value["calendar"][0].update(
        kind="cpi", symbol=None, precision="time", scheduled_at="2025-07-08T12:30:00Z"
    )
    result = market.market_point(
        dump(tmp_path / "snapshot", value), "2025-07-07", market.instant("2025-07-07T21:05:00Z")
    )
    assert (
        result["captions"][1]
        == "Demain, 2025-07-08 : publication de l'inflation américaine, 08 h 30 à New York."
    )


@pytest.mark.parametrize(
    "change", [None, "dataset", "event", "source_date", "future_capture", "symbol"]
)
def test_calendar_marker_recovers_only_its_bound_original_source(tmp_path, change):
    value = brief_snapshot()
    event = {"kind": "EARNINGS", "symbol": "EXM", "at": "2025-07-08T23:59:59Z"}
    calendar = {
        "schema": "agent-v2-quality-calendar-v2",
        "received_at": "2025-07-07T06:20:00Z",
        "events": [event],
        "source_hashes": ["b" * 64],
        "ready": True,
        "coverage_start": "2025-07-07",
        "coverage_end": "2025-07-09",
        "covered_symbols": ["EXM"],
        "covered_macro": [],
        "coverage_notes": [],
        "missing_earnings_dates": [],
        "not_applicable_symbols": [],
    }
    manifest = [
        {
            "method": "GET",
            "status": 200,
            "url": "https://api.nasdaq.com/api/calendar/earnings?date=2025-07-08",
            "body_sha256": "b" * 64,
            "authorization_file_sha256": "c" * 64,
            "started_at": "2025-07-07T06:10:00Z",
            "received_at": "2025-07-07T06:11:00Z",
        }
    ]
    entry = value["calendar"][0]
    entry["source"] = None
    entry["evidence"] = {
        "kind": "observed_day_marker",
        "dataset_sha256": market.digest(calendar),
        "sha256": market.digest(event),
        "received_at": calendar["received_at"],
    }
    if change == "dataset":
        entry["evidence"]["dataset_sha256"] = "f" * 64
    if change == "event":
        entry["evidence"]["sha256"] = "f" * 64
    if change == "symbol":
        entry["symbol"] = "OTHER"
    if change == "source_date":
        manifest[0]["url"] = "https://api.nasdaq.com/api/calendar/earnings?date=2025-07-09"
    if change == "future_capture":
        calendar["received_at"] = "2025-07-08T06:20:00Z"
        entry["evidence"]["dataset_sha256"] = market.digest(calendar)
    result = market.market_point(
        dump(tmp_path / "snapshot", value),
        "2025-07-07",
        market.instant("2025-07-07T21:05:00Z"),
        calendar_file=dump(tmp_path / "calendar", calendar),
        calendar_manifest=dump(tmp_path / "manifest", manifest),
    )
    if change:
        assert result is None
    else:
        assert result["source_evidence"][1]["published_at"] is None
        assert result["source_evidence"][1]["available_at"] == "2025-07-07T06:11:00Z"
        assert (
            result["sources"][0] == "https://api.nasdaq.com/api/calendar/earnings?date=2025-07-08"
        )
        assert "horaire non confirmé" in result["captions"][1]


def test_weekly_recap_pools_only_common_matured_published_scores_from_this_week(tmp_path):
    value = observed()
    second = observed("2025-07-08")
    old = observed("2025-07-03")
    value["generated_at"] = "2025-07-11T20:46:00Z"
    for other in (second, old):
        value["reveals"].extend(other["reveals"])
        value["forecasts"].extend(other["forecasts"])
    for row in value["forecasts"]:
        row["score_sha256"] = "e" * 64
    # A published single-agent row is not a paired observation.
    value["forecasts"].append({**value["forecasts"][0], "symbol": "EXM", "probability": 0.1})
    result = factory.weekly_recap(
        dump(tmp_path / "snapshot", value), "2025-07-11", market.instant("2025-07-11T21:05:00Z")
    )
    assert result["kind"] == "weekly"
    assert result["chart"] == [
        {"agent": "HypR-A1", "hits": 2, "n": 2},
        {"agent": "HypR-E1", "hits": 2, "n": 2},
    ]
    assert result["score_session"] == "2025-07-07 au 2025-07-11"
    assert len(result["source_evidence"]) == 4
    with pytest.raises(ValueError, match="Friday"):
        factory.weekly_recap(
            tmp_path / "snapshot", "2025-07-10", market.instant("2025-07-11T21:05:00Z")
        )


@pytest.mark.parametrize("change", ["empty", "missing_score_identity"])
def test_weekly_needs_real_scores(tmp_path, change):
    value = observed()
    value["generated_at"] = "2025-07-11T20:46:00Z"
    if change == "empty":
        value["forecasts"] = []
    p = dump(tmp_path / "snapshot", value)
    if change == "empty":
        assert factory.weekly_recap(p, "2025-07-11", market.instant("2025-07-11T21:05:00Z")) is None
    else:
        with pytest.raises(ValueError, match="published score"):
            factory.weekly_recap(p, "2025-07-11", market.instant("2025-07-11T21:05:00Z"))


@pytest.mark.parametrize(
    "day", ["2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11"]
)
def test_daily_prefers_market_and_only_friday_can_select_weekly(tmp_path, monkeypatch, day):
    kinds = []
    monkeypatch.setattr(factory, "trading_day", lambda _: True)
    monkeypatch.setattr(factory, "market_point", lambda *args, **kwargs: {"kind": "market"})

    def weekly(*args):
        assert day == "2025-07-11"
        return {"kind": "weekly"}

    monkeypatch.setattr(factory, "weekly_recap", weekly)

    def produce(config, identifier, content, publish=True):
        kinds.append(content["kind"])
        return {"status": "rendered"}

    monkeypatch.setattr(factory, "produce", produce)
    factory.daily(
        {"state_dir": str(tmp_path), "snapshot": "unused", "verdict_content": "must-not-be-read"},
        market.instant(day + "T21:05:00Z"),
        publish=False,
    )
    assert kinds == ["weekly" if day == "2025-07-11" else "market"]


def test_v2_rotation_has_product_copy_and_no_ai_hashtag():
    from scripts.video_factory.content import HASHTAGS

    assert [evergreen(i)["title"] for i in range(5)] == [
        "HyprL en 30 secondes",
        "Votre matinée de marché en deux minutes",
        "Ne ratez plus un événement",
        "La transparence comme règle",
        "L'accès anticipé est ouvert",
    ]
    assert evergreen(0) == evergreen(5)
    assert all(evergreen(i)["edition"] == "version présentation produit" for i in range(5))
    assert "#IA" not in HASHTAGS


def test_stale_v1_bundle_cannot_be_published_as_v2(tmp_path, monkeypatch):
    bundle = tmp_path / "evergreen-v2-01"
    bundle.mkdir()
    dump(bundle / "metadata.json", {"schema": "hyprl-video-v1", "title": "Old v1 copy"})
    monkeypatch.setattr(factory, "guard_clock", lambda _: None)

    def no_upload(*args):
        pytest.fail("stale bundle reached upload")

    monkeypatch.setattr(factory, "upload", no_upload)
    with pytest.raises(ValueError, match="differs from selected"):
        factory.produce(
            {"output_dir": str(tmp_path)}, "evergreen-v2-01", evergreen(0), publish=True
        )
