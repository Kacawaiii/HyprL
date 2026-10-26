import json
from pathlib import Path

import jsonschema
import pytest

from scripts.alert_router.model import KINDS, Alert, AlertError, synthetic_alert

CASES = [
    ("sec_filing", {"form": "8-K", "cik": "0000320193", "accession": "0000320193-26-000001"}),
    ("sec_filing", {"form": "10-Q", "cik": "0000320193", "accession": "0000320193-26-000002"}),
    ("sec_filing", {"form": "10-K", "cik": "0000320193", "accession": "0000320193-26-000003"}),
    ("sec_filing", {"form": "S-1", "cik": "0000320193", "accession": "0000320193-26-000004"}),
    ("sec_filing", {"form": "4", "cik": "0000320193", "accession": "0000320193-26-000005"}),
    ("fed_release", {"release": "statement"}),
    ("cpi_release", {"period": "2026-10", "yoy_pct": "2.1", "mom_pct": "0.1"}),
    ("nfp_release", {"period": "2026-10", "jobs_change": "100000", "unemployment_pct": "4.0"}),
    ("earnings_date", {"scheduled_at": "2026-10-27T20:00:00Z", "period": "FY2026Q4"}),
    (
        "earnings_result",
        {"eps": "2.1", "revenue": "100000000", "currency": "USD", "period": "FY2026Q4"},
    ),
    ("unusual_move", {"price_change_pct": "-2.0", "volume_multiple": "3.5", "window_minutes": 15}),
    ("radar_headline", {"category": "regulation", "impact": "high"}),
    (
        "duel_hashes",
        {
            "session": "2026-10-26",
            "open_at": "2026-10-26T13:30:00Z",
            "a1_sha256": "a" * 64,
            "e1_sha256": "b" * 64,
        },
    ),
    (
        "duel_results",
        {
            "session": "2026-10-23",
            "close_at": "2026-10-23T20:00:00Z",
            "a1_sha256": "a" * 64,
            "e1_sha256": "b" * 64,
            "a1_return_pct": "0.1",
            "e1_return_pct": "-0.2",
        },
    ),
    (
        "weekly_luck",
        {
            "week_ending": "2026-10-23",
            "a1_percentile": "55",
            "e1_percentile": "45",
            "observations": 50,
        },
    ),
    ("system_test", {"synthetic": True}),
]


@pytest.mark.parametrize("kind,facts", CASES)
def test_every_alert_kind_satisfies_published_schema_and_fits_channels(kind, facts):
    event = synthetic_alert()
    event.update(kind=kind, facts=facts, symbols=["AAPL"])
    alert = Alert.parse(event)
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/alerts/payload-v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(alert.payload(), schema)
    assert len(alert.text()) <= 2000
    assert {k for k, _ in CASES} == KINDS


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", []),
        ("schema_version", True),
        ("symbols", [{}]),
        ("agents", ["Claude"]),
        ("occurred_at", "2026-10-26T12:00:00"),
        ("source", {}),
        ("event_id", "arbitrary"),
    ],
)
def test_malformed_envelopes_fail_closed(field, value):
    event = synthetic_alert()
    event[field] = value
    with pytest.raises(AlertError):
        Alert.parse(event)


@pytest.mark.parametrize(
    "field",
    ["recommendation", "position", "stop_loss", "target_price", "headline", "summary", "body"],
)
def test_free_text_trading_or_raw_content_fields_are_rejected(field):
    event = synthetic_alert()
    event["facts"][field] = "Achetez maintenant, augmentez la position"
    with pytest.raises(AlertError):
        Alert.parse(event)


def test_cik_keeps_official_padded_string_shape():
    event = synthetic_alert()
    event.update(
        kind="sec_filing",
        symbols=["AAPL"],
        facts={"form": "8-K", "cik": 320193, "accession": "0000320193-26-000001"},
    )
    with pytest.raises(AlertError):
        Alert.parse(event)
