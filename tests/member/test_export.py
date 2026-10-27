"""Synthetic member handoffs, independent known-value metric checks, no network."""

import copy
import hashlib
import json
from datetime import datetime, timezone

import pytest
from jsonschema import ValidationError

from scripts.member_export.export import build, canonical, publish

NOW = datetime(2026, 1, 5, tzinfo=timezone.utc)


def handoff():
    source = {
        "url": "https://example.com/evidence",
        "sha256": "a" * 64,
        "published_at": "2026-01-01T20:00:00Z",
        "available_at": "2026-01-01T20:00:00Z",
    }
    forecasts = []
    for day, p, y in [(1, "0.8", 1), (2, "0.4", 0), (3, "0.6", 0)]:
        c = {
            "agent": "HypR-A1",
            "variant": "frozen-v1",
            "session": f"2026-01-{day:02}",
            "symbol": "DEMO",
            "event": "close_above_open",
            "probability": p,
            "base_rate": "0.5",
            "committed_at": f"2026-01-{day:02}T12:00:00Z",
            "reveal_at": f"2026-01-{day:02}T20:00:00Z",
            "nonce": str(day) * 64,
        }
        forecasts.append(
            {
                "commitment": c,
                "sha256": hashlib.sha256(canonical(c)).hexdigest(),
                "receipt_url": "https://example.com/receipt",
                "outcome": y,
                "scored_at": c["reveal_at"],
                "source": source,
            }
        )
    return {
        "schema_version": 1,
        "provenance": "synthetic",
        "generated_at": "2026-01-04T23:00:00Z",
        "briefs": [],
        "radar": [],
        "calendar": [],
        "forecasts": forecasts,
        "failures": [],
        "revenue": [],
        "claims": [],
    }


def run(value):
    return build(value, now=NOW, allow_synthetic=True)


def test_known_brier_hit_and_exact_binomial_tail():
    result = run(handoff())
    points = result["series"][0]["points"]
    # Errors: .04 + .16 + .36 = .56; reference: .25 * 3 = .75.
    assert points[-1]["brier_skill"] == pytest.approx(1 - 0.56 / 0.75)
    assert [p["rolling_hit_rate"] for p in points] == [1, 1, 2 / 3]
    # P(Bin(3,.5) >= 2) = 4/8. First two hits have tail 1/4.
    assert [p["luck_tail"] for p in points] == [0.5, 0.25, 0.5]


def test_heterogeneous_baselines_and_zero_denominator():
    value = handoff()
    value["forecasts"] = value["forecasts"][:2]
    for row, b in zip(value["forecasts"], ["0.2", "0.3"]):
        row["commitment"]["base_rate"] = b
        row["sha256"] = hashlib.sha256(canonical(row["commitment"])).hexdigest()
    # Both hits, q = .2 and 1-.3=.7; exact upper tail = .14.
    assert run(value)["series"][0]["points"][-1]["luck_tail"] == pytest.approx(0.14)
    value["forecasts"] = value["forecasts"][:1]
    row = value["forecasts"][0]
    row["commitment"]["base_rate"] = "1.0"
    row["sha256"] = hashlib.sha256(canonical(row["commitment"])).hexdigest()
    assert run(value)["series"][0]["points"][0]["brier_skill"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(model_id="private"),
        lambda v: v["forecasts"][0].update(account_number="hidden"),
        lambda v: v["forecasts"][0]["commitment"].update(probability="NaN"),
        lambda v: v["forecasts"][0].update(sha256="f" * 64),
        lambda v: v["forecasts"].append(copy.deepcopy(v["forecasts"][0])),
        lambda v: v["forecasts"][0]["commitment"].update(reveal_at="2099-01-01T00:00:00Z"),
        lambda v: v.update(generated_at="2099-01-01T00:00:00Z"),
        lambda v: v["forecasts"][0]["source"].update(available_at="2099-01-01T00:00:00Z"),
        lambda v: v["forecasts"][0]["source"].update(available_at="2025-01-01T00:00:00Z"),
    ],
)
def test_bad_private_duplicate_future_and_noncausal_handoffs_refused(mutate):
    value = handoff()
    mutate(value)
    with pytest.raises((ValueError, ValidationError)):
        run(value)


def test_synthetic_never_accepted_by_production_build():
    with pytest.raises(ValueError, match="synthetic"):
        build(handoff(), now=NOW)


def test_private_and_advice_text_refused():
    for text in ["Achetez DEMO", "person@example.com", "/home/synthetic/private", "Bearer token"]:
        value = handoff()
        value["claims"] = [
            {
                "claim": text,
                "verdict": "verified",
                "verified_at": "2026-01-01T20:00:00Z",
                "source": value["forecasts"][0]["source"],
            }
        ]
        with pytest.raises(ValueError, match="vocabulary"):
            run(value)


def revenue(value):
    source = {
        **value["forecasts"][0]["source"],
        "url": "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/report.htm",
    }
    return {
        "symbol": "DEMO",
        "cik": "0000000001",
        "concept": "Revenues",
        "periods": [
            {
                "start": "2025-01-01",
                "end": "2025-03-31",
                "revenue": "100",
                "currency": "USD",
                "source": source,
            },
            {
                "start": "2025-04-01",
                "end": "2025-06-30",
                "revenue": "110",
                "currency": "USD",
                "source": source,
            },
            {
                "start": "2025-07-01",
                "end": "2025-09-30",
                "revenue": "132",
                "currency": "USD",
                "source": source,
            },
        ],
    }


def test_factual_acceleration_and_zero_prior_revenue_unavailable():
    value = handoff()
    value["revenue"] = [revenue(value)]
    row = run(value)["screener"][0]
    assert (row["growth_previous"], row["growth_latest"], row["acceleration_pp"]) == (10, 20, 10)
    value["revenue"][0]["periods"][0]["revenue"] = "0"
    assert run(value)["screener"] == []


@pytest.mark.parametrize("field,new", [("currency", "EUR"), ("start", "2024-01-01")])
def test_revenue_comparability_required(field, new):
    value = handoff()
    value["revenue"] = [revenue(value)]
    value["revenue"][0]["periods"][1][field] = new
    with pytest.raises(ValueError):
        run(value)


def test_atomic_read_only_publication_and_input_untouched(tmp_path):
    value = run(handoff())
    original = copy.deepcopy(value)
    digest = publish(value, tmp_path)
    file = tmp_path / "member-v1.json"
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    assert file.stat().st_mode & 0o777 == 0o440
    assert json.loads(file.read_text()) == original
    assert publish(value, tmp_path) == digest  # atomic replacement, not write-through
    assert not list(tmp_path.glob(".member-*"))


def test_twenty_observation_hit_window_keeps_negative_skill():
    value = handoff()
    template = value["forecasts"][0]
    value["forecasts"] = []
    for i in range(21):
        row = copy.deepcopy(template)
        row["commitment"]["symbol"] = "D" + str(i)
        row["commitment"]["probability"] = "0.8"
        row["outcome"] = int(i == 0)
        row["sha256"] = hashlib.sha256(canonical(row["commitment"])).hexdigest()
        # Distinct real ordering, avoiding hash-dependent ties for this test.
        row["scored_at"] = f"2026-01-02T20:00:{i:02}Z"
        value["forecasts"].append(row)
    last = run(value)["series"][0]["points"][-1]
    assert last["rolling_hit_rate"] == 0
    assert last["brier_skill"] < 0


def test_reveal_guard_even_when_digest_matches_future_commitment():
    value = handoff()
    row = value["forecasts"][0]
    row["commitment"]["reveal_at"] = "2099-01-01T00:00:00Z"
    row["sha256"] = hashlib.sha256(canonical(row["commitment"])).hexdigest()
    with pytest.raises(ValueError, match="unrevealed"):
        run(value)


def test_python_publisher_and_node_reader_agree_on_canonical_commitments(tmp_path):
    import os
    import subprocess

    publish(run(handoff()), tmp_path)
    script = """
    const {LegacySnapshots:Snapshots}=require('./apps/hyprl-site/server/member/snapshots');
    const dir=process.env.TEST_SNAPSHOT_DIR;
    try { new Snapshots(dir).read(); process.exit(2); } catch(e) {}
    const data=new Snapshots(dir,{allowSynthetic:true}).read();
    if(data.forecasts.length!==3) process.exit(3);
    console.log('PASS');
    """
    result = subprocess.run(
        ["node", "-e", script],
        env={**os.environ, "TEST_SNAPSHOT_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"


def test_cli_refusal_does_not_log_rejected_private_body(tmp_path):
    import subprocess
    import sys

    value = handoff()
    value["internal_model_id"] = "synthetic-private-body-must-not-be-logged"
    file = tmp_path / "handoff.json"
    file.write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.member_export.export",
            "--handoff",
            str(file),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert result.stderr.strip() == "MEMBER_EXPORT_REFUSED"
    assert result.stdout == ""
    assert not (tmp_path / "member-v1.json").exists()
