"""Synthetic Git disclosures → observed adapter → v2 snapshot/Node boundary."""

import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from test_producer import sources

from scripts.member_export.export import canonical
from scripts.member_export.export_v2 import build, publish
from scripts.member_export.producer import research_digest
from scripts.member_export.producer_v2 import produce
from scripts.member_export.spec import FROZEN_RATE_SHA256

NOW = datetime(2026, 10, 27, 3, tzinfo=timezone.utc)


def observed(tmp_path):
    config = sources(tmp_path)
    base, ledger = Path(config["forward_dir"]), Path(config["ledger"])
    day, agent = "2026-10-22", "HypR-A1"
    snapshot_hash = "b" * 64
    original = {
        "schema": "live-forward-decision-v2",
        "provenance": "LIVE_FORWARD",
        "family": "opus",
        "model": "synthetic",
        "configuration_hash": "c" * 64,
        "snapshot_hash": snapshot_hash,
        "decision_time": day + "T13:05:00Z",
        "call_order": ["opus", "gpt"],
        "stored_at": day + "T13:06:00Z",
        "call": {
            "called_at": day + "T13:05:00Z",
            "family": "opus",
            "identity_evidence": "synthetic",
            "model": "synthetic",
            "prompt_sha256": "a" * 64,
            "reported": [],
            "transcript": "synthetic-call.json",
        },
        "output": {
            "analyst": {
                "regime": ["synthetic"],
                "views": [
                    {
                        "asset": "DEMO",
                        "horizon": "1d",
                        "p_outperform": 0.8,
                        "view": "ABSTAIN",
                        "confidence_reason": "Synthetic",
                        "catalysts": [],
                        "priced_in_assessment": "Synthetic",
                        "second_order": "Synthetic",
                        "counter_thesis": "Synthetic",
                        "falsifier": "Synthetic",
                        "event_risk": [],
                    },
                    {
                        "asset": "TEST",
                        "horizon": "1d",
                        "p_outperform": 0.2,
                        "view": "ABSTAIN",
                        "confidence_reason": "Synthetic",
                        "catalysts": [],
                        "priced_in_assessment": "Synthetic",
                        "second_order": "Synthetic",
                        "counter_thesis": "Synthetic",
                        "falsifier": "Synthetic",
                        "event_risk": [],
                    },
                ],
            },
            "news": {"forecasts": [], "views": [], "abstain_views_filled": 0},
        },
    }
    original["output"]["analyst"]["views"] += [
        {**v, "horizon": "5d"} for v in original["output"]["analyst"]["views"]
    ]
    digest = research_digest(original)
    failure = {
        "at": day + "T13:06:00Z",
        "error": "MODEL_USAGE_LIMIT",
        "snapshot_hash": snapshot_hash,
    }
    directory = base / day
    directory.mkdir()
    (directory / "opus.json").write_text(json.dumps({**original, "decision_hash": digest}))
    (directory / "gpt-failure.json").write_text(json.dumps(failure))
    ledger.mkdir()

    def git(*args, at=day + "T13:20:00Z"):
        return subprocess.check_output(
            ["git", "-C", str(ledger), *args],
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "Synthetic",
                "GIT_AUTHOR_EMAIL": "synthetic@example.com",
                "GIT_COMMITTER_NAME": "Synthetic",
                "GIT_COMMITTER_EMAIL": "synthetic@example.com",
                "GIT_AUTHOR_DATE": at,
                "GIT_COMMITTER_DATE": at,
            },
        )

    git("init", "-q", "-b", "ledger/forward-duel")
    line = {
        "kind": "forward",
        "key": day,
        "hashes": {"snapshot": snapshot_hash, "opus": digest, "gpt": research_digest(failure)},
        "protocol_spec_hash": "d" * 64,
        "committed_at": day + "T13:19:00Z",
    }
    (ledger / "forward-duel-ledger.jsonl").write_bytes(canonical(line) + b"\n")
    git("add", ".")
    git("commit", "-qm", "Synthetic prior commitment")
    prior = git("rev-parse", "HEAD").decode().strip()
    receipts = base.parent / "commitments/forward"
    receipts.mkdir(parents=True)
    (receipts / (day + ".json")).write_text(
        json.dumps(
            {
                "state": "PUBLISHED",
                "route": "discord",
                "messages": 1,
                "ledger_commit": prior,
                "line": line,
                "line_sha256": research_digest(line),
                "published_at": day + "T13:21:00Z",
            }
        )
    )
    filename = "reveals/" + day + "/" + agent + ".json"
    target = ledger / filename
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical(original))
    readme = "reveals/" + day + "/README.md"
    (ledger / readme).write_text(
        "| HypR-A1 | REVEALED | "
        + digest
        + " |\n| HypR-E1 | NO_DECISION | "
        + research_digest(failure)
        + " |\n"
    )
    git("add", ".")
    git("commit", "-qm", "Synthetic original reveal", at=day + "T21:00:00Z")
    commit = git("rev-parse", "HEAD").decode().strip()
    git("update-ref", "refs/remotes/origin/ledger/forward-duel", commit)
    reveal_dir = Path(config["reveal_dir"])
    reveal_dir.mkdir()
    (reveal_dir / (day + ".json")).write_text(
        json.dumps(
            {
                "state": "DELIVERED",
                "reveal_commit": commit,
                "delivered_at": day + "T21:00:05Z",
                "files": {
                    filename: digest,
                    readme: hashlib.sha256((ledger / readme).read_bytes()).hexdigest(),
                },
            }
        )
    )
    score = json.loads((base / "scores.json").read_text())
    score.update(
        base_rate_source="FROZEN_DEV_BASE_RATE:" + FROZEN_RATE_SHA256,
        primary_endpoint={"base_rate": 591 / 1216},
        trades=[],
        failures={
            "gpt": {"count": 1, "failed_sessions": [{"session": day, "error": failure["error"]}]}
        },
    )
    for symbol, probability, outcome in (("DEMO", 0.8, 1), ("TEST", 0.2, 0)):
        score["rows"].append(
            {
                "provenance": "LIVE_FORWARD",
                "mode": "REGULAR_0905",
                "decision_time": original["decision_time"],
                "entry_at": day + "T13:45:00Z",
                "session": day,
                "family": "opus",
                "group": "ab",
                "source_spec_hash": "d" * 64,
                "source_code_hash": "e" * 64,
                "panel": "a_v1_as_live:1d",
                "asset": symbol,
                "horizon": "1d",
                "snapshot_hash": snapshot_hash,
                "decision_hash": digest,
                "outcome": outcome,
                "outperform": bool(outcome),
                "p": probability,
                "brier": 0.04,
                "log_loss": 0.2231435513142097,
                "rel": 0.01 if outcome else -0.01,
            }
        )
    save_score(config, score)
    return config


def save_score(config, score):
    base = Path(config["forward_dir"])
    (base / "scores.json").write_text(json.dumps(score))
    (base / "score-runs" / (research_digest(score) + ".json")).write_text(json.dumps(score))


def test_public_original_records_feed_real_horizon_scores_and_date_only_markers(tmp_path):
    value, status = produce(observed(tmp_path), now=NOW)
    snapshot = build(value, now=NOW)
    row = next(r for r in snapshot["series"] if r["n"])
    assert (row["n"], row["hits"], row["brier"], row["rolling_hit_rate"]) == (
        2,
        2,
        pytest.approx(0.04),
        1,
    )
    assert row["brier_base_rate"] == pytest.approx(0.250195447757964)
    assert row["brier_skill"] == pytest.approx(0.840124988849935)
    assert all(not r["points"] for r in snapshot["series"] if not r["n"])
    failed = [
        r for r in snapshot["series"] if r["agent"] == "HypR-E1" and r["mode"] == "REGULAR_0905"
    ]
    assert all(r["model_failures"] == 1 and r["n"] == 0 and r["brier"] is None for r in failed)
    assert value["calendar"][0]["scheduled_at"] is None
    assert value["calendar"][0]["date"] == "2026-10-28"
    assert value["revenue"] == [] and snapshot["screener"] == []
    assert status["counts"]["reveals"] == 2
    assert all("commitment" not in r for r in value["forecasts"])


@pytest.mark.parametrize(
    "change", ["wrong_probability", "wrong_outcome", "duplicate", "synthetic", "unrevealed"]
)
def test_adapter_refuses_modified_scores_or_withholds_unpublished_decisions(tmp_path, change):
    config = observed(tmp_path)
    score = json.loads((Path(config["forward_dir"]) / "scores.json").read_text())
    if change == "wrong_probability":
        score["rows"][0]["p"] = 0.7
    if change == "wrong_outcome":
        score["rows"][0]["outcome"] = 0
    if change == "duplicate":
        score["rows"].append(score["rows"][0])
    if change == "synthetic":
        score["rows"][0]["provenance"] = "synthetic"
    if change == "unrevealed":
        (Path(config["reveal_dir"]) / "2026-10-22.json").unlink()
    save_score(config, score)
    if change == "unrevealed":
        value, status = produce(config, now=NOW)
        assert value["forecasts"] == [] and status["excluded"]["withheld_forecast_rows"] == 2
    else:
        with pytest.raises(ValueError):
            produce(config, now=NOW)


def test_negative_skill_empty_horizons_and_missing_baseline_are_honest(tmp_path):
    value, _ = produce(observed(tmp_path), now=NOW)
    for row in value["forecasts"]:
        row["probability"] = 1 - row["probability"]
    result = build(value, now=NOW)
    row = next(r for r in result["series"] if r["n"])
    assert row["brier"] == pytest.approx(0.64) and row["brier_skill"] < -1.5 and row["hits"] == 0
    value["baseline"] = None
    assert all(r["brier_skill"] is None for r in build(value, now=NOW)["series"])
    value["forecasts"] = []
    value["failures"] = []
    value["reveals"] = []
    assert all(r["n"] == 0 and r["points"] == [] for r in build(value, now=NOW)["series"])


@pytest.mark.parametrize("change", ["baseline", "hour", "proof", "future", "extra", "v1"])
def test_contract_rejects_fabricated_evidence_and_implicit_migration(tmp_path, change):
    value, _ = produce(observed(tmp_path), now=NOW)
    if change == "baseline":
        value["baseline"]["probability"] = 0.5
    if change == "hour":
        value["calendar"][0]["scheduled_at"] = "2026-10-28T00:00:00Z"
    if change == "proof":
        value["reveals"][0]["url"] = "https://example.com/substitute"
    if change == "future":
        value["reveals"][0]["available_at"] = "2099-01-01T00:00:00Z"
    if change == "extra":
        value["forecasts"][0]["unit_commitment"] = "f" * 64
    if change == "v1":
        value["schema_version"] = 1
    with pytest.raises(Exception):
        build(value, now=NOW)


def test_python_v2_publication_agrees_with_production_node_reader(tmp_path):
    value, _ = produce(observed(tmp_path), now=NOW)
    snapshot = build(value, now=NOW)
    output = tmp_path / "snapshots"
    output.mkdir(mode=0o700)
    digest = publish(snapshot, output)
    assert hashlib.sha256((output / "member-v2.json").read_bytes()).hexdigest() == digest
    assert not (output / "member-v1.json").exists()
    script = "const {Snapshots}=require('./apps/hyprl-site/server/member/snapshots-v2'); const v=new Snapshots(process.argv[1]).read(); console.log(v.forecasts.length)"
    assert subprocess.check_output(["node", "-e", script, str(output)], text=True).strip() == "2"
    bad = copy.deepcopy(snapshot)
    bad["series"][0]["points"].append(copy.deepcopy(bad["series"][0]["points"][0]))
    publish(bad, output)
    assert subprocess.run(["node", "-e", script, str(output)], capture_output=True).returncode != 0


def test_member_timer_keeps_v1_fresh_and_only_updates_member_jobs():
    from scripts.member_export.jobs_v2 import units

    rendered = units(
        checkout="/synthetic/checkout",
        python="/synthetic/python",
        legacy_config="/synthetic/legacy.json",
        config="/synthetic/v2.json",
        commit="a" * 40,
    )
    assert set(rendered) == {"hyprl-member-producer.service", "hyprl-member-producer.timer"}
    service = rendered["hyprl-member-producer.service"]
    assert service.count("ExecStart=") == 2
    assert service.index("scripts.member_export.producer --config") < service.index(
        "scripts.member_export.producer_v2 --config"
    )
    assert "RestrictAddressFamilies=AF_UNIX" in service
    assert "OnFailure=hyprl-member-data-alert.service" in service


def test_rolling_window_and_horizons_remain_separate(tmp_path):
    value, _ = produce(observed(tmp_path), now=NOW)
    prototype = value["forecasts"][0]
    value["forecasts"] = [
        {**prototype, "symbol": f"T{i:02}", "probability": 0.8, "outcome": int(i == 0)}
        for i in range(21)
    ]
    value["forecasts"].append({**prototype, "horizon": "5d", "probability": 0.8, "outcome": 1})
    result = build(value, now=NOW)
    one = next(
        r
        for r in result["series"]
        if r["agent"] == "HypR-A1" and r["mode"] == "REGULAR_0905" and r["horizon"] == "1d"
    )
    five = next(
        r
        for r in result["series"]
        if r["agent"] == "HypR-A1" and r["mode"] == "REGULAR_0905" and r["horizon"] == "5d"
    )
    assert one["n"] == 21 and one["hits"] == 1 and one["rolling_hit_rate"] == 0
    assert five["n"] == 1 and five["hits"] == 1 and five["rolling_hit_rate"] == 1


def test_unsafe_status_directory_refuses_cli_before_any_artifact_replacement(tmp_path, monkeypatch):
    from scripts.member_export import producer_v2

    config = observed(tmp_path)
    output = tmp_path / "snapshots"
    output.mkdir(mode=0o700)
    unsafe = tmp_path / "unsafe-status"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o770)
    handoff = tmp_path / "handoff.json"
    config.update(
        handoff=str(handoff), output_dir=str(output), status_file=str(unsafe / "status.json")
    )
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps(config))
    original_git = producer_v2.git

    def pinned_git(root, *args):
        if root == producer_v2.ROOT:
            return b"a" * 40 if args == ("rev-parse", "HEAD") else b""
        return original_git(root, *args)

    monkeypatch.setattr(producer_v2, "git", pinned_git)
    assert producer_v2.main(["--config", str(configuration), "--expected-commit", "a" * 40]) == 1
    assert not handoff.exists() and not (output / "member-v2.json").exists()
