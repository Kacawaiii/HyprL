"""Synthetic source shapes at the producer CLI → unchanged exporter boundary."""

import json
import sqlite3
from datetime import datetime, timezone
import pytest

from scripts.member_export.producer import produce

NOW = datetime(2026, 10, 27, 3, tzinfo=timezone.utc)


def sources(tmp_path):
    dbfile = tmp_path / "radar.sqlite"
    with sqlite3.connect(dbfile) as db:
        db.execute(
            "CREATE TABLE evidence (seq INTEGER PRIMARY KEY, kind TEXT, key TEXT, at TEXT, body TEXT)"
        )
    briefs = tmp_path / "briefs"
    briefs.mkdir()
    forward = tmp_path / "forward"
    forward.mkdir()
    (forward / "score-runs").mkdir()
    score = {
        "provenance": "LIVE_FORWARD",
        "evidence_class": "LIVE_FORWARD",
        "scored_at": "2026-10-26T20:45:00Z",
        "rows": [],
        "failures": {},
    }
    from scripts.member_export.producer import research_digest

    (forward / "scores.json").write_text(json.dumps(score))
    (forward / "score-runs" / (research_digest(score) + ".json")).write_text(json.dumps(score))
    revenue = tmp_path / "revenue.json"
    # The running hunter keeps only an end date and value: no filing/receipt/start.
    revenue.write_text(json.dumps({"DEMO": {"Revenues": {"CY2026Q2": ["20260630", 110]}}}))
    calendar = tmp_path / "calendar.json"
    calendar.write_text(
        json.dumps(
            {
                "schema": "agent-v2-quality-calendar-v2",
                "events": [{"at": "2026-10-28T00:00:00Z", "kind": "FOMC", "symbol": None}],
                "received_at": "2026-10-23T07:00:00Z",
            }
        )
    )
    return {
        "radar_db": str(dbfile),
        "brief_dir": str(briefs),
        "forward_dir": str(forward),
        "reveal_dir": str(tmp_path / "reveals"),
        "ledger": str(tmp_path / "ledger"),
        "calendar_file": str(calendar),
        "revenue_file": str(revenue),
    }


def news(config, **changes):
    row = {
        "source": "fed:press",
        "primary": True,
        "headline": "Federal Reserve Board announces a meeting",
        "headline_hash": "a" * 64,
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/demo.htm",
        "published_at": "2026-10-26T12:00:00Z",
        "received_at": "2026-10-26T12:01:00Z",
        "symbols": [],
    }
    row.update(changes)
    with sqlite3.connect(config["radar_db"]) as db:
        db.execute(
            "INSERT INTO evidence(kind,key,at,body) VALUES(?,?,?,?)",
            ("news", "synthetic", row["received_at"], json.dumps(row)),
        )
    return row


def test_real_shapes_produce_only_causal_facts_without_manufacturing_missing_evidence(tmp_path):
    config = sources(tmp_path)
    news(config)
    news(
        config,
        source="sec:8k",
        published_at=None,
        filing={"updated_at": "2026-10-26T11:00:00Z"},
        url="https://www.sec.gov/Archives/edgar/data/1/demo.htm",
    )
    handoff, status = produce(config, now=NOW)
    assert len(handoff["radar"]) == 1
    assert handoff["radar"][0]["source"]["available_at"] == "2026-10-26T12:01:00Z"
    assert handoff["provenance"] == "observed"
    assert all(handoff[k] == [] for k in ("forecasts", "revenue", "claims", "calendar"))
    assert status["excluded"]["radar_publication_unknown"] == 1
    assert status["excluded"]["calendar_unattested_time_or_source"] == 1
    assert status["excluded"]["revenue_missing_filing_intervals_and_receipts"] == 1


def test_future_private_advice_and_unknown_publication_never_enter_snapshot(tmp_path):
    config = sources(tmp_path)
    news(config, headline="Achetez DEMO")
    news(config, headline="Bearer secret")
    news(config, headline="GPT model result")
    news(config, received_at="2026-10-28T12:00:00Z")
    news(config, published_at="2026-10-27T12:00:00Z")
    handoff, _ = produce(config, now=NOW)
    assert handoff["radar"] == []


def test_intraday_and_multiday_predictions_cannot_be_rehashed_into_v1_proofs(tmp_path):
    config = sources(tmp_path)
    from pathlib import Path

    from scripts.member_export.producer import research_digest

    base = Path(config["forward_dir"])
    value = json.loads((base / "scores.json").read_text())
    value["rows"] = [
        {
            "provenance": "LIVE_FORWARD",
            "mode": mode,
            "panel": "a_v1_as_live:1d",
            "p": 0.6,
            "outcome": 1,
        }
        for mode in ("REGULAR_0905", "INTRADAY")
    ]
    (base / "scores.json").write_text(json.dumps(value))
    (base / "score-runs" / (research_digest(value) + ".json")).write_text(json.dumps(value))
    handoff, status = produce(config, now=NOW)
    assert handoff["forecasts"] == []
    assert status["excluded"]["forecast_event_and_commitment_incompatible_v1"] == 2


def test_cli_exports_atomically_and_keeps_prior_snapshot_on_refusal(tmp_path):
    import hashlib
    from pathlib import Path

    from scripts.member_export.producer import main

    config = sources(tmp_path)
    news(config)
    output = tmp_path / "snapshots"
    output.mkdir(mode=0o700)
    config.update(
        handoff=str(tmp_path / "handoff.json"),
        output_dir=str(output),
        status_file=str(tmp_path / "status.json"),
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    original = hashlib.sha256(Path(config["radar_db"]).read_bytes()).hexdigest()
    assert main(["--config", str(config_file)]) == 0
    published = output / "member-v1.json"
    before = published.read_bytes()
    handoff_before = Path(config["handoff"]).read_bytes()
    assert json.loads(before)["provenance"] == "observed"
    assert published.stat().st_mode & 0o777 == 0o440
    assert hashlib.sha256(Path(config["radar_db"]).read_bytes()).hexdigest() == original
    (Path(config["forward_dir"]) / "scores.json").write_text('{"provenance":"synthetic"}')
    assert main(["--config", str(config_file)]) == 1
    assert published.read_bytes() == before
    assert Path(config["handoff"]).read_bytes() == handoff_before


def test_briefs_require_confirmed_delivery_and_preexisting_source(tmp_path):
    import hashlib
    from pathlib import Path

    config = sources(tmp_path)
    story = news(config, headline="Fed announces its policy statement")
    report = {"live": True, "events": [{"stories": [story]}], "markdown": "Synthetic report"}
    stored = "2026-10-26T12:02:00Z"
    with sqlite3.connect(config["radar_db"]) as db:
        db.execute(
            "INSERT INTO evidence(kind,key,at,body) VALUES(?,?,?,?)",
            ("report", "2026-10-26-morning", stored, json.dumps(report)),
        )
    meta = {
        "id": "2026-10-26-semaine",
        "at": "2026-10-26T12:03:00Z",
        "report": {
            "key": "2026-10-26-morning",
            "stored_at": stored,
            "hash": hashlib.sha256(json.dumps("Synthetic report").encode()).hexdigest()[:16],
        },
        "items": [{"topic": "fed", "headline": story["headline"]}],
    }
    directory = Path(config["brief_dir"])
    (directory / "2026-10-26-semaine.json").write_text(json.dumps(meta))
    receipt = {
        "brief": meta["id"],
        "at": "2026-10-26T12:04:00Z",
        "status": "UNCONFIRMED",
        "messages": 2,
        "sent": 2,
    }
    ledger = directory / "deliveries.jsonl"
    ledger.write_text(json.dumps(receipt) + "\n")
    assert produce(config, now=NOW)[0]["briefs"] == []
    receipt["status"] = "SENT"
    ledger.write_text(json.dumps(receipt) + "\n")
    handoff, _ = produce(config, now=NOW)
    assert len(handoff["briefs"]) == 1
    assert handoff["briefs"][0]["facts"][0]["summary"] == story["headline"]
    # Rebuilt after delivery is not evidence for what was already sent.
    meta["at"] = "2026-10-26T12:05:00Z"
    (directory / "2026-10-26-semaine.json").write_text(json.dumps(meta))
    assert produce(config, now=NOW)[0]["briefs"] == []


def test_rendered_jobs_are_isolated_and_refresh_after_successful_reveal(tmp_path):
    from scripts.member_export.jobs import alert, units

    rendered = units(
        checkout="/synthetic/checkout",
        python="/synthetic/python",
        config="/synthetic/config.json",
        notify="/synthetic/notify",
        commit="a" * 40,
        research_python="/synthetic/research-python",
        research_environment="/synthetic/research.env",
    )
    assert len(rendered) == 5
    assert (
        "OnSuccess=hyprl-member-producer.service"
        in rendered["hyprl-research-forward-reveal.service"]
    )
    assert "OnFailure=hyprl-member-data-alert.service" in rendered["hyprl-member-producer.service"]
    assert "RestrictAddressFamilies=AF_UNIX" in rendered["hyprl-member-producer.service"]
    notify = tmp_path / "notify"
    notify.write_text(
        '#!/usr/bin/env python3\nimport sys\nassert sys.argv[1]=="alertes"\n'
        'assert "MEMBER_DATA_JOB_FAILED" in sys.stdin.read()\nprint("discord")\n'
    )
    notify.chmod(0o700)
    assert alert(notify) == 0
    notify.write_text('#!/bin/sh\nprintf "telegram-fallback\\n"\n')
    assert alert(notify) == 1


def test_failures_need_published_reveal_bytes_and_keep_public_agent_names(tmp_path):
    import hashlib
    import os
    import subprocess
    from pathlib import Path

    import pytest

    from scripts.member_export.producer import research_digest

    config = sources(tmp_path)
    base = Path(config["forward_dir"])
    day = "2026-10-26"
    original = {
        "at": day + "T13:25:00Z",
        "error": "INSUFFICIENT_CALL_WINDOW",
        "snapshot_hash": "b" * 64,
    }
    (base / day).mkdir()
    (base / day / "opus-failure.json").write_text(json.dumps(original))
    value = json.loads((base / "scores.json").read_text())
    value["failures"] = {
        "opus": {"count": 1, "failed_sessions": [{"session": day, "error": original["error"]}]}
    }
    (base / "scores.json").write_text(json.dumps(value))
    (base / "score-runs" / (research_digest(value) + ".json")).write_text(json.dumps(value))
    assert produce(config, now=NOW)[0]["failures"] == []
    ledger = Path(config["ledger"])
    ledger.mkdir()

    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(ledger), *args],
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "Synthetic",
                "GIT_AUTHOR_EMAIL": "synthetic@example.com",
                "GIT_COMMITTER_NAME": "Synthetic",
                "GIT_COMMITTER_EMAIL": "synthetic@example.com",
                "GIT_AUTHOR_DATE": day + "T21:00:00Z",
                "GIT_COMMITTER_DATE": day + "T21:00:00Z",
            },
        )

    run("init", "-q", "-b", "ledger/forward-duel")
    name = "reveals/" + day + "/README.md"
    readme = ledger / name
    readme.parent.mkdir(parents=True)
    readme.write_text("| HypR-A1 | NO_DECISION | " + research_digest(original) + " |\n")
    run("add", ".")
    run("commit", "-qm", "Synthetic reveal")
    commit = run("rev-parse", "HEAD").decode().strip()
    run("update-ref", "refs/remotes/origin/ledger/forward-duel", commit)
    reveals = Path(config["reveal_dir"])
    reveals.mkdir()
    marker = {
        "state": "DELIVERED",
        "reveal_commit": commit,
        "delivered_at": day + "T21:00:05Z",
        "files": {name: hashlib.sha256(readme.read_bytes()).hexdigest()},
    }
    proof = reveals / (day + ".json")
    proof.write_text(json.dumps(marker))
    failure = produce(config, now=NOW)[0]["failures"][0]
    assert (failure["agent"], failure["code"], failure["session"]) == ("HypR-A1", "late", day)
    assert failure["source"]["available_at"] == day + "T21:00:05Z"
    marker["files"][name] = "f" * 64
    proof.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="REVEAL_MANIFEST_MISMATCH"):
        produce(config, now=NOW)


@pytest.mark.parametrize('case', ['sourced', 'future', 'ambiguous', 'unrelated'])
def test_grouped_brief_projects_only_its_observed_source_headline(tmp_path, case):
    import hashlib
    from pathlib import Path
    config = sources(tmp_path)
    story = news(config, source='sec:8k', headline='Company reports a quarterly filing',
                 url='https://www.sec.gov/Archives/edgar/data/1/000000000126000001/report.htm')
    if case == 'future': story['received_at'] = '2026-10-26T12:05:00Z'
    grouped = 'Synthetic grouped results'
    report = {'live':True, 'markdown':'Synthetic report',
              'events':[{'headline':grouped,'stories':[story]}]}
    if case == 'ambiguous': report['events'].append(dict(report['events'][0]))
    if case == 'unrelated': report['events'][0]['headline'] = 'Another group'
    stored = '2026-10-26T12:02:00Z'
    with sqlite3.connect(config['radar_db']) as db:
        db.execute('INSERT INTO evidence(kind,key,at,body) VALUES(?,?,?,?)',
                   ('report','synthetic-report',stored,json.dumps(report)))
    meta = {'id':'2026-10-26-semaine','at':'2026-10-26T12:03:00Z',
            'report':{'key':'synthetic-report','stored_at':stored,
                      'hash':hashlib.sha256(json.dumps(report['markdown']).encode()).hexdigest()[:16]},
            'items':[{'topic':'resultats','headline':grouped}]}
    directory=Path(config['brief_dir'])
    (directory/'2026-10-26-semaine.json').write_text(json.dumps(meta))
    (directory/'deliveries.jsonl').write_text(json.dumps({'brief':meta['id'],'at':'2026-10-26T12:04:00Z',
                                                       'status':'SENT','messages':1,'sent':1})+'\n')
    value,_=produce(config,now=NOW)
    if case == 'sourced':
        row=value['briefs'][0]['facts'][0]
        assert row['kind']=='sec_8k' and row['summary']==story['headline']
        assert row['summary']!=grouped and row['source']['url']==story['url']
        assert row['source']['sha256']==story['headline_hash']
    else:
        assert value['briefs']==[]
