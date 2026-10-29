"""Synthetic-only safety and contract checks; no external requests."""
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone

import pytest

from scripts.video_factory import delivery, factory, render, study
from scripts.video_factory.content import DISCLAIMER, blocked_verdict, evergreen
from scripts.video_factory.install import units

UTC = timezone.utc


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


def observed(day="2025-07-07"):
    reveals, forecasts = [], []
    for agent in factory.AGENTS:
        proof = {"id": day+"-"+agent, "session": day, "mode": "REGULAR_0905", "agent": agent,
                 "status": "REVEALED", "sha256": "a"*64,
                 "published_at": day+"T20:15:00Z", "available_at": day+"T20:16:00Z",
                 "url": "https://github.com/Kacawaiii/HyprL/blob/"+"b"*40+"/reveals/"+day+"/"+agent+".json",
                 "commitment": {"committed_at": day+"T13:05:00Z", "published_at": day+"T13:06:00Z", "sha256": "a"*64}}
        reveals.append(proof)
        forecasts.append({"session": day, "mode": "REGULAR_0905", "variant": "a_v1_as_live", "horizon": "1d",
                          "agent": agent, "symbol": "SPY", "reveal_id": proof["id"], "decision_sha256": "a"*64,
                          "decision_time": day+"T13:05:00Z", "scored_at": day+"T20:45:00Z", "probability": .6, "outcome": 1})
    return {"provenance": "observed", "generated_at": day+"T20:46:00Z", "reveals": reveals, "forecasts": forecasts}


def test_all_cards_have_a_constant_visible_disclaimer():
    for content in [evergreen(i) for i in range(5)] + [blocked_verdict()]:
        footer = None
        for i in range(len(content["captions"])):
            frame = render.card(content, i)
            assert frame.size == (720, 1280)
            crop = frame.crop((60, 1170, 700, 1210)).tobytes()
            assert len(set(crop)) > 3  # actual glyph pixels, not an empty footer
            assert footer is None or crop == footer
            footer = crop


@pytest.mark.parametrize("at", ["11:30", "12:00", "17:14"])
def test_trading_job_window_refused(at):
    with pytest.raises(RuntimeError):
        factory.guard_clock(datetime.fromisoformat("2025-07-07T"+at+":00+00:00"))


@pytest.mark.parametrize("at", ["11:29", "17:15", "20:50", "21:50"])
def test_safe_clock_times(at):
    factory.guard_clock(datetime.fromisoformat("2025-07-07T"+at+":00+00:00"))


def test_real_exchange_calendar_skips_weekends_and_holidays():
    assert not factory.trading_day("2025-07-04")
    assert not factory.trading_day("2025-07-05")
    assert factory.trading_day("2025-07-07")


def test_duel_separates_session_panel_and_horizon(tmp_path):
    value = observed()
    value["forecasts"].append({**value["forecasts"][0], "horizon": "5d", "probability": 0, "outcome": 1})
    p = dump(tmp_path/"snapshot.json", value)
    now = study.instant("2025-07-07T21:05:00Z")
    result = factory.duel(p, "2025-07-07", now)
    assert result["chart"] == [{"agent": a, "hits": 1, "n": 1} for a in factory.AGENTS]
    assert "0.160" in result["captions"][1]
    assert factory.duel(p, "2025-07-08", now) is None


def test_matured_scores_keep_the_original_forecast_date(tmp_path):
    value = observed("2025-07-07")
    fresh = observed("2025-07-08")
    value["generated_at"] = fresh["generated_at"]
    value["reveals"].extend(fresh["reveals"])
    # A decision is published today, but its one-day scores are not ready yet.
    p = dump(tmp_path/"snapshot.json", value)
    result = factory.duel(p, "2025-07-08", study.instant("2025-07-08T21:05:00Z"))
    assert result["score_session"] == "2025-07-07"
    assert "2025-07-07" in result["captions"][1]
    assert "2025-07-08" in result["captions"][0]


def test_no_decision_is_visible_without_inventing_a_score(tmp_path):
    value = observed()
    value["forecasts"] = []
    for proof in value["reveals"]:
        proof["status"] = "NO_DECISION"
        proof["url"] = proof["url"].rsplit("/", 1)[0]+"/README.md"
    result = factory.duel(dump(tmp_path/"snapshot.json", value), "2025-07-07", study.instant("2025-07-07T21:05:00Z"))
    assert "aucune décision" in result["captions"][0]
    assert not result["chart"] and result["score_session"] is None
    assert "aucun score" in result["captions"][1]


def test_published_luck_is_scoped_and_never_recomputed(tmp_path):
    value = observed()
    value["series"] = [{"agent": "HypR-A1", "variant": "a_v1_as_live", "horizon": "1d", "mode": "REGULAR_0905",
                        "points": [{"at": value["generated_at"], "session": "2025-07-07", "luck_tail": .42}]}]
    p = dump(tmp_path/"snapshot.json", value)
    result = factory.duel(p, "2025-07-07", study.instant("2025-07-07T21:05:00Z"))
    assert "42.0 %" in result["captions"][3]
    value["series"][0]["horizon"] = "5d"
    result = factory.duel(dump(p, value), "2025-07-07", study.instant("2025-07-07T21:05:00Z"))
    assert "n'est pas disponible" in result["captions"][3]


@pytest.mark.parametrize("change", ["synthetic", "future", "stale", "missing_reveal"])
def test_duel_never_uses_unavailable_snapshot(tmp_path, change):
    value = observed()
    if change == "synthetic":
        value["provenance"] = "synthetic"
    if change == "future":
        value["generated_at"] = "2025-07-08T20:46:00Z"
    if change == "stale":
        value["generated_at"] = "2025-07-05T20:46:00Z"
    if change == "missing_reveal":
        value["reveals"].pop()
    p = dump(tmp_path/"snapshot.json", value)
    assert factory.duel(p, "2025-07-07", study.instant("2025-07-07T20:50:00Z")) is None


@pytest.mark.parametrize("change", ["bad_sha", "duplicate", "nan", "crypto", "future_score", "unpublished_commitment"])
def test_invalid_forecasts_are_refused(tmp_path, change):
    value = observed()
    row = value["forecasts"][0]
    if change == "bad_sha":
        row["decision_sha256"] = "c"*64
    if change == "duplicate":
        value["forecasts"].append(dict(row))
    if change == "nan":
        row["probability"] = float("nan")
    if change == "crypto":
        row["symbol"] = "BTC"
    if change == "future_score":
        row["scored_at"] = "2025-07-08T20:45:00Z"
    if change == "unpublished_commitment":
        value["reveals"][0]["commitment"]["published_at"] = "2025-07-07T20:30:00Z"
    with pytest.raises(ValueError):
        factory.duel(dump(tmp_path/"snapshot.json", value), "2025-07-07", study.instant("2025-07-07T20:50:00Z"))


def test_daily_falls_back_and_is_idempotent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(factory, "trading_day", lambda day: True)
    monkeypatch.setattr(factory, "duel", lambda *a: None)
    def produce(config, identifier, content, publish=True):
        calls.append(content)
        return {"status": "delivered", "id": identifier}
    monkeypatch.setattr(factory, "produce", produce)
    cfg = {"state_dir": str(tmp_path), "snapshot": "unused"}
    now = study.instant("2025-07-07T21:05:00Z")
    assert factory.daily(cfg, now)["status"] == "delivered"
    assert calls[0]["kind"] == "evergreen"
    assert factory.daily(cfg, now)["status"] == "already_delivered"
    assert len(calls) == 1


def test_retention_preserves_foreign_paths_and_unsent_videos(tmp_path):
    for i in range(1, 16):
        p = tmp_path/f"daily-2025-07-{i:02}"
        p.mkdir()
        dump(p/"metadata.json", {})
        dump(p/"delivery.json", {"status": "delivered"})
    foreign = tmp_path/"keep-me"
    foreign.mkdir()
    factory.retain(tmp_path)
    assert foreign.exists()
    assert len(list(tmp_path.glob("daily-*"))) == 14
    earliest = min(tmp_path.glob("daily-*"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    dump(earliest/"delivery.json", {"status": "pending"})
    with pytest.raises(RuntimeError):
        factory.retain(tmp_path, 13)
    assert earliest.exists()


def fake_bundle(tmp_path):
    bundle = tmp_path/"bundle"
    bundle.mkdir()
    video = b"synthetic-test-video"
    (bundle/"video.mp4").write_bytes(video)
    (bundle/"caption.txt").write_text("Synthetic caption")
    dump(bundle/"metadata.json", {"sha256": hashlib.sha256(video).hexdigest(), "title": "Test",
                                 "description": DISCLAIMER, "hashtags": [], "ai_label_reminder": "Test"})
    env = tmp_path/"webhook.env"
    env.write_text('DISCORD_WEBHOOK_BROUILLONS="https://discord.com/api/webhooks/000/synthetic"\n')
    return bundle, env


def test_multipart_attachment_delivery_acknowledged_once(tmp_path, monkeypatch):
    bundle, env = fake_bundle(tmp_path)
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"id":"1234","attachments":[{},{}]}'
    def post(req, timeout):
        calls.append(req)
        assert b'name="files[0]"; filename="video.mp4"' in req.data
        assert b'name="files[1]"; filename="caption.txt"' in req.data
        assert b'"parse": []' in req.data
        return Response()
    monkeypatch.setattr(delivery.urllib.request, "urlopen", post)
    assert delivery.upload(bundle, env)["status"] == "delivered"
    delivery.upload(bundle, env)
    assert len(calls) == 1


def test_timeout_is_never_blindly_retried_or_leaked(tmp_path, monkeypatch):
    bundle, env = fake_bundle(tmp_path)
    def post(*a, **kw): raise TimeoutError("private-url-would-be-here")
    monkeypatch.setattr(delivery.urllib.request, "urlopen", post)
    with pytest.raises(RuntimeError, match="unconfirmed") as error:
        delivery.upload(bundle, env)
    assert "private-url" not in str(error.value)
    with pytest.raises(RuntimeError, match="uncertain previous"):
        delivery.upload(bundle, env)


def grid(day, low=10):
    day = date.fromisoformat(day)
    at = datetime.combine(day, time(2), study.PARIS).astimezone(UTC)
    end = datetime.combine(day+timedelta(days=1), time(), study.PARIS).astimezone(UTC)
    rows = {}
    while at < end:
        rows[at] = (int(at.timestamp()), low, 20, 15, 15, 1)
        at += timedelta(hours=1)
    return rows


@pytest.mark.parametrize("day,n", [("2025-03-30", 21), ("2025-10-26", 23), ("2025-07-07", 22)])
def test_paris_dst_complete_sessions_and_strict_sweep(day, n):
    rows = grid(day)
    assert len(rows) == n
    d = date.fromisoformat(day)
    events, excluded = study.daily_events(rows, 3600, d, d+timedelta(days=1))
    assert not excluded and events[0]["taken"] is False  # equality is a touch, not a sweep
    at = datetime.combine(d, time(11), study.PARIS).astimezone(UTC)
    rows[at] = (int(at.timestamp()), 9, 20, 15, 15, 1)
    assert study.daily_events(rows, 3600, d, d+timedelta(days=1))[0][0]["taken"] is True
    del rows[at]
    assert study.daily_events(rows, 3600, d, d+timedelta(days=1)) == ([], [day])


@pytest.mark.parametrize("change", ["holdout", "daily", "unbound_plan", "no_authority"])
def test_manifest_refuses_unsafe_data_before_opening_shard(tmp_path, change):
    m = {"product": "BTC-USD", "source": "coinbase_exchange", "plan_sha256": study.plan_hash(),
         "shards": [{"file": "DOES-NOT-EXIST.json", "start": "2025-01-01T00:00:00Z", "end_exclusive": "2025-01-02T00:00:00Z",
                     "granularity_seconds": 3600, "authorization_id": "synthetic-grant", "sha256": "a"*64}]}
    if change == "holdout":
        m["shards"][0]["end_exclusive"] = "2026-12-01T00:00:00Z"
    if change == "daily":
        m["shards"][0]["granularity_seconds"] = 86400
    if change == "unbound_plan":
        m["plan_sha256"] = "b"*64
    if change == "no_authority":
        m["shards"][0]["authorization_id"] = ""
    with pytest.raises(ValueError):
        study.load_candles(dump(tmp_path/"manifest.json", m))


def test_candle_shape_hash_and_duplicate_validation(tmp_path):
    rows = list(grid("2025-07-07").values())
    p = dump(tmp_path/"candles.json", {"rows": rows})
    shard = {"file": p.name, "start": "2025-07-07T00:00:00Z", "end_exclusive": "2025-07-08T00:00:00Z",
             "granularity_seconds": 3600, "authorization_id": "synthetic-grant", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    m = dump(tmp_path/"manifest.json", {"product": "BTC-USD", "source": "coinbase_exchange", "plan_sha256": study.plan_hash(), "shards": [shard, shard]})
    bars, step = study.load_candles(m)
    assert step == 3600 and len(bars) == 22
    p.write_text("[]")
    with pytest.raises(ValueError, match="digest"):
        study.load_candles(m)


def test_statistics_do_not_assume_random_means_fifty_percent():
    events = []
    start = date(2025, 1, 1)
    for i in range(60):
        day = start+timedelta(days=i)
        events.append({"day": day.isoformat(), "weekday": day.weekday(), "taken": True})
    result = study.summarize(events, [])
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["random_day"]["rate"] == 1
    assert result["random_baseline_rate_ci95"] == [1, 1]
    assert result["years_friday_beats_other_days"] == 0
    assert study.wilson(0, 0)["ci95"] is None
    assert study.wilson(5, 10)["ci95"] == pytest.approx([.236593, .763407], abs=1e-6)


def test_units_only_schedule_this_job_with_bounded_resources():
    value = units("/synthetic/checkout", "/synthetic/python", "/synthetic/config")
    assert set(value) == {"hyprl-video-factory.service", "hyprl-video-factory.timer", "hyprl-video-factory-alert.service"}
    assert "17:05:00 America/New_York" in value["hyprl-video-factory.timer"]
    assert "Persistent=false" in value["hyprl-video-factory.timer"]
    assert "MemoryMax=400M" in value["hyprl-video-factory.service"]
    assert "Nice=19" in value["hyprl-video-factory.service"]
    assert "WorkingDirectory=/synthetic/checkout\n" in value["hyprl-video-factory.service"]
