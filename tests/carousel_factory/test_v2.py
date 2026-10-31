"""Fail-closed editorial validation, real-shaped source adapters and v2 geometry."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.carousel_factory import editorial, factory
from scripts.carousel_factory.content import evergreen
from scripts.carousel_factory.data import label
from scripts.carousel_factory.facts import select, to_post
from scripts.carousel_factory.public_study import verdict
from scripts.carousel_factory.render import render, verify

UTC = timezone.utc
NOW = datetime(2026, 10, 30, 11, 15, tzinfo=UTC)
DAY = "2026-10-30"


def citation(kind="earnings", day=DAY):
    return {
        "url": {
            "earnings": "https://api.nasdaq.com/api/calendar/earnings?date=" + day,
            "cpi": "https://www.bls.gov/news.release/cpi.nr0.htm",
            "nfp": "https://www.bls.gov/news.release/empsit.nr0.htm",
            "fed": "https://www.federalreserve.gov/newsevents/pressreleases/monetary.htm",
            "sec": "https://www.sec.gov/Archives/edgar/data/123/filing.htm",
        }[kind],
        "sha256": "a" * 64,
        "published_at": day + "T10:00:00Z",
        "available_at": day + "T10:01:00Z",
    }


def snapshot():
    return {
        "generated_at": DAY + "T11:00:00Z",
        "provenance": "observed",
        "briefs": [],
        "_provenance": {"snapshot_sha256": "b" * 64},
    }


def event(symbol="AAPL", day=DAY, kind="earnings"):
    return {
        "date": day,
        "kind": kind,
        "symbol": symbol,
        "source": citation(kind, day),
        "title": "Publication officielle",
        "timing": "horaire non confirmé",
    }


def facts():
    return select(snapshot(), [event(), event("MSFT")], DAY)


def draft(f=None):
    f = f or facts()
    return {
        "hook": "Apple et Microsoft publient leurs résultats",
        "slides": [
            {"fact_id": item["id"], "title": f"Résultats de {item['symbol']}", "body": item["text"]}
            for item in f["facts"]
        ],
    }


@pytest.fixture
def editor(config, tmp_path, monkeypatch):
    root = tmp_path / "skills"
    for name in (
        "copywriting/SKILL.md",
        "copywriting/references/ai-tells.md",
        "social/SKILL.md",
        "social/references/carousel-frameworks.md",
    ):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Synthetic style context, never a fact source.")
    context = tmp_path / "product.md"
    context.write_text("Synthetic product positioning with an unlicensed number 9999.")
    config.update(editorial_enabled=True, skills_dir=str(root), product_context=str(context))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(editorial, "datetime", Clock)
    return config


def test_verified_today_calendar_does_not_require_a_brief_or_tomorrow():
    f = facts()
    assert f["count"] == 2
    assert f["facts"][0]["symbol"] == "AAPL"
    assert select(snapshot(), [event("OBSCURE"), event(day="2026-10-29")], DAY) is None
    assert select(None, [event()], DAY) is None
    bad = event()
    bad["source"]["url"] = "https://api.nasdaq.com.attacker.test/api/calendar/earnings?date=" + DAY
    assert select(snapshot(), [bad], DAY) is None
    bad = event()
    bad["source"]["url"] = "https://api.nasdaq.com/api/calendar/earnings?date=2026-11-02"
    assert select(snapshot(), [bad], DAY) is None


def test_macro_and_large_cap_sec_facts_are_today_and_source_bound(tmp_path):
    x = snapshot()
    x["briefs"] = [
        {
            "date": DAY,
            "published_at": DAY + "T10:30:00Z",
            "facts": [
                {
                    "kind": "cpi",
                    "symbol": None,
                    "summary": "Inflation américaine : 2,5 %.",
                    "source": citation("cpi"),
                },
                {
                    "kind": "nfp",
                    "symbol": None,
                    "summary": "Emploi américain : 100000 emplois.",
                    "source": citation("nfp"),
                },
                {
                    "kind": "sec",
                    "symbol": "AAPL",
                    "summary": "AAPL publie un 8-K.",
                    "source": citation("sec"),
                },
            ],
        }
    ]
    assert [f["kind"] for f in select(x, [], DAY)["facts"]] == ["cpi", "nfp", "sec"]
    x["briefs"][0]["facts"][0]["source"] = citation("cpi", "2026-10-29")
    assert select(x, [], DAY)["count"] == 2
    x["briefs"][0]["facts"][1]["source"]["available_at"] = DAY + "T12:00:00Z"
    with pytest.raises(ValueError, match="noncausal"):
        select(x, [], DAY)


def test_date_marker_retains_collection_clock_without_invented_publication():
    r = event()
    r["source"].update(published_at=None, event_sha256="c" * 64, dataset_sha256="d" * 64)
    assert select(snapshot(), [r], DAY)["count"] == 1
    assert "publication non horodatée" in label(r["source"])
    del r["source"]["event_sha256"]
    assert select(snapshot(), [r], DAY) is None


@pytest.mark.parametrize(
    "invention",
    [
        "AAPL gagne 99 %.",
        "TSLA publie ses résultats.",
        "AAPL publie le 2027-10-30.",
        "AAPL publie le 31/10/2026.",
        "AAPL à 15 h 30.",
        "Achetez AAPL.",
        "Renforcez AAPL.",
        "$tsla publie ses résultats.",
        "Vendez AAPL.",
        "AAPL : rendement garanti.",
        "AAPL va monter.",
        "AAPL rapporte quatre milliards.",
        "AAPL publie demain.",
        "AAPL à midi.",
        "AAPL a doublé.",
        "AAPL atteint ９９ %.",
        "AAPL à 1\u200b5 heures.",
    ],
)
def test_copy_rejects_unlicensed_claims_and_advice(invention):
    f = facts()
    text = draft(f)
    text["slides"][0]["body"] = invention + " Horaire non confirmé."
    with pytest.raises(ValueError):
        editorial.validate_copy(text, f)


def test_validation_checks_each_fact_rather_than_global_number_bag():
    f = facts()
    f["facts"][0]["text"] += " 10 %."
    f["facts"][1]["text"] += " 20 %."
    text = draft(f)
    text["slides"][0]["body"] = "AAPL : 20 %. Horaire non confirmé."
    with pytest.raises(ValueError, match="unsupported number"):
        editorial.validate_copy(text, f)
    text = draft(f)
    text["slides"][0]["title"] = "MSFT publie"
    with pytest.raises(ValueError, match="ticker"):
        editorial.validate_copy(text, f)


def test_unknown_hour_generic_hook_missing_subject_and_missing_fact_rejected():
    text = draft()
    text["slides"][0]["body"] = "AAPL publie ses résultats."
    with pytest.raises(ValueError, match="unknown schedule"):
        editorial.validate_copy(text, facts())
    text = draft()
    text["hook"] = "Votre matinée de marché"
    with pytest.raises(ValueError, match="generic"):
        editorial.validate_copy(text, facts())
    text = draft()
    text["slides"].pop()
    with pytest.raises(ValueError, match="count"):
        editorial.validate_copy(text, facts())


def test_one_sonnet_call_has_no_tools_mcp_hooks_or_session_and_cached_copy(editor):
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        assert args[args.index("--model") + 1] == "sonnet"
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
        assert args[args.index("--settings") + 1] == '{"disableAllHooks":true}'
        assert "--no-session-persistence" in args and "--strict-mcp-config" in args
        assert json.loads(kwargs["input"])["schema"] == "carousel-daily-facts-v2"
        assert "9999" in args[-1] and kwargs["timeout"] == 120
        return SimpleNamespace(stdout=json.dumps({"result": json.dumps(draft())}))

    f = facts()
    assert editorial.edit(editor, f, NOW, runner=runner) == draft()
    # Fresh snapshot clocks do not consume a second call or invalidate unchanged facts.
    f["provenance"]["snapshot_sha256"] = "e" * 64
    assert editorial.edit(editor, f, NOW, runner=runner) == draft()
    assert len(calls) == 1
    f["facts"][0]["text"] += " Changed."
    assert editorial.edit(editor, f, NOW, runner=runner) is None
    assert len(calls) == 1


@pytest.mark.parametrize("result", ["timeout", "invalid", "bad_number", "envelope"])
def test_uncertain_or_rejected_copy_never_retries_or_posts(editor, result):
    calls = []

    def runner(*args, **kwargs):
        calls.append(1)
        if result == "timeout":
            raise subprocess.TimeoutExpired("synthetic", 120)
        if result == "invalid":
            return SimpleNamespace(stdout="not json")
        if result == "envelope":
            return SimpleNamespace(stdout="[]")
        value = draft()
        value["hook"] = "Apple annonce 9999 résultats"
        return SimpleNamespace(stdout=json.dumps({"result": json.dumps(value)}))

    assert editorial.edit(editor, facts(), NOW, runner=runner) is None
    assert editorial.edit(editor, facts(), NOW, runner=runner) is None
    assert calls == [1]


@pytest.mark.parametrize(
    "clock,allowed", [("11:29", True), ("11:30", False), ("17:14", False), ("17:15", True)]
)
def test_exact_utc_peak_window(clock, allowed):
    assert (
        editorial.outside_peak(datetime.fromisoformat(DAY + "T" + clock + ":00+00:00")) == allowed
    )


def test_peak_or_disabled_editorial_has_no_call_or_reservation(editor):
    def forbidden(*args, **kwargs):
        pytest.fail("editorial must not start")

    assert editorial.edit(editor, facts(), NOW.replace(hour=12), runner=forbidden) is None
    editor["editorial_enabled"] = False
    assert editorial.edit(editor, facts(), NOW, runner=forbidden) is None
    assert not list(Path(editor["state_dir"]).glob("editorial-daily-*.json"))


def test_prepared_copy_publishes_in_summer_peak_without_a_new_call(editor, monkeypatch):
    monkeypatch.setattr(editorial, "outside_peak", lambda now: True)
    editorial.edit(
        editor,
        facts(),
        NOW,
        runner=lambda *a, **k: SimpleNamespace(stdout=json.dumps({"result": json.dumps(draft())})),
    )
    monkeypatch.setattr(editorial, "outside_peak", lambda now: False)
    assert (
        editorial.edit(
            editor, facts(), NOW.replace(hour=16), runner=lambda *a, **k: pytest.fail("second call")
        )
        == draft()
    )


def test_real_daily_post_renders_even_with_one_good_fact(config):
    f = facts()
    f["facts"] = f["facts"][:1]
    f["count"] = 1
    text = draft(f)
    text["hook"] = "Apple publie ses résultats"
    editorial.validate_copy(text, f)
    p = to_post(f, text)
    meta = render(p, Path(config["output_dir"]) / "one-fact", config["font_dir"])
    assert len(meta["platforms"]["instagram"]) == 3
    assert "AAPL" in meta["platforms"]["instagram"][1]["body"]


def test_daily_no_fact_or_failed_editorial_means_no_product_no_publish(config, monkeypatch):
    monkeypatch.setattr(factory, "inputs", lambda *a: (snapshot(), [event()]))
    monkeypatch.setattr(factory, "edit", lambda *a: None)
    monkeypatch.setattr(factory, "produce", lambda *a, **k: pytest.fail("filler post"))
    result = factory.run(config, "daily", NOW.replace(hour=17, minute=30), True)
    assert result["status"] == "skipped" and "editorial" in result["reason"]


def test_weekly_evergreen_cap_reserves_before_an_uncertain_upload(config, monkeypatch):
    monkeypatch.setattr(factory, "inputs", lambda *a: (None, []))
    monkeypatch.setattr(factory, "product", lambda *a: evergreen(0))
    calls = []

    def uncertain(*args):
        calls.append(1)
        raise RuntimeError("synthetic uncertainty")

    monkeypatch.setattr(factory, "produce", uncertain)
    now = NOW.replace(hour=18, minute=15)
    with pytest.raises(RuntimeError):
        factory.run(config, "evergreen", now, True)
    result = factory.run(config, "evergreen", now, True)
    assert result["reason"] == "weekly evergreen already reserved" and calls == [1]


def test_public_btc_v2_has_honest_bars_plain_verdict_and_statistics_in_caption(config):
    p = verdict()
    visible = " ".join(s["title"] + " " + s["body"] for s in p["slides"])
    assert all(word not in visible for word in ("95 %", "p =", "holdout"))
    assert "p = 0,075" in p["caption_hook"]
    assert p["slides"][3]["chart"][0]["value"] == 251 / 450
    assert p["slides"][3]["chart"][1]["value"] == 1567 / 2701
    assert p["slides"][4]["title"] == "Aucun avantage du vendredi établi."
    dest = Path(config["output_dir"]) / "btc"
    meta = render(p, dest, config["font_dir"])
    assert meta["edition"] == "carrousels v2"
    with Image.open(dest / "instagram/05.png") as im:
        assert im.getpixel((0, 0)) == (143, 180, 255)
    assert meta["platforms"]["x"][1]["chart"]
    assert meta["style"]["background"] == "#0A0E1A"
    assert verify(dest)["schema"] == "hyprl-carousel-v2"


def test_logo_is_configurable_and_only_its_digest_is_in_metadata(config, tmp_path):
    logo = tmp_path / "logo.png"
    Image.new("RGBA", (100, 50), (255, 255, 255, 255)).save(logo)
    meta = render(
        evergreen(0), Path(config["output_dir"]) / "logo", config["font_dir"], {"logo": str(logo)}
    )
    assert len(meta["style"]["logo_sha256"]) == 64
    assert str(logo) not in json.dumps(meta)


def test_real_radar_shape_includes_today_sec_8k_and_excludes_old_news():
    x = snapshot()
    x["radar"] = [
        {
            "kind": "sec_8k",
            "symbol": "AAPL",
            "summary": "AAPL publie un 8-K.",
            "source": citation("sec"),
        },
        {
            "kind": "fed",
            "symbol": None,
            "summary": "La Fed publie sa décision.",
            "source": citation("fed", "2026-10-29"),
        },
    ]
    f = select(x, [], DAY)
    assert f["count"] == 1 and f["facts"][0]["kind"] == "sec"
    assert f["facts"][0]["symbol"] == "AAPL"


def test_reviewed_excerpt_translation_binds_exact_text_and_cannot_add_numbers():
    import hashlib

    from scripts.carousel_factory.data import excerpt

    x = snapshot()
    text = "The Fed publishes a document."
    x["briefs"] = [
        {
            "date": DAY,
            "published_at": DAY + "T10:30:00Z",
            "facts": [{"summary": text, "source": citation("fed")}],
        }
    ]
    key = hashlib.sha256(text.encode()).hexdigest()
    item = excerpt(x, {key: "La Fed publie un document."})
    assert item["translated"] and item["body"] == "La Fed publie un document."
    with pytest.raises(ValueError, match="number"):
        excerpt(x, {key: "La Fed publie 99 documents."})
    assert excerpt(x, {"unrelated": "Other"})["body"] == text


def test_pending_editorial_receipt_after_process_crash_never_starts_second_call(editor):
    root = Path(editor["state_dir"])
    root.mkdir(parents=True)
    (root / f"editorial-daily-{DAY}.json").write_text(json.dumps({"status": "pending"}))
    assert (
        editorial.edit(editor, facts(), NOW, runner=lambda *a, **k: pytest.fail("duplicate call"))
        is None
    )


def test_actual_clock_is_checked_again_before_spawn(editor, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.replace(hour=12)

    monkeypatch.setattr(editorial, "datetime", Clock)
    assert (
        editorial.edit(editor, facts(), NOW, runner=lambda *a, **k: pytest.fail("peak call"))
        is None
    )
    record = json.loads((Path(editor["state_dir"]) / f"editorial-daily-{DAY}.json").read_text())
    assert record["status"] == "skipped_window"


def test_evergreen_never_fills_an_empty_calendar_or_chance_slot(config, monkeypatch):
    monkeypatch.setattr(factory, "inputs", lambda *a: (None, []))
    config["public_study"] = False
    monkeypatch.setattr(factory, "produce", lambda *a, **k: pytest.fail("filler"))
    for slot, now in [
        ("calendar", datetime(2026, 11, 2, 7, tzinfo=UTC)),
        ("chance", datetime(2026, 10, 28, 17, 45, tzinfo=UTC)),
    ]:
        assert factory.run(config, slot, now, True)["status"] == "skipped"


def test_logo_change_cannot_silently_reuse_an_old_rendering(config, tmp_path):
    p = evergreen(0)
    factory.produce(config, "carousel-v2-logo-change", p)
    logo = tmp_path / "new-logo.png"
    Image.new("RGBA", (100, 50), (255, 255, 255, 255)).save(logo)
    config["brand"] = {"logo": str(logo)}
    with pytest.raises(ValueError, match="different logo"):
        factory.produce(config, "carousel-v2-logo-change", p)


def test_x_keeps_every_fact_and_calendar_company_promised_on_cover(config):
    from scripts.carousel_factory.data import weekly_calendar
    from scripts.carousel_factory.render import platform_slides

    f = select(snapshot(), [event("AAPL"), event("MSFT"), event("NVDA")], DAY)
    p = to_post(f, draft(f))
    projected = platform_slides(p, "x")
    assert len(projected) == 4 and projected[-1]["cta"]
    assert all(
        symbol in " ".join(s["body"] for s in projected) for symbol in ("AAPL", "MSFT", "NVDA")
    )
    days = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]
    rows = []
    for day, symbol in zip(days, ["AAPL", "MSFT", "NVDA", "AMZN", "META"]):
        row = event(symbol, day)
        row.update(precision="date", scheduled_at=None)
        rows.append(row)
    p = weekly_calendar(snapshot(), rows, DAY)
    projected = platform_slides(p, "x")
    assert len(projected) == 4
    text = " ".join(s["title"] + " " + s["body"] for s in projected)
    assert all(day in text for day in days)
    assert all(row["symbol"] in text for row in rows)
    meta = render(p, Path(config["output_dir"]) / "calendar-x", config["font_dir"])
    assert len(meta["platforms"]["x"]) == 4
