"""Home blocks over synthetic inputs: calendar, the AI decisions of the day and the system light."""
import json
from datetime import datetime, timezone

from scripts.radar import cockpit_today as today
from scripts.radar.cockpit_export import build_home, build_paper
from scripts.radar.cockpit_refresh import refresh
from tests.radar.test_cockpit_export import BOOK_STATUS, JOURNAL, PAPER_REPORT, RADAR, TRADER, view

NOW = datetime(2026, 10, 12, 21, 0, tzinfo=timezone.utc)
CALENDAR = {"schema": "cockpit-calendar-v1", "entries": [
    {"at": "2026-10-14T12:30:00Z", "title": "Inflation US (CPI)", "kind": "cpi", "verification": "OFFICIEL", "assets": ["SPY"]},
    {"date": "2026-10-28", "title": "Décision de la Fed", "kind": "fed", "verification": "FIL"},
    {"at": "2026-10-13T11:00:00Z", "title": "Résultats JPMorgan", "kind": "earnings", "verification": "OFFICIEL",
     "assets": ["JPM"]},
    {"at": "2026-10-12T12:00:00Z", "title": "Already past", "kind": "other"},
    {"date": "2026-10-12", "title": "Today, time unknown", "kind": "wizard"},
    {"title": "No date"}]}
WATCHLIST = {"JPM": {"catalyst": "Q3 results", "date": "2026-10-13 07:00 ET"},
             "GS": {"catalyst": "Q3 results", "date": "2026-10-13 07:30 ET"},
             "ETH/USD": {"catalyst": "Fusaka upgrade", "date": "2026-12-06"},
             "CRCL": {"catalyst": "OCC rules", "date": "2026-11"},
             "HOOD": {"catalyst": "rating", "date": "-"}}


def test_calendar_keeps_upcoming_entries_soonest_first_with_unknown_times_marked():
    entries = today.calendar_entries(CALENDAR, WATCHLIST, now=NOW)
    titles = [e["title"] for e in entries]
    assert titles == ["Today, time unknown", "Résultats JPMorgan", "GS : Q3 results", "Inflation US (CPI)",
                      "Décision de la Fed", "ETH/USD : Fusaka upgrade"]
    gs = entries[2]
    # 07:30 New York (EDT) is 11:30 UTC; the watchlist is not a verified source.
    assert gs["at"] == "2026-10-13T11:30:00Z" and gs["verification"] == "NON VERIFIE" and gs["kind"] == "earnings"
    fed = entries[4]
    assert fed["at"] is None and fed["time_known"] is False and fed["date"] == "2026-10-28"
    assert entries[0]["kind"] == "other" and entries[0]["verification"] == "NON VERIFIE"


def test_calendar_ignores_a_file_with_another_schema():
    assert today.calendar_entries({"schema": "x", "entries": CALENDAR["entries"]}, None, now=NOW) == []


def run_with(views, at="2026-10-12T12:17:59Z", slot="am"):
    return {"run_id": f"trader:{at[:10]}:real", "at": at, "status": "COMPLETE", "synthetic": False, "slot": slot,
            "views": views}


def test_ai_today_uses_the_executor_selection_and_the_broker_outcome():
    views = [view("analyst_claude", "XLE", "1d", "UP", 0.53, verdict="DOWNGRADE"),
             view("analyst_gpt", "XLE", "1d", "ABSTAIN", 0.5),
             view("analyst_claude", "XOM", "1d", "UP", 0.6), view("analyst_gpt", "XOM", "1d", "UP", 0.58),
             view("analyst_claude", "NVDA", "1d", "UP", 0.6), view("analyst_gpt", "NVDA", "1d", "DOWN", 0.4),
             view("analyst_claude", "BTC-USD", "1d", "UP", 0.6), view("analyst_gpt", "BTC-USD", "1d", "ABSTAIN", 0.5)]
    report = {"accounts": [{"account": "ia_actions", "halted": False, "downgrade_guard": {"state": "ACTIVE"}}],
              "execution_outcomes": [{"account": "ia_actions", "symbol": "XLE", "horizon": "1d", "purpose": "entry",
                                      "state": "NOT_TRADED", "broker_status": "expired", "at": "2026-10-12T15:07:25Z"}]}
    block = today.ai_today([run_with([]), run_with(views)], report)
    lines = {(l["account"], l["asset"]): l for l in block["lines"]}
    # Conflicting UP/DOWN pairs are skipped by the executor; so are they here.
    assert set(lines) == {("ia_actions", "XLE"), ("ia_actions", "XOM"), ("ia_crypto", "BTC-USD")}
    xle = lines[("ia_actions", "XLE")]
    assert (xle["status"], xle["size"], xle["direction"]) == ("expired", "half", "UP")
    # The executor already sent this run's XLE entry, so an XOM line without a trace was skipped, not planned.
    assert lines[("ia_actions", "XOM")]["status"] == "skipped" and lines[("ia_actions", "XOM")]["size"] == "full"
    # No outcome yet for the crypto account and the report is from the same day: still planned.
    assert lines[("ia_crypto", "BTC-USD")]["status"] == "planned"
    # A single kept crypto view is a half-size consensus-less entry for the crypto account.
    assert lines[("ia_crypto", "BTC-USD")]["size"] == "half"
    assert [l["status"] for l in block["lines"]][-1] == "expired"
    assert block["session"] == "2026-10-12"


def test_ai_today_marks_a_halted_downgrade_tier_as_blocked_and_handles_no_run():
    views = [view("analyst_claude", "XLE", "1d", "UP", 0.53, verdict="DOWNGRADE")]
    report = {"accounts": [{"account": "ia_actions", "downgrade_guard": {"state": "HALTED"}}]}
    assert today.ai_today([run_with(views)], report)["lines"][0]["status"] == "blocked"
    later = {"at": "2026-10-13T09:00:00Z", "accounts": []}
    assert today.ai_today([run_with([view("analyst_claude", "XOM", "1d", "UP", 0.6)])], later)["lines"][0]["status"] == "skipped"
    assert today.ai_today([], None) == {"run_id": None, "at": None, "session": None, "lines": []}


def test_system_light_counts_failures_and_staleness_but_not_sources_blocked_by_design():
    system = {"units": [{"unit": "hyprl-radar-collect.service", "state": "failed", "result": "exit-code"},
                        {"unit": "ok.timer", "state": "active", "result": "success"}],
              "trader": {"health": {"state": "HEALTHY"}, "paused": False,
                         "failures": [{"at": "2026-10-12T20:00:00Z", "code": "MISSING_ANCHOR_PRICE"},
                                      {"at": "2026-10-01T20:00:00Z", "code": "OLD"}]},
              "sources": [{"source": "x", "status": "BLOCKED"}]}
    light = today.system_light(system, radar_cutoff="2026-10-12T20:00:00Z", now=NOW)
    assert light["level"] == "red" and len(light["problems"]) == 2
    units, incidents = light["problems"]
    # Beginner text is a plain sentence; names and codes stay in `detail` for Expert mode.
    assert units["text"] == "1 tâche automatique en échec (voir l’onglet Tâches)."
    assert units["detail"] == "hyprl-radar-collect.service"
    assert "MISSING" not in incidents["text"] and incidents["detail"] == "MISSING_ANCHOR_PRICE"
    calm = today.system_light({"trader": {"health": {"state": "HEALTHY"}}}, radar_cutoff="2026-10-12T20:00:00Z", now=NOW)
    assert calm == {"level": "green", "problems": [], "checked_at": "2026-10-12T21:00:00Z"}
    stale = today.system_light({}, radar_cutoff="2026-10-12T01:00:00Z", now=NOW)
    assert stale["level"] == "orange"
    unchecked = today.system_light({"trader": {}, "units": []}, radar_cutoff="2026-10-12T20:00:00Z", now=NOW)
    assert unchecked["level"] == "unknown" and unchecked["problems"] == []


def test_home_carries_the_blocks_and_the_book_gets_its_previous_close():
    home = build_home(RADAR, TRADER, calendar=today.calendar_entries(CALENDAR, None, now=NOW),
                      ai_today=today.ai_today(TRADER["runs"]), system={"level": "green", "problems": []})
    assert home["calendar"][0]["title"] == "Today, time unknown"
    assert home["ai_today"]["run_id"] == "trader:2026-10-10:real"
    assert build_home(RADAR, TRADER)["calendar"] == [] and build_home(RADAR, TRADER)["system"] is None
    report = {**PAPER_REPORT, "claude_book": {"equity": "107290", "last_equity": "107277.83"}}
    book = build_paper(report, BOOK_STATUS, JOURNAL)["accounts"][-1]
    assert book["last_equity"] == 107277.83 and book["day_pnl"] == round(107303.58 - 107277.83, 2)
    assert build_paper(PAPER_REPORT, BOOK_STATUS, JOURNAL)["accounts"][-1]["day_pnl"] is None


def test_refresh_writes_calendar_ai_lines_light_and_fresh_benchmark(tmp_path):
    from scripts.trading_lab.research.store import ResearchStore
    runtime = tmp_path / "runtime"
    ResearchStore(runtime / "evidence").close()
    files = {"radar.json": RADAR, "calendar.json": CALENDAR, "watchlist.json": WATCHLIST}
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in JOURNAL))
    out = tmp_path / "snapshots"
    out.mkdir()
    (out / "paper.json").write_text(json.dumps(build_paper(PAPER_REPORT, BOOK_STATUS, JOURNAL,
                                                           generated_at="2026-10-10T17:10:00Z")))
    fresh = {**PAPER_REPORT, "at": "2026-10-12T20:30:00Z", "claude_book": {"last_equity": "107000"},
             "benchmarks": {"SPY": {"price": "779", "return_since_paper_start": "0.0042", "base_at": "2026-10-08T13:30:00Z"}}}
    (tmp_path / "report.json").write_text(json.dumps(fresh))
    config = {"out": str(out), "radar": str(tmp_path / "radar.json"), "trader_root": str(runtime),
              "book_journal": str(tmp_path / "journal.jsonl"), "paper_report": str(tmp_path / "report.json"),
              "calendar": str(tmp_path / "calendar.json"), "book_watchlist": str(tmp_path / "watchlist.json")}
    assert refresh(config, at=NOW, read_units=False)["state"] == "EXPORTED"
    home = json.loads((out / "radar-home.json").read_text())
    assert [e["title"] for e in home["calendar"]][:2] == ["Today, time unknown", "Résultats JPMorgan"]
    assert home["ai_today"] == {"run_id": None, "at": None, "session": None, "lines": []}
    assert home["system"]["level"] in ("green", "orange", "red")
    paper = json.loads((out / "paper.json").read_text())
    assert paper["benchmarks"]["SPY"]["return_since_paper_start"] == 0.0042
    assert paper["accounts"][-1]["last_equity"] == 107000.0
