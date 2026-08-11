"""Phase 5E: the local release bundle and the frontend size budgets.

Not marked `ml`: a release must be buildable and verifiable without the
optional model stack.

The budgets here are deliberately generous. Their job is not to police a few
kilobytes -- it is to make a future dependency that adds two megabytes fail
loudly instead of arriving unnoticed in a release.
"""

from __future__ import annotations

import gzip
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DIST = REPO_ROOT / "apps/web/dist"

# Generous, and real. The app chunk is currently ~5 kB gzip and the largest
# route ~3.3 kB; a page that reaches 50 kB has acquired a dependency worth
# arguing about.
MAX_ENTRY_GZIP_BYTES = 50 * 1024
MAX_ROUTE_GZIP_BYTES = 50 * 1024
# React and the router. Replacing them is a decision, not an accident.
MAX_VENDOR_GZIP_BYTES = 120 * 1024
MAX_TOTAL_FRONTEND_BYTES = 2 * 1024 * 1024


def _gzip_size(path: pathlib.Path) -> int:
    return len(gzip.compress(path.read_bytes(), 9))


needs_build = pytest.mark.skipif(
    not (DIST / "index.html").is_file(),
    reason="no frontend build; run ./scripts/hyprl.sh build")


# --- frontend budgets ------------------------------------------------------


@needs_build
def test_the_entry_chunk_stays_small():
    entry = [path for path in (DIST / "assets").glob("index-*.js")]
    assert entry, "no entry chunk in the build"
    size = _gzip_size(entry[0])
    assert size < MAX_ENTRY_GZIP_BYTES, f"entry chunk is {size} bytes gzipped"


@needs_build
def test_no_single_route_chunk_dominates_the_bundle():
    oversized = {}
    for path in (DIST / "assets").glob("*Page-*.js"):
        size = _gzip_size(path)
        if size >= MAX_ROUTE_GZIP_BYTES:
            oversized[path.name] = size
    assert not oversized, f"route chunks over budget: {oversized}"


@needs_build
def test_every_route_is_still_split_into_its_own_chunk():
    """A page that stops being lazy is invisible until the entry chunk grows."""
    names = {path.name.split('-')[0] for path in (DIST / "assets").glob("*.js")}
    for page in ("MarketsPage", "SignalsPage", "RiskPage", "PaperPage",
                 "BacktestsPage", "ResearchPage", "SystemPage", "SettingsPage"):
        assert page in names, f"{page} is no longer a separate chunk"


@needs_build
def test_the_vendor_chunk_is_bounded():
    vendor = [path for path in (DIST / "assets").glob("react-*.js")]
    assert vendor, "no vendor chunk"
    size = _gzip_size(vendor[0])
    assert size < MAX_VENDOR_GZIP_BYTES, f"vendor chunk is {size} bytes gzipped"


@needs_build
def test_the_whole_frontend_build_is_bounded():
    total = sum(path.stat().st_size for path in DIST.rglob("*") if path.is_file())
    assert total < MAX_TOTAL_FRONTEND_BYTES, f"frontend build is {total} bytes"


@needs_build
def test_the_build_references_no_remote_origin():
    """Offline by construction: no CDN, no font host, no analytics beacon."""
    for path in DIST.rglob("*"):
        if not path.is_file() or path.suffix not in (".html", ".js", ".css"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for origin in ("http://", "https://"):
            for index in range(len(text)):
                found = text.find(origin, index)
                if found < 0:
                    break
                tail = text[found:found + 60]
                # localhost references and schema/spec URLs in comments are fine
                if any(allowed in tail for allowed in
                       ("127.0.0.1", "localhost", "www.w3.org", "://schema",
                        "reactjs.org", "react.dev")):
                    index = found + 1
                    continue
                raise AssertionError(f"{path.name} references {tail!r}")
            break


# --- release bundle --------------------------------------------------------


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    from scripts.trading_lab.ops import release as release_module

    if not (DIST / "index.html").is_file():
        pytest.skip("no frontend build")
    output = tmp_path_factory.mktemp("release") / "hyprl-local"
    report = release_module.build_release(
        root=REPO_ROOT, output=output, include_research_data=False)
    return output, report


def test_a_release_carries_a_manifest_and_checksums(release):
    output, report = release
    assert (output / "manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert (output / "README-RUN.txt").is_file()
    assert (output / "hyprl-run.sh").is_file()
    assert (output / "frontend" / "index.html").is_file()
    assert report["files"] > 0


def test_a_release_verifies_against_itself(release):
    from scripts.trading_lab.ops import release as release_module

    output, _ = release
    report = release_module.verify_release(output)
    assert report["ok"] is True, report
    assert all(report["checks"].values())


def test_a_modified_file_breaks_verification(release, tmp_path):
    import shutil

    from scripts.trading_lab.ops import release as release_module

    output, _ = release
    copy = tmp_path / "tampered"
    shutil.copytree(output, copy)
    target = copy / "frontend" / "index.html"
    target.write_text(target.read_text() + "<!-- changed -->")
    report = release_module.verify_release(copy)
    assert report["ok"] is False
    assert report["checks"]["sha256"] is False


def test_a_release_identifies_its_commit(release):
    """A worktree keeps .git as a file, and a naive reader returns nothing."""
    output, report = release
    commit = report["git_commit"]
    assert commit and len(commit) == 40, f"no commit recorded: {commit!r}"
    assert all(character in "0123456789abcdef" for character in commit)


def test_two_builds_of_the_same_tree_share_a_content_hash(tmp_path):
    from scripts.trading_lab.ops import release as release_module

    if not (DIST / "index.html").is_file():
        pytest.skip("no frontend build")
    first = release_module.build_release(root=REPO_ROOT, output=tmp_path / "a",
                                         include_research_data=False)
    second = release_module.build_release(root=REPO_ROOT, output=tmp_path / "b",
                                          include_research_data=False)
    assert first["content_hash"] == second["content_hash"]
    # and the bytes on disk match, manifest aside
    for path in sorted((tmp_path / "a").rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        twin = (tmp_path / "b") / path.relative_to(tmp_path / "a")
        assert twin.read_bytes() == path.read_bytes(), path.name


def test_the_bundle_excludes_what_the_app_does_not_run(release):
    output, _ = release
    names = {path.name for path in output.rglob("*")}
    parts = {part for path in output.rglob("*") for part in path.parts}
    assert "__pycache__" not in parts
    assert ".git" not in parts
    assert "node_modules" not in parts
    assert "tests" not in parts
    assert not [name for name in names if name.endswith(".pyc")]
    # nor any runtime state
    assert not [name for name in names if name.endswith(".sqlite")]


def test_the_bundle_ships_the_frozen_models_the_engine_needs(release):
    output, _ = release
    for product in ("BTC-USD", "ETH-USD"):
        assert (output / "data/models/paper_v1" / f"{product}.json").is_file()


def test_research_data_is_included_only_when_asked_for(release):
    output, report = release
    assert report["includes_research_data"] is False
    assert not (output / "data/crypto").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["content"]["includes_research_data"] is False


def test_the_manifest_records_every_frozen_hash(release):
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    output, _ = release
    specs = json.loads((output / "manifest.json").read_text())["content"]["specs"]
    assert specs["signal_spec_hash"] == SIGNAL_SPEC_V1.spec_hash
    assert specs["risk_spec_hash"] == RISK_SPEC_V1.risk_spec_hash
    assert len(specs["execution_spec_hash"]) == 64
    assert len(specs["holdout_hash"]) == 64


def test_the_manifest_declares_no_real_money_and_no_broker(release):
    output, _ = release
    safety = json.loads((output / "manifest.json").read_text())["content"]["trading_safety"]
    assert safety == {"real_money": False, "broker_connected": False,
                      "live_trading": False, "shadow_mode": True}


def test_the_release_readme_states_what_the_software_will_not_do(release):
    output, _ = release
    readme = (output / "README-RUN.txt").read_text()
    assert "NO real money" in readme
    assert "NO broker connection" in readme
    assert "NO exchange API key" in readme
    assert "127.0.0.1" in readme


def test_the_launcher_binds_loopback_and_carries_the_marker(release):
    output, _ = release
    launcher = (output / "hyprl-run.sh").read_text()
    assert "127.0.0.1" in launcher
    assert "0.0.0.0" not in launcher
    assert "hyprl-local-app" in launcher
    assert (output / "hyprl-run.sh").stat().st_mode & 0o111, "launcher not executable"


def test_there_is_no_auto_updater_anywhere_in_the_release(release):
    """`curl | bash` is a supply chain, not an update mechanism."""
    output, _ = release
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix not in (".sh", ".txt", ".py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "curl -sSL" not in text
        assert "| bash" not in text or "no `curl | bash`" in text.lower() \
            or "No `curl | bash`" in text


def test_a_release_needs_a_frontend_build(tmp_path):
    from scripts.trading_lab.ops import release as release_module

    with pytest.raises(release_module.ReleaseError):
        release_module.build_release(root=tmp_path, output=tmp_path / "out")


# --- the holdout wall extends to the release ------------------------------


def test_the_release_manifest_records_the_window_unobserved(release):
    output, _ = release
    protection = json.loads(
        (output / "manifest.json").read_text())["content"]["research_protection"]
    assert protection["observed"] is False
    assert protection["spent"] is False
    assert protection["enforced"] is True
    assert protection["start"] == "2026-09-01T00:00:00Z"
    assert protection["end"] == "2026-11-30T23:00:00Z"


def test_the_release_carries_no_protected_market_data(release):
    """Nothing in the bundle may contain a bar from inside the window."""
    output, _ = release
    protected = (b"2026-09-", b"2026-10-", b"2026-11-")
    for path in output.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.suffix not in (".json", ".csv", ".txt", ".jsonl"):
            continue
        blob = path.read_bytes()
        for needle in protected:
            assert needle not in blob, f"{path.name} mentions {needle.decode()}"


def test_the_bundled_guard_cannot_be_disabled_by_editing_config(release):
    output, _ = release
    guard = (output / "scripts/trading_lab/protected_holdout.py").read_text()
    assert "2026-09-01T00:00:00Z" in guard
    for escape in ("os.environ", "getenv", "HYPRL_DISABLE", "bypass"):
        assert escape not in guard
