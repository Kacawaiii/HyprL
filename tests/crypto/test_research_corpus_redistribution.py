"""The redistribution boundary, re-proved with the corpus actually installed.

The earlier release tests ran on a machine where the corpus happened to be
absent from the paths a bundle copies. That is a weaker statement than it
looks: it shows the bundle did not contain the data, not that it would refuse
to. These tests exist so the guarantee is checked in the condition that makes
it interesting -- a real, verified, present corpus on the same machine that is
building a release, a frontend bundle and a support archive.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

from scripts.trading_lab.ops import release as release_module

REPO = pathlib.Path(__file__).resolve().parents[2]
REAL_STORE = REPO / "var/trading_lab/research/yahoo_us_equity_daily_v2"

needs_corpus = pytest.mark.skipif(
    not (REAL_STORE / "manifest.local.json").is_file(),
    reason="the local research corpus is not installed on this machine")

# Substrings that only occur in the corpus itself, never in a hash or an id.
PRICE_MARKERS = ("224.3699951171875", "62501000")


def _bundle_files(directory: pathlib.Path):
    return [path for path in directory.rglob("*") if path.is_file()]


@needs_corpus
def test_a_release_built_with_the_corpus_installed_contains_none_of_it(tmp_path):
    output = tmp_path / "release"
    release_module.build_release(root=REPO, output=output)

    files = _bundle_files(output)
    assert files, "the release produced no files"
    assert not [path for path in files if "yahoo_us_equity" in str(path)]
    assert not [path for path in files
                if path.relative_to(output).parts[:1] == ("var",)]
    assert not [path for path in files if path.suffix == ".jsonl"
                and "xnas" in path.name]


@needs_corpus
def test_no_release_file_contains_a_corpus_price(tmp_path):
    output = tmp_path / "release"
    release_module.build_release(root=REPO, output=output)
    for path in _bundle_files(output):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:                     # pragma: no cover - defensive
            continue
        for marker in PRICE_MARKERS:
            assert marker not in text, f"{path} carries corpus data"


@needs_corpus
def test_the_release_guard_still_refuses_a_planted_restricted_dataset(tmp_path):
    """The guard is the reason the above passes, so prove it still bites."""
    staged = tmp_path / "staged"
    (staged / "data").mkdir(parents=True)
    (staged / "data" / "manifest.json").write_text(
        json.dumps({"redistribution_permitted": False, "corpus_id": "x"}))
    with pytest.raises(release_module.ReleaseError):
        release_module.assert_no_restricted_data(staged)


@needs_corpus
def test_copying_the_local_manifest_into_a_bundle_is_caught(tmp_path):
    """The local manifest is its own tripwire.

    It keeps the unrenamed `redistribution_permitted: false`, so if a future
    change ever copies the store into an assembled bundle, the guard fires
    rather than shipping it.
    """
    staged = tmp_path / "staged"
    (staged / "research").mkdir(parents=True)
    shutil.copy(REAL_STORE / "manifest.local.json",
                staged / "research" / "manifest.local.json")
    with pytest.raises(release_module.ReleaseError):
        release_module.assert_no_restricted_data(staged)


@needs_corpus
def test_the_frontend_build_embeds_no_canonical_rows():
    """§29: the browser fetches a window at runtime; it ships with none."""
    dist = REPO / "apps/web/dist"
    if not dist.is_dir():
        pytest.skip("no frontend build present")
    for path in dist.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in PRICE_MARKERS:
            assert marker not in text, f"{path} embeds corpus data"
        assert "xnas_AAPL.jsonl" not in text


@needs_corpus
def test_the_support_bundle_reports_the_corpus_without_carrying_it():
    from scripts.trading_lab.ops import support_bundle
    from scripts.trading_lab.ops.runtime_paths import RuntimeLayout

    layout = RuntimeLayout(REPO / "var/trading_lab")
    bundle = support_bundle.build(layout=layout, root=REPO)
    section = bundle["research_corpus"]

    assert section["installed"] is True
    assert section["status"] == "AVAILABLE"
    assert section["fingerprint_match"] is True
    assert section["corpus_content_hash"] == (
        "64ac4485fc2541e671b899928f804bf3a7ceed5d380cb266145904446f71e024")
    assert section["redistribution_permitted"] is False
    assert section["source_data_included"] is False

    text = json.dumps(bundle)
    for marker in PRICE_MARKERS:
        assert marker not in text
    assert "bar_open_at" not in text


@needs_corpus
def test_git_still_tracks_no_corpus_file():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
        check=True).stdout.splitlines()
    assert not [path for path in tracked if "yahoo_us_equity" in path]
    assert not [path for path in tracked if path.startswith("var/")]


@needs_corpus
def test_the_store_is_ignored_by_git():
    result = subprocess.run(
        ["git", "check-ignore", str(REAL_STORE.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, "the local corpus store is not gitignored"
