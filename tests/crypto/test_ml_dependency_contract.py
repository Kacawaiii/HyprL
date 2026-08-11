"""The Phase 1/2 core must not drag in the optional ML stack.

Deliberately NOT marked `ml`: this is the test that proves the core suite is
runnable without scikit-learn or xgboost, so it has to run in that suite.

"The libraries happen to be installed on this machine" is not the same claim
as "the core does not require them", so the probe runs in a fresh interpreter
with those imports actively blocked rather than merely absent.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CORE_MODULES = (
    "market_bar",
    "coinbase_candles",
    "market_data_store",
    "market_snapshots",
    "market_series",
    "market_indicators",
    "market_dataset",
    "walk_forward",
)

_BLOCKER = '''
import sys

class _Blocked:
    """Refuse the ML stack without touching the installed environment."""

    BANNED = {"sklearn", "xgboost"}

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BANNED:
            raise ImportError(f"blocked for this probe: {name}")
        return None

sys.meta_path.insert(0, _Blocked())
'''


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-"], input=_BLOCKER + script,
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300)


def test_the_blocker_itself_actually_blocks() -> None:
    """A probe that cannot fail proves nothing."""
    result = _run("import sklearn\n")
    assert result.returncode != 0
    assert "blocked for this probe" in result.stderr
    assert _run("import xgboost\n").returncode != 0


def test_the_core_imports_with_the_ml_stack_blocked() -> None:
    script = "\n".join(
        f"import scripts.trading_lab.{name}" for name in CORE_MODULES
    ) + '''
import sys
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"sklearn", "xgboost"})
assert not leaked, leaked
print("CORE-OK")
'''
    result = _run(script)
    assert result.returncode == 0, result.stderr
    assert "CORE-OK" in result.stdout


def test_the_core_can_still_do_real_work_without_the_ml_stack() -> None:
    """Importing is cheap. The core must actually run: snapshot to dataset."""
    result = _run('''
import json, pathlib, tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from scripts.trading_lab.market_data_store import MarketDataStore
from scripts.trading_lab.market_snapshots import _materialize_snapshot
from scripts.trading_lab.market_series import load_market_series
from scripts.trading_lab.market_dataset import (
    DatasetConfig, FeatureDefinition, LabelSpec, build_dataset)
from scripts.trading_lab.walk_forward import WalkForwardConfig, build_folds

GRID = datetime(2027, 3, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
iso = lambda t: t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
tmp = pathlib.Path(tempfile.mkdtemp())
store = MarketDataStore(tmp / "core.sqlite3")
opens = [GRID + HOUR * i for i in range(60)]
closes = [Decimal(100 + (i * 7) % 23) for i in range(60)]
rows = [[int(o.timestamp()), str(c - 2), str(c + 2), str(c), str(c), "1.0"]
        for o, c in zip(opens, closes)]
store.ingest_coinbase_response(json.dumps(rows, separators=(",", ":")).encode(),
    product_id="BTC-USD", timeframe="1h", available_at=iso(GRID + HOUR * 200),
    ingested_at=iso(GRID + HOUR * 200 + timedelta(seconds=1)))
connection = store._connect()
try:
    snapshot = _materialize_snapshot(connection, provider="coinbase_exchange_rest",
        product_id="BTC-USD", timeframe="1h", range_start=iso(opens[0]),
        range_end=iso(opens[-1] + HOUR), as_of=iso(GRID + HOUR * 210))
    series = load_market_series(connection, snapshot_id=snapshot.snapshot_id)
finally:
    connection.close()
dataset = build_dataset(series, config=DatasetConfig(
    features=(FeatureDefinition("sma5", "sma", (("period", 5),)),),
    label=LabelSpec(horizon=4)))
folds = build_folds(dataset, config=WalkForwardConfig(
    min_train_rows=12, validation_rows=8, test_rows=5, step_rows=5, purge_rows=4))
assert len(folds) >= 2, len(folds)
print("CORE-PIPELINE-OK", len(dataset.rows), len(folds))
''')
    assert result.returncode == 0, result.stderr
    assert "CORE-PIPELINE-OK" in result.stdout


def test_the_phase_three_layer_names_the_missing_extra_at_import_time() -> None:
    """Not a puzzle three frames into a fit: say which extra is missing, early."""
    result = _run("import scripts.trading_lab.models\n")
    assert result.returncode != 0
    assert "pip install hyprl[ml]" in result.stderr
    assert "scikit-learn and xgboost" in result.stderr
    assert "Phase 1 and Phase 2" in result.stderr
    # and it is an ImportError, not a NameError or an AttributeError later on
    assert "ImportError" in result.stderr
    assert "NameError" not in result.stderr and "AttributeError" not in result.stderr


def _ml_importers() -> list[str]:
    """Modules that actually IMPORT the ML stack, found by parsing, not grepping.

    A substring scan was the original check and it was wrong: the string
    "xgboost" also appears as a candidate identifier in ordinary code, which
    made an innocent module look coupled. Only real import statements count.
    """
    import ast

    found = []
    for path in sorted((REPO_ROOT / "scripts" / "trading_lab").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            roots = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".")[0]]
            if any(root in {"sklearn", "xgboost"} for root in roots):
                found.append(path.name)
                break
    return found


def test_exactly_one_trading_lab_module_imports_the_ml_stack() -> None:
    """The direct boundary is a single file, and that is what keeps the core clean."""
    assert _ml_importers() == ["models.py"], _ml_importers()


def test_modules_built_on_top_of_models_are_declared_ml_coupled() -> None:
    """Importing `models` is transitive ML coupling, and must be acknowledged.

    These modules do not import sklearn or xgboost themselves, but importing
    them pulls the stack in anyway, so their tests carry the `ml` marker. The
    list is explicit: a new module joining it is a deliberate decision, not an
    accident nobody noticed.
    """
    import ast

    transitive = []
    for path in sorted((REPO_ROOT / "scripts" / "trading_lab").glob("*.py")):
        if path.name == "models.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.endswith("trading_lab.models"):
                transitive.append(path.name)
                break
    assert transitive == ["paper_model.py", "real_benchmark.py",
                          "real_benchmark_v2.py", "run_real_benchmark.py"], transitive


@pytest.mark.parametrize("module", CORE_MODULES)
def test_no_core_module_mentions_the_ml_stack_at_all(module) -> None:
    source = (REPO_ROOT / "scripts" / "trading_lab" / f"{module}.py").read_text()
    assert "sklearn" not in source and "xgboost" not in source


PHASE_THREE_TEST_MODULES = (
    "test_models.py",
    "test_model_selection.py",
    "test_model_robustness.py",
    "test_phase3_closure.py",
)


def test_the_core_only_selection_excludes_every_phase_three_test() -> None:
    """Without this, dropping a marker would go unnoticed while [ml] is installed.

    On a machine that has scikit-learn, an unmarked Phase 3 test still passes
    in the core-only run -- the contract only breaks for someone who installed
    the core alone. So the contract is asserted on the collection itself.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "not ml",
         "tests/crypto"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600)
    assert result.returncode == 0, result.stderr
    collected = [line for line in result.stdout.splitlines() if "::" in line]
    assert collected, result.stdout
    for module in PHASE_THREE_TEST_MODULES:
        assert not any(module in line for line in collected), module
    # and this very file must stay in the core selection
    assert any("test_ml_dependency_contract.py" in line for line in collected)


SAFETY_CRITICAL_CORE = (
    "research_holdout",
    "protected_holdout",
    "paper_event_store",
    "live_market",
)

_GUARD_PROBE = """
from scripts.trading_lab.protected_holdout import (
    ProtectedHoldoutError, require_tradeable_now)
try:
    require_tradeable_now("BTC-USD", now="2026-10-01T00:00:00+00:00")
    raise SystemExit("the embargo did not fire")
except ProtectedHoldoutError:
    pass
import sys
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"sklearn", "xgboost"})
assert not leaked, leaked
print("GUARD-OK")
"""


def test_the_holdout_guard_works_without_the_ml_stack() -> None:
    """A safety check that cannot load in a core environment is not a safety check.

    The confirmatory holdout must be enforceable whether or not scikit-learn is
    installed, so the guard reads its window from a dependency-free module
    rather than from the benchmark contract that imports the model classes.
    """
    imports = "\n".join(f"import scripts.trading_lab.{name}"
                         for name in SAFETY_CRITICAL_CORE)
    result = _run(imports + _GUARD_PROBE)
    assert result.returncode == 0, result.stderr
    assert "GUARD-OK" in result.stdout


def test_the_holdout_window_has_exactly_one_definition() -> None:
    """Two copies of these dates would eventually disagree by one hour."""
    import importlib
    contract = importlib.import_module("scripts.trading_lab.research_holdout")
    benchmark = importlib.import_module("scripts.trading_lab.real_benchmark_v2")
    guard = importlib.import_module("scripts.trading_lab.protected_holdout")
    assert benchmark.CONFIRMATORY_HOLDOUT_V2 is contract.CONFIRMATORY_HOLDOUT_V2
    window = guard.PROTECTED_WINDOW_V1
    assert window.start == contract.CONFIRMATORY_HOLDOUT_V2["range_start"]
    assert window.end == contract.CONFIRMATORY_HOLDOUT_V2["range_end"]
    assert list(window.products) == contract.CONFIRMATORY_HOLDOUT_V2["products"]
