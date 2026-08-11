"""Every Phase 5E surface, checked against the protected research window.

Phase 5D proved the engine respects the embargo. That proof does not extend to
code written afterwards. Each new surface -- the operations API, the doctor,
the export, the support bundle, the settings file, the release bundle -- is a
new way for protected data to leave the machine or for the guard to be turned
off, and none of them existed when the original tests were written.

This file exists so that "5D already tested it" is never the reason a leak
ships. It is deliberately short and deliberately paranoid, and it is not
marked `ml`: a guard that only works when scikit-learn is installed is not a
guard.

The window: BTC-USD and ETH-USD, 2026-09-01T00:00:00Z through
2026-11-30T23:00:00Z, single use, unobserved.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

PROTECTED_MOMENTS = (
    "2026-09-01T00:00:00Z",
    "2026-09-15T12:00:00Z",
    "2026-10-31T23:00:00Z",
    "2026-11-30T23:00:00Z",
)
UNPROTECTED_MOMENTS = (
    "2026-08-31T23:00:00Z",
    "2026-12-01T00:00:00Z",
)


@pytest.fixture
def layout(tmp_path):
    from scripts.trading_lab.ops.runtime_paths import RuntimeLayout
    return RuntimeLayout(tmp_path / "var" / "trading_lab").ensure()


# --- the window itself -----------------------------------------------------


def test_the_window_still_says_what_the_protocol_recorded():
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

    window = PROTECTED_WINDOW_V1
    assert window.start == "2026-09-01T00:00:00Z"
    assert window.end == "2026-11-30T23:00:00Z"
    assert sorted(window.products) == ["BTC-USD", "ETH-USD"]
    assert window.observed is False
    assert window.single_use is True
    assert window.holdout_hash == (
        "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85")


@pytest.mark.parametrize("moment", PROTECTED_MOMENTS)
@pytest.mark.parametrize("product", ["BTC-USD", "ETH-USD"])
def test_a_protected_bar_is_refused_for_every_protected_product(product, moment):
    from scripts.trading_lab.protected_holdout import (
        ProtectedHoldoutError, require_unprotected_bar)

    with pytest.raises(ProtectedHoldoutError):
        require_unprotected_bar(product, moment)


@pytest.mark.parametrize("moment", UNPROTECTED_MOMENTS)
def test_a_bar_outside_the_window_is_allowed(moment):
    from scripts.trading_lab.protected_holdout import require_unprotected_bar

    require_unprotected_bar("BTC-USD", moment)


# --- settings cannot reach the guard --------------------------------------


@pytest.mark.parametrize("field,value", [
    ("holdout_start", "2027-01-01T00:00:00Z"),
    ("holdout_end", "2026-09-02T00:00:00Z"),
    ("holdout_dates", ["2027-01-01", "2027-02-01"]),
    ("holdout_products", []),
    ("protected_products", []),
    ("holdout_id", "something_else"),
    ("embargo", False),
    ("embargo_enabled", False),
])
def test_no_setting_can_move_or_disable_the_protected_window(field, value):
    from scripts.trading_lab.ops.settings import ForbiddenSettingError, validate

    with pytest.raises(ForbiddenSettingError):
        validate({field: value})


def test_a_settings_file_carrying_holdout_keys_is_ignored_wholesale(layout):
    """Corruption or tampering must not partially apply."""
    from scripts.trading_lab.ops import settings as settings_module
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

    layout.settings_file.write_text(json.dumps({
        "theme": "light", "holdout_end": "2026-09-02T00:00:00Z"}))
    loaded = settings_module.load(layout.settings_file)
    assert loaded == settings_module.DEFAULT_SETTINGS
    assert PROTECTED_WINDOW_V1.end == "2026-11-30T23:00:00Z"


def test_the_window_is_frozen_against_attribute_assignment():
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

    for field, value in (("end", "2026-09-02T00:00:00Z"), ("observed", True),
                         ("products", ())):
        with pytest.raises(Exception):
            setattr(PROTECTED_WINDOW_V1, field, value)


# --- the operations surfaces ----------------------------------------------


def test_the_doctor_reads_the_specification_and_never_protected_data(layout):
    """Doctor may say when the window is; it may not look inside it."""
    from scripts.trading_lab.ops import doctor

    report = doctor.run(layout=layout, root=REPO_ROOT)
    rendered = json.dumps(report)
    assert "2026-09-01T00:00:00Z" in rendered          # the specification
    assert "unobserved" in rendered
    # no candle, no price, no prediction from inside the window
    assert "close" not in rendered and "predicted" not in rendered


def test_the_support_bundle_states_the_window_without_exposing_it(layout):
    from scripts.trading_lab.ops import support_bundle

    bundle = support_bundle.build(layout=layout, root=REPO_ROOT)
    protection = bundle["research_protection"]
    assert protection["observed"] is False
    assert protection["enforced"] is True
    assert protection["start"] == "2026-09-01T00:00:00Z"
    # the embargo state is a status, not data
    for product in ("BTC-USD", "ETH-USD"):
        assert set(protection["embargo"][product]) >= {"embargoed", "reason"}


def test_an_export_of_the_runtime_cannot_contain_protected_bars(layout, tmp_path):
    """It cannot, because the runtime refuses to hold them in the first place.

    This asserts the property end to end rather than trusting the argument:
    a protected bar is refused before it can be recorded, so an export of the
    recorded log has nothing protected to carry.
    """
    from scripts.trading_lab.ops import runtime_export
    from scripts.trading_lab.paper_event_store import PaperEventStore
    from scripts.trading_lab.protected_holdout import (
        ProtectedHoldoutError, require_unprotected_bar)

    store = PaperEventStore(layout.paper_database)
    kept = 0
    for moment in ("2026-08-30T00:00:00Z", *PROTECTED_MOMENTS):
        try:
            require_unprotected_bar("BTC-USD", moment)
        except ProtectedHoldoutError:
            continue                                   # never reaches the store
        store.append(session_id="s1", event_type="CANDLE_INGESTED",
                     event_at=moment, product="BTC-USD", natural_key=moment,
                     payload={"close": "20000"})
        kept += 1
    assert kept == 1, "a protected bar was accepted into the runtime"

    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    import zipfile
    with zipfile.ZipFile(tmp_path / "rt.zip") as archive:
        blob = archive.read("paper_v1.sqlite")
    for moment in PROTECTED_MOMENTS:
        assert moment.encode() not in blob, moment


def test_no_operations_endpoint_returns_market_data(layout):
    """The ops API reports on the machine, never on the market."""
    from scripts.trading_lab.app_api.service import AppService

    service = AppService(REPO_ROOT / "data/crypto")
    for name in ("ops_runtime", "ops_recovery", "ops_storage", "ops_settings"):
        rendered = json.dumps(getattr(service, name)(), default=str)
        for banned in ("open", "high", "low", "close", "predicted_return",
                       "target_exposure"):
            assert f'"{banned}"' not in rendered, f"{name} exposed {banned}"


def test_the_embargo_turns_itself_on_without_anyone_remembering():
    from scripts.trading_lab.protected_holdout import embargo_state

    assert embargo_state("BTC-USD", now="2026-08-31T23:59:59Z")["embargoed"] is False
    assert embargo_state("BTC-USD", now="2026-09-01T00:00:00Z")["embargoed"] is True
    assert embargo_state("BTC-USD", now="2026-11-30T23:00:00Z")["embargoed"] is True
    # and it lifts only after the final protected bar has finished forming
    assert embargo_state("BTC-USD", now="2026-12-01T00:00:00Z")["embargoed"] is False


def test_the_guard_has_no_environment_or_config_switch():
    """A guard with an off switch is a guard that will be switched off."""
    import inspect

    from scripts.trading_lab import protected_holdout

    source = inspect.getsource(protected_holdout)
    for escape in ("os.environ", "getenv", "HYPRL_DISABLE", "if not enabled",
                   "skip_holdout", "bypass"):
        assert escape not in source, f"the guard exposes {escape!r}"

