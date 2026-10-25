"""Synthetic checks for temporal/label leakage and honest savings arithmetic."""
import pytest

from scripts.trading_lab.local_triage import (
    budget_saving, evaluate, fingerprint, fit_teacher, metrics, select_fraction, split_rows,
)


def row(day, text="Synthetic headline", label=1):
    return {"session": day, "teacher": "synthetic", "text": text, "label": label}


def test_calendar_boundaries_and_unknown_labels():
    rows = [row("2024-12-31"), row("2025-01-01"), row("2026-08-31"),
            row("2026-09-01"), row("2026-10-31"), row("2026-11-01"),
            row("2026-08-01", label=None)]
    train, test = split_rows(rows)
    assert [r["session"] for r in train] == ["2025-01-01", "2026-08-31"]
    assert [r["session"] for r in test] == ["2026-09-01", "2026-10-31"]


@pytest.mark.parametrize("label", [-1, 2, "1", True])
def test_reject_malformed_teacher_label(label):
    with pytest.raises(ValueError):
        split_rows([row("2026-07-01", label=label)])


def test_top_fraction_is_per_session_and_stable():
    rows = [row("2026-09-01") for _ in range(10)] + [row("2026-09-02") for _ in range(5)]
    assert select_fraction(rows, [1] * 15, .2) == {0, 1, 10}


def test_unselected_positives_count_as_false_negatives():
    rows = [row("2026-09-01", label=y) for y in [1, 0, 1, 0]]
    result = metrics(rows, {0, 1})
    assert result["precision"] == result["recall"] == .5
    assert result["fn"] == result["fp"] == 1


def test_savings_preserve_fixed_work():
    assert budget_saving(100000, .1, .8) == pytest.approx(8000)
    assert budget_saving(100000, 0, .8) == 0


def test_missing_test_is_blocked_not_fabricated():
    result = evaluate([row("2026-07-01", label=0), row("2026-07-02", label=1)])
    assert result["synthetic"]["status"] == "BLOCKED_INSUFFICIENT_LABELLED_DATA"


def test_train_headline_component_excluded_from_clean_evaluation():
    pytest.importorskip("sklearn")
    rows = [row("2026-07-01", "Synthetic merger approved", 1),
            row("2026-07-02", "Synthetic routine weekly update", 0)] * 3
    test = row("2026-09-01", "Synthetic merger approved plus new details", 1)
    test["headline_fingerprints"] = [fingerprint(rows[0]["text"]), fingerprint("new details")]
    result = evaluate(rows + [test])["synthetic"]
    assert result["test_rows_with_train_overlap"] == 1
    assert result["evaluations"]["no_train_headline_overlap"]["rows"] == 0


def test_test_only_words_do_not_enter_fitted_vocabulary():
    pytest.importorskip("sklearn")
    train = [row("2026-07-01", "Synthetic merger approved", 1),
             row("2026-07-02", "Synthetic routine update", 0)] * 3
    model = fit_teacher(train)
    model.predict_proba(["unseenholdoutsentinel"])
    assert "unseenholdoutsentinel" not in model[0].vocabulary_
