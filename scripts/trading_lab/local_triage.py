"""Offline teacher distillation experiment. No network, trader imports or deployment.

Input JSONL is private: one row per teacher/session/candidate (or headline), with
session, teacher, text, label (0/1, or null for unknown), and optional payload_bytes.
Only the predecision headline text belongs in text. Never use outcomes or reasons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import time


def fingerprint(text):
    normalized = " ".join(re.findall(r"\w+", text.casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_rows(rows):
    """Keep the prescribed split; unknown labels never become negatives."""
    train, test = [], []
    for row in rows:
        day = date.fromisoformat(row["session"])
        if row.get("label") is None:
            continue
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("label must be 0, 1 or null")
        if not isinstance(row["text"], str) or not row["text"].strip():
            continue
        if date(2025, 1, 1) <= day < date(2026, 9, 1):
            train.append(row)
        elif date(2026, 9, 1) <= day < date(2026, 11, 1):
            test.append(row)
    return train, test


def select_fraction(rows, scores, fraction):
    """Rank within a complete decision session; deterministic input-order ties."""
    if len(rows) != len(scores) or not 0 < fraction <= 1:
        raise ValueError("invalid ranking inputs")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["session"]].append(index)
    selected = set()
    for indices in groups.values():
        count = max(1, math.ceil(len(indices) * fraction))
        selected.update(sorted(indices, key=lambda i: (-scores[i], i))[:count])
    return selected


def metrics(rows, selected):
    tp = sum(rows[i]["label"] for i in selected)
    positives = sum(r["label"] for r in rows)
    total_bytes = sum(r.get("payload_bytes", len(r["text"].encode())) for r in rows)
    kept_bytes = sum(rows[i].get("payload_bytes", len(rows[i]["text"].encode())) for i in selected)
    return {"rows": len(rows), "positive": positives, "sent": len(selected),
            "tp": tp, "fp": len(selected) - tp, "fn": positives - tp,
            "precision": tp / len(selected) if selected else None,
            "recall": tp / positives if positives else None,
            "send_share": len(selected) / len(rows) if rows else None,
            "total_payload_bytes": total_bytes, "kept_payload_bytes": kept_bytes,
            "payload_byte_saving": 1 - kept_bytes / total_bytes if total_bytes else None}


def budget_saving(total_new_tokens, eligible_share, payload_saving):
    """Scenario only: fixed prompt, output and other work remain in the denominator."""
    if total_new_tokens < 0 or not 0 <= eligible_share <= 1 or not 0 <= payload_saving <= 1:
        raise ValueError("invalid token scenario")
    return total_new_tokens * eligible_share * payload_saving


def fit_teacher(train):
    # Lazy import permits stdlib-only accounting and dataset validation.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    import numpy as np

    if len({row["label"] for row in train}) < 2:
        raise ValueError("training requires both teacher classes")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000,
                        sublinear_tf=True, dtype=np.float32),
        LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                           max_iter=300, random_state=17),
    )
    model.fit([row["text"] for row in train], [row["label"] for row in train])
    return model


def session_intervals(rows, selected, draws=1000):
    """Percentile intervals resample whole UTC sessions, preserving correlation."""
    groups = defaultdict(lambda: [0, 0, 0])
    for i, row in enumerate(rows):
        group = groups[row["session"]]
        group[0] += int(i in selected) * row["label"]
        group[1] += int(i in selected)
        group[2] += row["label"]
    values = list(groups.values())
    rng = random.Random(17)
    estimates = {"precision": [], "recall": []}
    for _ in range(draws):
        sample = rng.choices(values, k=len(values))
        tp, sent, positive = (sum(x[k] for x in sample) for k in range(3))
        if sent:
            estimates["precision"].append(tp / sent)
        if positive:
            estimates["recall"].append(tp / positive)
    return {name: [sorted(v)[int((len(v) - 1) * p)] for p in (.025, .975)] if v else None
            for name, v in estimates.items()}


def evaluate(rows, models_dir=None):
    """One prespecified model per teacher. Never tune on September/October."""
    result = {}
    for teacher in sorted({r["teacher"] for r in rows}):
        subset = [r for r in rows if r["teacher"] == teacher]
        train, test = split_rows(subset)
        summary = {"train_rows": len(train), "test_rows": len(test),
                   "train_positive": sum(r["label"] for r in train),
                   "unknown_rows": sum(r.get("label") is None for r in subset),
                   "train_sessions": len({r["session"] for r in train}),
                   "test_sessions": len({r["session"] for r in test}),
                   "train_months": dict(sorted(Counter(r["session"][:7] for r in train).items())),
                   "test_months": dict(sorted(Counter(r["session"][:7] for r in test).items()))}
        result[teacher] = summary
        if not train or not test or len({r["label"] for r in train}) < 2:
            summary["status"] = "BLOCKED_INSUFFICIENT_LABELLED_DATA"
            continue
        start = time.monotonic()
        model = fit_teacher(train)
        summary["fit_seconds"] = time.monotonic() - start
        if models_dir is not None:
            import joblib
            models_dir.mkdir(parents=True, exist_ok=True)
            # Teacher names are data, never usable as file paths.
            path = models_dir / (fingerprint(teacher)[:16] + ".joblib")
            if path.exists():
                raise ValueError("model artifact already exists")
            joblib.dump(model, path, compress=3)
            summary["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary["model_bytes"] = path.stat().st_size
        # Exclude even individual training headlines from the clean candidate
        # evaluation; an overlapping component could carry the teacher label.
        seen = {h for r in train for h in r.get("headline_fingerprints", [fingerprint(r["text"])])}
        clean = [r for r in test if not seen.intersection(
            r.get("headline_fingerprints", [fingerprint(r["text"])]))]
        summary["test_rows_with_train_overlap"] = len(test) - len(clean)
        summary["status"] = "EXPLORATORY"
        summary["evaluations"] = {}
        for name, sample in [("all", test), ("no_train_headline_overlap", clean)]:
            if not sample:
                summary["evaluations"][name] = {"rows": 0}
                continue
            scores = model.predict_proba([r["text"] for r in sample])[:, 1]
            summary["evaluations"][name] = {}
            for fraction in (0.1, 0.2):
                selected = select_fraction(sample, scores, fraction)
                measured = metrics(sample, selected)
                measured["session_bootstrap_95pct"] = session_intervals(sample, selected)
                measured["random_expected_precision"] = sum(r["label"] for r in sample) / len(sample)
                measured["random_expected_recall"] = len(selected) / len(sample)
                by_month = {}
                for month in sorted({r["session"][:7] for r in sample}):
                    indices = [i for i, r in enumerate(sample) if r["session"].startswith(month)]
                    local = [sample[i] for i in indices]
                    by_month[month] = metrics(local, {j for j, i in enumerate(indices) if i in selected})
                measured["by_month"] = by_month
                summary["evaluations"][name][str(fraction)] = measured
        # A fixed cheap event-word comparator; no test-driven vocabulary choice.
        event_words = re.compile(r"earnings|revenue|merger|acquir|contract|fda|approval|offering|guidance|buyback|investigation|layoff", re.I)
        heuristic = [len(event_words.findall(r["text"])) for r in test]
        summary["keyword_baseline"] = {str(f): metrics(test, select_fraction(test, heuristic, f))
                                       for f in (.1, .2)}
        summary["vocabulary_size"] = len(model[0].vocabulary_)
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    args = parser.parse_args()
    # This host's trading jobs take precedence, including weekend UTC windows.
    now = datetime.now(timezone.utc)
    minute = now.hour * 60 + now.minute
    if 690 <= minute < 1035:
        parser.error("batch training is forbidden during 11:30–17:15 UTC")
    if args.output.exists():
        parser.error("output already exists; preserve prior experiment")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    with args.dataset.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report = {"schema": "local-triage-research-v1", "status": "EXPLORATORY",
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "teachers": evaluate(rows, args.models_dir)}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
