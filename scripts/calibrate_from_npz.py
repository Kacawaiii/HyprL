#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate probability predictions from an NPZ holdout (y, p)."""

import argparse

import joblib
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(f"matplotlib required for plotting: {exc}") from exc

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs: list[float] = []
    ys: list[float] = []
    ns: list[int] = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    diff = p - y
    return float(np.mean(diff * diff))


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    loss = -y * np.log(p) - (1.0 - y) * np.log(1.0 - p)
    return float(np.mean(loss))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate holdout predictions from NPZ (y,p).")
    parser.add_argument("--npz", required=True, help="NPZ with arrays y (0/1) and p (proba).")
    parser.add_argument("--method", choices=["platt", "isotonic"], required=True)
    parser.add_argument("--out", required=True, help="Output calibrator joblib path.")
    parser.add_argument("--plot", required=True, help="Output PNG reliability plot.")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.npz)
    if "y" not in data or "p" not in data:
        raise SystemExit("NPZ must contain arrays 'y' and 'p'.")
    y = data["y"].astype(int).ravel()
    p = data["p"].astype(float).ravel()

    if args.method == "platt":
        X = _logit(p).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=200)
        lr.fit(X, y)
        p_cal = lr.predict_proba(X)[:, 1]
        calibrator = {"method": "platt", "model": lr}
    else:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        p_cal = ir.transform(p)
        calibrator = {"method": "isotonic", "model": ir}

    x0, y0, _ = _reliability_curve(y, p, bins=args.bins)
    x1, y1, _ = _reliability_curve(y, p_cal, bins=args.bins)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.scatter(x0, y0, label="raw")
    plt.scatter(x1, y1, label=args.method)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"Reliability ({args.method})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)

    joblib.dump(calibrator, args.out)
    print(f"[OK] calibrator={args.out}")
    print(f"[OK] plot={args.plot}")
    print(f"[METRICS] brier_raw={_brier(y, p):.4f} brier_cal={_brier(y, p_cal):.4f}")
    print(f"[METRICS] logloss_raw={_log_loss(y, p):.4f} logloss_cal={_log_loss(y, p_cal):.4f}")


if __name__ == "__main__":
    main()
