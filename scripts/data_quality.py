#!/usr/bin/env python3
"""
Data Quality Validation for Dukascopy M30 data.

Checks:
- Missing bars / gaps (weekends OK, intra-session gaps flagged)
- Outliers > 5σ
- Column integrity
- Descriptive statistics

Usage:
    python scripts/data_quality.py --dir data/dukascopy/
    python scripts/data_quality.py --file data/dukascopy/EURUSD_M30.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def detect_gaps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect gaps in M30 data. Weekends and holidays are expected."""
    df = df.sort_values("time").reset_index(drop=True)
    time_diff = df["time"].diff()

    # Expected gap: 30 min
    expected = pd.Timedelta(minutes=30)

    # Flag gaps > 30 min that aren't weekends
    gaps = []
    for i in range(1, len(df)):
        delta = time_diff.iloc[i]
        if delta > expected:
            t_prev = df["time"].iloc[i - 1]
            t_curr = df["time"].iloc[i]
            hours = delta.total_seconds() / 3600

            # Weekend gap: Friday evening -> Sunday evening / Monday
            is_weekend = (t_prev.dayofweek == 4 and t_curr.dayofweek == 0)
            # Holiday: gap > 2 days but starts/ends near weekend
            is_holiday = hours > 48 and t_prev.dayofweek >= 3

            gap_type = "weekend" if is_weekend else ("holiday" if is_holiday else "intra-session")

            gaps.append({
                "from": t_prev,
                "to": t_curr,
                "hours": hours,
                "type": gap_type,
            })

    return pd.DataFrame(gaps)


def detect_outliers(df: pd.DataFrame, sigma: float = 5.0) -> pd.DataFrame:
    """Detect price outliers > N sigma from rolling mean."""
    df = df.copy()
    # Use returns for outlier detection
    df["ret"] = df["close"].pct_change()

    # Rolling stats
    roll_mean = df["ret"].rolling(100, min_periods=20).mean()
    roll_std = df["ret"].rolling(100, min_periods=20).std()

    # Flag outliers
    z_score = (df["ret"] - roll_mean) / roll_std.replace(0, np.nan)
    outlier_mask = z_score.abs() > sigma

    outliers = df[outlier_mask][["time", "open", "high", "low", "close", "ret"]].copy()
    outliers["z_score"] = z_score[outlier_mask]
    return outliers


def clean_outliers(df: pd.DataFrame, sigma: float = 5.0) -> pd.DataFrame:
    """Remove outlier bars (> 5σ returns)."""
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    roll_mean = df["ret"].rolling(100, min_periods=20).mean()
    roll_std = df["ret"].rolling(100, min_periods=20).std()
    z_score = (df["ret"] - roll_mean) / roll_std.replace(0, np.nan)

    mask = z_score.abs() <= sigma
    # Keep first bar (no return) and non-outlier bars
    mask.iloc[0] = True
    # Also keep bars where z_score is NaN (not enough data yet)
    mask = mask | z_score.isna()

    cleaned = df[mask].drop(columns=["ret"]).reset_index(drop=True)
    return cleaned


def validate_columns(df: pd.DataFrame) -> list:
    """Check required columns exist and have correct types."""
    required = ["time", "open", "high", "low", "close", "volume"]
    issues = []

    for col in required:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif col == "time":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(f"Column 'time' is not datetime: {df[col].dtype}")
        else:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' is not numeric: {df[col].dtype}")

    # Check OHLC consistency
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl > 0:
            issues.append(f"{bad_hl} bars with high < low")

        bad_oh = (df["high"] < df["open"]).sum()
        if bad_oh > 0:
            issues.append(f"{bad_oh} bars with high < open")

        bad_ol = (df["low"] > df["open"]).sum()
        if bad_ol > 0:
            issues.append(f"{bad_ol} bars with low > open")

    # Check for zero/negative prices
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            zeros = (df[col] <= 0).sum()
            if zeros > 0:
                issues.append(f"{zeros} zero/negative values in '{col}'")

    return issues


def generate_report(df: pd.DataFrame, symbol: str, clean: bool = True) -> dict:
    """Generate quality report for one symbol."""
    report = {"symbol": symbol}

    # Basic info
    report["total_bars"] = len(df)
    report["date_range"] = f"{df['time'].min()} -> {df['time'].max()}"
    report["date_start"] = df["time"].min()
    report["date_end"] = df["time"].max()
    days = (df["time"].max() - df["time"].min()).days
    report["calendar_days"] = days
    report["years"] = days / 365.25

    # Expected bars (roughly 48 per business day × 252 days/year × years)
    expected_bars = int(report["years"] * 252 * 48)
    report["expected_bars_approx"] = expected_bars
    report["bar_coverage_pct"] = len(df) / expected_bars * 100 if expected_bars > 0 else 0

    # Column validation
    col_issues = validate_columns(df)
    report["column_issues"] = col_issues

    # Gaps
    gaps_df = detect_gaps(df, symbol)
    report["total_gaps"] = len(gaps_df)
    intra_gaps = gaps_df[gaps_df["type"] == "intra-session"]
    report["intra_session_gaps"] = len(intra_gaps)
    report["weekend_gaps"] = len(gaps_df[gaps_df["type"] == "weekend"])

    # Outliers
    outliers = detect_outliers(df)
    report["outliers_5sigma"] = len(outliers)

    # Descriptive stats
    report["stats"] = {
        "close_mean": df["close"].mean(),
        "close_std": df["close"].std(),
        "close_min": df["close"].min(),
        "close_max": df["close"].max(),
        "volume_mean": df["volume"].mean(),
        "volume_median": df["volume"].median(),
        "avg_range_pct": ((df["high"] - df["low"]) / df["close"]).mean() * 100,
    }

    # Clean if requested
    if clean and len(outliers) > 0:
        df_clean = clean_outliers(df)
        report["bars_after_clean"] = len(df_clean)
        report["bars_removed"] = len(df) - len(df_clean)
    else:
        report["bars_after_clean"] = len(df)
        report["bars_removed"] = 0

    return report


def print_report(report: dict):
    """Print formatted quality report."""
    s = report
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT: {s['symbol']}")
    print(f"{'='*60}")
    print(f"  Date range:     {s['date_range']}")
    print(f"  Calendar days:  {s['calendar_days']} ({s['years']:.1f} years)")
    print(f"  Total bars:     {s['total_bars']:,}")
    print(f"  Expected bars:  ~{s['expected_bars_approx']:,}")
    print(f"  Coverage:       {s['bar_coverage_pct']:.1f}%")
    print()

    if s["column_issues"]:
        print("  COLUMN ISSUES:")
        for issue in s["column_issues"]:
            print(f"    WARNING: {issue}")
    else:
        print("  Columns: OK")

    print(f"\n  Gaps:")
    print(f"    Total:          {s['total_gaps']}")
    print(f"    Weekend:        {s['weekend_gaps']}")
    print(f"    Intra-session:  {s['intra_session_gaps']}")

    print(f"\n  Outliers (>5σ):   {s['outliers_5sigma']}")
    if s["bars_removed"] > 0:
        print(f"  Bars removed:     {s['bars_removed']}")
        print(f"  Bars after clean: {s['bars_after_clean']:,}")

    stats = s["stats"]
    print(f"\n  Price stats:")
    print(f"    Mean close:     {stats['close_mean']:.5f}")
    print(f"    Std close:      {stats['close_std']:.5f}")
    print(f"    Min close:      {stats['close_min']:.5f}")
    print(f"    Max close:      {stats['close_max']:.5f}")
    print(f"    Avg range %:    {stats['avg_range_pct']:.4f}%")
    print(f"    Avg volume:     {stats['volume_mean']:.1f}")

    # Verdict
    critical = len(s["column_issues"]) > 0 or s["total_bars"] < 10000
    warning = s["intra_session_gaps"] > 50 or s["outliers_5sigma"] > 100
    print(f"\n  Verdict: ", end="")
    if critical:
        print("CRITICAL ISSUES - data may be unusable")
    elif warning:
        print("WARNINGS - review gaps/outliers")
    else:
        print("PASS - data quality OK")


def process_file(path: Path, clean_and_save: bool = False) -> dict:
    """Process a single parquet file."""
    symbol = path.stem.replace("_M30", "")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    report = generate_report(df, symbol)
    print_report(report)

    if clean_and_save:
        df_clean = clean_outliers(df)
        df_clean.to_parquet(path, index=False)
        print(f"  Saved cleaned data: {path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Data Quality Validation")
    parser.add_argument("--dir", type=str, default=None, help="Directory with parquet files")
    parser.add_argument("--file", type=str, default=None, help="Single parquet file")
    parser.add_argument("--clean", action="store_true", help="Clean outliers and overwrite files")
    args = parser.parse_args()

    if args.file:
        process_file(Path(args.file), args.clean)
    elif args.dir:
        dir_path = Path(args.dir)
        files = sorted(dir_path.glob("*_M30.parquet"))
        if not files:
            print(f"No *_M30.parquet files found in {dir_path}")
            return

        print(f"\nFound {len(files)} data files in {dir_path}")
        reports = []
        for f in files:
            r = process_file(f, args.clean)
            reports.append(r)

        # Summary table
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Symbol':<10} {'Bars':>10} {'Years':>6} {'Gaps':>6} {'IntraGap':>8} {'Outliers':>8} {'Status':<10}")
        print("-" * 70)
        for r in reports:
            critical = len(r["column_issues"]) > 0 or r["total_bars"] < 10000
            warning = r["intra_session_gaps"] > 50 or r["outliers_5sigma"] > 100
            status = "CRITICAL" if critical else ("WARNING" if warning else "OK")
            print(
                f"{r['symbol']:<10} {r['total_bars']:>10,} {r['years']:>6.1f} "
                f"{r['total_gaps']:>6} {r['intra_session_gaps']:>8} "
                f"{r['outliers_5sigma']:>8} {status:<10}"
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
