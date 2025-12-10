from __future__ import annotations


def map_decision_to_side(decision: str | None) -> int:
    """
    Normalize a trade decision into a signed side.

    Returns:
        1 for long, -1 for short, 0 for hold/flat/unknown.
    """
    if not decision:
        return 0
    normalized = str(decision).strip().lower()
    if normalized == "long":
        return 1
    if normalized == "short":
        return -1
    return 0


def map_side_to_label(side: int) -> str:
    """Convert numeric side to standard label."""
    if side > 0:
        return "LONG"
    if side < 0:
        return "SHORT"
    return "FLAT"


def fuse_probabilities(
    probabilities: dict[str, float],
    method: str = "mean",
    weights: dict[str, float] | None = None,
) -> float:
    """
    Fuse multi-timeframe probabilities with a deterministic rule.

    Supported methods:
      - "max": take the maximum probability_up
      - "min": take the minimum probability_up
      - default: weighted average (weights default to 1.0)
    """
    if not probabilities:
        return 0.0
    values = list(probabilities.values())
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    if weights:
        try:
            ordered = [(weights.get(k, 1.0), v) for k, v in probabilities.items()]
            total_w = sum(w for w, _ in ordered)
            if total_w <= 0:
                return float(sum(v for _, v in ordered) / len(ordered))
            return float(sum(w * v for w, v in ordered) / total_w)
        except Exception:
            return float(sum(values) / len(values))
    return float(sum(values) / len(values))
