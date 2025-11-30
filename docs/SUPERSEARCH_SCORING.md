# HyprL Supersearch Scoring System

## Overview

HyprL uses a **lexicographic ranking** system to order search results, followed by
**normalized rank-based scoring** to assign `base_score` values in [0, 1].

Optionally, a **meta-model** can adjust `final_score` to predict out-of-sample robustness.

---

## Lexicographic Ranking

### Scoring Tuple

Each strategy is ranked by a tuple of composite metrics (in priority order):

```python
score_tuple = (
    -pf + ror * 2.0 + sentiment_ratio,     # 1. Primary: reward PF, penalize RoR and sentiment
    -sharpe + ror,                          # 2. Risk-adjusted: reward Sharpe, penalize RoR
    primary_dd + ror * 100.0 - exp * 100.0 # 3. DD/expectancy: penalize DD and RoR, reward exp
)
```

### Comparison Logic

Python compares tuples **lexicographically** (left-to-right):

1. First, strategies are ranked by the **first element** (primary score).
2. If tied, they're ranked by the **second element** (risk-adjusted score).
3. If still tied, by the **third element** (DD/expectancy score).

**Lower tuple values rank better** (sorted ascending).

### Example

| Strategy | PF  | Sharpe | DD   | RoR  | Exp  | Tuple                         | Rank |
|----------|-----|--------|------|------|------|-------------------------------|------|
| A        | 2.5 | 1.8    | 0.20 | 0.05 | 0.5  | (-2.40, -1.75, 4.70)          | 1    |
| B        | 2.5 | 1.6    | 0.20 | 0.05 | 0.5  | (-2.40, -1.55, 4.70)          | 2    |
| C        | 3.0 | 2.0    | 0.15 | 0.03 | 0.6  | (-2.94, -1.97, 3.15)          | 0    |

**Explanation:**
- **C ranks #1**: Best primary score (-2.94), even though it's not #1 on all metrics.
- **A ranks #2**: Same primary score as B (-2.40), but better Sharpe (second element).
- **B ranks #3**: Worse Sharpe than A (second element loses).

---

## Base Score Assignment

After ranking, each strategy gets a **normalized rank score**:

```python
base_score = 1.0 - (rank / (N - 1))
```

Where:
- `rank` = 0 for best, 1 for second-best, ..., N-1 for worst
- `N` = total number of results

### Example (10 strategies)

| Rank | base_score |
|------|------------|
| 0    | 1.00       |
| 1    | 0.89       |
| 2    | 0.78       |
| 3    | 0.67       |
| ...  | ...        |
| 9    | 0.00       |

### Interpretation

- `base_score` is **relative** to this search batch.
- It does NOT indicate absolute quality (e.g., 0.8 ≠ "80% good").
- It means: "This strategy ranks in the top X% of this supersearch run."

---

## Meta-Model Adjustment (Optional)

If `--meta-model` is specified (or `meta_robustness_model_path` in config):

1. The meta-model predicts **out-of-sample robustness** based on backtest features:
   - PF, Sharpe, DD distribution
   - Win rate consistency
   - Risk-of-ruin tail risk
   - Portfolio correlation structure

2. `final_score` is computed as a **weighted blend**:

```python
final_score = (1 - meta_weight) * base_score + meta_weight * meta_prediction
```

Default: `meta_weight = 0.4` (60% base ranking, 40% meta-model).

### Purpose

- **Base ranking** reflects in-sample performance (backtest).
- **Meta-model** adds expected degradation on live/unseen data.
- Strategies with high `base_score` but low `meta_prediction` (fragile) get downranked.

### Example

| Strategy | base_score | meta_prediction | final_score (w=0.4) |
|----------|------------|-----------------|---------------------|
| A        | 1.00       | 0.80            | 0.92                |
| B        | 0.89       | 0.90            | 0.89                |
| C        | 0.78       | 0.95            | 0.85                |

**B ranks above C** after meta-adjustment (0.89 > 0.85), even though C has higher meta-prediction.

---

## Hard Constraints

Before scoring, all strategies must pass **hard constraints**:

| Constraint       | Default | Purpose                                  |
|------------------|---------|------------------------------------------|
| `--min-trades`   | 50      | Statistical confidence                   |
| `--min-pf`       | 1.2     | Minimum profitability                    |
| `--min-sharpe`   | 0.5     | Minimum risk-adjusted returns            |
| `--max-dd`       | 0.40    | Maximum tolerable drawdown (40%)         |
| `--max-ror`      | 0.10    | Maximum risk-of-ruin (10%)               |
| `--min-expectancy` | 0.0   | Must have positive edge                  |

Strategies failing any constraint are **rejected before scoring**.

---

## CLI Usage

### Default (no meta-model)

```bash
python scripts/run_supersearch.py \
    --ticker AAPL \
    --min-trades 50 \
    --min-pf 1.3 \
    --max-dd 0.30
```

### With meta-model

```bash
python scripts/run_supersearch.py \
    --ticker AAPL \
    --meta-model robustness@stable \
    --meta-weight 0.4
```

Or specify a path:

```bash
python scripts/run_supersearch.py \
    --ticker AAPL \
    --meta-robustness-model-path data/models/meta_robustness_v1.pkl \
    --meta-weight 0.5
```

---

## Comparison to Weighted Scoring

### Why Not Use Weighted Sums?

Example weighted formula:

```python
score = w_pf * PF + w_sharpe * Sharpe - w_dd * DD - w_ror * RoR + w_exp * Expectancy
```

**Problems:**

1. **Weight tuning**: Choosing `w_pf=1.0`, `w_sharpe=1.0`, `w_ror=-10.0` is arbitrary.
2. **Scale mismatch**: PF ∈ [1, 3], Sharpe ∈ [0, 2], RoR ∈ [0, 1] have different ranges.
3. **Masking**: A bad metric (e.g., RoR=0.15) can be masked by good PF (e.g., 3.0).

### Lexicographic Ranking Advantages

- **No arbitrary weights**: Priority is explicit (primary > risk-adjusted > DD/exp).
- **No masking**: A bad metric at higher priority rejects the strategy outright.
- **Interpretable**: Users know why strategy A ranks above B (inspect tuple element-by-element).

---

## Future Extensions

### Making Priorities Configurable

If needed, we can add CLI flags to reorder tuple elements:

```bash
--score-priority "pf,sharpe,dd,ror,expectancy"  # Custom priority order
```

Or switch to weighted mode:

```bash
--score-mode weighted \
--score-weight-pf 2.0 \
--score-weight-sharpe 1.5
```

For now, the **fixed lexicographic order** is sufficient for research.

---

## Implementation Details

### Code Location

- **Scoring logic**: `src/hyprl/search/optimizer.py`
  - `_score_tuple(result)`: Computes the ranking tuple
  - `_assign_base_scores(results)`: Assigns normalized base_score
  - `_apply_meta_scores(results, model, weight)`: Applies meta-model adjustment

### Key Invariants

1. **Lower tuple values rank better** (ascending sort).
2. **base_score ∈ [0, 1]**, where 1.0 = best, 0.0 = worst.
3. **final_score ∈ [0, 1]** after meta-model blend.
4. **Hard constraints are ALWAYS enforced** before scoring (no exceptions).

### Testing

Run scoring tests:

```bash
pytest tests/search/test_optimizer.py -v -k score
```

---

## Summary

| Component           | Purpose                                | Range       |
|---------------------|----------------------------------------|-------------|
| `_score_tuple`      | Lexicographic ranking (primary sort)   | (float, float, float) |
| `base_score`        | Normalized rank within batch           | [0, 1]      |
| `meta_prediction`   | Out-of-sample robustness (optional)    | [0, 1]      |
| `final_score`       | Blended score for final ranking        | [0, 1]      |

**Result:**  
Strategies are ranked by `final_score` (descending), with `base_score` as tiebreaker.
