I built a trading system in Python. It worked great until I needed to run parameter optimization.

**The problem:** A grid search over 1,000 parameter combinations took 7.5 minutes. Every time I changed something, I had to wait. Productivity killer.

**The solution:** Rewrite the hot path in Rust, keep Python for everything else.

**Result:** 56x speedup. Same grid search now takes 8 seconds.

## The Stack

- **Python**: ML pipeline, orchestration, data handling
- **Rust + PyO3**: Backtesting engine, indicators, grid search
- **Rayon**: Parallelization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (HyprL)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Strategy    │  │ Features    │  │ Grid Search         │ │
│  │ Engine      │  │ Pipeline    │  │ Optimizer           │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          │ PyO3 FFI                         │
├──────────────────────────┼──────────────────────────────────┤
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              hyprl_supercalc (Rust)                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ indicators/ │  │ backtest/   │  │ batch.rs    │  │   │
│  │  │ - SMA, EMA  │  │ - Engine    │  │ - Parallel  │  │   │
│  │  │ - RSI, MACD │  │ - Metrics   │  │   search    │  │   │
│  │  │ - ATR, BB   │  │ - Trades    │  │ - Rayon     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         Rust                                │
└─────────────────────────────────────────────────────────────┘
```

## Before: Pure Python

```python
def backtest(prices, signals, stop_loss, take_profit):
    equity = [initial_capital]
    for i in range(len(prices)):
        # ... lot of loops and conditions
    return calculate_metrics(equity)

# Grid search: 1000 combinations
results = [backtest(prices, signals, sl, tp)
           for sl, tp in param_grid]  # 7.5 minutes
```

## After: Rust + PyO3

```rust
#[pyfunction]
fn run_backtest_py(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    signals: PyReadonlyArray1<i8>,
    stop_loss: f64,
    take_profit: f64,
) -> PyResult<PyObject> {
    let prices = prices.as_slice()?;
    let signals = signals.as_slice()?;

    let result = run_backtest(prices, signals, stop_loss, take_profit);

    Ok(result.into_py(py))
}
```

Python side stays clean:

```python
from hyprl_supercalc import run_backtest

# Same API, 56x faster
results = [run_backtest(prices, signals, sl, tp)
           for sl, tp in param_grid]  # 8 seconds
```

## Key Optimizations

### 1. Zero-Copy NumPy Arrays

PyO3's `PyReadonlyArray` gives direct access to NumPy memory:

```rust
use numpy::PyReadonlyArray1;

let prices = prices.as_slice()?;  // No copy, just a pointer
```

### 2. Parallel Grid Search with Rayon

```rust
use rayon::prelude::*;

pub fn run_grid_search(params: &[Params]) -> Vec<BacktestResult> {
    params
        .par_iter()  // Runs on all cores
        .map(|p| run_single_backtest(p))
        .collect()
}
```

### 3. Pre-allocated Buffers

```rust
// Bad: allocates every iteration
for i in 0..n {
    let temp = Vec::new();
}

// Good: reuse buffer
let mut buffer = Vec::with_capacity(n);
for i in 0..n {
    buffer.clear();
    // reuse buffer
}
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Parallelism** | Rayon | Work-stealing scheduler for all CPU cores |
| **Data Engine** | Polars | SIMD-vectorized DataFrames |
| **FFI Bridge** | PyO3 | Zero-copy Python-Rust interop |
| **Mathematics** | ndarray | Cache-optimized matrix operations |

## Benchmarks

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| ATR 14 (10k bars) | 45ms | 1.2ms | **37x** |
| RSI 14 (10k bars) | 38ms | 0.9ms | **42x** |
| Single backtest | 450ms | 12ms | **37x** |
| Grid search (1000) | 7.5min | 8sec | **56x** |

**Memory**: 80% RAM reduction compared to pure Pandas.

## Try It Yourself

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install maturin (Python-Rust build tool)
pip install maturin
```

### Build from Source

```bash
# Clone the repo
git clone https://github.com/Kacawaiii/HyprL.git
cd HyprL/native/hyprl_supercalc

# Development build (faster compilation)
maturin develop

# Release build (faster runtime) - RECOMMENDED
maturin develop --release
```

### Verify Installation

```python
from hyprl_supercalc import compute_indicators_py, run_backtest_py
print("Rust engine loaded successfully!")
```

### Usage Example

```python
import numpy as np
from hyprl_supercalc import run_backtest_py

prices = np.array([100.0, 101.0, 99.5, 102.0, ...])
signals = np.array([1, 0, -1, 1, ...])  # 1=long, -1=short, 0=flat

result = run_backtest_py(
    prices=prices,
    signals=signals,
    stop_loss=0.02,
    take_profit=0.04
)

print(f"Sharpe: {result['sharpe']}")
print(f"Total return: {result['total_return']}")
```

## The Hard Parts

**Lifetimes with NumPy views.** Getting the borrow checker happy with numpy array views took trial and error.

**Building for multiple platforms.** Maturin helps, but testing Linux/Mac/Windows is still needed.

**Debugging across the boundary.** When Rust crashes from Python, stack traces are not great.

## Was It Worth It?

Absolutely.

- Development iteration is 56x faster
- I can explore more parameter combinations
- The Rust code is actually cleaner than Python

## Resources

- [Full source code](https://github.com/Kacawaiii/HyprL/tree/main/native)
- [Rust Engine Documentation](https://github.com/Kacawaiii/HyprL/blob/main/docs/RUST_ENGINE.md)
- [PyO3 User Guide](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)

If you're hitting performance walls in Python, consider rewriting just the hot path in Rust. PyO3 makes the integration surprisingly smooth.

---

*Questions? Drop a comment or find me on [LinkedIn](https://www.linkedin.com/in/thomas-grunfeld-b70470255).*
