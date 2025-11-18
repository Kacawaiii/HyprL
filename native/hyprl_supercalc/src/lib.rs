use pyo3::prelude::*;
use pyo3::types::PyDict;

mod indicators;
mod metrics;
mod backtest;

/// Candle data structure for OHLCV data
#[derive(Debug, Clone)]
struct Candle {
    high: f64,
    low: f64,
    close: f64,
}

/// Container for computed indicators
#[pyclass]
#[derive(Clone)]
struct IndicatorSet {
    #[pyo3(get)]
    sma_short: Vec<f64>,
    #[pyo3(get)]
    sma_long: Vec<f64>,
    #[pyo3(get)]
    ema_short: Vec<f64>,
    #[pyo3(get)]
    ema_long: Vec<f64>,
    #[pyo3(get)]
    rsi: Vec<f64>,
    #[pyo3(get)]
    macd_line: Vec<f64>,
    #[pyo3(get)]
    macd_signal: Vec<f64>,
    #[pyo3(get)]
    macd_histogram: Vec<f64>,
    #[pyo3(get)]
    bb_upper: Vec<f64>,
    #[pyo3(get)]
    bb_middle: Vec<f64>,
    #[pyo3(get)]
    bb_lower: Vec<f64>,
    #[pyo3(get)]
    atr: Vec<f64>,
    #[pyo3(get)]
    trend_ratio: Vec<f64>,
    #[pyo3(get)]
    volatility: Vec<f64>,
}

#[pymethods]
impl IndicatorSet {
    fn __repr__(&self) -> String {
        format!("IndicatorSet(length={})", self.sma_short.len())
    }
}

/// Compute all indicators for a given set of candles
fn compute_indicators(candles: &[Candle], sma_short: usize, sma_long: usize, rsi_window: usize, atr_window: usize) -> IndicatorSet {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    
    let sma_short_vals = indicators::sma(&closes, sma_short);
    let sma_long_vals = indicators::sma(&closes, sma_long);
    let ema_short_vals = indicators::ema(&closes, sma_short);
    let ema_long_vals = indicators::ema(&closes, sma_long);
    let rsi_vals = indicators::rsi(&closes, rsi_window);
    
    let (macd_line, macd_signal, macd_histogram) = indicators::macd(&closes, 12, 26, 9);
    let (bb_upper, bb_middle, bb_lower) = indicators::bollinger_bands(&closes, 20, 2.0);
    let atr_vals = indicators::atr(&highs, &lows, &closes, atr_window);
    let trend_ratio_vals = indicators::trend_ratio(&sma_short_vals, &sma_long_vals);
    let volatility = indicators::rolling_volatility(&closes, sma_short);
    
    IndicatorSet {
        sma_short: sma_short_vals,
        sma_long: sma_long_vals,
        ema_short: ema_short_vals,
        ema_long: ema_long_vals,
        rsi: rsi_vals,
        macd_line,
        macd_signal,
        macd_histogram,
        bb_upper,
        bb_middle,
        bb_lower,
        atr: atr_vals,
        trend_ratio: trend_ratio_vals,
        volatility,
    }
}

/// Python wrapper for compute_indicators
/// 
/// Takes a pandas DataFrame (as PyDict) and returns a dict of indicator arrays
#[pyfunction]
fn compute_indicators_py(py: Python, df: &Bound<'_, PyAny>, sma_short: usize, sma_long: usize, rsi_window: usize, atr_window: usize) -> PyResult<PyObject> {
    // Extract columns from DataFrame
    let _opens = extract_column(df, "open")?;
    let highs = extract_column(df, "high")?;
    let lows = extract_column(df, "low")?;
    let closes = extract_column(df, "close")?;
    let _volumes = extract_column(df, "volume")?;
    
    let n = closes.len();
    let mut candles = Vec::with_capacity(n);
    
    for i in 0..n {
        candles.push(Candle {
            high: highs[i],
            low: lows[i],
            close: closes[i],
        });
    }
    
    let indicators = compute_indicators(&candles, sma_short, sma_long, rsi_window, atr_window);
    
    // Convert to Python dict
    let result = PyDict::new_bound(py);
    result.set_item("sma_short", indicators.sma_short)?;
    result.set_item("sma_long", indicators.sma_long)?;
    result.set_item("ema_short", indicators.ema_short)?;
    result.set_item("ema_long", indicators.ema_long)?;
    result.set_item("rsi", indicators.rsi)?;
    result.set_item("macd_line", indicators.macd_line)?;
    result.set_item("macd_signal", indicators.macd_signal)?;
    result.set_item("macd_histogram", indicators.macd_histogram)?;
    result.set_item("bb_upper", indicators.bb_upper)?;
    result.set_item("bb_middle", indicators.bb_middle)?;
    result.set_item("bb_lower", indicators.bb_lower)?;
    result.set_item("atr", indicators.atr)?;
    result.set_item("trend_ratio", indicators.trend_ratio)?;
    result.set_item("volatility", indicators.volatility)?;
    
    Ok(result.into())
}

/// Helper function to extract a column from a DataFrame as Vec<f64>
fn extract_column(df: &Bound<'_, PyAny>, col_name: &str) -> PyResult<Vec<f64>> {
    let col = df.getattr(col_name)?;
    let values = col.getattr("values")?;
    let list = values.call_method0("tolist")?;
    
    let vec: Vec<f64> = list.extract()?;
    Ok(vec)
}

/// Python wrapper for Sharpe ratio calculation
#[pyfunction]
fn sharpe_ratio_py(returns: Vec<f64>) -> PyResult<Option<f64>> {
    Ok(metrics::sharpe_ratio(&returns))
}

/// Python wrapper for Sortino ratio calculation
#[pyfunction]
fn sortino_ratio_py(returns: Vec<f64>, target_return: f64) -> PyResult<Option<f64>> {
    Ok(metrics::sortino_ratio(&returns, target_return))
}

/// Python wrapper for max drawdown calculation
#[pyfunction]
fn max_drawdown_py(equity_curve: Vec<f64>) -> PyResult<f64> {
    Ok(metrics::max_drawdown(&equity_curve))
}

/// Python wrapper for win rate calculation
#[pyfunction]
fn win_rate_py(trade_pnls: Vec<f64>) -> PyResult<f64> {
    Ok(metrics::win_rate(&trade_pnls))
}

/// Python wrapper for profit factor calculation
#[pyfunction]
fn profit_factor_py(trade_pnls: Vec<f64>) -> PyResult<Option<f64>> {
    Ok(metrics::profit_factor(&trade_pnls))
}

/// Python wrapper for risk of ruin calculation
#[pyfunction]
fn risk_of_ruin_py(trade_returns: Vec<f64>, risk_pct: f64) -> PyResult<f64> {
    Ok(metrics::risk_of_ruin(&trade_returns, risk_pct))
}

/// Python wrapper for robustness score calculation
#[pyfunction]
#[pyo3(signature = (sharpe=None, profit_factor=None, win_rate=0.0, max_dd=0.0))]
fn robustness_score_py(
    sharpe: Option<f64>,
    profit_factor: Option<f64>,
    win_rate: f64,
    max_dd: f64,
) -> PyResult<f64> {
    Ok(metrics::robustness_score(sharpe, profit_factor, win_rate, max_dd))
}

/// Performance metrics container
#[pyclass]
#[derive(Clone)]
struct PerformanceMetrics {
    #[pyo3(get)]
    sharpe_ratio: Option<f64>,
    #[pyo3(get)]
    sortino_ratio: Option<f64>,
    #[pyo3(get)]
    profit_factor: Option<f64>,
    #[pyo3(get)]
    win_rate: f64,
    #[pyo3(get)]
    max_drawdown: f64,
    #[pyo3(get)]
    risk_of_ruin: f64,
    #[pyo3(get)]
    robustness_score: f64,
    #[pyo3(get)]
    expectancy: f64,
}

#[pymethods]
impl PerformanceMetrics {
    fn __repr__(&self) -> String {
        format!(
            "PerformanceMetrics(sharpe={:.2}, sortino={:.2}, pf={:.2}, wr={:.2}, mdd={:.2})",
            self.sharpe_ratio.unwrap_or(0.0),
            self.sortino_ratio.unwrap_or(0.0),
            self.profit_factor.unwrap_or(0.0),
            self.win_rate,
            self.max_drawdown
        )
    }
}

/// Compute all performance metrics
#[pyfunction]
fn compute_metrics_py(
    equity_curve: Vec<f64>,
    trade_returns: Vec<f64>,
    trade_pnls: Vec<f64>,
    risk_pct: f64,
) -> PyResult<PerformanceMetrics> {
    let sharpe = metrics::sharpe_ratio(&trade_returns);
    let sortino = metrics::sortino_ratio(&trade_returns, 0.0);
    let pf = metrics::profit_factor(&trade_pnls);
    let wr = metrics::win_rate(&trade_pnls);
    let mdd = metrics::max_drawdown(&equity_curve);
    let ror = metrics::risk_of_ruin(&trade_returns, risk_pct);
    let exp = metrics::expectancy(&trade_pnls);
    let robustness = metrics::robustness_score(sharpe, pf, wr, mdd);
    
    Ok(PerformanceMetrics {
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        profit_factor: pf,
        win_rate: wr,
        max_drawdown: mdd,
        risk_of_ruin: ror,
        robustness_score: robustness,
        expectancy: exp,
    })
}

/// Python wrapper for trade simulation
#[pyfunction]
fn simulate_trade_py(
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    direction: &str,
    entry_price: f64,
    stop_price: f64,
    take_profit_price: f64,
    position_size: i64,
) -> PyResult<(f64, usize, f64)> {
    let dir = backtest::Direction::from_str(direction)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid direction"))?;
    
    let risk = backtest::RiskOutcome {
        direction: dir,
        entry_price,
        position_size,
        stop_price,
        take_profit_price,
        risk_amount: 0.0,
        rr_multiple: 0.0,
    };
    
    let (exit_price, exit_idx) = backtest::simulate_trade(&highs, &lows, &closes, &risk);
    let pnl = backtest::compute_trade_pnl(&risk, entry_price, exit_price);
    
    Ok((exit_price, exit_idx, pnl))
}

/// Python wrapper for position sizing
#[pyfunction]
fn compute_position_size_py(
    balance: f64,
    risk_pct: f64,
    entry_price: f64,
    stop_price: f64,
    min_position_size: i64,
) -> PyResult<i64> {
    Ok(backtest::compute_position_size(
        balance,
        risk_pct,
        entry_price,
        stop_price,
        min_position_size,
    ))
}

/// Python wrapper for stop price calculation
#[pyfunction]
fn compute_stop_price_py(
    entry_price: f64,
    atr: f64,
    direction: &str,
    atr_multiplier: f64,
) -> PyResult<f64> {
    let dir = backtest::Direction::from_str(direction)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid direction"))?;
    
    Ok(backtest::compute_stop_price(entry_price, atr, dir, atr_multiplier))
}

/// Python wrapper for take profit calculation
#[pyfunction]
fn compute_take_profit_price_py(
    entry_price: f64,
    atr: f64,
    direction: &str,
    atr_multiplier: f64,
    reward_multiple: f64,
) -> PyResult<f64> {
    let dir = backtest::Direction::from_str(direction)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid direction"))?;
    
    Ok(backtest::compute_take_profit_price(
        entry_price,
        atr,
        dir,
        atr_multiplier,
        reward_multiple,
    ))
}

/// Python module definition
#[pymodule]
fn hyprl_supercalc(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Indicators
    m.add_function(wrap_pyfunction!(compute_indicators_py, m)?)?;
    m.add_class::<IndicatorSet>()?;
    
    // Metrics
    m.add_function(wrap_pyfunction!(sharpe_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(sortino_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown_py, m)?)?;
    m.add_function(wrap_pyfunction!(win_rate_py, m)?)?;
    m.add_function(wrap_pyfunction!(profit_factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk_of_ruin_py, m)?)?;
    m.add_function(wrap_pyfunction!(robustness_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_metrics_py, m)?)?;
    m.add_class::<PerformanceMetrics>()?;
    
    // Backtest
    m.add_function(wrap_pyfunction!(simulate_trade_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_position_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stop_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_take_profit_price_py, m)?)?;
    
    Ok(())
}
