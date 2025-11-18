use pyo3::prelude::*;
use pyo3::types::PyDict;

mod indicators;

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

/// Python module definition
#[pymodule]
fn hyprl_supercalc(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_indicators_py, m)?)?;
    m.add_class::<IndicatorSet>()?;
    Ok(())
}
