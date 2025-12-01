//! PyO3 bindings: expose a minimal Python API for HyprL.

use anyhow::anyhow;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_polars::PyDataFrame;

use crate::batch::{evaluate_batch, run_native_search, SearchConstraint};
use crate::core::{BacktestConfig, BacktestReport, Candle, EquityPoint, PerformanceMetrics};
use crate::indicators::compute_indicators;

/// Convert a Polars DataFrame to Vec<Candle> following the HyprL schema.
pub fn df_to_candles(df: &DataFrame) -> anyhow::Result<Vec<Candle>> {
    let ts_col = df.column("ts")?.i64()?;
    let open_col = df.column("open")?.f64()?;
    let high_col = df.column("high")?.f64()?;
    let low_col = df.column("low")?.f64()?;
    let close_col = df.column("close")?.f64()?;
    let vol_col = df.column("volume")?.f64()?;

    let n = df.height();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let ts = ts_col.get(i).ok_or_else(|| anyhow!("ts[{i}] is null"))?;
        let open = open_col
            .get(i)
            .ok_or_else(|| anyhow!("open[{i}] is null"))?;
        let high = high_col
            .get(i)
            .ok_or_else(|| anyhow!("high[{i}] is null"))?;
        let low = low_col.get(i).ok_or_else(|| anyhow!("low[{i}] is null"))?;
        let close = close_col
            .get(i)
            .ok_or_else(|| anyhow!("close[{i}] is null"))?;
        let volume = vol_col
            .get(i)
            .ok_or_else(|| anyhow!("volume[{i}] is null"))?;

        out.push(Candle {
            ts,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    Ok(out)
}

fn to_py_value_error(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn dict_required<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: FromPyObject<'py>,
{
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key '{key}' in BacktestConfig")))?
        .extract::<T>()
}

fn dict_optional<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
where
    T: FromPyObject<'py>,
{
    match dict.get_item(key)? {
        Some(value) => Ok(Some(value.extract::<T>()?)),
        None => Ok(None),
    }
}

fn dict_with_default<'py, T>(dict: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: FromPyObject<'py>,
{
    match dict.get_item(key)? {
        Some(value) => value.extract::<T>(),
        None => Ok(default),
    }
}

fn extract_search_constraint<'py>(obj: &Bound<'py, PyAny>) -> PyResult<SearchConstraint> {
    let dict = obj.downcast::<PyDict>()?;
    let defaults = SearchConstraint::default();
    Ok(SearchConstraint {
        min_trades: dict_with_default(&dict, "min_trades", defaults.min_trades)?,
        min_profit_factor: dict_with_default(
            &dict,
            "min_profit_factor",
            defaults.min_profit_factor,
        )?,
        min_sharpe: dict_with_default(&dict, "min_sharpe", defaults.min_sharpe)?,
        max_drawdown: dict_with_default(&dict, "max_drawdown", defaults.max_drawdown)?,
        max_risk_of_ruin: dict_with_default(
            &dict,
            "max_risk_of_ruin",
            defaults.max_risk_of_ruin,
        )?,
        min_expectancy: dict_with_default(&dict, "min_expectancy", defaults.min_expectancy)?,
        min_robustness: dict_with_default(
            &dict,
            "min_robustness",
            defaults.min_robustness,
        )?,
        max_maxdd_p95: dict_with_default(&dict, "max_maxdd_p95", defaults.max_maxdd_p95)?,
        min_pnl_p05: dict_with_default(&dict, "min_pnl_p05", defaults.min_pnl_p05)?,
        min_pnl_p50: dict_with_default(&dict, "min_pnl_p50", defaults.min_pnl_p50)?,
        min_pnl_p95: dict_with_default(&dict, "min_pnl_p95", defaults.min_pnl_p95)?,
    })
}

fn extract_backtest_config<'py>(obj: &Bound<'py, PyAny>) -> PyResult<BacktestConfig> {
    let dict = obj.downcast::<PyDict>()?;
    Ok(BacktestConfig {
        risk_pct: dict_required(&dict, "risk_pct")?,
        commission_pct: dict_with_default(&dict, "commission_pct", 0.0)?,
        slippage_pct: dict_with_default(&dict, "slippage_pct", 0.0)?,
        max_leverage: dict_with_default(&dict, "max_leverage", 1.0)?,
        params: dict_with_default(&dict, "params", Vec::<f64>::new())?,
        allow_short: dict_with_default(&dict, "allow_short", false)?,
        label: dict_optional(&dict, "label")?,
        atr_window: dict_with_default(&dict, "atr_window", 14_usize)?,
        atr_mult_stop: dict_with_default(&dict, "atr_mult_stop", 2.0)?,
        atr_mult_tp: dict_with_default(&dict, "atr_mult_tp", 4.0)?,
        use_atr_position_sizing: dict_with_default(&dict, "use_atr_position_sizing", false)?,
        trailing_activation_r: dict_with_default(&dict, "trailing_activation_r", 0.0)?,
        trailing_distance_r: dict_with_default(&dict, "trailing_distance_r", 0.0)?,
    })
}

impl<'py> FromPyObject<'py> for BacktestConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        extract_backtest_config(obj)
    }
}

impl<'py> FromPyObject<'py> for SearchConstraint {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        extract_search_constraint(obj)
    }
}

impl IntoPy<PyObject> for BacktestConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("risk_pct", self.risk_pct).unwrap();
        dict.set_item("commission_pct", self.commission_pct)
            .unwrap();
        dict.set_item("slippage_pct", self.slippage_pct).unwrap();
        dict.set_item("max_leverage", self.max_leverage).unwrap();
        dict.set_item("params", self.params).unwrap();
        dict.set_item("allow_short", self.allow_short).unwrap();
        dict.set_item("label", self.label).unwrap();
        dict.set_item("atr_window", self.atr_window).unwrap();
        dict.set_item("atr_mult_stop", self.atr_mult_stop).unwrap();
        dict.set_item("atr_mult_tp", self.atr_mult_tp).unwrap();
        dict.set_item("use_atr_position_sizing", self.use_atr_position_sizing)
            .unwrap();
        dict.set_item("trailing_activation_r", self.trailing_activation_r)
            .unwrap();
        dict.set_item("trailing_distance_r", self.trailing_distance_r)
            .unwrap();
        dict.into_py(py)
    }
}

impl IntoPy<PyObject> for SearchConstraint {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("min_trades", self.min_trades).unwrap();
        dict.set_item("min_profit_factor", self.min_profit_factor)
            .unwrap();
        dict.set_item("min_sharpe", self.min_sharpe).unwrap();
        dict.set_item("max_drawdown", self.max_drawdown).unwrap();
        dict.set_item("max_risk_of_ruin", self.max_risk_of_ruin)
            .unwrap();
        dict.set_item("min_expectancy", self.min_expectancy).unwrap();
        dict.set_item("min_robustness", self.min_robustness).unwrap();
        dict.set_item("max_maxdd_p95", self.max_maxdd_p95).unwrap();
        dict.set_item("min_pnl_p05", self.min_pnl_p05).unwrap();
        dict.set_item("min_pnl_p50", self.min_pnl_p50).unwrap();
        dict.set_item("min_pnl_p95", self.min_pnl_p95).unwrap();
        dict.into_py(py)
    }
}

impl IntoPy<PyObject> for EquityPoint {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("ts", self.ts).unwrap();
        dict.set_item("equity", self.equity).unwrap();
        dict.into_py(py)
    }
}

impl IntoPy<PyObject> for PerformanceMetrics {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_return", self.total_return).unwrap();
        dict.set_item("cagr", self.cagr).unwrap();
        dict.set_item("sharpe", self.sharpe).unwrap();
        dict.set_item("sortino", self.sortino).unwrap();
        dict.set_item("calmar", self.calmar).unwrap();
        dict.set_item("max_drawdown", self.max_drawdown).unwrap();
        dict.set_item("max_drawdown_duration", self.max_drawdown_duration)
            .unwrap();
        dict.set_item("profit_factor", self.profit_factor).unwrap();
        dict.set_item("win_rate", self.win_rate).unwrap();
        dict.set_item("expectancy", self.expectancy).unwrap();
        dict.set_item("risk_of_ruin", self.risk_of_ruin).unwrap();
        dict.set_item("maxdd_p05", self.maxdd_p05).unwrap();
        dict.set_item("maxdd_p95", self.maxdd_p95).unwrap();
        dict.set_item("pnl_p05", self.pnl_p05).unwrap();
        dict.set_item("pnl_p50", self.pnl_p50).unwrap();
        dict.set_item("pnl_p95", self.pnl_p95).unwrap();
        dict.set_item("robustness_score", self.robustness_score)
            .unwrap();
        dict.into_py(py)
    }
}

impl IntoPy<PyObject> for BacktestReport {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        let config_obj = self.config.into_py(py);
        let metrics_obj = self.metrics.into_py(py);
        let equity_curve_list = {
            let list = PyList::empty_bound(py);
            for point in self.equity_curve {
                list.append(point.into_py(py)).unwrap();
            }
            list
        };
        dict.set_item("config", config_obj).unwrap();
        dict.set_item("metrics", metrics_obj).unwrap();
        dict.set_item("equity_curve", equity_curve_list).unwrap();
        dict.set_item("n_trades", self.n_trades).unwrap();
        dict.set_item("debug_info", self.debug_info).unwrap();
        dict.into_py(py)
    }
}

#[pyfunction]
pub fn run_batch_backtest_py(
    df: PyDataFrame,
    signal: Vec<f64>,
    configs: Vec<BacktestConfig>,
) -> PyResult<Vec<BacktestReport>> {
    let df = df.0;
    let candles = df_to_candles(&df).map_err(to_py_value_error)?;

    if candles.len() != signal.len() {
        return Err(PyValueError::new_err("signal and candles length mismatch"));
    }

    let reports = evaluate_batch(&candles, &signal, &configs);
    Ok(reports)
}

#[pyfunction]
pub fn run_native_search_py(
    df: PyDataFrame,
    signal: Vec<f64>,
    configs: Vec<BacktestConfig>,
    constraints: SearchConstraint,
    top_k: usize,
) -> PyResult<Vec<BacktestReport>> {
    let df = df.0;
    let candles = df_to_candles(&df).map_err(to_py_value_error)?;
    if candles.len() != signal.len() {
        return Err(PyValueError::new_err("signal and candles length mismatch"));
    }
    if top_k == 0 {
        return Ok(Vec::new());
    }
    let reports = run_native_search(&candles, &signal, &configs, &constraints, top_k);
    Ok(reports)
}

#[pyfunction]
pub fn compute_indicators_py(py: Python<'_>, df: PyDataFrame) -> PyResult<PyObject> {
    let df = df.0;
    let candles = df_to_candles(&df).map_err(to_py_value_error)?;
    let indicators = compute_indicators(&candles);
    let dict = PyDict::new_bound(py);
    dict.set_item("sma_20", indicators.sma_20)?;
    dict.set_item("ema_20", indicators.ema_20)?;
    dict.set_item("rsi_14", indicators.rsi_14)?;
    dict.set_item("macd", indicators.macd)?;
    dict.set_item("macd_signal", indicators.macd_signal)?;
    dict.set_item("macd_hist", indicators.macd_hist)?;
    dict.set_item("bb_upper_20", indicators.bb_upper_20)?;
    dict.set_item("bb_mid_20", indicators.bb_mid_20)?;
    dict.set_item("bb_lower_20", indicators.bb_lower_20)?;
    dict.set_item("atr_14", indicators.atr_14)?;
    dict.set_item("trend_ratio_50_200", indicators.trend_ratio_50_200)?;
    dict.set_item("rolling_vol_20", indicators.rolling_vol_20)?;
    Ok(dict.into_py(py))
}

#[pymodule]
pub fn hyprl_supercalc(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_batch_backtest_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_native_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_indicators_py, m)?)?;
    Ok(())
}
