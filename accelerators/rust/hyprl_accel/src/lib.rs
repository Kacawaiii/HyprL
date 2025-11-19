use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod indicators;

fn simulate_path_core(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    direction: &str,
    entry_price: f64,
    stop_price: f64,
    take_profit_price: f64,
) -> f64 {
    let len = highs.len().min(lows.len()).min(closes.len());
    if len == 0 {
        return entry_price;
    }
    let exit_price = *closes.last().unwrap_or(&entry_price);
    let dir = direction.to_ascii_lowercase();
    let is_long = dir == "long";
    for idx in 0..len {
        let high = highs[idx];
        let low = lows[idx];
        let hit_stop = if is_long {
            low <= stop_price
        } else {
            high >= stop_price
        };
        let hit_take = if is_long {
            high >= take_profit_price
        } else {
            low <= take_profit_price
        };
        if hit_stop {
            return stop_price;
        }
        if hit_take {
            return take_profit_price;
        }
    }
    exit_price
}

#[pyfunction]
fn simulate_trade_path(
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    direction: &str,
    entry_price: f64,
    stop_price: f64,
    take_profit_price: f64,
) -> PyResult<f64> {
    Ok(simulate_path_core(
        &highs,
        &lows,
        &closes,
        direction,
        entry_price,
        stop_price,
        take_profit_price,
    ))
}

/// Convert Python DataFrame to Vec<Candle>
fn df_to_candles(_py: Python, df: &PyAny) -> PyResult<Vec<indicators::Candle>> {
    let open_col: Vec<f64> = df.getattr("open")?.call_method0("tolist")?.extract()?;
    let high_col: Vec<f64> = df.getattr("high")?.call_method0("tolist")?.extract()?;
    let low_col: Vec<f64> = df.getattr("low")?.call_method0("tolist")?.extract()?;
    let close_col: Vec<f64> = df.getattr("close")?.call_method0("tolist")?.extract()?;
    let volume_col: Vec<f64> = df.getattr("volume")?.call_method0("tolist")?.extract()?;
    
    let len = open_col.len().min(high_col.len()).min(low_col.len()).min(close_col.len()).min(volume_col.len());
    
    let mut candles = Vec::with_capacity(len);
    for i in 0..len {
        candles.push(indicators::Candle {
            open: open_col[i],
            high: high_col[i],
            low: low_col[i],
            close: close_col[i],
            volume: volume_col[i],
        });
    }
    
    Ok(candles)
}

/// Compute indicators from a DataFrame and return as a Python dict
#[pyfunction]
fn compute_indicators_py(
    py: Python,
    df: &PyAny,
    sma_short_window: usize,
    sma_long_window: usize,
    rsi_window: usize,
    atr_window: usize,
) -> PyResult<PyObject> {
    let candles = df_to_candles(py, df)?;
    let indicator_set = indicators::compute_indicators(
        &candles,
        sma_short_window,
        sma_long_window,
        rsi_window,
        atr_window,
    );
    
    let result = PyDict::new(py);
    let dict = indicator_set.to_dict();
    
    for (key, values) in dict.iter() {
        let py_list = PyList::empty(py);
        for val in values {
            match val {
                Some(v) => py_list.append(v)?,
                None => py_list.append(py.None())?,
            }
        }
        result.set_item(key, py_list)?;
    }
    
    Ok(result.into())
}

#[pymodule]
fn hyprl_accel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trade_path, m)?)?;
    m.add_function(wrap_pyfunction!(compute_indicators_py, m)?)?;
    Ok(())
}
