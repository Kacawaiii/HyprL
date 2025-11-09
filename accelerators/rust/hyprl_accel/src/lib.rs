use pyo3::prelude::*;

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
    let mut exit_price = *closes.last().unwrap_or(&entry_price);
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

#[pymodule]
fn hyprl_accel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trade_path, m)?)?;
    Ok(())
}
