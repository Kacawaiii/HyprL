//! Core technical indicators used by HyprL supersearch.

use crate::core::Candle;

const NAN: f64 = f64::NAN;

#[derive(Debug, Clone)]
pub struct IndicatorSet {
    pub sma_20: Vec<f64>,
    pub ema_20: Vec<f64>,
    pub rsi_14: Vec<f64>,
    pub macd: Vec<f64>,
    pub macd_signal: Vec<f64>,
    pub macd_hist: Vec<f64>,
    pub bb_upper_20: Vec<f64>,
    pub bb_mid_20: Vec<f64>,
    pub bb_lower_20: Vec<f64>,
    pub atr_14: Vec<f64>,
    pub trend_ratio_50_200: Vec<f64>,
    pub rolling_vol_20: Vec<f64>,
}

pub fn compute_indicators(candles: &[Candle]) -> IndicatorSet {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();

    let sma_20 = sma(&closes, 20);
    let ema_20 = ema(&closes, 20);
    let rsi_14 = rsi(&closes, 14);
    let (macd, macd_signal, macd_hist) = macd_full(&closes);
    let (bb_mid_20, bb_upper_20, bb_lower_20) = bollinger_bands(&closes, 20, 2.0);
    let mut atr_14 = atr(&highs, &lows, &closes, 14);
    zero_warmup(&mut atr_14, 14);
    let trend_ratio_50_200 = trend_ratio(&closes, 50, 200);
    let rolling_vol_20 = rolling_volatility(&closes, 20);

    IndicatorSet {
        sma_20,
        ema_20,
        rsi_14,
        macd,
        macd_signal,
        macd_hist,
        bb_upper_20,
        bb_mid_20,
        bb_lower_20,
        atr_14,
        trend_ratio_50_200,
        rolling_vol_20,
    }
}

/// Simple moving average with NaN warmup until `window` samples.
pub fn sma(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![NAN; n];
    if window == 0 || n == 0 {
        return out;
    }
    let w = window;
    let mut sum = 0.0;
    let mut nan_count = 0usize;
    for i in 0..n {
        let val = series[i];
        if val.is_finite() {
            sum += val;
        } else {
            nan_count += 1;
        }

        if i >= w {
            let old = series[i - w];
            if old.is_finite() {
                sum -= old;
            } else if nan_count > 0 {
                nan_count -= 1;
            }
        }

        if i + 1 >= w && nan_count == 0 {
            out[i] = sum / w as f64;
        }
    }
    out
}

/// Exponential moving average matching pandas ewm(adjust=False).
pub fn ema(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![NAN; n];
    if n == 0 {
        return out;
    }
    let alpha = 2.0 / (window.max(1) as f64 + 1.0);
    let mut prev: Option<f64> = None;
    for (idx, value) in series.iter().enumerate() {
        if !value.is_finite() {
            out[idx] = NAN;
            prev = None;
            continue;
        }
        let ema_val = match prev {
            Some(prev_val) => alpha * value + (1.0 - alpha) * prev_val,
            None => *value,
        };
        out[idx] = ema_val;
        prev = Some(ema_val);
    }
    out
}

/// Relative Strength Index matching ta.momentum RSIIndicator semantics.
pub fn rsi(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![NAN; n];
    if n <= 1 || window == 0 {
        return out;
    }

    let mut gains = vec![0.0_f64; n];
    let mut losses = vec![0.0_f64; n];
    for i in 1..n {
        let curr = series[i];
        let prev = series[i - 1];
        let change = if curr.is_finite() && prev.is_finite() {
            curr - prev
        } else {
            0.0
        };
        gains[i] = if change > 0.0 { change } else { 0.0 };
        losses[i] = if change < 0.0 { -change } else { 0.0 };
    }

    let alpha = 1.0 / window as f64;
    let ema_up = ewm(&gains, alpha, window);
    let ema_down = ewm(&losses, alpha, window);

    for i in 0..n {
        let up = ema_up[i];
        let down = ema_down[i];
        if down == 0.0 && up.is_finite() {
            out[i] = 100.0;
        } else if up.is_finite() && down.is_finite() && down != 0.0 {
            let rs = up / down;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    out
}

/// Average True Range (ATR) matching existing backtest semantics.
pub fn atr(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    let n = high.len().min(low.len()).min(close.len());
    if n == 0 {
        return Vec::new();
    }
    let mut tr = vec![0.0_f64; n];
    for i in 0..n {
        let hl = (high[i] - low[i]).abs();
        if i == 0 {
            tr[i] = hl.max(0.0);
        } else {
            let prev_close = close[i - 1];
            let up = (high[i] - prev_close).abs();
            let down = (low[i] - prev_close).abs();
            tr[i] = hl.max(up.max(down));
        }
    }

    let window = window.max(1);
    let mut atr_vals = vec![0.0_f64; n];
    let mut init_len = window.min(n);
    if init_len == 0 {
        init_len = n;
    }

    let mut running_sum = 0.0;
    for i in 0..init_len {
        running_sum += tr[i];
        let denom = (i + 1) as f64;
        atr_vals[i] = if denom > 0.0 { running_sum / denom } else { tr[i] };
    }

    if init_len == n {
        return atr_vals;
    }

    let mut prev_atr = atr_vals[init_len - 1];
    let window_f = window as f64;
    for i in init_len..n {
        prev_atr = ((prev_atr * (window_f - 1.0)) + tr[i]) / window_f;
        atr_vals[i] = prev_atr;
    }

    atr_vals
}

/// Bollinger Bands: (middle, upper, lower) using SMA + population std.
pub fn bollinger_bands(
    series: &[f64],
    window: usize,
    num_std: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mid = sma(series, window);
    let std = rolling_std(series, window);
    let mut upper = vec![NAN; series.len()];
    let mut lower = vec![NAN; series.len()];
    for i in 0..series.len() {
        if mid[i].is_finite() && std[i].is_finite() {
            upper[i] = mid[i] + num_std * std[i];
            lower[i] = mid[i] - num_std * std[i];
        }
    }
    (mid, upper, lower)
}

fn macd_full(series: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema12 = ema(series, 12);
    let ema26 = ema(series, 26);
    let n = series.len();
    let mut macd = vec![NAN; n];
    for i in 0..n {
        if ema12[i].is_finite() && ema26[i].is_finite() {
            macd[i] = ema12[i] - ema26[i];
        }
    }
    let signal = ema(&macd, 9);
    let mut hist = vec![NAN; n];
    for i in 0..n {
        if macd[i].is_finite() && signal[i].is_finite() {
            hist[i] = macd[i] - signal[i];
        }
    }
    (macd, signal, hist)
}

fn trend_ratio(series: &[f64], short_window: usize, long_window: usize) -> Vec<f64> {
    let short = sma(series, short_window);
    let long = sma(series, long_window);
    let mut out = vec![NAN; series.len()];
    for i in 0..series.len() {
        if short[i].is_finite() && long[i].is_finite() && long[i].abs() > f64::EPSILON {
            let ratio = short[i] / long[i] - 1.0;
            out[i] = ratio.clamp(-0.05, 0.05);
        }
    }
    out
}

fn rolling_volatility(series: &[f64], window: usize) -> Vec<f64> {
    let returns = log_returns(series);
    rolling_std(&returns, window)
}

fn log_returns(series: &[f64]) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![NAN; n];
    for i in 1..n {
        let curr = series[i];
        let prev = series[i - 1];
        if curr.is_finite() && prev.is_finite() && prev > 0.0 {
            out[i] = (curr / prev).ln();
        }
    }
    out
}

fn rolling_std(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![NAN; n];
    if window == 0 || n == 0 {
        return out;
    }
    let w = window;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut nan_count = 0usize;
    for i in 0..n {
        let val = series[i];
        if val.is_finite() {
            sum += val;
            sum_sq += val * val;
        } else {
            nan_count += 1;
        }

        if i >= w {
            let old = series[i - w];
            if old.is_finite() {
                sum -= old;
                sum_sq -= old * old;
            } else if nan_count > 0 {
                nan_count -= 1;
            }
        }

        if i + 1 >= w && nan_count == 0 {
            let mean = sum / w as f64;
            let var = (sum_sq / w as f64) - mean * mean;
            out[i] = if var > 0.0 { var.sqrt() } else { 0.0 };
        }
    }
    out
}

fn ewm(series: &[f64], alpha: f64, min_periods: usize) -> Vec<f64> {
    let mut out = vec![NAN; series.len()];
    if series.is_empty() {
        return out;
    }
    let mut prev: Option<f64> = None;
    let mut valid = 0usize;
    for (idx, value) in series.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        prev = Some(match prev {
            Some(prev_val) => alpha * value + (1.0 - alpha) * prev_val,
            None => *value,
        });
        valid += 1;
        if valid >= min_periods {
            out[idx] = prev.unwrap_or(NAN);
        }
    }
    out
}

fn zero_warmup(series: &mut [f64], window: usize) {
    if window <= 1 {
        return;
    }
    let limit = window.saturating_sub(1).min(series.len());
    for value in series.iter_mut().take(limit) {
        *value = 0.0;
    }
}
