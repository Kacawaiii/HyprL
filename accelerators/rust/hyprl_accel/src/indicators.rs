//! Technical indicator calculations for OHLCV data

use std::collections::HashMap;

/// Represents a single OHLCV candle
#[derive(Debug, Clone)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Container for computed indicators
#[derive(Debug, Clone)]
pub struct IndicatorSet {
    pub price: Vec<f64>,
    pub sma_short: Vec<Option<f64>>,
    pub sma_long: Vec<Option<f64>>,
    pub ema_short: Vec<Option<f64>>,
    pub ema_long: Vec<Option<f64>>,
    pub rsi: Vec<Option<f64>>,
    pub macd: Vec<Option<f64>>,
    pub macd_signal: Vec<Option<f64>>,
    pub macd_hist: Vec<Option<f64>>,
    pub bb_upper: Vec<Option<f64>>,
    pub bb_middle: Vec<Option<f64>>,
    pub bb_lower: Vec<Option<f64>>,
    pub atr: Vec<Option<f64>>,
    pub trend_ratio: Vec<Option<f64>>,
    pub rolling_vol: Vec<Option<f64>>,
    pub rsi_normalized: Vec<Option<f64>>,
}

impl IndicatorSet {
    pub fn to_dict(&self) -> HashMap<String, Vec<Option<f64>>> {
        let mut map = HashMap::new();
        map.insert("price".to_string(), self.price.iter().map(|&v| Some(v)).collect());
        map.insert("sma_short".to_string(), self.sma_short.clone());
        map.insert("sma_long".to_string(), self.sma_long.clone());
        map.insert("ema_short".to_string(), self.ema_short.clone());
        map.insert("ema_long".to_string(), self.ema_long.clone());
        map.insert("rsi".to_string(), self.rsi.clone());
        map.insert("macd".to_string(), self.macd.clone());
        map.insert("macd_signal".to_string(), self.macd_signal.clone());
        map.insert("macd_hist".to_string(), self.macd_hist.clone());
        map.insert("bb_upper".to_string(), self.bb_upper.clone());
        map.insert("bb_middle".to_string(), self.bb_middle.clone());
        map.insert("bb_lower".to_string(), self.bb_lower.clone());
        map.insert("atr".to_string(), self.atr.clone());
        map.insert("trend_ratio".to_string(), self.trend_ratio.clone());
        map.insert("rolling_vol".to_string(), self.rolling_vol.clone());
        map.insert("rsi_normalized".to_string(), self.rsi_normalized.clone());
        map
    }
}

/// Simple Moving Average
pub fn sma(values: &[f64], window: usize) -> Vec<Option<f64>> {
    let mut result = Vec::with_capacity(values.len());
    
    for i in 0..values.len() {
        if i + 1 < window {
            result.push(None);
        } else {
            let sum: f64 = values[i + 1 - window..=i].iter().sum();
            result.push(Some(sum / window as f64));
        }
    }
    
    result
}

/// Exponential Moving Average
pub fn ema(values: &[f64], window: usize) -> Vec<Option<f64>> {
    let mut result = Vec::with_capacity(values.len());
    
    if values.is_empty() {
        return result;
    }
    
    let alpha = 2.0 / (window as f64 + 1.0);
    let mut ema_val = values[0];
    
    // First value is just the value itself
    result.push(Some(ema_val));
    
    for &val in &values[1..] {
        ema_val = alpha * val + (1.0 - alpha) * ema_val;
        result.push(Some(ema_val));
    }
    
    result
}

/// Relative Strength Index
pub fn rsi(closes: &[f64], window: usize) -> Vec<Option<f64>> {
    let mut result = Vec::with_capacity(closes.len());
    
    if closes.len() < window + 1 {
        return vec![None; closes.len()];
    }
    
    // Calculate price changes
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    for i in 1..closes.len() {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    // First window: simple average
    result.push(None); // First value has no change
    
    for i in 0..gains.len() {
        if i + 1 < window {
            result.push(None);
        } else if i + 1 == window {
            let avg_gain: f64 = gains[i + 1 - window..=i].iter().sum::<f64>() / window as f64;
            let avg_loss: f64 = losses[i + 1 - window..=i].iter().sum::<f64>() / window as f64;
            
            if avg_loss == 0.0 {
                result.push(Some(100.0));
            } else {
                let rs = avg_gain / avg_loss;
                result.push(Some(100.0 - (100.0 / (1.0 + rs))));
            }
        } else {
            // Wilder's smoothing
            if let Some(_prev_rsi) = result[result.len() - 1] {
                let prev_idx = i - 1;
                let prev_avg_gain = if prev_idx >= window - 1 {
                    gains[prev_idx + 1 - window..=prev_idx].iter().sum::<f64>() / window as f64
                } else {
                    0.0
                };
                let prev_avg_loss = if prev_idx >= window - 1 {
                    losses[prev_idx + 1 - window..=prev_idx].iter().sum::<f64>() / window as f64
                } else {
                    0.0
                };
                
                let avg_gain = (prev_avg_gain * (window as f64 - 1.0) + gains[i]) / window as f64;
                let avg_loss = (prev_avg_loss * (window as f64 - 1.0) + losses[i]) / window as f64;
                
                if avg_loss == 0.0 {
                    result.push(Some(100.0));
                } else {
                    let rs = avg_gain / avg_loss;
                    result.push(Some(100.0 - (100.0 / (1.0 + rs))));
                }
            } else {
                result.push(None);
            }
        }
    }
    
    result
}

/// MACD (Moving Average Convergence Divergence)
pub fn macd(closes: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let ema_fast = ema(closes, fast);
    let ema_slow = ema(closes, slow);
    
    let mut macd_line = Vec::with_capacity(closes.len());
    
    for i in 0..closes.len() {
        if let (Some(fast_val), Some(slow_val)) = (ema_fast[i], ema_slow[i]) {
            macd_line.push(Some(fast_val - slow_val));
        } else {
            macd_line.push(None);
        }
    }
    
    // Convert Option<f64> to f64 for signal line calculation
    let macd_values: Vec<f64> = macd_line.iter().filter_map(|&x| x).collect();
    let signal_line_raw = ema(&macd_values, signal);
    
    // Align signal line back to original length
    let mut signal_line = vec![None; closes.len()];
    let mut signal_idx = 0;
    for i in 0..closes.len() {
        if macd_line[i].is_some() {
            if signal_idx < signal_line_raw.len() {
                signal_line[i] = signal_line_raw[signal_idx];
                signal_idx += 1;
            }
        }
    }
    
    let mut histogram = Vec::with_capacity(closes.len());
    for i in 0..closes.len() {
        if let (Some(macd_val), Some(signal_val)) = (macd_line[i], signal_line[i]) {
            histogram.push(Some(macd_val - signal_val));
        } else {
            histogram.push(None);
        }
    }
    
    (macd_line, signal_line, histogram)
}

/// Bollinger Bands
pub fn bollinger_bands(closes: &[f64], window: usize, num_std: f64) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let sma_values = sma(closes, window);
    let mut upper = Vec::with_capacity(closes.len());
    let middle = sma_values.clone();
    let mut lower = Vec::with_capacity(closes.len());
    
    for i in 0..closes.len() {
        if let Some(sma_val) = sma_values[i] {
            if i + 1 >= window {
                let slice = &closes[i + 1 - window..=i];
                let variance: f64 = slice.iter()
                    .map(|&x| (x - sma_val).powi(2))
                    .sum::<f64>() / window as f64;
                let std_dev = variance.sqrt();
                
                upper.push(Some(sma_val + num_std * std_dev));
                lower.push(Some(sma_val - num_std * std_dev));
            } else {
                upper.push(None);
                lower.push(None);
            }
        } else {
            upper.push(None);
            lower.push(None);
        }
    }
    
    (upper, middle, lower)
}

/// Average True Range
pub fn atr(candles: &[Candle], window: usize) -> Vec<Option<f64>> {
    let mut true_ranges = Vec::with_capacity(candles.len());
    
    // First TR is just high - low
    if !candles.is_empty() {
        true_ranges.push(candles[0].high - candles[0].low);
    }
    
    // Subsequent TRs
    for i in 1..candles.len() {
        let high_low = candles[i].high - candles[i].low;
        let high_close = (candles[i].high - candles[i - 1].close).abs();
        let low_close = (candles[i].low - candles[i - 1].close).abs();
        
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    // Apply SMA to true ranges
    sma(&true_ranges, window)
}

/// Rolling Volatility (standard deviation of returns)
pub fn rolling_volatility(closes: &[f64], window: usize) -> Vec<Option<f64>> {
    let mut result = Vec::with_capacity(closes.len());
    
    // Calculate returns
    let mut returns = Vec::with_capacity(closes.len());
    returns.push(0.0); // First return is 0
    
    for i in 1..closes.len() {
        if closes[i - 1] != 0.0 {
            returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
        } else {
            returns.push(0.0);
        }
    }
    
    // Calculate rolling std dev with ddof=1 (sample std, matching pandas)
    for i in 0..returns.len() {
        if i + 1 < window {
            result.push(None);
        } else {
            let slice = &returns[i + 1 - window..=i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (window as f64 - 1.0);  // ddof=1 for sample std
            result.push(Some(variance.sqrt()));
        }
    }
    
    result
}

/// Compute all indicators for a set of candles
pub fn compute_indicators(
    candles: &[Candle],
    sma_short_window: usize,
    sma_long_window: usize,
    rsi_window: usize,
    atr_window: usize,
) -> IndicatorSet {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    
    let sma_short = sma(&closes, sma_short_window);
    let sma_long = sma(&closes, sma_long_window);
    let ema_short = ema(&closes, sma_short_window);
    let ema_long = ema(&closes, sma_long_window);
    let rsi_values = rsi(&closes, rsi_window);
    
    let (macd_line, macd_signal, macd_hist) = macd(&closes, 12, 26, 9);
    let (bb_upper, bb_middle, bb_lower) = bollinger_bands(&closes, 20, 2.0);
    let atr_values = atr(candles, atr_window);
    let rolling_vol = rolling_volatility(&closes, sma_short_window);
    
    // Calculate trend_ratio: (sma_short - sma_long) / sma_long, clipped to [-0.05, 0.05]
    let mut trend_ratio = Vec::with_capacity(closes.len());
    for i in 0..closes.len() {
        if let (Some(short), Some(long)) = (sma_short[i], sma_long[i]) {
            if long != 0.0 {
                let ratio = (short - long) / long;
                trend_ratio.push(Some(ratio.clamp(-0.05, 0.05)));
            } else {
                trend_ratio.push(None);
            }
        } else {
            trend_ratio.push(None);
        }
    }
    
    // Calculate rsi_normalized: (rsi - 50) / 50
    let rsi_normalized: Vec<Option<f64>> = rsi_values.iter()
        .map(|&opt_rsi| opt_rsi.map(|r| (r - 50.0) / 50.0))
        .collect();
    
    IndicatorSet {
        price: closes,
        sma_short,
        sma_long,
        ema_short,
        ema_long,
        rsi: rsi_values,
        macd: macd_line,
        macd_signal,
        macd_hist,
        bb_upper,
        bb_middle,
        bb_lower,
        atr: atr_values,
        trend_ratio,
        rolling_vol,
        rsi_normalized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&values, 3);
        
        assert_eq!(result[0], None);
        assert_eq!(result[1], None);
        assert_eq!(result[2], Some(2.0));
        assert_eq!(result[3], Some(3.0));
        assert_eq!(result[4], Some(4.0));
    }
    
    #[test]
    fn test_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&values, 3);
        
        assert!(result[0].is_some());
        assert!(result[4].is_some());
        assert_eq!(result.len(), values.len());
    }
}
