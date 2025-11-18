//! Technical indicators implementation
//! 
//! This module provides implementations of common technical indicators
//! that match the Python versions for parity testing.

use std::f64;

/// Simple Moving Average
pub fn sma(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    if window == 0 || window > n {
        return result;
    }
    
    for i in (window - 1)..n {
        let sum: f64 = values[(i - window + 1)..=i].iter().sum();
        result[i] = sum / window as f64;
    }
    
    result
}

/// Exponential Moving Average
pub fn ema(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    if window == 0 || window > n {
        return result;
    }
    
    let alpha = 2.0 / (window as f64 + 1.0);
    
    // Start with SMA for the first window
    let mut sum = 0.0;
    for i in 0..window {
        sum += values[i];
    }
    result[window - 1] = sum / window as f64;
    
    // Calculate EMA for remaining values
    for i in window..n {
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
    }
    
    result
}

/// Relative Strength Index
pub fn rsi(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    if window == 0 || n < window + 1 {
        return result;
    }
    
    // Calculate price changes
    let mut gains = Vec::with_capacity(n - 1);
    let mut losses = Vec::with_capacity(n - 1);
    
    for i in 1..n {
        let change = values[i] - values[i - 1];
        gains.push(if change > 0.0 { change } else { 0.0 });
        losses.push(if change < 0.0 { -change } else { 0.0 });
    }
    
    // Calculate initial average gain and loss
    let mut avg_gain: f64 = gains[..window].iter().sum::<f64>() / window as f64;
    let mut avg_loss: f64 = losses[..window].iter().sum::<f64>() / window as f64;
    
    // Calculate RSI for first valid point
    let rs = if avg_loss == 0.0 { 100.0 } else { avg_gain / avg_loss };
    result[window] = 100.0 - (100.0 / (1.0 + rs));
    
    // Calculate RSI for remaining points using smoothed averages
    for i in window..gains.len() {
        avg_gain = (avg_gain * (window - 1) as f64 + gains[i]) / window as f64;
        avg_loss = (avg_loss * (window - 1) as f64 + losses[i]) / window as f64;
        
        let rs = if avg_loss == 0.0 { 100.0 } else { avg_gain / avg_loss };
        result[i + 1] = 100.0 - (100.0 / (1.0 + rs));
    }
    
    result
}

/// MACD (Moving Average Convergence Divergence)
/// Returns (macd_line, signal_line, histogram)
pub fn macd(values: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = values.len();
    let mut macd_line = vec![f64::NAN; n];
    let mut signal_line = vec![f64::NAN; n];
    let mut histogram = vec![f64::NAN; n];
    
    if n < slow_period {
        return (macd_line, signal_line, histogram);
    }
    
    let fast_ema = ema(values, fast_period);
    let slow_ema = ema(values, slow_period);
    
    // Calculate MACD line
    for i in 0..n {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }
    
    // Calculate signal line as EMA of MACD line
    // Find first valid MACD value
    let first_valid = macd_line.iter().position(|&x| !x.is_nan());
    if let Some(start_idx) = first_valid {
        let valid_macd: Vec<f64> = macd_line[start_idx..].to_vec();
        let signal_ema = ema(&valid_macd, signal_period);
        
        for (i, &val) in signal_ema.iter().enumerate() {
            if !val.is_nan() {
                signal_line[start_idx + i] = val;
            }
        }
    }
    
    // Calculate histogram
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }
    
    (macd_line, signal_line, histogram)
}

/// Bollinger Bands
/// Returns (upper_band, middle_band, lower_band)
pub fn bollinger_bands(values: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = values.len();
    let mut upper = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    
    if window == 0 || window > n {
        return (upper, middle, lower);
    }
    
    // Calculate SMA as middle band
    middle = sma(values, window);
    
    // Calculate standard deviation and bands
    for i in (window - 1)..n {
        if !middle[i].is_nan() {
            let slice = &values[(i - window + 1)..=i];
            let mean = middle[i];
            
            // Calculate standard deviation
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            let std_dev = variance.sqrt();
            
            upper[i] = mean + num_std * std_dev;
            lower[i] = mean - num_std * std_dev;
        }
    }
    
    (upper, middle, lower)
}

/// Average True Range
pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], window: usize) -> Vec<f64> {
    let n = highs.len().min(lows.len()).min(closes.len());
    let mut result = vec![f64::NAN; n];
    
    if window == 0 || n < 2 {
        return result;
    }
    
    // Calculate true range for each period
    let mut true_ranges = Vec::with_capacity(n);
    true_ranges.push(highs[0] - lows[0]); // First TR is just high - low
    
    for i in 1..n {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    // Calculate ATR as SMA of true ranges
    if true_ranges.len() >= window {
        // Initial ATR is SMA of first window periods
        let mut atr_val: f64 = true_ranges[..window].iter().sum::<f64>() / window as f64;
        result[window - 1] = atr_val;
        
        // Smoothed ATR for remaining periods
        for i in window..n {
            atr_val = (atr_val * (window - 1) as f64 + true_ranges[i]) / window as f64;
            result[i] = atr_val;
        }
    }
    
    result
}

/// Trend ratio: (short_ma - long_ma) / long_ma, clipped to [-0.05, 0.05]
pub fn trend_ratio(short_ma: &[f64], long_ma: &[f64]) -> Vec<f64> {
    let n = short_ma.len().min(long_ma.len());
    let mut result = vec![f64::NAN; n];
    
    for i in 0..n {
        if !short_ma[i].is_nan() && !long_ma[i].is_nan() && long_ma[i] != 0.0 {
            let ratio = (short_ma[i] - long_ma[i]) / long_ma[i];
            result[i] = ratio.clamp(-0.05, 0.05);
        }
    }
    
    result
}

/// Rolling volatility (standard deviation of returns)
pub fn rolling_volatility(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    if window == 0 || n < 2 {
        return result;
    }
    
    // Calculate returns
    let mut returns = Vec::with_capacity(n - 1);
    for i in 1..n {
        if values[i - 1] != 0.0 {
            returns.push((values[i] - values[i - 1]) / values[i - 1]);
        } else {
            returns.push(0.0);
        }
    }
    
    // Calculate rolling standard deviation
    for i in window..=returns.len() {
        let slice = &returns[(i - window)..i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;
        result[i] = variance.sqrt();
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&values, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&values, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // First EMA is SMA
        assert!(result[3] > 2.0 && result[3] < 4.0);
    }

    #[test]
    fn test_atr() {
        let highs = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let lows = vec![9.0, 10.0, 11.0, 12.0, 13.0];
        let closes = vec![9.5, 10.5, 11.5, 12.5, 13.5];
        let result = atr(&highs, &lows, &closes, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(!result[2].is_nan());
    }
}
