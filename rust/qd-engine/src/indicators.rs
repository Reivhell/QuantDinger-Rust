//! Port of `backend_api_python/app/utils/technical_indicators.py`.
//!
//! KDJ(9,3,3) with K/D seeded at 50 (CN terminal convention: 同花顺/东方财富)
//! and Wilder RSI (first average = SMA of the first N changes).
//!
//! Parity note: Python `round(x, 4)` is round-half-even on the decimal form;
//! [`round4`] uses half-away-from-zero. The two agree except on exact ties at
//! the 4th decimal, which do not occur for these smoothed series in practice.

/// Round to 4 decimals (see module note on tie-breaking vs Python).
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// KDJ with K/D initial value 50.
///
/// Returns `(k, d, j)`; entries before index `period - 1` are `None`.
/// Mirrors `compute_kdj_cn(high, low, close, period, k_smooth, d_smooth)`.
/// `k_smooth` / `d_smooth` must be >= 1, as in the Python caller contract.
pub fn compute_kdj_cn(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    k_smooth: usize,
    d_smooth: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = close.len();
    let mut k_out: Vec<Option<f64>> = vec![None; n];
    let mut d_out: Vec<Option<f64>> = vec![None; n];
    let mut j_out: Vec<Option<f64>> = vec![None; n];
    if n < period || period < 1 {
        return (k_out, d_out, j_out);
    }
    let ks = k_smooth as f64;
    let ds = d_smooth as f64;
    let mut k_prev = 50.0;
    let mut d_prev = 50.0;
    for i in (period - 1)..n {
        let mut window_high = f64::NEG_INFINITY;
        let mut window_low = f64::INFINITY;
        for j in (i + 1 - period)..=i {
            if high[j] > window_high {
                window_high = high[j];
            }
            if low[j] < window_low {
                window_low = low[j];
            }
        }
        let rsv = if window_high == window_low {
            50.0
        } else {
            (close[i] - window_low) / (window_high - window_low) * 100.0
        };
        k_prev = (k_prev * (ks - 1.0) + rsv) / ks;
        d_prev = (d_prev * (ds - 1.0) + k_prev) / ds;
        let j_val = 3.0 * k_prev - 2.0 * d_prev;
        k_out[i] = Some(round4(k_prev));
        d_out[i] = Some(round4(d_prev));
        j_out[i] = Some(round4(j_val));
    }
    (k_out, d_out, j_out)
}

fn rsi_from_avgs(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_loss == 0.0 {
        return 100.0;
    }
    let rs = avg_gain / avg_loss;
    round4(100.0 - (100.0 / (1.0 + rs)))
}

/// Wilder RSI; first valid value at index `period`.
/// Mirrors `compute_rsi_wilder(closes, period)`.
pub fn compute_rsi_wilder(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out: Vec<Option<f64>> = vec![None; n];
    if n < period + 1 || period < 1 {
        return out;
    }
    let mut gains = Vec::with_capacity(n - 1);
    let mut losses = Vec::with_capacity(n - 1);
    for i in 1..n {
        let chg = closes[i] - closes[i - 1];
        gains.push(if chg > 0.0 { chg } else { 0.0 });
        losses.push(if chg < 0.0 { -chg } else { 0.0 });
    }
    let p = period as f64;
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / p;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / p;
    out[period] = Some(rsi_from_avgs(avg_gain, avg_loss));
    for i in period..gains.len() {
        avg_gain = (avg_gain * (p - 1.0) + gains[i]) / p;
        avg_loss = (avg_loss * (p - 1.0) + losses[i]) / p;
        out[i + 1] = Some(rsi_from_avgs(avg_gain, avg_loss));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors tests/test_technical_indicators.py
    #[test]
    fn kdj_seeds_k_and_d_at_fifty() {
        let n = 12;
        let flat = vec![10.0; n];
        let (k, d, j) = compute_kdj_cn(&flat, &flat, &flat, 9, 3, 3);
        assert_eq!(k[8], Some(50.0));
        assert_eq!(d[8], Some(50.0));
        assert_eq!(j[8], Some(50.0));
        assert_eq!(k[11], Some(50.0));
    }

    #[test]
    fn kdj_first_valid_bar_blends_seed_with_rsv() {
        let high: Vec<f64> = (0..15).map(|i| i as f64 + 10.0).collect();
        let low: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let close: Vec<f64> = (0..15).map(|i| i as f64 + 5.0).collect();
        let (k, _, _) = compute_kdj_cn(&high, &low, &close, 9, 3, 3);
        let k8 = k[8].expect("first K must exist");
        assert!((50.0..100.0).contains(&k8), "K[8]={k8}, want 50 < K < 100");
    }

    #[test]
    fn rsi_wilder_first_index() {
        let closes = [
            100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 102.0,
            103.0, 104.0, 105.0, 106.0,
        ];
        let rsi = compute_rsi_wilder(&closes, 14);
        assert_eq!(rsi[13], None);
        let v = rsi[14].expect("RSI[14] must exist");
        assert!((0.0..=100.0).contains(&v));
    }

    #[test]
    fn rsi_all_gains_is_100() {
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = compute_rsi_wilder(&closes, 14);
        assert_eq!(rsi[14], Some(100.0));
    }

    #[test]
    fn short_series_returns_all_none() {
        let c = vec![1.0, 2.0, 3.0];
        assert!(compute_rsi_wilder(&c, 14).iter().all(|v| v.is_none()));
        let (k, _, _) = compute_kdj_cn(&c, &c, &c, 9, 3, 3);
        assert!(k.iter().all(|v| v.is_none()));
    }
}
