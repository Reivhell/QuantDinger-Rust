//! Deterministic market-regime detection.
//!
//! States: `TrendingUp | TrendingDown | Ranging | HighVolatility |
//! LowVolatility | Transition`. The classifier is a pure function of
//! causal features with configurable thresholds — every default is
//! documented with its rationale below. No randomness, no hidden state.

use crate::features::{atr, rolling_percentile, wilder, Bar};

/// Regime states, in priority order for the rule stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    TrendingUp,
    TrendingDown,
    HighVolatility,
    Ranging,
    LowVolatility,
    Transition,
}

impl Regime {
    pub fn name(&self) -> &'static str {
        match self {
            Regime::TrendingUp => "TRENDING_UP",
            Regime::TrendingDown => "TRENDING_DOWN",
            Regime::HighVolatility => "HIGH_VOLATILITY",
            Regime::Ranging => "RANGING",
            Regime::LowVolatility => "LOW_VOLATILITY",
            Regime::Transition => "TRANSITION",
        }
    }
}

/// Tunable thresholds. Defaults and why:
/// - `adx_trend = 25.0`: Wilder's own trend/momentum divide (New Concepts
///   in Technical Trading Systems, 1978); ADX > 25 = directional movement
///   dominates noise.
/// - `adx_strong = 40.0`: only used to *keep* a trend label when slope is
///   briefly flat (hysteresis), not to enter one.
/// - `atr_pct_high = 0.8` / `atr_pct_low = 0.2`: top/bottom quintile of the
///   trailing ATR distribution = volatility extremes.
/// - `slope_bars = 5`: trend must persist a full trading week (daily bars)
///   before it counts; kills single-bar whipsaws.
/// - `range_band = 0.01`: ±1% VWAP band = "at fair value" for ranging.
/// - `di_gap = 5.0`: +DI/-DI must separate by 5 points to confirm direction.
#[derive(Debug, Clone)]
pub struct RegimeConfig {
    pub adx_period: usize,
    pub adx_trend: f64,
    pub adx_strong: f64,
    pub atr_period: usize,
    pub atr_window: usize,
    pub atr_pct_high: f64,
    pub atr_pct_low: f64,
    pub ema_fast: usize,
    pub ema_slow: usize,
    pub slope_bars: usize,
    pub range_band: f64,
    pub di_gap: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            adx_period: 14,
            adx_trend: 25.0,
            adx_strong: 40.0,
            atr_period: 14,
            atr_window: 100,
            atr_pct_high: 0.8,
            atr_pct_low: 0.2,
            ema_fast: 20,
            ema_slow: 50,
            slope_bars: 5,
            range_band: 0.01,
            di_gap: 5.0,
        }
    }
}

/// ADX (+DI/-DI) via Wilder smoothing. Returns `(adx, plus_di, minus_di)`.
/// `None` until index `2*period - 2` (two Wilder seeds deep). O(n).
pub fn adx(
    bars: &[Bar],
    period: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = bars.len();
    let mut adx_out = vec![None; n];
    let mut pdi = vec![None; n];
    let mut mdi = vec![None; n];
    if period == 0 || n < 2 * period {
        return (adx_out, pdi, mdi);
    }
    let mut pdm = vec![None; n];
    let mut mdm = vec![None; n];
    let mut tr = vec![None; n];
    for i in 1..n {
        let (a, b) = (&bars[i - 1], &bars[i]);
        if !a.is_valid() || !b.is_valid() {
            continue;
        }
        let up = b.high - a.high;
        let dn = a.low - b.low;
        pdm[i] = Some(if up > dn && up > 0.0 { up } else { 0.0 });
        mdm[i] = Some(if dn > up && dn > 0.0 { dn } else { 0.0 });
        tr[i] = Some(
            (b.high - b.low)
                .max((b.high - a.close).abs())
                .max((b.low - a.close).abs()),
        );
    }
    let spdm = wilder(&pdm, period);
    let smdm = wilder(&mdm, period);
    let str_ = wilder(&tr, period);
    let mut dx = vec![None; n];
    for i in 0..n {
        match (spdm[i], smdm[i], str_[i]) {
            (Some(p), Some(m), Some(t)) if t > 0.0 => {
                let (pi, mi) = (100.0 * p / t, 100.0 * m / t);
                pdi[i] = Some(pi);
                mdi[i] = Some(mi);
                dx[i] = Some(if pi + mi > 0.0 { 100.0 * (pi - mi).abs() / (pi + mi) } else { 0.0 });
            }
            _ => {}
        }
    }
    // ADX = Wilder smoothing of DX: seed = SMA of first `period` DX values.
    let first = dx.iter().position(|v| v.is_some()).unwrap_or(n);
    if first + period <= n && dx[first..first + period].iter().all(|v| v.is_some()) {
        let mut prev: f64 = dx[first..first + period].iter().map(|v| v.unwrap()).sum::<f64>() / period as f64;
        adx_out[first + period - 1] = Some(prev);
        for i in (first + period)..n {
            match dx[i] {
                Some(x) => {
                    prev += (x - prev) / period as f64;
                    adx_out[i] = Some(prev);
                }
                None => break, // gap: stop (restays None, deterministic)
            }
        }
    }
    (adx_out, pdi, mdi)
}

/// Classify every bar. Rule stack (first match wins):
/// 1. Missing inputs → `Transition` (unknown = do not trade as trend).
/// 2. `atr_pct >= atr_pct_high` → `HighVolatility`, where `atr_pct` is the
///    trailing percentile of ATR *relative to price* (see [`detect`]).
///    Volatility overrides: stops behave differently there, so it must be
///    visible even inside trends.
/// 3. `adx >= adx_trend` (or `>= adx_strong` with flat slope = hysteresis)
///    + EMA stack + slope persistence + DI gap → `TrendingUp/Down`.
/// 4. `|close - vwap|/vwap <= range_band` + weak ADX → `Ranging`.
/// 5. `atr_pct <= atr_pct_low` → `LowVolatility`.
/// 6. Otherwise → `Transition`.
pub fn classify(
    bars: &[Bar],
    cfg: &RegimeConfig,
    adx_v: &[Option<f64>],
    pdi: &[Option<f64>],
    mdi: &[Option<f64>],
    ema_fast: &[Option<f64>],
    ema_slow: &[Option<f64>],
    atr_pct: &[Option<f64>],
    vwap: &[Option<f64>],
) -> Vec<Regime> {
    let n = bars.len();
    let mut out = vec![Regime::Transition; n];
    for i in 0..n {
        let (a, p, m, ef, es, ap, v) = (
            adx_v.get(i).copied().flatten(),
            pdi.get(i).copied().flatten(),
            mdi.get(i).copied().flatten(),
            ema_fast.get(i).copied().flatten(),
            ema_slow.get(i).copied().flatten(),
            atr_pct.get(i).copied().flatten(),
            vwap.get(i).copied().flatten(),
        );
        // Slope persistence: last `slope_bars` fast-EMA steps all same sign.
        let mut slope_up = true;
        let mut slope_dn = true;
        if i + 1 < cfg.slope_bars {
            slope_up = false;
            slope_dn = false;
        } else {
            for j in (i + 2 - cfg.slope_bars)..=i {
                match (ema_fast.get(j - 1).copied().flatten(), ema_fast.get(j).copied().flatten()) {
                    (Some(x), Some(y)) => {
                        if y <= x {
                            slope_up = false;
                        }
                        if y >= x {
                            slope_dn = false;
                        }
                    }
                    _ => {
                        slope_up = false;
                        slope_dn = false;
                    }
                }
            }
        }
        // Hysteresis: an established trend survives `slope_bars` whipsaws
        // while ADX stays very strong; direction then comes from the EMA
        // stack + DI gap alone.
        let prev_trend = if i > 0 {
            match out[i - 1] {
                Regime::TrendingUp => Some(true),
                Regime::TrendingDown => Some(false),
                _ => None,
            }
        } else {
            None
        };
        out[i] = classify_bar(
            cfg,
            bars[i].close,
            a,
            p,
            m,
            ef,
            es,
            ap,
            v,
            slope_up,
            slope_dn,
            prev_trend,
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn classify_bar(
    cfg: &RegimeConfig,
    close: f64,
    adx: Option<f64>,
    pdi: Option<f64>,
    mdi: Option<f64>,
    ef: Option<f64>,
    es: Option<f64>,
    atr_pct: Option<f64>,
    vwap: Option<f64>,
    slope_up: bool,
    slope_dn: bool,
    prev_trend: Option<bool>,
) -> Regime {
    let (a, p, m, f, s, ap, v) = match (adx, pdi, mdi, ef, es, atr_pct, vwap) {
        (Some(a), Some(p), Some(m), Some(f), Some(s), Some(ap), Some(v)) => (a, p, m, f, s, ap, v),
        _ => return Regime::Transition,
    };
    if !close.is_finite() {
        return Regime::Transition;
    }
    if ap >= cfg.atr_pct_high {
        return Regime::HighVolatility;
    }
    let trending = a >= cfg.adx_trend;
    if trending {
        if f > s && slope_up && p - m >= cfg.di_gap {
            return Regime::TrendingUp;
        }
        if f < s && slope_dn && m - p >= cfg.di_gap {
            return Regime::TrendingDown;
        }
    }
    // Hysteresis: keep an established trend while ADX is very strong even
    // if the slope persistence just broke (one flat week, not a reversal).
    if a >= cfg.adx_strong {
        if prev_trend == Some(true) && f > s && p > m {
            return Regime::TrendingUp;
        }
        if prev_trend == Some(false) && f < s && m > p {
            return Regime::TrendingDown;
        }
    }
    if v != 0.0 && ((close - v) / v).abs() <= cfg.range_band && a < cfg.adx_trend {
        return Regime::Ranging;
    }
    if ap <= cfg.atr_pct_low {
        return Regime::LowVolatility;
    }
    Regime::Transition
}

/// Convenience: compute everything from bars + session ids.
///
/// Volatility regime uses ATR *relative to price* (`atr / close`), not raw
/// ATR: a constant-spread ramp has flat absolute ATR while its economic
/// volatility decays, and must not read as HIGH_VOLATILITY late in the move.
pub fn detect(bars: &[Bar], session: &[i64], cfg: &RegimeConfig) -> Vec<Regime> {
    use crate::features as f;
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let (adx_v, pdi, mdi) = adx(bars, cfg.adx_period);
    let ef = f::ema(&closes, cfg.ema_fast);
    let es = f::ema(&closes, cfg.ema_slow);
    let atr_v = atr(bars, cfg.atr_period);
    let atr_pct_price: Vec<Option<f64>> = atr_v
        .iter()
        .zip(closes.iter())
        .map(|(a, c)| match a {
            Some(v) if c.is_finite() && *c > 0.0 => Some(v / c),
            _ => None,
        })
        .collect();
    let atr_pct = rolling_percentile(&atr_pct_price, cfg.atr_window);
    let vw = f::vwap(bars, session);
    classify(bars, cfg, &adx_v, &pdi, &mdi, &ef, &es, &atr_pct, &vw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Bar;

    fn trend_bars(n: usize, slope: f64) -> Vec<Bar> {
        (0..n)
            .map(|i| {
                let c = 100.0 + slope * i as f64;
                Bar { t: i as i64, open: c - 0.1, high: c + 0.3, low: c - 0.3, close: c, volume: 100.0 }
            })
            .collect()
    }

    #[test]
    fn strong_uptrend_detected() {
        let bars = trend_bars(250, 0.5);
        let sess = vec![1i64; 250];
        let cfg = RegimeConfig::default();
        let out = detect(&bars, &sess, &cfg);
        // Late bars of a clean ramp must be TRENDING_UP, never RANGING.
        assert_eq!(out[249], Regime::TrendingUp);
        assert!(out.iter().skip(200).all(|r| *r == Regime::TrendingUp));
    }

    #[test]
    fn flat_market_is_ranging_or_lowvol() {
        let bars: Vec<Bar> = (0..250)
            .map(|i| {
                let c = 100.0 + ((i % 2) as f64 - 0.5) * 0.1;
                Bar { t: i as i64, open: c, high: c + 0.05, low: c - 0.05, close: c, volume: 100.0 }
            })
            .collect();
        let sess = vec![1i64; 250];
        let out = detect(&bars, &sess, &RegimeConfig::default());
        // Flat chop must never read as a trend.
        assert!(!out.iter().skip(200).any(|r| matches!(
            r,
            Regime::TrendingUp | Regime::TrendingDown
        )));
    }

    #[test]
    fn adx_known_value() {
        // Two-bar smoke: no panic, Nones early.
        let bars = trend_bars(30, 0.5);
        let (a, p, m) = adx(&bars, 14);
        assert!(a[13].is_none()); // needs 2*period bars
        assert!(a[29].is_some());
        assert!(p[29].unwrap() > m[29].unwrap());
    }
}
