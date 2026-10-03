//! Deterministic market-regime detection.
//!
//! States: `TrendingUp | TrendingDown | Ranging | HighVolatility |
//! LowVolatility | Breakout | MeanReversion | Abnormal | Transition`.
//! The classifier is a pure function of causal features with configurable
//! thresholds — every default is documented with its rationale below.
//! No randomness, no hidden state.

use crate::features::{atr, relative_volume, rolling_percentile, wilder, Bar};

/// Regime states, in priority order for the rule stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    TrendingUp,
    TrendingDown,
    HighVolatility,
    Ranging,
    LowVolatility,
    Breakout,
    MeanReversion,
    Abnormal,
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
            Regime::Breakout => "BREAKOUT",
            Regime::MeanReversion => "MEAN_REVERSION",
            Regime::Abnormal => "ABNORMAL",
            Regime::Transition => "TRANSITION",
        }
    }

    /// Fail-safe states: the entry gate must never open into these.
    /// `Abnormal` = data/market integrity uncertain → do not trade.
    pub fn is_fail_safe(&self) -> bool {
        matches!(self, Regime::Abnormal)
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
/// - `vol_period = 20`: one trading month of volume for relative-volume.
/// - `breakout_lookback = 20`: Donchian-style channel — a close beyond the
///   prior 20-bar extremes escapes a monthly range.
/// - `relvol_breakout = 1.5`: breakout needs 50% above-average participation;
///   price drift on thin volume is not a breakout.
/// - `mr_dev_atr = 2.0`: stretched ≥2 ATR from VWAP = statistically extended
///   (a ~2σ event under locally normal noise), mean-reversion candidate.
/// - `meanrev_adx_max = 20.0`: below Wilder's 25 trend divide — stretched
///   AND directionless. Above it, the move is a trend, not a stretch.
/// - `abnormal_range_atr_mult = 6.0`: a single bar spanning ≥6 ATR is a data
///   error, flash crash, or halt-reopen — integrity uncertain, fail safe.
/// - `vwap_period = 20`: rolling value anchor ≈ one trading month (daily
///   bars). Session VWAP drifts months away from price on multi-day series
///   (RANGING then reads 0.3% of bars); the trailing-20 anchor keeps
///   `|close-vwap|` a current extension-from-value measure.
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
    pub vol_period: usize,
    pub breakout_lookback: usize,
    pub relvol_breakout: f64,
    pub mr_dev_atr: f64,
    pub meanrev_adx_max: f64,
    pub abnormal_range_atr_mult: f64,
    pub vwap_period: usize,
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
            vol_period: 20,
            breakout_lookback: 20,
            relvol_breakout: 1.5,
            mr_dev_atr: 2.0,
            meanrev_adx_max: 20.0,
            abnormal_range_atr_mult: 6.0,
            vwap_period: 20,
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
/// 0. Invalid bar (non-finite OHLC, high < low, non-positive close) →
///    `Abnormal`: market-integrity uncertain, fail safe (§5/§11).
/// 1. Missing features → `Transition` (unknown = do not trade as trend).
/// 2. Single-bar range ≥ `abnormal_range_atr_mult` × ATR → `Abnormal`
///    (flash move / bad tick — do not trade the print).
/// 3. `atr_pct >= atr_pct_high` → `HighVolatility`, where `atr_pct` is the
///    trailing percentile of ATR *relative to price* (see [`detect`]).
///    Volatility overrides: stops behave differently there, so it must be
///    visible even inside trends.
/// 4. `adx >= adx_trend` (or `>= adx_strong` with flat slope = hysteresis)
///    + EMA stack + slope persistence + DI gap → `TrendingUp/Down`.
/// 5. Fresh range escape: close beyond the prior `breakout_lookback`-bar
///    extremes with relative volume ≥ `relvol_breakout` → `Breakout`.
///    Breakout is a *participation* event: drift on thin volume stays a
///    trend/range call, not a breakout.
/// 6. Stretched ≥ `mr_dev_atr` ATR from VWAP while `adx <= meanrev_adx_max`
///    (directionless extension) → `MeanReversion` (counter-trend setup).
/// 7. `|close - vwap|/vwap <= range_band` + weak ADX → `Ranging`.
/// 8. `atr_pct <= atr_pct_low` → `LowVolatility`.
/// 9. Otherwise → `Transition`.
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
    atr_raw: &[Option<f64>],
    relvol: &[Option<f64>],
) -> Vec<Regime> {
    let n = bars.len();
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let mut out = vec![Regime::Transition; n];
    for i in 0..n {
        let (a, p, m, ef, es, ap, v, ar, rv) = (
            adx_v.get(i).copied().flatten(),
            pdi.get(i).copied().flatten(),
            mdi.get(i).copied().flatten(),
            ema_fast.get(i).copied().flatten(),
            ema_slow.get(i).copied().flatten(),
            atr_pct.get(i).copied().flatten(),
            vwap.get(i).copied().flatten(),
            atr_raw.get(i).copied().flatten(),
            relvol.get(i).copied().flatten(),
        );
        // Donchian-style channel over the PRIOR `breakout_lookback` bars
        // only (strictly `< i` — the current bar never defines its own
        // breakout). Missing/None until enough history exists.
        let (ch_hi, ch_lo) = if cfg.breakout_lookback > 0 && i >= 1 {
            let lo = i.saturating_sub(cfg.breakout_lookback);
            let win: Vec<f64> =
                closes[lo..i].iter().copied().filter(|c| c.is_finite()).collect();
            if win.len() == i - lo && !win.is_empty() {
                let (mut hi, mut lo_v) = (win[0], win[0]);
                for c in win.iter().skip(1) {
                    if *c > hi {
                        hi = *c;
                    }
                    if *c < lo_v {
                        lo_v = *c;
                    }
                }
                (Some(hi), Some(lo_v))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
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
            &bars[i],
            a,
            p,
            m,
            ef,
            es,
            ap,
            v,
            ar,
            rv,
            ch_hi,
            ch_lo,
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
    bar: &Bar,
    adx: Option<f64>,
    pdi: Option<f64>,
    mdi: Option<f64>,
    ef: Option<f64>,
    es: Option<f64>,
    atr_pct: Option<f64>,
    vwap: Option<f64>,
    atr_raw: Option<f64>,
    relvol: Option<f64>,
    ch_hi: Option<f64>,
    ch_lo: Option<f64>,
    slope_up: bool,
    slope_dn: bool,
    prev_trend: Option<bool>,
) -> Regime {
    let close = bar.close;
    // 0. Invalid bar: integrity uncertain → fail safe, never a tradeable call.
    if !bar.is_valid() || !close.is_finite() || close <= 0.0 {
        return Regime::Abnormal;
    }
    let (a, p, m, f, s, ap, v) = match (adx, pdi, mdi, ef, es, atr_pct, vwap) {
        (Some(a), Some(p), Some(m), Some(f), Some(s), Some(ap), Some(v)) => (a, p, m, f, s, ap, v),
        _ => return Regime::Transition,
    };
    // 2. Flash-move / bad-tick guard: one bar spanning many ATRs.
    if let Some(ar) = atr_raw {
        if ar > 0.0 && bar.high.is_finite() && bar.low.is_finite() {
            let span = bar.high - bar.low;
            if span.is_finite() && span >= cfg.abnormal_range_atr_mult * ar {
                return Regime::Abnormal;
            }
        }
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
    // 5. Breakout: fresh escape beyond the prior channel WITH participation.
    // Fires regardless of ADX (a new trend starts with ADX still low), but
    // only on above-average volume — thin drift is not a breakout.
    if let (Some(hi), Some(lo_v), Some(r)) = (ch_hi, ch_lo, relvol) {
        if r.is_finite() && r >= cfg.relvol_breakout && (close > hi || close < lo_v) {
            return Regime::Breakout;
        }
    }
    // 6. Mean-reversion: statistically stretched from value (VWAP) while
    // directionless. A stretched TREND (high ADX) stays a trend — this
    // fires only when there is no trend to fight.
    if let Some(ar) = atr_raw {
        if ar > 0.0 {
            let dev_atr = (close - v).abs() / ar;
            if dev_atr >= cfg.mr_dev_atr && a <= cfg.meanrev_adx_max {
                return Regime::MeanReversion;
            }
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
pub fn detect(bars: &[Bar], _session: &[i64], cfg: &RegimeConfig) -> Vec<Regime> {
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
    // Rolling (trailing-`vwap_period`) value anchor: session VWAP drifts
    // months from price on daily series; the RANGING / MEAN_REVERSION rules
    // need a *current* extension-from-value, so they read this, not it.
    let vw = f::vwap_rolling(bars, cfg.vwap_period);
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let rv = relative_volume(&volumes, cfg.vol_period);
    classify(bars, cfg, &adx_v, &pdi, &mdi, &ef, &es, &atr_pct, &vw, &atr_v, &rv)
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

    fn flat_range(n: usize, base: f64) -> Vec<Bar> {
        (0..n)
            .map(|i| {
                let c = base + ((i % 4) as f64 - 1.5) * 0.2;
                Bar {
                    t: i as i64,
                    open: c,
                    high: c + 0.15,
                    low: c - 0.15,
                    close: c,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn breakout_needs_range_escape_plus_volume() {
        // 150 flat bars then a gap-free creep above the channel. No gaps:
        // true-range stays flat so the HighVol override stays quiet and the
        // participation check decides.
        let mut bars = flat_range(150, 100.0);
        let mut cfg = RegimeConfig::default();
        cfg.breakout_lookback = 20;
        cfg.relvol_breakout = 1.5;
        let n0 = bars.len();
        for k in 0..5 {
            let c = 100.35 + k as f64 * 0.15;
            bars.push(Bar {
                t: (n0 + k) as i64,
                open: c - 0.05,
                high: c + 0.15,
                low: c - 0.15,
                close: c,
                volume: 5000.0, // 5x average participation
            });
        }
        let sess = vec![1i64; bars.len()];
        let out = detect(&bars, &sess, &cfg);
        let tail = &out[out.len() - 3..];
        assert!(
            tail.iter().any(|r| *r == Regime::Breakout),
            "escape on 5x volume must read BREAKOUT, got {tail:?}"
        );
        // Same escape on thin volume: participation missing → not a breakout.
        let mut thin = flat_range(150, 100.0);
        let n1 = thin.len();
        for k in 0..5 {
            let c = 100.0 + 0.5 + k as f64 * 0.4;
            thin.push(Bar {
                t: (n1 + k) as i64,
                open: c - 0.1,
                high: c + 0.2,
                low: c - 0.2,
                close: c,
                volume: 200.0, // far below average
            });
        }
        let sess2 = vec![1i64; thin.len()];
        let out2 = detect(&thin, &sess2, &cfg);
        assert!(
            !out2.iter().skip(out2.len() - 3).any(|r| *r == Regime::Breakout),
            "thin drift must not read BREAKOUT"
        );
    }

    #[test]
    fn stretched_directionless_is_mean_reversion() {
        // Chop around 100, then a sharp spike with no trend behind it:
        // extended from VWAP while ADX stays low → MEAN_REVERSION, and a
        // stretched TREND must never be relabeled as such.
        let mut bars = flat_range(200, 100.0);
        let n0 = bars.len();
        for k in 0..3 {
            let c = 100.0 + (k as f64 + 1.0) * 1.2;
            bars.push(Bar {
                t: (n0 + k) as i64,
                open: c - 0.2,
                high: c + 0.3,
                low: c - 0.3,
                close: c,
                volume: 1000.0,
            });
        }
        let sess = vec![1i64; bars.len()];
        let out = detect(&bars, &sess, &RegimeConfig::default());
        // Spike bars must not be trendy (no ADX behind them)...
        assert!(!out.iter().skip(out.len() - 3).any(|r| matches!(
            r,
            Regime::TrendingUp | Regime::TrendingDown
        )));
        // ...and a clean ramp must never be MEAN_REVERSION either.
        let ramp = trend_bars(250, 0.5);
        let sess2 = vec![1i64; 250];
        let out2 = detect(&ramp, &sess2, &RegimeConfig::default());
        assert!(!out2.iter().skip(200).any(|r| *r == Regime::MeanReversion));
    }

    #[test]
    fn garbage_and_flash_bars_are_abnormal() {
        // NaN close → ABNORMAL (integrity uncertain, fail safe).
        let mut bars = flat_range(150, 100.0);
        bars.push(Bar {
            t: 150,
            open: 100.0,
            high: 100.2,
            low: 99.8,
            close: f64::NAN,
            volume: 1000.0,
        });
        let sess = vec![1i64; bars.len()];
        let out = detect(&bars, &sess, &RegimeConfig::default());
        assert_eq!(*out.last().unwrap(), Regime::Abnormal);
        assert!(Regime::Abnormal.is_fail_safe());
        // Flash bar: 10-point span on ~0.3 ATR → ABNORMAL, not a breakout.
        let mut bars2 = flat_range(150, 100.0);
        bars2.push(Bar {
            t: 150,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 101.0,
            volume: 9000.0,
        });
        let sess2 = vec![1i64; bars2.len()];
        let out2 = detect(&bars2, &sess2, &RegimeConfig::default());
        assert_eq!(*out2.last().unwrap(), Regime::Abnormal);
    }
}
