//! Reference signal strategies over the feature/regime stack.
//!
//! These exist to exercise the full pipeline deterministically — they are
//! starting points for research, not recommendations. Every signal consumes
//! only data `<= i` (index-bounded slices, enforced by the function
//! signatures taking full series + the decision index).

use crate::backtest::Signal;
use crate::features;
use crate::regime::Regime;

/// EMA-cross trend follower: long when `fast > slow`, short when
/// `fast < slow`, filtered by regime. Trades only where momentum has a
/// thesis: TRENDING_*, TRANSITION (early trend), and BREAKOUT (fresh
/// escape in the signal's direction). Flat everywhere else — RANGING
/// (chop), HIGH_VOLATILITY (stops misbehave), LOW_VOLATILITY (no edge),
/// MEAN_REVERSION (counter-trend territory), ABNORMAL (fail safe, §11).
pub struct EmaCrossTrend {
    pub fast: usize,
    pub slow: usize,
}

impl EmaCrossTrend {
    pub fn signals(
        &self,
        closes: &[f64],
        regimes: &[Regime],
    ) -> Vec<Signal> {
        let f = features::ema(closes, self.fast);
        let s = features::ema(closes, self.slow);
        closes
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let reg = regimes.get(i).copied().unwrap_or(Regime::Transition);
                match reg {
                    Regime::TrendingUp
                    | Regime::TrendingDown
                    | Regime::Transition
                    | Regime::Breakout => {}
                    _ => return Signal::Flat,
                }
                match (f.get(i).copied().flatten(), s.get(i).copied().flatten()) {
                    (Some(a), Some(b)) if a > b => Signal::Long,
                    (Some(a), Some(b)) if a < b => Signal::Short,
                    _ => Signal::Flat,
                }
            })
            .collect()
    }
}

/// RSI mean-reversion: long when RSI < `oversold`, flat when RSI >
/// `overbought` (exits), shorts disabled by default (`allow_shorts`).
/// Gated to RANGING / TRANSITION / MEAN_REVERSION only — counter-trend
/// systems must not fight trends, chase breakouts, or trade garbage:
/// TRENDING_*, BREAKOUT, HIGH_VOLATILITY, LOW_VOLATILITY, and ABNORMAL
/// all force flat.
pub struct RsiMeanReversion {
    pub period: usize,
    pub oversold: f64,
    pub overbought: f64,
    pub allow_shorts: bool,
}

impl RsiMeanReversion {
    pub fn signals(&self, closes: &[f64], regimes: &[Regime]) -> Vec<Signal> {
        let rsi = qd_engine::indicators::compute_rsi_wilder(closes, self.period);
        closes
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let reg = regimes.get(i).copied().unwrap_or(Regime::Transition);
                match reg {
                    Regime::Ranging | Regime::Transition | Regime::MeanReversion => {}
                    _ => return Signal::Flat,
                }
                match rsi.get(i).copied().flatten() {
                    Some(v) if v < self.oversold => Signal::Long,
                    Some(v) if v > self.overbought && self.allow_shorts => Signal::Short,
                    Some(v) if v > self.overbought => Signal::Flat,
                    _ => Signal::Flat,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_cross_follows_sustained_move() {
        let closes: Vec<f64> = (0..120).map(|i| 100.0 + i as f64 * 0.5).collect();
        let regs = vec![Regime::TrendingUp; 120];
        let sig = EmaCrossTrend { fast: 5, slow: 20 }.signals(&closes, &regs);
        assert_eq!(sig[119], Signal::Long);
        // Ranging gate forces flat even with a cross present.
        let rg = vec![Regime::Ranging; 120];
        let sig2 = EmaCrossTrend { fast: 5, slow: 20 }.signals(&closes, &rg);
        assert!(sig2.iter().all(|s| *s == Signal::Flat));
    }

    #[test]
    fn rsi_buys_the_dip() {
        // Sharp drop then flat: RSI must dip below 30 mid-fall.
        let mut closes: Vec<f64> = vec![100.0; 30];
        for i in 0..10 {
            closes.push(100.0 - (i + 1) as f64 * 3.0);
        }
        let regs = vec![Regime::Ranging; closes.len()];
        let sig = RsiMeanReversion { period: 14, oversold: 30.0, overbought: 70.0, allow_shorts: false }
            .signals(&closes, &regs);
        assert!(sig.iter().any(|s| *s == Signal::Long));
        assert!(!sig.iter().any(|s| *s == Signal::Short));
    }

    #[test]
    fn nine_state_gates_are_conservative() {
        // Trend system may ride trends and genuine breakouts — never chop,
        // counter-trend stretches, or garbage.
        let closes: Vec<f64> = (0..120).map(|i| 100.0 + i as f64 * 0.5).collect();
        let ema = EmaCrossTrend { fast: 5, slow: 20 };
        for ok in [Regime::TrendingUp, Regime::Breakout, Regime::Transition] {
            let regs = vec![ok; 120];
            assert_eq!(ema.signals(&closes, &regs)[119], Signal::Long, "{ok:?} must pass");
        }
        for bad in [
            Regime::Ranging,
            Regime::HighVolatility,
            Regime::LowVolatility,
            Regime::MeanReversion,
            Regime::Abnormal,
        ] {
            let regs = vec![bad; 120];
            assert!(
                ema.signals(&closes, &regs).iter().all(|s| *s == Signal::Flat),
                "{bad:?} must force flat"
            );
        }
        // Counter-trend system trades its home turf only.
        let mut dip: Vec<f64> = vec![100.0; 30];
        for i in 0..10 {
            dip.push(100.0 - (i + 1) as f64 * 3.0);
        }
        let rsi = RsiMeanReversion { period: 14, oversold: 30.0, overbought: 70.0, allow_shorts: false };
        for ok in [Regime::Ranging, Regime::MeanReversion, Regime::Transition] {
            let regs = vec![ok; dip.len()];
            assert!(
                rsi.signals(&dip, &regs).iter().any(|s| *s == Signal::Long),
                "{ok:?} must allow dip-buying"
            );
        }
        for bad in [
            Regime::TrendingUp,
            Regime::TrendingDown,
            Regime::Breakout,
            Regime::HighVolatility,
            Regime::LowVolatility,
            Regime::Abnormal,
        ] {
            let regs = vec![bad; dip.len()];
            assert!(
                rsi.signals(&dip, &regs).iter().all(|s| *s == Signal::Flat),
                "{bad:?} must force flat"
            );
        }
    }
}
