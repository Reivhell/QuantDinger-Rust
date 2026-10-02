//! Deterministic randomness (SplitMix64) + Monte Carlo / bootstrap on
//! completed trades.
//!
//! All stochastic entry points take an explicit `seed: u64`. Same seed +
//! same inputs = identical outputs, always. No thread-local RNG, no
//! system entropy.

/// SplitMix64: fast, high-quality, fully deterministic. Used for every
/// shuffle / resample in this crate.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa → [0, 1).
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Fisher–Yates shuffle with the crate RNG. O(n).
pub fn shuffle<T>(xs: &mut [T], rng: &mut SplitMix64) {
    for i in (1..xs.len()).rev() {
        let j = rng.range(i + 1);
        xs.swap(i, j);
    }
}

/// One Monte Carlo path statistic set, computed on an ordered return
/// sequence (fractions of starting equity).
#[derive(Debug, Clone, Default)]
pub struct PathStats {
    pub total_return: f64,
    pub max_drawdown: f64,
    pub sharpe_per_trade: f64,
    pub longest_loss_streak: usize,
    /// Bars (trades) from the max-DD trough back to breakeven; `None` if
    /// the path never recovers within its own length.
    pub recovery_trades: Option<usize>,
}

pub fn path_stats(ordered_rets: &[f64]) -> PathStats {
    let n = ordered_rets.len();
    let mut eq = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    let mut trough_at = 0usize;
    let mut rec: Option<usize> = None;
    let mut cl = 0usize;
    let mut bl = 0usize;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for (i, &r) in ordered_rets.iter().enumerate() {
        eq *= 1.0 + r;
        sum += r;
        sum2 += r * r;
        if eq > peak {
            peak = eq;
            if rec.is_none() && max_dd > 0.0 {
                rec = Some(i - trough_at);
            }
        }
        let dd = if peak > 0.0 { (peak - eq) / peak } else { 0.0 };
        if dd > max_dd {
            max_dd = dd;
            trough_at = i;
            rec = None; // deeper trough resets the recovery clock
        }
        if r <= 0.0 {
            cl += 1;
            if cl > bl {
                bl = cl;
            }
        } else {
            cl = 0;
        }
    }
    let mean = if n > 0 { sum / n as f64 } else { 0.0 };
    let var = if n > 0 { (sum2 / n as f64 - mean * mean).max(0.0) } else { 0.0 };
    PathStats {
        total_return: eq - 1.0,
        max_drawdown: max_dd,
        sharpe_per_trade: if var > 0.0 { mean / var.sqrt() } else { 0.0 },
        longest_loss_streak: bl,
        recovery_trades: rec,
    }
}

/// Distribution summary: median + quartiles + tails + extremes.
/// `min`/`max` are plain extrema (for drawdowns the "worst case" is `max`,
/// for returns it is `min` — interpret per metric, see [`report`]).
#[derive(Debug, Clone)]
pub struct DistSummary {
    pub median: f64,
    pub p5: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

pub fn summarize(mut xs: Vec<f64>) -> DistSummary {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        if xs.is_empty() {
            return 0.0;
        }
        let pos = p * (xs.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        xs[lo] + (xs[hi] - xs[lo]) * (pos - lo as f64)
    };
    DistSummary {
        median: q(0.5),
        p5: q(0.05),
        p25: q(0.25),
        p75: q(0.75),
        p95: q(0.95),
        min: xs.first().copied().unwrap_or(0.0),
        max: xs.last().copied().unwrap_or(0.0),
        count: xs.len(),
    }
}

/// Monte Carlo report over trade-order randomization AND bootstrap
/// resampling (both, separately — they answer different questions:
/// ordering luck vs sampling luck).
#[derive(Debug, Clone)]
pub struct MonteCarloReport {
    pub simulations: usize,
    pub seed: u64,
    pub shuffled_return: DistSummary,
    pub shuffled_max_dd: DistSummary,
    pub shuffled_longest_loss: DistSummary,
    pub shuffled_recovery: DistSummary,
    pub boot_return: DistSummary,
    pub boot_max_dd: DistSummary,
    pub boot_sharpe: DistSummary,
}

/// Run `simulations` shuffles (trade-order randomization) and
/// `simulations` bootstraps (sample with replacement, same length).
/// Returns are per-trade fractions of starting equity.
///
/// Note: under compounding, shuffled total return and per-trade Sharpe are
/// order-invariant (multiplication commutes) — they act as a determinism
/// self-check. Ordering luck shows up in max drawdown, longest losing
/// streak and recovery time. Bootstrap varies the multiset, so all of its
/// distributions are informative.
pub fn run_monte_carlo(returns: &[f64], simulations: usize, seed: u64) -> MonteCarloReport {
    let n = returns.len();
    let mut rng = SplitMix64(seed);
    let (mut sr, mut sd, mut sl, mut src) = (vec![], vec![], vec![], vec![]);
    let (mut br, mut bd, mut bs) = (vec![], vec![], vec![]);
    for _ in 0..simulations {
        // Shuffle: preserve the distribution, destroy the order.
        let mut order: Vec<f64> = returns.to_vec();
        shuffle(&mut order, &mut rng);
        let st = path_stats(&order);
        sr.push(st.total_return);
        sd.push(st.max_drawdown);
        sl.push(st.longest_loss_streak as f64);
        src.push(st.recovery_trades.map(|v| v as f64).unwrap_or(n as f64));
        // Bootstrap: sample with replacement.
        let mut sample = Vec::with_capacity(n);
        for _ in 0..n {
            sample.push(returns[rng.range(n.max(1))]);
        }
        let bt = path_stats(&sample);
        br.push(bt.total_return);
        bd.push(bt.max_drawdown);
        bs.push(bt.sharpe_per_trade);
    }
    MonteCarloReport {
        simulations,
        seed,
        shuffled_return: summarize(sr),
        shuffled_max_dd: summarize(sd),
        shuffled_longest_loss: summarize(sl),
        shuffled_recovery: summarize(src),
        boot_return: summarize(br),
        boot_max_dd: summarize(bd),
        boot_sharpe: summarize(bs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = SplitMix64(42);
        let mut b = SplitMix64(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = SplitMix64(43);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn shuffle_preserves_multiset() {
        let mut rng = SplitMix64(7);
        let mut xs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        shuffle(&mut xs, &mut rng);
        let mut s = xs.clone();
        s.sort();
        assert_eq!(s, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // A second seed shuffles differently (overwhelmingly likely).
        let mut rng2 = SplitMix64(8);
        let mut ys = vec![1, 2, 3, 4, 5, 6, 7, 8];
        shuffle(&mut ys, &mut rng2);
        assert_ne!(xs, ys);
    }

    #[test]
    fn path_stats_known_path() {
        // +10%, -20%, +10%: eq 1 → 1.1 → 0.88 → 0.968; MDD = (1.1-0.88)/1.1.
        let st = path_stats(&[0.10, -0.20, 0.10]);
        assert!((st.total_return - (-0.032)).abs() < 1e-12);
        assert!((st.max_drawdown - 0.2).abs() < 1e-12);
        assert_eq!(st.longest_loss_streak, 1);
    }

    #[test]
    fn shuffle_preserves_compounded_return() {
        // Order-invariant under compounding: every shuffle must reproduce
        // the realized total return exactly (determinism self-check).
        let rets = vec![0.02, -0.01, 0.03, -0.02, 0.01, -0.005, 0.015, -0.01];
        let realized = path_stats(&rets).total_return;
        let r = run_monte_carlo(&rets, 50, 42);
        assert!((r.shuffled_return.median - realized).abs() < 1e-12);
        assert!((r.shuffled_return.p5 - realized).abs() < 1e-12);
        // ...while drawdown DOES vary with order.
        assert!(r.shuffled_max_dd.max >= r.shuffled_max_dd.median);
    }

    #[test]
    fn mc_report_seed_stability() {
        let rets = vec![0.02, -0.01, 0.03, -0.02, 0.01, -0.005, 0.015, -0.01];
        let r1 = run_monte_carlo(&rets, 200, 42);
        let r2 = run_monte_carlo(&rets, 200, 42);
        assert_eq!(r1.shuffled_return.median, r2.shuffled_return.median);
        assert_eq!(r1.shuffled_return.count, 200);
        assert!(r1.shuffled_max_dd.max >= r1.shuffled_max_dd.median);
        assert!(r1.boot_return.p5 <= r1.boot_return.median);
    }
}
