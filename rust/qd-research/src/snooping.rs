//! Data-snooping controls: White's Reality Check (2000) + Hansen's SPA
//! (2005), implemented for real over the strategy-grid OOS score matrix.
//!
//! The question they answer: the grid tests K configs and we showcase the
//! best — is its OOS edge real, or the luckiest of K draws? Both tests
//! bootstrap the null "no rule beats the benchmark" and return p-values.
//! High p = the best result is consistent with luck. Low p = the edge
//! survives the snooping adjustment.
//!
//! Inputs: `diffs[k][t]` = OOS score of grid config `k` on OOS period `t`
//! minus the benchmark (benchmark = 0, i.e. cash — a strategy must beat
//! doing nothing). Periods are CPCV contiguous test runs scored
//! flat-to-flat; they overlap across paths, so they are NOT independent.
//! The stationary bootstrap (Politis–Romano) handles weak dependence via
//! block resampling, but overlapping CPCV windows stretch that assumption —
//! the resulting p-values are approximate, stated as such, and the gate
//! treats them as one confirmation among three (WF + PBO + snooping), not
//! as proof.
//!
//! White RC (non-studentized max):
//!   Vbar = max_k sqrt(T) * dbar_k
//!   Vbar*_b = max_k sqrt(T) * (dbar*_k,b − dbar_k), p = fraction ≥ Vbar.
//! Hansen SPA (studentized, re-centered null):
//!   T_spa = max(0, max_k sqrt(T) * dbar_k / σ̂_k)
//!   null_b = max(0, max_k sqrt(T) * (dbar*_k,b − μ_k) / σ̂_k)
//!   μ^lower = 0, μ^consistent = dbar_k·1{t_k ≥ −√(2 ln ln T)},
//!   μ^upper = min(dbar_k, 0). Guaranteed: p_lower ≤ p_upper and
//!   p_consistent ≤ p_upper (μ^c, μ^l both ≥ μ^u pointwise, so their nulls
//!   are stochastically smaller). NO guaranteed order between p_lower and
//!   p_consistent: for a positive-mean rule μ^c = dbar > 0 = μ^l, so the
//!   consistent null sits below the LF null and p_consistent < p_lower.
//!   (That is the point: the μ=0 null is misspecified when good rules
//!   exist — Hansen prefers the consistent p.) Decisions use p_consistent;
//!   lower/upper are reported as context, not ranked against each other.
//!   σ̂²_k = bootstrap variance of sqrt(T)·dbar* (Hansen's estimator).
//! p-values use add-one smoothing (1+count)/(B+1): never exactly 0.
//!
//! Fail-closed: ragged matrices, T < min_periods, or degenerate inputs
//! return `None` ("INSUFFICIENT DATA"), never a fabricated pass. NaN
//! cannot escape: zero-variance rules get ratio 0 (dbar ≤ 0) or a capped
//! large value (dbar > 0, genuine constant edge — vanishingly rare).

use crate::montecarlo::SplitMix64;

/// Stationary-bootstrap tuning + decision thresholds. Thresholds are module
/// constants where the gate reads them; sim count / block length / seed
/// come from config (reproducibility knobs, not bar-lowering: raising sims
/// only sharpens the p-value).
#[derive(Debug, Clone)]
pub struct SnoopConfig {
    /// Bootstrap resamples (default 5000: K×T resample stats are cheap —
    /// no backtests inside the loop, only means over the score matrix).
    pub boot_sims: usize,
    pub seed: u64,
    /// Mean geometric block length (default 4 ≈ one CPCV-window quarter).
    pub mean_block: usize,
    /// Minimum OOS periods for asymptotics to mean anything (default 10).
    pub min_periods: usize,
    /// Pass bar on the SPA consistent p-value (default 0.05).
    pub alpha: f64,
}

impl Default for SnoopConfig {
    fn default() -> Self {
        Self { boot_sims: 5000, seed: 43, mean_block: 4, min_periods: 10, alpha: 0.05 }
    }
}

/// Full data-snooping outcome: real numbers for the report, Clone for gate.
#[derive(Debug, Clone)]
pub struct SnoopReport {
    pub rules: usize,
    pub periods: usize,
    pub boot_sims: usize,
    pub mean_block: usize,
    pub best_rule: usize,
    pub best_mean: f64,
    pub white_stat: f64,
    pub white_p: f64,
    pub spa_stat: f64,
    pub spa_p_lower: f64,
    pub spa_p_consistent: f64,
    pub spa_p_upper: f64,
    pub assessment: &'static str,
}

/// One stationary-bootstrap index series over `0..n` (circular: blocks wrap
/// past the end — Politis–Romano's circular scheme, valid under
/// stationarity; our CPCV windows only approximate it, see module docs).
pub fn stationary_indices(n: usize, mean_block: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let p = 1.0 / mean_block.max(1) as f64; // restart probability
    let mut idx = rng.range(n);
    for _ in 0..n {
        out.push(idx);
        if rng.next_f64() < p {
            idx = rng.range(n); // new block
        } else {
            idx = (idx + 1) % n; // continue block, circular
        }
    }
    out
}

/// White RC + Hansen SPA over the K×T differential matrix. `None` when the
/// input cannot support the asymptotics (fail-closed, never faked).
pub fn data_snoop(diffs: &[Vec<f64>], cfg: &SnoopConfig) -> Option<SnoopReport> {
    let k = diffs.len();
    if k == 0 {
        return None;
    }
    let t = diffs[0].len();
    if t < cfg.min_periods.max(3) || cfg.boot_sims == 0 {
        return None;
    }
    if diffs.iter().any(|r| r.len() != t || r.iter().any(|v| !v.is_finite())) {
        return None; // ragged or non-finite: refuse, don't trim silently
    }
    let sq = (t as f64).sqrt();
    let means: Vec<f64> = diffs.iter().map(|r| r.iter().sum::<f64>() / t as f64).collect();
    let (mut best_rule, mut best_mean) = (0, means[0]);
    for (i, &m) in means.iter().enumerate().skip(1) {
        if m > best_mean {
            best_rule = i;
            best_mean = m;
        }
    }
    // One bootstrap loop → B×K re-centered means; everything derives.
    let mut rng = SplitMix64(cfg.seed);
    let mut boot_means: Vec<Vec<f64>> = Vec::with_capacity(cfg.boot_sims);
    for _ in 0..cfg.boot_sims {
        let idx = stationary_indices(t, cfg.mean_block, &mut rng);
        let mut row = Vec::with_capacity(k);
        for r in diffs {
            row.push(idx.iter().map(|&j| r[j]).sum::<f64>() / t as f64);
        }
        boot_means.push(row);
    }
    // White: non-studentized max of re-centered means.
    let white_stat = best_mean * sq;
    let mut white_ge = 0usize;
    for b in &boot_means {
        let mut m = f64::NEG_INFINITY;
        for (kk, &v) in b.iter().enumerate() {
            m = m.max((v - means[kk]) * sq);
        }
        if m >= white_stat {
            white_ge += 1;
        }
    }
    let white_p = (white_ge + 1) as f64 / (cfg.boot_sims + 1) as f64;
    // Hansen: bootstrap long-run std per rule.
    let mut sigma = vec![0.0f64; k];
    for kk in 0..k {
        let mb: f64 = boot_means.iter().map(|b| b[kk]).sum::<f64>() / cfg.boot_sims as f64;
        let v: f64 =
            boot_means.iter().map(|b| (b[kk] * sq - mb * sq).powi(2)).sum::<f64>()
                / cfg.boot_sims as f64;
        sigma[kk] = v.sqrt();
    }
    // Studentized ratio with zero-variance guard (no NaN escapes).
    let ratio = |kk: usize, centered: f64| -> f64 {
        if sigma[kk] > 0.0 {
            centered * sq / sigma[kk]
        } else if centered > 0.0 {
            1e6 // constant positive edge: enormous, directionally honest
        } else {
            0.0
        }
    };
    let mut spa_stat: f64 = 0.0;
    for kk in 0..k {
        spa_stat = spa_stat.max(ratio(kk, means[kk]).max(0.0));
    }
    // Re-centering thresholds: consistent uses Hansen's √(2 ln ln T) rule.
    let q = (2.0 * (t as f64).ln().ln()).sqrt();
    let tstat = |kk: usize| ratio(kk, means[kk]);
    let mut ge_l = 0usize;
    let mut ge_c = 0usize;
    let mut ge_u = 0usize;
    for b in &boot_means {
        let (mut nl, mut nc, mut nu) = (0.0f64, 0.0f64, 0.0f64);
        for kk in 0..k {
            let mu_l = 0.0;
            let mu_c = if tstat(kk) >= -q { means[kk] } else { 0.0 };
            let mu_u = means[kk].min(0.0);
            nl = nl.max(ratio(kk, b[kk] - mu_l).max(0.0));
            nc = nc.max(ratio(kk, b[kk] - mu_c).max(0.0));
            nu = nu.max(ratio(kk, b[kk] - mu_u).max(0.0));
        }
        if nl >= spa_stat {
            ge_l += 1;
        }
        if nc >= spa_stat {
            ge_c += 1;
        }
        if nu >= spa_stat {
            ge_u += 1;
        }
    }
    let n1 = (cfg.boot_sims + 1) as f64;
    let (spa_p_lower, spa_p_consistent, spa_p_upper) =
        ((ge_l + 1) as f64 / n1, (ge_c + 1) as f64 / n1, (ge_u + 1) as f64 / n1);
    let assessment = if spa_p_consistent < cfg.alpha && white_p < cfg.alpha {
        "EDGE SURVIVES SNOOPING"
    } else if spa_p_consistent < 0.10 || white_p < 0.10 {
        "MARGINAL (snooping-adjusted)"
    } else {
        "NO EDGE AFTER SNOOPING ADJUSTMENT"
    };
    Some(SnoopReport {
        rules: k,
        periods: t,
        boot_sims: cfg.boot_sims,
        mean_block: cfg.mean_block,
        best_rule,
        best_mean,
        white_stat,
        white_p,
        spa_stat,
        spa_p_lower,
        spa_p_consistent,
        spa_p_upper,
        assessment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(sims: usize, seed: u64) -> SnoopConfig {
        SnoopConfig { boot_sims: sims, seed, ..SnoopConfig::default() }
    }

    #[test]
    fn stationary_indices_deterministic_and_bounded() {
        let (mut a, mut b) = (SplitMix64(7), SplitMix64(7));
        let x = stationary_indices(23, 4, &mut a);
        let y = stationary_indices(23, 4, &mut b);
        assert_eq!(x, y);
        assert_eq!(x.len(), 23);
        assert!(x.iter().all(|&i| i < 23));
        // mean_block=1 → every draw restarts (p=1): still bounded, still
        // deterministic across identical seeds.
        let z = stationary_indices(23, 1, &mut SplitMix64(7));
        assert_eq!(z.len(), 23);
        assert!(z.iter().all(|&i| i < 23));
        assert!(stationary_indices(0, 4, &mut SplitMix64(1)).is_empty());
    }

    #[test]
    fn all_zero_differentials_give_p_one() {
        // No rule ever beats cash: Vbar=0, every Vbar*=0 → p exactly 1.
        let d = vec![vec![0.0; 20]; 5];
        let r = data_snoop(&d, &cfg(500, 1)).expect("computable");
        assert_eq!(r.white_p, 1.0);
        assert_eq!(r.spa_p_lower, 1.0);
        assert_eq!(r.assessment, "NO EDGE AFTER SNOOPING ADJUSTMENT");
    }

    #[test]
    fn constant_edge_is_detected() {
        // Every rule +1% every period: overwhelming evidence, both tests
        // must clear 5% (fixed seed → deterministic verdict).
        let d = vec![vec![0.01; 20]; 4];
        let r = data_snoop(&d, &cfg(500, 1)).expect("computable");
        assert!(r.white_p < 0.05, "white p={}", r.white_p);
        assert!(r.spa_p_consistent < 0.05, "spa p={}", r.spa_p_consistent);
        assert_eq!(r.assessment, "EDGE SURVIVES SNOOPING");
    }

    #[test]
    fn spa_p_values_bounded_by_upper() {
        // Guaranteed orders: lower ≤ upper and consistent ≤ upper (both
        // nulls re-center weakly above μ^u, so their nulls sit below the
        // upper null). NO order between lower and consistent: for a
        // positive-mean rule μ^c = dbar > 0 = μ^l, so p_consistent can sit
        // below p_lower — that is Hansen's design, not a bug.
        let mut rng = SplitMix64(99);
        let mut d = vec![vec![0.0; 25]; 6];
        for row in d.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.next_f64() * 0.04 - 0.02;
            }
        }
        let r = data_snoop(&d, &cfg(500, 2)).expect("computable");
        assert!(r.spa_p_lower <= r.spa_p_upper, "{} > {}", r.spa_p_lower, r.spa_p_upper);
        assert!(r.spa_p_consistent <= r.spa_p_upper, "{} > {}", r.spa_p_consistent, r.spa_p_upper);
        for p in [r.white_p, r.spa_p_lower, r.spa_p_consistent, r.spa_p_upper] {
            assert!((0.0..=1.0).contains(&p), "p out of range: {p}");
        }
    }

    #[test]
    fn insufficient_or_bad_input_returns_none() {
        let c = cfg(500, 1);
        assert!(data_snoop(&[], &c).is_none()); // no rules
        assert!(data_snoop(&[vec![0.01; 5]], &c).is_none()); // T < min_periods
        // Ragged matrix: refuse, don't trim.
        assert!(data_snoop(&[vec![0.01; 20], vec![0.01; 19]], &c).is_none());
        // Non-finite: refuse.
        let mut bad = vec![vec![0.01; 20]; 2];
        bad[1][7] = f64::NAN;
        assert!(data_snoop(&bad, &c).is_none());
        // Zero sims: refuse.
        assert!(data_snoop(&vec![vec![0.01; 20]; 2], &cfg(0, 1)).is_none());
    }

    #[test]
    fn single_lucky_rule_does_not_survive() {
        // One rule spikes once (+50% in a single period, flat else),
        // five rules total: the max is luck, the tests must say so.
        let mut d = vec![vec![0.0; 20]; 5];
        d[2][10] = 0.50;
        let r = data_snoop(&d, &cfg(2000, 3)).expect("computable");
        assert_eq!(r.best_rule, 2);
        assert!(r.spa_p_consistent >= 0.05, "lucky spike survived: {}", r.spa_p_consistent);
    }
}
