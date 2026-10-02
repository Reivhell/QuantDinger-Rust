//! Backtest-overfitting detection: PBO estimate, degradation accounting,
//! and configuration bookkeeping.
//!
//! Method (Bailey et al., "Pseudo-Mathematics and Financial Charlatanism",
//! 2014; López de Prado, 2018 ch. 11): run `S` strategy configurations
//! through CPCV-style IS/OOS splits; for each split rank configs by IS
//! performance, take the IS-best, and record its OOS rank as a logit
//! `λ = ln(rank/(1-rank))`-style. PBO = fraction of splits where the
//! IS-selected config underperforms the median OOS — i.e. `P(OOS rank of
//! IS-best < median)`. This implementation works directly on caller
//! supplied IS/OOS score matrices (config × split), so any scored
//! parameter grid can be assessed.
//!
//! Assessment bands (documented, configurable via [`PboBands`]):
//! - PBO < 0.2 → LOW, 0.2–0.5 → MODERATE, > 0.5 → HIGH overfitting risk.
//! - IS→OOS degradation > 50% of the IS edge always escalates one level.

use crate::montecarlo::summarize;

/// IS/OOS scores: `is[c][s]`, `oos[c][s]` for config `c`, split `s`.
#[derive(Debug, Clone)]
pub struct ScoreMatrix {
    pub configs: Vec<String>,
    pub is: Vec<Vec<f64>>,
    pub oos: Vec<Vec<f64>>,
}

/// PBO + degradation report.
#[derive(Debug, Clone)]
pub struct PboReport {
    pub configs_tested: usize,
    pub splits: usize,
    /// P(PBO): fraction of splits where IS-best lands below median OOS.
    pub pbo: f64,
    pub best_is: f64,
    pub median_is: f64,
    pub median_oos_of_is_best: f64,
    pub median_oos_overall: f64,
    /// 1 - median_oos(is-best)/best_is, clipped to [0, 1].
    pub degradation: f64,
    pub assessment: &'static str,
}

/// Assessment bands. Defaults: LOW < 0.2 ≤ MODERATE ≤ 0.5 < HIGH; any
/// degradation above `degradation_escalate` (default 0.5) escalates one band.
#[derive(Debug, Clone)]
pub struct PboBands {
    pub low: f64,
    pub high: f64,
    pub degradation_escalate: f64,
}

impl Default for PboBands {
    fn default() -> Self {
        Self { low: 0.2, high: 0.5, degradation_escalate: 0.5 }
    }
}

pub fn analyze_overfitting(scores: &ScoreMatrix, bands: &PboBands) -> PboReport {
    let c = scores.configs.len();
    let s = scores.is.first().map(|r| r.len()).unwrap_or(0);
    if c == 0 || s == 0 {
        return PboReport {
            configs_tested: 0,
            splits: 0,
            pbo: 1.0,
            best_is: 0.0,
            median_is: 0.0,
            median_oos_of_is_best: 0.0,
            median_oos_overall: 0.0,
            degradation: 1.0,
            assessment: "HIGH OVERFITTING RISK",
        };
    }
    // Per-split: IS-best config → its OOS rank (fraction below median).
    let mut underperform = 0usize;
    let mut is_best_oos: Vec<f64> = Vec::with_capacity(s);
    for split in 0..s {
        let mut best_c = 0;
        for cfg in 1..c {
            if scores.is[cfg][split] > scores.is[best_c][split] {
                best_c = cfg;
            }
        }
        let oos_best = scores.oos[best_c][split];
        is_best_oos.push(oos_best);
        let below_or_eq = (0..c).filter(|&k| scores.oos[k][split] <= oos_best).count();
        // Below median ⇔ rank position in the bottom half.
        if (below_or_eq as f64) <= (c as f64 / 2.0) {
            underperform += 1;
        }
    }
    let pbo = underperform as f64 / s as f64;
    // Best/median IS across configs (median over splits of per-split values).
    let mut per_split_best_is = Vec::with_capacity(s);
    let mut per_split_med_is = Vec::with_capacity(s);
    let mut per_split_med_oos = Vec::with_capacity(s);
    for split in 0..s {
        let mut col_is: Vec<f64> = (0..c).map(|k| scores.is[k][split]).collect();
        let mut col_oos: Vec<f64> = (0..c).map(|k| scores.oos[k][split]).collect();
        col_is.sort_by(|a, b| a.partial_cmp(b).unwrap());
        col_oos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        per_split_best_is.push(*col_is.last().unwrap());
        per_split_med_is.push(col_is[c / 2]);
        per_split_med_oos.push(col_oos[c / 2]);
    }
    let med = |v: Vec<f64>| summarize(v).median;
    let best_is = med(per_split_best_is);
    let median_is = med(per_split_med_is);
    let median_oos_overall = med(per_split_med_oos);
    let median_oos_best = med(is_best_oos);
    let degradation = if best_is.abs() > 1e-12 {
        ((best_is - median_oos_best) / best_is.abs()).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let mut level = if pbo < bands.low {
        0
    } else if pbo <= bands.high {
        1
    } else {
        2
    };
    if degradation > bands.degradation_escalate && level < 2 {
        level += 1;
    }
    PboReport {
        configs_tested: c,
        splits: s,
        pbo,
        best_is,
        median_is,
        median_oos_of_is_best: median_oos_best,
        median_oos_overall,
        degradation,
        assessment: match level {
            0 => "LOW OVERFITTING RISK",
            1 => "MODERATE OVERFITTING RISK",
            _ => "HIGH OVERFITTING RISK",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(is: Vec<Vec<f64>>, oos: Vec<Vec<f64>>) -> ScoreMatrix {
        ScoreMatrix {
            configs: (0..is.len()).map(|i| format!("c{i}")).collect(),
            is,
            oos,
        }
    }

    #[test]
    fn persistent_edge_scores_low_pbo() {
        // Config 0 always best IS and always best OOS → PBO 0.
        let m = matrix(
            vec![vec![2.0, 2.1, 1.9], vec![1.0, 1.0, 1.0], vec![0.5, 0.4, 0.6]],
            vec![vec![1.5, 1.6, 1.4], vec![0.5, 0.5, 0.5], vec![0.0, 0.0, 0.0]],
        );
        let r = analyze_overfitting(&m, &PboBands::default());
        assert_eq!(r.pbo, 0.0);
        assert_eq!(r.configs_tested, 3);
        assert_eq!(r.assessment, "LOW OVERFITTING RISK");
    }

    #[test]
    fn random_is_best_scores_high_pbo() {
        // IS-best rotates and always lands worst OOS → every split below median.
        let m = matrix(
            vec![
                vec![3.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 3.0],
            ],
            vec![
                vec![-1.0, 1.0, 1.0, 1.0],
                vec![1.0, -1.0, 1.0, 1.0],
                vec![1.0, 1.0, -1.0, 1.0],
                vec![1.0, 1.0, 1.0, -1.0],
            ],
        );
        let r = analyze_overfitting(&m, &PboBands::default());
        assert_eq!(r.pbo, 1.0);
        assert_eq!(r.assessment, "HIGH OVERFITTING RISK");
        assert!(r.degradation > 0.5);
    }
}
