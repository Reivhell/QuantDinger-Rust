//! Parameter-sensitivity testing: re-score perturbed parameter grids and
//! classify stability. Prefer flat plateaus over isolated spikes — a ±1-step
//! perturbation that destroys performance means the optimum is luck, not edge.
//! (Transaction-cost stress lives in [`crate::costs`].)

/// One evaluated parameter point.
#[derive(Debug, Clone)]
pub struct ParamPoint {
    pub name: String,
    pub value: f64,
    /// Score (e.g. OOS Sharpe or net return) at this point.
    pub score: f64,
}

/// Stability verdict over a perturbation sweep around a baseline.
#[derive(Debug, Clone)]
pub struct SensitivityReport {
    pub baseline: ParamPoint,
    /// min(score)/baseline over the sweep (1.0 = perfectly flat).
    pub worst_relative: f64,
    /// Fraction of sweep points with score > 0 (still viable).
    pub viable_fraction: f64,
    pub verdict: &'static str,
}

/// Classify a sweep. `stable_band` (default 0.8): worst point keeps 80% of
/// baseline. Below `collapse_band` (default 0.3) any single point = spike.
pub fn analyze_sensitivity(
    baseline: ParamPoint,
    sweep: &[ParamPoint],
    stable_band: f64,
    collapse_band: f64,
) -> SensitivityReport {
    if sweep.is_empty() || baseline.score <= 0.0 {
        return SensitivityReport {
            baseline,
            worst_relative: 0.0,
            viable_fraction: 0.0,
            verdict: "UNSTABLE",
        };
    }
    let worst = sweep.iter().map(|p| p.score / baseline.score).fold(f64::INFINITY, f64::min);
    let viable = sweep.iter().filter(|p| p.score > 0.0).count() as f64 / sweep.len() as f64;
    let verdict = if worst >= stable_band && viable >= 0.8 {
        "STABLE"
    } else if worst < collapse_band {
        "UNSTABLE"
    } else {
        "FRAGILE"
    };
    SensitivityReport { baseline, worst_relative: worst, viable_fraction: viable, verdict }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateau_is_stable_spike_is_not() {
        let base = ParamPoint { name: "ema".into(), value: 20.0, score: 1.0 };
        let flat: Vec<ParamPoint> = [18.0, 19.0, 21.0, 22.0]
            .iter()
            .map(|v| ParamPoint { name: "ema".into(), value: *v, score: 0.9 })
            .collect();
        assert_eq!(analyze_sensitivity(base.clone(), &flat, 0.8, 0.3).verdict, "STABLE");
        let spike: Vec<ParamPoint> = [18.0, 19.0, 21.0, 22.0]
            .iter()
            .map(|v| ParamPoint { name: "ema".into(), value: *v, score: -0.5 })
            .collect();
        assert_eq!(analyze_sensitivity(base, &spike, 0.8, 0.3).verdict, "UNSTABLE");
    }
}
