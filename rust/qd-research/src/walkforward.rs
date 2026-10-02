//! Walk-forward analysis: train → pick params → validate OOS, roll forward.
//!
//! The contract: parameter selection sees ONLY the in-sample window. The
//! OOS window is evaluated once per step with the selected configuration
//! and never feeds back into selection. Report stability across windows —
//! never present the best window as representative.

/// One walk-forward fold, as index ranges over the bar series.
#[derive(Debug, Clone)]
pub struct Fold {
    /// In-sample range `[is_start, is_end)`.
    pub is_start: usize,
    pub is_end: usize,
    /// Out-of-sample range `[oos_start, oos_end)`.
    pub oos_start: usize,
    pub oos_end: usize,
}

/// Build rolling (or anchored) folds.
///
/// - `train`: IS window length in bars; `oos`: OOS length; `step`: how far
///   the window advances per fold.
/// - `anchored`: IS always starts at 0 (expanding) instead of rolling.
/// - At least one full OOS window must fit; trailing partial windows are
///   dropped (documented, deterministic).
pub fn build_folds(n: usize, train: usize, oos: usize, step: usize, anchored: bool) -> Vec<Fold> {
    let mut folds = Vec::new();
    if train == 0 || oos == 0 || step == 0 || n < train + oos {
        return folds;
    }
    let mut start = 0;
    loop {
        let is_start = if anchored { 0 } else { start };
        let is_end = start + train;
        let oos_start = is_end;
        let oos_end = oos_start + oos;
        if oos_end > n {
            break;
        }
        folds.push(Fold { is_start, is_end, oos_start, oos_end });
        start += step;
    }
    folds
}

/// OOS outcome of one fold.
#[derive(Debug, Clone)]
pub struct FoldOutcome {
    pub fold: usize,
    pub selected_config: String,
    pub is_score: f64,
    pub oos_return: f64,
    pub oos_max_dd: f64,
    pub oos_sharpe: f64,
    pub oos_sortino: f64,
    pub oos_trades: usize,
}

/// Aggregate stability verdict over fold outcomes.
#[derive(Debug, Clone)]
pub struct WalkForwardReport {
    pub folds: Vec<FoldOutcome>,
    /// Fraction of folds with `oos_return > 0`.
    pub positive_oos_fraction: f64,
    /// Median OOS Sharpe across folds.
    pub median_oos_sharpe: f64,
    /// Worst OOS max-DD across folds.
    pub worst_oos_dd: f64,
    /// `true` when at least `min_fraction` of folds are OOS-positive.
    pub stable: bool,
}

pub fn summarize_walkforward(folds: Vec<FoldOutcome>, min_fraction: f64) -> WalkForwardReport {
    let pos = folds.iter().filter(|f| f.oos_return > 0.0).count();
    let frac = if folds.is_empty() { 0.0 } else { pos as f64 / folds.len() as f64 };
    let mut sharpes: Vec<f64> = folds.iter().map(|f| f.oos_sharpe).collect();
    sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if sharpes.is_empty() {
        0.0
    } else {
        sharpes[sharpes.len() / 2]
    };
    let worst_dd = folds.iter().map(|f| f.oos_max_dd).fold(0.0, f64::max);
    WalkForwardReport {
        folds,
        positive_oos_fraction: frac,
        median_oos_sharpe: med,
        worst_oos_dd: worst_dd,
        stable: frac >= min_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_tile_without_overlap_or_gap_use() {
        let fs = build_folds(100, 40, 10, 10, false);
        assert_eq!(fs.len(), 6); // starts 0..=50; 60+40+10 > 100 stops
        assert_eq!((fs[0].is_start, fs[0].is_end, fs[0].oos_start, fs[0].oos_end), (0, 40, 40, 50));
        assert_eq!((fs[5].is_start, fs[5].is_end, fs[5].oos_start, fs[5].oos_end), (50, 90, 90, 100));
        // OOS of fold k never overlaps IS of fold k+1's selection data... IS windows overlap
        // by design in rolling mode; OOS windows never overlap each other.
        for w in fs.windows(2) {
            assert!(w[0].oos_end <= w[1].oos_start);
        }
    }

    #[test]
    fn anchored_expands_is() {
        let fs = build_folds(100, 40, 10, 10, true);
        assert!(fs.iter().all(|f| f.is_start == 0));
        assert_eq!(fs.len(), 6);
    }

    #[test]
    fn summary_marks_instability() {
        let mk = |r: f64| FoldOutcome {
            fold: 0,
            selected_config: "a".into(),
            is_score: 1.0,
            oos_return: r,
            oos_max_dd: 0.05,
            oos_sharpe: 0.5,
            oos_sortino: 0.5,
            oos_trades: 10,
        };
        let rep = summarize_walkforward(vec![mk(0.1), mk(-0.2), mk(0.05), mk(-0.1)], 0.6);
        assert_eq!(rep.positive_oos_fraction, 0.5);
        assert!(!rep.stable);
    }
}
