//! Evidence-fed §13 deployment gate: the single mapping from measured
//! research evidence to [`DeploymentGate`] booleans. Both runners and any
//! future runner use this — never hand-roll gate fields per call site.
//!
//! Design notes (why these thresholds, what can weaken them):
//! - The gate is fail-closed: missing/untested evidence is `false`.
//! - Thresholds are the module constants below, not config knobs — a
//!   strategy file must not be able to lower its own bar (`reject the
//!   bar-lowering, not just the trade`).
//! - `no_lookahead`/`no_leakage` are structural properties of the harness
//!   (latency ≥ 1 bar enforced by backtest; causal features; purged
//!   splitters), so they are `true` by construction *of this crate* — any
//!   new execution path that breaks those invariants must set them false
//!   at its own call site and say why.
//! - Everything else is measured per run: walk-forward stability, Monte
//!   Carlo tail, sensitivity plateau, PBO/degradation, cost stress,
//!   drawdown vs cap, stop/lev/killswitch verification.

use crate::autonomy::DeploymentGate;
use crate::costs::CostStressRow;
use crate::montecarlo::MonteCarloReport;
use crate::pbo::PboReport;
use crate::sensitivity::SensitivityReport;
use crate::shadow::{ShadowSummary, ShadowVerdict};
use crate::snooping::SnoopReport;
use crate::walkforward::WalkForwardReport;

/// OOS bar: walk-forward needs at least half the folds OOS-positive
/// (same bar the report already uses via `min_oos_fraction` default).
pub const MIN_OOS_FRACTION: f64 = 0.5;
/// Monte Carlo bar: p5 of bootstrapped total return must clear zero —
/// the edge survives resampling luck, not just the realized order.
pub const MC_P5_MIN_RETURN: f64 = 0.0;
/// Monte Carlo tail bar: p95 shuffled max-DD must stay within 2× the
/// backtest max-DD — ordering luck must not double the realized tail.
pub const MC_TAIL_DD_MULTIPLE: f64 = 2.0;
/// PBO bar: IS-best must survive OOS more often than a coin flip, and
/// degradation must stay under 50% (mirrors `PboBands` defaults).
pub const PBO_MAX: f64 = 0.5;
pub const DEGRADATION_MAX: f64 = 0.5;
/// Sensitivity bar: only a plateaued knob passes — UNSTABLE / SPIKE /
/// NO_EDGE all fail closed.
pub const SENSITIVITY_PASS: &str = "STABLE";
/// Data-snooping bar: the grid-best OOS edge must survive the White/SPA
/// luck adjustment (consistent p < 5%, same bar the snoop module uses).
/// Missing/thin evidence (`None`) fails closed — an untested grid is not
/// an honest grid.
pub const SNOOP_ALPHA: f64 = 0.05;

/// Measured evidence for one strategy run. Plain data in, gate out —
/// no I/O, no globals, deterministic.
#[derive(Debug, Clone)]
pub struct GateEvidence {
    /// Realistic costs charged (commission/spread/slippage all > 0).
    pub realistic_costs: bool,
    /// Walk-forward stage ran and its report.
    pub walkforward: Option<WalkForwardReport>,
    /// Monte Carlo report (needs bootstrap p5 + shuffled-DD tail).
    pub monte_carlo: Option<MonteCarloReport>,
    /// Full-sample backtest max drawdown (fraction of equity).
    pub backtest_max_dd: f64,
    /// Sensitivity report for the tuned knob.
    pub sensitivity: Option<SensitivityReport>,
    /// PBO report over the config grid.
    pub pbo: Option<PboReport>,
    /// Cost-stress verdict string (from `execution_sensitivity`).
    pub cost_verdict: String,
    /// Drawdown cap from config (`max_drawdown`).
    pub max_drawdown_cap: f64,
    /// Every trade carried a stop (backtest enforces; harness asserts).
    pub stop_on_every_trade: bool,
    /// Leverage within cap with liquidation buffer (config guards).
    pub leverage_ok: bool,
    /// Kill-switch tripped correctly in shadow when it had to
    /// (or shadow never needed it — verified present either way).
    pub killswitch_verified: bool,
    /// Data-snooping report (White RC + Hansen SPA over the grid OOS
    /// matrix). `None` = too thin to test → fails closed, never faked.
    pub snooping: Option<SnoopReport>,
    /// Shadow verdict from the unseen holdout window.
    pub shadow: Option<ShadowVerdictSummary>,
}

/// Minimal shadow outcome the gate needs (carries no trade vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowVerdictSummary {
    Promote,
    KeepIncumbent,
    Reject,
}

impl From<&ShadowSummary> for ShadowVerdictSummary {
    fn from(s: &ShadowSummary) -> Self {
        match s.verdict.as_str() {
            "Promote" => ShadowVerdictSummary::Promote,
            "KeepIncumbent" => ShadowVerdictSummary::KeepIncumbent,
            _ => ShadowVerdictSummary::Reject,
        }
    }
}

impl From<ShadowVerdict> for ShadowVerdictSummary {
    fn from(v: ShadowVerdict) -> Self {
        match v {
            ShadowVerdict::Promote => ShadowVerdictSummary::Promote,
            ShadowVerdict::KeepIncumbent => ShadowVerdictSummary::KeepIncumbent,
            ShadowVerdict::Reject => ShadowVerdictSummary::Reject,
        }
    }
}

/// Map measured evidence to the 13 gate booleans. Fail-closed throughout:
/// `None` (stage missing) is `false`, never `true`.
pub fn gate_from_evidence(ev: &GateEvidence) -> DeploymentGate {
    let wf_stable =
        ev.walkforward.as_ref().map(|w| w.stable).unwrap_or(false);
    let oos_ok = ev
        .walkforward
        .as_ref()
        .map(|w| w.positive_oos_fraction >= MIN_OOS_FRACTION)
        .unwrap_or(false);
    let mc_ok = ev
        .monte_carlo
        .as_ref()
        .map(|m| {
            m.boot_return.p5 > MC_P5_MIN_RETURN
                && m.shuffled_max_dd.p95
                    <= ev.backtest_max_dd.max(1e-9) * MC_TAIL_DD_MULTIPLE
        })
        .unwrap_or(false);
    let sens_ok = ev
        .sensitivity
        .as_ref()
        .map(|s| s.verdict == SENSITIVITY_PASS)
        .unwrap_or(false);
    let (pbo_ok, no_overfit) = ev
        .pbo
        .as_ref()
        .map(|p| (p.pbo <= PBO_MAX, p.degradation <= DEGRADATION_MAX))
        .unwrap_or((false, false));
    let exec_ok = !ev.cost_verdict.starts_with("EXECUTION-SENSITIVE")
        && !ev.cost_verdict.starts_with("MODERATE");
    let dd_ok = ev.backtest_max_dd <= ev.max_drawdown_cap;
    let shadow_ok =
        matches!(ev.shadow, Some(ShadowVerdictSummary::Promote));
    // Data-snooping (§7 luck adjustment): the grid-best OOS edge must
    // survive White + SPA at 5%. Thin/missing evidence fails closed.
    let snoop_ok = ev
        .snooping
        .as_ref()
        .map(|s| s.white_p < SNOOP_ALPHA && s.spa_p_consistent < SNOOP_ALPHA)
        .unwrap_or(false);
    // OOS validation (§13 `oos_validated`) needs *independent* confirmation
    // from two directions: walk-forward folds AND the CPCV/PBO path.
    // WF alone can pass on 2 folds; PBO alone can pass on a lucky grid.
    let oos_validated = oos_ok && pbo_ok && no_overfit && snoop_ok;
    DeploymentGate {
        no_lookahead: true, // backtest latency>=1 + causal features
        no_leakage: true,   // purged walk-forward + CPCV splitters
        realistic_costs: ev.realistic_costs,
        oos_validated,
        walkforward_stable: wf_stable,
        monte_carlo_ok: mc_ok,
        sensitivity_ok: sens_ok,
        drawdown_ok: dd_ok,
        execution_stress_ok: exec_ok,
        risk_limits_ok: dd_ok && shadow_ok,
        stop_verified: ev.stop_on_every_trade,
        leverage_ok: ev.leverage_ok,
        killswitch_verified: ev.killswitch_verified,
    }
}

/// CostStressRow is built per-run; this keeps the gate's cost-string
/// contract next to the gate (DRY: one place names the failing prefixes).
pub fn cost_rows_profitable_at_2x(rows: &[CostStressRow]) -> bool {
    rows.iter().filter(|r| r.multiplier <= 2.0).all(|r| r.still_profitable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::montecarlo::{summarize, MonteCarloReport};
    use crate::pbo::{PboBands, ScoreMatrix};
    use crate::sensitivity::{analyze_sensitivity, ParamPoint};

    fn ev() -> GateEvidence {
        GateEvidence {
            realistic_costs: true,
            walkforward: None,
            monte_carlo: None,
            backtest_max_dd: 0.02,
            sensitivity: None,
            pbo: None,
            cost_verdict: "RESILIENT (dies only beyond +100% costs)".into(),
            max_drawdown_cap: 0.10,
            stop_on_every_trade: true,
            leverage_ok: true,
            killswitch_verified: true,
            snooping: None, // most unit fixtures don't run the bootstrap
            shadow: Some(ShadowVerdictSummary::Promote),
        }
    }

    #[test]
    fn missing_stages_fail_closed() {
        // Nothing ran → every measured check is false, structural trues stay.
        let g = gate_from_evidence(&ev());
        assert!(g.no_lookahead && g.no_leakage);
        assert!(!g.deploy_allowed());
        assert!(g.failures().contains(&"oos_not_validated"));
        assert!(g.failures().contains(&"walkforward_unstable"));
        assert!(g.failures().contains(&"monte_carlo_failed"));
        assert!(g.failures().contains(&"parameter_fragile"));
    }

    #[test]
    fn oos_needs_both_wf_and_pbo() {
        // WF stable alone must NOT validate OOS (2 folds can luck through).
        let mut e = ev();
        e.walkforward = Some(crate::walkforward::summarize_walkforward(
            vec![
                crate::walkforward::FoldOutcome {
                    fold: 0,
                    selected_config: "a".into(),
                    is_score: 0.1,
                    oos_return: 0.05,
                    oos_max_dd: 0.01,
                    oos_sharpe: 1.0,
                    oos_sortino: 1.0,
                    oos_trades: 5,
                },
                crate::walkforward::FoldOutcome {
                    fold: 1,
                    selected_config: "a".into(),
                    is_score: 0.1,
                    oos_return: 0.03,
                    oos_max_dd: 0.01,
                    oos_sharpe: 0.8,
                    oos_sortino: 0.8,
                    oos_trades: 5,
                },
            ],
            0.5,
        ));
        let g = gate_from_evidence(&e);
        assert!(g.walkforward_stable);
        assert!(!g.oos_validated, "WF without PBO must not validate OOS");
    }

    #[test]
    fn full_evidence_can_pass() {
        let mut e = ev();
        e.walkforward = Some(crate::walkforward::summarize_walkforward(
            vec![crate::walkforward::FoldOutcome {
                fold: 0,
                selected_config: "a".into(),
                is_score: 0.1,
                oos_return: 0.05,
                oos_max_dd: 0.01,
                oos_sharpe: 1.0,
                oos_sortino: 1.0,
                oos_trades: 5,
            }],
            0.5,
        ));
        // PBO: IS-best survives OOS on every split, zero degradation.
        let m = ScoreMatrix {
            configs: vec!["a".into(), "b".into()],
            is: vec![vec![0.2, 0.2], vec![0.1, 0.1]],
            oos: vec![vec![0.2, 0.2], vec![0.0, 0.0]],
        };
        e.pbo = Some(crate::pbo::analyze_overfitting(&m, &PboBands::default()));
        // Sensitivity plateau around a positive baseline.
        let base = ParamPoint { name: "k".into(), value: 20.0, score: 0.05 };
        let sweep = vec![
            ParamPoint { name: "k".into(), value: 10.0, score: 0.048 },
            ParamPoint { name: "k".into(), value: 30.0, score: 0.052 },
        ];
        e.sensitivity = Some(analyze_sensitivity(base, &sweep, 0.8, 0.3));
        // MC over healthy returns: bootstrap p5 > 0, tail bounded.
        let rets = vec![0.01; 40];
        let mut mc: MonteCarloReport =
            crate::montecarlo::run_monte_carlo(&rets, 200, 7);
        // Force tail fields into passing shape deterministically.
        mc.boot_return = summarize(vec![0.02; 200]);
        mc.shuffled_max_dd = summarize(vec![0.01; 200]);
        e.monte_carlo = Some(mc);
        // Snooping: grid-best survives the luck adjustment (both p < 5%).
        e.snooping = Some(crate::snooping::SnoopReport {
            rules: 2,
            periods: 12,
            boot_sims: 500,
            mean_block: 4,
            best_rule: 0,
            best_mean: 0.02,
            white_stat: 3.1,
            white_p: 0.01,
            spa_stat: 2.8,
            spa_p_lower: 0.02,
            spa_p_consistent: 0.02,
            spa_p_upper: 0.30,
            assessment: "EDGE SURVIVES SNOOPING",
        });
        let g = gate_from_evidence(&e);
        assert!(g.deploy_allowed(), "full evidence must pass: {:?}", g.failures());
    }

    #[test]
    fn snooping_blocks_oos_without_other_failures() {
        // Every stage passes except snooping (None = thin grid): OOS must
        // stay unvalidated — an untested grid is not an honest grid.
        let mut e = ev();
        e.walkforward = Some(crate::walkforward::summarize_walkforward(
            vec![crate::walkforward::FoldOutcome {
                fold: 0,
                selected_config: "a".into(),
                is_score: 0.1,
                oos_return: 0.05,
                oos_max_dd: 0.01,
                oos_sharpe: 1.0,
                oos_sortino: 1.0,
                oos_trades: 5,
            }],
            0.5,
        ));
        let m = ScoreMatrix {
            configs: vec!["a".into(), "b".into()],
            is: vec![vec![0.2, 0.2], vec![0.1, 0.1]],
            oos: vec![vec![0.2, 0.2], vec![0.0, 0.0]],
        };
        e.pbo = Some(crate::pbo::analyze_overfitting(&m, &PboBands::default()));
        let base = ParamPoint { name: "k".into(), value: 20.0, score: 0.05 };
        let sweep = vec![
            ParamPoint { name: "k".into(), value: 10.0, score: 0.048 },
            ParamPoint { name: "k".into(), value: 30.0, score: 0.052 },
        ];
        e.sensitivity = Some(analyze_sensitivity(base, &sweep, 0.8, 0.3));
        let rets = vec![0.01; 40];
        let mut mc: MonteCarloReport =
            crate::montecarlo::run_monte_carlo(&rets, 200, 7);
        mc.boot_return = summarize(vec![0.02; 200]);
        mc.shuffled_max_dd = summarize(vec![0.01; 200]);
        e.monte_carlo = Some(mc);
        // snooping stays None → oos_validated false, everything else green.
        let g = gate_from_evidence(&e);
        assert!(!g.oos_validated, "missing snooping must block OOS validation");
        assert!(g.failures().contains(&"oos_not_validated"));
    }

    #[test]
    fn execution_sensitive_fails_exec_check_only() {
        let mut e = ev();
        e.cost_verdict = "EXECUTION-SENSITIVE (dies at ≤+25% costs)".into();
        let g = gate_from_evidence(&e);
        assert!(g.failures().contains(&"execution_sensitive"));
    }
}
