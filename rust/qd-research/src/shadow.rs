//! Paper/shadow testing bulkhead (spec §12).
//!
//! Validation (backtest → walk-forward → Monte Carlo → CPCV) answers "did it
//! work on history". Shadow answers the different question "does it survive
//! unseen data under autonomous control". A candidate strategy runs here —
//! never on real capital — against data the research stage never saw, behind
//! the full entry gate ([`crate::autonomy`]) with a live kill-switch. Only a
//! candidate that survives shadow may advance `Shadow → Deploy` in the
//! research loop ([`crate::autonomy::advance_loop`]); the comparison verdict
//! below is what feeds that transition's `checks_passed` flag.
//!
//! Halt simulation is exact, not approximate: the first pass runs unhalted,
//! the kill-switch's first trip bar is found on the realized equity curve,
//! and a second pass re-runs with every signal at/after the trip forced
//! `Flat` (existing position still managed to exit — STOP NEW ENTRIES ↓
//! MONITOR EXISTING ↓ WAIT, §4). Fills before the trip are bit-identical
//! between passes because fills depend only on past bars.
//!
//! Single-name assumption: exposure/correlated notionals are tracked as 0;
//! the daily/weekly/drawdown trips are the live dimensions.

use crate::autonomy::{gate_signals, EntryCtx};
use crate::backtest::{run_backtest, ExecConfig, Signal};
use crate::features::Bar;
use crate::metrics::drawdown_stats;
use crate::regime::Regime;
use crate::risk::KillSwitch;

/// Everything shadow needs. Kill-switch fields (`ks_*`) mirror
/// [`crate::config::ResearchConfig`] caps; duplicated here (not imported) so
/// this module compiles against explicit caller-supplied limits only.
pub struct ShadowConfig {
    pub starting_equity: f64,
    pub max_positions: usize,
    pub max_open_risk: f64,
    pub blocked_regimes: Vec<Regime>,
    pub leverage_ok: bool,
    pub stop_present: bool,
    pub execution_ok: bool,
    pub ks_daily: f64,
    pub ks_weekly: f64,
    pub ks_drawdown: f64,
    pub ks_exposure: f64,
    pub ks_corr: f64,
    pub week_bars: u32,
    /// Bars per paper session (daily marker). Bar-count based: shadow has
    /// no wall clock, so session boundaries come from bar index, exactly
    /// like the autonomy check's paper daily marker.
    pub session_bars: usize,
}

/// Outcome of one shadow run. Paper only — no orders, no capital.
pub struct ShadowResult {
    pub equity_curve: Vec<f64>,
    pub trade_count: usize,
    /// Idle bars: gate-suppressed + halt-suppressed. Reported, never silent.
    pub suppressed: usize,
    /// First bar where the kill-switch latched, if it did.
    pub halt_bar: Option<usize>,
    pub halt_reason: Option<String>,
    pub total_return: f64,
    pub max_drawdown: f64,
}

/// Run one contender through the shadow bulkhead. `signals`/`regimes` must
/// cover `bars` 1:1 (shorter slices read as missing → gated out).
pub fn run_shadow(
    bars: &[Bar],
    signals: &[Signal],
    regimes: &[Regime],
    feature_ids: &[u64],
    exec: &ExecConfig,
    sizer: impl Fn(f64, f64) -> f64 + Copy,
    cfg: &ShadowConfig,
) -> Result<ShadowResult, String> {
    if bars.is_empty() {
        return Err("no bars".to_string());
    }
    // Fail safe first: corrupt input bars → no paper trading either.
    if !bars.iter().all(|b| b.is_valid()) {
        return Ok(ShadowResult {
            equity_curve: vec![cfg.starting_equity; bars.len()],
            trade_count: 0,
            suppressed: signals.iter().filter(|s| **s != Signal::Flat).count(),
            halt_bar: Some(0),
            halt_reason: Some("unreliable_data".to_string()),
            total_return: 0.0,
            max_drawdown: 0.0,
        });
    }
    let gate = EntryCtx {
        data_ok: true,
        emergency_halted: false,
        risk_halted: false,
        blocked_regimes: &cfg.blocked_regimes,
        open_positions: 0,
        max_positions: cfg.max_positions,
        open_risk: 0.0,
        max_open_risk: cfg.max_open_risk,
        leverage_ok: cfg.leverage_ok,
        stop_present: cfg.stop_present,
        execution_ok: cfg.execution_ok,
    };
    let (gated, mut suppressed) = gate_signals(signals, regimes, &gate);
    let first = run_backtest(bars, &gated, regimes, feature_ids, exec, cfg.starting_equity, sizer);
    let first_eq = first.equity_curve.clone();
    let trip: Option<(usize, String)> = first_trip(&first_eq, cfg);
    let (equity_curve, trade_count, halt_bar, halt_reason) = match trip {
        None => (first_eq, first.trades.len(), None, None),
        Some((k, reason)) => {
            // Halt: no new entries at/after k; the open position (if any) is
            // still managed to its exit by the engine.
            let mut halted = gated.clone();
            let mut extra = 0usize;
            for (i, s) in halted.iter_mut().enumerate() {
                if i >= k && *s != Signal::Flat {
                    *s = Signal::Flat;
                    extra += 1;
                }
            }
            suppressed += extra;
            let second =
                run_backtest(bars, &halted, regimes, feature_ids, exec, cfg.starting_equity, sizer);
            // Latch audit: the halted curve must agree with the unhalted one
            // strictly before k (fills depend only on the past).
            debug_assert!(second.equity_curve.iter().zip(first_eq.iter()).take(k).all(|(a, b)| a == b));
            let n_trades = second.trades.len();
            let curve = second.equity_curve;
            (curve, n_trades, Some(k), Some(reason))
        }
    };
    let total_return = equity_curve.last().copied().unwrap_or(cfg.starting_equity) / cfg.starting_equity - 1.0;
    let max_drawdown = drawdown_stats(&equity_curve).0;
    Ok(ShadowResult {
        equity_curve,
        trade_count,
        suppressed,
        halt_bar,
        halt_reason,
        total_return,
        max_drawdown,
    })
}

/// First kill-switch trip over a realized equity curve. Session boundaries
/// are bar-count based (`i % session_bars == 0`): shadow has no wall clock,
/// and using caller timestamps would silently disable the daily trip on
/// fixtures whose `t` is a bar index (all 600 bars in one "day" — the daily
/// trip could never fire). Explicit is safer than clever.
fn first_trip(equity: &[f64], cfg: &ShadowConfig) -> Option<(usize, String)> {
    let mut ks = KillSwitch::with_weekly(
        cfg.ks_daily,
        cfg.ks_weekly,
        cfg.ks_drawdown,
        cfg.ks_exposure,
        cfg.ks_corr,
        cfg.week_bars,
    );
    let session = cfg.session_bars.max(1);
    for (i, eq) in equity.iter().enumerate() {
        let new_day = i % session == 0;
        if let Some(reason) = ks.update_full(*eq, 0.0, 0.0, new_day) {
            return Some((i, reason));
        }
    }
    None
}

/// Shadow verdict for the `Shadow → Deploy` loop transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowVerdict {
    /// Candidate survives shadow AND improves on the incumbent without
    /// worsening tail risk: may advance to Deploy (still gated by §13).
    Promote,
    /// Candidate survives but does not beat the incumbent: stay put.
    KeepIncumbent,
    /// Candidate tripped, degraded vs its own validation, or blew tail
    /// risk: back to Research, never Deploy.
    Reject,
}

pub struct ShadowComparison {
    pub verdict: ShadowVerdict,
    pub reason: &'static str,
}

/// Report-ready shadow outcome (Clone so examples can print AND report).
#[derive(Debug, Clone, Default)]
pub struct ShadowSummary {
    pub unseen_bars: usize,
    pub candidate_trades: usize,
    pub candidate_return: f64,
    pub candidate_max_dd: f64,
    pub candidate_halt: Option<String>,
    pub candidate_suppressed: usize,
    pub incumbent_trades: usize,
    pub incumbent_return: f64,
    pub incumbent_max_dd: f64,
    pub incumbent_halt: Option<String>,
    pub verdict: String,
    pub verdict_reason: String,
}

/// Drawdown-first candidate comparison (§9, §15):
/// 1. Halt in shadow → Reject (uncontrolled, whatever the return).
/// 2. Shadow return below `validation_median_oos - max_degradation` →
///    Reject (the edge did not survive unseen data).
/// 3. Max DD worse than incumbent × (1 + `dd_tolerance`) → KeepIncumbent
///    (higher return never buys higher tail risk).
/// 4. Otherwise the higher return wins.
pub fn compare_candidate(
    candidate: &ShadowResult,
    incumbent: &ShadowResult,
    validation_median_oos: f64,
    max_degradation: f64,
    dd_tolerance: f64,
) -> ShadowComparison {
    if candidate.halt_bar.is_some() {
        return ShadowComparison {
            verdict: ShadowVerdict::Reject,
            reason: "kill-switch tripped in shadow — uncontrolled, return irrelevant",
        };
    }
    if !validation_median_oos.is_finite() || !max_degradation.is_finite() || max_degradation < 0.0 {
        return ShadowComparison {
            verdict: ShadowVerdict::Reject,
            reason: "invalid validation baseline — cannot promote without one",
        };
    }
    if candidate.total_return < validation_median_oos - max_degradation {
        return ShadowComparison {
            verdict: ShadowVerdict::Reject,
            reason: "shadow return degraded beyond tolerance vs validation OOS",
        };
    }
    if candidate.max_drawdown > incumbent.max_drawdown * (1.0 + dd_tolerance) {
        return ShadowComparison {
            verdict: ShadowVerdict::KeepIncumbent,
            reason: "candidate worsens tail risk vs incumbent — return does not compensate",
        };
    }
    if candidate.total_return > incumbent.total_return {
        ShadowComparison { verdict: ShadowVerdict::Promote, reason: "survives shadow, beats incumbent, tail risk controlled" }
    } else {
        ShadowComparison {
            verdict: ShadowVerdict::KeepIncumbent,
            reason: "no improvement over incumbent — stay put",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::synthetic_bars;

    fn harness_cfg() -> ShadowConfig {
        ShadowConfig {
            starting_equity: 100_000.0,
            max_positions: 3,
            max_open_risk: 0.06,
            blocked_regimes: vec![Regime::Ranging, Regime::HighVolatility, Regime::LowVolatility],
            leverage_ok: true,
            stop_present: true,
            execution_ok: true,
            ks_daily: 0.02,
            ks_weekly: 0.05,
            ks_drawdown: 0.10,
            ks_exposure: 3.0,
            ks_corr: 0.0,
            week_bars: 5,
            session_bars: 50,
        }
    }

    fn exec_cfg() -> ExecConfig {
        ExecConfig {
            commission: 0.0005,
            spread: 0.0002,
            slippage: 0.0003,
            latency_bars: 1,
            stop_loss: 0.03,
            take_profit: 0.06,
            ..ExecConfig::default()
        }
    }

    #[test]
    fn empty_input_is_an_error() {
        let cfg = harness_cfg();
        assert!(run_shadow(&[], &[], &[], &[], &exec_cfg(), |_, _| 1.0, &cfg).is_err());
    }

    #[test]
    fn corrupt_bars_halt_before_any_trade() {
        let mut bars = synthetic_bars(200, 7);
        bars[100].close = f64::NAN;
        let n = bars.len();
        let sig = vec![Signal::Long; n];
        let reg = vec![Regime::TrendingUp; n];
        let ids = vec![1u64; n];
        let r = run_shadow(&bars, &sig, &reg, &ids, &exec_cfg(), |_, _| 10.0, &harness_cfg())
            .expect("returns halted result");
        assert_eq!(r.trade_count, 0);
        assert_eq!(r.halt_bar, Some(0));
        assert_eq!(r.halt_reason.as_deref(), Some("unreliable_data"));
    }

    #[test]
    fn fully_blocked_regimes_trade_nothing() {
        let bars = synthetic_bars(300, 7);
        let n = bars.len();
        let sig = vec![Signal::Long; n];
        let reg = vec![Regime::Ranging; n]; // blocked
        let ids = vec![1u64; n];
        let r = run_shadow(&bars, &sig, &reg, &ids, &exec_cfg(), |_, _| 10.0, &harness_cfg())
            .expect("runs");
        assert_eq!(r.trade_count, 0);
        assert_eq!(r.suppressed, n); // every signal idled, all counted
        assert!(r.halt_bar.is_none());
    }

    #[test]
    fn kill_switch_halts_and_freezes_entries() {
        // Hair-trigger daily loss on a volatile fixture: must trip, and no
        // trade may open at/after the trip bar.
        let bars = synthetic_bars(600, 99);
        let n = bars.len();
        let sig = vec![Signal::Long; n];
        let reg = vec![Regime::TrendingUp; n];
        let ids = vec![1u64; n];
        let mut cfg = harness_cfg();
        cfg.ks_daily = 0.004; // hair trigger vs 2%-vol shock phase
        let r = run_shadow(&bars, &sig, &reg, &ids, &exec_cfg(), |_, _| 50.0, &cfg).expect("runs");
        let k = r.halt_bar.expect("hair trigger must trip on shock fixture");
        assert!(k > 0, "entries before the trip are legitimate");
        assert!(r.halt_reason.is_some());
    }

    #[test]
    fn halted_curve_matches_unhalted_before_trip() {
        // Determinism + prefix agreement: re-running the unhalted pass must
        // reproduce the halted curve strictly before the trip bar.
        let bars = synthetic_bars(600, 99);
        let n = bars.len();
        let sig = vec![Signal::Long; n];
        let reg = vec![Regime::TrendingUp; n];
        let ids = vec![1u64; n];
        let mut cfg = harness_cfg();
        cfg.ks_daily = 0.004;
        let exec = exec_cfg();
        let r = run_shadow(&bars, &sig, &reg, &ids, &exec, |_, _| 50.0, &cfg).expect("runs");
        let k = r.halt_bar.unwrap();
        let gate = EntryCtx {
            data_ok: true,
            emergency_halted: false,
            risk_halted: false,
            blocked_regimes: &cfg.blocked_regimes,
            open_positions: 0,
            max_positions: cfg.max_positions,
            open_risk: 0.0,
            max_open_risk: cfg.max_open_risk,
            leverage_ok: true,
            stop_present: true,
            execution_ok: true,
        };
        let (gated, _) = gate_signals(&sig, &reg, &gate);
        let raw = run_backtest(&bars, &gated, &reg, &ids, &exec, cfg.starting_equity, |_, _| 50.0);
        assert!(r.equity_curve.iter().zip(raw.equity_curve.iter()).take(k).all(|(a, b)| a == b));
    }

    fn result(ret: f64, dd: f64, halted: bool) -> ShadowResult {
        ShadowResult {
            equity_curve: vec![],
            trade_count: 1,
            suppressed: 0,
            halt_bar: halted.then_some(10),
            halt_reason: halted.then(|| "max_drawdown".to_string()),
            total_return: ret,
            max_drawdown: dd,
        }
    }

    #[test]
    fn comparison_is_drawdown_first() {
        let inc = result(0.05, 0.04, false);
        // Tripped candidate: rejected no matter the return.
        let c = compare_candidate(&result(0.50, 0.01, true), &inc, 0.04, 0.10, 0.10);
        assert_eq!(c.verdict, ShadowVerdict::Reject);
        // Degraded vs its own validation: rejected.
        let c = compare_candidate(&result(0.0, 0.02, false), &inc, 0.20, 0.10, 0.10);
        assert_eq!(c.verdict, ShadowVerdict::Reject);
        // Better return but worse tail: stays incumbent.
        let c = compare_candidate(&result(0.09, 0.06, false), &inc, 0.04, 0.10, 0.10);
        assert_eq!(c.verdict, ShadowVerdict::KeepIncumbent);
        // Beats incumbent, tail controlled: promote.
        let c = compare_candidate(&result(0.09, 0.04, false), &inc, 0.04, 0.10, 0.10);
        assert_eq!(c.verdict, ShadowVerdict::Promote);
        // Survives, tail fine, but no improvement: stay put.
        let c = compare_candidate(&result(0.03, 0.03, false), &inc, 0.02, 0.10, 0.10);
        assert_eq!(c.verdict, ShadowVerdict::KeepIncumbent);
        // No validation baseline: cannot promote.
        let c = compare_candidate(&result(0.09, 0.02, false), &inc, f64::NAN, 0.10, 0.10);
        assert_eq!(c.verdict, ShadowVerdict::Reject);
    }

    #[test]
    fn shadow_is_deterministic() {
        let bars = synthetic_bars(400, 11);
        let n = bars.len();
        let sig = vec![Signal::Long; n];
        let reg = vec![Regime::TrendingUp; n];
        let ids = vec![1u64; n];
        let cfg = harness_cfg();
        let exec = exec_cfg();
        let a = run_shadow(&bars, &sig, &reg, &ids, &exec, |_, _| 10.0, &cfg).unwrap();
        let b = run_shadow(&bars, &sig, &reg, &ids, &exec, |_, _| 10.0, &cfg).unwrap();
        assert_eq!(a.equity_curve, b.equity_curve);
        assert_eq!(a.trade_count, b.trade_count);
        assert_eq!(a.halt_bar, b.halt_bar);
    }
}
