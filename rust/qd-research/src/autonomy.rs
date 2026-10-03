//! Autonomous decision engine, emergency safety, deployment gate and
//! versioned strategy management.
//!
//! Design rule (spec §6, §15): determinism first. The strategy/signal layer
//! proposes; this module disposes. No signal, optimizer output, or LLM
//! context can bypass the layers here — a failing layer forces `NoTrade`,
//! and `NoTrade` (remain idle) is always a valid outcome.
//!
//! Layers, in order:
//! 1. Data/exchange/state health (fail-safe: uncertain state → no trade).
//! 2. Emergency halt + portfolio kill-switch (latched).
//! 3. Signal presence (Flat → idle, counted not errored).
//! 4. Regime filter (blocked regimes → idle).
//! 5. Portfolio caps (positions / open risk / correlated exposure).
//! 6. Leverage + liquidation buffer + mandatory protective stop.
//! 7. Execution conditions (spread / latency / order health).
//! Only then: `Trade`.

use crate::backtest::Signal;
use crate::regime::Regime;

/// Machine-readable reason codes for `NoTrade`. Callers log these; idle
/// time is evidence of risk control, not a failure.
pub const IDLE_FLAT_SIGNAL: &str = "idle_flat_signal";
pub const NO_TRADE_DATA: &str = "no_trade_data_unreliable";
pub const NO_TRADE_EMERGENCY: &str = "no_trade_emergency_halt";
pub const NO_TRADE_RISK: &str = "no_trade_risk_halted";
pub const NO_TRADE_REGIME: &str = "no_trade_regime_blocked";
pub const NO_TRADE_POSITIONS: &str = "no_trade_max_positions";
pub const NO_TRADE_OPEN_RISK: &str = "no_trade_max_open_risk";
pub const NO_TRADE_LEVERAGE: &str = "no_trade_leverage_unsafe";
pub const NO_TRADE_NO_STOP: &str = "no_trade_no_protective_stop";
pub const NO_TRADE_EXECUTION: &str = "no_trade_execution_unhealthy";

/// Entry-gate context: everything the gate needs, all caller-supplied.
/// Pure data — no hidden state, no I/O.
#[derive(Debug, Clone)]
pub struct EntryCtx<'a> {
    /// Data feed + exchange + position-state certainty (§11).
    pub data_ok: bool,
    /// Emergency halt latched (§11).
    pub emergency_halted: bool,
    /// Portfolio kill-switch tripped (§4).
    pub risk_halted: bool,
    /// Regimes where new entries are forbidden (caller policy, §5).
    pub blocked_regimes: &'a [Regime],
    /// Current open positions vs cap (§4).
    pub open_positions: usize,
    pub max_positions: usize,
    /// Current open risk (fraction of equity) vs cap (§4).
    pub open_risk: f64,
    pub max_open_risk: f64,
    /// Liquidation buffer check passed (§2).
    pub leverage_ok: bool,
    /// A protective stop will be attached (§1).
    pub stop_present: bool,
    /// Spread/latency/order-failure health (§11).
    pub execution_ok: bool,
}

/// Entry verdict. `Trade` means every layer passed; anything else names
/// the first failing layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryVerdict {
    Trade,
    NoTrade(&'static str),
}

/// Pure multi-layer entry decision. First failing layer wins.
pub fn assess_entry(signal: Signal, regime: Regime, ctx: &EntryCtx) -> EntryVerdict {
    if !ctx.data_ok {
        return EntryVerdict::NoTrade(NO_TRADE_DATA);
    }
    if ctx.emergency_halted {
        return EntryVerdict::NoTrade(NO_TRADE_EMERGENCY);
    }
    if ctx.risk_halted {
        return EntryVerdict::NoTrade(NO_TRADE_RISK);
    }
    if signal == Signal::Flat {
        return EntryVerdict::NoTrade(IDLE_FLAT_SIGNAL);
    }
    if ctx.blocked_regimes.contains(&regime) {
        return EntryVerdict::NoTrade(NO_TRADE_REGIME);
    }
    if ctx.open_positions >= ctx.max_positions.max(1) {
        return EntryVerdict::NoTrade(NO_TRADE_POSITIONS);
    }
    if ctx.open_risk >= ctx.max_open_risk {
        return EntryVerdict::NoTrade(NO_TRADE_OPEN_RISK);
    }
    if !ctx.leverage_ok {
        return EntryVerdict::NoTrade(NO_TRADE_LEVERAGE);
    }
    if !ctx.stop_present {
        return EntryVerdict::NoTrade(NO_TRADE_NO_STOP);
    }
    if !ctx.execution_ok {
        return EntryVerdict::NoTrade(NO_TRADE_EXECUTION);
    }
    EntryVerdict::Trade
}

/// Filter a signal series through the entry gate. Non-flat signals that
/// fail become `Flat`; returns the filtered series + suppression count.
/// Suppressed (idle) bars are reported, never silently dropped.
pub fn gate_signals(
    signals: &[Signal],
    regimes: &[Regime],
    ctx: &EntryCtx,
) -> (Vec<Signal>, usize) {
    let mut suppressed = 0usize;
    let out = signals
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if *s == Signal::Flat {
                return Signal::Flat;
            }
            let reg = regimes.get(i).copied().unwrap_or(Regime::Transition);
            match assess_entry(*s, reg, ctx) {
                EntryVerdict::Trade => *s,
                EntryVerdict::NoTrade(_) => {
                    suppressed += 1;
                    Signal::Flat
                }
            }
        })
        .collect();
    (out, suppressed)
}

/// What the autonomous manager may do with an open position (§10).
/// There is deliberately NO `RemoveStop` variant: the protective stop can
/// be tightened, never removed to avoid realizing a loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorAction {
    Hold,
    Reduce,
    TightenStop,
    Exit(&'static str),
}

/// Pure position monitor. Never increases exposure, never drops the stop:
/// - stop distance gone → `Exit("stop_hit")` (the stop did its job);
/// - regime aborted the thesis or execution unhealthy → `Exit`;
/// - abnormal spread / extreme vol → `TightenStop`;
/// - deep adverse drift while still inside the stop → `Reduce`;
/// - otherwise → `Hold`.
pub fn monitor_position(
    pnl_frac: f64,
    dist_to_stop: f64,
    thesis_aborted: bool,
    spread_abnormal: bool,
    vol_extreme: bool,
) -> MonitorAction {
    if dist_to_stop <= 0.0 {
        return MonitorAction::Exit("stop_hit");
    }
    if thesis_aborted {
        return MonitorAction::Exit("regime_exit");
    }
    if spread_abnormal || vol_extreme {
        return MonitorAction::TightenStop;
    }
    if pnl_frac <= -0.02 {
        return MonitorAction::Reduce;
    }
    MonitorAction::Hold
}

/// Latched emergency halt (§11). Trips permanently on unreliable data,
/// unstable exchange, unknown position state, breached drawdown, abnormal
/// spread/latency/volatility, or repeated order failures. Cleared only by
/// an explicit operator `reset` — never implicitly.
#[derive(Debug, Clone, Default)]
pub struct EmergencyState {
    halted: bool,
    reason: Option<String>,
    consecutive_failures: u32,
}

impl EmergencyState {
    pub fn halt(&mut self, reason: &str) {
        self.halted = true;
        self.reason = Some(reason.to_string());
    }

    /// Record one order failure; halts when `consecutive_failures >= limit`.
    pub fn note_order_result(&mut self, ok: bool, limit: u32) {
        if self.halted {
            return;
        }
        if ok {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures += 1;
            if self.consecutive_failures >= limit.max(1) {
                self.halt("repeated_order_failures");
            }
        }
    }

    /// Evaluate one health snapshot. Returns the halt reason if this call
    /// trips (or the latched reason if already halted).
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &mut self,
        data_ok: bool,
        exchange_ok: bool,
        state_known: bool,
        drawdown_within_limit: bool,
        spread_ok: bool,
        latency_ok: bool,
        vol_ok: bool,
    ) -> Option<String> {
        if self.halted {
            return self.reason.clone();
        }
        let bad = if !data_ok {
            Some("data_unreliable")
        } else if !exchange_ok {
            Some("exchange_unstable")
        } else if !state_known {
            Some("position_state_unknown")
        } else if !drawdown_within_limit {
            Some("drawdown_limit_breach")
        } else if !spread_ok {
            Some("spread_abnormal")
        } else if !latency_ok {
            Some("latency_abnormal")
        } else if !vol_ok {
            Some("volatility_extreme")
        } else {
            None
        };
        if let Some(r) = bad {
            self.halt(r);
        }
        self.reason.clone()
    }

    pub fn halted(&self) -> bool {
        self.halted
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// Explicit operator reset after review. The only path out of halt.
    pub fn reset(&mut self) {
        self.halted = false;
        self.reason = None;
        self.consecutive_failures = 0;
    }
}

/// Deployment gate (§13): every check must pass before a strategy version
/// may trade. One `false` anywhere → DO NOT DEPLOY.
#[derive(Debug, Clone, Default)]
pub struct DeploymentGate {
    pub no_lookahead: bool,
    pub no_leakage: bool,
    pub realistic_costs: bool,
    pub oos_validated: bool,
    pub walkforward_stable: bool,
    pub monte_carlo_ok: bool,
    pub sensitivity_ok: bool,
    pub drawdown_ok: bool,
    pub execution_stress_ok: bool,
    pub risk_limits_ok: bool,
    pub stop_verified: bool,
    pub leverage_ok: bool,
    pub killswitch_verified: bool,
}

impl DeploymentGate {
    /// All-blocking failures, machine-readable. Empty = deployable.
    pub fn failures(&self) -> Vec<&'static str> {
        let mut f = Vec::new();
        if !self.no_lookahead { f.push("lookahead_bias"); }
        if !self.no_leakage { f.push("data_leakage"); }
        if !self.realistic_costs { f.push("unrealistic_costs"); }
        if !self.oos_validated { f.push("oos_not_validated"); }
        if !self.walkforward_stable { f.push("walkforward_unstable"); }
        if !self.monte_carlo_ok { f.push("monte_carlo_failed"); }
        if !self.sensitivity_ok { f.push("parameter_fragile"); }
        if !self.drawdown_ok { f.push("drawdown_exceeded"); }
        if !self.execution_stress_ok { f.push("execution_sensitive"); }
        if !self.risk_limits_ok { f.push("risk_limits_unverified"); }
        if !self.stop_verified { f.push("stop_unverified"); }
        if !self.leverage_ok { f.push("leverage_unsafe"); }
        if !self.killswitch_verified { f.push("killswitch_unverified"); }
        f
    }

    pub fn deploy_allowed(&self) -> bool {
        self.failures().is_empty()
    }
}

/// One immutable strategy version (§14). Never mutated after publish;
/// the registry only appends.
#[derive(Debug, Clone)]
pub struct StrategyVersion {
    pub id: String,
    pub version: String,
    pub params: String,
    pub data_range: String,
    pub train_range: String,
    pub oos_range: String,
    pub gate: DeploymentGate,
    pub risk: String,
    pub exec: String,
}

/// Append-only version registry. `publish` never overwrites: re-publishing
/// an id+version pair appends a new record (history is the audit trail).
#[derive(Debug, Default)]
pub struct VersionRegistry {
    versions: Vec<StrategyVersion>,
}

impl VersionRegistry {
    pub fn publish(&mut self, v: StrategyVersion) {
        self.versions.push(v);
    }

    pub fn len(&self) -> usize {
        self.versions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Latest record for `id` (last published wins, history retained).
    pub fn latest(&self, id: &str) -> Option<&StrategyVersion> {
        self.versions.iter().rev().find(|v| v.id == id)
    }
}

/// Autonomous research-loop stages (§12). Auto-generated changes flow
/// forward only through validation; failures return to research, never to
/// live trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopStage {
    Trade,
    Record,
    Analyze,
    Research,
    Backtest,
    WalkForward,
    MonteCarlo,
    Validate,
    Shadow,
    Deploy,
    Halt,
}

/// Advance the loop. `checks_passed` = the current stage's gates green;
/// `critical` = a safety failure (halt now, operator review required).
pub fn advance_loop(stage: LoopStage, checks_passed: bool, critical: bool) -> LoopStage {
    if critical {
        return LoopStage::Halt;
    }
    match stage {
        LoopStage::Trade => LoopStage::Record,
        LoopStage::Record => LoopStage::Analyze,
        LoopStage::Analyze => LoopStage::Research,
        LoopStage::Research => LoopStage::Backtest,
        LoopStage::Backtest => LoopStage::WalkForward,
        LoopStage::WalkForward => LoopStage::MonteCarlo,
        LoopStage::MonteCarlo => LoopStage::Validate,
        // Validation + shadow are the deployment bulkhead: pass moves
        // forward, fail returns to research — never to Deploy.
        LoopStage::Validate => {
            if checks_passed { LoopStage::Shadow } else { LoopStage::Research }
        }
        LoopStage::Shadow => {
            if checks_passed { LoopStage::Deploy } else { LoopStage::Research }
        }
        LoopStage::Deploy => LoopStage::Trade, // new version live → observe
        LoopStage::Halt => LoopStage::Halt,    // operator reset only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(blocked: &'a [Regime]) -> EntryCtx<'a> {
        EntryCtx {
            data_ok: true,
            emergency_halted: false,
            risk_halted: false,
            blocked_regimes: blocked,
            open_positions: 0,
            max_positions: 3,
            open_risk: 0.0,
            max_open_risk: 0.06,
            leverage_ok: true,
            stop_present: true,
            execution_ok: true,
        }
    }

    #[test]
    fn layers_fail_in_order() {
        let blocked = [Regime::HighVolatility];
        let mut c = ctx(&blocked);
        assert_eq!(assess_entry(Signal::Long, Regime::TrendingUp, &c), EntryVerdict::Trade);
        c.data_ok = false;
        assert_eq!(assess_entry(Signal::Long, Regime::TrendingUp, &c), EntryVerdict::NoTrade(NO_TRADE_DATA));
        c.data_ok = true;
        c.risk_halted = true;
        assert_eq!(assess_entry(Signal::Long, Regime::TrendingUp, &c), EntryVerdict::NoTrade(NO_TRADE_RISK));
        c.risk_halted = false;
        assert_eq!(
            assess_entry(Signal::Flat, Regime::TrendingUp, &c),
            EntryVerdict::NoTrade(IDLE_FLAT_SIGNAL)
        );
        assert_eq!(
            assess_entry(Signal::Long, Regime::HighVolatility, &c),
            EntryVerdict::NoTrade(NO_TRADE_REGIME)
        );
        c.open_positions = 3;
        assert_eq!(
            assess_entry(Signal::Long, Regime::TrendingUp, &c),
            EntryVerdict::NoTrade(NO_TRADE_POSITIONS)
        );
        c.open_positions = 0;
        c.stop_present = false;
        assert_eq!(
            assess_entry(Signal::Long, Regime::TrendingUp, &c),
            EntryVerdict::NoTrade(NO_TRADE_NO_STOP)
        );
    }

    #[test]
    fn gate_suppresses_and_counts() {
        let blocked = [Regime::HighVolatility];
        let c = ctx(&blocked);
        let sig = vec![Signal::Long, Signal::Long, Signal::Flat, Signal::Short];
        let reg = vec![Regime::TrendingUp, Regime::HighVolatility, Regime::TrendingUp, Regime::Ranging];
        let (out, n) = gate_signals(&sig, &reg, &c);
        assert_eq!(out, vec![Signal::Long, Signal::Flat, Signal::Flat, Signal::Short]);
        assert_eq!(n, 1);
    }

    #[test]
    fn monitor_never_drops_the_stop() {
        // No RemoveStop variant exists; worst case tightens or exits.
        assert_eq!(monitor_position(0.01, 0.01, false, false, false), MonitorAction::Hold);
        assert_eq!(monitor_position(0.05, 0.01, false, true, false), MonitorAction::TightenStop);
        assert_eq!(monitor_position(0.05, 0.01, false, false, true), MonitorAction::TightenStop);
        assert_eq!(monitor_position(-0.025, 0.005, false, false, false), MonitorAction::Reduce);
        assert_eq!(monitor_position(-0.05, 0.0, false, false, false), MonitorAction::Exit("stop_hit"));
        assert_eq!(monitor_position(0.05, 0.01, true, false, false), MonitorAction::Exit("regime_exit"));
    }

    #[test]
    fn emergency_latches_and_needs_reset() {
        let mut e = EmergencyState::default();
        assert_eq!(e.check(true, true, true, true, true, true, true), None);
        assert_eq!(e.check(true, true, false, true, true, true, true).as_deref(), Some("position_state_unknown"));
        assert!(e.halted());
        // Latched: healthy snapshot does not clear.
        assert!(e.check(true, true, true, true, true, true, true).is_some());
        e.reset();
        assert!(!e.halted());
        assert_eq!(e.check(true, true, true, true, true, true, true), None);
    }

    #[test]
    fn order_failures_trip() {
        let mut e = EmergencyState::default();
        e.note_order_result(false, 3);
        e.note_order_result(false, 3);
        assert!(!e.halted());
        e.note_order_result(false, 3);
        assert!(e.halted());
        assert_eq!(e.reason(), Some("repeated_order_failures"));
    }

    #[test]
    fn gate_blocks_single_failure() {
        let mut g = DeploymentGate {
            no_lookahead: true, no_leakage: true, realistic_costs: true,
            oos_validated: true, walkforward_stable: true, monte_carlo_ok: true,
            sensitivity_ok: true, drawdown_ok: true, execution_stress_ok: true,
            risk_limits_ok: true, stop_verified: true, leverage_ok: true,
            killswitch_verified: true,
        };
        assert!(g.deploy_allowed());
        g.sensitivity_ok = false;
        assert!(!g.deploy_allowed());
        assert_eq!(g.failures(), vec!["parameter_fragile"]);
    }

    #[test]
    fn registry_appends_never_overwrites() {
        let mk = |v: &str| StrategyVersion {
            id: "mom".into(), version: v.into(), params: String::new(),
            data_range: String::new(), train_range: String::new(), oos_range: String::new(),
            gate: DeploymentGate::default(), risk: String::new(), exec: String::new(),
        };
        let mut r = VersionRegistry::default();
        r.publish(mk("3.2.0"));
        r.publish(mk("3.2.1"));
        assert_eq!(r.len(), 2);
        assert_eq!(r.latest("mom").unwrap().version, "3.2.1");
        assert!(r.latest("nope").is_none());
    }

    #[test]
    fn loop_never_shortcircuits_validation() {
        assert_eq!(advance_loop(LoopStage::Validate, true, false), LoopStage::Shadow);
        assert_eq!(advance_loop(LoopStage::Validate, false, false), LoopStage::Research);
        assert_eq!(advance_loop(LoopStage::Shadow, true, false), LoopStage::Deploy);
        assert_eq!(advance_loop(LoopStage::Shadow, false, false), LoopStage::Research);
        assert_eq!(advance_loop(LoopStage::Deploy, true, false), LoopStage::Trade);
        assert_eq!(advance_loop(LoopStage::Backtest, true, true), LoopStage::Halt);
        assert_eq!(advance_loop(LoopStage::Halt, true, false), LoopStage::Halt);
    }
}
