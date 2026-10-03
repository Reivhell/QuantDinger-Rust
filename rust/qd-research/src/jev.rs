//! JEV — typed judgment/decision layer between proposers and execution.
//!
//! Architecture: a strategy engine or LLM proposes a trade; JEV formalizes
//! the proposal into a typed decision, validates it deterministically,
//! enforces the hard risk contract, and only then hands a
//! [`ValidatedDecision`] to the execution gate. The LLM NEVER controls
//! execution directly: `execute(ValidatedDecision)`, never `execute(llm_output)`.
//!
//! Layers, in authority order (SAFETY > RISK > DATA > EXECUTION > STRATEGY > LLM):
//! 1. Schema validation — every field present, typed, in range (§1).
//! 2. Scorecard — mandatory evidence checks, all must PASS (§7).
//! 3. Risk contract — SL ≤ 3%, leverage policy, exposure caps, no
//!    martingale/averaging-down (§4). The proposer cannot modify these.
//! 4. State machine — legal transitions only, expiry enforced (§2, §8).
//! 5. Execution gate — accepts ONLY [`ValidatedDecision`] (§10).
//!
//! Fail-closed throughout (§11): UNKNOWN / STALE / INVALID / MISSING /
//! CONFLICTING input → [`Action::NoTrade`]. The most important capability
//! of this layer is saying NO (§3).
//!
//! Confidence is advisory only (§5): it is recorded on the decision and
//! reported, but position sizing comes from the risk engine
//! ([`crate::risk::adaptive_qty`]), never from confidence.

use crate::backtest::Signal;
use crate::regime::Regime;

/// BUY / SELL / HOLD / NO_TRADE (§1). Anything that is not an explicit,
/// fully-validated BUY/SELL reads as NO_TRADE downstream — HOLD keeps a
/// position, NO_TRADE forbids opening one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
    NoTrade,
}

/// Structured evidence a proposer must supply (§6). Free-form reasoning may
/// accompany a decision but is NEVER sufficient alone: every flag below
/// must be present, and the mandatory ones must be true.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Evidence {
    pub trend: bool,
    pub momentum: bool,
    pub volatility_ok: bool,
    pub volume_ok: bool,
    pub regime_supported: bool,
    pub liquidity_ok: bool,
    pub higher_timeframe_ok: bool,
    pub strategy_signal: bool,
}

/// Risk declaration attached to the proposal (§1). Checked against the
/// hard contract (§4); the proposer declares, JEV disposes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskTerms {
    /// Stop-loss distance as fraction of entry (mandatory, ≤ [`MAX_SL`]).
    pub stop_loss: f64,
    /// Take-profit distance as fraction of entry (0 = disabled).
    pub take_profit: f64,
    /// Position notional as fraction of equity.
    pub position_size: f64,
    /// Margin multiple (1.0 = spot; hard cap [`crate::risk::MAX_LEVERAGE`]).
    pub leverage: f64,
    /// Declared risk per trade as fraction of equity.
    pub risk_per_trade: f64,
    /// Declared worst-case loss as fraction of equity.
    pub max_loss: f64,
    /// Declared total portfolio exposure as multiple of equity.
    pub portfolio_exposure: f64,
}

/// Raw proposal from a strategy engine or LLM (§1, §10). Untrusted input:
/// nothing here is actionable until [`validate`] succeeds.
#[derive(Debug, Clone)]
pub struct RawDecision {
    pub action: Action,
    pub asset: String,
    pub timeframe: String,
    pub strategy_id: String,
    pub strategy_version: String,
    /// Advisory only — never sizes positions (§5).
    pub confidence: f64,
    pub regime: Regime,
    pub entry: f64,
    pub terms: RiskTerms,
    /// Free-form text: recorded for audit, never a pass criterion (§6).
    pub reasoning: String,
    /// Evidence flags: the mandatory subset must all hold (§6, §7).
    pub evidence: Evidence,
    /// Bar timestamp the decision was made on (seconds).
    pub timestamp: i64,
    /// Bar timestamp after which the decision is dead (§8).
    pub expiry: i64,
}

/// Caps the proposer cannot move (§4). The LLM cannot modify these; they
/// are module constants, not config knobs on the decision.
pub struct RiskContract {
    /// Maximum SL distance: 3%.
    pub max_sl: f64,
    /// Maximum leverage JEV will approve (stricter than the risk module's
    /// absolute ceiling [`crate::risk::MAX_LEVERAGE`]: 5x reads as
    /// excessive margin even when the liquidation buffer math passes —
    /// preferred leverage is 1x, no unnecessary margin).
    pub max_leverage: f64,
    /// Maximum position notional as fraction of equity.
    pub max_position: f64,
    /// Maximum portfolio exposure as multiple of equity.
    pub max_portfolio_exposure: f64,
    /// Liquidation-buffer multiple over the stop distance (§2 leverage rule).
    pub min_liq_buffer: f64,
    /// Maximum bar age of market data before it reads as STALE (§8, §11).
    pub max_data_age_bars: i64,
}

impl Default for RiskContract {
    fn default() -> Self {
        Self {
            max_sl: MAX_SL,
            max_leverage: MAX_JEV_LEVERAGE,
            max_position: 0.10,
            max_portfolio_exposure: 1.0,
            min_liq_buffer: 3.0,
            max_data_age_bars: 1,
        }
    }
}

/// Maximum SL distance: 3% (§4).
pub const MAX_SL: f64 = 0.03;
/// Maximum leverage JEV approves: 3x. Above this reads as unnecessary
/// margin (§4) even when the liquidation buffer covers the stop — the
/// risk module's 5x remains the absolute ceiling, JEV is stricter.
pub const MAX_JEV_LEVERAGE: f64 = 3.0;

/// Machine-readable rejection reasons (§3, §13). Logged on every rejection;
/// a rejected trade is evidence the layer worked, not a system failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    InvalidSchema,
    EvidenceMissing,
    StaleData,
    Expired,
    RegimeUnsupported,
    StopTooWide,
    StopMissing,
    LeverageUnsafe,
    PositionTooLarge,
    ExposureExceeded,
    RiskHalted,
    ExecutionUnhealthy,
    StateViolation,
}

/// One mandatory scorecard line (§7). All lines must PASS for VALIDATED —
/// there is no "85/100 = buy": mandatory constraints and evidence outrank
/// any cosmetic score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreLine {
    pub name: &'static str,
    pub pass: bool,
}

/// Deterministic pre-trade scorecard (§7). `passed()` is true only when
/// every mandatory line passes.
#[derive(Debug, Clone, PartialEq)]
pub struct Scorecard {
    pub lines: Vec<ScoreLine>,
}

impl Scorecard {
    pub fn passed(&self) -> bool {
        !self.lines.is_empty() && self.lines.iter().all(|l| l.pass)
    }

    pub fn failures(&self) -> Vec<&'static str> {
        self.lines.iter().filter(|l| !l.pass).map(|l| l.name).collect()
    }
}

/// Decision lifecycle (§2). Legal flow:
/// OBSERVE → ANALYZE → SIGNAL → RISK_CHECK → VALIDATED → EXECUTE → MONITOR → EXIT.
/// Rejection sinks: REJECTED_SIGNAL / REJECTED_RISK / REJECTED_DATA /
/// REJECTED_EXECUTION / EXPIRED / INVALID_DECISION.
/// ANALYZE → EXECUTE is impossible by construction (no such transition).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionState {
    Observe,
    Analyze,
    Signal,
    RiskCheck,
    Validated,
    Execute,
    Monitor,
    Exit,
    RejectedSignal,
    RejectedRisk,
    RejectedData,
    RejectedExecution,
    Expired,
    InvalidDecision,
}

/// Legal next states for each state. Terminal states (EXIT + all rejection
/// sinks) have no outgoing transitions.
pub fn next_states(s: DecisionState) -> &'static [DecisionState] {
    use DecisionState as S;
    match s {
        S::Observe => &[S::Analyze, S::RejectedData, S::InvalidDecision],
        S::Analyze => &[S::Signal, S::RejectedSignal, S::RejectedData, S::InvalidDecision],
        S::Signal => &[S::RiskCheck, S::RejectedSignal, S::RejectedData, S::Expired, S::InvalidDecision],
        S::RiskCheck => &[S::Validated, S::RejectedRisk, S::RejectedData, S::Expired, S::InvalidDecision],
        S::Validated => &[S::Execute, S::RejectedExecution, S::Expired, S::InvalidDecision],
        S::Execute => &[S::Monitor, S::RejectedExecution, S::Expired],
        S::Monitor => &[S::Exit, S::Expired],
        S::Exit
        | S::RejectedSignal
        | S::RejectedRisk
        | S::RejectedData
        | S::RejectedExecution
        | S::Expired
        | S::InvalidDecision => &[],
    }
}

/// Attempt a transition. `Err` on any illegal jump (notably ANALYZE →
/// EXECUTE); the caller maps the error to a rejection sink.
pub fn transition(from: DecisionState, to: DecisionState) -> Result<DecisionState, RejectReason> {
    if next_states(from).contains(&to) {
        Ok(to)
    } else {
        Err(RejectReason::StateViolation)
    }
}

/// Immutable validated decision (§9, §10). The ONLY type the execution
/// gate accepts. Created solely by [`validate`]; fields are read-only
/// downstream (clone for a new validation — never mutate).
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedDecision {
    /// Unique id, assigned at validation (§9, §13).
    pub id: u64,
    pub action: Action, // Buy or Sell only — Hold/NoTrade never validate
    pub asset: String,
    pub timeframe: String,
    pub strategy_id: String,
    pub strategy_version: String,
    pub confidence: f64, // advisory, echoed for audit (§5)
    pub regime: Regime,
    pub entry: f64,
    pub terms: RiskTerms,
    pub evidence: Evidence,
    pub timestamp: i64,
    pub expiry: i64,
    pub scorecard: Scorecard,
}

/// Full audit record for one proposal, approved or rejected (§13).
#[derive(Debug, Clone)]
pub struct DecisionRecord {
    pub id: u64,
    pub state: DecisionState,
    pub action: Action,
    pub asset: String,
    pub strategy_id: String,
    pub strategy_version: String,
    pub regime: Regime,
    pub timestamp: i64,
    pub reason: Option<RejectReason>,
    pub scorecard: Vec<ScoreLine>,
    pub executed: bool,
}

/// Append-only audit trail (§13). Records every proposal including
/// rejections — `rejections_by_reason` / `by_regime` feed the research
/// layer (§14) and observability (§15).
#[derive(Debug, Default)]
pub struct AuditTrail {
    records: Vec<DecisionRecord>,
    next_id: u64,
}

impl AuditTrail {
    pub fn issue_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub fn record(&mut self, r: DecisionRecord) {
        self.records.push(r);
    }

    /// One-line rejection: issue an id, append the audit record, hand back
    /// the reason for `return Err(...)`. Every early exit in [`validate`]
    /// goes through here — 7 call sites, one shape.
    pub fn reject(
        &mut self,
        raw: &RawDecision,
        state: DecisionState,
        reason: RejectReason,
        lines: &[ScoreLine],
    ) -> RejectReason {
        let id = self.issue_id();
        self.record(reject_record(id, raw, state, reason, lines));
        reason
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Rejection counts by reason (§14: which rules fire, §15: rejection rate).
    pub fn rejections_by_reason(&self) -> Vec<(RejectReason, usize)> {
        use std::collections::BTreeMap;
        let mut m: BTreeMap<u8, (RejectReason, usize)> = BTreeMap::new();
        for r in &self.records {
            if let Some(reason) = r.reason {
                let k = reason as u8;
                m.entry(k).or_insert((reason, 0)).1 += 1;
            }
        }
        m.into_values().collect()
    }

    /// Rejection counts by regime (§14: which regimes produce the most
    /// rejected signals).
    pub fn rejections_by_regime(&self) -> Vec<(String, usize)> {
        use std::collections::BTreeMap;
        let mut m: BTreeMap<String, usize> = BTreeMap::new();
        for r in &self.records {
            if r.reason.is_some() {
                *m.entry(r.regime.name().to_string()).or_default() += 1;
            }
        }
        m.into_iter().collect()
    }

    /// Approval rate over all recorded proposals (§15).
    pub fn approval_rate(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let ok = self.records.iter().filter(|r| r.reason.is_none()).count();
        ok as f64 / self.records.len() as f64
    }
}

/// What JEV needs from the world to judge a proposal. All caller-supplied,
/// all fail-closed (§11): any `false` on data/risk/execution health, or
/// `None` on account/position knowledge, forces NO_TRADE.
#[derive(Debug, Clone, Copy)]
pub struct WorldView {
    /// Market data fresh (timestamp within [`RiskContract::max_data_age_bars`]).
    pub data_fresh: bool,
    /// Position + account state fully known (§11).
    pub state_known: bool,
    /// Kill-switch / risk halt NOT tripped.
    pub risk_ok: bool,
    /// Spread / latency / order-path healthy.
    pub execution_ok: bool,
    /// Current bar timestamp (for expiry + staleness checks).
    pub now: i64,
}

/// Validate a raw proposal end to end: schema → evidence scorecard → state
/// path → risk contract → world health. `Ok(ValidatedDecision)` is the
/// single passport to execution; `Err(RejectReason)` names the first
/// failing layer. HOLD/NoTrade proposals validate to `Err` with no failure
/// semantics — the caller reads them as idle, not errors.
pub fn validate(
    raw: &RawDecision,
    world: &WorldView,
    contract: &RiskContract,
    audit: &mut AuditTrail,
) -> Result<ValidatedDecision, RejectReason> {
    // §1 schema: every field present, typed, in range. Incomplete →
    // rejected, never defaulted.
    if raw.asset.trim().is_empty()
        || raw.timeframe.trim().is_empty()
        || raw.strategy_id.trim().is_empty()
        || raw.strategy_version.trim().is_empty()
        || !raw.entry.is_finite()
        || raw.entry <= 0.0
        || !raw.confidence.is_finite()
        || !(0.0..=1.0).contains(&raw.confidence)
        || raw.expiry <= raw.timestamp
    {
        return Err(audit.reject(raw, DecisionState::InvalidDecision, RejectReason::InvalidSchema, &[]));
    }
    // HOLD / NO_TRADE are idle by construction — not failures, not trades,
    // not audit entries. Only real BUY/SELL proposals enter the trail.
    if matches!(raw.action, Action::Hold | Action::NoTrade) {
        return Err(RejectReason::InvalidSchema);
    }
    // §8 expiry + staleness before any other work: stale reasoning is dead.
    if world.now > raw.expiry {
        return Err(audit.reject(raw, DecisionState::Expired, RejectReason::Expired, &[]));
    }
    if !world.data_fresh || world.now - raw.timestamp > contract.max_data_age_bars {
        return Err(audit.reject(raw, DecisionState::RejectedData, RejectReason::StaleData, &[]));
    }
    if !world.state_known {
        return Err(audit.reject(raw, DecisionState::RejectedData, RejectReason::StaleData, &[]));
    }

    // RISK veto before evidence (§12: SAFETY > RISK > DATA > EXECUTION >
    // STRATEGY > LLM). A halted risk engine rejects even a fully-evidenced
    // proposal — evidence never outranks a halt.
    if !world.risk_ok {
        return Err(audit.reject(raw, DecisionState::RejectedRisk, RejectReason::RiskHalted, &[]));
    }

    // §7 scorecard: every mandatory line must PASS. Confidence appears
    // nowhere here — a 99% LLM conviction passes zero lines by itself (§5).
    let e = &raw.evidence;
    let t = &raw.terms;
    let scorecard = Scorecard {
        lines: vec![
            ScoreLine { name: "TREND", pass: e.trend },
            ScoreLine { name: "MOMENTUM", pass: e.momentum },
            ScoreLine { name: "VOLATILITY", pass: e.volatility_ok },
            ScoreLine { name: "VOLUME", pass: e.volume_ok },
            ScoreLine { name: "REGIME", pass: e.regime_supported && regime_supported(raw.regime) },
            ScoreLine { name: "LIQUIDITY", pass: e.liquidity_ok },
            ScoreLine { name: "HTF_CONTEXT", pass: e.higher_timeframe_ok },
            ScoreLine { name: "STRATEGY_SIGNAL", pass: e.strategy_signal },
            ScoreLine { name: "SL<=3%", pass: t.stop_loss > 0.0 && t.stop_loss <= contract.max_sl },
            ScoreLine {
                name: "LEVERAGE",
                pass: t.leverage >= 1.0
                    && t.leverage <= contract.max_leverage
                    && crate::risk::liquidation_ok(t.leverage, t.stop_loss, contract.min_liq_buffer),
            },
            ScoreLine { name: "EXPOSURE", pass: t.position_size > 0.0 && t.position_size <= contract.max_position
                && t.portfolio_exposure >= 0.0 && t.portfolio_exposure <= contract.max_portfolio_exposure },
            ScoreLine { name: "RISK_TERMS_SANE", pass: t.risk_per_trade > 0.0 && t.risk_per_trade <= contract.max_position
                && t.max_loss > 0.0 && t.take_profit >= 0.0 && t.take_profit.is_finite()
                && t.stop_loss.is_finite() && t.position_size.is_finite() && t.leverage.is_finite() },
            ScoreLine { name: "EXECUTION", pass: world.execution_ok },
            ScoreLine { name: "DATA_FRESHNESS", pass: world.data_fresh },
        ],
    };
    if !scorecard.passed() {
        let reason = first_risk_reason(raw, world, contract);
        let state = match reason {
            RejectReason::RegimeUnsupported => DecisionState::RejectedSignal,
            RejectReason::StaleData => DecisionState::RejectedData,
            RejectReason::ExecutionUnhealthy => DecisionState::RejectedExecution,
            _ => DecisionState::RejectedRisk,
        };
        return Err(audit.reject(raw, state, reason, &scorecard.lines));
    }

    // Walk the legal state path explicitly (§2): a shortcut (e.g.
    // ANALYZE → EXECUTE) cannot typecheck here because every hop goes
    // through `transition`.
    let mut s = DecisionState::Observe;
    for next in [
        DecisionState::Analyze,
        DecisionState::Signal,
        DecisionState::RiskCheck,
        DecisionState::Validated,
    ] {
        s = transition(s, next).map_err(|e| {
            audit.reject(raw, DecisionState::InvalidDecision, e, &scorecard.lines)
        })?;
    }

    let id = audit.issue_id();
    let v = ValidatedDecision {
        id,
        action: raw.action,
        asset: raw.asset.clone(),
        timeframe: raw.timeframe.clone(),
        strategy_id: raw.strategy_id.clone(),
        strategy_version: raw.strategy_version.clone(),
        confidence: raw.confidence,
        regime: raw.regime,
        entry: raw.entry,
        terms: raw.terms,
        evidence: raw.evidence,
        timestamp: raw.timestamp,
        expiry: raw.expiry,
        scorecard: scorecard.clone(),
    };
    audit.record(DecisionRecord {
        id,
        state: s,
        action: raw.action,
        asset: raw.asset.clone(),
        strategy_id: raw.strategy_id.clone(),
        strategy_version: raw.strategy_version.clone(),
        regime: raw.regime,
        timestamp: raw.timestamp,
        reason: None,
        scorecard: scorecard.lines,
        executed: false,
    });
    Ok(v)
}

/// Regimes where a directional proposal has a thesis. Mirrors the reference
/// strategy discipline (TRENDING_*, TRANSITION, BREAKOUT) — enforced here
/// even when the proposer ignores regime entirely (§3, §12).
fn regime_supported(r: Regime) -> bool {
    matches!(r, Regime::TrendingUp | Regime::TrendingDown | Regime::Transition | Regime::Breakout)
}

/// Map the first failing scorecard line to its rejection reason (§3).
fn first_risk_reason(raw: &RawDecision, world: &WorldView, c: &RiskContract) -> RejectReason {
    let e = &raw.evidence;
    let t = &raw.terms;
    if !(e.trend && e.momentum && e.volatility_ok && e.volume_ok && e.liquidity_ok
        && e.higher_timeframe_ok && e.strategy_signal)
    {
        return RejectReason::EvidenceMissing;
    }
    if !e.regime_supported || !regime_supported(raw.regime) {
        return RejectReason::RegimeUnsupported;
    }
    if !(t.stop_loss > 0.0) {
        return RejectReason::StopMissing;
    }
    if t.stop_loss > c.max_sl {
        return RejectReason::StopTooWide;
    }
    if !(t.leverage >= 1.0
        && t.leverage <= c.max_leverage
        && crate::risk::liquidation_ok(t.leverage, t.stop_loss, c.min_liq_buffer))
    {
        return RejectReason::LeverageUnsafe;
    }
    if !(t.position_size > 0.0 && t.position_size <= c.max_position) {
        return RejectReason::PositionTooLarge;
    }
    if !(t.portfolio_exposure >= 0.0 && t.portfolio_exposure <= c.max_portfolio_exposure) {
        return RejectReason::ExposureExceeded;
    }
    if !(t.risk_per_trade > 0.0 && t.max_loss > 0.0) {
        return RejectReason::InvalidSchema;
    }
    if !world.execution_ok {
        return RejectReason::ExecutionUnhealthy;
    }
    RejectReason::RiskHalted
}

fn reject_record(
    id: u64,
    raw: &RawDecision,
    state: DecisionState,
    reason: RejectReason,
    lines: &[ScoreLine],
) -> DecisionRecord {
    DecisionRecord {
        id,
        state,
        action: raw.action,
        asset: raw.asset.clone(),
        strategy_id: raw.strategy_id.clone(),
        strategy_version: raw.strategy_version.clone(),
        regime: raw.regime,
        timestamp: raw.timestamp,
        reason: Some(reason),
        scorecard: lines.to_vec(),
        executed: false,
    }
}

/// Execution gate (§10). Accepts ONLY a [`ValidatedDecision`] — there is no
/// overload taking raw LLM output, so unvalidated proposals cannot
/// typecheck at the call site. Re-checks expiry at fire time: a validated
/// decision that went stale waits for a fresh proposal (§8).
/// Returns the [`Signal`] the backtester/exchange layer acts on.
pub fn execute_gate(d: &ValidatedDecision, now: i64, audit: &mut AuditTrail) -> Result<Signal, RejectReason> {
    if now > d.expiry {
        audit.record(DecisionRecord {
            id: d.id,
            state: DecisionState::Expired,
            action: d.action,
            asset: d.asset.clone(),
            strategy_id: d.strategy_id.clone(),
            strategy_version: d.strategy_version.clone(),
            regime: d.regime,
            timestamp: d.timestamp,
            reason: Some(RejectReason::Expired),
            scorecard: d.scorecard.lines.clone(),
            executed: false,
        });
        return Err(RejectReason::Expired);
    }
    // Mark the audit record executed: immutable history — the decision is
    // never mutated, only its record gains the execution flag (§9).
    if let Some(r) = audit.records.iter_mut().find(|r| r.id == d.id) {
        r.executed = true;
        r.state = DecisionState::Execute;
    }
    Ok(match d.action {
        Action::Buy => Signal::Long,
        Action::Sell => Signal::Short,
        Action::Hold | Action::NoTrade => Signal::Flat, // unreachable: validate rejects these
    })
}

/// Observability snapshot over the audit trail (§15).
#[derive(Debug, Clone, Default)]
pub struct JevStats {
    pub decisions: usize,
    pub approved: usize,
    pub rejected: usize,
    pub expired: usize,
    pub approval_rate: f64,
    pub avg_scorecard_pass: f64,
}

pub fn observe(audit: &AuditTrail) -> JevStats {
    let decisions = audit.len();
    let mut approved = 0usize;
    let mut rejected = 0usize;
    let mut expired = 0usize;
    let mut pass_sum = 0usize;
    let mut pass_n = 0usize;
    for r in &audit.records {
        match r.reason {
            None => approved += 1,
            Some(RejectReason::Expired) => expired += 1,
            Some(_) => rejected += 1,
        }
        if !r.scorecard.is_empty() {
            pass_sum += r.scorecard.iter().filter(|l| l.pass).count();
            pass_n += r.scorecard.len();
        }
    }
    JevStats {
        decisions,
        approved,
        rejected,
        expired,
        approval_rate: audit.approval_rate(),
        avg_scorecard_pass: if pass_n > 0 { pass_sum as f64 / pass_n as f64 } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_raw() -> RawDecision {
        RawDecision {
            action: Action::Buy,
            asset: "BTC-USDT".into(),
            timeframe: "1D".into(),
            strategy_id: "ema-cross".into(),
            strategy_version: "0.1.0".into(),
            confidence: 0.95, // high confidence must NOT bypass anything (§3)
            regime: Regime::TrendingUp,
            entry: 100.0,
            terms: RiskTerms {
                stop_loss: 0.02,
                take_profit: 0.04,
                position_size: 0.05,
                leverage: 1.0,
                risk_per_trade: 0.01,
                max_loss: 0.02,
                portfolio_exposure: 0.5,
            },
            reasoning: "LLM says strong trend".into(), // never sufficient alone
            evidence: Evidence {
                trend: true, momentum: true, volatility_ok: true, volume_ok: true,
                regime_supported: true, liquidity_ok: true, higher_timeframe_ok: true,
                strategy_signal: true,
            },
            timestamp: 105,
            expiry: 115,
        }
    }

    fn world() -> WorldView {
        WorldView { data_fresh: true, state_known: true, risk_ok: true, execution_ok: true, now: 105 }
    }

    // --- schema (§1, §16) ---
    #[test]
    fn valid_decision_passes() {
        let mut a = AuditTrail::default();
        let v = validate(&good_raw(), &world(), &RiskContract::default(), &mut a).expect("good validates");
        assert_eq!(v.action, Action::Buy);
        assert_eq!(a.len(), 1);
        assert!((a.approval_rate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn missing_fields_rejected() {
        let mut a = AuditTrail::default();
        for mut m in [good_raw(), good_raw(), good_raw()] {
            m.asset.clear();
            assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::InvalidSchema));
            m = good_raw(); m.strategy_version.clear();
            assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::InvalidSchema));
            m = good_raw(); m.expiry = m.timestamp; // expiry <= timestamp
            assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::InvalidSchema));
            break;
        }
    }

    #[test]
    fn malformed_values_rejected() {
        let mut a = AuditTrail::default();
        let mut m = good_raw(); m.entry = f64::NAN;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::InvalidSchema));
        let mut m = good_raw(); m.confidence = 1.5;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::InvalidSchema));
        let mut m = good_raw(); m.confidence = f64::INFINITY;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::InvalidSchema));
    }

    #[test]
    fn hold_and_notrade_are_idle_not_trades() {
        let mut a = AuditTrail::default();
        for act in [Action::Hold, Action::NoTrade] {
            let mut m = good_raw(); m.action = act;
            assert!(validate(&m, &world(), &RiskContract::default(), &mut a).is_err());
        }
    }

    // --- risk (§3, §4, §16) ---
    #[test]
    fn jev_says_no_sl_wide() {
        let mut a = AuditTrail::default();
        let mut m = good_raw(); m.terms.stop_loss = 0.038; // 3.8% > 3%
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::StopTooWide));
    }

    #[test]
    fn jev_says_no_leverage_unsafe() {
        let mut a = AuditTrail::default();
        let mut m = good_raw(); m.terms.leverage = 5.0; // liq 20% < 3x stop 6%
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::LeverageUnsafe));
    }

    #[test]
    fn jev_says_no_exposure() {
        let mut a = AuditTrail::default();
        let mut m = good_raw(); m.terms.position_size = 0.50;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::PositionTooLarge));
        let mut m = good_raw(); m.terms.portfolio_exposure = 2.0;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::ExposureExceeded));
    }

    #[test]
    fn jev_says_no_bad_regime() {
        let mut a = AuditTrail::default();
        let mut m = good_raw(); m.regime = Regime::Ranging;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::RegimeUnsupported));
    }

    #[test]
    fn high_confidence_never_bypasses() {
        let mut a = AuditTrail::default();
        let mut m = good_raw();
        m.confidence = 0.999;
        m.terms.stop_loss = 0.05; // still rejected despite near-certainty
        m.evidence.trend = false; // and missing evidence still fails
        assert!(validate(&m, &world(), &RiskContract::default(), &mut a).is_err());
    }

    // --- state (§2, §8, §16) ---
    #[test]
    fn analyze_never_jumps_to_execute() {
        assert_eq!(
            transition(DecisionState::Analyze, DecisionState::Execute),
            Err(RejectReason::StateViolation)
        );
        // Every legal hop succeeds; terminal states have no exits.
        let mut s = DecisionState::Observe;
        for n in [DecisionState::Analyze, DecisionState::Signal, DecisionState::RiskCheck, DecisionState::Validated] {
            s = transition(s, n).expect("legal hop");
        }
        assert_eq!(s, DecisionState::Validated);
        assert!(next_states(DecisionState::Exit).is_empty());
        assert!(next_states(DecisionState::RejectedRisk).is_empty());
    }

    #[test]
    fn expired_and_stale_rejected() {
        let mut a = AuditTrail::default();
        let mut m = good_raw(); m.timestamp = 90; // older than max_data_age
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::StaleData));
        let m = good_raw();
        let mut w = world(); w.now = 200; // past expiry
        assert_eq!(validate(&m, &w, &RiskContract::default(), &mut a), Err(RejectReason::Expired));
        // Gate re-checks expiry at fire time.
        let v = validate(&good_raw(), &world(), &RiskContract::default(), &mut a).expect("valid");
        assert_eq!(execute_gate(&v, 500, &mut a), Err(RejectReason::Expired));
    }

    #[test]
    fn duplicate_execution_marks_record_once() {
        let mut a = AuditTrail::default();
        let v = validate(&good_raw(), &world(), &RiskContract::default(), &mut a).expect("valid");
        assert_eq!(execute_gate(&v, 106, &mut a), Ok(Signal::Long));
        assert_eq!(execute_gate(&v, 107, &mut a), Ok(Signal::Long)); // idempotent signal, one record
        assert_eq!(a.len(), 1, "no silent new records on re-fire");
        assert!(a.records[0].executed);
    }

    // --- safety (§11, §16) ---
    #[test]
    fn unknown_or_unhealthy_world_is_notrade() {
        let mut a = AuditTrail::default();
        let mut w = world(); w.state_known = false; // unknown position state
        assert!(validate(&good_raw(), &w, &RiskContract::default(), &mut a).is_err());
        let mut w = world(); w.data_fresh = false; // stale market data
        assert!(validate(&good_raw(), &w, &RiskContract::default(), &mut a).is_err());
        let mut w = world(); w.risk_ok = false; // risk engine halted
        assert_eq!(validate(&good_raw(), &w, &RiskContract::default(), &mut a), Err(RejectReason::RiskHalted));
        let mut w = world(); w.execution_ok = false; // execution unhealthy
        assert_eq!(validate(&good_raw(), &w, &RiskContract::default(), &mut a), Err(RejectReason::ExecutionUnhealthy));
    }

    // --- integration + research linkage (§12, §14, §15, §16) ---
    #[test]
    fn audit_trail_feeds_research() {
        let mut a = AuditTrail::default();
        let c = RiskContract::default();
        let _ = validate(&good_raw(), &world(), &c, &mut a);
        let mut bad = good_raw(); bad.terms.stop_loss = 0.05;
        let _ = validate(&bad, &world(), &c, &mut a);
        let mut bad2 = good_raw(); bad2.regime = Regime::HighVolatility;
        let _ = validate(&bad2, &world(), &c, &mut a);
        assert_eq!(a.len(), 3);
        assert!((a.approval_rate() - 1.0 / 3.0).abs() < 1e-12);
        let by_reason = a.rejections_by_reason();
        assert_eq!(by_reason.len(), 2, "two distinct rules fired: {by_reason:?}");
        let by_regime = a.rejections_by_regime();
        assert!(by_regime.iter().any(|(r, _)| r == "HIGH_VOLATILITY"));
        let s = observe(&a);
        assert_eq!((s.decisions, s.approved, s.rejected), (3, 1, 2));
    }

    #[test]
    fn scorecard_is_all_or_nothing() {
        let mut a = AuditTrail::default();
        let v = validate(&good_raw(), &world(), &RiskContract::default(), &mut a).expect("valid");
        assert!(v.scorecard.passed());
        assert!(v.scorecard.failures().is_empty());
        let mut m = good_raw(); m.evidence.volume_ok = false;
        assert_eq!(validate(&m, &world(), &RiskContract::default(), &mut a), Err(RejectReason::EvidenceMissing));
    }

    #[test]
    fn sell_path_validates_and_gates() {
        let mut a = AuditTrail::default();
        let mut m = good_raw();
        m.action = Action::Sell;
        m.regime = Regime::TrendingDown;
        let v = validate(&m, &world(), &RiskContract::default(), &mut a).expect("sell validates");
        assert_eq!(execute_gate(&v, 106, &mut a), Ok(Signal::Short));
    }
}
