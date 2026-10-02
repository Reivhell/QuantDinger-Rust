//! Port of `backend_api_python/app/services/strategy_v2/protection.py`.
//!
//! Shared position-protection semantics for backtest and live execution:
//! stop-loss, take-profit, trailing-stop (with activation + scale-in modes),
//! time-limit exits, and intrabar candidate priority.
//!
//! Timestamps are `i64` epoch seconds. The Python code uses `pd.Timestamp`
//! only for the time-limit difference (`total_seconds()`), so epoch seconds
//! carry the full decision semantics without a datetime dependency.
//! Serialization helpers (`metadata`/`from_metadata`) are intentionally not
//! ported — they belong to the live session's snapshot layer, not the engine.

/// Stop-loss cap: ratios above 1.0 (100%) are clamped.
pub const MAX_STOP_LOSS: f64 = 1.0;
/// Take-profit / trailing-activation cap: 5.0 (500%).
pub const MAX_PROFIT: f64 = 5.0;

fn num(value: f64, default: f64) -> f64 {
    let v = if value.is_nan() { default } else { value };
    if v.is_nan() { default } else { v }
}

fn ratio(value: f64, maximum: f64) -> f64 {
    num(value, 0.0).clamp(0.0, maximum)
}

/// Protection parameters. Mirrors `ProtectionSpec` (including clamping).
#[derive(Debug, Clone, PartialEq)]
pub struct ProtectionSpec {
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub trailing_stop_pct: f64,
    pub trailing_activation_pct: f64,
    pub time_limit_seconds: i64,
    pub trailing_rebase_on_scale_in: bool,
}

impl Default for ProtectionSpec {
    fn default() -> Self {
        Self {
            stop_loss_pct: 0.0,
            take_profit_pct: 0.0,
            trailing_stop_pct: 0.0,
            trailing_activation_pct: 0.0,
            time_limit_seconds: 0,
            trailing_rebase_on_scale_in: true,
        }
    }
}

impl ProtectionSpec {
    pub fn new(
        stop_loss_pct: f64,
        take_profit_pct: f64,
        trailing_stop_pct: f64,
        trailing_activation_pct: f64,
        time_limit_seconds: i64,
        trailing_rebase_on_scale_in: bool,
    ) -> Self {
        Self {
            stop_loss_pct: ratio(stop_loss_pct, MAX_STOP_LOSS),
            take_profit_pct: ratio(take_profit_pct, MAX_PROFIT),
            trailing_stop_pct: ratio(trailing_stop_pct, MAX_STOP_LOSS),
            trailing_activation_pct: ratio(trailing_activation_pct, MAX_PROFIT),
            time_limit_seconds: time_limit_seconds.max(0),
            trailing_rebase_on_scale_in,
        }
    }

    pub fn enabled(&self) -> bool {
        self.stop_loss_pct > 0.0
            || self.take_profit_pct > 0.0
            || self.trailing_stop_pct > 0.0
            || self.time_limit_seconds > 0
    }
}

/// Mutable protected-position state. Mirrors `ProtectionState`.
#[derive(Debug, Clone)]
pub struct ProtectionState {
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub spec: ProtectionSpec,
    pub opened_at_secs: i64,
    pub highest_price: f64,
    pub lowest_price: f64,
    pub trailing_active: bool,
}

impl ProtectionState {
    pub fn open(
        symbol: &str,
        side: &str,
        entry_price: f64,
        spec: ProtectionSpec,
        opened_at_secs: i64,
    ) -> Self {
        // Python: float(entry_price or 0.0) — falsy (0/None) -> 0.0,
        // but NaN/inf are truthy and pass through untouched.
        let price = if entry_price == 0.0 {
            0.0
        } else {
            entry_price
        };
        Self {
            symbol: symbol.to_string(),
            side: side.to_lowercase(),
            entry_price: price,
            spec,
            opened_at_secs,
            highest_price: price,
            lowest_price: price,
            trailing_active: false,
        }
    }

    /// Same-side basket enlargement without discarding active protection.
    /// Mirrors `apply_scale_in`.
    pub fn apply_scale_in(
        &mut self,
        entry_price: f64,
        fill_price: f64,
        spec: Option<ProtectionSpec>,
        scaled_at_secs: Option<i64>,
    ) {
        if entry_price > 0.0 && entry_price.is_finite() {
            self.entry_price = entry_price;
        }
        if let Some(s) = spec {
            self.spec = s;
        }
        let mut reference = fill_price;
        if !(reference > 0.0) || !reference.is_finite() {
            reference = self.entry_price;
        }
        if self.spec.trailing_rebase_on_scale_in {
            self.trailing_active = false;
            self.highest_price = self.entry_price.max(reference);
            self.lowest_price = self.entry_price.min(reference);
            if let Some(ts) = scaled_at_secs {
                self.opened_at_secs = ts;
            }
            return;
        }
        if self.trailing_active {
            return;
        }
        self.highest_price = self.entry_price.max(reference);
        self.lowest_price = self.entry_price.min(reference);
    }
}

/// Fired protection exit. Mirrors `ProtectionDecision` (timestamp as epoch).
#[derive(Debug, Clone, PartialEq)]
pub struct ProtectionDecision {
    pub symbol: String,
    pub side: String,
    pub reason: &'static str,
    pub price: f64,
    pub trigger_price: f64,
    pub timestamp_secs: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrabarMode {
    Conservative,
    Balanced,
    Aggressive,
}

/// Deterministic protection evaluator used by both execution modes.
pub struct ProtectionEngine {
    pub intrabar_mode: IntrabarMode,
}

impl Default for ProtectionEngine {
    fn default() -> Self {
        Self {
            intrabar_mode: IntrabarMode::Conservative,
        }
    }
}

impl ProtectionEngine {
    pub fn new(mode: &str) -> Self {
        let intrabar_mode = match mode.trim().to_lowercase().as_str() {
            "aggressive" => IntrabarMode::Aggressive,
            "balanced" => IntrabarMode::Balanced,
            _ => IntrabarMode::Conservative,
        };
        Self { intrabar_mode }
    }

    /// Bar-close evaluation with gap handling. Mirrors `evaluate_bar`.
    pub fn evaluate_bar(
        &self,
        state: &mut ProtectionState,
        timestamp_secs: i64,
        open_price: f64,
        high_price: f64,
        low_price: f64,
    ) -> Option<ProtectionDecision> {
        let open = num(open_price, 0.0);
        let high = open.max(num(high_price, open));
        let low = open.min(num(low_price, open));
        let candidates = self.bar_candidates(state, timestamp_secs, open, high, low);
        let decision = Self::choose(self.intrabar_mode, &candidates, open);
        if decision.is_none() {
            state.highest_price = state.highest_price.max(high);
            state.lowest_price = state.lowest_price.min(low);
        }
        decision
    }

    /// Tick evaluation. Mirrors `evaluate_price`.
    pub fn evaluate_price(
        &self,
        state: &mut ProtectionState,
        timestamp_secs: i64,
        price: f64,
    ) -> Option<ProtectionDecision> {
        let current = num(price, 0.0);
        if !(current > 0.0) {
            return None;
        }
        state.highest_price = state.highest_price.max(current);
        state.lowest_price = state.lowest_price.min(current);
        let candidates = self.price_candidates(state, timestamp_secs, current);
        Self::choose(self.intrabar_mode, &candidates, current)
    }

    fn bar_candidates(
        &self,
        state: &mut ProtectionState,
        ts: i64,
        open: f64,
        high: f64,
        low: f64,
    ) -> Vec<ProtectionDecision> {
        let entry = state.entry_price;
        let is_long = state.side == "long";
        let sl = state.spec.stop_loss_pct;
        let tp = state.spec.take_profit_pct;
        let mut out = Vec::new();
        if sl > 0.0 {
            let trigger = entry * if is_long { 1.0 - sl } else { 1.0 + sl };
            let touched = if is_long { low <= trigger } else { high >= trigger };
            let gapped = if is_long { open <= trigger } else { open >= trigger };
            if touched {
                out.push(decide(state, "stop_loss", if gapped { open } else { trigger }, trigger, ts));
            }
        }
        if let Some(trailing) = Self::trailing_trigger(state) {
            let touched = if is_long { low <= trailing } else { high >= trailing };
            let gapped = if is_long { open <= trailing } else { open >= trailing };
            if touched {
                out.push(decide(
                    state,
                    "trailing_stop",
                    if gapped { open } else { trailing },
                    trailing,
                    ts,
                ));
            }
        }
        if tp > 0.0 {
            let trigger = entry * if is_long { 1.0 + tp } else { 1.0 - tp };
            let touched = if is_long { high >= trigger } else { low <= trigger };
            let gapped = if is_long { open >= trigger } else { open <= trigger };
            if touched {
                out.push(decide(
                    state,
                    "take_profit",
                    if gapped { open } else { trigger },
                    trigger,
                    ts,
                ));
            }
        }
        if Self::time_limit_reached(state, ts) {
            out.push(decide(state, "time_limit", open, open, ts));
        }
        out
    }

    fn price_candidates(
        &self,
        state: &mut ProtectionState,
        ts: i64,
        price: f64,
    ) -> Vec<ProtectionDecision> {
        let entry = state.entry_price;
        let is_long = state.side == "long";
        let sl = state.spec.stop_loss_pct;
        let tp = state.spec.take_profit_pct;
        let mut out = Vec::new();
        if sl > 0.0 {
            let trigger = entry * if is_long { 1.0 - sl } else { 1.0 + sl };
            if (is_long && price <= trigger) || (!is_long && price >= trigger) {
                out.push(decide(state, "stop_loss", price, trigger, ts));
            }
        }
        if let Some(trailing) = Self::trailing_trigger(state) {
            if (is_long && price <= trailing) || (!is_long && price >= trailing) {
                out.push(decide(state, "trailing_stop", price, trailing, ts));
            }
        }
        if tp > 0.0 {
            let trigger = entry * if is_long { 1.0 + tp } else { 1.0 - tp };
            if (is_long && price >= trigger) || (!is_long && price <= trigger) {
                out.push(decide(state, "take_profit", price, trigger, ts));
            }
        }
        if Self::time_limit_reached(state, ts) {
            out.push(decide(state, "time_limit", price, price, ts));
        }
        out
    }

    fn trailing_trigger(state: &mut ProtectionState) -> Option<f64> {
        let spec = &state.spec;
        if !(spec.trailing_stop_pct > 0.0) {
            return None;
        }
        if state.side == "long" {
            if !state.trailing_active
                && state.highest_price >= state.entry_price * (1.0 + spec.trailing_activation_pct)
            {
                state.trailing_active = true;
            }
            if !state.trailing_active {
                return None;
            }
            Some(state.highest_price * (1.0 - spec.trailing_stop_pct))
        } else {
            if !state.trailing_active
                && state.lowest_price <= state.entry_price * (1.0 - spec.trailing_activation_pct)
            {
                state.trailing_active = true;
            }
            if !state.trailing_active {
                return None;
            }
            Some(state.lowest_price * (1.0 + spec.trailing_stop_pct))
        }
    }

    fn time_limit_reached(state: &ProtectionState, timestamp_secs: i64) -> bool {
        let limit = state.spec.time_limit_seconds;
        limit > 0 && timestamp_secs - state.opened_at_secs >= limit
    }

    fn choose(
        mode: IntrabarMode,
        candidates: &[ProtectionDecision],
        reference_price: f64,
    ) -> Option<ProtectionDecision> {
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            return Some(candidates[0].clone());
        }
        let pick = match mode {
            IntrabarMode::Aggressive => {
                // take_profit > trailing_stop > time_limit > stop_loss
                min_by(candidates, &|d| match d.reason {
                    "take_profit" => 0,
                    "trailing_stop" => 1,
                    "time_limit" => 2,
                    "stop_loss" => 3,
                    _ => 9,
                })
            }
            IntrabarMode::Balanced => {
                let mut best = &candidates[0];
                let mut best_dist = (best.trigger_price - reference_price).abs();
                for c in &candidates[1..] {
                    let dist = (c.trigger_price - reference_price).abs();
                    if dist < best_dist {
                        best = c;
                        best_dist = dist;
                    }
                }
                best
            }
            IntrabarMode::Conservative => {
                // stop_loss > trailing_stop > time_limit > take_profit
                min_by(candidates, &|d| match d.reason {
                    "stop_loss" => 0,
                    "trailing_stop" => 1,
                    "time_limit" => 2,
                    "take_profit" => 3,
                    _ => 9,
                })
            }
        };
        Some(pick.clone())
    }
}

fn min_by<'a>(items: &'a [ProtectionDecision], key: &dyn Fn(&ProtectionDecision) -> i32) -> &'a ProtectionDecision {
    let mut best = &items[0];
    let mut best_key = key(best);
    for c in &items[1..] {
        let k = key(c);
        if k < best_key {
            best = c;
            best_key = k;
        }
    }
    best
}

fn decide(
    state: &ProtectionState,
    reason: &'static str,
    price: f64,
    trigger_price: f64,
    timestamp_secs: i64,
) -> ProtectionDecision {
    ProtectionDecision {
        symbol: state.symbol.clone(),
        side: state.side.clone(),
        reason,
        price,
        trigger_price,
        timestamp_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1767225600; // 2026-01-01 00:00:00 UTC

    fn long_state(spec: ProtectionSpec) -> ProtectionState {
        ProtectionState::open("Crypto:BTC/USDT@spot", "long", 100.0, spec, T0)
    }

    // Mirrors test_conservative_intrabar_mode_prioritizes_stop_loss
    #[test]
    fn conservative_prefers_stop_loss_inside_bar() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = long_state(spec);
        let d = ProtectionEngine::new("conservative")
            .evaluate_bar(&mut state, T0 + 4 * 3600, 100.0, 106.0, 97.0)
            .expect("must fire");
        assert_eq!(d.reason, "stop_loss");
        assert!((d.price - 98.0).abs() < 1e-9);
    }

    #[test]
    fn aggressive_prefers_take_profit_inside_bar() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = long_state(spec);
        let d = ProtectionEngine::new("aggressive")
            .evaluate_bar(&mut state, T0 + 4 * 3600, 100.0, 106.0, 97.0)
            .expect("must fire");
        assert_eq!(d.reason, "take_profit");
        assert!((d.price - 105.0).abs() < 1e-9);
    }

    // Mirrors test_backtest_protection_fills_at_gap_open (stop 98, gap to 95)
    #[test]
    fn gap_open_fills_at_open() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = long_state(spec);
        let d = ProtectionEngine::new("conservative")
            .evaluate_bar(&mut state, T0 + 8 * 3600, 95.0, 96.0, 94.0)
            .expect("must fire");
        assert_eq!(d.reason, "stop_loss");
        assert!((d.price - 95.0).abs() < 1e-9);
        assert!((d.trigger_price - 98.0).abs() < 1e-9);
    }

    // Mirrors test_backtest_protection_fills_at_stop_inside_bar
    #[test]
    fn stop_inside_bar_fills_at_trigger() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = long_state(spec);
        let d = ProtectionEngine::new("conservative")
            .evaluate_bar(&mut state, T0 + 8 * 3600, 100.0, 101.0, 97.0)
            .expect("must fire");
        assert_eq!(d.reason, "stop_loss");
        assert!((d.price - 98.0).abs() < 1e-9);
    }

    #[test]
    fn no_touch_updates_extremes() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = long_state(spec);
        let d = ProtectionEngine::new("conservative")
            .evaluate_bar(&mut state, T0 + 3600, 100.0, 101.0, 99.0);
        assert!(d.is_none());
        assert!((state.highest_price - 101.0).abs() < 1e-9);
        assert!((state.lowest_price - 99.0).abs() < 1e-9);
    }

    // Mirrors test_live_protection_uses_price_ticks_without_new_bar
    #[test]
    fn tick_stop_loss_fires() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = long_state(spec);
        let d = ProtectionEngine::new("conservative")
            .evaluate_price(&mut state, T0 + 30, 97.5)
            .expect("must fire");
        assert_eq!(d.reason, "stop_loss");
        assert!((d.price - 97.5).abs() < 1e-9);
    }

    // Mirrors test_scale_in_preserves_an_activated_trailing_peak
    #[test]
    fn scale_in_preserves_activated_trailing_peak() {
        let spec = ProtectionSpec::new(0.0, 0.0, 0.003, 0.01, 0, false);
        let mut state = long_state(spec.clone());
        let engine = ProtectionEngine::new("conservative");
        assert!(engine.evaluate_price(&mut state, T0 + 60, 102.0).is_none());
        assert!(state.trailing_active);
        state.apply_scale_in(95.0, 90.0, Some(spec), None);
        assert!((state.entry_price - 95.0).abs() < 1e-9);
        assert!(state.trailing_active);
        assert!((state.highest_price - 102.0).abs() < 1e-9);
        let d = engine
            .evaluate_price(&mut state, T0 + 120, 101.6)
            .expect("trailing must fire");
        assert_eq!(d.reason, "trailing_stop");
        assert!((d.trigger_price - 102.0 * (1.0 - 0.003)).abs() < 1e-9);
    }

    // Mirrors test_scale_in_rebases_unactivated_trailing_extremes
    #[test]
    fn scale_in_rebases_unactivated_trailing() {
        let spec = ProtectionSpec::new(0.0, 0.0, 0.003, 0.05, 0, false);
        let mut state = long_state(spec.clone());
        state.highest_price = 103.0;
        state.apply_scale_in(95.0, 90.0, Some(spec), None);
        assert!(!state.trailing_active);
        assert!((state.highest_price - 95.0).abs() < 1e-9);
        assert!((state.lowest_price - 90.0).abs() < 1e-9);
    }

    // Mirrors test_scale_in_defaults_to_legacy_trailing_reset
    #[test]
    fn scale_in_defaults_to_legacy_reset() {
        let spec = ProtectionSpec::new(0.0, 0.0, 0.003, 0.01, 0, true);
        let mut state = long_state(spec);
        state.highest_price = 103.0;
        state.trailing_active = true;
        state.apply_scale_in(95.0, 90.0, None, Some(T0 + 86400));
        assert!(!state.trailing_active);
        assert!((state.highest_price - 95.0).abs() < 1e-9);
        assert!((state.lowest_price - 90.0).abs() < 1e-9);
        assert_eq!(state.opened_at_secs, T0 + 86400);
    }

    #[test]
    fn time_limit_fires() {
        let spec = ProtectionSpec::new(0.0, 0.0, 0.0, 0.0, 3600, true);
        let mut state = long_state(spec);
        assert!(
            ProtectionEngine::new("conservative")
                .evaluate_price(&mut state, T0 + 3599, 100.0)
                .is_none()
        );
        let d = ProtectionEngine::new("conservative")
            .evaluate_price(&mut state, T0 + 3600, 100.0)
            .expect("time limit must fire");
        assert_eq!(d.reason, "time_limit");
    }

    #[test]
    fn short_side_mirrors() {
        let spec = ProtectionSpec::new(0.02, 0.05, 0.0, 0.0, 0, true);
        let mut state = ProtectionState::open("X", "short", 100.0, spec, T0);
        let d = ProtectionEngine::new("conservative")
            .evaluate_bar(&mut state, T0 + 3600, 100.0, 103.0, 94.0)
            .expect("must fire");
        assert_eq!(d.reason, "stop_loss");
        assert!((d.price - 102.0).abs() < 1e-9);
    }

    #[test]
    fn spec_clamps_and_disables() {
        let s = ProtectionSpec::new(9.0, 99.0, -1.0, 0.0, -5, true);
        assert_eq!(s.stop_loss_pct, 1.0);
        assert_eq!(s.take_profit_pct, 5.0);
        assert_eq!(s.trailing_stop_pct, 0.0);
        assert!(!ProtectionSpec::default().enabled());
        assert!(s.enabled());
    }

    #[test]
    fn unknown_mode_falls_back_to_conservative() {
        assert_eq!(
            ProtectionEngine::new("whatever").intrabar_mode,
            IntrabarMode::Conservative
        );
    }
}
