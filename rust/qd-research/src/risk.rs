//! Risk management, strictly separated from signal generation.
//!
//! Signals say *which way*; this module says *how much* and *when to stop*.
//! Sizers are pure functions `(price, equity) -> quantity`; kill-switches
//! are stateful guards fed with realized equity.

/// Fixed-fractional sizer: risk `risk_fraction` of equity per trade with a
/// `stop_fraction` stop distance → `qty = equity*risk / (price*stop)`.
/// Clamped to `max_position_notional` and non-negative.
pub fn fixed_fractional_qty(
    price: f64,
    equity: f64,
    risk_fraction: f64,
    stop_fraction: f64,
    max_position_notional: f64,
) -> f64 {
    if price <= 0.0 || equity <= 0.0 || stop_fraction <= 0.0 || risk_fraction <= 0.0 {
        return 0.0;
    }
    let qty = equity * risk_fraction / (price * stop_fraction);
    let max_qty = if max_position_notional > 0.0 { max_position_notional / price } else { qty };
    qty.min(max_qty).max(0.0)
}

/// Volatility-target sizer: scale quantity so the position's trailing
/// `vol_window`-bar realized volatility matches `target_vol` (annualized).
/// Falls back to `fallback_qty` when volatility is unavailable.
pub fn volatility_target_qty(
    price: f64,
    equity: f64,
    realized_vol_ann: Option<f64>,
    target_vol: f64,
    max_position_notional: f64,
    fallback_qty: f64,
) -> f64 {
    match realized_vol_ann {
        Some(rv) if rv > 0.0 && price > 0.0 && equity > 0.0 && target_vol > 0.0 => {
            let notional = equity * (target_vol / rv).min(1.0);
            let qty = notional / price;
            let max_qty = if max_position_notional > 0.0 { max_position_notional / price } else { qty };
            qty.min(max_qty).max(0.0)
        }
        _ => fallback_qty.max(0.0),
    }
}

/// Drawdown-scaled risk fraction: at zero drawdown the full `base_risk`
/// applies; risk linearly shrinks to zero as `current_dd` approaches the
/// `max_dd_allowance` (the KillSwitch trip level). A losing streak — or a
/// deep hole — automatically *reduces* the next trade's risk, never
/// increases it (no martingale, no revenge sizing, §1/§3).
/// Returns 0 on nonsense inputs.
pub fn drawdown_scaled_risk(base_risk: f64, current_dd: f64, max_dd_allowance: f64) -> f64 {
    if !(base_risk > 0.0) || !(max_dd_allowance > 0.0) || !current_dd.is_finite() {
        return 0.0;
    }
    let dd = current_dd.max(0.0);
    if dd >= max_dd_allowance {
        return 0.0;
    }
    base_risk * (1.0 - dd / max_dd_allowance)
}

/// Volatility-scaled risk fraction: full `base_risk` at or below
/// `target_vol_ann`, shrinking proportionally above it. High volatility
/// cuts size; calm markets never inflate it above the base (§3).
/// `None`/non-positive realized vol → full base risk (fail-open would
/// oversize; fail-closed here would halt research on thin data — the
/// KillSwitch remains the hard backstop).
pub fn volatility_scaled_risk(
    base_risk: f64,
    realized_vol_ann: Option<f64>,
    target_vol_ann: f64,
) -> f64 {
    if !(base_risk > 0.0) {
        return 0.0;
    }
    match realized_vol_ann {
        Some(rv) if rv.is_finite() && rv > 0.0 && target_vol_ann > 0.0 => {
            base_risk * (target_vol_ann / rv).min(1.0)
        }
        _ => base_risk,
    }
}

/// Position-count share of the risk budget: with `open` of `max` slots
/// filled, only the remaining share of risk is available. At cap → 0
/// (the entry gate must already block this; the sizer refuses as well —
/// defense in depth, §4). Returns a multiplier in [0, 1].
pub fn position_budget_scale(open_positions: usize, max_positions: usize) -> f64 {
    if max_positions == 0 || open_positions >= max_positions {
        return 0.0;
    }
    (max_positions - open_positions) as f64 / max_positions as f64
}

/// Adaptive sizer inputs (§3): every factor that may *shrink* size.
/// Nothing here can inflate size above `risk_fraction` — conviction,
/// signal strength, and recent wins are deliberately absent.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveSizeCtx {
    /// Base per-trade risk fraction (config `risk_fraction`).
    pub risk_fraction: f64,
    /// Stop distance as fraction of price.
    pub stop_fraction: f64,
    /// Trailing realized vol (annualized), if available.
    pub realized_vol_ann: Option<f64>,
    /// Vol level at/below which full size applies.
    pub target_vol_ann: f64,
    /// Current portfolio drawdown as fraction of peak.
    pub current_dd: f64,
    /// Drawdown allowance (KillSwitch `max_drawdown`).
    pub max_dd_allowance: f64,
    /// Open / max position counts.
    pub open_positions: usize,
    pub max_positions: usize,
    /// Hard notional cap (0 = none beyond the risk math).
    pub max_position_notional: f64,
}

/// Adaptive position size: fixed-fractional base × vol scale × drawdown
/// scale × position-budget scale, clamped to `max_position_notional`.
/// Monotone non-increasing in risk, vol, drawdown, and crowding — a
/// strong signal NEVER receives a larger position than an identical weak
/// one (§3). Pure function, no loss-memory: consecutive losses shrink
/// size only through realized equity/drawdown, and size recovers only
/// through recovery, never through escalation.
pub fn adaptive_qty(price: f64, equity: f64, c: AdaptiveSizeCtx) -> f64 {
    if price <= 0.0 || equity <= 0.0 || c.stop_fraction <= 0.0 {
        return 0.0;
    }
    let r = volatility_scaled_risk(c.risk_fraction, c.realized_vol_ann, c.target_vol_ann);
    let r = drawdown_scaled_risk(r, c.current_dd, c.max_dd_allowance);
    let r = r * position_budget_scale(c.open_positions, c.max_positions);
    if r <= 0.0 {
        return 0.0;
    }
    let qty = equity * r / (price * c.stop_fraction);
    let max_qty = if c.max_position_notional > 0.0 { c.max_position_notional / price } else { qty };
    qty.min(max_qty).max(0.0)
}

/// Portfolio / account kill-switches. Feed realized equity bar by bar;
/// `tripped()` goes true permanently once any guard fires (latched — a
/// real desk does not auto-resume after a max-loss day).
#[derive(Debug, Clone)]
pub struct KillSwitch {
    /// Halt if intraday equity drops this fraction below the day's first mark.
    pub max_daily_loss: f64,
    /// Halt if equity drops this fraction below the 5-session running peak
    /// (weekly loss proxy on daily bars; 0 = disabled).
    pub max_weekly_loss: f64,
    /// Halt if equity drops this fraction below the running peak.
    pub max_drawdown: f64,
    /// Halt if exposure notional exceeds this multiple of equity.
    pub max_exposure: f64,
    /// Halt if correlated exposure (same-direction notional across
    /// correlated names) exceeds this multiple of equity (0 = disabled).
    pub max_correlated_exposure: f64,
    day_start: f64,
    week_peak: f64,
    week_len: u32,
    week_count: u32,
    peak: f64,
    tripped_reason: Option<String>,
}

impl KillSwitch {
    pub fn new(max_daily_loss: f64, max_drawdown: f64, max_exposure: f64) -> Self {
        Self {
            max_daily_loss,
            max_weekly_loss: 0.0,
            max_drawdown,
            max_exposure,
            max_correlated_exposure: 0.0,
            day_start: f64::NAN,
            week_peak: f64::NEG_INFINITY,
            week_len: 0,
            week_count: 0,
            peak: f64::NEG_INFINITY,
            tripped_reason: None,
        }
    }

    /// Full constructor including the weekly-loss and correlated-exposure
    /// guards (§4). `week_bars` = bars per week on the traded timeframe
    /// (5 for daily bars); the weekly peak resets every `week_bars` bars.
    pub fn with_weekly(
        max_daily_loss: f64,
        max_weekly_loss: f64,
        max_drawdown: f64,
        max_exposure: f64,
        max_correlated_exposure: f64,
        week_bars: u32,
    ) -> Self {
        let mut k = Self::new(max_daily_loss, max_drawdown, max_exposure);
        k.max_weekly_loss = max_weekly_loss;
        k.max_correlated_exposure = max_correlated_exposure;
        k.week_len = week_bars.max(1);
        k
    }

    /// `new_day` resets the daily reference. Returns the trip reason if
    /// this update trips a guard.
    pub fn update(&mut self, equity: f64, exposure_notional: f64, new_day: bool) -> Option<String> {
        self.update_full(equity, exposure_notional, 0.0, new_day)
    }

    /// Full update with correlated-exposure notional (§4). Pass 0 when the
    /// book holds a single name (no correlation dimension).
    pub fn update_full(
        &mut self,
        equity: f64,
        exposure_notional: f64,
        correlated_notional: f64,
        new_day: bool,
    ) -> Option<String> {
        if self.tripped_reason.is_some() {
            return self.tripped_reason.clone();
        }
        if !equity.is_finite() {
            return None;
        }
        if new_day || self.day_start.is_nan() {
            self.day_start = equity;
        }
        // Weekly window: every `week_len` new-day bars the weekly peak
        // resets to current equity (a fresh week starts at par). The reset
        // fires on the first bar AFTER a full window, so a drawdown into
        // the window's last bar still trips first.
        if self.week_len > 0 && new_day {
            self.week_count += 1;
            if self.week_count > self.week_len {
                self.week_count = 1;
                self.week_peak = equity;
            }
        }
        if equity > self.week_peak {
            self.week_peak = equity;
        }
        if equity > self.peak {
            self.peak = equity;
        }
        if self.max_daily_loss > 0.0 && self.day_start > 0.0
            && (self.day_start - equity) / self.day_start >= self.max_daily_loss
        {
            self.tripped_reason = Some("max_daily_loss".to_string());
        } else if self.max_weekly_loss > 0.0 && self.week_peak > 0.0
            && (self.week_peak - equity) / self.week_peak >= self.max_weekly_loss
        {
            self.tripped_reason = Some("max_weekly_loss".to_string());
        } else if self.max_drawdown > 0.0 && self.peak > 0.0
            && (self.peak - equity) / self.peak >= self.max_drawdown
        {
            self.tripped_reason = Some("max_drawdown".to_string());
        } else if self.max_exposure > 0.0 && equity > 0.0
            && exposure_notional / equity >= self.max_exposure
        {
            self.tripped_reason = Some("max_exposure".to_string());
        } else if self.max_correlated_exposure > 0.0 && equity > 0.0
            && correlated_notional / equity >= self.max_correlated_exposure
        {
            self.tripped_reason = Some("max_correlated_exposure".to_string());
        }
        self.tripped_reason.clone()
    }

    pub fn tripped(&self) -> bool {
        self.tripped_reason.is_some()
    }

    pub fn reason(&self) -> Option<&str> {
        self.tripped_reason.as_deref()
    }

    /// Manual risk reset after a halt (new session / operator review).
    /// Latched trips are never cleared implicitly — this explicit call is
    /// the "WAIT FOR RISK RESET" path.
    pub fn reset(&mut self) {
        self.day_start = f64::NAN;
        self.week_peak = f64::NEG_INFINITY;
        self.week_count = 0;
        self.peak = f64::NEG_INFINITY;
        self.tripped_reason = None;
    }
}

/// Hard ceiling for configured leverage: anything above is rejected as
/// excessive regardless of justification (spot = 1.0 is the default).
pub const MAX_LEVERAGE: f64 = 5.0;

/// Approximate liquidation distance as a fraction of notional for isolated
/// margin: the adverse move that wipes the position (`1/leverage`).
/// `None` for spot (`leverage <= 1`): no liquidation exists.
/// Conservative by design — ignores maintenance margin, so real
/// liquidation is *closer* than this number.
pub fn liquidation_distance(leverage: f64) -> Option<f64> {
    if leverage <= 1.0 {
        None
    } else if leverage.is_finite() && leverage > 0.0 {
        Some(1.0 / leverage)
    } else {
        None
    }
}

/// Liquidation safety: spot always passes; a margined position passes only
/// when its liquidation distance covers the stop distance with at least
/// `min_buffer_multiple` headroom (e.g. 3x: a 3% stop needs ≥9% to
/// liquidation). Rejects leverage that turns a normal stop-out into a
/// liquidation-risk trade.
pub fn liquidation_ok(leverage: f64, stop_loss: f64, min_buffer_multiple: f64) -> bool {
    if leverage <= 1.0 {
        return stop_loss > 0.0; // spot: only a protective stop is required
    }
    match liquidation_distance(leverage) {
        Some(d) => stop_loss > 0.0 && d >= stop_loss * min_buffer_multiple,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_math() {
        // 10k equity, 1% risk, 2% stop at price 100 → qty = 100*... = 50.
        assert_eq!(fixed_fractional_qty(100.0, 10_000.0, 0.01, 0.02, 0.0), 50.0);
        // Cap at 2000 notional → 20 shares.
        assert_eq!(fixed_fractional_qty(100.0, 10_000.0, 0.01, 0.02, 2000.0), 20.0);
        assert_eq!(fixed_fractional_qty(0.0, 10_000.0, 0.01, 0.02, 0.0), 0.0);
    }

    #[test]
    fn kill_switch_latches() {
        let mut k = KillSwitch::new(0.02, 0.10, 3.0);
        assert_eq!(k.update(10_000.0, 0.0, true), None);
        assert_eq!(k.update(9_900.0, 0.0, false), None);
        assert_eq!(k.update(9_700.0, 0.0, false).as_deref(), Some("max_daily_loss"));
        // Latched: recovery does not untrip.
        assert!(k.tripped());
        assert_eq!(k.update(10_500.0, 0.0, true).as_deref(), Some("max_daily_loss"));
    }

    #[test]
    fn vol_target_scales_down_in_high_vol() {
        let calm = volatility_target_qty(100.0, 10_000.0, Some(0.10), 0.20, 0.0, 0.0);
        let wild = volatility_target_qty(100.0, 10_000.0, Some(0.80), 0.20, 0.0, 0.0);
        assert!(calm > wild && wild > 0.0);
        assert_eq!(volatility_target_qty(100.0, 10_000.0, None, 0.20, 0.0, 5.0), 5.0);
    }

    fn actx() -> AdaptiveSizeCtx {
        AdaptiveSizeCtx {
            risk_fraction: 0.01,
            stop_fraction: 0.03,
            realized_vol_ann: Some(0.15),
            target_vol_ann: 0.20,
            current_dd: 0.0,
            max_dd_allowance: 0.10,
            open_positions: 0,
            max_positions: 3,
            max_position_notional: 0.0,
        }
    }

    #[test]
    fn adaptive_full_size_at_par() {
        // 10k × 1% / (100 × 3%) = 33.33… shares.
        let q = adaptive_qty(100.0, 10_000.0, actx());
        assert!((q - 100.0 / 3.0).abs() < 1e-9, "qty={q}");
    }

    #[test]
    fn adaptive_shrinks_monotonically() {
        let base = adaptive_qty(100.0, 10_000.0, actx());
        // Higher vol → smaller.
        let mut c = actx();
        c.realized_vol_ann = Some(0.60);
        assert!(adaptive_qty(100.0, 10_000.0, c) < base);
        // Deeper drawdown → smaller; at allowance → zero.
        let mut c = actx();
        c.current_dd = 0.05;
        let half = adaptive_qty(100.0, 10_000.0, c);
        assert!(half < base && half > 0.0);
        let mut c = actx();
        c.current_dd = 0.10;
        assert_eq!(adaptive_qty(100.0, 10_000.0, c), 0.0);
        // Crowding → smaller; at cap → zero.
        let mut c = actx();
        c.open_positions = 2;
        assert!(adaptive_qty(100.0, 10_000.0, c) < base);
        let mut c = actx();
        c.open_positions = 3;
        assert_eq!(adaptive_qty(100.0, 10_000.0, c), 0.0);
        // Calm vol never inflates above the fixed-fractional base.
        let mut c = actx();
        c.realized_vol_ann = Some(0.01);
        assert!((adaptive_qty(100.0, 10_000.0, c) - base).abs() < 1e-9);
    }

    #[test]
    fn adaptive_never_escalates_after_losses() {
        // Simulating consecutive losses: equity falls AND drawdown rises —
        // size must fall on both legs, never recover without recovery.
        let full = adaptive_qty(100.0, 10_000.0, actx());
        let mut c = actx();
        c.current_dd = 0.04;
        let after_loss = adaptive_qty(100.0, 9_600.0, c);
        assert!(after_loss < full, "losses must shrink size, not grow it");
    }

    #[test]
    fn weekly_and_correlated_guards_trip() {
        // Weekly: 5-bar window, 3% allowance. Week 1 peaks at 10_200, then
        // slides to 9_800 (−3.9%) → trips before the 10% max-DD would.
        let mut k = KillSwitch::with_weekly(0.50, 0.03, 0.10, 99.0, 0.0, 5);
        for (i, eq) in [10_000.0, 10_100.0, 10_200.0, 10_000.0, 9_800.0].iter().enumerate() {
            let r = k.update(*eq, 0.0, true);
            if i < 4 {
                assert!(r.is_none(), "bar {i}: {r:?}");
            } else {
                assert_eq!(r.as_deref(), Some("max_weekly_loss"));
            }
        }
        // Legacy constructor: weekly guard disabled, same path trips daily.
        let mut k2 = KillSwitch::new(0.02, 0.10, 3.0);
        assert_eq!(k2.update(10_000.0, 0.0, true), None);
        assert_eq!(k2.update(9_700.0, 0.0, false).as_deref(), Some("max_daily_loss"));
        // Correlated exposure trips independently of gross exposure.
        let mut k3 = KillSwitch::with_weekly(0.50, 0.50, 0.50, 99.0, 0.5, 5);
        assert_eq!(k3.update_full(10_000.0, 1_000.0, 6_000.0, true).as_deref(), Some("max_correlated_exposure"));
    }
}
