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

/// Portfolio / account kill-switches. Feed realized equity bar by bar;
/// `tripped()` goes true permanently once any guard fires (latched — a
/// real desk does not auto-resume after a max-loss day).
#[derive(Debug, Clone)]
pub struct KillSwitch {
    /// Halt if intraday equity drops this fraction below the day's first mark.
    pub max_daily_loss: f64,
    /// Halt if equity drops this fraction below the running peak.
    pub max_drawdown: f64,
    /// Halt if exposure notional exceeds this multiple of equity.
    pub max_exposure: f64,
    day_start: f64,
    peak: f64,
    tripped_reason: Option<String>,
}

impl KillSwitch {
    pub fn new(max_daily_loss: f64, max_drawdown: f64, max_exposure: f64) -> Self {
        Self {
            max_daily_loss,
            max_drawdown,
            max_exposure,
            day_start: f64::NAN,
            peak: f64::NEG_INFINITY,
            tripped_reason: None,
        }
    }

    /// `new_day` resets the daily reference. Returns the trip reason if
    /// this update trips a guard.
    pub fn update(&mut self, equity: f64, exposure_notional: f64, new_day: bool) -> Option<String> {
        if self.tripped_reason.is_some() {
            return self.tripped_reason.clone();
        }
        if !equity.is_finite() {
            return None;
        }
        if new_day || self.day_start.is_nan() {
            self.day_start = equity;
        }
        if equity > self.peak {
            self.peak = equity;
        }
        if self.max_daily_loss > 0.0 && self.day_start > 0.0
            && (self.day_start - equity) / self.day_start >= self.max_daily_loss
        {
            self.tripped_reason = Some("max_daily_loss".to_string());
        } else if self.max_drawdown > 0.0 && self.peak > 0.0
            && (self.peak - equity) / self.peak >= self.max_drawdown
        {
            self.tripped_reason = Some("max_drawdown".to_string());
        } else if self.max_exposure > 0.0 && equity > 0.0
            && exposure_notional / equity >= self.max_exposure
        {
            self.tripped_reason = Some("max_exposure".to_string());
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
}
