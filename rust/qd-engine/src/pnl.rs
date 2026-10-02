//! Port of `backend_api_python/app/utils/pnl.py`.
//!
//! Shared PnL helpers — futures margin semantics kept consistent across routes.

/// `swap` / `futures` / `perp` / `perpetual` count as derivatives.
pub fn is_derivatives_market(market_type: &str) -> bool {
    matches!(
        market_type.trim().to_lowercase().as_str(),
        "swap" | "futures" | "future" | "perp" | "perpetual"
    )
}

/// Absolute PnL in quote currency (USDT) from base-asset size.
pub fn calc_unrealized_pnl(side: &str, entry_price: f64, current_price: f64, size: f64) -> f64 {
    if !(entry_price > 0.0) || !(current_price > 0.0) || !(size > 0.0) {
        return 0.0;
    }
    if side.trim().to_lowercase() == "short" {
        (entry_price - current_price) * size
    } else {
        (current_price - entry_price) * size
    }
}

/// Notional value `entry_price * size`, or 0 on bad input.
pub fn calc_notional_value(entry_price: f64, size: f64) -> f64 {
    if !(entry_price > 0.0) || !(size > 0.0) {
        return 0.0;
    }
    entry_price * size
}

/// Margin used for a linear USDT-margined position.
pub fn calc_margin_notional(notional: f64, leverage: f64, market_type: &str) -> f64 {
    if !(notional > 0.0) {
        return 0.0;
    }
    if !is_derivatives_market(market_type) {
        return notional;
    }
    let lev = if leverage > 0.0 { leverage } else { 1.0 };
    notional / lev
}

/// Return-on-margin % for derivatives, price-change % for spot.
pub fn calc_pnl_percent(
    entry_price: f64,
    size: f64,
    pnl: f64,
    leverage: f64,
    market_type: &str,
) -> f64 {
    let denom = calc_notional_value(entry_price, size);
    if !(denom > 0.0) {
        return 0.0;
    }
    let lev = if leverage > 0.0 { leverage } else { 1.0 };
    let mult = if is_derivatives_market(market_type) {
        lev
    } else {
        1.0
    };
    pnl / denom * 100.0 * mult
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_and_short_pnl() {
        assert_eq!(calc_unrealized_pnl("long", 100.0, 110.0, 2.0), 20.0);
        assert_eq!(calc_unrealized_pnl("short", 110.0, 100.0, 2.0), 20.0);
        assert_eq!(calc_unrealized_pnl("long", 110.0, 100.0, 2.0), -20.0);
    }

    #[test]
    fn bad_inputs_yield_zero() {
        assert_eq!(calc_unrealized_pnl("long", 0.0, 100.0, 1.0), 0.0);
        assert_eq!(calc_unrealized_pnl("long", 100.0, 100.0, -1.0), 0.0);
        assert_eq!(calc_notional_value(0.0, 5.0), 0.0);
        assert_eq!(calc_margin_notional(0.0, 10.0, "swap"), 0.0);
        assert_eq!(calc_pnl_percent(0.0, 1.0, 10.0, 1.0, "spot"), 0.0);
    }

    #[test]
    fn margin_semantics() {
        assert_eq!(calc_margin_notional(200.0, 10.0, "swap"), 20.0);
        assert_eq!(calc_margin_notional(200.0, 10.0, "spot"), 200.0);
        assert_eq!(calc_margin_notional(200.0, 0.0, "perp"), 200.0);
    }

    #[test]
    fn pnl_percent_matches_backtest_semantics() {
        // spot: price-change %
        assert_eq!(calc_pnl_percent(100.0, 2.0, 20.0, 1.0, "spot"), 10.0);
        // swap 10x: return-on-margin %
        assert_eq!(calc_pnl_percent(100.0, 2.0, 20.0, 10.0, "swap"), 100.0);
    }
}
