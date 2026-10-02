//! Port of `backend_api_python/app/utils/risk_guard.py`.
//!
//! Risk-control guard helpers shared by live execution and backtests.

/// Default taker fee rate (0.1%).
pub const DEFAULT_TAKER_FEE_RATE: f64 = 0.001;
/// Absolute ceiling for a sane decimal fee rate (5%).
pub const MAX_FEE_RATE: f64 = 0.05;

/// Sane decimal fee rate, e.g. 0.001 for 0.1%.
///
/// A caller accidentally passing a percent number (e.g. 0.1 meaning 0.1%)
/// is interpreted defensively, exactly like the Python helper.
pub fn coerce_fee_rate(value: Option<f64>, default: f64) -> f64 {
    let mut fee = value.unwrap_or(default);
    if !fee.is_finite() {
        fee = default;
    }
    if fee < 0.0 {
        return 0.0;
    }
    if fee > MAX_FEE_RATE {
        fee /= 100.0;
    }
    fee.min(MAX_FEE_RATE)
}

/// True when a trailing exit price is beyond round-trip fee breakeven.
pub fn trailing_exit_locks_net_profit(
    side: &str,
    entry_price: f64,
    exit_price: f64,
    fee_rate: f64,
    extra_buffer: f64,
) -> bool {
    if !(entry_price > 0.0) || !(exit_price > 0.0) {
        return false;
    }
    let fee = coerce_fee_rate(Some(fee_rate), 0.0);
    let extra = if extra_buffer > 0.0 && extra_buffer.is_finite() {
        extra_buffer
    } else {
        0.0
    };
    let min_move = 2.0 * fee + extra;
    match side {
        "long" => exit_price >= entry_price * (1.0 + min_move),
        "short" => exit_price <= entry_price * (1.0 - min_move),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fee_coercion() {
        assert_eq!(coerce_fee_rate(None, DEFAULT_TAKER_FEE_RATE), 0.001);
        assert_eq!(coerce_fee_rate(Some(-0.5), DEFAULT_TAKER_FEE_RATE), 0.0);
        // 0.1 passed as "percent" (0.1%) -> 0.001 decimal
        assert!((coerce_fee_rate(Some(0.1), DEFAULT_TAKER_FEE_RATE) - 0.001).abs() < 1e-12);
        assert_eq!(coerce_fee_rate(Some(0.02), DEFAULT_TAKER_FEE_RATE), 0.02);
    }

    #[test]
    fn trailing_exit_must_clear_round_trip_fees() {
        // long, 0.1% fee -> needs >= +0.2%
        assert!(trailing_exit_locks_net_profit("long", 100.0, 100.3, 0.001, 0.0));
        assert!(!trailing_exit_locks_net_profit("long", 100.0, 100.1, 0.001, 0.0));
        // short mirror
        assert!(trailing_exit_locks_net_profit(
            "short", 100.0, 99.7, 0.001, 0.0
        ));
        assert!(!trailing_exit_locks_net_profit(
            "short", 100.0, 99.9, 0.001, 0.0
        ));
        assert!(!trailing_exit_locks_net_profit(
            "sideways", 100.0, 200.0, 0.001, 0.0
        ));
        assert!(!trailing_exit_locks_net_profit("long", 0.0, 200.0, 0.001, 0.0));
    }
}
