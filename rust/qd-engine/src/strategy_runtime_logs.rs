//! Port of the DB-free slice of `backend_api_python/app/utils/strategy_runtime_logs.py`:
//! [`MARKET_DATA_LOG_PREFIX`], [`format_market_data_log`],
//! [`parse_market_data_log`], and the row-normalization inside
//! `append_strategy_log`.
//!
//! The actual INSERT (`get_db_connection`, logger, clock) stays Python-side —
//! it is IO, not computation. [`normalize_log_row`] captures everything the
//! function computes before touching the DB: `int(strategy_id)` coercion,
//! level normalization (`(level or "info").strip().lower()[:20]`), and message
//! trimming (`str(message or "").strip()[:8000]`, empty → skip).
//!
//! Note: `int(x)` semantics for floats-as-strings etc. are caller-side; the
//! port takes the already-coerced `i64` plus a flag for "coercion failed".

use crate::json_helpers::{parse_json, JsonInput, JsonVal};
use crate::market_data_errors::MarketDataFailure;

/// Mirrors `MARKET_DATA_LOG_PREFIX`.
pub const MARKET_DATA_LOG_PREFIX: &str = "market-data|";

/// Mirrors `format_market_data_log`: prefix + compact JSON, byte-identical to
/// `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`.
pub fn format_market_data_log(failure: &MarketDataFailure) -> String {
    format!("{MARKET_DATA_LOG_PREFIX}{}", JsonVal::Obj(failure.as_dict()).dump_raw())
}

/// Mirrors `parse_market_data_log`: prefix check, then JSON-decode; only a
/// JSON *object* is returned, anything else yields `None`.
/// (`str(message or "")` stringification is caller-side.)
pub fn parse_market_data_log(raw: &str) -> Option<Vec<(String, JsonVal)>> {
    let body = raw.strip_prefix(MARKET_DATA_LOG_PREFIX)?;
    match parse_json(body) {
        Ok(JsonVal::Obj(kv)) => Some(kv),
        _ => None,
    }
}

/// Mirrors the pure prelude of `append_strategy_log`.
/// Returns `None` when the row must be skipped (empty message after trim,
/// or `int(strategy_id)` failed — `coerce_ok == false`).
pub fn normalize_log_row(
    strategy_id: i64,
    coerce_ok: bool,
    level: Option<&str>,
    message: Option<&str>,
) -> Option<(i64, String, String)> {
    if !coerce_ok {
        return None;
    }
    let lv: String = level.unwrap_or("info").trim().to_lowercase().chars().take(20).collect();
    let msg = message.unwrap_or("").trim();
    if msg.is_empty() {
        return None;
    }
    Some((strategy_id, lv, msg.chars().take(8000).collect()))
}

/// Convenience mirror of `safe_json_loads` use on `JsonInput` for tests.
pub fn loads_input(input: &JsonInput, default: &JsonVal) -> JsonVal {
    crate::json_helpers::safe_json_loads(input, default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_data_errors::classify_market_data_failure;

    #[test]
    fn format_parse_round_trip() {
        let f = classify_market_data_failure("timeout", "Binance", "Spot", "BTC/USDT", "1h");
        let line = format_market_data_log(&f);
        assert!(line.starts_with("market-data|"));
        let kv = parse_market_data_log(&line).unwrap();
        let back = MarketDataFailure::from_mapping(&kv);
        assert_eq!(back, f);
    }

    #[test]
    fn parse_rejects_non_objects_and_junk() {
        assert_eq!(parse_market_data_log("plain line"), None);
        assert_eq!(parse_market_data_log("market-data|[1,2]"), None);
        assert_eq!(parse_market_data_log("market-data|{bad"), None);
        assert_eq!(parse_market_data_log("market-data|"), None);
    }

    #[test]
    fn row_normalization_edges() {
        assert_eq!(
            normalize_log_row(7, true, Some(" WARNING "), Some("  hi  ")),
            Some((7, "warning".to_string(), "hi".to_string()))
        );
        assert_eq!(normalize_log_row(7, true, None, Some("m")), Some((7, "info".to_string(), "m".to_string())));
        assert_eq!(normalize_log_row(7, false, None, Some("m")), None);
        assert_eq!(normalize_log_row(7, true, None, Some("   ")), None);
        assert_eq!(normalize_log_row(7, true, None, None), None);
        let long_level = "x".repeat(40);
        let (_, lv, _) = normalize_log_row(1, true, Some(&long_level), Some("m")).unwrap();
        assert_eq!(lv.len(), 20);
        let long_msg = "y".repeat(9000);
        let (_, _, m) = normalize_log_row(1, true, None, Some(&long_msg)).unwrap();
        assert_eq!(m.chars().count(), 8000);
    }
}
