//! Port of `backend_api_python/app/utils/trade_close_reason.py`
//! plus `enrich_execution_reference` from
//! `backend_api_python/app/utils/trade_execution.py`
//! (with its `positive_number` helper from
//! `app/services/live_trading/fill_evidence.py`).
//!
//! Machine-readable close-reason codes, zh/en labels, legacy inference,
//! row enrichment, and the execution-reference splitter that keeps
//! instruction benchmarks separate from recorded fill prices.
//!
//! Notes on faithful corners:
//! - `label_for_reason` falls back to the code itself for unknown reasons,
//!   and to `""` for empty input; `lang` starting with "zh" picks Chinese.
//! - `float(x or 0.0)` float-normalisation: `NaN` is truthy in Python and
//!   passes through; empty string / `None` become `0.0`; unparsable text
//!   becomes `0.0`.
//! - `positive_number` accepts numeric strings (`float()` trims whitespace)
//!   and rejects NaN / infinite / non-positive values.

/// Loose numeric-or-text value, mirroring Python dict cells.
#[derive(Debug, Clone, PartialEq)]
pub enum NumVal {
    Num(f64),
    Text(String),
    Null,
}

impl NumVal {
    /// Python `float(v or 0.0)` with unparsable text -> `0.0`.
    pub fn or_zero(&self) -> f64 {
        match self {
            NumVal::Num(v) => *v,
            NumVal::Null => 0.0,
            NumVal::Text(s) => {
                if s.trim().is_empty() {
                    0.0
                } else {
                    s.trim().parse::<f64>().unwrap_or(0.0)
                }
            }
        }
    }
}

/// Python `positive_number`: finite and strictly positive, else `None`.
pub fn positive_number(value: &NumVal) -> Option<f64> {
    let n = match value {
        NumVal::Num(v) => *v,
        NumVal::Null => return None,
        NumVal::Text(s) => s.trim().parse::<f64>().ok()?,
    };
    if n.is_finite() && n > 0.0 {
        Some(n)
    } else {
        None
    }
}

// --- reason codes (verbatim from trade_close_reason.py) ---
pub const GRID_LONG_ENTRY: &str = "long_entry";
pub const GRID_LONG_EXIT: &str = "long_exit";
pub const GRID_SHORT_ENTRY: &str = "short_entry";
pub const GRID_SHORT_EXIT: &str = "short_exit";
pub const GRID_INITIAL_LONG: &str = "grid_initial_long";
pub const GRID_INITIAL_SHORT: &str = "grid_initial_short";
pub const GRID_REDUCE_LONG: &str = "grid_reduce_long";
pub const GRID_REDUCE_SHORT: &str = "grid_reduce_short";
pub const GRID_CLOSE_ALL: &str = "grid_close_all";
pub const GRID_WATERFALL_CLOSE: &str = "grid_waterfall_close";
pub const GRID_EQUITY_STOP_LOSS: &str = "grid_equity_stop_loss";
pub const GRID_EQUITY_TAKE_PROFIT: &str = "grid_equity_take_profit";
pub const GRID_EQUITY_TRAILING_STOP: &str = "grid_equity_trailing_stop";
pub const DCA_EQUITY_STOP_LOSS: &str = "dca_equity_stop_loss";
pub const DCA_EQUITY_TAKE_PROFIT: &str = "dca_equity_take_profit";
pub const DCA_EQUITY_TRAILING_STOP: &str = "dca_equity_trailing_stop";
pub const ROBOT_EQUITY_STOP_LOSS: &str = "robot_equity_stop_loss";
pub const ROBOT_EQUITY_TAKE_PROFIT: &str = "robot_equity_take_profit";
pub const ROBOT_EQUITY_TRAILING_STOP: &str = "robot_equity_trailing_stop";
pub const GRID_OUT_OF_BOUNDS_UP: &str = "grid_out_of_bounds_up";
pub const GRID_OUT_OF_BOUNDS_DOWN: &str = "grid_out_of_bounds_down";
pub const SERVER_STOP_LOSS: &str = "server_stop_loss";
pub const SERVER_TAKE_PROFIT: &str = "server_take_profit";
pub const SERVER_TRAILING_STOP: &str = "server_trailing_stop";
pub const INDICATOR_SIGNAL: &str = "indicator_signal";
pub const LEGACY_SIGNAL_TRIGGER: &str = "signal_trigger";

/// Human label for a reason code. Unknown codes pass through; empty -> `""`.
pub fn label_for_reason(reason: &str, lang: &str) -> String {
    let code = reason.trim();
    if code.is_empty() {
        return String::new();
    }
    let zh = lang.trim().to_lowercase().starts_with("zh");
    let label = if zh { label_zh(code) } else { label_en(code) };
    label.unwrap_or(code).to_string()
}

fn label_zh(code: &str) -> Option<&'static str> {
    Some(match code {
        "long_entry" => "网格买入开多",
        "long_exit" => "网格卖出平多",
        "short_entry" => "网格卖出开空",
        "short_exit" => "网格买入平空",
        "grid_initial_long" => "初始底仓开多",
        "grid_initial_short" => "初始底仓开空",
        "grid_reduce_long" => "网格减多",
        "grid_reduce_short" => "网格减空",
        "grid_close_all" => "网格全平",
        "grid_waterfall_close" => "防瀑布平仓",
        "grid_equity_stop_loss" => "网格净值止损",
        "grid_equity_take_profit" => "网格净值止盈",
        "grid_equity_trailing_stop" => "网格净值追踪止盈",
        "dca_equity_stop_loss" => "定投总权益止损",
        "dca_equity_take_profit" => "定投总权益止盈",
        "dca_equity_trailing_stop" => "定投总权益追踪止盈",
        "robot_equity_stop_loss" => "机器人总权益止损",
        "robot_equity_take_profit" => "机器人总权益止盈",
        "robot_equity_trailing_stop" => "机器人总权益追踪止盈",
        "grid_out_of_bounds_up" => "网格上轨突破平仓",
        "grid_out_of_bounds_down" => "网格下轨突破平仓",
        "server_stop_loss" => "止损平仓",
        "server_take_profit" => "止盈平仓",
        "server_trailing_stop" => "追踪止损平仓",
        "indicator_signal" => "信号触发平仓",
        "signal_trigger" => "信号触发平仓",
        _ => return None,
    })
}

fn label_en(code: &str) -> Option<&'static str> {
    Some(match code {
        "long_entry" => "Grid buy long",
        "long_exit" => "Grid sell close long",
        "short_entry" => "Grid sell short",
        "short_exit" => "Grid buy cover short",
        "grid_initial_long" => "Initial long position",
        "grid_initial_short" => "Initial short position",
        "grid_reduce_long" => "Grid reduce long",
        "grid_reduce_short" => "Grid reduce short",
        "grid_close_all" => "Grid close all",
        "grid_waterfall_close" => "Waterfall protection close",
        "grid_equity_stop_loss" => "Grid equity stop-loss",
        "grid_equity_take_profit" => "Grid equity take-profit",
        "grid_equity_trailing_stop" => "Grid equity trailing take-profit",
        "dca_equity_stop_loss" => "DCA total-equity stop-loss",
        "dca_equity_take_profit" => "DCA total-equity take-profit",
        "dca_equity_trailing_stop" => "DCA total-equity trailing take-profit",
        "robot_equity_stop_loss" => "Robot total-equity stop-loss",
        "robot_equity_take_profit" => "Robot total-equity take-profit",
        "robot_equity_trailing_stop" => "Robot total-equity trailing take-profit",
        "grid_out_of_bounds_up" => "Grid upper-bound breakout",
        "grid_out_of_bounds_down" => "Grid lower-bound breakdown",
        "server_stop_loss" => "Stop-loss close",
        "server_take_profit" => "Take-profit close",
        "server_trailing_stop" => "Trailing stop close",
        "indicator_signal" => "Signal close",
        "signal_trigger" => "Signal close",
        _ => return None,
    })
}

/// `"close_*"` / `"reduce_*"` trade types are exits.
pub fn is_exit_trade_type_local(trade_type: &str) -> bool {
    let t = trade_type.trim().to_lowercase();
    t.starts_with("close_") || t.starts_with("reduce_")
}

/// Best-effort reason for rows written before `close_reason` existed.
pub fn infer_legacy_close_reason(trade_type: &str, bot_type: &str, stored_reason: &str) -> String {
    let stored = stored_reason.trim();
    if !stored.is_empty() {
        return stored.to_string();
    }
    if !is_exit_trade_type_local(trade_type) {
        return String::new();
    }
    let bt = bot_type.trim().to_lowercase();
    let t = trade_type.trim().to_lowercase();
    if bt == "grid" || bt == "dca" {
        if t.contains("long") {
            return GRID_REDUCE_LONG.to_string();
        }
        if t.contains("short") {
            return GRID_REDUCE_SHORT.to_string();
        }
    }
    LEGACY_SIGNAL_TRIGGER.to_string()
}

/// DB `close_reason` choice when persisting a close/reduce fill.
/// `bot_type` stands in for the `trading_config` dict's `bot_type` key.
pub fn resolve_close_reason_for_record(
    trade_type: &str,
    signal_reason: &str,
    bot_type: &str,
) -> String {
    if !is_exit_trade_type_local(trade_type) {
        return String::new();
    }
    let explicit = signal_reason.trim();
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    let bt = bot_type.trim().to_lowercase();
    if bt == "grid" || bt == "dca" {
        return infer_legacy_close_reason(trade_type, &bt, "");
    }
    String::new()
}

#[derive(Debug, Clone)]
pub struct TradeRowIn {
    pub trade_type: String,
    pub close_reason: String,
    pub bot_type: String,
    pub lang: String,
    pub matched_entry_price: Option<NumVal>,
    pub grid_matched_profit: Option<NumVal>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnrichedTradeRow {
    pub close_reason: String,
    pub action_note: String,
    pub action_note_en: String,
    pub matched_entry_price: Option<f64>,
    pub grid_matched_profit: Option<f64>,
}

/// Add resolved `close_reason`, zh/en notes, normalised matched columns.
pub fn enrich_trade_row(row: &TradeRowIn) -> EnrichedTradeRow {
    let stored = row.close_reason.trim();
    let (reason, note, note_en) = if !stored.is_empty() {
        (
            stored.to_string(),
            label_for_reason(stored, &row.lang),
            label_for_reason(stored, "en"),
        )
    } else {
        let inferred = if row.bot_type == "grid" || row.bot_type == "dca" {
            infer_legacy_close_reason(&row.trade_type, &row.bot_type, "")
        } else {
            String::new()
        };
        if !inferred.is_empty() {
            (
                inferred.clone(),
                label_for_reason(&inferred, &row.lang),
                label_for_reason(&inferred, "en"),
            )
        } else {
            (String::new(), String::new(), String::new())
        }
    };
    EnrichedTradeRow {
        close_reason: reason,
        action_note: note,
        action_note_en: note_en,
        matched_entry_price: row.matched_entry_price.as_ref().map(|v| v.or_zero()),
        grid_matched_profit: row.grid_matched_profit.as_ref().map(|v| v.or_zero()),
    }
}

// --- execution reference (trade_execution.py) ---

#[derive(Debug, Clone, Default)]
pub struct ExecRowIn {
    /// Raw `request_payload` — JSON object string, or absent.
    pub request_payload_json: Option<String>,
    pub request_price: NumVal,
    pub grid_request_price: NumVal,
    pub grid_client_reference: String,
    pub price: NumVal,
    pub trade_type: String,
}

impl Default for NumVal {
    fn default() -> Self {
        NumVal::Null
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecRowOut {
    pub grid_client_reference: String,
    pub reference_price: Option<f64>,
    pub reference_kind: Option<String>,
    pub price_deviation_pct: Option<f64>,
}

/// Split instruction benchmarks from the recorded fill price.
/// Mirrors `enrich_execution_reference` field-for-field.
pub fn enrich_execution_reference(row: &ExecRowIn) -> ExecRowOut {
    let client_ref = if row.grid_client_reference.is_empty() {
        json_string_field(row.request_payload_json.as_deref(), "client_order_id").unwrap_or_default()
    } else {
        row.grid_client_reference.clone()
    };
    let mut reference = positive_number(&row.grid_request_price);
    let mut kind: Option<String> = if reference.is_some() {
        Some("limit".to_string())
    } else {
        None
    };
    if reference.is_none() {
        reference = json_number_field(row.request_payload_json.as_deref(), "ref_price");
        if reference.is_some() {
            kind = Some("signal".to_string());
        } else {
            reference = json_number_field(row.request_payload_json.as_deref(), "limit_price")
                .or_else(|| json_number_field(row.request_payload_json.as_deref(), "price"))
                .or_else(|| positive_number(&row.request_price));
            kind = if reference.is_some() {
                Some("instruction".to_string())
            } else {
                None
            };
        }
    }
    let mut deviation: Option<f64> = None;
    if let (Some(reference), Some(actual)) = (reference, positive_number(&row.price)) {
        let action = row.trade_type.trim().to_lowercase();
        let direction: Option<f64> = match action.as_str() {
            "open_long" | "add_long" | "close_short" | "reduce_short" => Some(1.0),
            "open_short" | "add_short" | "close_long" | "reduce_long" => Some(-1.0),
            _ => None,
        };
        if let Some(d) = direction {
            deviation = Some(d * (actual / reference - 1.0) * 100.0);
        }
    }
    ExecRowOut {
        grid_client_reference: client_ref,
        reference_price: reference,
        reference_kind: kind,
        price_deviation_pct: deviation,
    }
}

/// Extract a JSON string field from a flat object payload.
/// Returns `None` for absent / non-string / unparsable input.
pub fn json_string_field(payload: Option<&str>, key: &str) -> Option<String> {
    let raw = json_raw_field(payload?, key)?;
    let s = raw.strip_prefix('"')?.strip_suffix('"')?;
    Some(unescape_json_string(s))
}

/// Extract a JSON number field from a flat object payload.
/// String values are accepted via `positive_number`-style parsing only when
/// they are finite and positive — mirroring `positive_number(payload.get(..))`.
pub fn json_number_field(payload: Option<&str>, key: &str) -> Option<f64> {
    let raw = json_raw_field(payload?, key)?;
    let val = if raw.starts_with('"') {
        positive_number(&NumVal::Text(unescape_json_string(
            raw.strip_prefix('"')?.strip_suffix('"')?,
        )))?
    } else {
        raw.parse::<f64>().ok()?
    };
    positive_number(&NumVal::Num(val))
}

/// Locate the raw JSON value following `"key":` (flat scan; nested objects
/// are not supported — real request payloads here are flat dicts).
fn json_raw_field<'a>(payload: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let mut search = payload;
    loop {
        let pos = search.find(needle.as_str())?;
        let mut rest = &search[pos + needle.len()..];
        rest = rest.trim_start();
        if !rest.starts_with(':') {
            search = &search[pos + needle.len()..];
            continue;
        }
        rest = rest[1..].trim_start();
        if rest.starts_with('"') {
            let mut end = 1;
            let bytes = rest.as_bytes();
            while end < bytes.len() {
                if bytes[end] == b'\\' {
                    end += 2;
                    continue;
                }
                if bytes[end] == b'"' {
                    end += 1;
                    break;
                }
                end += 1;
            }
            return Some(&rest[..end.min(rest.len())]);
        }
        let end = rest
            .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
            .unwrap_or(rest.len());
        return Some(rest[..end].trim_end_matches(',').trim());
    }
}

fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some(other) => out.push(other),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Mirrors tests/test_trade_close_reason.py
    #[test]
    fn exit_types() {
        assert!(is_exit_trade_type_local("close_long"));
        assert!(is_exit_trade_type_local("reduce_short"));
        assert!(!is_exit_trade_type_local("open_long"));
    }

    #[test]
    fn legacy_grid_inference() {
        assert_eq!(
            infer_legacy_close_reason("close_long", "grid", ""),
            GRID_REDUCE_LONG
        );
        assert_eq!(
            infer_legacy_close_reason("close_short", "grid", ""),
            "grid_reduce_short"
        );
    }

    #[test]
    fn stored_reason_labels() {
        let row = enrich_trade_row(&TradeRowIn {
            trade_type: "close_long".into(),
            close_reason: SERVER_STOP_LOSS.into(),
            bot_type: "grid".into(),
            lang: "zh".into(),
            matched_entry_price: None,
            grid_matched_profit: None,
        });
        assert_eq!(row.close_reason, SERVER_STOP_LOSS);
        assert_eq!(row.action_note, "止损平仓");
    }

    #[test]
    fn waterfall_label_contains() {
        let row = enrich_trade_row(&TradeRowIn {
            trade_type: "close_short".into(),
            close_reason: GRID_WATERFALL_CLOSE.into(),
            bot_type: "grid".into(),
            lang: "zh".into(),
            matched_entry_price: None,
            grid_matched_profit: None,
        });
        assert!(row.action_note.contains("防瀑布"));
    }

    #[test]
    fn unknown_label_passthrough_and_empty() {
        assert_eq!(label_for_reason("custom_reason", "zh"), "custom_reason");
        assert_eq!(label_for_reason("", "zh"), "");
    }

    #[test]
    fn indicator_close_without_reason_stays_empty() {
        let row = enrich_trade_row(&TradeRowIn {
            trade_type: "close_long".into(),
            close_reason: "".into(),
            bot_type: "".into(),
            lang: "zh".into(),
            matched_entry_price: None,
            grid_matched_profit: None,
        });
        assert_eq!(row.action_note, "");
        assert_eq!(row.close_reason, "");
    }

    #[test]
    fn take_profit_and_grid_entry_labels() {
        let row = enrich_trade_row(&TradeRowIn {
            trade_type: "close_long".into(),
            close_reason: "server_take_profit".into(),
            bot_type: "".into(),
            lang: "zh".into(),
            matched_entry_price: None,
            grid_matched_profit: None,
        });
        assert_eq!(row.action_note, "止盈平仓");
        assert_eq!(label_for_reason(GRID_LONG_ENTRY, "zh"), "网格买入开多");
        assert_eq!(label_for_reason(GRID_INITIAL_LONG, "en"), "Initial long position");
        let entry = enrich_trade_row(&TradeRowIn {
            trade_type: "open_long".into(),
            close_reason: GRID_LONG_ENTRY.into(),
            bot_type: "grid".into(),
            lang: "zh".into(),
            matched_entry_price: None,
            grid_matched_profit: None,
        });
        assert_eq!(entry.action_note, "网格买入开多");
        assert_eq!(entry.action_note_en, "Grid buy long");
    }

    #[test]
    fn matched_columns_normalise() {
        let row = enrich_trade_row(&TradeRowIn {
            trade_type: "close_long".into(),
            close_reason: "".into(),
            bot_type: "".into(),
            lang: "zh".into(),
            matched_entry_price: Some(NumVal::Text("12.5".into())),
            grid_matched_profit: Some(NumVal::Text("bad".into())),
        });
        assert_eq!(row.matched_entry_price, Some(12.5));
        assert_eq!(row.grid_matched_profit, Some(0.0));
    }

    // Mirrors tests/test_execution_price_evidence.py reference behavior
    #[test]
    fn grid_limit_reference_wins() {
        let out = enrich_execution_reference(&ExecRowIn {
            grid_request_price: NumVal::Num(100.0),
            price: NumVal::Num(101.0),
            trade_type: "open_long".into(),
            ..Default::default()
        });
        assert_eq!(out.reference_price, Some(100.0));
        assert_eq!(out.reference_kind.as_deref(), Some("limit"));
        assert!((out.price_deviation_pct.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn signal_reference_and_short_direction() {
        let out = enrich_execution_reference(&ExecRowIn {
            request_payload_json: Some(r#"{"ref_price": 200.0}"#.into()),
            price: NumVal::Num(198.0),
            trade_type: "open_short".into(),
            ..Default::default()
        });
        assert_eq!(out.reference_price, Some(200.0));
        assert_eq!(out.reference_kind.as_deref(), Some("signal"));
        assert!((out.price_deviation_pct.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn instruction_fallback_chain_and_client_ref() {
        let out = enrich_execution_reference(&ExecRowIn {
            request_payload_json: Some(
                r#"{"client_order_id": "abc", "limit_price": 50.0, "price": 51.0}"#.into(),
            ),
            price: NumVal::Num(50.5),
            trade_type: "close_long".into(),
            ..Default::default()
        });
        assert_eq!(out.reference_price, Some(50.0));
        assert_eq!(out.reference_kind.as_deref(), Some("instruction"));
        assert_eq!(out.grid_client_reference, "abc");
    }

    #[test]
    fn unknown_reference_stays_unknown() {
        let out = enrich_execution_reference(&ExecRowIn {
            price: NumVal::Num(10.0),
            trade_type: "open_long".into(),
            ..Default::default()
        });
        assert_eq!(out.reference_price, None);
        assert_eq!(out.reference_kind, None);
        assert_eq!(out.price_deviation_pct, None);
    }

    #[test]
    fn positive_number_corners() {
        assert_eq!(positive_number(&NumVal::Num(1.5)), Some(1.5));
        assert_eq!(positive_number(&NumVal::Num(0.0)), None);
        assert_eq!(positive_number(&NumVal::Num(f64::NAN)), None);
        assert_eq!(positive_number(&NumVal::Num(f64::INFINITY)), None);
        assert_eq!(positive_number(&NumVal::Text(" 2.5 ".into())), Some(2.5));
        assert_eq!(positive_number(&NumVal::Text("".into())), None);
        assert_eq!(positive_number(&NumVal::Null), None);
        // keeps HashMap import used for future row-map helpers
        let _map: HashMap<String, String> = HashMap::new();
    }
}
