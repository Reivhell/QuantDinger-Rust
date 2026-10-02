//! Port of `backend_api_python/app/utils/trade_net_pnl.py`.
//!
//! Net realised P&L: gross profit minus open + close commissions.
//! `qd_strategy_trades` stores one row per fill; closes carry gross `profit`
//! plus close commission, and the matched open-leg fee is allocated FIFO
//! per (symbol, side).
//!
//! Helpers inlined from their Python homes so this module is self-contained:
//! - `is_exit_trade_type` — `app/utils/trade_close_reason.py`
//! - `normalize_strategy_symbol` — `app/services/live_trading/records.py`

use std::collections::{HashMap, VecDeque};

/// Round to 8 decimals (matches Python `round(x, 8)` on these magnitudes).
fn round8(v: f64) -> f64 {
    (v * 100_000_000.0).round() / 100_000_000.0
}

/// `"close_*"` / `"reduce_*"` trade types are exits.
pub fn is_exit_trade_type(trade_type: &str) -> bool {
    let t = trade_type.trim().to_lowercase();
    t.starts_with("close_") || t.starts_with("reduce_")
}

/// Canonical symbol for positions/trades (e.g. `BTC/USDT`).
///
/// Mirrors `normalize_strategy_symbol`: uppercases, strips `-`; if a `/`
/// is present, drops a `:SETTLEMENT` suffix (but keeps `@VENUE`); otherwise
/// splits a trailing `USDT|USDC|USD|BUSD|EUR` quote with `/`.
pub fn normalize_strategy_symbol(symbol: &str) -> String {
    let s = symbol.trim().to_uppercase().replace('-', "");
    if s.is_empty() {
        return String::new();
    }
    if let Some(slash) = s.find('/') {
        let after = &s[slash + 1..];
        let settle = after.find(':').map(|i| slash + 1 + i);
        let venue = after.find('@').map(|i| slash + 1 + i);
        match (settle, venue) {
            (Some(st), Some(v)) if st < v => return s[..st].to_string(),
            (Some(st), None) => return s[..st].to_string(),
            _ => return s,
        }
    }
    for quote in ["USDT", "USDC", "USD", "BUSD", "EUR"] {
        if s.ends_with(quote) && s.len() > quote.len() {
            return format!("{}/{}", &s[..s.len() - quote.len()], quote);
        }
    }
    s
}

/// Minimal trade-row view. `created_at_secs` covers the Python int/float
/// timestamp path (the ORM `datetime` path sorts by epoch seconds too).
#[derive(Debug, Clone)]
pub struct TradeRow {
    pub id: i64,
    pub symbol: String,
    pub symbol_canonical: Option<String>,
    pub trade_type: String,
    pub amount: f64,
    pub commission: f64,
    pub commission_quote: Option<f64>,
    pub profit: Option<f64>,
    pub created_at_secs: i64,
    // Enrichment output (None until `enrich_trades_net_pnl` fills them).
    pub profit_gross: Option<f64>,
    pub open_commission_allocated: Option<f64>,
    pub close_commission: Option<f64>,
    pub total_commission: Option<f64>,
    pub net_pnl: Option<f64>,
    pub grid_matched_profit: Option<f64>,
}

impl TradeRow {
    pub fn new(
        id: i64,
        symbol: &str,
        trade_type: &str,
        amount: f64,
        commission: f64,
        profit: Option<f64>,
        created_at_secs: i64,
    ) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            symbol_canonical: None,
            trade_type: trade_type.to_string(),
            amount,
            commission,
            commission_quote: None,
            profit,
            created_at_secs,
            profit_gross: None,
            open_commission_allocated: None,
            close_commission: None,
            total_commission: None,
            net_pnl: None,
            grid_matched_profit: None,
        }
    }
}

fn leg_side(trade_type: &str) -> &str {
    let t = trade_type.trim().to_lowercase();
    if t.contains("long") {
        "long"
    } else if t.contains("short") {
        "short"
    } else {
        ""
    }
}

fn symbol_key(row: &TradeRow) -> String {
    let raw = row
        .symbol_canonical
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&row.symbol);
    let norm = normalize_strategy_symbol(raw);
    if !norm.is_empty() {
        norm
    } else {
        raw.trim().to_uppercase()
    }
}

fn quote_commission(row: &TradeRow) -> f64 {
    match row.commission_quote {
        Some(v) if v.is_finite() => v,
        _ => {
            if row.commission.is_finite() {
                row.commission
            } else {
                0.0
            }
        }
    }
}

/// Walk trades chronologically; return `{exit_trade_id: allocated_open_fee}`.
pub fn allocate_open_commissions_fifo(trades: &[TradeRow]) -> HashMap<i64, f64> {
    let mut out = HashMap::new();
    if trades.is_empty() {
        return out;
    }
    let mut ordered: Vec<&TradeRow> = trades.iter().collect();
    ordered.sort_by_key(|r| (r.created_at_secs, r.id));
    // (symbol_key, side) -> FIFO lots of (remaining_amount, comm_per_unit)
    let mut lots: HashMap<(String, String), VecDeque<(f64, f64)>> = HashMap::new();
    for row in ordered {
        let side = leg_side(&row.trade_type);
        if side.is_empty() {
            continue;
        }
        let key = (symbol_key(row), side.to_string());
        let amount = if row.amount.is_finite() { row.amount } else { 0.0 };
        let commission = quote_commission(row);
        if !is_exit_trade_type(&row.trade_type) {
            if amount > 1e-12 {
                let cpu = if amount > 0.0 { commission / amount } else { 0.0 };
                lots.entry(key).or_default().push_back((amount, cpu));
            }
            continue;
        }
        if row.profit.is_none() {
            continue;
        }
        let mut close_qty = amount;
        let mut open_comm = 0.0;
        let queue = lots.entry(key).or_default();
        while close_qty > 1e-12 {
            let front = match queue.front_mut() {
                Some(f) => f,
                None => break,
            };
            if front.0 <= 1e-12 {
                queue.pop_front();
                continue;
            }
            let take = front.0.min(close_qty);
            open_comm += take * front.1;
            front.0 -= take;
            close_qty -= take;
            if front.0 <= 1e-12 {
                queue.pop_front();
            }
        }
        // Drop exhausted lots (mirrors the Python list-compaction).
        if let Some(q) = lots.get_mut(&(symbol_key(row), side.to_string())) {
            q.retain(|(rem, _)| *rem > 1e-12);
        }
        if row.id > 0 {
            out.insert(row.id, open_comm);
        }
    }
    out
}

/// Gross profit minus close commission minus allocated open commission.
///
/// `profit=None` rows (entries) yield `None`, mirroring the Python guard
/// `if trade.get("profit") is None: return None`.
pub fn net_realized_pnl(
    profit: Option<f64>,
    profit_gross: Option<f64>,
    close_commission: Option<f64>,
    quote_comm_fallback: f64,
    open_commission: f64,
) -> Option<f64> {
    if profit.is_none() {
        return None;
    }
    let gross = profit_gross.or(profit).unwrap_or(0.0);
    let close_comm = close_commission.unwrap_or(quote_comm_fallback);
    Some(gross - close_comm - open_commission)
}

/// Mutate exit rows in place: gross/open/close/total/net fields, `profit` -> net.
pub fn enrich_trades_net_pnl(trades: &mut [TradeRow]) {
    if trades.is_empty() {
        return;
    }
    let open_map = allocate_open_commissions_fifo(trades);
    for row in trades.iter_mut() {
        if !is_exit_trade_type(&row.trade_type) || row.profit.is_none() {
            continue;
        }
        let open_comm = if row.id > 0 {
            open_map.get(&row.id).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let gross = row.profit.unwrap_or(0.0);
        let close_comm = quote_commission(row);
        let net = gross - close_comm - open_comm;
        row.profit_gross = Some(gross);
        row.open_commission_allocated = Some(round8(open_comm));
        row.close_commission = Some(close_comm);
        row.total_commission = Some(round8(close_comm + open_comm));
        row.net_pnl = Some(round8(net));
        row.profit = Some(round8(net));
        if let Some(gmp) = row.grid_matched_profit {
            if (gmp - gross).abs() <= (1e-8f64).max(gross.abs() * 1e-6) {
                row.grid_matched_profit = Some(round8(net));
            }
        }
    }
}

/// Single-row equity delta for a chronological equity curve.
///
/// Entry rows debit commission now; exit rows add gross minus close fee only
/// (the open fee was already debited on its entry row).
pub fn net_pnl_for_equity_step(row: &TradeRow) -> f64 {
    if row.profit.is_some() {
        if let Some(net) = row.net_pnl {
            let open_comm = row.open_commission_allocated.unwrap_or(0.0);
            return net + open_comm;
        }
        let open_comm = row.open_commission_allocated.unwrap_or(0.0);
        // Mirrors Python: net_realized_pnl(trade, open_commission=open) + open.
        let val = net_realized_pnl(
            row.profit,
            row.profit_gross,
            row.close_commission,
            quote_commission(row),
            open_comm,
        )
        .unwrap_or(0.0);
        return val + open_comm;
    }
    -quote_commission(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(id: i64, sym: &str, amt: f64, comm: f64, ts: i64) -> TradeRow {
        TradeRow::new(id, sym, "open_long", amt, comm, None, ts)
    }
    fn close(id: i64, sym: &str, amt: f64, comm: f64, profit: f64, ts: i64) -> TradeRow {
        TradeRow::new(id, sym, "close_long", amt, comm, Some(profit), ts)
    }

    // Mirrors tests/test_trade_net_pnl.py
    #[test]
    fn allocate_single_round_trip() {
        let trades = vec![open(1, "BTC/USDT", 0.1, 5.0, 100), close(2, "BTC/USDT", 0.1, 3.0, 100.0, 200)];
        let alloc = allocate_open_commissions_fifo(&trades);
        assert_eq!(alloc[&2], 5.0);
    }

    #[test]
    fn enrich_full_round_trip() {
        let mut trades = vec![open(1, "BTC/USDT", 0.1, 5.0, 100), close(2, "BTC/USDT", 0.1, 3.0, 100.0, 200)];
        enrich_trades_net_pnl(&mut trades);
        let c = &trades[1];
        assert_eq!(c.profit_gross, Some(100.0));
        assert_eq!(c.open_commission_allocated, Some(5.0));
        assert_eq!(c.profit, Some(92.0));
        assert_eq!(c.net_pnl, Some(92.0));
        assert_eq!(c.total_commission, Some(8.0));
        let equity: f64 = trades.iter().map(net_pnl_for_equity_step).sum();
        assert!((equity - 92.0).abs() < 1e-9, "equity={equity}");
    }

    #[test]
    fn partial_close_fifo_across_two_opens() {
        let mut trades = vec![
            open(1, "ETH/USDT", 1.0, 10.0, 1),
            TradeRow::new(2, "ETH/USDT", "add_long", 1.0, 8.0, None, 2),
            TradeRow::new(3, "ETH/USDT", "reduce_long", 1.5, 2.0, Some(30.0), 3),
        ];
        enrich_trades_net_pnl(&mut trades);
        let c = &trades[2];
        assert_eq!(c.open_commission_allocated, Some(14.0));
        assert_eq!(c.profit, Some(30.0 - 2.0 - 14.0));
    }

    #[test]
    fn equity_step_open_and_close() {
        let o = TradeRow::new(0, "X", "open_long", 1.0, 4.0, None, 0);
        assert_eq!(net_pnl_for_equity_step(&o), -4.0);
        let mut c = TradeRow::new(0, "X", "close_long", 1.0, 3.0, Some(90.0), 0);
        c.profit_gross = Some(100.0);
        c.net_pnl = Some(90.0);
        c.open_commission_allocated = Some(7.0);
        assert_eq!(net_pnl_for_equity_step(&c), 97.0);
        assert_eq!(
            net_realized_pnl(Some(90.0), Some(100.0), None, 3.0, 7.0),
            Some(90.0)
        );
    }

    #[test]
    fn short_leg_does_not_cross_with_long() {
        let mut trades = vec![
            open(1, "BTC/USDT", 1.0, 5.0, 1),
            TradeRow::new(2, "BTC/USDT", "open_short", 1.0, 6.0, None, 2),
            close(3, "BTC/USDT", 1.0, 1.0, 20.0, 3),
        ];
        enrich_trades_net_pnl(&mut trades);
        assert_eq!(trades[2].open_commission_allocated, Some(5.0));
        assert_eq!(trades[2].profit, Some(20.0 - 1.0 - 5.0));
    }

    #[test]
    fn quote_commission_preferred_over_raw() {
        let mut o = open(1, "BTC/USDT", 1.0, 0.001, 1);
        o.commission_quote = Some(60.0);
        let mut c = close(2, "BTC/USDT", 1.0, 10.0, 100.0, 2);
        c.commission_quote = Some(10.0);
        let mut trades = vec![o, c];
        enrich_trades_net_pnl(&mut trades);
        assert_eq!(trades[1].open_commission_allocated, Some(60.0));
        assert_eq!(trades[1].profit, Some(30.0));
    }

    #[test]
    fn symbol_normalization_matches_python() {
        assert_eq!(normalize_strategy_symbol("btcusdt"), "BTC/USDT");
        assert_eq!(normalize_strategy_symbol("BTC/USDT"), "BTC/USDT");
        assert_eq!(normalize_strategy_symbol("BTC/USDT:USDT"), "BTC/USDT");
        assert_eq!(normalize_strategy_symbol("eth-usd"), "ETH/USD");
    }
}
