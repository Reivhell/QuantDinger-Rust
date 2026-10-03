//! Realistic event-driven backtesting layer.
//!
//! The loop walks bars in order; signals decide on bar `i` and the fill
//! happens at bar `i + latency_bars` (default 1 — never same-bar fills,
//! which would be look-ahead). Costs are never zero by default:
//! commission + spread + slippage apply to every fill. Every trade is
//! recorded individually with MAE/MFE, holding period, regime and the
//! feature-snapshot id the signal used.

use crate::features::Bar;
use crate::regime::Regime;

/// Execution assumptions. All rates are fractions of notional unless noted.
#[derive(Debug, Clone)]
pub struct ExecConfig {
    /// Commission per side, fraction of fill notional (e.g. 0.0005 = 5 bps).
    pub commission: f64,
    /// Half-spread crossed on each fill, fraction of price.
    pub spread: f64,
    /// Slippage per side, fraction of price, adverse by construction.
    pub slippage: f64,
    /// Bars between signal and fill (>= 1 enforces no look-ahead).
    pub latency_bars: usize,
    /// Fraction of the order filled when the bar's range can't absorb it
    /// (partial fills); remainder chases the next bar. `1.0` = always full.
    pub fill_fraction: f64,
    /// Stop-loss distance as fraction of entry. MANDATORY (> 0): every
    /// position must carry a protective stop (hard 3% cap enforced in
    /// [`crate::config`]). `0` disables — unit tests only, never research.
    pub stop_loss: f64,
    /// Take-profit distance as fraction of entry (0 = disabled).
    pub take_profit: f64,
    /// Trailing-stop distance as fraction of best excursion (0 = disabled).
    pub trailing: f64,
    /// Leverage on margin (>= 1). Margin calls are not simulated; leverage
    /// scales both PnL and the funding charge.
    pub leverage: f64,
    /// Overnight funding per bar held, fraction of notional (0 = disabled).
    pub funding_per_bar: f64,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            commission: 0.0005,
            spread: 0.0002,
            slippage: 0.0003,
            latency_bars: 1,
            fill_fraction: 1.0,
            stop_loss: 0.03, // mandatory protective stop (hard cap, see config)
            take_profit: 0.0,
            trailing: 0.0,
            leverage: 1.0,
            funding_per_bar: 0.0,
        }
    }
}

/// Signal intent for the next fill opportunity. Produced from data `<= i`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Flat,
    Long,
    Short,
}

/// One completed (or open) trade with full execution accounting.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: Option<usize>,
    /// Caller timestamp of the entry bar (`Bar.t`).
    pub entry_t: i64,
    /// Caller timestamp of the exit bar.
    pub exit_t: Option<i64>,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub direction: i8, // +1 long, -1 short
    pub quantity: f64,
    pub gross_pnl: f64,
    pub fees: f64,
    pub slippage_cost: f64,
    pub net_pnl: f64,
    /// Maximum adverse excursion as fraction of entry.
    pub mae: f64,
    /// Maximum favorable excursion as fraction of entry.
    pub mfe: f64,
    pub holding_bars: usize,
    pub entry_regime: Regime,
    pub exit_regime: Option<Regime>,
    pub exit_reason: String,
    /// Opaque id of the feature snapshot the signal consumed.
    pub feature_id: u64,
}

impl Trade {
    pub fn is_open(&self) -> bool {
        self.exit_idx.is_none()
    }
}

struct OpenPosition {
    entry_idx: usize,
    entry_t: i64,
    entry_price: f64,
    direction: i8,
    quantity: f64,
    entry_regime: Regime,
    feature_id: u64,
    best: f64,  // best excursion price (high for long, low for short)
    worst: f64, // worst excursion price
    trail_level: Option<f64>,
}

/// Run the event loop.
///
/// - `signals[i]` is decided on bar `i` from data `<= i`; the fill lands on
///   bar `i + latency_bars` at that bar's open (a price known only then).
/// - `sizer(entry_price, equity) -> quantity` converts risk decisions to size.
/// - `regimes[i]` / `feature_ids[i]` are recorded on the trade.
/// - Only one position at a time (single-name research harness); a new
///   opposing signal closes the open trade first.
/// - Open trades at the end are closed at the last close (`exit_reason =
///   "eod"`).
pub fn run_backtest(
    bars: &[Bar],
    signals: &[Signal],
    regimes: &[Regime],
    feature_ids: &[u64],
    cfg: &ExecConfig,
    starting_equity: f64,
    sizer: impl Fn(f64, f64) -> f64,
) -> BacktestResult {
    let n = bars.len();
    let latency = cfg.latency_bars.max(1);
    let mut equity = starting_equity;
    let mut equity_curve = Vec::with_capacity(n);
    let mut trades: Vec<Trade> = Vec::new();
    let mut open: Option<OpenPosition> = None;
    // pending entries: (fill_idx, direction, feature_id, forced_qty).
    // forced_qty is Some only for partial-fill top-ups, which must NOT be
    // re-sized (re-running the sizer would overfill).
    let mut pending: Vec<(usize, i8, u64, Option<f64>)> = Vec::new();

    for i in 0..n {
        let bar = &bars[i];
        // 1. Manage the open position against this bar (SL/TP/trailing).
        if let Some(pos) = open.as_mut() {
            if bar.open.is_finite() && bar.high.is_finite() && bar.low.is_finite() {
                update_excursion(pos, bar);
                if let Some(reason) = check_exits(pos, bar, cfg) {
                    let exit_price = exit_fill_price(pos, bar, &reason, cfg);
                    close_position(
                        &mut trades,
                        &mut equity,
                        pos,
                        i,
                        bar.t,
                        exit_price,
                        reason,
                        regimes.get(i).copied().unwrap_or(Regime::Transition),
                        cfg,
                    );
                    open = None;
                }
            }
        }
        // 2. Process pending entries due on this bar. Flat-exit orders
        // (direction 0) are NOT consumed here — step 4 handles them.
        let due: Vec<(i8, u64, Option<f64>)> = pending
            .iter()
            .filter(|(idx, d, _, _)| *idx == i && *d != 0)
            .map(|(_, d, f, q)| (*d, *f, *q))
            .collect();
        pending.retain(|(idx, d, _, _)| !(*idx == i && *d != 0));
        for (dir, fid, forced) in due {
            // Opposing pending entry closes an open position first.
            if let Some(mut pos) = open.take() {
                if pos.direction != dir {
                    let px = market_fill_price(bar.open, dir, cfg);
                    close_position(
                        &mut trades,
                        &mut equity,
                        &pos,
                        i,
                        bar.t,
                        px,
                        "signal".to_string(),
                        regimes.get(i).copied().unwrap_or(Regime::Transition),
                        cfg,
                    );
                } else {
                    // Same-direction top-up (partial-fill remainder): add the
                    // carried quantity at this bar's fill price, averaging the
                    // entry price. Fresh signals never reach this branch with
                    // forced quantity.
                    if let Some(q) = forced {
                        let px = market_fill_price(bar.open, dir, cfg);
                        let tot = pos.quantity + q;
                        if tot > 0.0 {
                            pos.entry_price = (pos.entry_price * pos.quantity + px * q) / tot;
                            pos.quantity = tot;
                        }
                    }
                    open = Some(pos);
                    continue; // already positioned this way
                }
            }
            if open.is_none() && bar.open.is_finite() && bar.open > 0.0 {
                let px = market_fill_price(bar.open, dir, cfg);
                // Partial-fill top-ups carry their exact remainder; fresh
                // entries are sized once, here.
                let qty = match forced {
                    Some(q) => q,
                    None => sizer(px, equity),
                };
                if qty > 0.0 {
                    let fill_qty = if forced.is_some() {
                        qty // top-up fills in full (the remainder, by definition)
                    } else {
                        qty * cfg.fill_fraction.clamp(0.0, 1.0)
                    };
                    open = Some(OpenPosition {
                        entry_idx: i,
                        entry_t: bar.t,
                        entry_price: px,
                        direction: dir,
                        quantity: fill_qty,
                        entry_regime: regimes.get(i).copied().unwrap_or(Regime::Transition),
                        feature_id: fid,
                        best: px,
                        worst: px,
                        trail_level: None,
                    });
                    if qty - fill_qty > 0.0 {
                        pending.push((i + 1, dir, fid, Some(qty - fill_qty)));
                    }
                }
            }
        }
        // 3. New signal on this bar → schedule fill at i + latency.
        if let Some(sig) = signals.get(i) {
            let want = match sig {
                Signal::Long => 1,
                Signal::Short => -1,
                Signal::Flat => 0,
            };
            let cur = open.as_ref().map(|p| p.direction).unwrap_or(0);
            if want == 0 {
                if open.is_some() {
                    // Close at next bar open (no look-ahead: schedule it).
                    pending.push((i + latency, 0, 0, None));
                }
            } else if want != cur {
                pending.push((i + latency, want, feature_ids.get(i).copied().unwrap_or(0), None));
            }
            // Flat-exit orders (dir == 0) are handled below in pending processing.
        }
        // 4. Flat-exit orders due now.
        let flat_due = pending.iter().any(|(idx, d, _, _)| *idx == i && *d == 0);
        if flat_due {
            pending.retain(|(idx, d, _, _)| !(*idx == i && *d == 0));
            if let Some(pos) = open.take() {
                let px = market_fill_price(bar.open, pos.direction, cfg);
                close_position(
                    &mut trades,
                    &mut equity,
                    &pos,
                    i,
                    bar.t,
                    px,
                    "signal".to_string(),
                    regimes.get(i).copied().unwrap_or(Regime::Transition),
                    cfg,
                );
            }
        }
        // 5. Funding accrual on open positions.
        if let Some(pos) = open.as_ref() {
            if cfg.funding_per_bar > 0.0 {
                equity -= pos.entry_price * pos.quantity * cfg.funding_per_bar * cfg.leverage.max(1.0);
            }
        }
        // 6. Mark-to-market equity.
        let mtm = match open.as_ref() {
            Some(pos) if bar.close.is_finite() => {
                pos.direction as f64 * (bar.close - pos.entry_price) * pos.quantity * cfg.leverage.max(1.0)
            }
            _ => 0.0,
        };
        equity_curve.push(equity + mtm);
    }
    // End-of-data: close anything still open at the last close.
    if let Some(pos) = open.take() {
        let last = n - 1;
        let px = if bars[last].close.is_finite() { bars[last].close } else { pos.entry_price };
        close_position(
            &mut trades,
            &mut equity,
            &pos,
            last,
            bars[last].t,
            px,
            "eod".to_string(),
            regimes.get(last).copied().unwrap_or(Regime::Transition),
            cfg,
        );
        if let Some(v) = equity_curve.last_mut() {
            *v = equity;
        }
    }
    BacktestResult { trades, equity_curve, starting_equity }
}

fn market_fill_price(open: f64, direction: i8, cfg: &ExecConfig) -> f64 {
    // Cross half the spread + adverse slippage, always against the taker.
    open * (1.0 + direction as f64 * (cfg.spread + cfg.slippage))
}

fn update_excursion(pos: &mut OpenPosition, bar: &Bar) {
    if pos.direction > 0 {
        if bar.high > pos.best {
            pos.best = bar.high;
        }
        if bar.low < pos.worst {
            pos.worst = bar.low;
        }
    } else {
        if bar.low < pos.best {
            pos.best = bar.low;
        }
        if bar.high > pos.worst {
            pos.worst = bar.high;
        }
    }
}

// Stored on the position at creation would be cleaner, but cfg is passed to
// check_exits; the trailing fraction is threaded through there instead.
fn check_exits(pos: &mut OpenPosition, bar: &Bar, cfg: &ExecConfig) -> Option<String> {
    if pos.entry_price <= 0.0 {
        return None;
    }
    // Stop-loss / take-profit checked on the bar's adverse/favorable edge.
    // Conservative: if both trigger in one bar, the stop wins.
    if cfg.stop_loss > 0.0 {
        let adverse = if pos.direction > 0 { bar.low } else { bar.high };
        let dist = (pos.entry_price - adverse) * pos.direction as f64 / pos.entry_price;
        if dist >= cfg.stop_loss {
            return Some("stop_loss".to_string());
        }
    }
    if cfg.take_profit > 0.0 {
        let favor = if pos.direction > 0 { bar.high } else { bar.low };
        let dist = (favor - pos.entry_price) * pos.direction as f64 / pos.entry_price;
        if dist >= cfg.take_profit {
            return Some("take_profit".to_string());
        }
    }
    if cfg.trailing > 0.0 {
        let level = if pos.direction > 0 {
            pos.best * (1.0 - cfg.trailing)
        } else {
            pos.best * (1.0 + cfg.trailing)
        };
        let hit = if pos.direction > 0 { bar.low <= level } else { bar.high >= level };
        // Arm only after the position was profitable at some point.
        let was_profitable = (pos.best - pos.entry_price) * pos.direction as f64 > 0.0;
        if hit && was_profitable {
            pos.trail_level = Some(level);
            return Some("trailing".to_string());
        }
    }
    None
}

fn exit_fill_price(pos: &OpenPosition, bar: &Bar, reason: &str, cfg: &ExecConfig) -> f64 {
    // Stop/take/trailing levels are theoretical triggers; the fill itself is
    // a market order and crosses spread + slippage like any other exit.
    // (EOD exits mark at the last close with no extra cost — documented.)
    let level = match reason {
        "stop_loss" if cfg.stop_loss > 0.0 => {
            pos.entry_price * (1.0 - pos.direction as f64 * cfg.stop_loss)
        }
        "take_profit" if cfg.take_profit > 0.0 => {
            pos.entry_price * (1.0 + pos.direction as f64 * cfg.take_profit)
        }
        "trailing" => pos.trail_level.unwrap_or(bar.close),
        _ => return market_fill_price(bar.open, -pos.direction, cfg),
    };
    // Gap risk: a stop level is not a guaranteed fill. If the bar opened
    // beyond the level (overnight/weekend gap), the market order fills near
    // the open — worse for protective stops. Favorable gaps (take-profit)
    // fill near the open too, which is what a real market order receives.
    let protective = reason != "take_profit";
    let base = if bar.open.is_finite() {
        if pos.direction > 0 {
            if protective { level.min(bar.open) } else { level.max(bar.open) }
        } else if protective {
            level.max(bar.open)
        } else {
            level.min(bar.open)
        }
    } else {
        level
    };
    market_fill_price(base, -pos.direction, cfg)
}

fn close_position(
    trades: &mut Vec<Trade>,
    equity: &mut f64,
    pos: &OpenPosition,
    exit_idx: usize,
    exit_t: i64,
    exit_price: f64,
    reason: String,
    exit_regime: Regime,
    cfg: &ExecConfig,
) {
    let lev = cfg.leverage.max(1.0);
    let gross = pos.direction as f64 * (exit_price - pos.entry_price) * pos.quantity * lev;
    let notional = pos.entry_price * pos.quantity;
    let fees = cfg.commission * notional + cfg.commission * exit_price * pos.quantity;
    let slip = cfg.slippage * notional + cfg.slippage * exit_price * pos.quantity;
    let net = gross - fees - slip;
    *equity += net;
    let mae = if pos.entry_price > 0.0 {
        ((pos.entry_price - pos.worst) * pos.direction as f64 / pos.entry_price).max(0.0)
    } else {
        0.0
    };
    let mfe = if pos.entry_price > 0.0 {
        ((pos.best - pos.entry_price) * pos.direction as f64 / pos.entry_price).max(0.0)
    } else {
        0.0
    };
    trades.push(Trade {
        entry_idx: pos.entry_idx,
        exit_idx: Some(exit_idx),
        entry_t: pos.entry_t,
        exit_t: Some(exit_t),
        entry_price: pos.entry_price,
        exit_price: Some(exit_price),
        direction: pos.direction,
        quantity: pos.quantity,
        gross_pnl: gross,
        fees,
        slippage_cost: slip,
        net_pnl: net,
        mae,
        mfe,
        holding_bars: exit_idx.saturating_sub(pos.entry_idx),
        entry_regime: pos.entry_regime,
        exit_regime: Some(exit_regime),
        exit_reason: reason,
        feature_id: pos.feature_id,
    });
}

/// Result of [`run_backtest`].
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub starting_equity: f64,
}

impl BacktestResult {
    pub fn net_returns(&self) -> Vec<f64> {
        self.trades.iter().map(|t| t.net_pnl).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(closes: &[f64]) -> Vec<Bar> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Bar {
                t: i as i64,
                open: c,
                high: c * 1.001,
                low: c * 0.999,
                close: c,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn latency_means_no_same_bar_fill() {
        let b = bars(&[100.0, 101.0, 102.0, 103.0]);
        let sig = vec![Signal::Long, Signal::Flat, Signal::Flat, Signal::Flat];
        let reg = vec![Regime::Transition; 4];
        let res = run_backtest(&b, &sig, &reg, &[0; 4], &ExecConfig::default(), 10_000.0, |_, _| 1.0);
        assert_eq!(res.trades.len(), 1);
        // signal on bar 0, latency 1 → entry on bar 1's open (101.0 + costs)
        assert_eq!(res.trades[0].entry_idx, 1);
        assert!(res.trades[0].entry_price > 101.0);
    }

    #[test]
    fn partial_fill_does_not_resize() {
        // fill_fraction 0.5: first fill 2.0, remainder 2.0 top-up next bar →
        // final quantity equals the sizer's single decision (4.0).
        let b = bars(&[100.0, 101.0, 102.0, 103.0, 104.0]);
        let sig = vec![Signal::Long, Signal::Flat, Signal::Flat, Signal::Flat, Signal::Flat];
        let reg = vec![Regime::Transition; 5];
        let cfg = ExecConfig { fill_fraction: 0.5, ..ExecConfig::default() };
        let res = run_backtest(&b, &sig, &reg, &[0; 5], &cfg, 10_000.0, |_, _| 4.0);
        assert_eq!(res.trades.len(), 1);
        assert!((res.trades[0].quantity - 4.0).abs() < 1e-12);
    }

    #[test]
    fn stop_exits_pay_exit_costs() {
        // Stop trigger 95.0 on 10 @ ~100: exit must be WORSE than the raw
        // level (spread + slippage crossed on the market exit order).
        let b = bars(&[100.0, 99.0, 90.0, 91.0]);
        let sig = vec![Signal::Long, Signal::Flat, Signal::Flat, Signal::Flat];
        let reg = vec![Regime::Transition; 4];
        let cfg = ExecConfig { stop_loss: 0.05, ..ExecConfig::default() };
        let res = run_backtest(&b, &sig, &reg, &[0; 4], &cfg, 10_000.0, |_, _| 10.0);
        let t = &res.trades[0];
        assert_eq!(t.exit_reason, "stop_loss");
        let raw = t.entry_price * 0.95;
        assert!(t.exit_price.unwrap() < raw, "exit {:?} must be below raw level {raw}", t.exit_price);
    }

    #[test]
    fn trades_carry_bar_timestamps() {
        let mut b = bars(&[100.0, 101.0, 102.0, 103.0]);
        for (i, bar) in b.iter_mut().enumerate() {
            bar.t = 1_700_000_000 + i as i64 * 86_400;
        }
        let sig = vec![Signal::Long, Signal::Flat, Signal::Flat, Signal::Flat];
        let reg = vec![Regime::Transition; 4];
        let res = run_backtest(&b, &sig, &reg, &[0; 4], &ExecConfig::default(), 10_000.0, |_, _| 1.0);
        let t = &res.trades[0];
        assert_eq!(t.entry_t, 1_700_000_000 + 86_400); // entered bar 1
        assert_eq!(t.exit_t.unwrap(), 1_700_000_000 + 86_400 * 2); // flat-exit bar 2
    }

    #[test]
    fn stop_loss_caps_loss_and_fees_apply() {
        let b = bars(&[100.0, 99.0, 90.0, 91.0]);
        let sig = vec![Signal::Long, Signal::Flat, Signal::Flat, Signal::Flat];
        let reg = vec![Regime::Transition; 4];
        let cfg = ExecConfig { stop_loss: 0.05, ..ExecConfig::default() };
        let res = run_backtest(&b, &sig, &reg, &[0; 4], &cfg, 10_000.0, |_, _| 10.0);
        assert_eq!(res.trades[0].exit_reason, "stop_loss");
        assert!(res.trades[0].fees > 0.0);
        assert!(res.trades[0].net_pnl < res.trades[0].gross_pnl);
        assert!(res.trades[0].mae >= 0.05);
    }

    #[test]
    fn every_trade_records_full_accounting() {
        let b = bars(&[100.0, 102.0, 104.0, 103.0]);
        let sig = vec![Signal::Short, Signal::Flat, Signal::Flat, Signal::Flat];
        let reg = vec![Regime::Ranging; 4];
        let res = run_backtest(&b, &sig, &reg, &[7; 4], &ExecConfig::default(), 10_000.0, |_, _| 2.0);
        let t = &res.trades[0];
        assert_eq!(t.direction, -1);
        assert_eq!(t.entry_regime, Regime::Ranging);
        assert_eq!(t.feature_id, 7);
        assert!(t.mfe >= 0.0 && t.holding_bars >= 1);
        assert_eq!(res.equity_curve.len(), 4);
    }
}
