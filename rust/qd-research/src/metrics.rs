//! Performance metrics: returns, risk, risk-adjusted, trade statistics.
//!
//! Conventions (documented so results are comparable):
//! - Returns are simple net-PnL fractions of `starting_equity` unless noted.
//! - Sharpe/Sortino annualize with `sqrt(periods_per_year)`; risk-free = 0
//!   (configurable via `risk_free`).
//! - VaR/CVaR are historical (empirical quantiles of per-trade returns),
//!   reported as positive loss magnitudes.
//! - Drawdowns are computed on the equity curve as fractions of running peak.

use crate::backtest::Trade;

/// Full metric bundle for one backtest result.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    // Return
    pub total_return: f64,
    pub cagr: f64,
    pub avg_trade_return: f64,
    pub expectancy: f64,
    // Risk
    pub max_drawdown: f64,
    pub max_drawdown_duration: usize,
    pub avg_drawdown: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub volatility: f64,
    // Risk-adjusted
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    // Trade statistics
    pub win_rate: f64,
    pub loss_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub payoff_ratio: f64,
    pub num_trades: usize,
    pub longest_win_streak: usize,
    pub longest_loss_streak: usize,
    pub avg_holding_bars: f64,
}

/// Compute [`Metrics`] from trades + equity curve.
///
/// `periods_per_year` annualizes CAGR/volatility/Sharpe/Sortino/Calmar;
/// `bars_held_total` is the span of the backtest in bars (for CAGR when the
/// curve is flat, pass `bars.len()` and `periods_per_year`).
pub fn compute_metrics(
    trades: &[Trade],
    equity_curve: &[f64],
    starting_equity: f64,
    periods_per_year: f64,
) -> Metrics {
    let n = trades.len();
    let mut m = Metrics { num_trades: n, ..Metrics::default() };
    if equity_curve.is_empty() || starting_equity <= 0.0 {
        return m;
    }
    let end_equity = *equity_curve.last().unwrap();
    m.total_return = end_equity / starting_equity - 1.0;
    let years = equity_curve.len() as f64 / periods_per_year.max(1.0);
    m.cagr = if years > 0.0 && end_equity > 0.0 {
        (end_equity / starting_equity).powf(1.0 / years) - 1.0
    } else {
        0.0
    };
    if n == 0 {
        // Drawdowns still defined on the curve.
        let dd = drawdown_stats(equity_curve);
        m.max_drawdown = dd.0;
        m.max_drawdown_duration = dd.1;
        m.avg_drawdown = dd.2;
        m.calmar = if m.max_drawdown > 0.0 { m.cagr / m.max_drawdown } else { 0.0 };
        return m;
    }
    let rets: Vec<f64> = trades.iter().map(|t| t.net_pnl / starting_equity).collect();
    m.avg_trade_return = rets.iter().sum::<f64>() / n as f64;
    m.expectancy = m.avg_trade_return;
    let wins: Vec<f64> = rets.iter().copied().filter(|r| *r > 0.0).collect();
    let losses: Vec<f64> = rets.iter().copied().filter(|r| *r <= 0.0).collect();
    m.win_rate = wins.len() as f64 / n as f64;
    m.loss_rate = 1.0 - m.win_rate;
    m.avg_win = if wins.is_empty() { 0.0 } else { wins.iter().sum::<f64>() / wins.len() as f64 };
    m.avg_loss = if losses.is_empty() {
        0.0
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };
    let gross_win: f64 = wins.iter().sum();
    let gross_loss: f64 = -losses.iter().sum::<f64>();
    m.profit_factor = if gross_loss > 0.0 { gross_win / gross_loss } else { f64::INFINITY };
    m.payoff_ratio = if m.avg_loss != 0.0 { m.avg_win / m.avg_loss.abs() } else { 0.0 };
    let (ws, ls) = streaks(&rets);
    m.longest_win_streak = ws;
    m.longest_loss_streak = ls;
    m.avg_holding_bars = trades.iter().map(|t| t.holding_bars as f64).sum::<f64>() / n as f64;
    // Risk on the equity curve.
    let dd = drawdown_stats(equity_curve);
    m.max_drawdown = dd.0;
    m.max_drawdown_duration = dd.1;
    m.avg_drawdown = dd.2;
    // Historical VaR/CVaR on per-trade returns (positive magnitudes).
    let mut sorted = rets.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let var_level = quantile_sorted(&sorted, 0.05);
    m.var_95 = (-var_level).max(0.0);
    // CVaR = mean of returns at or below the VaR level (worst 5% mass).
    let tail: Vec<f64> = sorted.iter().copied().take_while(|r| *r <= var_level).collect();
    m.cvar_95 = if tail.is_empty() { m.var_95 } else { -tail.iter().sum::<f64>() / tail.len() as f64 };
    // Volatility + Sharpe/Sortino on per-trade returns, annualized.
    let mean = m.avg_trade_return;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    m.volatility = var.sqrt() * periods_per_year.max(1.0).sqrt();
    m.sharpe = if var > 0.0 {
        mean / var.sqrt() * periods_per_year.max(1.0).sqrt()
    } else {
        0.0
    };
    let downside: f64 = rets.iter().map(|r| r.min(0.0).powi(2)).sum::<f64>() / n as f64;
    m.sortino = if downside > 0.0 {
        mean / downside.sqrt() * periods_per_year.max(1.0).sqrt()
    } else {
        0.0
    };
    m.calmar = if m.max_drawdown > 0.0 { m.cagr / m.max_drawdown } else { 0.0 };
    m
}

/// `(max_dd, max_duration_bars, avg_dd)`: drawdown as fraction of peak.
/// Duration = longest bars-under-water stretch; avg over bars with dd > 0.
pub fn drawdown_stats(equity: &[f64]) -> (f64, usize, f64) {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    let mut acc = 0.0;
    let mut cnt = 0usize;
    let mut cur_dur = 0usize;
    let mut max_dur = 0usize;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = if peak > 0.0 { (peak - e) / peak } else { 0.0 };
        if dd > 0.0 {
            acc += dd;
            cnt += 1;
            cur_dur += 1;
            if cur_dur > max_dur {
                max_dur = cur_dur;
            }
        } else {
            cur_dur = 0;
        }
        if dd > max_dd {
            max_dd = dd;
        }
    }
    (max_dd, max_dur, if cnt > 0 { acc / cnt as f64 } else { 0.0 })
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn streaks(rets: &[f64]) -> (usize, usize) {
    let (mut bw, mut bl, mut cw, mut cl) = (0, 0, 0, 0);
    for &r in rets {
        if r > 0.0 {
            cw += 1;
            cl = 0;
        } else {
            cl += 1;
            cw = 0;
        }
        if cw > bw {
            bw = cw;
        }
        if cl > bl {
            bl = cl;
        }
    }
    (bw, bl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::Trade;
    use crate::regime::Regime;

    fn trade(pnl: f64) -> Trade {
        Trade {
            entry_idx: 0,
            exit_idx: Some(1),
            entry_t: 0,
            exit_t: Some(1),
            entry_price: 100.0,
            exit_price: Some(100.0 + pnl),
            direction: 1,
            quantity: 1.0,
            gross_pnl: pnl,
            fees: 0.0,
            slippage_cost: 0.0,
            net_pnl: pnl,
            mae: 0.0,
            mfe: pnl.max(0.0) / 100.0,
            holding_bars: 1,
            entry_regime: Regime::Transition,
            exit_regime: None,
            exit_reason: "signal".into(),
            feature_id: 0,
        }
    }

    #[test]
    fn known_trade_stats() {
        // +2, -1, +2, -1 on 1000 equity → rets .002,-.001,.002,-.001
        let ts = vec![trade(2.0), trade(-1.0), trade(2.0), trade(-1.0)];
        let eq = vec![1000.0, 1002.0, 1001.0, 1003.0, 1002.0];
        let m = compute_metrics(&ts, &eq, 1000.0, 252.0);
        assert_eq!(m.num_trades, 4);
        assert_eq!(m.win_rate, 0.5);
        assert_eq!(m.longest_win_streak, 1);
        assert_eq!(m.longest_loss_streak, 1);
        assert!((m.profit_factor - 2.0).abs() < 1e-12);
        assert!((m.total_return - 0.002).abs() < 1e-12);
        assert!(m.max_drawdown > 0.0 && m.max_drawdown < 0.01);
        assert!(m.var_95 > 0.0 && m.cvar_95 >= m.var_95);
        assert!(m.sharpe.is_finite() && m.sortino.is_finite());
    }

    #[test]
    fn empty_trades_still_drawdown() {
        let eq = vec![100.0, 90.0, 95.0];
        let m = compute_metrics(&[], &eq, 100.0, 252.0);
        assert!((m.max_drawdown - 0.1).abs() < 1e-12);
        assert_eq!(m.max_drawdown_duration, 2);
        assert_eq!(m.num_trades, 0);
    }
}
