//! MAE / MFE analysis: excursion distributions, regime splits, and
//! stop/take placement evaluation (descriptive only — no auto-optimization).

use crate::backtest::Trade;
use crate::montecarlo::DistSummary;
use std::collections::BTreeMap;

/// MAE/MFE distribution summaries + outcome-conditioned analysis.
#[derive(Debug, Clone)]
pub struct MaeMfeReport {
    pub mae: DistSummary,
    pub mfe: DistSummary,
    /// Mean MAE/MFE split by entry regime.
    pub by_regime: BTreeMap<String, (f64, f64, usize)>,
    /// Mean MAE of winners vs losers (stops: do losers suffer large MAE
    /// before dying?); mean MFE of winners vs losers (takes: do losers
    /// give back large MFE?).
    pub mae_winners: f64,
    pub mae_losers: f64,
    pub mfe_winners: f64,
    pub mfe_losers: f64,
}

pub fn analyze_mae_mfe(trades: &[Trade]) -> MaeMfeReport {
    use crate::montecarlo::summarize;
    let mae: Vec<f64> = trades.iter().map(|t| t.mae).collect();
    let mfe: Vec<f64> = trades.iter().map(|t| t.mfe).collect();
    let mut by_regime: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    for t in trades {
        let e = by_regime.entry(t.entry_regime.name().to_string()).or_insert((0.0, 0.0, 0));
        e.0 += t.mae;
        e.1 += t.mfe;
        e.2 += 1;
    }
    for (_, v) in by_regime.iter_mut() {
        v.0 /= v.2 as f64;
        v.1 /= v.2 as f64;
    }
    let (mut mw, mut ml, mut fw, mut fl) = (vec![], vec![], vec![], vec![]);
    for t in trades {
        if t.net_pnl > 0.0 {
            mw.push(t.mae);
            fw.push(t.mfe);
        } else {
            ml.push(t.mae);
            fl.push(t.mfe);
        }
    }
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    MaeMfeReport {
        mae: summarize(mae),
        mfe: summarize(mfe),
        by_regime,
        mae_winners: mean(&mw),
        mae_losers: mean(&ml),
        mfe_winners: mean(&fw),
        mfe_losers: mean(&fl),
    }
}

/// Stop/take placement readout: for each candidate stop fraction, what
/// fraction of trades would have survived (MAE < stop) and what the
/// winners' MFE capture looks like. Descriptive — choosing the "best"
/// row on the same data is in-sample optimization and must be validated
/// out-of-sample ([`crate::walkforward`]).
#[derive(Debug, Clone)]
pub struct StopRow {
    pub stop: f64,
    pub survival_rate: f64,
    pub winner_mfe_p50: f64,
}

pub fn stop_sweep(trades: &[Trade], stops: &[f64]) -> Vec<StopRow> {
    stops
        .iter()
        .map(|&stop| {
            let surv = trades.iter().filter(|t| t.mae < stop).count();
            let mut wmfe: Vec<f64> = trades
                .iter()
                .filter(|t| t.net_pnl > 0.0)
                .map(|t| t.mfe)
                .collect();
            wmfe.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = if wmfe.is_empty() {
                0.0
            } else {
                wmfe[wmfe.len() / 2]
            };
            StopRow {
                stop,
                survival_rate: if trades.is_empty() {
                    0.0
                } else {
                    surv as f64 / trades.len() as f64
                },
                winner_mfe_p50: p50,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::Trade;
    use crate::regime::Regime;

    fn trade(mae: f64, mfe: f64, pnl: f64, reg: Regime) -> Trade {
        Trade {
            entry_idx: 0,
            exit_idx: Some(1),
            entry_price: 100.0,
            exit_price: Some(100.0),
            direction: 1,
            quantity: 1.0,
            gross_pnl: pnl,
            fees: 0.0,
            slippage_cost: 0.0,
            net_pnl: pnl,
            mae,
            mfe,
            holding_bars: 1,
            entry_regime: reg,
            exit_regime: None,
            exit_reason: "signal".into(),
            feature_id: 0,
        }
    }

    #[test]
    fn report_splits_winners_losers() {
        let ts = vec![
            trade(0.01, 0.05, 3.0, Regime::TrendingUp),
            trade(0.04, 0.01, -2.0, Regime::Ranging),
        ];
        let r = analyze_mae_mfe(&ts);
        assert_eq!(r.mae_winners, 0.01);
        assert_eq!(r.mae_losers, 0.04);
        assert_eq!(r.mfe_losers, 0.01);
        assert_eq!(r.by_regime["TRENDING_UP"].2, 1);
        let sweep = stop_sweep(&ts, &[0.02, 0.05]);
        assert_eq!(sweep[0].survival_rate, 0.5);
        assert_eq!(sweep[1].survival_rate, 1.0);
    }
}
