//! Research-report generation: machine-readable JSON + human-readable Markdown.
//!
//! The final assessment reports evidence, never verdicts like "GOOD
//! STRATEGY". Data-snooping controls (White's Reality Check, Hansen SPA)
//! are reported as NOT IMPLEMENTED with reasons — never faked.

use crate::costs::CostStressRow;
use crate::mae_mfe::MaeMfeReport;
use crate::metrics::Metrics;
use crate::montecarlo::MonteCarloReport;
use crate::pbo::PboReport;
use crate::sensitivity::SensitivityReport;
use crate::walkforward::WalkForwardReport;
use std::collections::BTreeMap;

/// Regime-conditioned performance slice.
#[derive(Debug, Clone, Default)]
pub struct RegimeSlice {
    pub trades: usize,
    pub win_rate: f64,
    pub expectancy: f64,
    pub sharpe_per_trade: f64,
    pub max_drawdown: f64,
}

pub fn regime_slices(
    trades: &[crate::backtest::Trade],
    starting_equity: f64,
) -> BTreeMap<String, RegimeSlice> {
    let mut map: BTreeMap<String, Vec<&crate::backtest::Trade>> = BTreeMap::new();
    for t in trades {
        map.entry(t.entry_regime.name().to_string()).or_default().push(t);
    }
    map.into_iter()
        .map(|(k, ts)| {
            let n = ts.len();
            let rets: Vec<f64> = ts.iter().map(|t| t.net_pnl / starting_equity).collect();
            let wins = rets.iter().filter(|r| **r > 0.0).count();
            let mean = if n > 0 { rets.iter().sum::<f64>() / n as f64 } else { 0.0 };
            let var = if n > 0 {
                rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64
            } else {
                0.0
            };
            // Regime-local drawdown on the regime's own trade sequence.
            let mut eq = 1.0;
            let mut peak = 1.0;
            let mut mdd = 0.0;
            for r in &rets {
                eq *= 1.0 + r;
                if eq > peak {
                    peak = eq;
                }
                let dd = if peak > 0.0 { (peak - eq) / peak } else { 0.0 };
                if dd > mdd {
                    mdd = dd;
                }
            }
            (
                k,
                RegimeSlice {
                    trades: n,
                    win_rate: if n > 0 { wins as f64 / n as f64 } else { 0.0 },
                    expectancy: mean,
                    sharpe_per_trade: if var > 0.0 { mean / var.sqrt() } else { 0.0 },
                    max_drawdown: mdd,
                },
            )
        })
        .collect()
}

/// Everything the report needs (caller assembles from pipeline stages).
#[derive(Debug, Clone, Default)]
pub struct ResearchInput {
    pub strategy_name: String,
    pub strategy_config: String,
    pub asset: String,
    pub timeframe: String,
    pub date_range: String,
    pub n_bars: usize,
    pub metrics: Metrics,
    pub mae_mfe: Option<MaeMfeReport>,
    pub regime_perf: BTreeMap<String, RegimeSlice>,
    pub walkforward: Option<WalkForwardReport>,
    pub montecarlo: Option<MonteCarloReport>,
    pub cpcv_paths: usize,
    pub cpcv_runs_scored: usize,
    pub cpcv_median_oos: f64,
    pub cpcv_p25_oos: f64,
    pub cpcv_p75_oos: f64,
    pub pbo: Option<PboReport>,
    pub sensitivity: Vec<SensitivityReport>,
    pub cost_stress: Vec<CostStressRow>,
    pub cost_verdict: String,
    pub primary_weakness: String,
    pub regime_dependency: String,
}

/// Escape a string for JSON output.
fn jestr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn pct(x: f64) -> String {
    format!("{:.2}%", x * 100.0)
}

/// Machine-readable JSON report.
pub fn report_json(r: &ResearchInput) -> String {
    let m = &r.metrics;
    let mut o = String::from("{");
    o.push_str(&format!("\"strategy\":{},\"config\":{},", jestr(&r.strategy_name), jestr(&r.strategy_config)));
    o.push_str(&format!(
        "\"data\":{{\"asset\":{},\"timeframe\":{},\"range\":{},\"bars\":{}}},",
        jestr(&r.asset),
        jestr(&r.timeframe),
        jestr(&r.date_range),
        r.n_bars
    ));
    o.push_str(&format!(
        "\"backtest\":{{\"total_return\":{:.6},\"cagr\":{:.6},\"sharpe\":{:.4},\"sortino\":{:.4},\"calmar\":{:.4},\"max_drawdown\":{:.6},\"expectancy\":{:.6},\"trades\":{}}},",
        m.total_return, m.cagr, m.sharpe, m.sortino, m.calmar, m.max_drawdown, m.expectancy, m.num_trades
    ));
    o.push_str("\"data_snooping\":{\"whites_reality_check\":\"NOT IMPLEMENTED\",\"hansen_spa\":\"NOT IMPLEMENTED\",\"reason\":\"insufficient statistical assumptions / data requirements for reliable bootstrap null distributions at this sample size; reported honestly instead of faked\"},");
    if let Some(p) = &r.pbo {
        o.push_str(&format!(
            "\"overfitting\":{{\"configs_tested\":{},\"pbo\":{:.4},\"degradation\":{:.4},\"assessment\":{}}},",
            p.configs_tested, p.pbo, p.degradation, jestr(p.assessment)
        ));
    }
    o.push_str(&format!(
        "\"cpcv\":{{\"paths\":{},\"runs_scored\":{},\"median_oos\":{:.6},\"p25_oos\":{:.6},\"p75_oos\":{:.6}}},",
        r.cpcv_paths, r.cpcv_runs_scored, r.cpcv_median_oos, r.cpcv_p25_oos, r.cpcv_p75_oos
    ));
    if let Some(w) = &r.walkforward {
        o.push_str(&format!(
            "\"walkforward\":{{\"folds\":{},\"positive_oos_fraction\":{:.4},\"median_oos_sharpe\":{:.4},\"worst_oos_dd\":{:.6},\"stable\":{}}},",
            w.folds.len(),
            w.positive_oos_fraction,
            w.median_oos_sharpe,
            w.worst_oos_dd,
            w.stable
        ));
    }
    o.push_str(&format!("\"cost_verdict\":{},", jestr(&r.cost_verdict)));
    o.push_str(&format!("\"regime_dependency\":{},", jestr(&r.regime_dependency)));
    o.push_str(&format!("\"primary_weakness\":{}", jestr(&r.primary_weakness)));
    o.push('}');
    o
}

/// Human-readable Markdown report (sections 1–15 of the research spec).
pub fn report_markdown(r: &ResearchInput) -> String {
    let m = &r.metrics;
    let mut o = String::new();
    o.push_str(&format!("# Strategy Research Report: {}\n\n", r.strategy_name));
    o.push_str(&format!("Config: `{}`\n\n", r.strategy_config));
    o.push_str("## 1-2. Data & Strategy\n");
    o.push_str(&format!(
        "- Asset: {} | Timeframe: {} | Range: {} | Bars: {}\n\n",
        r.asset, r.timeframe, r.date_range, r.n_bars
    ));
    o.push_str("## 3-5. Backtest, Trades & Risk\n");
    o.push_str(&format!(
        "- Total return: {} | CAGR: {} | Expectancy/trade: {:.4}%\n- Sharpe: {:.3} | Sortino: {:.3} | Calmar: {:.3}\n- Max DD: {} ({} bars) | Avg DD: {} | VaR95: {} | CVaR95: {}\n- Win rate: {} | Profit factor: {:.3} | Payoff: {:.3} | Trades: {}\n- Win streak: {} | Loss streak: {} | Avg hold: {:.1} bars\n\n",
        pct(m.total_return),
        pct(m.cagr),
        m.expectancy * 100.0,
        m.sharpe,
        m.sortino,
        m.calmar,
        pct(m.max_drawdown),
        m.max_drawdown_duration,
        pct(m.avg_drawdown),
        pct(m.var_95),
        pct(m.cvar_95),
        pct(m.win_rate),
        m.profit_factor,
        m.payoff_ratio,
        m.num_trades,
        m.longest_win_streak,
        m.longest_loss_streak,
        m.avg_holding_bars,
    ));
    if let Some(mm) = &r.mae_mfe {
        o.push_str("## 6. MAE / MFE\n");
        o.push_str(&format!(
            "- MAE p50/p95: {:.3}% / {:.3}% | MFE p50/p95: {:.3}% / {:.3}%\n- Loser MAE: {:.3}% vs winner MAE: {:.3}% | Loser MFE given back: {:.3}%\n\n",
            mm.mae.median * 100.0,
            mm.mae.p95 * 100.0,
            mm.mfe.median * 100.0,
            mm.mfe.p95 * 100.0,
            mm.mae_losers * 100.0,
            mm.mae_winners * 100.0,
            mm.mfe_losers * 100.0,
        ));
    }
    o.push_str("## 7/12. Regime Analysis\n");
    o.push_str("| Regime | Trades | Win rate | Expectancy | Sharpe/t | Max DD |\n|---|---|---|---|---|---|\n");
    for (name, s) in &r.regime_perf {
        o.push_str(&format!(
            "| {} | {} | {} | {:.4}% | {:.3} | {} |\n",
            name,
            s.trades,
            pct(s.win_rate),
            s.expectancy * 100.0,
            s.sharpe_per_trade,
            pct(s.max_drawdown)
        ));
    }
    o.push_str(&format!("\nRegime dependency: {}\n\n", r.regime_dependency));
    if let Some(w) = &r.walkforward {
        o.push_str("## 8. Walk-Forward\n");
        o.push_str(&format!(
            "- OOS-positive windows: {:.0}% ({}/{}), median OOS Sharpe {:.3}, worst OOS DD {}\n- Stability: {}\n\n",
            w.positive_oos_fraction * 100.0,
            w.folds.iter().filter(|f| f.oos_return > 0.0).count(),
            w.folds.len(),
            w.median_oos_sharpe,
            pct(w.worst_oos_dd),
            if w.stable { "STABLE" } else { "UNSTABLE" },
        ));
    }
    if let Some(mc) = &r.montecarlo {
        o.push_str("## 9-10. Monte Carlo (shuffle + bootstrap)\n");
        o.push_str(&format!(
            "- {} sims (seed {}). Shuffled return: median {} (p5 {} / p95 {})\n- Shuffled MDD: median {} (p95 {}) | Longest loss: median {:.0} (p95 {:.0})\n- Bootstrap Sharpe/t: median {:.3} (p5 {:.3})\n- NOTE: Monte Carlo is not proof of future profitability.\n\n",
            mc.simulations,
            mc.seed,
            pct(mc.shuffled_return.median),
            pct(mc.shuffled_return.p5),
            pct(mc.shuffled_return.p95),
            pct(mc.shuffled_max_dd.median),
            pct(mc.shuffled_max_dd.p95),
            mc.shuffled_longest_loss.median,
            mc.shuffled_longest_loss.p95,
            mc.boot_sharpe.median,
            mc.boot_sharpe.p5,
        ));
    }
    o.push_str(&format!("## 11. CPCV\n- Paths: {} | Runs scored: {} | Median OOS return: {:.4}% (p25 {:.4}% / p75 {:.4}%)\n", r.cpcv_paths, r.cpcv_runs_scored, r.cpcv_median_oos * 100.0, r.cpcv_p25_oos * 100.0, r.cpcv_p75_oos * 100.0));
    o.push_str("- White's Reality Check: NOT IMPLEMENTED. Hansen SPA: NOT IMPLEMENTED.\n- Reason: insufficient statistical assumptions / data requirements — reported honestly, never faked.\n\n");
    if let Some(p) = &r.pbo {
        o.push_str("## 12. PBO / Overfitting\n");
        o.push_str(&format!(
            "- Configs tested: {} (all tracked, failures included) | Best IS: {:.3} | Median IS: {:.3} | Median OOS of IS-best: {:.3}\n- Degradation: {} | Assessment: {}\n\n",
            p.configs_tested,
            p.best_is,
            p.median_is,
            p.median_oos_of_is_best,
            pct(p.degradation),
            p.assessment
        ));
    }
    if !r.sensitivity.is_empty() {
        o.push_str("## 13. Sensitivity\n");
        for s in &r.sensitivity {
            o.push_str(&format!(
                "- {}={}: worst-relative {:.2}, viable {:.0}%, verdict {}\n",
                s.baseline.name,
                s.baseline.value,
                s.worst_relative,
                s.viable_fraction * 100.0,
                s.verdict
            ));
        }
        o.push('\n');
    }
    if !r.cost_stress.is_empty() {
        o.push_str("## 14. Cost Stress\n");
        for c in &r.cost_stress {
            o.push_str(&format!(
                "- x{:.2} costs → net {:+.2}: {}\n",
                c.multiplier,
                c.net_pnl,
                if c.still_profitable { "profitable" } else { "UNPROFITABLE" }
            ));
        }
        o.push_str(&format!("\nVerdict: {}\n\n", r.cost_verdict));
    }
    o.push_str("## 15. Robustness Assessment (evidence, not verdicts)\n");
    o.push_str(&format!("- Primary weakness: {}\n", r.primary_weakness));
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime::Regime;

    #[test]
    fn json_is_machine_readable() {
        let r = ResearchInput {
            strategy_name: "demo".into(),
            cost_verdict: "ROBUST".into(),
            regime_dependency: "LOW".into(),
            primary_weakness: "none observed".into(),
            ..ResearchInput::default()
        };
        let j = report_json(&r);
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(j.contains("NOT IMPLEMENTED"));
        assert!(j.contains("\"strategy\":\"demo\""));
        let md = report_markdown(&r);
        assert!(md.contains("# Strategy Research Report: demo"));
        assert!(md.contains("never faked"));
    }

    #[test]
    fn regime_slice_math() {
        use crate::backtest::Trade;
        let t = |pnl: f64, reg: Regime| Trade {
            entry_idx: 0,
            exit_idx: Some(1),
            entry_t: 0,
            exit_t: Some(1),
            entry_price: 100.0,
            exit_price: Some(100.0),
            direction: 1,
            quantity: 1.0,
            gross_pnl: pnl,
            fees: 0.0,
            slippage_cost: 0.0,
            net_pnl: pnl,
            mae: 0.0,
            mfe: 0.0,
            holding_bars: 1,
            entry_regime: reg,
            exit_regime: None,
            exit_reason: "signal".into(),
            feature_id: 0,
        };
        let ts = vec![t(2.0, Regime::TrendingUp), t(-1.0, Regime::TrendingUp)];
        let s = regime_slices(&ts, 1000.0);
        assert_eq!(s["TRENDING_UP"].trades, 2);
        assert_eq!(s["TRENDING_UP"].win_rate, 0.5);
    }
}
