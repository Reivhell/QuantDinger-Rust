//! Transaction-cost sensitivity: re-price results under multiplied
//! execution assumptions (spread, commission, slippage).
//!
//! If profitability disappears with small cost increases, the strategy is
//! execution-sensitive and is flagged as such — no matter how good the
//! base-case backtest looks.

use crate::backtest::ExecConfig;

/// Scale every cost leg of an [`ExecConfig`] by `multiplier`.
pub fn scaled_costs(base: &ExecConfig, multiplier: f64) -> ExecConfig {
    let mut cfg = base.clone();
    cfg.commission *= multiplier;
    cfg.spread *= multiplier;
    cfg.slippage *= multiplier;
    cfg
}

/// One cost-stress row: P&L under a cost multiplier.
#[derive(Debug, Clone)]
pub struct CostStressRow {
    pub multiplier: f64,
    pub net_pnl: f64,
    pub still_profitable: bool,
}

/// Re-price a set of per-trade `(gross, cost)` pairs under multipliers of
/// the cost leg. `multipliers` e.g. `[1.0, 1.25, 1.5, 2.0, 3.0]`.
pub fn cost_stress(gross_costs: &[(f64, f64)], multipliers: &[f64]) -> Vec<CostStressRow> {
    multipliers
        .iter()
        .map(|&m| {
            let net: f64 = gross_costs.iter().map(|(g, c)| g - c * m).sum();
            CostStressRow { multiplier: m, net_pnl: net, still_profitable: net > 0.0 }
        })
        .collect()
}

/// Execution-sensitivity verdict: the smallest multiplier that flips the
/// total negative, if any.
pub fn execution_sensitivity(rows: &[CostStressRow]) -> &'static str {
    match rows.iter().find(|r| !r.still_profitable) {
        None => "ROBUST (profitable at all tested multipliers)",
        Some(r) if r.multiplier <= 1.25 => "EXECUTION-SENSITIVE (dies at ≤+25% costs)",
        Some(r) if r.multiplier <= 2.0 => "MODERATE (dies at ≤+100% costs)",
        Some(_) => "RESILIENT (dies only beyond +100% costs)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_flip_detected() {
        // gross 100, costs 60 → net 40 at 1x, -20 at 2x.
        let rows = cost_stress(&[(100.0, 60.0)], &[1.0, 1.5, 2.0]);
        assert!(rows[0].still_profitable);
        assert!(!rows[2].still_profitable);
        assert!(execution_sensitivity(&rows).contains("MODERATE"));
    }

    #[test]
    fn scaling_touches_only_cost_legs() {
        let base = ExecConfig::default();
        let s = scaled_costs(&base, 2.0);
        assert_eq!(s.commission, base.commission * 2.0);
        assert_eq!(s.spread, base.spread * 2.0);
        assert_eq!(s.slippage, base.slippage * 2.0);
        assert_eq!(s.stop_loss, base.stop_loss);
        assert_eq!(s.leverage, base.leverage);
    }
}
