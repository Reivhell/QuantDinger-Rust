//! Shared research pipeline: one choke point for config → bars → signals →
//! backtest → gate inputs.
//!
//! Both real runners call this; no pipeline logic lives in the binaries.
//! This is the DRY home for constants and builders previously copy-pasted
//! across `research_run` and `autonomy_check` (starting equity, periods per
//! year, position cap, blocked-regime policy, exec mapping, signal builders).
//!
//! Body/brain contract: the Python backend (`backend_api_python/`) is the
//! brain — it owns data fetching, scheduling, and order submission. This
//! crate is the body: pure computation over bars it is given. It never does
//! I/O except reading the CSV path the caller hands it (`load_bars`), never
//! touches the network/DB, and reports evidence the brain decides on.

use crate::backtest::{ExecConfig, Signal};
use crate::config::ResearchConfig;
use crate::features::Bar;
use crate::regime::Regime;
use crate::shadow::ShadowConfig;
use crate::strategies::{DonchianBreakout, EmaCrossTrend, RsiMeanReversion};

/// Starting equity every backtest/shadow run is scored on (USD).
pub const STARTING_EQUITY: f64 = 100_000.0;
/// Annualisation periods: daily bars → 252 trading days/year.
pub const PERIODS_PER_YEAR: f64 = 252.0;
/// Hard per-position notional cap (USD) for the fixed-fractional sizer.
pub const MAX_POSITION_NOTIONAL: f64 = 20_000.0;
/// Bars per paper session marker (kill-switch daily accounting in shadow).
/// Bar-count based: shadow has no wall clock.
pub const SESSION_BARS: usize = 50;
/// Weekly window for the kill-switch, in paper sessions.
pub const WEEK_BARS: u32 = 5;
/// Cost-stress multipliers for [`crate::costs::cost_stress`]: baseline,
/// +25%, +50%, +100%, +200%.
pub const COST_MULTIPLIERS: [f64; 5] = [1.0, 1.25, 1.5, 2.0, 3.0];
/// Vol level at/below which the autonomy check's adaptive sizer applies
/// full size (20% annualized). High trailing vol shrinks size; calm
/// markets never inflate it above the base risk fraction.
pub const TARGET_VOL_ANN: f64 = 0.20;
/// Trailing window (bars) for the autonomy check's realized-vol estimate.
pub const VOL_WINDOW: usize = 20;

/// Strategy family selected by `config.strategy.name`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyFam {
    Ema,
    Rsi,
    Donchian,
}

/// Map a config strategy name to its family. Unknown names cannot occur
/// (config validation rejects them); they read as EMA defensively.
pub fn family_of(name: &str) -> StrategyFam {
    match name {
        "RsiMeanReversion" => StrategyFam::Rsi,
        "DonchianBreakout" => StrategyFam::Donchian,
        _ => StrategyFam::Ema,
    }
}

/// Load REAL market bars for a research run. `cfg.csv` must point at a
/// strict-CSV bar file (`t,open,high,low,close,volume`, e.g. via
/// `fetch_okx.sh`). Empty `csv` is an error — the deterministic synthetic
/// fixture (`data::synthetic_bars`) is unit-test only, never a substitute
/// for market data.
pub fn load_bars(cfg: &ResearchConfig) -> Result<(Vec<Bar>, String), String> {
    if cfg.csv.trim().is_empty() {
        return Err(
            "config: data.csv is required — real market bars only (fetch via \
             qd-research/fetch_okx.sh); the synthetic fixture is unit-test only"
                .to_string(),
        );
    }
    let text = std::fs::read_to_string(&cfg.csv)
        .map_err(|e| format!("cannot read data.csv {}: {e}", cfg.csv))?;
    let loaded =
        crate::data::load_bars_csv(&text).map_err(|e| format!("invalid CSV {}: {e}", cfg.csv))?;
    let label = format!(
        "REAL {} ({} bars, t {}..{})",
        cfg.csv,
        loaded.len(),
        loaded.first().map(|b| b.t).unwrap_or(0),
        loaded.last().map(|b| b.t).unwrap_or(0)
    );
    Ok((loaded, label))
}

/// Build the configured strategy's full-series signals.
pub fn build_signals(
    fam: StrategyFam,
    cfg: &ResearchConfig,
    bars: &[Bar],
    closes: &[f64],
    regimes: &[Regime],
) -> Vec<Signal> {
    match fam {
        StrategyFam::Rsi => RsiMeanReversion {
            period: cfg.rsi_period,
            oversold: cfg.rsi_oversold,
            overbought: cfg.rsi_overbought,
            allow_shorts: cfg.rsi_shorts,
        }
        .signals(closes, regimes),
        StrategyFam::Donchian => DonchianBreakout {
            lookback: cfg.donchian_lookback,
            relvol_min: cfg.donchian_relvol,
        }
        .signals(bars, regimes),
        StrategyFam::Ema => EmaCrossTrend { fast: cfg.ema_fast, slow: cfg.ema_slow }
            .signals(closes, regimes),
    }
}

/// Short per-fold label for walk-forward bookkeeping (`ema20/50`,
/// `rsi14/30-70`, `donch20/rv1.5`).
pub fn fold_label(fam: StrategyFam, cfg: &ResearchConfig) -> String {
    match fam {
        StrategyFam::Rsi => {
            format!("rsi{}/{}-{}", cfg.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought)
        }
        StrategyFam::Donchian => {
            format!("donch{}/rv{:.1}", cfg.donchian_lookback, cfg.donchian_relvol)
        }
        StrategyFam::Ema => format!("ema{}/{}", cfg.ema_fast, cfg.ema_slow),
    }
}

/// Full strategy identity for the report + version registry:
/// `(display_label, param_string)`.
pub fn strategy_identity(fam: StrategyFam, cfg: &ResearchConfig) -> (String, String) {
    match fam {
        StrategyFam::Donchian => (
            format!("DonchianBreakout({},rv{:.1})", cfg.donchian_lookback, cfg.donchian_relvol),
            format!(
                "lookback={} relvol={:.2} stop={:.3} take={:.3} risk={:.3}",
                cfg.donchian_lookback,
                cfg.donchian_relvol,
                cfg.stop_loss,
                cfg.take_profit,
                cfg.risk_fraction
            ),
        ),
        StrategyFam::Rsi => (
            format!(
                "RsiMeanReversion({},{:.0},{:.0}{})",
                cfg.rsi_period,
                cfg.rsi_oversold,
                cfg.rsi_overbought,
                if cfg.rsi_shorts { ",shorts" } else { ",long-only" }
            ),
            format!(
                "period={} os={:.0} ob={:.0} shorts={} stop={:.3} take={:.3} risk={:.3}",
                cfg.rsi_period,
                cfg.rsi_oversold,
                cfg.rsi_overbought,
                cfg.rsi_shorts,
                cfg.stop_loss,
                cfg.take_profit,
                cfg.risk_fraction
            ),
        ),
        StrategyFam::Ema => (
            format!("EmaCrossTrend({},{})", cfg.ema_fast, cfg.ema_slow),
            format!(
                "fast={} slow={} stop={:.3} take={:.3} risk={:.3}",
                cfg.ema_fast, cfg.ema_slow, cfg.stop_loss, cfg.take_profit, cfg.risk_fraction
            ),
        ),
    }
}

/// Execution pricing from config. `fill_fraction`/`trailing`/`funding`
/// stay at neutral defaults (full fills, no trailing, no funding) — the
/// config surface covers commission/spread/slippage/latency/leverage/stop/take.
pub fn exec_of(cfg: &ResearchConfig) -> ExecConfig {
    ExecConfig {
        commission: cfg.commission,
        spread: cfg.spread,
        slippage: cfg.slippage,
        latency_bars: cfg.latency_bars,
        stop_loss: cfg.stop_loss,
        take_profit: cfg.take_profit,
        leverage: cfg.leverage,
        ..ExecConfig::default()
    }
}

/// Leverage policy from config: within `[1.0, MAX_LEVERAGE]` with a
/// liquidation buffer of at least 3x the stop distance.
pub fn leverage_ok(cfg: &ResearchConfig) -> bool {
    cfg.leverage >= 1.0
        && cfg.leverage <= crate::risk::MAX_LEVERAGE
        && crate::risk::liquidation_ok(cfg.leverage, cfg.stop_loss, 3.0)
}
/// Fixed-fractional position size on the shared caps: risk
/// `cfg.risk_fraction` of equity against the `cfg.stop_loss` distance,
/// capped at [`MAX_POSITION_NOTIONAL`].
pub fn fixed_qty(price: f64, equity: f64, cfg: &ResearchConfig) -> f64 {
    crate::risk::fixed_fractional_qty(price, equity, cfg.risk_fraction, cfg.stop_loss, MAX_POSITION_NOTIONAL)
}

/// Regimes where new entries are forbidden (caller policy, §5): chop,
/// volatility extremes, counter-trend turf, and integrity-uncertain prints.
/// Shared by the research shadow bulkhead and the autonomy gate.
pub fn blocked_regimes() -> [Regime; 5] {
    [
        Regime::Ranging,
        Regime::HighVolatility,
        Regime::LowVolatility,
        Regime::MeanReversion,
        Regime::Abnormal,
    ]
}

/// Shadow bulkhead config from research caps. `leverage_ok` comes from the
/// caller (config validation + liquidation-buffer check).
pub fn shadow_config(cfg: &ResearchConfig, leverage_ok: bool) -> ShadowConfig {
    ShadowConfig {
        starting_equity: STARTING_EQUITY,
        max_positions: cfg.max_positions,
        max_open_risk: cfg.max_open_risk,
        blocked_regimes: blocked_regimes().to_vec(),
        leverage_ok,
        stop_present: cfg.stop_loss > 0.0,
        execution_ok: true,
        ks_daily: cfg.max_daily_loss,
        ks_weekly: cfg.max_weekly_loss,
        ks_drawdown: cfg.max_drawdown,
        ks_exposure: cfg.max_exposure,
        ks_corr: cfg.max_correlated_exposure,
        week_bars: WEEK_BARS,
        session_bars: SESSION_BARS,
    }
}

/// Duller incumbent sibling of the same family for the shadow comparison:
/// slower/longer parameters on the same unseen window.
pub fn incumbent_signals(
    fam: StrategyFam,
    cfg: &ResearchConfig,
    bars: &[Bar],
    closes: &[f64],
    regimes: &[Regime],
) -> Vec<Signal> {
    match fam {
        StrategyFam::Donchian => DonchianBreakout {
            lookback: cfg.donchian_lookback + 10,
            relvol_min: cfg.donchian_relvol,
        }
        .signals(bars, regimes),
        StrategyFam::Rsi => RsiMeanReversion {
            period: cfg.rsi_period + 7,
            oversold: cfg.rsi_oversold - 5.0,
            overbought: cfg.rsi_overbought,
            allow_shorts: cfg.rsi_shorts,
        }
        .signals(closes, regimes),
        StrategyFam::Ema => EmaCrossTrend { fast: cfg.ema_fast + 10, slow: cfg.ema_slow + 50 }
            .signals(closes, regimes),
    }
}

/// Length-normalized grid score: CAGR on any window that traded, `0.0` on
/// genuinely flat (0-trade) windows. Raw totals are length-biased (a 600-bar
/// IS window vs a 150-bar OOS window); CAGR annualizes so IS/OOS levels are
/// comparable. 1–2 trade windows keep their real CAGR — zeroing them
/// fabricates flat windows and pins short-window medians at exactly 0.0.
pub fn robust_cagr(num_trades: usize, cagr: f64) -> f64 {
    if num_trades == 0 { 0.0 } else { cagr }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_maps_all_known_names() {
        assert_eq!(family_of("EmaCrossTrend"), StrategyFam::Ema);
        assert_eq!(family_of("RsiMeanReversion"), StrategyFam::Rsi);
        assert_eq!(family_of("DonchianBreakout"), StrategyFam::Donchian);
    }

    #[test]
    fn empty_csv_is_an_error_not_a_fixture() {
        let cfg = ResearchConfig::default();
        assert!(cfg.csv.is_empty());
        let err = load_bars(&cfg).expect_err("empty csv must fail");
        assert!(err.contains("data.csv is required"));
    }

    #[test]
    fn robust_cagr_floors_only_flat_windows() {
        assert_eq!(robust_cagr(0, 0.5), 0.0);
        assert!((robust_cagr(2, -0.13) - -0.13).abs() < 1e-12);
    }

    #[test]
    fn blocked_regimes_are_the_five_no_thesis_states() {
        let b = blocked_regimes();
        assert_eq!(b.len(), 5);
        assert!(b.contains(&Regime::Abnormal));
        assert!(!b.contains(&Regime::TrendingUp));
    }

    #[test]
    fn exec_maps_config_pricing() {
        let cfg = ResearchConfig::default();
        let e = exec_of(&cfg);
        assert_eq!((e.commission, e.spread, e.slippage), (cfg.commission, cfg.spread, cfg.slippage));
        assert_eq!(e.stop_loss, cfg.stop_loss);
    }
}
