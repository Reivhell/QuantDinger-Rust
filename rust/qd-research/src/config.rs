//! Typed research-run configuration: one JSON document drives an entire
//! research run (dataset + strategy + execution + robustness stages +
//! thresholds). No tuning constant lives anywhere else.
//!
//! Example (`research.json`):
//! ```json
//! {
//!   "data": {"asset": "BTC-USDT", "timeframe": "1D", "bars": 1500, "seed": 42,
//!            "csv": "/tmp/btc_usdt_1d.csv"},
//!   "strategy": {"name": "EmaCrossTrend", "fast": 20, "slow": 50,
//!                "stop_loss": 0.03, "take_profit": 0.06, "risk_fraction": 0.01},
//!   "execution": {"commission": 0.0005, "spread": 0.0002, "slippage": 0.0003,
//!                 "latency_bars": 1},
//!   "robustness": {"mc_sims": 2000, "cpcv_partitions": 6, "cpcv_test": 2,
//!                  "wf_train": 600, "wf_oos": 150, "wf_step": 150,
//!                  "min_oos_fraction": 0.5, "seed": 42}
//! }
//! ```

use qd_engine::json_helpers::{parse_json, JsonVal};

/// Full research-run configuration with documented defaults.
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    // data
    pub asset: String,
    pub timeframe: String,
    pub bars: usize,
    pub data_seed: u64,
    /// Path to a strict-CSV bar file (`t,open,high,low,close,volume`).
    /// Empty = deterministic synthetic fixture. Non-empty = REAL market
    /// data via [`crate::data::load_bars_csv`]; the fixture is never used.
    pub csv: String,
    // strategy
    pub strategy_name: String,
    pub ema_fast: usize,
    pub ema_slow: usize,
    pub rsi_period: usize,
    pub rsi_oversold: f64,
    pub rsi_overbought: f64,
    pub rsi_shorts: bool,
    pub donchian_lookback: usize,
    pub donchian_relvol: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub risk_fraction: f64,
    // hard risk limits (§1/§2/§4): deterministic caps the strategy,
    // optimizer, LLM context, and agent can never override.
    /// Hard ceiling on stop-loss distance (default 0.03 = 3%). Any
    /// configured `stop_loss` above this is rejected, not clamped.
    pub max_stop_loss: f64,
    /// Configured leverage (default 1.0 = spot). Rejected above
    /// [`crate::risk::MAX_LEVERAGE`]; below that, the liquidation buffer
    /// check ([`crate::risk::liquidation_ok`]) still applies.
    pub leverage: f64,
    /// Maximum simultaneous positions (§4).
    pub max_positions: usize,
    /// Maximum total open risk as fraction of equity (§4).
    pub max_open_risk: f64,
    /// Latch halt if intraday equity falls this fraction (§4).
    pub max_daily_loss: f64,
    /// Latch halt if equity falls this fraction below the weekly window
    /// peak (§4, 0 = disabled). Window = 5 daily sessions.
    pub max_weekly_loss: f64,
    /// Latch halt if equity falls this fraction below peak (§4).
    pub max_drawdown: f64,
    /// Latch halt if exposure notional exceeds this multiple of equity (§4).
    pub max_exposure: f64,
    /// Latch halt if correlated notional exceeds this multiple of equity
    /// (§4, 0 = disabled; single-name books set 0).
    pub max_correlated_exposure: f64,
    // execution
    pub commission: f64,
    pub spread: f64,
    pub slippage: f64,
    pub latency_bars: usize,
    // robustness
    pub mc_sims: usize,
    pub mc_seed: u64,
    pub cpcv_partitions: usize,
    pub cpcv_test: usize,
    pub wf_train: usize,
    pub wf_oos: usize,
    pub wf_step: usize,
    pub min_oos_fraction: f64,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            asset: "SYNTH".into(),
            timeframe: "1d".into(),
            bars: 1500,
            data_seed: 42,
            csv: String::new(),
            strategy_name: "EmaCrossTrend".into(),
            ema_fast: 20,
            ema_slow: 50,
            rsi_period: 14,
            rsi_oversold: 30.0,
            rsi_overbought: 70.0,
            rsi_shorts: false,
            donchian_lookback: 20,
            donchian_relvol: 1.5,
            stop_loss: 0.03,
            take_profit: 0.06,
            risk_fraction: 0.01,
            max_stop_loss: 0.03,
            leverage: 1.0,
            max_positions: 3,
            max_open_risk: 0.06,
            max_daily_loss: 0.02,
            max_weekly_loss: 0.05,
            max_drawdown: 0.10,
            max_exposure: 3.0,
            max_correlated_exposure: 0.0,
            commission: 0.0005,
            spread: 0.0002,
            slippage: 0.0003,
            latency_bars: 1,
            mc_sims: 10_000,
            mc_seed: 42,
            cpcv_partitions: 6,
            cpcv_test: 2,
            wf_train: 600,
            wf_oos: 150,
            wf_step: 150,
            min_oos_fraction: 0.5,
        }
    }
}

fn obj<'a>(v: &'a JsonVal, key: &str) -> Option<&'a JsonVal> {
    match v {
        JsonVal::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn str_of(v: &JsonVal) -> Option<String> {
    match v {
        JsonVal::Str(s) => Some(s.clone()),
        _ => None,
    }
}

fn num(v: &JsonVal) -> Option<f64> {
    v.as_f64()
}

/// Parse a JSON document into [`ResearchConfig`]. Unknown keys are ignored;
/// missing keys fall back to [`ResearchConfig::default`]. Type errors
/// (e.g. `"fast": "abc"`) are reported, never silently defaulted.
pub fn load_config(json: &str) -> Result<ResearchConfig, String> {
    let root = parse_json(json).map_err(|e| format!("config: invalid JSON: {e}"))?;
    if !matches!(root, JsonVal::Obj(_)) {
        return Err("config: top level must be an object".into());
    }
    let mut cfg = ResearchConfig::default();
    let mut bad: Vec<String> = Vec::new();
    for (sec, key, slot) in [
        ("data", "asset", &mut cfg.asset),
        ("data", "timeframe", &mut cfg.timeframe),
        ("data", "csv", &mut cfg.csv),
        ("strategy", "name", &mut cfg.strategy_name),
    ] {
        if let Some(s) = obj(&root, sec).and_then(|o| obj(o, key)) {
            match str_of(s) {
                Some(v) => *slot = v,
                None => bad.push(format!("{sec}.{key}: expected string")),
            }
        }
    }
    fn want(root: &JsonVal, bad: &mut Vec<String>, sec: &str, key: &str) -> Option<f64> {
        match obj(root, sec).and_then(|o| obj(o, key)) {
            None => None,
            Some(v) => match num(v) {
                Some(x) => Some(x),
                None => {
                    bad.push(format!("{sec}.{key}: expected number"));
                    None
                }
            },
        }
    }
    if let Some(v) = want(&root, &mut bad, "data", "bars") {
        cfg.bars = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "data", "seed") {
        cfg.data_seed = v as u64;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "fast") {
        cfg.ema_fast = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "slow") {
        cfg.ema_slow = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "rsi_period") {
        cfg.rsi_period = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "rsi_oversold") {
        cfg.rsi_oversold = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "rsi_overbought") {
        cfg.rsi_overbought = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "donchian_lookback") {
        cfg.donchian_lookback = (v as usize).max(1);
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "donchian_relvol") {
        cfg.donchian_relvol = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "stop_loss") {
        cfg.stop_loss = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "take_profit") {
        cfg.take_profit = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "risk_fraction") {
        cfg.risk_fraction = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_stop_loss") {
        cfg.max_stop_loss = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "leverage") {
        cfg.leverage = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_positions") {
        cfg.max_positions = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_open_risk") {
        cfg.max_open_risk = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_daily_loss") {
        cfg.max_daily_loss = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_weekly_loss") {
        cfg.max_weekly_loss = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_drawdown") {
        cfg.max_drawdown = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_exposure") {
        cfg.max_exposure = v;
    }
    if let Some(v) = want(&root, &mut bad, "strategy", "max_correlated_exposure") {
        cfg.max_correlated_exposure = v;
    }
    if let Some(v) = want(&root, &mut bad, "execution", "commission") {
        cfg.commission = v;
    }
    if let Some(v) = want(&root, &mut bad, "execution", "spread") {
        cfg.spread = v;
    }
    if let Some(v) = want(&root, &mut bad, "execution", "slippage") {
        cfg.slippage = v;
    }
    if let Some(v) = want(&root, &mut bad, "execution", "latency_bars") {
        cfg.latency_bars = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "mc_sims") {
        cfg.mc_sims = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "seed") {
        cfg.mc_seed = v as u64;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "cpcv_partitions") {
        cfg.cpcv_partitions = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "cpcv_test") {
        cfg.cpcv_test = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "wf_train") {
        cfg.wf_train = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "wf_oos") {
        cfg.wf_oos = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "wf_step") {
        cfg.wf_step = v as usize;
    }
    if let Some(v) = want(&root, &mut bad, "robustness", "min_oos_fraction") {
        cfg.min_oos_fraction = v;
    }
    if let Some(v) = obj(&root, "strategy").and_then(|o| obj(o, "rsi_shorts")) {
        match v {
            JsonVal::Bool(b) => cfg.rsi_shorts = *b,
            _ => bad.push("strategy.rsi_shorts: expected boolean".into()),
        }
    }
    if !bad.is_empty() {
        return Err(format!("config: type errors: {}", bad.join(", ")));
    }
    // Cross-field guards: nonsense configs fail loudly, not silently.
    if cfg.ema_fast == 0 || cfg.ema_slow == 0 {
        return Err("config: strategy.fast/slow must be >= 1".into());
    }
    if cfg.ema_fast >= cfg.ema_slow {
        return Err("config: strategy.fast must be < strategy.slow".into());
    }
    if cfg.strategy_name == "RsiMeanReversion" {
        if cfg.rsi_period < 2 {
            return Err("config: strategy.rsi_period must be >= 2".into());
        }
        if !(0.0 < cfg.rsi_oversold && cfg.rsi_oversold < cfg.rsi_overbought && cfg.rsi_overbought < 100.0) {
            return Err("config: need 0 < rsi_oversold < rsi_overbought < 100".into());
        }
    } else if cfg.strategy_name == "DonchianBreakout" {
        if cfg.donchian_relvol <= 0.0 {
            return Err("config: strategy.donchian_relvol must be > 0".into());
        }
    } else if cfg.strategy_name != "EmaCrossTrend" {
        return Err(format!("config: unknown strategy.name '{}' (EmaCrossTrend | RsiMeanReversion | DonchianBreakout)", cfg.strategy_name));
    }
    if cfg.bars < cfg.wf_train + cfg.wf_oos {
        return Err("config: data.bars must fit one walk-forward train+oos window".into());
    }
    if cfg.cpcv_test == 0 || cfg.cpcv_test >= cfg.cpcv_partitions.max(2) {
        return Err("config: need 1 <= robustness.cpcv_test < cpcv_partitions".into());
    }
    // Hard risk guards (§1/§2/§4): reject, never clamp. The strategy,
    // optimizer, LLM context, and agent cannot override these.
    if cfg.stop_loss <= 0.0 {
        return Err("config: strategy.stop_loss must be > 0 (mandatory protective stop)".into());
    }
    if cfg.max_stop_loss <= 0.0 || cfg.stop_loss > cfg.max_stop_loss {
        return Err("config: strategy.stop_loss exceeds strategy.max_stop_loss (3% cap)".into());
    }
    if cfg.leverage < 1.0 || cfg.leverage > crate::risk::MAX_LEVERAGE {
        return Err(format!(
            "config: strategy.leverage must be in [1.0, {}] (spot default)",
            crate::risk::MAX_LEVERAGE
        ));
    }
    if !crate::risk::liquidation_ok(cfg.leverage, cfg.stop_loss, 3.0) {
        return Err("config: liquidation buffer < 3x stop distance (leverage unsafe)".into());
    }
    if cfg.risk_fraction <= 0.0 || cfg.risk_fraction > 0.05 {
        return Err("config: strategy.risk_fraction must be in (0, 0.05]".into());
    }
    if cfg.max_positions == 0 {
        return Err("config: strategy.max_positions must be >= 1".into());
    }
    if cfg.max_open_risk <= 0.0 || cfg.max_drawdown <= 0.0 || cfg.max_daily_loss <= 0.0 {
        return Err("config: max_open_risk/max_drawdown/max_daily_loss must all be > 0".into());
    }
    if cfg.max_weekly_loss < 0.0 || cfg.max_correlated_exposure < 0.0 {
        return Err("config: max_weekly_loss/max_correlated_exposure must be >= 0".into());
    }
    Ok(cfg)
}

/// Canonical default config document (what [`ResearchConfig::default`]
/// parses from). Kept next to the parser so the two cannot drift.
pub const DEFAULT_CONFIG_JSON: &str = r#"{
  "data": {"asset": "SYNTH", "timeframe": "1d", "bars": 1500, "seed": 42},
  "strategy": {"name": "EmaCrossTrend", "fast": 20, "slow": 50,
               "stop_loss": 0.03, "take_profit": 0.06, "risk_fraction": 0.01},
  "execution": {"commission": 0.0005, "spread": 0.0002, "slippage": 0.0003,
                "latency_bars": 1},
  "robustness": {"mc_sims": 10000, "seed": 42, "cpcv_partitions": 6,
                 "cpcv_test": 2, "wf_train": 600, "wf_oos": 150,
                 "wf_step": 150, "min_oos_fraction": 0.5}
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_json_round_trips() {
        let c = load_config(DEFAULT_CONFIG_JSON).expect("default parses");
        let d = ResearchConfig::default();
        assert_eq!(c.bars, d.bars);
        assert_eq!((c.ema_fast, c.ema_slow), (d.ema_fast, d.ema_slow));
        assert_eq!((c.mc_sims, c.mc_seed), (d.mc_sims, d.mc_seed));
        assert_eq!(c.strategy_name, d.strategy_name);
    }

    #[test]
    fn partial_config_falls_back() {
        let c = load_config(r#"{"strategy": {"fast": 10, "slow": 30}}"#).unwrap();
        assert_eq!((c.ema_fast, c.ema_slow), (10, 30));
        assert_eq!(c.bars, 1500); // default kept
    }

    #[test]
    fn csv_path_parses_and_defaults_empty() {
        let c = load_config(DEFAULT_CONFIG_JSON).expect("default parses");
        assert!(c.csv.is_empty()); // synthetic fixture unless configured
        let r = load_config(r#"{"data": {"csv": "/tmp/btc_usdt_1d.csv", "bars": 1000}}"#).unwrap();
        assert_eq!(r.csv, "/tmp/btc_usdt_1d.csv");
        assert!(load_config(r#"{"data": {"csv": 42}}"#).is_err()); // type errors are loud
    }

    #[test]
    fn type_errors_are_loud() {
        assert!(load_config(r#"{"strategy": {"fast": "abc"}}"#).is_err());
        assert!(load_config(r#"{"strategy": {"fast": 50, "slow": 20}}"#).is_err());
        assert!(load_config(r#"[1,2]"#).is_err());
        assert!(load_config(r#"{"data": {"bars": 10}}"#).is_err()); // too small for WF
    }

    #[test]
    fn hard_risk_guards_reject_never_clamp() {
        // Stop above the 3% cap: rejected outright.
        assert!(load_config(r#"{"strategy": {"stop_loss": 0.05}}"#).is_err());
        // No protective stop: rejected.
        assert!(load_config(r#"{"strategy": {"stop_loss": 0.0}}"#).is_err());
        // Excessive leverage: rejected.
        assert!(load_config(r#"{"strategy": {"leverage": 20.0}}"#).is_err());
        // Leverage whose liquidation buffer (< 3x stop) is unsafe: rejected.
        // 5x → distance 0.20 vs 3x stop 0.09: passes; 0.20/0.03 = 6.7x ok.
        assert!(load_config(r#"{"strategy": {"leverage": 5.0}}"#).is_ok());
        // Oversized per-trade risk: rejected.
        assert!(load_config(r#"{"strategy": {"risk_fraction": 0.25}}"#).is_err());
        // Zero position cap: rejected.
        assert!(load_config(r#"{"strategy": {"max_positions": 0}}"#).is_err());
    }
}
