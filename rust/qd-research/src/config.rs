//! Typed research-run configuration: one JSON document drives an entire
//! research run (dataset + strategy + execution + robustness stages +
//! thresholds). No tuning constant lives anywhere else.
//!
//! Example (`research.json`):
//! ```json
//! {
//!   "data": {"asset": "SYNTH", "timeframe": "1d", "bars": 1500, "seed": 42},
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
    // strategy
    pub strategy_name: String,
    pub ema_fast: usize,
    pub ema_slow: usize,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub risk_fraction: f64,
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
            strategy_name: "EmaCrossTrend".into(),
            ema_fast: 20,
            ema_slow: 50,
            stop_loss: 0.03,
            take_profit: 0.06,
            risk_fraction: 0.01,
            commission: 0.0005,
            spread: 0.0002,
            slippage: 0.0003,
            latency_bars: 1,
            mc_sims: 2000,
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
        ("strategy", "name", &mut cfg.strategy_name),
    ] {
        if let Some(s) = obj(&root, sec).and_then(|o| obj(o, key)) {
            match str_of(s) {
                Some(v) => *slot = v,
                None => bad.push(format!("{sec}.{key}: expected string")),
            }
        }
    }
    let mut want_num = |sec: &str, key: &str| -> Option<f64> {
        match obj(&root, sec).and_then(|o| obj(o, key)) {
            None => None,
            Some(v) => match num(v) {
                Some(x) => Some(x),
                None => {
                    bad.push(format!("{sec}.{key}: expected number"));
                    None
                }
            },
        }
    };
    if let Some(v) = want_num("data", "bars") {
        cfg.bars = v as usize;
    }
    if let Some(v) = want_num("data", "seed") {
        cfg.data_seed = v as u64;
    }
    if let Some(v) = want_num("strategy", "fast") {
        cfg.ema_fast = v as usize;
    }
    if let Some(v) = want_num("strategy", "slow") {
        cfg.ema_slow = v as usize;
    }
    if let Some(v) = want_num("strategy", "stop_loss") {
        cfg.stop_loss = v;
    }
    if let Some(v) = want_num("strategy", "take_profit") {
        cfg.take_profit = v;
    }
    if let Some(v) = want_num("strategy", "risk_fraction") {
        cfg.risk_fraction = v;
    }
    if let Some(v) = want_num("execution", "commission") {
        cfg.commission = v;
    }
    if let Some(v) = want_num("execution", "spread") {
        cfg.spread = v;
    }
    if let Some(v) = want_num("execution", "slippage") {
        cfg.slippage = v;
    }
    if let Some(v) = want_num("execution", "latency_bars") {
        cfg.latency_bars = v as usize;
    }
    if let Some(v) = want_num("robustness", "mc_sims") {
        cfg.mc_sims = v as usize;
    }
    if let Some(v) = want_num("robustness", "seed") {
        cfg.mc_seed = v as u64;
    }
    if let Some(v) = want_num("robustness", "cpcv_partitions") {
        cfg.cpcv_partitions = v as usize;
    }
    if let Some(v) = want_num("robustness", "cpcv_test") {
        cfg.cpcv_test = v as usize;
    }
    if let Some(v) = want_num("robustness", "wf_train") {
        cfg.wf_train = v as usize;
    }
    if let Some(v) = want_num("robustness", "wf_oos") {
        cfg.wf_oos = v as usize;
    }
    if let Some(v) = want_num("robustness", "wf_step") {
        cfg.wf_step = v as usize;
    }
    if let Some(v) = want_num("robustness", "min_oos_fraction") {
        cfg.min_oos_fraction = v;
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
    if cfg.bars < cfg.wf_train + cfg.wf_oos {
        return Err("config: data.bars must fit one walk-forward train+oos window".into());
    }
    if cfg.cpcv_test == 0 || cfg.cpcv_test >= cfg.cpcv_partitions.max(2) {
        return Err("config: need 1 <= robustness.cpcv_test < cpcv_partitions".into());
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
  "robustness": {"mc_sims": 2000, "seed": 42, "cpcv_partitions": 6,
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
    fn type_errors_are_loud() {
        assert!(load_config(r#"{"strategy": {"fast": "abc"}}"#).is_err());
        assert!(load_config(r#"{"strategy": {"fast": 50, "slow": 20}}"#).is_err());
        assert!(load_config(r#"[1,2]"#).is_err());
        assert!(load_config(r#"{"data": {"bars": 10}}"#).is_err()); // too small for WF
    }
}
