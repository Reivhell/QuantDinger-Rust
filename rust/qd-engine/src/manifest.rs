//! Port of `backend_api_python/app/services/strategy_v2/models.py`.
//!
//! Immutable contract models: universe / subscription / schedule specs and
//! the strategy manifest. Only the computable derivations are ported:
//! `key` (reused from [`crate::instruments`]), `metadata` shapes,
//! `markets`, `primary_frequency`, `frequencies`, `driving_frequency`.
//!
//! `metadata` dicts become ordered `Vec<(key, JsonVal)>` pairs — field order
//! mirrors the Python literal insertion order exactly (the parity check
//! compares `json.dumps(..., sort_keys=True)` on both sides, so order is
//! cosmetic but kept faithful anyway).

use std::collections::BTreeSet;

use crate::frequencies::{driving_frequency, unique_frequencies};
use crate::instruments::InstrumentSpec;

/// Minimal JSON value for metadata shapes (no floats needed — models carry
/// none except `max_leverage`, handled as raw bits string... see below).
#[derive(Debug, Clone, PartialEq)]
pub enum Meta {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    List(Vec<Meta>),
    Map(Vec<(String, Meta)>),
}

impl Meta {
    pub fn dump(&self) -> String {
        match self {
            Meta::Str(s) => format!("{s:?}"),
            Meta::Int(i) => i.to_string(),
            Meta::Float(f) => {
                let s = format!("{f:?}");
                if s.contains(['.', 'e']) { s } else { format!("{s}.0") }
            }
            Meta::Bool(b) => b.to_string(),
            Meta::Null => "null".to_string(),
            Meta::List(xs) => format!("[{}]", xs.iter().map(|x| x.dump()).collect::<Vec<_>>().join(",")),
            Meta::Map(kv) => format!(
                "{{{}}}",
                kv.iter().map(|(k, v)| format!("{k:?}:{}", v.dump())).collect::<Vec<_>>().join(",")
            ),
        }
    }
}

pub fn instrument_metadata(spec: &InstrumentSpec) -> Meta {
    // asdict field order: market, symbol, exchange_id, market_type, instrument_id
    Meta::Map(vec![
        ("market".into(), Meta::Str(spec.market.clone())),
        ("symbol".into(), Meta::Str(spec.symbol.clone())),
        ("exchange_id".into(), Meta::Str(spec.exchange_id.clone())),
        ("market_type".into(), Meta::Str(spec.market_type.clone())),
        ("instrument_id".into(), Meta::Str(spec.instrument_id.clone())),
    ])
}

#[derive(Debug, Clone, Default)]
pub struct UniverseSpec {
    pub kind: String,
    pub reference: String,
    pub instruments: Vec<InstrumentSpec>,
}

impl UniverseSpec {
    pub fn metadata(&self) -> Meta {
        Meta::Map(vec![
            ("kind".into(), Meta::Str(self.kind.clone())),
            ("reference".into(), Meta::Str(self.reference.clone())),
            (
                "instruments".into(),
                Meta::List(self.instruments.iter().map(instrument_metadata).collect()),
            ),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct SubscriptionSpec {
    pub instruments: Vec<InstrumentSpec>,
    pub universe_reference: String,
    pub frequency: String,
    pub fields: Vec<String>,
}

impl Default for SubscriptionSpec {
    fn default() -> Self {
        Self {
            instruments: vec![],
            universe_reference: String::new(),
            frequency: "1d".to_string(),
            fields: ["open", "high", "low", "close", "volume"].iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl SubscriptionSpec {
    pub fn metadata(&self) -> Meta {
        Meta::Map(vec![
            (
                "instruments".into(),
                Meta::List(self.instruments.iter().map(instrument_metadata).collect()),
            ),
            ("universeReference".into(), Meta::Str(self.universe_reference.clone())),
            ("frequency".into(), Meta::Str(self.frequency.clone())),
            ("fields".into(), Meta::List(self.fields.iter().map(|f| Meta::Str(f.clone())).collect())),
        ])
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScheduleSpec {
    pub frequency: String,
    pub callback: String,
    pub time: String,
    pub weekday: Option<i64>,
    pub monthday: Option<i64>,
}

impl ScheduleSpec {
    pub fn metadata(&self) -> Meta {
        Meta::Map(vec![
            ("frequency".into(), Meta::Str(self.frequency.clone())),
            ("callback".into(), Meta::Str(self.callback.clone())),
            ("time".into(), Meta::Str(self.time.clone())),
            ("weekday".into(), self.weekday.map(Meta::Int).unwrap_or(Meta::Null)),
            ("monthday".into(), self.monthday.map(Meta::Int).unwrap_or(Meta::Null)),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct StrategyManifest {
    pub api_version: i64,
    pub code_hash: String,
    pub strategy_type: String,
    pub universe: UniverseSpec,
    pub subscriptions: Vec<SubscriptionSpec>,
    pub schedules: Vec<ScheduleSpec>,
    pub benchmark: Option<InstrumentSpec>,
    pub handlers: Vec<String>,
    pub factor_dependencies: Vec<String>,
    pub fundamental_dependencies: Vec<String>,
    pub warmup_bars: i64,
    pub leverage_allowed: bool,
    pub max_leverage: f64,
    pub direction_mode: String,
    pub metadata_fields: Vec<(String, String)>,
}

impl StrategyManifest {
    /// Sorted unique market names. Mirrors `markets`.
    pub fn markets(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        if let Some(b) = &self.benchmark {
            set.insert(b.market.clone());
        }
        for i in &self.universe.instruments {
            set.insert(i.market.clone());
        }
        for s in &self.subscriptions {
            for i in &s.instruments {
                set.insert(i.market.clone());
            }
        }
        set.into_iter().collect()
    }

    /// First subscription's frequency, else `"1d"`.
    pub fn primary_frequency(&self) -> String {
        self.subscriptions.first().map(|s| s.frequency.clone()).unwrap_or_else(|| "1d".to_string())
    }

    /// Deduped subscription frequencies in first-seen order.
    pub fn frequencies(&self) -> Vec<String> {
        let vals: Vec<&str> = self.subscriptions.iter().map(|s| s.frequency.as_str()).collect();
        unique_frequencies(&vals, "1d")
    }

    pub fn driving_frequency(&self) -> String {
        let freqs = self.frequencies();
        let refs: Vec<&str> = freqs.iter().map(|s| s.as_str()).collect();
        driving_frequency(&refs, "1d")
    }

    pub fn metadata(&self) -> Meta {
        Meta::Map(vec![
            ("apiVersion".into(), Meta::Int(self.api_version)),
            ("codeHash".into(), Meta::Str(self.code_hash.clone())),
            ("strategyType".into(), Meta::Str(self.strategy_type.clone())),
            ("primaryFrequency".into(), Meta::Str(self.primary_frequency())),
            ("drivingFrequency".into(), Meta::Str(self.driving_frequency())),
            (
                "frequencies".into(),
                Meta::List(self.frequencies().into_iter().map(Meta::Str).collect()),
            ),
            ("markets".into(), Meta::List(self.markets().into_iter().map(Meta::Str).collect())),
            ("universe".into(), self.universe.metadata()),
            (
                "subscriptions".into(),
                Meta::List(self.subscriptions.iter().map(|s| s.metadata()).collect()),
            ),
            ("schedules".into(), Meta::List(self.schedules.iter().map(|s| s.metadata()).collect())),
            ("benchmark".into(), self.benchmark.as_ref().map(instrument_metadata).unwrap_or(Meta::Null)),
            ("handlers".into(), Meta::List(self.handlers.iter().map(|h| Meta::Str(h.clone())).collect())),
            (
                "factorDependencies".into(),
                Meta::List(self.factor_dependencies.iter().map(|h| Meta::Str(h.clone())).collect()),
            ),
            (
                "fundamentalDependencies".into(),
                Meta::List(self.fundamental_dependencies.iter().map(|h| Meta::Str(h.clone())).collect()),
            ),
            ("warmupBars".into(), Meta::Int(self.warmup_bars)),
            ("leverageAllowed".into(), Meta::Bool(self.leverage_allowed)),
            ("maxLeverage".into(), Meta::Float(self.max_leverage)),
            ("directionMode".into(), Meta::Str(self.direction_mode.clone())),
            (
                "metadata".into(),
                Meta::Map(self.metadata_fields.iter().map(|(k, v)| (k.clone(), Meta::Str(v.clone()))).collect()),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(market: &str, symbol: &str) -> InstrumentSpec {
        InstrumentSpec {
            market: market.into(),
            symbol: symbol.into(),
            exchange_id: String::new(),
            market_type: String::new(),
            instrument_id: format!("{market}:{symbol}"),
        }
    }

    fn manifest() -> StrategyManifest {
        StrategyManifest {
            api_version: 2,
            code_hash: "abc".into(),
            strategy_type: "test".into(),
            universe: UniverseSpec {
                kind: "static".into(),
                reference: String::new(),
                instruments: vec![inst("USStock", "AAPL"), inst("Crypto", "BTC/USDT")],
            },
            subscriptions: vec![
                SubscriptionSpec {
                    instruments: vec![inst("USStock", "AAPL")],
                    frequency: "1h".into(),
                    ..Default::default()
                },
                SubscriptionSpec {
                    instruments: vec![inst("Crypto", "BTC/USDT")],
                    frequency: "1d".into(),
                    ..Default::default()
                },
            ],
            schedules: vec![ScheduleSpec {
                frequency: "1d".into(),
                callback: "on_open".into(),
                time: "09:30".into(),
                weekday: None,
                monthday: None,
            }],
            benchmark: Some(inst("USStock", "SPY")),
            handlers: vec!["on_bar".into()],
            factor_dependencies: vec![],
            fundamental_dependencies: vec!["pe".into()],
            warmup_bars: 50,
            leverage_allowed: false,
            max_leverage: 1.0,
            direction_mode: "long".into(),
            metadata_fields: vec![],
        }
    }

    #[test]
    fn derived_sets_match_python() {
        let m = manifest();
        assert_eq!(m.markets(), vec!["Crypto".to_string(), "USStock".to_string()]);
        assert_eq!(m.primary_frequency(), "1h");
        assert_eq!(m.frequencies(), vec!["1h".to_string(), "1d".to_string()]);
        assert_eq!(m.driving_frequency(), "1h");
    }

    #[test]
    fn empty_manifest_defaults() {
        let m = StrategyManifest {
            api_version: 2,
            code_hash: String::new(),
            strategy_type: String::new(),
            universe: UniverseSpec::default(),
            subscriptions: vec![],
            schedules: vec![],
            benchmark: None,
            handlers: vec![],
            factor_dependencies: vec![],
            fundamental_dependencies: vec![],
            warmup_bars: 0,
            leverage_allowed: false,
            max_leverage: 1.0,
            direction_mode: String::new(),
            metadata_fields: vec![],
        };
        assert!(m.markets().is_empty());
        assert_eq!(m.primary_frequency(), "1d");
        // unique_frequencies([]) -> (default,) — never empty, mirrors Python
        assert_eq!(m.frequencies(), vec!["1d".to_string()]);
        assert_eq!(m.driving_frequency(), "1d");
    }

    #[test]
    fn metadata_keys_mirror_python() {
        let md = manifest().metadata().dump();
        for key in [
            "apiVersion", "codeHash", "strategyType", "primaryFrequency",
            "drivingFrequency", "frequencies", "markets", "universe",
            "subscriptions", "schedules", "benchmark", "handlers",
            "factorDependencies", "fundamentalDependencies", "warmupBars",
            "leverageAllowed", "maxLeverage", "directionMode", "metadata",
        ] {
            assert!(md.contains(&format!("\"{key}\":")), "{key}");
        }
    }
}
