//! Port of the `ast`-free slice of
//! `backend_api_python/app/services/strategy_v2/contract.py`:
//! [`source_code_hash`], [`is_strategy_v2_code`], [`parse_many`], the
//! [`DiscoveryContext`] builder methods (`set_universe`, `set_benchmark`,
//! `subscribe`, `set_warmup`, `allow_leverage`, `set_metadata`), the schedule
//! bindings (`daily`/`weekly`/`monthly` via [`schedule`]), and
//! [`schedule_callback`].
//!
//! Everything that needs `ast.parse` (`compile_strategy_v2` validation,
//! `_literal_requested_frequencies`, `_discover_dependencies`,
//! `_collect_static_api_values` and friends) plus the sandboxed exec stays
//! Python-side. Errors mirror the Python exception kinds: [`BuilderError`]
//! separates `strategyV2.*` contract errors from `ValueError`/`TypeError`
//! raised by `int()`/`float()` coercions and bad `set_metadata` shapes.

use crate::direction::{is_falsy, or_str};
use crate::frequencies::normalize_frequency;
use crate::instruments::{is_index_reference, normalize_index_reference, normalize_pool_reference, parse_instrument, InstrumentSpec};
use crate::json_helpers::JsonVal;
use crate::manifest::{ScheduleSpec, SubscriptionSpec};
use crate::snapshot::sha256_hex;

/// Contract (`StrategyV2ContractError`) vs coercion errors.
#[derive(Debug, Clone, PartialEq)]
pub enum BuilderError {
    Contract(String),
    Value(String),
    Type(String),
}

/// Mirrors `strategy_source_code_hash`: `str(code or "")`, strip, sha256 hex.
pub fn source_code_hash(code: Option<&str>) -> String {
    sha256_hex(code.unwrap_or("").trim().as_bytes())
}

/// Mirrors `is_strategy_v2_code` (pure substring checks on `str(code or "")`).
pub fn is_strategy_v2_code(code: Option<&str>) -> bool {
    let raw = code.unwrap_or("");
    raw.contains("def initialize(")
        && raw.contains("context.set_universe(")
        && ["context.subscribe(", "run_daily(", "run_weekly(", "run_monthly("]
            .iter()
            .any(|t| raw.contains(t))
}

/// Python `int(x)` for JSON values (`set_warmup`, `weekday`, `monthday`).
/// `Err` mirrors `ValueError`.
fn py_int(value: &JsonVal) -> Result<i64, BuilderError> {
    match value {
        JsonVal::Bool(true) => Ok(1),
        JsonVal::Bool(false) => Ok(0),
        JsonVal::Num(raw) => {
            // Python int(float-token) truncates: int(20.9) == 20, int(1e3)==1000.
            // Try the integer grammar first so huge int tokens keep full precision.
            if let Some(i) = parse_py_int(raw.trim()) {
                return Ok(i);
            }
            let f: f64 = raw.parse().map_err(|_| BuilderError::Value(format!("invalid int: {raw}")))?;
            if !f.is_finite() {
                return Err(BuilderError::Value(format!("invalid int: {raw}")));
            }
            if f.abs() >= 9007199254740992.0 {
                // Python int(huge float) raises OverflowError (a ValueError subclass
                // only in spirit); surface as Value here.
                return Err(BuilderError::Value(format!("int too large: {raw}")));
            }
            Ok(f.trunc() as i64)
        }
        JsonVal::Str(s) => parse_py_int(s.trim())
            .ok_or_else(|| BuilderError::Value(format!("invalid int: {s:?}"))),
        _ => Err(BuilderError::Value("invalid int: non-scalar".to_string())),
    }
}

/// Python `int()` integer-token grammar: optional sign, digits with
/// single underscores between digit runs, arbitrary precision. Overflow
/// yields `None` (the caller surfaces `ValueError`, like Python's
/// `OverflowError`-on-huge-int path collapsing into the int-failure kind).
fn parse_py_int(t: &str) -> Option<i64> {
    let (neg, digits) = match t.strip_prefix(['+', '-']) {
        Some(rest) => (t.starts_with('-'), rest),
        None => (false, t),
    };
    if digits.is_empty() {
        return None;
    }
    let parts: Vec<&str> = digits.split('_').collect();
    if parts.iter().any(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let mut acc: i64 = 0;
    for b in parts.concat().bytes() {
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as i64)?;
    }
    Some(if neg { acc.checked_neg()? } else { acc })
}

/// Python `float(x)` for JSON values (`allow_leverage`).
fn py_float(value: &JsonVal) -> Result<f64, BuilderError> {
    match value {
        JsonVal::Bool(true) => Ok(1.0),
        JsonVal::Bool(false) => Ok(0.0),
        JsonVal::Num(raw) => raw.parse().map_err(|_| BuilderError::Value(format!("invalid float: {raw}"))),
        JsonVal::Str(s) => {
            let t: String = s.trim().chars().filter(|&c| c != '_').collect();
            let l = t.to_lowercase();
            if l == "inf" || l == "infinity" {
                return Ok(f64::INFINITY);
            }
            if l == "-inf" || l == "-infinity" {
                return Ok(f64::NEG_INFINITY);
            }
            if l == "+inf" || l == "+infinity" {
                return Ok(f64::INFINITY);
            }
            if l == "nan" || l == "+nan" || l == "-nan" {
                return Ok(f64::NAN);
            }
            t.parse().map_err(|_| BuilderError::Value(format!("invalid float: {s:?}")))
        }
        _ => Err(BuilderError::Value("invalid float: non-scalar".to_string())),
    }
}

/// Input shapes for [`parse_many`] mirroring `_parse_many`'s coercions:
/// `None` → `[]`, str → single, mapping → keys, else iterate / single.
pub enum ManyInput<'a> {
    Null,
    Single(&'a JsonVal),
    List(&'a [JsonVal]),
    MapKeys(Vec<JsonVal>),
}

/// Mirrors `_parse_many`: index references skipped, dedup by key, first wins.
pub fn parse_many(input: &ManyInput) -> Result<Vec<InstrumentSpec>, BuilderError> {
    let items: Vec<JsonVal> = match input {
        ManyInput::Null => Vec::new(),
        ManyInput::Single(v) => vec![(*v).clone()],
        ManyInput::List(xs) => xs.to_vec(),
        ManyInput::MapKeys(ks) => ks.clone(),
    };
    let mut unique: Vec<(String, InstrumentSpec)> = Vec::new();
    for value in &items {
        let raw = or_str(value, "");
        if is_index_reference(&raw) {
            continue;
        }
        let item = parse_instrument(&raw, "").map_err(BuilderError::Contract)?;
        if !unique.iter().any(|(k, _)| k == &item.key()) {
            unique.push((item.key(), item));
        }
    }
    Ok(unique.into_iter().map(|(_, v)| v).collect())
}

/// A schedule-callback argument: either a callable (with optional
/// `__name__`) or any other value (never callable, like Python).
pub enum CallbackArg<'a> {
    Callable { name: Option<&'a str> },
    Value(&'a JsonVal),
}

/// Mirrors `_schedule_callback`: kwarg `callback` wins unless a positional
/// callable exists (first positional callable wins over everything).
pub fn schedule_callback<'a>(
    args: &'a [CallbackArg<'a>],
    kw_callback: Option<&'a CallbackArg<'a>>,
    kw_time: &JsonVal,
) -> Result<(String, String), BuilderError> {
    let mut callback = kw_callback;
    for value in args {
        if matches!(value, CallbackArg::Callable { .. }) {
            callback = Some(value);
            break;
        }
    }
    let name = match callback {
        Some(CallbackArg::Callable { name }) => name.unwrap_or("scheduled").to_string(),
        _ => return Err(BuilderError::Contract("strategyV2.scheduleCallbackRequired".to_string())),
    };
    Ok((name, or_str(kw_time, "")))
}

/// Mirrors `DiscoveryContext` (value semantics; caller persists manifests).
#[derive(Debug, Clone, Default)]
pub struct DiscoveryContext {
    pub universe_reference: String,
    pub instruments: Vec<InstrumentSpec>,
    pub subscriptions: Vec<SubscriptionSpec>,
    pub schedules: Vec<ScheduleSpec>,
    pub benchmark: Option<InstrumentSpec>,
    pub warmup_bars: i64,
    pub leverage_allowed: bool,
    pub max_leverage: f64,
    pub metadata: Vec<(String, JsonVal)>,
}

impl DiscoveryContext {
    pub fn new() -> Self {
        Self { max_leverage: 1.0, ..Self::default() }
    }

    /// Mirrors `set_universe`. `values`/`index`/`pool`: `None` = arg absent
    /// (`pool=None` is *absent*; an explicit null `JsonVal` is present and —
    /// like Python's `""` — fails `normalize_pool_reference`).
    pub fn set_universe(
        &mut self,
        values: Option<&JsonVal>,
        index: Option<&JsonVal>,
        pool: Option<&JsonVal>,
    ) -> Result<(), BuilderError> {
        if let Some(p) = pool {
            self.universe_reference = normalize_pool_reference(&or_str(p, "")).map_err(BuilderError::Contract)?;
            self.instruments = Vec::new();
            return Ok(());
        }
        let source = index.or(values);
        let source = source.ok_or_else(|| BuilderError::Contract("strategyV2.universeRequired".to_string()))?;
        // _parse_many shapes: str checked directly; lists iterated; mappings
        // contribute keys; other scalars wrapped as singletons.
        match source {
            JsonVal::Str(raw) => {
                if is_index_reference(raw) {
                    self.universe_reference = normalize_index_reference(raw);
                    return Ok(());
                }
                self.instruments = parse_many(&ManyInput::Single(source))?;
                Ok(())
            }
            JsonVal::Null => Err(BuilderError::Contract("strategyV2.universeRequired".to_string())),
            JsonVal::Arr(xs) => {
                self.instruments = parse_many(&ManyInput::List(xs))?;
                Ok(())
            }
            JsonVal::Obj(kv) => {
                let keys = kv.iter().map(|(k, _)| JsonVal::Str(k.clone())).collect::<Vec<_>>();
                self.instruments = parse_many(&ManyInput::MapKeys(keys))?;
                Ok(())
            }
            _ => {
                let raw = or_str(source, "");
                if is_index_reference(&raw) {
                    self.universe_reference = normalize_index_reference(&raw);
                    return Ok(());
                }
                self.instruments = parse_many(&ManyInput::Single(source))?;
                Ok(())
            }
        }
    }

    pub fn set_benchmark(&mut self, value: &JsonVal) -> Result<(), BuilderError> {
        self.benchmark = Some(parse_instrument(&or_str(value, ""), "").map_err(BuilderError::Contract)?);
        Ok(())
    }

    /// Mirrors `subscribe`. `frequency`: `str(frequency or "1d")` — pass the
    /// raw JSON value (missing arg = `Null`, which falls back to `"1d"`).
    pub fn subscribe(
        &mut self,
        symbols: Option<&[JsonVal]>,
        frequency: &JsonVal,
        fields: Option<&[JsonVal]>,
    ) -> Result<(), BuilderError> {
        let instruments = match symbols {
            Some(xs) => parse_many(&ManyInput::List(xs))?,
            None => self.instruments.clone(),
        };
        let reference = if instruments.is_empty() { self.universe_reference.clone() } else { String::new() };
        let freq_input = or_str(frequency, "1d");
        let default_fields = ["open", "high", "low", "close", "volume"]
            .iter()
            .map(|s| JsonVal::Str(s.to_string()))
            .collect::<Vec<_>>();
        let field_vals = fields.unwrap_or(&default_fields);
        let norm_fields = field_vals.iter().map(|f| or_str(f, "").trim().to_lowercase()).collect();
        self.subscriptions.push(SubscriptionSpec {
            instruments,
            universe_reference: reference,
            frequency: normalize_frequency(&freq_input, "1d"),
            fields: norm_fields,
        });
        Ok(())
    }

    /// Mirrors `set_warmup`: `max(0, int(bars or 0))`.
    pub fn set_warmup(&mut self, bars: &JsonVal) -> Result<(), BuilderError> {
        let n = if is_falsy(bars) { 0 } else { py_int(bars)? };
        self.warmup_bars = n.max(0);
        Ok(())
    }

    /// Mirrors `allow_leverage`: `value = max(1.0, float(x or 1.0))`.
    pub fn allow_leverage(&mut self, max_leverage: &JsonVal) -> Result<(), BuilderError> {
        let value = if is_falsy(max_leverage) { 1.0 } else { py_float(max_leverage)? };
        let value = value.max(1.0);
        // Python: max(1.0, nan) → nan (max compares, nan loses); then
        // nan > 1.0 is False → allowed False, max_leverage nan.
        self.leverage_allowed = value > 1.0;
        self.max_leverage = value;
        Ok(())
    }

    /// Mirrors `set_metadata` positional shapes; kwargs merged afterwards.
    pub fn set_metadata(
        &mut self,
        args: &[JsonVal],
        kwargs: &[(String, JsonVal)],
    ) -> Result<(), BuilderError> {
        if args.len() == 1 {
            match &args[0] {
                JsonVal::Obj(kv) => {
                    for (k, v) in kv {
                        upsert(&mut self.metadata, k.clone(), v.clone());
                    }
                }
                _ => {
                    return Err(BuilderError::Type(
                        "set_metadata expects keyword arguments, a single key/value pair, or a mapping".to_string(),
                    ));
                }
            }
        } else if args.len() == 2 {
            upsert(&mut self.metadata, or_str(&args[0], ""), args[1].clone());
        } else if !args.is_empty() {
            return Err(BuilderError::Type(
                "set_metadata expects keyword arguments, a single key/value pair, or a mapping".to_string(),
            ));
        }
        for (k, v) in kwargs {
            upsert(&mut self.metadata, k.clone(), v.clone());
        }
        Ok(())
    }

    pub fn schedule_daily(
        &mut self,
        args: &[CallbackArg],
        kw_callback: Option<&CallbackArg>,
        kw_time: &JsonVal,
    ) -> Result<(), BuilderError> {
        let (callback, time) = schedule_callback(args, kw_callback, kw_time)?;
        self.schedules.push(ScheduleSpec {
            frequency: "daily".to_string(),
            callback,
            time,
            weekday: None,
            monthday: None,
        });
        Ok(())
    }

    pub fn schedule_weekly(
        &mut self,
        args: &[CallbackArg],
        kwargs: &[(String, JsonVal)],
        kw_callback: Option<&CallbackArg>,
    ) -> Result<(), BuilderError> {
        let kw_time = find_kw(kwargs, "time").cloned().unwrap_or(JsonVal::Null);
        let (callback, time) = schedule_callback(args, kw_callback, &kw_time)?;
        let weekday = match find_kw(kwargs, "weekday") {
            None => 1,
            Some(v) => py_int(v)?,
        };
        self.schedules.push(ScheduleSpec {
            frequency: "weekly".to_string(),
            callback,
            time,
            weekday: Some(weekday),
            monthday: None,
        });
        Ok(())
    }

    pub fn schedule_monthly(
        &mut self,
        args: &[CallbackArg],
        kwargs: &[(String, JsonVal)],
        kw_callback: Option<&CallbackArg>,
    ) -> Result<(), BuilderError> {
        let kw_time = find_kw(kwargs, "time").cloned().unwrap_or(JsonVal::Null);
        let (callback, time) = schedule_callback(args, kw_callback, &kw_time)?;
        let monthday = match find_kw(kwargs, "monthday") {
            None => 1,
            Some(v) => py_int(v)?,
        };
        self.schedules.push(ScheduleSpec {
            frequency: "monthly".to_string(),
            callback,
            time,
            weekday: None,
            monthday: Some(monthday),
        });
        Ok(())
    }
}

fn upsert(meta: &mut Vec<(String, JsonVal)>, key: String, value: JsonVal) {
    if let Some(slot) = meta.iter_mut().find(|(k, _)| k == &key) {
        slot.1 = value;
    } else {
        meta.push((key, value));
    }
}

fn find_kw<'a>(kwargs: &'a [(String, JsonVal)], name: &str) -> Option<&'a JsonVal> {
    kwargs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> JsonVal {
        JsonVal::Str(v.to_string())
    }

    #[test]
    fn py_int_covers_grammar() {
        assert_eq!(py_int(&s("3")), Ok(3));
        assert_eq!(py_int(&s("  -12 ")), Ok(-12));
        assert_eq!(py_int(&s("+7")), Ok(7));
        assert_eq!(py_int(&s("1_0")), Ok(10));
        assert_eq!(py_int(&s("007")), Ok(7));
        assert_eq!(py_int(&s("-0")), Ok(0));
        assert!(py_int(&s("")).is_err());
        assert!(py_int(&s("3.5")).is_err());
        assert!(py_int(&s("1__0")).is_err());
        assert!(py_int(&s("abc")).is_err());
        assert!(py_int(&s("inf")).is_err());
        assert_eq!(py_int(&JsonVal::Bool(true)), Ok(1));
        assert_eq!(py_int(&JsonVal::Num("3.9".into())), Ok(3));
        assert_eq!(py_int(&JsonVal::Num("-3.9".into())), Ok(-3));
        assert!(py_int(&JsonVal::Num("1e400".into())).is_err());
        assert!(py_int(&JsonVal::Null).is_err());
    }

    #[test]
    fn py_float_covers_grammar() {
        assert_eq!(py_float(&s("2.5")), Ok(2.5));
        assert_eq!(py_float(&s("1_0")), Ok(10.0));
        assert_eq!(py_float(&s("inf")), Ok(f64::INFINITY));
        assert_eq!(py_float(&s("-INF")), Ok(f64::NEG_INFINITY));
        assert!(py_float(&s("nan")).unwrap().is_nan());
        assert!(py_float(&s("")).is_err());
        assert!(py_float(&s("x")).is_err());
        assert_eq!(py_float(&JsonVal::Bool(false)), Ok(0.0));
    }

    #[test]
    fn universe_branches() {
        let mut c = DiscoveryContext::new();
        assert_eq!(
            c.set_universe(None, None, None).unwrap_err(),
            BuilderError::Contract("strategyV2.universeRequired".to_string())
        );
        c.set_universe(None, None, Some(&s("my-pool"))).unwrap();
        assert_eq!(c.universe_reference, "POOL:my-pool");
        assert!(c.instruments.is_empty());
        // explicit null pool is *present* → normalize fails like Python
        assert!(c.set_universe(None, None, Some(&JsonVal::Null)).is_err());
        c.set_universe(None, Some(&s("INDEX:HS300")), None).unwrap();
        assert_eq!(c.universe_reference, "INDEX:HS300");
        c.set_universe(Some(&JsonVal::Arr(vec![s("AAPL"), s("AAPL"), s("INDEX:HS300")])), None, None).unwrap();
        assert_eq!(c.instruments.len(), 1);
        assert_eq!(c.instruments[0].symbol, "AAPL");
    }

    #[test]
    fn subscribe_defaults_and_reference() {
        let mut c = DiscoveryContext::new();
        c.set_universe(Some(&JsonVal::Arr(vec![s("AAPL")])), None, None).unwrap();
        c.subscribe(None, &JsonVal::Null, None).unwrap();
        assert_eq!(c.subscriptions[0].frequency, "1d");
        assert_eq!(c.subscriptions[0].universe_reference, "");
        assert_eq!(c.subscriptions[0].fields, vec!["open", "high", "low", "close", "volume"]);
        let mut c2 = DiscoveryContext::new();
        c2.universe_reference = "POOL:x".to_string();
        c2.subscribe(None, &s("1H"), Some(&[JsonVal::Bool(true), JsonVal::Num("5".into())])).unwrap();
        assert_eq!(c2.subscriptions[0].frequency, "1h");
        assert_eq!(c2.subscriptions[0].universe_reference, "POOL:x");
        assert_eq!(c2.subscriptions[0].fields, vec!["true", "5"]);
    }

    #[test]
    fn warmup_leverage_metadata() {
        let mut c = DiscoveryContext::new();
        c.set_warmup(&JsonVal::Null).unwrap();
        assert_eq!(c.warmup_bars, 0);
        c.set_warmup(&s("-5")).unwrap();
        assert_eq!(c.warmup_bars, 0);
        c.set_warmup(&s("20")).unwrap();
        assert_eq!(c.warmup_bars, 20);
        assert!(c.set_warmup(&s("x")).is_err());
        c.allow_leverage(&JsonVal::Null).unwrap();
        assert!(!c.leverage_allowed);
        assert_eq!(c.max_leverage, 1.0);
        c.allow_leverage(&s("3")).unwrap();
        assert!(c.leverage_allowed);
        // max(1.0, nan) → 1.0 on both sides (nan loses the comparison)
        c.allow_leverage(&s("nan")).unwrap();
        assert!(!c.leverage_allowed);
        assert_eq!(c.max_leverage, 1.0);
        c.set_metadata(&[JsonVal::Obj(vec![("a".to_string(), s("1"))])], &[("b".to_string(), s("2"))]).unwrap();
        c.set_metadata(&[s("k"), s("v")], &[]).unwrap();
        assert_eq!(c.metadata.len(), 3);
        assert!(c.set_metadata(&[s("only")], &[]).is_err());
        assert!(c.set_metadata(&[s("a"), s("b"), s("c")], &[]).is_err());
    }

    #[test]
    fn schedules_and_callback_rules() {
        let mut c = DiscoveryContext::new();
        let cb = CallbackArg::Callable { name: Some("on_open") };
        c.schedule_daily(&[cb], None, &s("09:30")).unwrap();
        assert_eq!(c.schedules[0].callback, "on_open");
        assert_eq!(c.schedules[0].time, "09:30");
        // positional callable beats kw callback
        let kw = CallbackArg::Callable { name: Some("kw_cb") };
        let pos = CallbackArg::Callable { name: None };
        c.schedule_weekly(&[pos], &[("weekday".to_string(), s("3"))], Some(&kw)).unwrap();
        assert_eq!(c.schedules[1].callback, "scheduled");
        assert_eq!(c.schedules[1].weekday, Some(3));
        // no callable anywhere → Contract error
        let v = CallbackArg::Value(&JsonVal::Num("1".into()));
        assert_eq!(
            c.schedule_monthly(&[v], &[], None).unwrap_err(),
            BuilderError::Contract("strategyV2.scheduleCallbackRequired".to_string())
        );
        c.schedule_monthly(&[], &[], Some(&kw)).unwrap();
        assert_eq!(c.schedules[2].monthday, Some(1));
    }

    #[test]
    fn hash_and_detect() {
        // surrounding whitespace ignored (strip) — same source, same hash
        assert_eq!(source_code_hash(None), source_code_hash(Some("  ")));
        assert_eq!(source_code_hash(Some("x")), source_code_hash(Some(" x ")));
        assert_eq!(source_code_hash(Some("x")).len(), 64);
        assert!(is_strategy_v2_code(Some("def initialize(context):\n context.set_universe(['A'])\n run_daily(f)")));
        assert!(!is_strategy_v2_code(Some("def initialize(context):\n pass")));
        assert!(!is_strategy_v2_code(None));
    }
}
