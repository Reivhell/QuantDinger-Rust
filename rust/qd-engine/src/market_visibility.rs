//! Port of `backend_api_python/app/utils/market_visibility.py`.
//!
//! Operator-controlled market visibility. Resolution order (first match wins):
//! 1. `ENABLED_MARKETS` CSV whitelist — when non-empty ONLY listed markets
//!    are visible; overrides the legacy flags completely.
//! 2. `SHOW_CN_STOCK` (legacy, default false).
//! 3. `SHOW_HK_STOCK` (legacy, default true).
//! 4. Everything else defaults to visible.
//!
//! The environment is read through [`Env`] so the logic stays pure and
//! testable; [`Env::live`] reads the real process environment, exactly like
//! Python's `os.getenv`.

use std::collections::{HashMap, HashSet};

/// Known markets. Mirrors `_KNOWN_MARKETS`.
pub const KNOWN_MARKETS: &[&str] =
    &["Crypto", "USStock", "CNStock", "HKStock", "Forex", "Futures", "MOEX"];

/// Environment snapshot. `None` value = variable unset
/// (mirrors `os.getenv(name)` returning `None`).
#[derive(Debug, Clone, Default)]
pub struct Env {
    vars: HashMap<String, String>,
}

impl Env {
    /// Read the three variables Python touches from the live environment.
    pub fn live() -> Self {
        let mut vars = HashMap::new();
        for name in ["ENABLED_MARKETS", "SHOW_CN_STOCK", "SHOW_HK_STOCK"] {
            if let Ok(v) = std::env::var(name) {
                vars.insert(name.to_string(), v);
            }
        }
        Self { vars }
    }

    pub fn with(mut self, name: &str, value: &str) -> Self {
        self.vars.insert(name.to_string(), value.to_string());
        self
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(|s| s.as_str())
    }
}

/// Mirrors `_flag`: `str(getenv(name, default)).strip().lower()` in
/// `{'1','true','yes','on'}`. Note Python stringifies the default, so an
/// unset var falls back to parsing `default`.
pub fn flag(env: &Env, name: &str, default: &str) -> bool {
    let raw = env.get(name).unwrap_or(default);
    matches!(raw.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

/// Mirrors `_parse_csv`: empty/missing → empty set, else trimmed non-empty parts.
pub fn parse_csv(env: &Env, name: &str) -> HashSet<String> {
    match env.get(name).map(str::trim) {
        None | Some("") => HashSet::new(),
        Some(raw) => raw.split(',').map(str::trim).filter(|p| !p.is_empty()).map(str::to_string).collect(),
    }
}

/// Mirrors `enabled_markets_whitelist`.
pub fn enabled_markets_whitelist(env: &Env) -> HashSet<String> {
    parse_csv(env, "ENABLED_MARKETS")
}

/// Mirrors `is_market_visible`.
pub fn is_market_visible(env: &Env, market: &str) -> bool {
    let m = market.trim();
    if m.is_empty() {
        return false;
    }
    let whitelist = enabled_markets_whitelist(env);
    if !whitelist.is_empty() {
        return whitelist.contains(m);
    }
    if m == "CNStock" {
        return flag(env, "SHOW_CN_STOCK", "false");
    }
    if m == "HKStock" {
        return flag(env, "SHOW_HK_STOCK", "true");
    }
    true
}

/// One filterable item: either a bare market string or a dict-like map.
/// Mirrors the `isinstance` branches of `filter_market_items`.
#[derive(Debug, Clone, PartialEq)]
pub enum MarketItem {
    Str(String),
    Map(HashMap<String, String>),
}

impl MarketItem {
    fn market_of(&self, key: &str) -> Option<String> {
        match self {
            MarketItem::Str(s) => Some(s.trim().to_string()),
            MarketItem::Map(m) => m.get(key).map(|v| v.trim().to_string()),
        }
    }
}

/// Mirrors `filter_market_items`: falsy/unknown/hidden markets dropped,
/// survivors keep relative order. Non-string items (`else: continue`) have
/// no representation here — the caller simply omits them.
pub fn filter_market_items(
    env: &Env,
    items: &[MarketItem],
    key: &str,
) -> Vec<MarketItem> {
    items
        .iter()
        .filter(|it| match it.market_of(key) {
            Some(mk) if !mk.is_empty() => is_market_visible(env, &mk),
            _ => false,
        })
        .cloned()
        .collect()
}

/// Mirrors `hidden_markets`.
pub fn hidden_markets(env: &Env) -> HashSet<String> {
    KNOWN_MARKETS
        .iter()
        .filter(|m| !is_market_visible(env, m))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        Env::default()
    }

    #[test]
    fn defaults_hide_cn_but_show_hk() {
        let e = env();
        assert!(!is_market_visible(&e, "CNStock"));
        assert!(is_market_visible(&e, "HKStock"));
        assert!(is_market_visible(&e, "Crypto"));
        assert!(!is_market_visible(&e, ""));
        assert!(!is_market_visible(&e, "   "));
    }

    #[test]
    fn legacy_flags_toggle() {
        let e = env().with("SHOW_CN_STOCK", "yes").with("SHOW_HK_STOCK", "0");
        assert!(is_market_visible(&e, "CNStock"));
        assert!(!is_market_visible(&e, "HKStock"));
        // case-insensitive, whitespace-tolerant like Python
        let e2 = env().with("SHOW_CN_STOCK", "  True ");
        assert!(is_market_visible(&e2, "CNStock"));
    }

    #[test]
    fn whitelist_overrides_everything() {
        let e = env()
            .with("ENABLED_MARKETS", "Crypto, USStock")
            .with("SHOW_CN_STOCK", "true");
        assert!(is_market_visible(&e, "Crypto"));
        assert!(!is_market_visible(&e, "CNStock")); // flag ignored
        assert!(!is_market_visible(&e, "HKStock")); // default ignored
        assert!(!is_market_visible(&e, "Unknown")); // must be listed
    }

    #[test]
    fn whitelist_parsing_ignores_empties() {
        let e = env().with("ENABLED_MARKETS", " Crypto ,,USStock, ");
        let w = enabled_markets_whitelist(&e);
        assert_eq!(w, ["Crypto".to_string(), "USStock".to_string()].into_iter().collect());
    }

    #[test]
    fn filter_keeps_order_and_drops_hidden() {
        let e = env(); // CN hidden by default
        let items = vec![
            MarketItem::Str("Crypto".into()),
            MarketItem::Str("CNStock".into()),
            MarketItem::Str("  ".into()),
            MarketItem::Map([("value".into(), "USStock".into())].into_iter().collect()),
            MarketItem::Map([("value".into(), "CNStock".into())].into_iter().collect()),
            MarketItem::Map([("other".into(), "Crypto".into())].into_iter().collect()),
        ];
        let out = filter_market_items(&e, &items, "value");
        assert_eq!(out, vec![items[0].clone(), items[3].clone()]);
    }

    #[test]
    fn hidden_lists_exactly_cn_by_default() {
        assert_eq!(hidden_markets(&env()), ["CNStock".to_string()].into_iter().collect());
        let e = env().with("ENABLED_MARKETS", "Crypto");
        let h = hidden_markets(&e);
        assert_eq!(h.len(), KNOWN_MARKETS.len() - 1);
        assert!(!h.contains("Crypto"));
    }
}
