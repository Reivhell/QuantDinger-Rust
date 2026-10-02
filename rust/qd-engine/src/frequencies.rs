//! Port of `backend_api_python/app/services/strategy_v2/frequencies.py`.
//!
//! Canonical Strategy API V2 frequency helpers: alias normalisation,
//! per-bar seconds, frequency-set helpers, and annualisation periods.
//!
//! Faithful corners:
//! - Empty/whitespace input falls back to `default` (mirrors `value or default`
//!   followed by `raw or default`).
//! - Chinese `"分钟"`/`"小时"` suffixes are rewritten before alias lookup.
//! - `periods_per_year` replicates `int(prefix or 1)` via float digit parsing
//!   (no overflow on absurd inputs) and returns `Err` where Python raises
//!   `ValueError` (e.g. `"1.5h"`).
//! - `"Crypto"` membership is an exact, case-sensitive element match.

/// Canonical bar seconds, mirroring `FREQUENCY_SECONDS`.
pub const FREQUENCY_SECONDS: &[(&str, i64)] = &[
    ("1m", 60),
    ("3m", 180),
    ("5m", 300),
    ("15m", 900),
    ("30m", 1800),
    ("1h", 3600),
    ("4h", 14_400),
    ("1d", 86_400),
    ("1w", 604_800),
];

pub const DEFAULT_FREQUENCY: &str = "1d";
pub const DEFAULT_FREQUENCY_SECONDS: i64 = 86_400;

fn alias(raw: &str) -> Option<&'static str> {
    Some(match raw {
        "daily" | "day" | "d" | "1day" => "1d",
        "weekly" | "week" | "w" => "1w",
        "monthly" | "month" => "1mo",
        "m1" => "1m",
        "h1" => "1h",
        "d1" => "1d",
        _ => return None,
    })
}

/// Normalise a frequency label. Mirrors `normalize_frequency`.
pub fn normalize_frequency(value: &str, default: &str) -> String {
    let base = if value.trim().is_empty() { default } else { value };
    let raw = base.trim().to_lowercase().replace("分钟", "m").replace("小时", "h");
    if let Some(a) = alias(&raw) {
        return a.to_string();
    }
    if raw.is_empty() {
        default.to_string()
    } else {
        raw
    }
}

/// Bar seconds for a label, defaulting to one day. Mirrors `frequency_seconds`.
pub fn frequency_seconds(value: &str) -> i64 {
    let norm = normalize_frequency(value, DEFAULT_FREQUENCY);
    FREQUENCY_SECONDS
        .iter()
        .find(|(k, _)| *k == norm)
        .map(|(_, s)| *s)
        .unwrap_or(DEFAULT_FREQUENCY_SECONDS)
}

/// Deduped normalised labels, order-preserving. Mirrors `unique_frequencies`.
pub fn unique_frequencies(values: &[&str], default: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for v in values {
        let n = normalize_frequency(v, default);
        if !out.contains(&n) {
            out.push(n);
        }
    }
    if out.is_empty() {
        out.push(normalize_frequency(default, default));
    }
    out
}

/// Fastest (finest) frequency, first occurrence winning ties.
/// Mirrors `driving_frequency`.
pub fn driving_frequency(values: &[&str], default: &str) -> String {
    let freqs = unique_frequencies(values, default);
    let mut best = &freqs[0];
    let mut best_key = (frequency_seconds(best), 0usize);
    for (i, f) in freqs.iter().enumerate().skip(1) {
        let key = (frequency_seconds(f), i);
        if key < best_key {
            best = f;
            best_key = key;
        }
    }
    best.clone()
}

/// Parse Python `int()` semantics for the numeric prefix: optional sign,
/// ASCII digits with single internal underscores. Returns the value as `f64`
/// so absurd magnitudes cannot overflow.
fn parse_int_prefix(s: &str) -> Option<f64> {
    let t = s.trim();
    let (negative, digits) = match t.strip_prefix(['+', '-']) {
        Some(rest) => (t.starts_with('-'), rest),
        None => (false, t),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value: f64 = 0.0;
    let mut prev_underscore = true; // leading '_' invalid
    for c in digits.chars() {
        if c == '_' {
            if prev_underscore {
                return None;
            }
            prev_underscore = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return None;
        }
        prev_underscore = false;
        value = value * 10.0 + (c as u8 - b'0') as f64;
    }
    if prev_underscore {
        return None;
    }
    Some(if negative { -value } else { value })
}

/// Annualisation periods. Mirrors `periods_per_year`; `Err` where Python
/// raises `ValueError` on a non-integer prefix.
pub fn periods_per_year(frequency: &str, markets: &[&str]) -> Result<f64, &'static str> {
    let normalized = normalize_frequency(frequency, DEFAULT_FREQUENCY);
    let is_crypto = markets.iter().any(|m| *m == "Crypto");
    let trading_days = if is_crypto { 365.25 } else { 252.0 };
    if let Some(prefix) = normalized.strip_suffix('m') {
        let minutes = if prefix.is_empty() {
            1.0
        } else {
            parse_int_prefix(prefix).ok_or("invalid minute frequency")?
        };
        let minutes = minutes.max(1.0);
        let session_minutes = if is_crypto { 1440.0 } else { 390.0 };
        return Ok(trading_days * session_minutes / minutes);
    }
    if let Some(prefix) = normalized.strip_suffix('h') {
        let hours = if prefix.is_empty() {
            1.0
        } else {
            parse_int_prefix(prefix).ok_or("invalid hourly frequency")?
        };
        let hours = hours.max(1.0);
        let session_hours = if is_crypto { 24.0 } else { 6.5 };
        return Ok(trading_days * session_hours / hours);
    }
    if normalized.ends_with('w') {
        return Ok(52.0);
    }
    Ok(trading_days)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_and_aliases() {
        assert_eq!(frequency_seconds("1m"), 60);
        assert_eq!(frequency_seconds("4h"), 14_400);
        assert_eq!(frequency_seconds("1w"), 604_800);
        assert_eq!(normalize_frequency("daily", "1d"), "1d");
        assert_eq!(normalize_frequency("m1", "1d"), "1m");
        assert_eq!(normalize_frequency(" 4H ", "1d"), "4h");
        assert_eq!(normalize_frequency("15分钟", "1d"), "15m");
        assert_eq!(normalize_frequency("2小时", "1d"), "2h");
        assert_eq!(normalize_frequency("", "4h"), "4h");
        assert_eq!(normalize_frequency("tick", "1d"), "tick");
        assert_eq!(frequency_seconds("tick"), 86_400);
    }

    #[test]
    fn unique_and_driving() {
        assert_eq!(
            unique_frequencies(&["1d", "4h", "1d", "1h"], "1d"),
            vec!["1d", "4h", "1h"]
        );
        assert_eq!(unique_frequencies(&[], "1d"), vec!["1d"]);
        assert_eq!(driving_frequency(&["1d", "4h", "1h"], "1d"), "1h");
        // unknown labels fall back to 1d seconds; first occurrence wins ties
        assert_eq!(driving_frequency(&["tick", "tock"], "1d"), "tick");
        assert_eq!(driving_frequency(&[], "4h"), "4h");
    }

    #[test]
    fn periods() {
        // crypto trades 365.25 x 24h sessions
        assert!((periods_per_year("1h", &["Crypto"]).unwrap() - 365.25 * 24.0).abs() < 1e-9);
        assert!((periods_per_year("4h", &["Crypto"]).unwrap() - 365.25 * 6.0).abs() < 1e-9);
        assert!((periods_per_year("15m", &["Crypto"]).unwrap() - 365.25 * 96.0).abs() < 1e-9);
        // equities: 252 days x 6.5h / 390min sessions
        assert!((periods_per_year("1h", &["Stocks"]).unwrap() - 252.0 * 6.5).abs() < 1e-9);
        assert!((periods_per_year("1d", &["Stocks"]).unwrap() - 252.0).abs() < 1e-9);
        assert_eq!(periods_per_year("1w", &[]).unwrap(), 52.0);
        assert_eq!(periods_per_year("1mo", &["Crypto"]).unwrap(), 365.25);
        // bare suffix means one unit
        assert!((periods_per_year("m", &["Crypto"]).unwrap() - 365.25 * 1440.0).abs() < 1e-9);
        // fractional prefix raises in Python -> Err here
        assert!(periods_per_year("1.5h", &["Stocks"]).is_err());
        // market tag must be exactly "Crypto"
        assert!((periods_per_year("1d", &["Crypto:BTC/USDT"]).unwrap() - 252.0).abs() < 1e-9);
    }
}
