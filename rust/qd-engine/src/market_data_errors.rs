//! Port of the pure slice of `backend_api_python/app/data_sources/errors.py`:
//! [`MarketDataFailure`], its mapping round-trip, and
//! `classify_market_data_failure`.
//!
//! Exception classes (`DataSourceError`, `MarketDataUnavailableError`,
//! `UnsupportedMarketError`) stay Python-side — they are control flow, not
//! computation.
//!
//! Faithful corners:
//! - `technical_detail` truncated to 500 *chars* (Unicode scalar count, like
//!   Python slicing), both in `from_mapping` and `classify`.
//! - Credential scrub `(?i)(https?://)([^/@\s:]+):([^/@\s]+)@` → `\1***:***@`
//!   re-implemented by hand (no regex crate; std only), applied globally.
//! - Classification reads `detail.lower()` and matches literal substrings,
//!   branch order preserved.
//! - `exchange_id`/`market_type` are stripped + lowercased; `symbol` and
//!   `timeframe` are only stringified.

use crate::json_helpers::JsonVal;

/// Mirrors `truthiness` of `bool(value.get("retryable", True))` for JSON values.
pub fn is_truthy(v: Option<&JsonVal>) -> bool {
    match v {
        None => true, // key missing → default True
        Some(JsonVal::Null) => false,
        Some(JsonVal::Bool(b)) => *b,
        Some(JsonVal::Num(raw)) => {
            // Python: numeric 0/0.0 false, else true; unparsable → true (non-empty str)
            match raw.parse::<f64>() {
                Ok(f) => f != 0.0,
                Err(_) => true,
            }
        }
        Some(JsonVal::Str(s)) => !s.is_empty(),
        Some(JsonVal::Arr(xs)) => !xs.is_empty(),
        Some(JsonVal::Obj(kv)) => !kv.is_empty(),
    }
}

/// Mirrors `MarketDataFailure`.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketDataFailure {
    pub code: String,
    pub message: String,
    pub technical_detail: String,
    pub exchange_id: String,
    pub market_type: String,
    pub symbol: String,
    pub timeframe: String,
    pub retryable: bool,
}

impl MarketDataFailure {
    pub fn as_dict(&self) -> Vec<(String, JsonVal)> {
        vec![
            ("code".to_string(), JsonVal::Str(self.code.clone())),
            ("message".to_string(), JsonVal::Str(self.message.clone())),
            ("technical_detail".to_string(), JsonVal::Str(self.technical_detail.clone())),
            ("exchange_id".to_string(), JsonVal::Str(self.exchange_id.clone())),
            ("market_type".to_string(), JsonVal::Str(self.market_type.clone())),
            ("symbol".to_string(), JsonVal::Str(self.symbol.clone())),
            ("timeframe".to_string(), JsonVal::Str(self.timeframe.clone())),
            ("retryable".to_string(), JsonVal::Bool(self.retryable)),
        ]
    }

    /// Mirrors `from_mapping`: `str(x or default)` + `[:500]` on detail.
    pub fn from_mapping(fields: &[(String, JsonVal)]) -> Self {
        let get = |k: &str| fields.iter().find(|(kk, _)| kk == k).map(|(_, v)| v);
        let s = |k: &str, dflt: &str| match get(k) {
            Some(JsonVal::Str(s)) if !s.is_empty() => s.clone(),
            Some(JsonVal::Num(raw)) if !raw.is_empty() => raw.clone(),
            Some(JsonVal::Bool(true)) => "True".to_string(),
            Some(JsonVal::Bool(false)) => String::new().into(),
            _ => dflt.to_string(),
        };
        // Python: str(value.get(k) or "") — falsy (None/""/0/False/[]) → "".
        let s_or_empty = |k: &str| match get(k) {
            Some(JsonVal::Str(s)) => {
                if s.is_empty() {
                    String::new()
                } else {
                    s.clone()
                }
            }
            Some(JsonVal::Num(raw)) => {
                if raw.parse::<f64>().map(|f| f == 0.0).unwrap_or(false) {
                    String::new()
                } else {
                    raw.clone()
                }
            }
            Some(JsonVal::Bool(true)) => "True".to_string(),
            _ => String::new(),
        };
        let _ = s;
        Self {
            code: or_default(get("code"), "no_market_data"),
            message: or_default(get("message"), "No usable market data is available."),
            technical_detail: truncate_chars(&s_or_empty("technical_detail"), 500),
            exchange_id: s_or_empty("exchange_id"),
            market_type: s_or_empty("market_type"),
            symbol: s_or_empty("symbol"),
            timeframe: s_or_empty("timeframe"),
            retryable: is_truthy(get("retryable")),
        }
    }
}

/// `str(v or default)`: falsy JSON values fall back to the default.
fn or_default(v: Option<&JsonVal>, dflt: &str) -> String {
    match v {
        Some(JsonVal::Str(s)) if !s.is_empty() => s.clone(),
        Some(JsonVal::Num(raw)) => {
            if raw.parse::<f64>().map(|f| f == 0.0).unwrap_or(false) || raw.is_empty() {
                dflt.to_string()
            } else {
                raw.clone()
            }
        }
        Some(JsonVal::Bool(true)) => "True".to_string(),
        // False / Null / empty containers / missing → default
        _ => dflt.to_string(),
    }
}

/// First `n` Unicode scalar values, mirroring Python `s[:n]`.
pub fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0C)
}

/// Case-insensitive match of `http://` / `https://` at `b[pos..]`;
/// returns the scheme length (7 or 8) on match.
fn scheme_len(b: &[u8], pos: usize) -> Option<usize> {
    let rest = &b[pos..];
    let lower = |i: usize| rest.get(i).map(|c| c.to_ascii_lowercase());
    if lower(0) == Some(b'h') && lower(1) == Some(b't') && lower(2) == Some(b't') && lower(3) == Some(b'p') {
        if lower(4) == Some(b's') {
            if rest.get(5) == Some(&b':') && rest.get(6) == Some(&b'/') && rest.get(7) == Some(&b'/') {
                return Some(8);
            }
        } else if rest.get(4) == Some(&b':') && rest.get(5) == Some(&b'/') && rest.get(6) == Some(&b'/') {
            return Some(7);
        }
    }
    None
}

/// Mirrors `re.sub(r"(?i)(https?://)([^/@\s:]+):([^/@\s]+)@", r"\1***:***@", detail)`.
pub fn scrub_credentials(detail: &str) -> String {
    let b = detail.as_bytes();
    let mut out = String::with_capacity(detail.len());
    let mut i = 0;
    let mut last = 0;
    while i < b.len() {
        if let Some(sl) = scheme_len(b, i) {
            let mut j = i + sl;
            // user = [^/@\s:]+
            while j < b.len() && b[j] != b'/' && b[j] != b'@' && b[j] != b':' && !is_ws(b[j]) {
                j += 1;
            }
            if j > i + sl && b.get(j) == Some(&b':') {
                // pass = [^/@\s]+ then '@'
                let mut k = j + 1;
                while k < b.len() && b[k] != b'/' && b[k] != b'@' && !is_ws(b[k]) {
                    k += 1;
                }
                if k > j + 1 && b.get(k) == Some(&b'@') {
                    out.push_str(&detail[last..i + sl]);
                    out.push_str("***:***@");
                    i = k + 1;
                    last = i;
                    continue;
                }
            }
        }
        i += 1;
    }
    out.push_str(&detail[last..]);
    out
}

const REGION: &[&str] = &[
    "451",
    "restricted location",
    "legal reasons",
    "region restricted",
    "block access from your country",
    "blocked access from your country",
];
const PROXY: &[&str] = &["proxyerror", "proxy error", "proxyconnect", "proxy connection", "tunnel connection", "socks"];
const SYMBOL: &[&str] = &[
    "does not have market symbol",
    "symbol not found",
    "invalid symbol",
    "market does not exist",
    "trading pair not found",
];
const RATE: &[&str] = &["429", "too many requests", "rate limit", "ratelimit"];
const INCOMPLETE: &[&str] = &[
    "incomplete k-line",
    "incomplete kline",
    "incomplete candle",
    "incomplete market data",
    "incomplete history",
];
const UNAVAIL: &[&str] = &[
    "timeout",
    "timed out",
    "network error",
    "connection reset",
    "connection refused",
    "exchange not available",
    "service unavailable",
    "502",
    "503",
    "504",
];
const TF_BAD: &[&str] = &["unsupported", "not support", "cannot serve"];

fn contains_any(text: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|t| text.contains(t))
}

/// Mirrors `classify_market_data_failure`. `error_text` is the already
/// stringified `str(error or "")` (stringification is caller-side).
pub fn classify_market_data_failure(
    error_text: &str,
    exchange_id: &str,
    market_type: &str,
    symbol: &str,
    timeframe: &str,
) -> MarketDataFailure {
    let detail = scrub_credentials(error_text.trim());
    let text = detail.to_lowercase();
    let (code, message, retryable) = if contains_any(&text, REGION) {
        ("region_restricted", "The exchange market-data endpoint is unavailable in this region.", false)
    } else if contains_any(&text, PROXY) {
        ("proxy_failure", "The market-data proxy could not connect to the exchange.", true)
    } else if contains_any(&text, SYMBOL) {
        ("symbol_not_found", "The trading pair does not exist for this exchange and market type.", false)
    } else if contains_any(&text, RATE) {
        ("rate_limited", "The exchange rate limit was reached. Market data will be retried.", true)
    } else if contains_any(&text, INCOMPLETE) {
        (
            "incomplete_market_data",
            "The exchange returned incomplete K-line coverage. The missing interval will be retried.",
            true,
        )
    } else if contains_any(&text, UNAVAIL) {
        ("exchange_unavailable", "The exchange market-data service is temporarily unreachable.", true)
    } else if text.contains("timeframe") && contains_any(&text, TF_BAD) {
        ("unsupported_timeframe", "This exchange does not provide the requested K-line timeframe.", false)
    } else {
        ("no_market_data", "The exchange returned no usable market data.", true)
    };
    MarketDataFailure {
        code: code.to_string(),
        message: message.to_string(),
        technical_detail: truncate_chars(&detail, 500),
        exchange_id: exchange_id.trim().to_lowercase(),
        market_type: market_type.trim().to_lowercase(),
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        retryable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_branch_classifies() {
        let cases: &[(&str, &str, bool)] = &[
            ("451 restricted location", "region_restricted", false),
            ("ProxyError: tunnel connection failed", "proxy_failure", true),
            ("symbol not found on market", "symbol_not_found", false),
            ("429 too many requests", "rate_limited", true),
            ("incomplete kline coverage", "incomplete_market_data", true),
            ("connection refused by peer", "exchange_unavailable", true),
            ("timeframe unsupported here", "unsupported_timeframe", false),
            ("weird thing", "no_market_data", true),
            ("", "no_market_data", true),
        ];
        for (err, code, retry) in cases {
            let f = classify_market_data_failure(err, " Binance ", "SPOT", "BTC", "1h");
            assert_eq!(&f.code, code, "{err}");
            assert_eq!(f.retryable, *retry, "{err}");
            assert_eq!(f.exchange_id, "binance");
            assert_eq!(f.market_type, "spot");
        }
    }

    #[test]
    fn branch_order_region_beats_proxy() {
        let f = classify_market_data_failure("451 proxyerror", "", "", "", "");
        assert_eq!(f.code, "region_restricted");
    }

    #[test]
    fn credentials_scrubbed_globally_case_insensitive() {
        let f = classify_market_data_failure(
            "get https://user:pass@api.x.com/a and HTTP://a:b@h.io/b failed: timeout",
            "", "", "", "",
        );
        assert_eq!(f.technical_detail, "get https://***:***@api.x.com/a and HTTP://***:***@h.io/b failed: timeout");
        assert_eq!(f.code, "exchange_unavailable");
    }

    #[test]
    fn scrub_rejects_bare_paths() {
        assert_eq!(scrub_credentials("https://host/path no creds"), "https://host/path no creds");
        assert_eq!(scrub_credentials("https://u@host/"), "https://u@host/");
    }

    #[test]
    fn mapping_defaults_and_truncation() {
        let f = MarketDataFailure::from_mapping(&[]);
        assert_eq!(f.code, "no_market_data");
        assert!(f.retryable);
        let long = "é".repeat(600);
        let f2 = MarketDataFailure::from_mapping(&[
            ("technical_detail".to_string(), JsonVal::Str(long.clone())),
            ("code".to_string(), JsonVal::Str(String::new())),
            ("retryable".to_string(), JsonVal::Bool(false)),
        ]);
        assert_eq!(f2.technical_detail.chars().count(), 500);
        assert_eq!(f2.code, "no_market_data");
        assert!(!f2.retryable);
    }
}
