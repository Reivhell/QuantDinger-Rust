//! Port of `backend_api_python/app/services/strategy_v2/instruments.py`.
//!
//! Canonical instrument parsing for Strategy API V2: market prefix routing,
//! venue (`@exchange:type`) splitting, per-market symbol normalisation, and
//! pool / index reference helpers.
//!
//! The unused `normalize_frequency` import in the Python module is dropped;
//! everything else is a line-for-line port. Errors carry the exact
//! `strategyV2.*` message strings the Python `InstrumentParseError` raises.

/// Parsed instrument. Mirrors `InstrumentSpec` (key/metadata only — the
/// frozen-dataclass wrapper adds nothing computable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstrumentSpec {
    pub market: String,
    pub symbol: String,
    pub exchange_id: String,
    pub market_type: String,
    pub instrument_id: String,
}

impl InstrumentSpec {
    /// Canonical key: `Market:SYMBOL[@exchange[:type] | @type]`.
    /// Mirrors `InstrumentSpec.key`.
    pub fn key(&self) -> String {
        let mut suffix = String::new();
        if !self.exchange_id.is_empty() {
            suffix.push('@');
            suffix.push_str(&self.exchange_id);
            if !self.market_type.is_empty() {
                suffix.push(':');
                suffix.push_str(&self.market_type);
            }
        } else if !self.market_type.is_empty() {
            suffix.push('@');
            suffix.push_str(&self.market_type);
        }
        format!("{}:{}{}", self.market, self.symbol, suffix)
    }
}

fn normalize_market(value: &str) -> String {
    let raw = value.trim();
    if raw.is_empty() {
        return String::new();
    }
    let canonical = match raw.to_lowercase().as_str() {
        "cnstock" => "CNStock",
        "usstock" => "USStock",
        "hkstock" => "HKStock",
        "crypto" => "Crypto",
        "forex" => "Forex",
        "futures" => "Futures",
        "moex" => "MOEX",
        _ => "",
    };
    if !canonical.is_empty() {
        return canonical.to_string();
    }
    // Case-sensitive fallback: already-canonical names pass through.
    match raw {
        "CNStock" | "USStock" | "HKStock" | "Crypto" | "Forex" | "Futures" | "MOEX" => {
            raw.to_string()
        }
        _ => String::new(),
    }
}

fn normalize_symbol(value: &str) -> String {
    let mut raw = value.trim().to_uppercase();
    for (old, new) in [
        (".XSHG", ".SH"),
        (".XSHE", ".SZ"),
        (".XBJS", ".BJ"),
        (".XHKG", ".HK"),
    ] {
        if raw.ends_with(old) {
            raw = format!("{}{}", &raw[..raw.len() - old.len()], new);
            break;
        }
    }
    raw
}

fn normalize_crypto_symbol(value: &str) -> String {
    let raw = value.trim().to_uppercase().replace('-', "/").replace('_', "/");
    if !raw.contains('/') {
        for quote in ["USDT", "USDC", "USD", "BTC", "ETH"] {
            if raw.ends_with(quote) && raw.len() > quote.len() {
                return format!("{}/{}", &raw[..raw.len() - quote.len()], quote);
            }
        }
    }
    raw
}

/// Guess the market from a normalised symbol. Mirrors `infer_market`.
pub fn infer_market(symbol: &str) -> String {
    let value = symbol.trim().to_uppercase();
    if value.is_empty() {
        return String::new();
    }
    if value.contains('/') || value.ends_with("USDT") || value.ends_with("USDC") {
        return "Crypto".to_string();
    }
    if value.ends_with(".HK") || value.ends_with(".XHKG") {
        return "HKStock".to_string();
    }
    if [".SH", ".SZ", ".BJ", ".XSHG", ".XSHE", ".XBJS"]
        .iter()
        .any(|s| value.ends_with(s))
    {
        return "CNStock".to_string();
    }
    if value.len() == 6 && value.bytes().all(|b| b.is_ascii_digit()) {
        return "CNStock".to_string();
    }
    // `re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", value)`
    let mut chars = value.chars();
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() => {}
        _ => return String::new(),
    }
    let mut len = 1;
    for c in chars {
        if !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '.' || c == '-') {
            return String::new();
        }
        len += 1;
        if len > 15 {
            return String::new();
        }
    }
    "USStock".to_string()
}

/// Parse a user-supplied instrument reference. Mirrors `parse_instrument`;
/// `Err` carries the exact `strategyV2.*` message.
pub fn parse_instrument(value: &str, default_market: &str) -> Result<InstrumentSpec, String> {
    let raw = value.trim();
    if raw.is_empty() {
        return Err("strategyV2.instrumentRequired".to_string());
    }
    let mut market = normalize_market(default_market);
    let mut body = raw;
    if let Some(colon) = raw.find(':') {
        let normalized = normalize_market(&raw[..colon]);
        if !normalized.is_empty() {
            market = normalized;
            body = raw[colon + 1..].trim_start();
            if body.is_empty() {
                body = "";
            }
        }
    }
    let mut exchange_id = String::new();
    let mut market_type = String::new();
    if let Some(at) = body.rfind('@') {
        let venue = &body[at + 1..];
        body = body[..at].trim_end();
        if venue.trim().to_lowercase() == "spot" || venue.trim().to_lowercase() == "swap" {
            market_type = venue.to_string();
        } else if venue.contains(':') {
            let vc = venue.find(':').unwrap();
            exchange_id = venue[..vc].to_string();
            market_type = venue[vc + 1..].to_string();
        } else {
            exchange_id = venue.to_string();
        }
        exchange_id = exchange_id.trim().to_lowercase();
        market_type = market_type.trim().to_lowercase();
    }
    let mut symbol = normalize_symbol(body);
    if market.is_empty() {
        market = infer_market(&symbol);
    }
    if market.is_empty() {
        return Err(format!("strategyV2.marketUnknown:{raw}"));
    }
    if market == "Crypto" {
        symbol = normalize_crypto_symbol(&symbol);
        if market_type != "spot" && market_type != "swap" {
            market_type = "spot".to_string();
        }
    } else {
        exchange_id = String::new();
        market_type = String::new();
    }
    Ok(InstrumentSpec {
        market,
        symbol,
        exchange_id,
        market_type,
        instrument_id: raw.to_string(),
    })
}

/// Mirrors `is_index_reference`.
pub fn is_index_reference(value: &str) -> bool {
    let raw = value.trim().to_uppercase();
    raw.starts_with("INDEX:") || raw.ends_with(".XBHS") || raw.ends_with(".XSHG_INDEX")
}

/// Mirrors `normalize_index_reference`.
pub fn normalize_index_reference(value: &str) -> String {
    let raw = value.trim();
    if raw.to_uppercase().starts_with("INDEX:") {
        return raw.to_string();
    }
    let upper = raw.to_uppercase();
    if upper.ends_with(".XBHS") {
        return format!("CNStock:{}.SH", &upper[..upper.len() - 5]);
    }
    raw.to_string()
}

/// Mirrors `normalize_pool_reference`; `Err` carries `strategyV2.universeRequired`.
pub fn normalize_pool_reference(value: &str) -> Result<String, String> {
    let mut raw = value.trim();
    if raw.is_empty() {
        return Err("strategyV2.universeRequired".to_string());
    }
    if raw.to_uppercase().starts_with("POOL:") {
        raw = raw.split_once(':').map(|(_, rest)| rest).unwrap_or(raw);
    }
    Ok(format!("POOL:{}", raw.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors test_instrument_parser_normalizes_ptrade_and_crypto_symbols
    #[test]
    fn ptrade_and_crypto_keys() {
        assert_eq!(parse_instrument("600519.XSHG", "").unwrap().key(), "CNStock:600519.SH");
        assert_eq!(parse_instrument("USStock:MSFT", "").unwrap().key(), "USStock:MSFT");
        assert_eq!(
            parse_instrument("Crypto:BTCUSDT@okx:swap", "").unwrap().key(),
            "Crypto:BTC/USDT@okx:swap"
        );
        assert_eq!(
            parse_instrument("Crypto:BTC/USDT@swap", "").unwrap().key(),
            "Crypto:BTC/USDT@swap"
        );
    }

    #[test]
    fn odd_venue_shape_from_contract_tests() {
        // test_manifest ... instruments[0]: "Crypto:00700/HKD@gate:spot"
        let spec = parse_instrument("Crypto:00700/HKD@gate:spot", "").unwrap();
        assert_eq!(spec.key(), "Crypto:00700/HKD@gate:spot");
        assert_eq!(spec.market, "Crypto");
        assert_eq!(spec.symbol, "00700/HKD");
        assert_eq!(spec.exchange_id, "gate");
        assert_eq!(spec.market_type, "spot");
    }

    #[test]
    fn default_market_and_inference() {
        let spec = parse_instrument("MSFT", "USStock").unwrap();
        assert_eq!(spec.key(), "USStock:MSFT");
        assert_eq!(parse_instrument("BTCUSDT", "").unwrap().key(), "Crypto:BTC/USDT@spot");
        assert_eq!(parse_instrument("000001", "").unwrap().market, "CNStock");
        assert_eq!(parse_instrument("00700.HK", "").unwrap().market, "HKStock");
    }

    #[test]
    fn errors_carry_exact_codes() {
        assert_eq!(
            parse_instrument("", "").unwrap_err(),
            "strategyV2.instrumentRequired"
        );
        assert_eq!(
            parse_instrument("!!!", "").unwrap_err(),
            "strategyV2.marketUnknown:!!!"
        );
        assert_eq!(
            normalize_pool_reference("").unwrap_err(),
            "strategyV2.universeRequired"
        );
    }

    #[test]
    fn non_crypto_drops_venue() {
        let spec = parse_instrument("USStock:MSFT@nasdaq:spot", "").unwrap();
        assert_eq!(spec.exchange_id, "");
        assert_eq!(spec.market_type, "");
        assert_eq!(spec.key(), "USStock:MSFT");
    }

    #[test]
    fn crypto_type_defaults_to_spot() {
        assert_eq!(
            parse_instrument("Crypto:BTC/USDT@binance", "").unwrap().key(),
            "Crypto:BTC/USDT@binance:spot"
        );
        assert_eq!(
            parse_instrument("Crypto:BTC/USDT", "").unwrap().market_type,
            "spot"
        );
    }

    #[test]
    fn index_and_pool_references() {
        assert!(is_index_reference("INDEX:CSI300"));
        assert!(is_index_reference("000300.xbhs"));
        assert!(!is_index_reference("Crypto:BTC/USDT"));
        assert_eq!(normalize_index_reference("000300.XBHS"), "CNStock:000300.SH");
        assert_eq!(normalize_index_reference("INDEX:x"), "INDEX:x");
        assert_eq!(normalize_pool_reference("POOL:ABC").unwrap(), "POOL:abc");
        assert_eq!(normalize_pool_reference("abc").unwrap(), "POOL:abc");
    }
}
