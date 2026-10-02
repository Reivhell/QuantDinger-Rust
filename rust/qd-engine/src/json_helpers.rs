//! Port of `backend_api_python/app/utils/json_helpers.py`.
//!
//! `safe_json_loads` plus the minimal JSON value model it needs (std-only:
//! no serde — crates.io is unreachable from this environment).
//!
//! Faithful corners:
//! - dict/list input returns as-is (here: [`JsonInput::Value`] clones back).
//! - Only non-blank strings are parsed; anything else yields `default`.
//! - Malformed JSON yields `default` instead of raising.
//! - The default when none is given is `{}` (empty object).
//! - Numbers keep their raw token so re-emission is byte-faithful
//!   (`1e+16` stays `1e+16`, `100.0` stays `100.0`, big ints don't lose
//!   precision through f64).

#[derive(Debug, Clone, PartialEq)]
pub enum JsonVal {
    Null,
    Bool(bool),
    /// Raw number token, exactly as written.
    Num(String),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(Vec<(String, JsonVal)>),
}

impl JsonVal {
    /// Numeric value when the token parses as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonVal::Num(raw) => raw.parse().ok(),
            _ => None,
        }
    }

    /// Emit canonical JSON (`ensure_ascii=True`, compact separators).
    pub fn dump(&self) -> String {
        self.dump_inner(false)
    }

    /// Emit compact JSON with raw UTF-8 passthrough (`ensure_ascii=False`),
    /// matching `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`.
    pub fn dump_raw(&self) -> String {
        self.dump_inner(true)
    }

    fn dump_inner(&self, raw_utf8: bool) -> String {
        match self {
            JsonVal::Null => "null".to_string(),
            JsonVal::Bool(true) => "true".to_string(),
            JsonVal::Bool(false) => "false".to_string(),
            JsonVal::Num(raw) => raw.clone(),
            JsonVal::Str(s) => dump_str_inner(s, raw_utf8),
            JsonVal::Arr(xs) => {
                format!("[{}]", xs.iter().map(|x| x.dump_inner(raw_utf8)).collect::<Vec<_>>().join(","))
            }
            JsonVal::Obj(kv) => format!(
                "{{{}}}",
                kv.iter()
                    .map(|(k, v)| format!("{}:{}", dump_str_inner(k, raw_utf8), v.dump_inner(raw_utf8)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

fn dump_str(s: &str) -> String {
    dump_str_inner(s, false)
}

fn dump_str_inner(s: &str, raw_utf8: bool) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) > 0x7F && !raw_utf8 => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Input accepted by `safe_json_loads`.
#[derive(Debug, Clone)]
pub enum JsonInput {
    Value(JsonVal),
    Text(String),
    Other,
}

/// Mirrors `safe_json_loads(value, default=None)`.
pub fn safe_json_loads(input: &JsonInput, default: &JsonVal) -> JsonVal {
    match input {
        JsonInput::Value(v) => v.clone(),
        JsonInput::Text(s) if !s.trim().is_empty() => parse_json(s).unwrap_or_else(|_| default.clone()),
        _ => default.clone(),
    }
}

/// Parse one JSON document; `Err` on any malformation (incl. trailing data).
pub fn parse_json(s: &str) -> Result<JsonVal, String> {
    let mut p = Parser { b: s.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, s: &str) -> bool {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<JsonVal, String> {
        match *self.b.get(self.i).ok_or("unexpected end".to_string())? {
            b'n' => self.lit("null").then(|| JsonVal::Null).ok_or("bad literal".to_string()),
            b't' => self.lit("true").then(|| JsonVal::Bool(true)).ok_or("bad literal".to_string()),
            b'f' => self.lit("false").then(|| JsonVal::Bool(false)).ok_or("bad literal".to_string()),
            b'"' => Ok(JsonVal::Str(self.string()?)),
            b'[' => self.array(),
            b'{' => self.object(),
            c if c == b'-' || c.is_ascii_digit() => Ok(JsonVal::Num(self.number()?)),
            c => Err(format!("unexpected byte {c:#x} at {}", self.i)),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string".to_string())?;
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    match *self.b.get(self.i).ok_or("bad escape".to_string())? {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.i) == Some(&b'\\')
                                    && self.b.get(self.i + 1) == Some(&b'u')
                                {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&lo) {
                                        return Err("bad low surrogate".to_string());
                                    }
                                    let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                    out.push(char::from_u32(cp).ok_or("bad codepoint".to_string())?);
                                    continue;
                                }
                                return Err("lone surrogate".to_string());
                            }
                            if (0xDC00..0xE000).contains(&hi) {
                                return Err("lone surrogate".to_string());
                            }
                            out.push(char::from_u32(hi).ok_or("bad codepoint".to_string())?);
                            continue;
                        }
                        e => return Err(format!("bad escape \\{e}")),
                    }
                    self.i += 1;
                }
                0x00..=0x1F => return Err("unescaped control".to_string()),
                _ => {
                    let s = std::str::from_utf8(&self.b[self.i..]).map_err(|e| e.to_string())?;
                    let c = s.chars().next().ok_or("unterminated string".to_string())?;
                    out.push(c);
                    self.i += c.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("bad \\u escape".to_string());
        }
        let h = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        let v = u32::from_str_radix(h, 16).map_err(|_| "bad \\u escape".to_string())?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<String, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        match self.b.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(c) if c.is_ascii_digit() => {
                while self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return Err("bad number".to_string()),
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if !self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) {
                return Err("bad number".to_string());
            }
            while self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.b.get(self.i), Some(b'e') | Some(b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            if !self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) {
                return Err("bad number".to_string());
            }
            while self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        std::str::from_utf8(&self.b[start..self.i]).map(|s| s.to_string()).map_err(|e| e.to_string())
    }

    fn array(&mut self) -> Result<JsonVal, String> {
        self.i += 1;
        let mut xs = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(JsonVal::Arr(xs));
        }
        loop {
            self.ws();
            xs.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(JsonVal::Arr(xs));
                }
                _ => return Err("expected , or ]".to_string()),
            }
        }
    }

    fn object(&mut self) -> Result<JsonVal, String> {
        self.i += 1;
        let mut kv = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(JsonVal::Obj(kv));
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err("expected string key".to_string());
            }
            let k = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err("expected :".to_string());
            }
            self.i += 1;
            self.ws();
            kv.push((k, self.value()?));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(JsonVal::Obj(kv));
                }
                _ => return Err("expected , or }".to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dflt() -> JsonVal {
        JsonVal::Obj(vec![])
    }

    #[test]
    fn round_trips_documents() {
        for doc in [
            r#"{"a":1,"b":[1.5,-2e-3,true,null,"x"]}"#,
            r#"{"caf\u00e9":"\u0041"}"#,
            r#"[1,2,3]"#,
            r#""s""#,
            "123",
            "-0.5",
            "1E+16",
        ] {
            let v = parse_json(doc).unwrap();
            let back = parse_json(&v.dump()).unwrap();
            assert_eq!(v, back, "{doc}");
        }
    }

    #[test]
    fn numbers_keep_raw_tokens() {
        assert_eq!(parse_json("100.0").unwrap(), JsonVal::Num("100.0".into()));
        assert_eq!(parse_json("1e+16").unwrap().dump(), "1e+16");
        assert_eq!(
            parse_json("123456789123456789123456789").unwrap().as_f64().is_some(),
            true
        );
    }

    #[test]
    fn malformed_inputs_error() {
        for bad in ["", "  ", "{bad", "[1,]", "{\"a\":}", "01", "1.", "nul", "[1 2]", "{\"a\":1} x"] {
            assert!(parse_json(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn safe_loads_mirrors_python() {
        assert_eq!(
            safe_json_loads(&JsonInput::Text("  ".into()), &dflt()),
            JsonVal::Obj(vec![])
        );
        assert_eq!(
            safe_json_loads(&JsonInput::Text("{bad".into()), &dflt()),
            JsonVal::Obj(vec![])
        );
        assert_eq!(
            safe_json_loads(&JsonInput::Text("[1,2]".into()), &dflt()),
            JsonVal::Arr(vec![JsonVal::Num("1".into()), JsonVal::Num("2".into())])
        );
        let v = JsonVal::Bool(true);
        assert_eq!(safe_json_loads(&JsonInput::Value(v.clone()), &dflt()), v);
        assert_eq!(safe_json_loads(&JsonInput::Other, &dflt()), dflt());
    }

    #[test]
    fn surrogate_pairs_decode() {
        assert_eq!(parse_json(r#""\uD83D\uDE00""#).unwrap(), JsonVal::Str("😀".into()));
        assert!(parse_json(r#""\uD83D""#).is_err());
    }
}
