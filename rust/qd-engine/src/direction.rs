//! Port of `backend_api_python/app/services/strategy_direction.py` except
//! `infer_direction_mode_from_code` (needs `ast.parse`; stays Python-side).
//!
//! All inputs are [`JsonVal`](crate::json_helpers::JsonVal): every helper
//! starts from `str(value or "")` semantics, so falsy JSON
//! (`null`/`""`/`0`/`false`/`[]`/`{}`) collapses to `""` exactly like Python.

use crate::json_helpers::JsonVal;

/// `str(value or "")`: falsy → `""`, else Python-`str` rendering.
/// Numbers keep their raw token (`1.0` stays `1.0`); bools render
/// `True`/`False` (lowercased by callers, matching Python).
pub fn or_str(value: &JsonVal, default: &str) -> String {
    if is_falsy(value) {
        return default.to_string();
    }
    match value {
        JsonVal::Str(s) => s.clone(),
        JsonVal::Num(raw) => raw.clone(),
        JsonVal::Bool(true) => "True".to_string(),
        JsonVal::Bool(false) => "False".to_string(),
        JsonVal::Null => "None".to_string(),
        JsonVal::Arr(xs) => {
            format!("[{}]", xs.iter().map(py_repr).collect::<Vec<_>>().join(", "))
        }
        JsonVal::Obj(kv) => format!(
            "{{{}}}",
            kv.iter().map(|(k, v)| format!("{k:?}: {}", py_repr(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python truthiness for JSON values.
pub fn is_falsy(value: &JsonVal) -> bool {
    match value {
        JsonVal::Null => true,
        JsonVal::Bool(b) => !b,
        JsonVal::Num(raw) => raw.parse::<f64>().map(|f| f == 0.0).unwrap_or(false),
        JsonVal::Str(s) => s.is_empty(),
        JsonVal::Arr(xs) => xs.is_empty(),
        JsonVal::Obj(kv) => kv.is_empty(),
    }
}

/// Python `repr` for scalar JSON values (used inside container rendering).
fn py_repr(value: &JsonVal) -> String {
    match value {
        JsonVal::Str(s) => format!("{s:?}"),
        JsonVal::Num(raw) => raw.clone(),
        JsonVal::Bool(true) => "True".to_string(),
        JsonVal::Bool(false) => "False".to_string(),
        JsonVal::Null => "None".to_string(),
        v => or_str(v, ""),
    }
}

/// Mirrors `normalize_direction_mode`.
pub fn normalize_direction_mode(value: &JsonVal) -> String {
    let normalized = or_str(value, "").trim().to_lowercase().replace('-', "_");
    let aliased = match normalized.as_str() {
        "long" | "buy" | "1" | "1.0" | "+1" | "longonly" => "long_only",
        "short" | "sell" | "_1" | "_1.0" | "-1" | "shortonly" => "short_only",
        "oneway" | "net" | "net_position" | "single_position" => "one_way",
        "dual" | "hedged" | "bidirectional" | "two_way" => "both",
        other => other,
    };
    match aliased {
        "long_only" | "short_only" | "one_way" | "both" | "neutral" => aliased.to_string(),
        _ => String::new(),
    }
}

/// Mirrors `direction_mode_position_side`.
pub fn direction_mode_position_side(value: &JsonVal) -> String {
    match normalize_direction_mode(value).as_str() {
        "long_only" => "long",
        "short_only" => "short",
        "both" | "neutral" => "neutral",
        _ => "",
    }
    .to_string()
}

/// Mirrors `direction_mode_owned_legs` (sorted; Python returns a set).
pub fn direction_mode_owned_legs(value: &JsonVal) -> Vec<String> {
    match normalize_direction_mode(value).as_str() {
        "long_only" => vec!["long".to_string()],
        "short_only" => vec!["short".to_string()],
        _ => vec!["long".to_string(), "short".to_string()],
    }
}

/// Mirrors `direction_mode_allows`.
pub fn direction_mode_allows(value: &JsonVal, position_side: &JsonVal) -> bool {
    let mode = normalize_direction_mode(value);
    let side = or_str(position_side, "").trim().to_lowercase();
    if !matches!(side.as_str(), "long" | "short") {
        return true;
    }
    if matches!(mode.as_str(), "one_way" | "both" | "neutral") {
        return true;
    }
    (mode == "long_only" && side == "long") || (mode == "short_only" && side == "short")
}

/// Mirrors `direction_mode_from_manifest`. `top` holds the manifest mapping;
/// the nested `metadata` object is read when present and a mapping.
pub fn direction_mode_from_manifest(top: &[(String, JsonVal)]) -> String {
    let get = |k: &str| top.iter().find(|(kk, _)| kk == k).map(|(_, v)| v);
    let meta: Vec<(String, JsonVal)> = match get("metadata") {
        Some(JsonVal::Obj(kv)) => kv.clone(),
        _ => Vec::new(),
    };
    let mget = |k: &str| meta.iter().find(|(kk, _)| kk == k).map(|(_, v)| v);
    let null = JsonVal::Null;
    let candidates = [
        get("directionMode").unwrap_or(&null),
        get("direction_mode").unwrap_or(&null),
        mget("direction_mode").unwrap_or(&null),
        mget("directionMode").unwrap_or(&null),
        mget("trade_direction").unwrap_or(&null),
        mget("position_side").unwrap_or(&null),
        mget("side").unwrap_or(&null),
    ];
    for c in candidates {
        let mode = normalize_direction_mode(c);
        if !mode.is_empty() {
            return mode;
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> JsonVal {
        JsonVal::Str(v.to_string())
    }

    #[test]
    fn aliases_and_passthrough() {
        let cases = [
            ("long", "long_only"), ("BUY", "long_only"), ("1", "long_only"),
            ("1.0", "long_only"), ("+1", "long_only"), ("longonly", "long_only"),
            ("short", "short_only"), ("sell", "short_only"), ("-1", "short_only"),
            ("_1", "short_only"), ("oneway", "one_way"), ("net", "one_way"),
            ("dual", "both"), ("two_way", "both"), ("hedged", "both"),
            ("long_only", "long_only"), ("neutral", "neutral"),
            ("LONG-ONLY", "long_only"), ("  both  ", "both"),
            ("sideways", ""), ("", ""),
        ];
        for (inp, want) in cases {
            assert_eq!(normalize_direction_mode(&s(inp)), want, "{inp}");
        }
        assert_eq!(normalize_direction_mode(&JsonVal::Null), "");
        assert_eq!(normalize_direction_mode(&JsonVal::Num("1".into())), "long_only");
        assert_eq!(normalize_direction_mode(&JsonVal::Num("1.0".into())), "long_only");
        assert_eq!(normalize_direction_mode(&JsonVal::Bool(true)), "");
        assert_eq!(normalize_direction_mode(&JsonVal::Num("0".into())), "");
    }

    #[test]
    fn side_legs_allows_matrix() {
        assert_eq!(direction_mode_position_side(&s("long")), "long");
        assert_eq!(direction_mode_position_side(&s("short")), "short");
        assert_eq!(direction_mode_position_side(&s("both")), "neutral");
        assert_eq!(direction_mode_position_side(&s("neutral")), "neutral");
        assert_eq!(direction_mode_position_side(&s("one_way")), "");
        assert_eq!(direction_mode_position_side(&s("bogus")), "");
        assert_eq!(direction_mode_owned_legs(&s("long")), vec!["long"]);
        assert_eq!(direction_mode_owned_legs(&s("short")), vec!["short"]);
        assert_eq!(direction_mode_owned_legs(&s("both")), vec!["long", "short"]);
        assert_eq!(direction_mode_owned_legs(&s("bogus")), vec!["long", "short"]);
        assert!(direction_mode_allows(&s("long"), &s("long")));
        assert!(!direction_mode_allows(&s("long"), &s("short")));
        assert!(!direction_mode_allows(&s("short"), &s("long")));
        assert!(direction_mode_allows(&s("one_way"), &s("short")));
        assert!(direction_mode_allows(&s("both"), &s("short")));
        assert!(direction_mode_allows(&s("bogus"), &s("sideways")));
        assert!(!direction_mode_allows(&s("bogus"), &s("long")));
    }

    #[test]
    fn manifest_priority_order() {
        let top = vec![
            ("directionMode".to_string(), s("short")),
            ("direction_mode".to_string(), s("long")),
            ("metadata".to_string(), JsonVal::Obj(vec![("side".to_string(), s("both"))])),
        ];
        assert_eq!(direction_mode_from_manifest(&top), "short_only");
        let top2 = vec![("metadata".to_string(), JsonVal::Obj(vec![("side".to_string(), s("both"))]))];
        assert_eq!(direction_mode_from_manifest(&top2), "both");
        // first valid wins: invalid top-level falls through to metadata
        let top3 = vec![
            ("direction_mode".to_string(), s("bogus")),
            ("metadata".to_string(), JsonVal::Obj(vec![("trade_direction".to_string(), s("net"))])),
        ];
        assert_eq!(direction_mode_from_manifest(&top3), "one_way");
        assert_eq!(direction_mode_from_manifest(&[]), "");
        // non-dict metadata ignored
        let top4 = vec![("metadata".to_string(), JsonVal::Num("1".into()))];
        assert_eq!(direction_mode_from_manifest(&top4), "");
    }
}
