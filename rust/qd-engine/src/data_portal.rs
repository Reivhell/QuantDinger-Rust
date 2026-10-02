//! Port of the pandas-free logic of
//! `backend_api_python/app/services/strategy_v2/data.py`
//! (`MultiAssetDataPortal`).
//!
//! The portal is the strategy's point-in-time data window — the anti-lookahead
//! core: a strategy only ever sees bars at or before the simulation clock.
//! Everything pandas-specific (DatetimeIndex, DataFrame slicing) is replaced
//! by sorted `Vec<i64>` epoch-second indexes with binary search; all error
//! strings, resolution rules, and NaN/default semantics are preserved.
//!
//! Deliberately NOT ported (stay Python): `history`/`panel`/`visible_frame`
//! DataFrame assembly, `universe` resolver callback, `_normalize_frame`'s
//! `time`-column coercion (`pd.to_datetime(errors="coerce")` — callers pass
//! `ts: None` for unparseable stamps, which are dropped like NaT).
//!
//! Faithful corners:
//! - `resolve_key`: direct hit → alias hit → `parse_instrument` hit → unique
//!   symbol-suffix match → `strategyV2.dataUnavailable:{raw}`.
//! - Visible end = `searchsorted(cutoff, side="right")` = count of `ts <=
//!   cutoff`; cutoff `None` (clock unset) → `0`, so nothing is visible.
//! - `current`: `end == 0` or unknown field → `default`; NaN → `default`
//!   (mirrors `value if value == value else default`).
//! - `open_at`/`close_at`: `float(... or 0.0)` then `> 0` else `None`.
//! - Duplicate bars with conflicting OHLCV raise
//!   `strategyV2.conflictingDuplicateBar:{key}:{iso}`; identical dupes keep
//!   the last (mirrors `duplicated(keep="last")` after the conflict scan).
//! - Missing `volume` column defaults to `0.0`; a present-but-NaN volume
//!   stays NaN (`nan or 0.0` is `nan` — NaN is truthy).

use std::collections::HashMap;

use crate::instruments::parse_instrument;

/// One raw input bar. `ts: None` = unparseable stamp (dropped like NaT).
#[derive(Debug, Clone)]
pub struct RawBar {
    pub ts: Option<i64>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// `None` + volume column present = NaN (kept, truthy `or`); column
    /// absent = `0.0`. Carries `(column_present, value)`.
    pub volume: (bool, Option<f64>),
    /// Extra numeric columns (`suspended`, `limit_up`, …) passed through.
    pub extras: Vec<(String, f64)>,
}

/// Raw input frame: column names as given + rows.
#[derive(Debug, Clone)]
pub struct RawFrame {
    pub columns: Vec<String>,
    pub rows: Vec<RawBar>,
}

/// Normalized frame: sorted, deduped, epoch-indexed columns.
#[derive(Debug, Clone)]
pub struct Frame {
    pub key: String,
    pub ts: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub extras: HashMap<String, Vec<f64>>,
}

/// A bar lookup result. Mirrors `bar_at`'s dict.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub extras: Vec<(String, f64)>,
}

/// Format epoch seconds as tz-naive ISO-8601 (`2026-01-01T00:00:00`),
/// matching `pd.Timestamp(ts).isoformat()` on the tz-naive stamps the
/// portal works with (including inside `conflictingDuplicateBar` messages).
pub fn iso_naive(ts: i64) -> String {
    let days = ts.div_euclid(86_400);
    let secs = ts.rem_euclid(86_400);
    // Howard Hinnant's civil-from-days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    y += if m <= 2 { 1 } else { 0 };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        y,
        m,
        d,
        secs / 3_600,
        (secs % 3_600) / 60,
        secs % 60
    )
}

/// Normalize one raw frame. Mirrors `_normalize_frame` (minus timestamp
/// parsing and column lowering — callers pass those pre-done).
pub fn normalize_frame(key: &str, raw: &RawFrame) -> Result<Frame, String> {
    if raw.rows.is_empty() {
        return Err(format!("strategyV2.emptyData:{key}"));
    }
    let lowered: Vec<String> = raw.columns.iter().map(|c| c.trim().to_lowercase()).collect();
    let missing: Vec<&str> = ["open", "high", "low", "close"]
        .into_iter()
        .filter(|c| !lowered.iter().any(|x| x == c))
        .collect();
    if !missing.is_empty() {
        return Err(format!("strategyV2.ohlcRequired:{key}:{}", missing.join(",")));
    }
    // Stable sort by ts, dropping NaT (None).
    let mut order: Vec<usize> = (0..raw.rows.len()).collect();
    let mut kept: Vec<usize> = order
        .drain(..)
        .filter(|&i| raw.rows[i].ts.is_some())
        .collect();
    kept.sort_by_key(|&i| raw.rows[i].ts.unwrap());
    let vol_present = lowered.iter().any(|x| x == "volume");
    let mut frame = Frame {
        key: key.to_string(),
        ts: Vec::new(),
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        volume: Vec::new(),
        extras: HashMap::new(),
    };
    let mut i = 0;
    while i < kept.len() {
        let ts = raw.rows[kept[i]].ts.unwrap();
        let mut j = i + 1;
        while j < kept.len() && raw.rows[kept[j]].ts == Some(ts) {
            j += 1;
        }
        // Conflict scan over the dupe group (baseline = first row).
        let base = &raw.rows[kept[i]];
        for &k in &kept[i + 1..j] {
            let r = &raw.rows[k];
            let same = r.open.to_bits() == base.open.to_bits()
                && r.high.to_bits() == base.high.to_bits()
                && r.low.to_bits() == base.low.to_bits()
                && r.close.to_bits() == base.close.to_bits()
                && vol_of(r, vol_present).to_bits() == vol_of(base, vol_present).to_bits();
            if !same {
                return Err(format!(
                    "strategyV2.conflictingDuplicateBar:{key}:{}",
                    iso_naive(ts)
                ));
            }
        }
        // keep="last": the group's final row wins.
        let last = &raw.rows[kept[j - 1]];
        frame.ts.push(ts);
        frame.open.push(last.open);
        frame.high.push(last.high);
        frame.low.push(last.low);
        frame.close.push(last.close);
        frame.volume.push(vol_of(last, vol_present));
        for (name, val) in &last.extras {
            frame.extras.entry(name.clone()).or_default().push(*val);
        }
        i = j;
    }
    // Align extra columns across rows (NaN where a row lacked the column).
    let n = frame.ts.len();
    let mut seen: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    // rebuild: walk kept-last rows again
    let mut i2 = 0;
    let mut gi = 0;
    while i2 < kept.len() {
        let mut j = i2 + 1;
        while j < kept.len() && raw.rows[kept[j]].ts == raw.rows[kept[i2]].ts {
            j += 1;
        }
        let last = &raw.rows[kept[j - 1]];
        for (name, val) in &last.extras {
            seen.entry(name.clone()).or_insert_with(|| vec![None; n])[gi] = Some(*val);
        }
        gi += 1;
        i2 = j;
    }
    for (name, col) in seen {
        frame.extras.insert(name, col.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect());
    }
    Ok(frame)
}

fn vol_of(bar: &RawBar, present: bool) -> f64 {
    match (present, bar.volume.1) {
        (false, _) => 0.0,
        (true, Some(v)) => v,
        (true, None) => f64::NAN,
    }
}

/// Resolve a symbol to a canonical key. Mirrors `resolve_key`.
/// `frames`: keys subscribed at this frequency; `aliases`: symbol/raw → key.
pub fn resolve_key(
    frames: &HashMap<String, Frame>,
    aliases: &HashMap<String, String>,
    symbol: &str,
) -> Result<String, String> {
    let raw = symbol.trim();
    if let Some(f) = frames.get(raw) {
        return Ok(f.key.clone());
    }
    if let Some(a) = aliases.get(raw) {
        if frames.contains_key(a) {
            return Ok(a.clone());
        }
    }
    if let Ok(parsed) = parse_instrument(raw, "") {
        if frames.contains_key(&parsed.key()) {
            return Ok(parsed.key());
        }
        let matching: Vec<&String> = frames
            .keys()
            .filter(|k| {
                let after_colon = k.split_once(':').map(|(_, r)| r).unwrap_or(k.as_str());
                after_colon.split('@').next().unwrap_or("") == parsed.symbol
            })
            .collect();
        if matching.len() == 1 {
            return Ok(matching[0].clone());
        }
    }
    Err(format!("strategyV2.dataUnavailable:{raw}"))
}

/// Visible-bar cutoff for a clock position. Mirrors `_visible_cutoff`.
pub fn visible_cutoff(
    current: Option<i64>,
    include_current: bool,
    driving_secs: i64,
    freq_secs: i64,
) -> Option<i64> {
    current.map(|now| now + if include_current { driving_secs } else { 0 } - freq_secs)
}

/// Count of bars at or before the cutoff (`searchsorted(cutoff, "right")`).
/// `None` cutoff → `0`.
pub fn visible_end(ts: &[i64], cutoff: Option<i64>) -> usize {
    match cutoff {
        None => 0,
        Some(c) => ts.partition_point(|&t| t <= c),
    }
}

/// Latest visible value of a column, else `default`. Mirrors `current`.
pub fn current_value(col: &[f64], end: usize, default: f64) -> f64 {
    if end == 0 || end > col.len() {
        return default;
    }
    let v = col[end - 1];
    if v == v { v } else { default }
}

/// Exact-timestamp bar lookup. Mirrors `bar_at` (no cache — callers memoize).
pub fn bar_at(frame: &Frame, ts: i64) -> Option<Bar> {
    let idx = frame.ts.binary_search(&ts).ok()?;
    Some(Bar {
        open: frame.open[idx],
        high: frame.high[idx],
        low: frame.low[idx],
        close: frame.close[idx],
        volume: frame.volume[idx],
        extras: frame
            .extras
            .iter()
            .map(|(k, col)| (k.clone(), col[idx]))
            .collect(),
    })
}

/// Positive-price gate. Mirrors `open_at`/`close_at`.
pub fn positive_price(v: f64) -> Option<f64> {
    let v = if v == 0.0 { 0.0 } else { v }; // `or 0.0`: None/0 → 0.0
    if v > 0.0 { Some(v) } else { None }
}

/// Mirrors `_as_list` for pre-decoded inputs.
#[derive(Debug, Clone)]
pub enum ListInput {
    Nil,
    Str(String),
    Keys(Vec<String>),
    List(Vec<String>),
    Single(String),
}

pub fn as_list(v: &ListInput) -> Vec<String> {
    match v {
        ListInput::Nil => vec![],
        ListInput::Str(s) => vec![s.clone()],
        ListInput::Keys(k) => k.clone(),
        ListInput::List(l) => l.clone(),
        ListInput::Single(s) => vec![s.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: i64 = 86_400;
    const B: i64 = 1_767_225_600; // Thu 2026-01-01 00:00 UTC

    fn bar(ts: i64, o: f64, c: f64) -> RawBar {
        RawBar {
            ts: Some(ts),
            open: o,
            high: o.max(c),
            low: o.min(c),
            close: c,
            volume: (true, Some(100.0)),
            extras: vec![],
        }
    }

    fn cols() -> Vec<String> {
        ["open", "high", "low", "close", "volume"].iter().map(|s| s.to_string()).collect()
    }

    fn frame3() -> Frame {
        normalize_frame(
            "Crypto:BTC/USDT@spot",
            &RawFrame {
                columns: cols(),
                rows: vec![bar(B, 100.0, 101.0), bar(B + D, 101.0, 102.0), bar(B + 2 * D, 102.0, 103.0)],
            },
        )
        .unwrap()
    }

    #[test]
    fn empty_and_missing_columns_err_with_python_messages() {
        assert_eq!(
            normalize_frame("K", &RawFrame { columns: cols(), rows: vec![] }).unwrap_err(),
            "strategyV2.emptyData:K"
        );
        assert_eq!(
            normalize_frame(
                "K",
                &RawFrame { columns: vec!["open".into(), "close".into()], rows: vec![bar(B, 1.0, 2.0)] }
            )
            .unwrap_err(),
            "strategyV2.ohlcRequired:K:high,low"
        );
    }

    #[test]
    fn nat_dropped_and_volume_defaults_without_column() {
        let f = normalize_frame(
            "K",
            &RawFrame {
                columns: vec!["open".into(), "high".into(), "low".into(), "close".into()],
                rows: vec![
                    RawBar { ts: None, ..bar(B, 1.0, 2.0) },
                    RawBar { volume: (false, None), ..bar(B + D, 3.0, 4.0) },
                ],
            },
        )
        .unwrap();
        assert_eq!(f.ts, vec![B + D]);
        assert_eq!(f.volume, vec![0.0]);
    }

    #[test]
    fn conflicting_dupe_errors_identical_dupe_keeps_last() {
        let rows = vec![bar(B, 1.0, 2.0), bar(B, 1.0, 9.0)];
        assert_eq!(
            normalize_frame("K", &RawFrame { columns: cols(), rows }).unwrap_err(),
            format!("strategyV2.conflictingDuplicateBar:K:{}", iso_naive(B))
        );
        let f = normalize_frame(
            "K",
            &RawFrame {
                columns: cols(),
                rows: vec![
                    RawBar { volume: (true, Some(1.0)), ..bar(B, 1.0, 2.0) },
                    RawBar { volume: (true, Some(2.0)), ..bar(B, 1.0, 2.0) },
                ],
            },
        )
        .unwrap_err();
        assert!(f.starts_with("strategyV2.conflictingDuplicateBar:K:"));
        // identical dupes collapse to one row
        let f = normalize_frame(
            "K",
            &RawFrame { columns: cols(), rows: vec![bar(B, 1.0, 2.0), bar(B, 1.0, 2.0)] },
        )
        .unwrap();
        assert_eq!(f.ts, vec![B]);
    }

    #[test]
    fn cutoff_and_visible_end_gate_the_future() {
        // clock Tue 06:00, daily bars: cutoff Mon 06:00 → only Mon's bar.
        let c = visible_cutoff(Some(B + D + 6 * 3_600), false, D, D);
        assert_eq!(c, Some(B + 6 * 3_600));
        let f = frame3();
        assert_eq!(visible_end(&f.ts, c), 1);
        assert_eq!(visible_end(&f.ts, None), 0);
        // include_current pushes one driving period further
        let c2 = visible_cutoff(Some(B + D + 6 * 3_600), true, D, D);
        assert_eq!(c2, Some(B + D + 6 * 3_600));
        assert_eq!(visible_end(&f.ts, c2), 2);
    }

    #[test]
    fn current_nan_and_empty_default() {
        assert_eq!(current_value(&[1.0, f64::NAN], 2, 7.0), 7.0);
        assert_eq!(current_value(&[1.0], 0, 7.0), 7.0);
        assert_eq!(current_value(&[1.0, 2.0], 2, 7.0), 2.0);
    }

    #[test]
    fn bar_lookup_and_price_gate() {
        let f = frame3();
        let b = bar_at(&f, B + D).unwrap();
        assert_eq!((b.open, b.close, b.volume), (101.0, 102.0, 100.0));
        assert!(bar_at(&f, B + 3 * D).is_none());
        assert_eq!(positive_price(5.0), Some(5.0));
        assert_eq!(positive_price(0.0), None);
        assert_eq!(positive_price(-1.0), None);
        assert_eq!(positive_price(f64::NAN), None);
    }

    #[test]
    fn resolve_key_hits_and_misses() {
        let mut frames = HashMap::new();
        frames.insert("Crypto:BTC/USDT@spot".to_string(), frame3());
        let mut aliases = HashMap::new();
        aliases.insert("BTCUSDT".to_string(), "Crypto:BTC/USDT@spot".to_string());
        assert_eq!(
            resolve_key(&frames, &aliases, "Crypto:BTC/USDT@spot").unwrap(),
            "Crypto:BTC/USDT@spot"
        );
        assert_eq!(resolve_key(&frames, &aliases, "BTCUSDT").unwrap(), "Crypto:BTC/USDT@spot");
        // parse path: canonical form with different spelling
        assert_eq!(
            resolve_key(&frames, &aliases, "Crypto:BTC/USDT").unwrap(),
            "Crypto:BTC/USDT@spot"
        );
        assert_eq!(
            resolve_key(&frames, &aliases, "NOPE").unwrap_err(),
            "strategyV2.dataUnavailable:NOPE"
        );
    }

    #[test]
    fn iso_format_matches_pandas() {
        assert_eq!(iso_naive(1_767_225_600), "2026-01-01T00:00:00");
    }
}
