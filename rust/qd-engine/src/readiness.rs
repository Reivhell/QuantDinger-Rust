//! Port of `backend_api_python/app/services/strategy_v2/readiness.py`.
//!
//! Point-in-time gates evaluated before a strategy runs: universe-history
//! availability, per-symbol warmup-bar counts, and fundamental-field
//! presence. All pandas indexing is replaced by epoch-second arithmetic;
//! every error string and truncation rule is preserved.
//!
//! Timestamps arrive as `(epoch_secs, utc_offset_secs)`: `None` offset means
//! the stamp was tz-naive (its `.date()` is the UTC-calendar date of the
//! epoch, exactly like `pd.Timestamp("2026-01-01").date()`). Aware stamps
//! carry their fixed UTC offset so `.date()` resolves in the stamp's own
//! zone. (DST zones: the caller passes the applicable offset.)
//!
//! Faithful corners:
//! - `universeHistoryUnavailable` renders `earliest` RAW (the original
//!   string), and `code` via `universe.get('code')` (missing → `None`).
//! - Warmup counts bars with `index < first_active` (strict) where all four
//!   OHLC values are numeric and `> 0` (`to_numeric(coerce)` → NaN fails
//!   both `notna` and `> 0`; `None` here = NaN/coerced).
//! - `joined` maps member `key → valid_from`; only truthy `valid_from`
//!   values participate (`if joined.get(symbol)`).
//! - Problems across frequencies/symbols join with `;`, capped at 6.
//! - Fundamentals: `±inf` counts as missing; absent field = empty series;
//!   `as_of` filters rows to `index <= as_of`; fields iterate `sorted`.

/// A timestamp with its zone: epoch seconds + fixed UTC offset (0 = naive).
#[derive(Debug, Clone, Copy)]
pub struct Zoned {
    pub epoch: i64,
    pub offset: i64,
}

impl Zoned {
    pub fn utc(epoch: i64) -> Self {
        Self { epoch, offset: 0 }
    }
}

/// Calendar date (y, m, d) of a zoned stamp in its own zone.
/// Mirrors `pd.Timestamp(...).date()`.
pub fn cal_date(z: Zoned) -> (i32, u8, u8) {
    let days = (z.epoch + z.offset).div_euclid(86_400);
    let z2 = days + 719_468;
    let era = z2.div_euclid(146_097);
    let doe = z2.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    y += if m <= 2 { 1 } else { 0 };
    (y as i32, m as u8, d as u8)
}

/// Mirrors `validate_universe_history`. `earliest_raw`: the original
/// `history_from or snapshot_as_of` string (`None` = missing/falsy).
/// `code`: `universe.get('code')` rendered Python-style (`None` when absent).
pub fn validate_universe_history(
    code: Option<&str>,
    history_from: Option<Zoned>,
    snapshot_as_of: Option<Zoned>,
    snapshot_only: bool,
    earliest_raw: Option<&str>,
    start: Zoned,
) -> Result<(), String> {
    let earliest = match (history_from, snapshot_as_of) {
        (Some(z), _) => Some(z),
        (None, Some(z)) => Some(z),
        (None, None) => None,
    };
    if snapshot_only {
        if let (Some(e), Some(raw)) = (earliest, earliest_raw) {
            if cal_date(start) < cal_date(e) {
                return Err(format!(
                    "strategyV2.universeHistoryUnavailable:{}:{raw}",
                    code.unwrap_or("None")
                ));
            }
        }
    }
    Ok(())
}

/// One frame for warmup counting: bar stamps + OHLC where `None` = NaN.
#[derive(Debug, Clone)]
pub struct WarmupFrame {
    pub ts: Vec<i64>,
    pub open: Vec<Option<f64>>,
    pub high: Vec<Option<f64>>,
    pub low: Vec<Option<f64>>,
    pub close: Vec<Option<f64>>,
}

/// A universe member: join key + optional `valid_from` (epoch, UTC).
#[derive(Debug, Clone)]
pub struct Member {
    pub key: String,
    pub valid_from: Option<i64>,
}

/// Mirrors `validate_warmup`. `frames` preserves caller order
/// (`Vec<(frequency, Vec<(symbol, frame)>)>`).
pub fn validate_warmup(
    frames: &[(String, Vec<(String, WarmupFrame)>)],
    warmup_bars: i64,
    start_utc: i64,
    members: &[Member],
) -> Result<(), String> {
    if warmup_bars <= 0 {
        return Ok(());
    }
    let mut problems = Vec::new();
    for (frequency, symbols) in frames {
        for (symbol, frame) in symbols {
            let mut first_active = start_utc;
            if let Some(m) = members.iter().find(|m| m.key == *symbol) {
                if let Some(vf) = m.valid_from {
                    first_active = first_active.max(vf);
                }
            }
            let n = frame.ts.len();
            let mut count = 0;
            for i in 0..n {
                if frame.ts[i] >= first_active {
                    continue;
                }
                let row = [&frame.open, &frame.high, &frame.low, &frame.close];
                if row.iter().all(|col| matches!(col.get(i), Some(Some(v)) if *v > 0.0)) {
                    count += 1;
                }
            }
            if count < warmup_bars {
                problems.push(format!("{symbol}@{frequency}={count}/{warmup_bars}"));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("strategyV2.insufficientWarmupData:{}", problems[..problems.len().min(6)].join(";")))
    }
}

/// One frame for fundamental checks: stamps + named columns.
#[derive(Debug, Clone, Default)]
pub struct FundFrame {
    pub ts: Vec<i64>,
    pub cols: std::collections::HashMap<String, Vec<Option<f64>>>,
}

/// Mirrors `validate_fundamentals`. `required` iterates sorted.
pub fn validate_fundamentals(
    frames: &[(String, FundFrame)],
    required: &std::collections::HashSet<String>,
    as_of: Option<i64>,
) -> Result<(), String> {
    let mut fields: Vec<&String> = required.iter().collect();
    fields.sort();
    let mut problems = Vec::new();
    for (symbol, frame) in frames {
        let upto = match as_of {
            None => frame.ts.len(),
            Some(t) => frame.ts.partition_point(|&ts| ts <= t),
        };
        for field in &fields {
            let ok = match frame.cols.get(*field) {
                None => false,
                Some(col) => col.iter().take(upto).any(|v| matches!(v, Some(x) if x.is_finite())),
            };
            if !ok {
                problems.push(format!("{symbol}/{field}"));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("strategyV2.fundamentalDataMissing:{}", problems[..problems.len().min(6)].join(";")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    const D: i64 = 86_400;
    const B: i64 = 1_767_225_600; // 2026-01-01 00:00 UTC

    fn wframe(n: usize, bad_at: Option<usize>) -> WarmupFrame {
        WarmupFrame {
            ts: (0..n as i64).map(|i| B + i * D).collect(),
            open: (0..n).map(|i| if Some(i) == bad_at { Some(0.0) } else { Some(100.0) }).collect(),
            high: (0..n).map(|_| Some(101.0)).collect(),
            low: (0..n).map(|_| Some(99.0)).collect(),
            close: (0..n).map(|i| if Some(i) == bad_at { None } else { Some(100.5) }).collect(),
        }
    }

    #[test]
    fn universe_gate_fires_only_for_snapshot_only() {
        let start = Zoned::utc(B - D); // day before earliest
        let err = validate_universe_history(
            Some("U1"), Some(Zoned::utc(B)), None, true, Some("2026-01-01"), start,
        )
        .unwrap_err();
        assert_eq!(err, "strategyV2.universeHistoryUnavailable:U1:2026-01-01");
        // not snapshot_only → no error
        assert!(validate_universe_history(Some("U1"), Some(Zoned::utc(B)), None, false, Some("2026-01-01"), start).is_ok());
        // start on the day → ok
        assert!(validate_universe_history(Some("U1"), Some(Zoned::utc(B)), None, true, Some("2026-01-01"), Zoned::utc(B)).is_ok());
        // missing code renders None
        let err = validate_universe_history(None, Some(Zoned::utc(B)), None, true, Some("raw"), start).unwrap_err();
        assert_eq!(err, "strategyV2.universeHistoryUnavailable:None:raw");
    }

    #[test]
    fn warmup_counts_strictly_before_and_requires_positive() {
        let frames = vec![("1d".to_string(), vec![("S".to_string(), wframe(5, Some(1)))])];
        // 4 usable before B+5D (bar 1 has 0.0/None) → warmup 5 fails
        let err = validate_warmup(&frames, 5, B + 5 * D, &[]).unwrap_err();
        assert_eq!(err, "strategyV2.insufficientWarmupData:S@1d=4/5");
        assert!(validate_warmup(&frames, 4, B + 5 * D, &[]).is_ok());
        assert!(validate_warmup(&frames, 99, B, &[]).unwrap_err().contains("=0/99"));
        assert!(validate_warmup(&frames, 0, B, &[]).is_ok());
    }

    #[test]
    fn warmup_respects_member_valid_from() {
        let frames = vec![("1d".to_string(), vec![("S".to_string(), wframe(5, None))])];
        let members = vec![Member { key: "S".to_string(), valid_from: Some(B + 3 * D) }];
        // only bars before B+3D count → 3 < 5
        let err = validate_warmup(&frames, 5, B, &members).unwrap_err();
        assert_eq!(err, "strategyV2.insufficientWarmupData:S@1d=3/5");
    }

    #[test]
    fn fundamentals_flag_missing_inf_and_absent() {
        let mut cols = HashMap::new();
        cols.insert("pe".to_string(), vec![Some(10.0), Some(f64::INFINITY), None]);
        cols.insert("pb".to_string(), vec![None, None, None]);
        let frames = vec![("S".to_string(), FundFrame { ts: vec![B, B + D, B + 2 * D], cols })];
        let req: HashSet<String> = ["pe".into(), "pb".into(), "roe".into()].into_iter().collect();
        let err = validate_fundamentals(&frames, &req, None).unwrap_err();
        assert_eq!(err, "strategyV2.fundamentalDataMissing:S/pb;S/roe");
        // as_of before the first bar cuts everything → pe missing too
        let err = validate_fundamentals(&frames, &req, Some(B - D)).unwrap_err();
        assert!(err.contains("S/pe"));
    }

    #[test]
    fn cal_date_respects_offset() {
        // 2026-01-01 00:30 UTC+2 → still 2026-01-01 locally... use 23:00 UTC, +2 → next day
        let z = Zoned { epoch: B - 3_600, offset: 7_200 };
        assert_eq!(cal_date(z), (2026, 1, 1));
        assert_eq!(cal_date(Zoned::utc(B)), (2026, 1, 1));
    }
}
