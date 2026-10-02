//! Port of `backend_api_python/app/utils/timeutil.py` (`to_utc_iso`).
//!
//! Serializes datetimes/epochs/ISO strings to UTC ISO-8601 with a `Z`
//! suffix. All inputs arrive pre-decoded (no `datetime` objects cross the
//! boundary): aware datetimes as epoch+offset, naive ones as bare epoch.
//!
//! Faithful corners:
//! - `None`/`""` (and whitespace-only strings) → `None`.
//! - Numbers `> 1e12` are milliseconds; `bool` counts as `1.0`/`0.0`
//!   (mirrors `isinstance(True, int)`).
//! - Out-of-range epochs (year outside `1..=9999`, NaN/inf) → `None`
//!   (mirrors `datetime.fromtimestamp` raising).
//! - String preprocessing mirrors the code exactly: trailing `Z` → all
//!   `Z` become `+00:00`; first space becomes `T` when no `T` present.
//! - Naive stamps are assumed to be UTC wall clock (the production default:
//!   the PG pool pins `timezone=UTC`). A non-UTC `DB_NAIVE_TIMESTAMP_TZ`
//!   override needs the tz database and stays Python-side —
//!   [`to_utc_iso_naive`] documents the assumption at the call site.
//! - Output drops sub-second precision by truncation (mirrors
//!   `replace(microsecond=0)`), always `YYYY-MM-DDTHH:MM:SSZ`.
//!
//! `fromisoformat` coverage: `YYYY-MM-DD` with optional `THH:MM[:SS[.frac]]`
//! (space separator accepted), tz `Z` or `±HH[:MM]` / `±HHMM`. Week dates,
//! ordinals, and other ISO shapes Python accepts are NOT ported — they never
//! occur in this codebase's traffic; unknown shapes return `None` exactly
//! like a `fromisoformat` failure.

/// A timestamp input. Mirrors the accepted Python value types.
#[derive(Debug, Clone)]
pub enum TimeInput {
    /// Aware datetime: epoch seconds + fixed UTC offset.
    Aware { epoch: f64, offset: i64 },
    /// Naive datetime (assumed UTC wall clock) / bare epoch seconds.
    Epoch(f64),
    /// Milliseconds are divided out when `> 1e12` — same rule.
    Text(String),
    /// `None`, `""`, and every other type → `None`.
    Other,
}

/// Convert to `YYYY-MM-DDTHH:MM:SSZ`, or `None`. `naive_is_utc` must be
/// `true` unless the caller resolved `DB_NAIVE_TIMESTAMP_TZ` itself;
/// non-UTC naive zones are out of scope (see module docs).
pub fn to_utc_iso(input: &TimeInput, naive_is_utc: bool) -> Option<String> {
    let epoch = match input {
        TimeInput::Aware { epoch, offset } => epoch - *offset as f64,
        TimeInput::Epoch(ts) => normalize_number(*ts)?,
        TimeInput::Text(s) => {
            if s.trim().is_empty() {
                return None;
            }
            parse_iso_to_epoch(s, naive_is_utc)?
        }
        TimeInput::Other => return None,
    };
    format_epoch_utc(epoch)
}

/// `to_utc_iso` with the production assumption (naive = UTC wall clock).
pub fn to_utc_iso_naive(input: &TimeInput) -> Option<String> {
    to_utc_iso(input, true)
}

fn normalize_number(ts: f64) -> Option<f64> {
    if !ts.is_finite() {
        return None;
    }
    Some(if ts > 1e12 { ts / 1000.0 } else { ts })
}

fn parse_iso_to_epoch(s: &str, naive_is_utc: bool) -> Option<f64> {
    let s = s.trim();
    // Mirror the preprocessing verbatim.
    let mut n = if s.ends_with('Z') { s.replace('Z', "+00:00") } else { s.to_string() };
    if n.contains(' ') && !n.contains('T') {
        n = n.replacen(' ', "T", 1);
    }
    let (date_part, rest) = match n.split_once('T') {
        Some((d, r)) => (d, Some(r)),
        None => (n.as_str(), None),
    };
    let (y, mo, d) = parse_date(date_part)?;
    let (hh, mi, ss, frac_ns, offset) = match rest {
        None => (0, 0, 0, 0i64, None),
        Some(t) => parse_time(t)?,
    };
    if !naive_is_utc && offset.is_none() {
        return None; // caller must resolve the naive zone itself
    }
    let days = days_from_civil(y, mo, d)?;
    let secs = days * 86_400 + hh as i64 * 3_600 + mi as i64 * 60 + ss as i64
        - offset.unwrap_or(0);
    Some(secs as f64 + frac_ns as f64 / 1e9)
}

fn parse_date(s: &str) -> Option<(i32, u8, u8)> {
    let mut p = s.split('-');
    let (y, mo, d) = (p.next()?, p.next()?, p.next()?);
    if p.next().is_some() || y.len() != 4 {
        return None;
    }
    let (y, mo, d): (i32, u8, u8) = (y.parse().ok()?, mo.parse().ok()?, d.parse().ok()?);
    if !(1..=9999).contains(&y) || !(1..=12).contains(&mo) {
        return None;
    }
    if d < 1 || d > days_in_month(y, mo) {
        return None;
    }
    Some((y, mo, d))
}

fn parse_time(s: &str) -> Option<(u8, u8, u8, i64, Option<i64>)> {
    // Split trailing tz: Z (already rewritten) or ±HH[:MM]/±HHMM.
    let (core, offset) = split_offset(s)?;
    let mut p = core.split(':');
    let (h, m) = (p.next()?, p.next().unwrap_or("00"));
    let (sec_part, third) = (p.next(), p.next());
    if third.is_some() {
        return None;
    }
    let (sec_str, frac_ns) = match sec_part {
        None => ("00", 0),
        Some(v) => {
            let (a, b) = match v.split_once(['.', ',']) {
                Some((a, b)) => (a, Some(b)),
                None => (v, None),
            };
            let mut ns: i64 = 0;
            if let Some(f) = b {
                if f.is_empty() || f.len() > 9 || !f.bytes().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                let mut s = f.to_string();
                while s.len() < 9 {
                    s.push('0');
                }
                ns = s.parse().ok()?;
            }
            (a, ns)
        }
    };
    let (h, m, sec): (u8, u8, u8) = (h.parse().ok()?, m.parse().ok()?, sec_str.parse().ok()?);
    if h > 23 || m > 59 || sec > 59 {
        return None;
    }
    Some((h, m, sec, frac_ns, offset))
}

fn split_offset(s: &str) -> Option<(&str, Option<i64>)> {
    for (i, c) in s.char_indices().rev() {
        if c == '+' || c == '-' {
            let body = &s[i + 1..];
            if body.is_empty()
                || !body.bytes().all(|b| b.is_ascii_digit() || b == b':')
            {
                return None;
            }
            let off = match body.len() {
                2 => body.parse::<i64>().ok()? * 3_600,
                4 => {
                    body[..2].parse::<i64>().ok()? * 3_600
                        + body[2..].parse::<i64>().ok()? * 60
                }
                _ => {
                    // HH:MM with colon
                    let (h, m) = body.split_once(':')?;
                    if h.len() != 2 || m.len() != 2 {
                        return None;
                    }
                    h.parse::<i64>().ok()? * 3_600 + m.parse::<i64>().ok()? * 60
                }
            };
            if off.abs() >= 24 * 3_600 {
                return None;
            }
            let signed = if c == '-' { -off } else { off };
            return Some((&s[..i], Some(signed)));
        } else if !c.is_ascii_digit() && c != '.' && c != ',' && c != ':' {
            return None;
        }
    }
    Some((s, None))
}

fn days_in_month(y: i32, m: u8) -> u8 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        _ => 28,
    }
}

/// Days since 1970-01-01, or `None` outside year `1..=9999`.
fn days_from_civil(y: i32, m: u8, d: u8) -> Option<i64> {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = ((m as i64 + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// Format epoch seconds as UTC `Z` string; `None` when out of range.
fn format_epoch_utc(epoch: f64) -> Option<String> {
    if !epoch.is_finite() {
        return None;
    }
    let secs = epoch.floor() as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    if !(0..3_652_059).contains(&z) {
        return None; // year 1..=9999
    }
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    y += if m <= 2 { 1 } else { 0 };
    Some(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3_600,
        (rem % 3_600) / 60,
        rem % 60
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_and_millis() {
        assert_eq!(to_utc_iso_naive(&TimeInput::Epoch(0.0)), Some("1970-01-01T00:00:00Z".into()));
        assert_eq!(
            to_utc_iso_naive(&TimeInput::Epoch(1_767_225_600.0)),
            Some("2026-01-01T00:00:00Z".into())
        );
        assert_eq!(
            to_utc_iso_naive(&TimeInput::Epoch(1_767_225_600_123.0)),
            Some("2026-01-01T00:00:00Z".into())
        );
        assert_eq!(to_utc_iso_naive(&TimeInput::Epoch(f64::NAN)), None);
        assert_eq!(to_utc_iso_naive(&TimeInput::Epoch(1e20)), None);
    }

    #[test]
    fn aware_shifts_by_offset() {
        let t = TimeInput::Aware { epoch: 1_767_225_600.0, offset: 8 * 3_600 };
        assert_eq!(to_utc_iso_naive(&t), Some("2025-12-31T16:00:00Z".into()));
    }

    #[test]
    fn strings_cover_common_shapes() {
        for (inp, out) in [
            ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            ("2026-01-01 00:00:00", "2026-01-01T00:00:00Z"),
            ("2026-01-01", "2026-01-01T00:00:00Z"),
            ("2026-01-01T08:00:00+08:00", "2026-01-01T00:00:00Z"),
            ("2026-01-01T08:00:00+0800", "2026-01-01T00:00:00Z"),
            ("2026-01-01T00:00:00.123456", "2026-01-01T00:00:00Z"),
            ("  2026-01-01T00:00:00Z  ", "2026-01-01T00:00:00Z"),
        ] {
            assert_eq!(to_utc_iso_naive(&TimeInput::Text(inp.into())), Some(out.into()), "{inp}");
        }
        for bad in ["", "   ", "not-a-date", "2026-13-01", "2026-01-01T25:00:00", "2026-01-01T00:00:00+99:99"] {
            assert_eq!(to_utc_iso_naive(&TimeInput::Text(bad.into())), None, "{bad}");
        }
        assert_eq!(to_utc_iso_naive(&TimeInput::Other), None);
    }
}
