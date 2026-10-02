//! Port of the pure-math core of
//! `backend_api_python/app/services/backtest/metrics.py`.
//!
//! Benchmark-relative performance: information ratio from levels observed at
//! common timestamps, band classification, benchmark level extraction.
//!
//! Deliberately NOT ported (stay Python, need pandas/exchange calendars):
//! - `_valid_interval_mask` equity-session path (`exchange_calendars` +
//!   `equity_bar_session_date`); the Crypto + exact-delta paths ARE ported.
//! - `pd.to_datetime` timestamp coercion; callers pass epoch seconds (`i64`)
//!   with unparseable points already dropped.
//!
//! Faithful corners:
//! - Levels keep only finite values `> 0`; duplicate timestamps keep the LAST
//!   (mirrors `duplicated(keep="last")`), then sort by time.
//! - Inner join on common timestamps, `pct_change` needs ≥2 levels; a return
//!   row is dropped when either leg is non-finite (mirrors the
//!   `math.isfinite` row filter).
//! - Exact-delta interval check first; Crypto/`Cryptocurrency` markets reject
//!   any non-exact gap; `1w` allows 4–10 days; other markets need the
//!   calendar path → here reported as `NeedsCalendar` (caller decides;
//!   Python returns `False` when the calendar is unavailable).
//! - Sample std (`ddof=1`); zero-tracking-error threshold `abs_tol=1e-15`
//!   with `rel_tol=0` (mirrors `math.isclose(..., rel_tol=0.0, abs_tol=1e-15)`).
//! - Bands: `(limit, label)` pairs, limits `f64`, labels trimmed; empty /
//!   NaN-limit / empty-label / non-increasing → `Err` with the exact Python
//!   messages.

/// Default bands. Mirrors `DEFAULT_INFORMATION_RATIO_BANDS`.
pub const DEFAULT_BANDS: &[(f64, &str)] = &[
    (0.30, "weak"),
    (0.50, "acceptable"),
    (1.00, "good"),
    (f64::INFINITY, "exceptional"),
];

/// One level observation: epoch seconds + level value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelPoint {
    pub epoch_secs: i64,
    pub value: f64,
}

/// Outcome of [`calculate_information_ratio`].
#[derive(Debug, Clone, PartialEq)]
pub struct InformationRatio {
    pub status: &'static str,
    pub portfolio_return_annualized: Option<f64>,
    pub benchmark_return_annualized: Option<f64>,
    pub active_return_annualized: Option<f64>,
    pub tracking_error_annualized: Option<f64>,
    pub information_ratio: Option<f64>,
    pub classification: Option<String>,
    pub observations: usize,
    pub frequency: String,
    pub annualization_factor: f64,
}

/// Interval validity that needs the exchange-calendar path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalVerdict {
    Valid,
    Invalid,
    NeedsCalendar,
}

/// Bar seconds table. Mirrors `_FREQUENCY_SECONDS`.
pub fn frequency_seconds(frequency: &str) -> i64 {
    match frequency {
        "1m" => 60,
        "3m" => 180,
        "5m" => 300,
        "15m" => 900,
        "30m" => 1800,
        "1h" => 3600,
        "4h" => 14_400,
        "1d" => 86_400,
        "1w" => 604_800,
        _ => 86_400,
    }
}

/// Mirrors metrics-local `_normalize_frequency` (smaller alias set than the
/// strategy_v2 one — kept separate on purpose).
pub fn normalize_frequency(value: &str) -> String {
    let n = value.trim().to_lowercase();
    let n = if n.is_empty() { "1d".to_string() } else { n };
    match n.as_str() {
        "daily" | "day" | "d" => "1d".to_string(),
        "weekly" | "week" | "w" => "1w".to_string(),
        _ => n,
    }
}

/// Validate classification bands. `Err` carries the exact Python messages.
pub fn validate_bands(bands: &[(f64, &str)]) -> Result<Vec<(f64, String)>, String> {
    if bands.is_empty() {
        return Err("classification_bands must contain labelled upper bounds".to_string());
    }
    let mut out = Vec::with_capacity(bands.len());
    for (limit, label) in bands {
        let l = label.trim().to_string();
        if l.is_empty() || limit.is_nan() {
            return Err("classification_bands must contain labelled upper bounds".to_string());
        }
        out.push((*limit, l));
    }
    for w in out.windows(2) {
        if w[1].0 <= w[0].0 {
            return Err(
                "classification band upper bounds must be strictly increasing".to_string(),
            );
        }
    }
    Ok(out)
}

/// First band whose upper bound exceeds `value`. Mirrors
/// `_classify_information_ratio` (Python returns `None` when no band matches —
/// only possible with a custom finite-last-bound set; here that is `""`).
pub fn classify_information_ratio(value: f64, bands: &[(f64, String)]) -> String {
    for (upper, label) in bands {
        if value < *upper {
            return label.clone();
        }
    }
    String::new()
}

/// Dedupe (keep last) + sort + drop non-positive/non-finite.
/// Mirrors `_curve_levels` minus timestamp parsing.
pub fn clean_levels(points: &[LevelPoint]) -> Vec<LevelPoint> {
    use std::collections::HashMap;
    let mut last: HashMap<i64, (usize, f64)> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        if p.value.is_finite() && p.value > 0.0 {
            last.insert(p.epoch_secs, (i, p.value));
        }
    }
    let mut order: Vec<(usize, LevelPoint)> = last
        .into_iter()
        .map(|(t, (i, v))| (i, LevelPoint { epoch_secs: t, value: v }))
        .collect();
    // Python: dedupe keeps last occurrence, then sort_index (stable on time).
    order.sort_by_key(|(_, p)| p.epoch_secs);
    order.into_iter().map(|(_, p)| p).collect()
}

/// Interval check for one gap. Mirrors `_valid_interval_mask` per-pair logic.
pub fn check_interval(
    start_secs: i64,
    end_secs: i64,
    frequency: &str,
    market: &str,
) -> IntervalVerdict {
    let expected = frequency_seconds(frequency);
    let delta = end_secs - start_secs;
    if delta == expected {
        return IntervalVerdict::Valid;
    }
    if matches!(market.trim(), "Crypto" | "Cryptocurrency") {
        return IntervalVerdict::Invalid;
    }
    if frequency == "1w" {
        let four = 4 * 86_400;
        let ten = 10 * 86_400;
        return if (four..=ten).contains(&delta) {
            IntervalVerdict::Valid
        } else {
            IntervalVerdict::Invalid
        };
    }
    IntervalVerdict::NeedsCalendar
}

/// Core IR computation over epoch-second level points.
/// `calendar_ok` resolves `NeedsCalendar` verdicts (ask the exchange calendar
/// in Python; pass the boolean here).
/// `bands` defaults to [`DEFAULT_BANDS`] when `None`.
pub fn calculate_information_ratio(
    portfolio_curve: &[LevelPoint],
    benchmark_curve: &[LevelPoint],
    frequency: &str,
    annualization_factor: f64,
    market: &str,
    bands: Option<&[(f64, &str)]>,
    calendar_ok: &dyn Fn(i64, i64) -> bool,
) -> Result<InformationRatio, String> {
    let normalized = normalize_frequency(frequency);
    if !annualization_factor.is_finite() || annualization_factor <= 0.0 {
        return Err("annualization_factor must be a positive finite number".to_string());
    }
    let bands_src: Vec<(f64, &str)> = match bands {
        Some(b) => b.to_vec(),
        None => DEFAULT_BANDS.to_vec(),
    };
    let bands = validate_bands(&bands_src)?;

    let pf = clean_levels(portfolio_curve);
    let bm = clean_levels(benchmark_curve);
    use std::collections::HashMap;
    let pf_map: HashMap<i64, f64> = pf.iter().map(|p| (p.epoch_secs, p.value)).collect();
    let bm_map: HashMap<i64, f64> = bm.iter().map(|p| (p.epoch_secs, p.value)).collect();
    let mut common: Vec<i64> = pf_map.keys().filter(|t| bm_map.contains_key(*t)).copied().collect();
    common.sort_unstable();

    // pct_change + dropna + interval mask + finite-row filter.
    let mut active: Vec<f64> = Vec::new();
    let mut pf_rets: Vec<f64> = Vec::new();
    let mut bm_rets: Vec<f64> = Vec::new();
    for w in common.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let (p0, p1) = (pf_map[&t0], pf_map[&t1]);
        let (b0, b1) = (bm_map[&t0], bm_map[&t1]);
        if p0 == 0.0 || b0 == 0.0 {
            continue; // pct_change would be inf — pandas keeps inf but the
                      // finite-row filter below drops it all the same.
        }
        let pr = p1 / p0 - 1.0;
        let br = b1 / b0 - 1.0;
        if !pr.is_finite() || !br.is_finite() {
            continue;
        }
        match check_interval(t0, t1, &normalized, market) {
            IntervalVerdict::Invalid => continue,
            IntervalVerdict::NeedsCalendar => {
                if !calendar_ok(t0, t1) {
                    continue;
                }
            }
            IntervalVerdict::Valid => {}
        }
        pf_rets.push(pr);
        bm_rets.push(br);
        active.push(pr - br);
    }

    let observations = active.len();
    let mut result = InformationRatio {
        status: "insufficient_history",
        portfolio_return_annualized: None,
        benchmark_return_annualized: None,
        active_return_annualized: None,
        tracking_error_annualized: None,
        information_ratio: None,
        classification: None,
        observations,
        frequency: normalized,
        annualization_factor,
    };
    if observations == 0 {
        return Ok(result);
    }
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    result.portfolio_return_annualized = Some(mean(&pf_rets) * annualization_factor);
    result.benchmark_return_annualized = Some(mean(&bm_rets) * annualization_factor);
    let active_ann = mean(&active) * annualization_factor;
    result.active_return_annualized = Some(active_ann);
    if observations < 2 {
        return Ok(result);
    }
    let m = mean(&active);
    let var = active.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (active.len() - 1) as f64;
    let periodic_te = var.sqrt();
    let te_ann = periodic_te * annualization_factor.sqrt();
    result.tracking_error_annualized = Some(te_ann);
    if periodic_te <= 1e-15 {
        // math.isclose(te, 0.0, rel_tol=0.0, abs_tol=1e-15)
        result.status = "zero_tracking_error";
        return Ok(result);
    }
    result.status = "available";
    result.information_ratio = Some(active_ann / te_ann);
    result.classification = Some(classify_information_ratio(active_ann / te_ann, &bands));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(vals: &[(i64, f64)]) -> Vec<LevelPoint> {
        vals.iter().map(|(t, v)| LevelPoint { epoch_secs: *t, value: *v }).collect()
    }

    const DAY: i64 = 86_400;

    #[test]
    fn empty_curves_give_insufficient_history() {
        let r = calculate_information_ratio(&[], &[], "1d", 252.0, "Stocks", None, &|_, _| true)
            .unwrap();
        assert_eq!(r.status, "insufficient_history");
        assert_eq!(r.observations, 0);
    }

    #[test]
    fn identical_curves_give_zero_tracking_error() {
        let pts: Vec<(i64, f64)> = (0..5).map(|i| (i * DAY, 100.0 + i as f64)).collect();
        let c = curve(&pts);
        let r = calculate_information_ratio(&c, &c, "1d", 252.0, "Crypto", None, &|_, _| true)
            .unwrap();
        assert_eq!(r.observations, 4);
        assert_eq!(r.status, "zero_tracking_error");
        assert!(r.portfolio_return_annualized.is_some());
        assert!(r.information_ratio.is_none());
    }

    #[test]
    fn single_observation_returns_means_only() {
        let pf = curve(&[(0, 100.0), (DAY, 101.0)]);
        let bm = curve(&[(0, 100.0), (DAY, 100.5)]);
        let r = calculate_information_ratio(&pf, &bm, "1d", 252.0, "Crypto", None, &|_, _| true)
            .unwrap();
        assert_eq!(r.observations, 1);
        assert_eq!(r.status, "insufficient_history");
        assert!(r.active_return_annualized.is_some());
        assert!(r.tracking_error_annualized.is_none());
    }

    #[test]
    fn available_ratio_matches_hand_computation() {
        // 4 daily points -> 3 returns; active = pf - bm.
        let pf = curve(&[(0, 100.0), (DAY, 102.0), (2 * DAY, 101.0), (3 * DAY, 103.0)]);
        let bm = curve(&[(0, 100.0), (DAY, 101.0), (2 * DAY, 100.5), (3 * DAY, 101.0)]);
        let r = calculate_information_ratio(&pf, &bm, "1d", 252.0, "Crypto", None, &|_, _| true)
            .unwrap();
        assert_eq!(r.status, "available");
        let pr = [0.02, 101.0 / 102.0 - 1.0, 103.0 / 101.0 - 1.0];
        let br = [0.01, 100.5 / 101.0 - 1.0, 101.0 / 100.5 - 1.0];
        let act: Vec<f64> = pr.iter().zip(&br).map(|(a, b)| a - b).collect();
        let m = act.iter().sum::<f64>() / 3.0;
        let sd = (act.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 2.0).sqrt();
        let expected = m * 252.0 / (sd * 252.0_f64.sqrt());
        assert!((r.information_ratio.unwrap() - expected).abs() < 1e-12);
        assert!(r.classification.is_some());
    }

    #[test]
    fn crypto_rejects_gapped_intervals() {
        // skip one day: 2d gap is not a valid 1d interval for Crypto.
        let pf = curve(&[(0, 100.0), (DAY, 101.0), (3 * DAY, 102.0)]);
        let bm = curve(&[(0, 100.0), (DAY, 100.5), (3 * DAY, 101.0)]);
        let r = calculate_information_ratio(&pf, &bm, "1d", 252.0, "Crypto", None, &|_, _| true)
            .unwrap();
        assert_eq!(r.observations, 1); // only the exact-delta return survives
    }

    #[test]
    fn inner_join_uses_common_timestamps() {
        let pf = curve(&[(0, 100.0), (DAY, 101.0), (2 * DAY, 102.0)]);
        let bm = curve(&[(DAY, 200.0), (2 * DAY, 202.0), (3 * DAY, 203.0)]);
        let r = calculate_information_ratio(&pf, &bm, "1d", 252.0, "Crypto", None, &|_, _| true)
            .unwrap();
        assert_eq!(r.observations, 1);
    }

    #[test]
    fn bad_factor_and_bands_err_with_python_messages() {
        let c = curve(&[(0, 100.0), (DAY, 101.0)]);
        assert_eq!(
            calculate_information_ratio(&c, &c, "1d", 0.0, "", None, &|_, _| true)
                .unwrap_err(),
            "annualization_factor must be a positive finite number"
        );
        assert_eq!(
            calculate_information_ratio(&c, &c, "1d", f64::NAN, "", None, &|_, _| true)
                .unwrap_err(),
            "annualization_factor must be a positive finite number"
        );
        assert_eq!(
            validate_bands(&[]).unwrap_err(),
            "classification_bands must contain labelled upper bounds"
        );
        assert_eq!(
            validate_bands(&[(0.5, "a"), (0.5, "b")]).unwrap_err(),
            "classification band upper bounds must be strictly increasing"
        );
        assert_eq!(
            validate_bands(&[(f64::NAN, "a")]).unwrap_err(),
            "classification_bands must contain labelled upper bounds"
        );
    }

    #[test]
    fn classification_thresholds() {
        let b = validate_bands(DEFAULT_BANDS).unwrap();
        assert_eq!(classify_information_ratio(0.1, &b), "weak");
        assert_eq!(classify_information_ratio(0.3, &b), "acceptable");
        assert_eq!(classify_information_ratio(0.99, &b), "good");
        assert_eq!(classify_information_ratio(5.0, &b), "exceptional");
    }

    #[test]
    fn dupes_keep_last_and_bad_levels_dropped() {
        let pts = curve(&[(0, 100.0), (0, 110.0), (DAY, -5.0), (2 * DAY, 112.0)]);
        let cleaned = clean_levels(&pts);
        assert_eq!(cleaned, curve(&[(0, 110.0), (2 * DAY, 112.0)]));
    }

    #[test]
    fn interval_verdicts() {
        assert_eq!(check_interval(0, DAY, "1d", "Crypto"), IntervalVerdict::Valid);
        assert_eq!(check_interval(0, 2 * DAY, "1d", "Crypto"), IntervalVerdict::Invalid);
        assert_eq!(
            check_interval(0, 2 * DAY, "1d", "Cryptocurrency"),
            IntervalVerdict::Invalid
        );
        assert_eq!(
            check_interval(0, 7 * DAY, "1w", "Stocks"),
            IntervalVerdict::Valid
        );
        assert_eq!(
            check_interval(0, 2 * DAY, "1d", "Stocks"),
            IntervalVerdict::NeedsCalendar
        );
        assert_eq!(normalize_frequency("daily"), "1d");
        assert_eq!(normalize_frequency(" 4H "), "4h");
    }
}
