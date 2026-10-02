//! Port of `backend_api_python/app/services/strategy_v2/curve_sampling.py`.
//!
//! Bounded historical payloads: keep endpoints, global low/high, the peak of
//! the maximum drawdown plus its trough, the most-negative stored drawdown,
//! and the insolvency boundary; fill remaining slots with evenly spaced
//! candidates (`round(i * step)`), mirroring Python banker's rounding.
//!
//! Faithful corners:
//! - `value` coercion mirrors `float(item["value"])` (strings accepted,
//!   unparsable -> point skipped like Python's `ValueError` — here the caller
//!   supplies `f64` with `None` for unparsable, which is skipped the same way
//!   non-finite values are: Python `continue`s on any non-finite value).
//! - `saved` mirrors `float(item.get("drawdown"))`: absent/`None` is ignored;
//!   callers pass `None` for those.
//! - Strict comparisons (`<`/`>`), so ties keep the earliest index.
//! - `round()` is banker's (half-to-even), matching CPython on `.5` steps.
//! - `slots = 0` yields no fill (Python `range(0)`); `slots = 1` takes the
//!   first candidate (Python would raise `ZeroDivisionError` there — the Rust
//!   port degrades gracefully instead). Callers must pass `limit >= 10`
//!   (Python raises `ValueError` below ten).

/// One equity-curve point: display value, optional stored drawdown %.
#[derive(Debug, Clone, PartialEq)]
pub struct CurvePoint {
    pub value: Option<f64>,
    pub drawdown: Option<f64>,
}

impl CurvePoint {
    pub fn new(value: f64, drawdown: Option<f64>) -> Self {
        Self {
            value: Some(value),
            drawdown,
        }
    }
}

/// Python `round()` banker's rounding for non-negative finite inputs.
fn py_round(x: f64) -> usize {
    // round-half-to-even on the fractional part.
    let f = x.floor();
    let frac = x - f;
    let mut r = f as usize;
    if frac > 0.5 || (frac == 0.5 && (r % 2 == 1)) {
        r += 1;
    }
    r
}

/// Downsample to at most `limit` indices (ascending), preserving key risk
/// observations. Returns the kept indices; `Err` when `limit < 10`.
/// Mirrors `sample_equity_curve` (which returns the items themselves).
pub fn sample_equity_curve_indices(
    items: &[CurvePoint],
    limit: usize,
    initial: f64,
) -> Result<Vec<usize>, &'static str> {
    if items.len() <= limit {
        return Ok((0..items.len()).collect());
    }
    if limit < 10 {
        return Err("Equity curve sampling requires at least ten points");
    }
    let mut required = vec![false; items.len()];
    required[0] = true;
    required[items.len() - 1] = true;

    let mut peak = initial;
    let mut peak_index = 0;
    let mut worst = 0.0;
    let mut worst_pair = (0usize, 0usize);
    let mut low = f64::INFINITY;
    let mut low_index = 0;
    let mut high = f64::NEG_INFINITY;
    let mut high_index = 0;
    let mut saved_worst = 0.0;
    let mut saved_index = 0;
    let mut first_insolvent: Option<usize> = None;

    for (index, item) in items.iter().enumerate() {
        let value = match item.value {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        if value < low {
            low = value;
            low_index = index;
        }
        if value > high {
            high = value;
            high_index = index;
        }
        if peak <= 0.0 || value > peak {
            peak = value;
            peak_index = index;
        }
        let drawdown = if peak > 0.0 { value / peak - 1.0 } else { 0.0 } * 100.0;
        if drawdown < worst {
            worst = drawdown;
            worst_pair = (peak_index, index);
        }
        if let Some(saved) = item.drawdown {
            if saved < saved_worst {
                saved_worst = saved;
                saved_index = index;
            }
        }
        if value <= 0.0 && first_insolvent.is_none() {
            first_insolvent = Some(index);
        }
    }

    required[worst_pair.0] = true;
    required[worst_pair.1] = true;
    required[low_index] = true;
    required[high_index] = true;
    required[saved_index] = true;
    if let Some(fi) = first_insolvent {
        required[fi.saturating_sub(1)] = true;
        required[fi] = true;
    }
    let required_count = required.iter().filter(|&&b| b).count();
    let candidates: Vec<usize> = (0..items.len()).filter(|i| !required[*i]).collect();
    let slots = limit.saturating_sub(required_count);
    if slots > 0 && !candidates.is_empty() {
        if slots == 1 {
            required[candidates[0]] = true;
        } else {
            let step = (candidates.len() - 1) as f64 / (slots - 1) as f64;
            for i in 0..slots {
                let pick = candidates[py_round(i as f64 * step).min(candidates.len() - 1)];
                required[pick] = true;
            }
        }
    }
    Ok((0..items.len()).filter(|i| required[*i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(values: &[f64]) -> Vec<CurvePoint> {
        values.iter().map(|v| CurvePoint::new(*v, None)).collect()
    }

    #[test]
    fn short_input_returned_whole_and_limit_guarded() {
        let items = pts(&[1.0, 2.0, 3.0]);
        assert_eq!(sample_equity_curve_indices(&items, 10, 100.0).unwrap(), vec![0, 1, 2]);
        assert!(sample_equity_curve_indices(&pts(&[1.0; 20]), 9, 1.0).is_err());
    }

    #[test]
    fn preserves_extrema_and_drawdown_pair() {
        // rise then crash: peak at 5, trough at 8, low at 8, high at 5.
        let values = [
            100.0, 102.0, 105.0, 110.0, 108.0, 120.0, 115.0, 90.0, 80.0, 85.0,
            88.0, 92.0, 95.0, 97.0, 99.0, 101.0, 103.0, 104.0, 106.0, 107.0,
        ];
        let kept = sample_equity_curve_indices(&pts(&values), 10, 100.0).unwrap();
        assert!(kept.len() <= 10);
        assert!(kept.contains(&0) && kept.contains(&19)); // endpoints
        assert!(kept.contains(&5)); // peak 120
        assert!(kept.contains(&8)); // trough 80 / worst drawdown
        assert!(kept.windows(2).all(|w| w[0] < w[1])); // ascending
    }

    #[test]
    fn insolvency_boundary_kept() {
        let mut values = [100.0f64; 20];
        values[12] = 0.0;
        values[13] = -5.0;
        let kept = sample_equity_curve_indices(&pts(&values), 10, 100.0).unwrap();
        assert!(kept.contains(&11) && kept.contains(&12));
    }

    #[test]
    fn non_finite_values_skipped_like_python_continue() {
        let mut items = pts(&[100.0; 20]);
        items[5] = CurvePoint::new(f64::NAN, None);
        items[6] = CurvePoint::new(f64::INFINITY, None);
        let kept = sample_equity_curve_indices(&items, 10, 100.0).unwrap();
        assert!(!kept.contains(&5) || kept.len() <= 10);
        assert!(kept.contains(&0) && kept.contains(&19));
    }

    #[test]
    fn saved_drawdown_minimum_kept() {
        let mut items = pts(&[100.0; 20]);
        items[15] = CurvePoint::new(100.0, Some(-42.0));
        let kept = sample_equity_curve_indices(&items, 10, 100.0).unwrap();
        assert!(kept.contains(&15));
    }

    #[test]
    fn banker_rounding_matches_python() {
        assert_eq!(py_round(0.5), 0);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(2.6), 3);
    }
}
