//! Port of `backend_api_python/app/services/grid/levels.py`.
//!
//! Grid level / cell generation shared with the legacy script.
//!
//! `grid_count` is a **boundary-line** count (the live-grid storage contract);
//! robot templates exposing a cell count convert via
//! `GridBotConfig.grid_line_count` before calling here.

use crate::precise::clean_number;

/// One grid cell between two adjacent levels.
#[derive(Debug, Clone, PartialEq)]
pub struct GridCellSpec {
    pub index: usize,
    pub lower_price: f64,
    pub upper_price: f64,
}

/// Generate `grid_count` boundary lines (arithmetic or geometric).
///
/// Returns `[]` when `upper <= lower`, mirroring the Python helper.
pub fn generate_levels(lower: f64, upper: f64, grid_count: i64, mode: &str) -> Vec<f64> {
    let n = max2(2, grid_count) as usize;
    if !(upper > lower) {
        return vec![];
    }
    let mut levels: Vec<f64> = if mode.trim().to_lowercase() == "geometric" && lower > 0.0 {
        let ratio = (upper / lower).powf(1.0 / (n - 1) as f64);
        (0..n)
            .map(|i| clean_number(lower * ratio.powi(i as i32), 12))
            .collect()
    } else {
        let step = (upper - lower) / (n - 1) as f64;
        (0..n)
            .map(|i| clean_number(lower + step * i as f64, 12))
            .collect()
    };
    if levels.len() >= 2 {
        let last_idx = levels.len() - 1;
        levels[0] = clean_number(lower, 12);
        levels[last_idx] = clean_number(upper, 12);
    }
    levels
}

fn max2(a: i64, b: i64) -> i64 {
    if a > b {
        a
    } else {
        b
    }
}

/// Pair adjacent levels into cells; non-ascending pairs are skipped.
pub fn generate_cells(levels: &[f64]) -> Vec<GridCellSpec> {
    if levels.len() < 2 {
        return vec![];
    }
    let mut out = Vec::new();
    for i in 0..levels.len() - 1 {
        if levels[i + 1] > levels[i] {
            out.push(GridCellSpec {
                index: i,
                lower_price: levels[i],
                upper_price: levels[i + 1],
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_levels_span_bounds() {
        let lv = generate_levels(100.0, 200.0, 5, "arithmetic");
        assert_eq!(lv.len(), 5);
        assert_eq!(lv[0], 100.0);
        assert_eq!(lv[4], 200.0);
        assert_eq!(lv[2], 150.0);
    }

    #[test]
    fn geometric_levels_are_multiplicative() {
        let lv = generate_levels(100.0, 400.0, 3, "geometric");
        assert_eq!(lv.len(), 3);
        assert_eq!(lv[0], 100.0);
        assert_eq!(lv[2], 400.0);
        assert!((lv[1] - 200.0).abs() < 1e-9, "mid={}", lv[1]);
    }

    #[test]
    fn inverted_bounds_yield_empty() {
        assert!(generate_levels(200.0, 100.0, 5, "arithmetic").is_empty());
        assert!(generate_levels(100.0, 100.0, 5, "arithmetic").is_empty());
    }

    #[test]
    fn cells_pair_adjacent_levels() {
        let cells = generate_cells(&[100.0, 150.0, 200.0]);
        assert_eq!(cells.len(), 2);
        assert_eq!(
            cells[0],
            GridCellSpec {
                index: 0,
                lower_price: 100.0,
                upper_price: 150.0,
            }
        );
        assert!(generate_cells(&[100.0]).is_empty());
    }
}
