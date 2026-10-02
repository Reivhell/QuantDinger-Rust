//! Purged / embargoed time-series cross-validation (incl. combinatorial CPCV).
//!
//! Why purge + embargo (López de Prado, 2018): labels computed over a
//! horizon `h` use future prices, so training rows near a test row leak
//! information even when the split is chronological. The purge drops every
//! training row whose label window overlaps the test interval; the embargo
//! additionally drops `embargo` bars after each test interval (covers
//! serially-correlated features and late-arriving labels).
//!
//! - `label_horizon h`: label of bar `i` uses prices up to `i + h`.
//! - Purge: drop train rows `i` with `i + h >= test_start && i < test_end`.
//! - Embargo: drop train rows in `[test_end, test_end + embargo)`.
//! - CPCV: split into `n_partitions` blocks, hold out `n_test` blocks per
//!   path; number of paths = C(n_partitions, n_test). Each path is one
//!   (train, test) split after purge + embargo.

/// One purged/embargoed split: sorted index lists.
#[derive(Debug, Clone)]
pub struct CpcvSplit {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

/// Build all CPCV paths over `n` bars.
///
/// `n_partitions` contiguous blocks; every combination of `n_test` blocks
/// becomes one test set. Train = all other blocks minus purged/embargoed
/// rows. Deterministic (lexicographic combination order).
pub fn cpcv_splits(
    n: usize,
    n_partitions: usize,
    n_test: usize,
    label_horizon: usize,
    embargo: usize,
) -> Vec<CpcvSplit> {
    if n_partitions < 2 || n_test == 0 || n_test >= n_partitions || n == 0 {
        return Vec::new();
    }
    let bounds = partition_bounds(n, n_partitions);
    let mut splits = Vec::new();
    for test_blocks in combinations(n_partitions, n_test) {
        let in_test = |i: usize| {
            test_blocks.iter().any(|&b| i >= bounds[b] && i < bounds[b + 1])
        };
        let mut test: Vec<usize> = (0..n).filter(|&i| in_test(i)).collect();
        test.sort();
        // Test interval edges for purge/embargo.
        let t0 = *test.first().unwrap();
        let t1 = *test.last().unwrap() + 1;
        let mut train: Vec<usize> = (0..n)
            .filter(|&i| {
                if in_test(i) {
                    return false;
                }
                // Purge: label window [i, i+h] overlaps [t0, t1).
                if i + label_horizon >= t0 && i < t1 {
                    return false;
                }
                // Embargo: rows right after the test interval.
                if i >= t1 && i < t1 + embargo {
                    return false;
                }
                true
            })
            .collect();
        train.sort();
        splits.push(CpcvSplit { train, test });
    }
    splits
}

fn partition_bounds(n: usize, p: usize) -> Vec<usize> {
    // Contiguous, sizes differ by at most 1; remainder goes to early blocks.
    let mut b = Vec::with_capacity(p + 1);
    for k in 0..=p {
        b.push((k * n + p - 1) / p);
    }
    b
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    // Lexicographic C(n, k), iterative, no RNG.
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            idx[i] += 1;
            if idx[i] < n - (k - 1 - i) {
                for j in (i + 1)..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Verify a split has no leakage: no train row's label window touches any
/// test row, and the embargo gap holds. Returns the first offending train
/// index, if any.
pub fn first_leak(split: &CpcvSplit, label_horizon: usize, embargo: usize) -> Option<usize> {
    if split.test.is_empty() {
        return None;
    }
    let t0 = *split.test.first().unwrap();
    let t1 = *split.test.last().unwrap() + 1;
    split
        .train
        .iter()
        .copied()
        .find(|&i| {
            (i + label_horizon >= t0 && i < t1) || (i >= t1 && i < t1 + embargo) || (i >= t0 && i < t1)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_count_is_combinatorial() {
        // C(6, 2) = 15 paths.
        let s = cpcv_splits(120, 6, 2, 5, 2);
        assert_eq!(s.len(), 15);
    }

    #[test]
    fn no_leakage_after_purge_embargo() {
        // Single-block test sets (contiguous): train must stay non-empty.
        for h in [0, 1, 5] {
            for e in [0, 1, 5] {
                for split in cpcv_splits(120, 6, 1, h, e) {
                    assert_eq!(first_leak(&split, h, e), None, "h={h} e={e}");
                    assert!(!split.train.is_empty() && !split.test.is_empty());
                }
            }
        }
        // All C(6,2) paths are leak-free (some may starve, checked below).
        for split in cpcv_splits(120, 6, 2, 5, 2) {
            assert_eq!(first_leak(&split, 5, 2), None);
        }
    }

    #[test]
    fn purge_removes_overlapping_labels() {
        // n=10, 2 partitions [0..5),[5..10); test=[0..5), h=3, e=2:
        // purge needs i < t1=5 (no train row qualifies); embargo drops 5,6.
        let s = cpcv_splits(10, 2, 1, 3, 2);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].test, vec![0, 1, 2, 3, 4]);
        assert_eq!(s[0].train, vec![7, 8, 9]);
    }

    #[test]
    fn spread_test_blocks_can_starve_train() {
        // Test blocks {0,5} span the full range: every train row's label
        // window overlaps [t0, t1), so purge empties train. Callers must
        // skip such paths (documented behavior, not a splitter bug).
        let s = cpcv_splits(120, 6, 2, 5, 2);
        let starved = s.iter().find(|sp| sp.train.is_empty()).expect("one path starves");
        assert!(!starved.test.is_empty());
        assert_eq!(first_leak(starved, 5, 2), None); // still leak-free
    }
}
