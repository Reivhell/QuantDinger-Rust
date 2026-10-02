//! Port of `backend_api_python/app/utils/thread_capacity.py`.
//!
//! [`snapshot`] reads the same cgroup files as Python's `_read_first` (first
//! non-empty wins, missing/unreadable skipped) and [`format_capacity`]
//! mirrors `format_thread_capacity` (`None` renders as `?`).
//!
//! Thread count: Python uses `threading.active_count()`. The std-only port
//! reads `/proc/self/stat` field 20 (`num_threads`), which counts the same
//! live-thread population for the current process. On read failure the count
//! falls back to 1 (the calling thread), never to an error.

use std::fs;

/// Candidate paths, in the same order as Python.
pub const PIDS_CURRENT: &[&str] = &["/sys/fs/cgroup/pids.current", "/sys/fs/cgroup/pids/pids.current"];
pub const PIDS_MAX: &[&str] = &["/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids/pids.max"];
pub const MEMORY_CURRENT: &[&str] = &["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"];
pub const MEMORY_MAX: &[&str] = &["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"];

/// Mirrors `_read_first`: first path whose content is non-empty after trim.
pub fn read_first(paths: &[&str]) -> Option<String> {
    for p in paths {
        match fs::read_to_string(p) {
            Ok(text) => {
                let v = text.trim().to_string();
                if !v.is_empty() {
                    return Some(v);
                }
            }
            Err(_) => continue,
        }
    }
    None
}

/// Live thread count of this process via `/proc/self/stat` field 20.
pub fn active_thread_count() -> u32 {
    fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| parse_num_threads(&s))
        .unwrap_or(1)
}

/// Parse `/proc/self/stat`: `pid (comm) rest...`; `num_threads` is field 20,
/// i.e. the 18th whitespace field after the closing `)` (comm may contain
/// spaces and parens, so split at the *last* `)`).
pub fn parse_num_threads(stat: &str) -> Option<u32> {
    let close = stat.rfind(')')?;
    let mut fields = stat[close + 1..].split_whitespace();
    fields.nth(17)?.parse().ok()
}

/// Mirrors `thread_capacity_snapshot`.
#[derive(Debug, Clone, PartialEq)]
pub struct CapacitySnapshot {
    pub threads: u32,
    pub pids_current: Option<String>,
    pub pids_max: Option<String>,
    pub memory_current: Option<String>,
    pub memory_max: Option<String>,
}

pub fn snapshot() -> CapacitySnapshot {
    CapacitySnapshot {
        threads: active_thread_count(),
        pids_current: read_first(PIDS_CURRENT),
        pids_max: read_first(PIDS_MAX),
        memory_current: read_first(MEMORY_CURRENT),
        memory_max: read_first(MEMORY_MAX),
    }
}

/// Mirrors `format_thread_capacity`.
pub fn format_capacity(s: &CapacitySnapshot) -> String {
    format!(
        "python_threads={}, pids={}/{}, memory={}/{}",
        s.threads,
        s.pids_current.as_deref().unwrap_or("?"),
        s.pids_max.as_deref().unwrap_or("?"),
        s.memory_current.as_deref().unwrap_or("?"),
        s.memory_max.as_deref().unwrap_or("?")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stat_with_tricky_comm() {
        // pid (comm with spaces and parens) + fields; 18th after ')' = 5.
        let stat = "1234 (my app (v2)) R 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 5 6 7";
        assert_eq!(parse_num_threads(stat), Some(5));
        assert_eq!(parse_num_threads("no parens here"), None);
        assert_eq!(parse_num_threads("1 (x) R"), None);
    }

    #[test]
    fn live_count_is_sane() {
        // Under the parallel test harness the process thread count fluctuates
        // as tests start/finish, so only structural checks here; exact
        // equality is verified by the single-threaded parity binary.
        let raw = fs::read_to_string("/proc/self/stat").unwrap();
        let parsed = parse_num_threads(&raw).unwrap();
        assert!(parsed >= 1);
        assert!(active_thread_count() >= 1);
    }

    #[test]
    fn format_renders_missing_as_qmark() {
        let s = CapacitySnapshot {
            threads: 3,
            pids_current: Some("12".into()),
            pids_max: None,
            memory_current: None,
            memory_max: Some("max".into()),
        };
        assert_eq!(format_capacity(&s), "python_threads=3, pids=12/?, memory=?/max");
    }

    #[test]
    fn read_first_skips_missing_and_empty() {
        assert_eq!(read_first(&["/nonexistent-qd-1", "/nonexistent-qd-2"]), None);
        assert_eq!(read_first(&["/proc/self/stat"]).is_some(), true);
    }
}
