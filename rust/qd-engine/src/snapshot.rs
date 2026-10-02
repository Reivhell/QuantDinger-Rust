//! Port of the content-addressing core of
//! `backend_api_python/app/services/strategy_v2/snapshot.py`.
//!
//! A snapshot is the canonical JSON bytes of an OHLCV frame; its SHA-256 hex
//! is the snapshot id. Save/load stay Python-side (gzip file layout +
//! pandas); what is ported is everything needed to reproduce or verify an
//! id byte-for-byte: column selection, stable sort, keep-last dedupe,
//! Python-`repr` float rendering, compact separators, and id validation.
//!
//! Faithful corners:
//! - Columns = `[c for c in SNAPSHOT_COLUMNS if c in frame]` —
//!   `SNAPSHOT_COLUMNS` order, not the frame's.
//! - `sort_index(kind="stable")` then `duplicated(keep="last")`.
//! - `json.dumps(..., allow_nan=False)`: non-finite float → `Err`
//!   (mirrors the `ValueError`), never `null`/`Infinity`.
//! - `save` writes only when the target is absent and returns the fixed
//!   `snapshotFormat` tag; `load` lowercases/strips the id, demands 64
//!   lowercase hex chars (`strategyV2.snapshotIdInvalid`), then demands the
//!   content hash match (`strategyV2.snapshotHashMismatch`).

use std::collections::HashMap;

/// Canonical column order. Mirrors `SNAPSHOT_COLUMNS`.
pub const SNAPSHOT_COLUMNS: &[&str] = &["open", "high", "low", "close", "volume"];

/// Format tag returned by `save`. Mirrors the Python literal.
pub const SNAPSHOT_FORMAT: &str = "strategy-v2-ohlcv-json-gzip-v1";

/// One normalized row: nanosecond timestamp + one value per selected column.
#[derive(Debug, Clone)]
pub struct SnapshotRow {
    pub ns: i64,
    pub values: Vec<f64>,
}

/// Python `repr` for a finite float, as `json` with `allow_nan=False` emits
/// it. Rust `{:?}` already gives shortest round-trip digits; the only fixup
/// is the exponent (`1e16` → `1e+16`, `1e-7` → `1e-07`: sign always shown,
/// at least two digits). Non-finite → `Err` like `allow_nan=False`.
pub fn py_float_repr(v: f64) -> Result<String, String> {
    if !v.is_finite() {
        return Err("Out of range float values are not JSON compliant".to_string());
    }
    let s = format!("{v:?}");
    match s.find('e') {
        None => Ok(s),
        Some(i) => {
            let (mant, exp) = (&s[..i], &s[i + 1..]);
            let (sign, digits) = match exp.strip_prefix(['+', '-']) {
                Some(d) => (&exp[..1], d),
                None => ("+", exp),
            };
            let neg = sign == "-";
            Ok(format!("{mant}e{}{digits:0>2}", if neg { "-" } else { "+" }))
        }
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Canonical snapshot bytes. `present` = columns the frame has (any order);
/// `rows` may be unsorted and contain duplicate timestamps (last wins).
/// Mirrors `canonical_frame_bytes` exactly, minus pandas indexing.
pub fn canonical_frame_bytes(present: &[&str], rows: &[SnapshotRow]) -> Result<Vec<u8>, String> {
    let columns: Vec<&str> =
        SNAPSHOT_COLUMNS.iter().copied().filter(|c| present.contains(c)).collect();
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].ns); // stable
    let mut last: HashMap<i64, usize> = HashMap::new();
    for &i in &order {
        last.insert(rows[i].ns, i);
    }
    let mut stamps: Vec<i64> = last.keys().copied().collect();
    stamps.sort_unstable();
    let mut doc = String::from("{\"columns\":[");
    doc.push_str(&columns.iter().map(|c| json_str(c)).collect::<Vec<_>>().join(","));
    doc.push_str("],\"rows\":[");
    for (ri, ns) in stamps.iter().enumerate() {
        let row = &rows[last[ns]];
        if row.values.len() != columns.len() {
            return Err(format!(
                "row width {} does not match {} columns",
                row.values.len(),
                columns.len()
            ));
        }
        if ri > 0 {
            doc.push(',');
        }
        doc.push('[');
        doc.push_str(&ns.to_string());
        for v in &row.values {
            doc.push(',');
            doc.push_str(&py_float_repr(*v)?);
        }
        doc.push(']');
    }
    doc.push_str("]}");
    Ok(doc.into_bytes())
}

/// Validate a snapshot id. Mirrors the `load` prefix check
/// (`strip().lower()`, 64 lowercase hex chars).
pub fn validate_snapshot_id(snapshot_id: &str) -> Result<String, String> {
    let normalized = snapshot_id.trim().to_lowercase();
    if normalized.len() != 64
        || !normalized.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("strategyV2.snapshotIdInvalid".to_string());
    }
    Ok(normalized)
}

/// SHA-256 hex digest (FIPS 180-4, std-only). Mirrors
/// `hashlib.sha256(payload).hexdigest()` used for snapshot ids.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python_forms() {
        assert_eq!(py_float_repr(100.0).unwrap(), "100.0");
        assert_eq!(py_float_repr(0.1).unwrap(), "0.1");
        assert_eq!(py_float_repr(1e16).unwrap(), "1e+16");
        assert_eq!(py_float_repr(1e-7).unwrap(), "1e-07");
        assert_eq!(py_float_repr(-2.5e100).unwrap(), "-2.5e+100");
        assert!(py_float_repr(f64::NAN).is_err());
        assert!(py_float_repr(f64::INFINITY).is_err());
    }

    #[test]
    fn id_validation() {
        let good = "a".repeat(64);
        assert_eq!(validate_snapshot_id(&format!("  {good}  ")).unwrap(), good);
        assert_eq!(
            validate_snapshot_id("xyz").unwrap_err(),
            "strategyV2.snapshotIdInvalid"
        );
        assert_eq!(
            validate_snapshot_id(&"G".repeat(64)).unwrap_err(),
            "strategyV2.snapshotIdInvalid"
        );
    }

    #[test]
    fn canonical_bytes_shape() {
        let rows = vec![
            SnapshotRow { ns: 2, values: vec![2.0, 3.0] },
            SnapshotRow { ns: 1, values: vec![1.0, 1.5] },
            SnapshotRow { ns: 2, values: vec![20.0, 30.0] }, // last wins
        ];
        let b = canonical_frame_bytes(&["close", "open", "extra"], &rows).unwrap();
        // column order follows SNAPSHOT_COLUMNS, unknown dropped
        assert_eq!(
            String::from_utf8(b).unwrap(),
            "{\"columns\":[\"open\",\"close\"],\"rows\":[[1,1.0,1.5],[2,20.0,30.0]]}"
        );
    }

    #[test]
    fn non_finite_value_errors() {
        let rows = vec![SnapshotRow { ns: 1, values: vec![f64::NAN] }];
        assert!(canonical_frame_bytes(&["open"], &rows).is_err());
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
