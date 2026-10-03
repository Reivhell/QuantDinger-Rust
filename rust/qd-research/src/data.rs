//! Market-data ingress: strict CSV bar loader + deterministic synthetic
//! fixture generator.
//!
//! This is the pipeline's first stage (Market Data, spec §12). Strictness is
//! deliberate: any malformed row is an error, never a silent skip — trading
//! on misaligned data is worse than not trading (§11: uncertain state →
//! fail safe). No I/O here: the caller reads the file, this parses text.
//!
//! Expected format (exact header, one bar per line):
//! ```text
//! t,open,high,low,close,volume
//! 1767225600,100.0,100.5,99.8,100.2,1234.5
//! ```
//! Rules: 6 columns; `t` is an i64 bar timestamp (unix seconds or index);
//! prices finite with `high >= low`, `close > 0`, `open > 0`; `volume >= 0`;
//! `t` strictly increasing (misordered history = corrupt history).

use crate::features::Bar;
use crate::montecarlo::SplitMix64;

pub const CSV_HEADER: &str = "t,open,high,low,close,volume";

/// Parse CSV text into bars. `Err` names the first offending line.
pub fn load_bars_csv(text: &str) -> Result<Vec<Bar>, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| "empty file: no header".to_string())?;
    if header.trim() != CSV_HEADER {
        return Err(format!("bad header {header:?}: expected {CSV_HEADER:?}"));
    }
    let mut bars: Vec<Bar> = Vec::new();
    for (k, raw) in lines.enumerate() {
        let line_no = k + 2; // 1-based incl. header
        let line = raw.trim();
        if line.is_empty() {
            continue; // tolerate trailing newline, not missing data
        }
        let c: Vec<&str> = line.split(',').map(str::trim).collect();
        if c.len() != 6 {
            return Err(format!("line {line_no}: want 6 columns, got {}", c.len()));
        }
        let num = |s: &str, what: &str| -> Result<f64, String> {
            s.parse::<f64>()
                .map_err(|_| format!("line {line_no}: {what} {s:?} not a number"))
        };
        let t: i64 = c[0]
            .parse()
            .map_err(|_| format!("line {line_no}: t {:?} not an integer", c[0]))?;
        let (open, high, low, close, volume) = (
            num(c[1], "open")?,
            num(c[2], "high")?,
            num(c[3], "low")?,
            num(c[4], "close")?,
            num(c[5], "volume")?,
        );
        let bar = Bar { t, open, high, low, close, volume };
        if !bar.is_valid() {
            return Err(format!("line {line_no}: non-finite OHLCV or high<low"));
        }
        if open <= 0.0 || close <= 0.0 {
            return Err(format!("line {line_no}: non-positive open/close"));
        }
        if volume < 0.0 {
            return Err(format!("line {line_no}: negative volume"));
        }
        if let Some(prev) = bars.last() {
            let prev_t: i64 = prev.t;
            if t <= prev_t {
                return Err(format!(
                    "line {line_no}: t={t} not after prev t={prev_t} (history must be ordered)"
                ));
            }
        }
        bars.push(bar);
    }
    if bars.is_empty() {
        return Err("no data rows".to_string());
    }
    Ok(bars)
}

/// Deterministic GBM-ish fixture: trend phase, range phase, shock phase.
/// Same `(n, seed)` → identical bars, always. Test/demo fixture only —
/// never a substitute for real market data.
pub fn synthetic_bars(n: usize, seed: u64) -> Vec<Bar> {
    let mut rng = SplitMix64(seed);
    let mut bars = Vec::with_capacity(n);
    let mut px = 100.0;
    for i in 0..n {
        let drift = if i < n / 3 {
            0.0012
        } else if i < 2 * n / 3 {
            0.0
        } else {
            -0.0008
        };
        let vol = if i >= 2 * n / 3 { 0.020 } else { 0.008 };
        // Box-Muller from the seeded RNG.
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let ret = drift + vol * z;
        let open = px;
        let close = open * (1.0 + ret);
        let high = open.max(close) * (1.0 + vol * rng.next_f64() * 0.3);
        let low = open.min(close) * (1.0 - vol * rng.next_f64() * 0.3);
        bars.push(Bar {
            t: i as i64,
            open,
            high,
            low,
            close,
            volume: 1000.0 + rng.next_f64() * 500.0,
        });
        px = close;
    }
    bars
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "t,open,high,low,close,volume\n\
        1,100.0,100.5,99.8,100.2,1200.0\n\
        2,100.2,101.0,100.0,100.8,900.0\n";

    #[test]
    fn parses_clean_csv() {
        let bars = load_bars_csv(GOOD).expect("clean csv parses");
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].t, 1);
        assert!((bars[1].close - 100.8).abs() < 1e-12);
    }

    #[test]
    fn rejects_header_body_garbage() {
        assert!(load_bars_csv("").is_err());
        assert!(load_bars_csv("t,open,high,low,close\n1,2,3,4,5\n").is_err());
        assert!(load_bars_csv("t,open,high,low,close,volume\n1,100,99,101,100,10\n").is_err()); // high<low
        assert!(load_bars_csv("t,open,high,low,close,volume\n1,100,101,99,NaN,10\n").is_err());
        assert!(load_bars_csv("t,open,high,low,close,volume\n1,100,101,99,-5,10\n").is_err()); // neg close
        assert!(load_bars_csv("t,open,high,low,close,volume\n1,100,101,99,100,-3\n").is_err()); // neg vol
        assert!(load_bars_csv("t,open,high,low,close,volume\n").is_err()); // header only
    }

    #[test]
    fn rejects_unordered_timestamps() {
        let csv = "t,open,high,low,close,volume\n\
            5,100,101,99,100,10\n\
            5,100,101,99,100,10\n";
        assert!(load_bars_csv(csv).is_err());
        let csv2 = "t,open,high,low,close,volume\n\
            9,100,101,99,100,10\n\
            3,100,101,99,100,10\n";
        assert!(load_bars_csv(csv2).is_err());
    }

    #[test]
    fn synthetic_is_deterministic_and_valid() {
        let a = synthetic_bars(300, 42);
        let b = synthetic_bars(300, 42);
        assert_eq!(a.len(), 300);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| x.close == y.close));
        assert!(a.iter().all(|b| b.is_valid() && b.close > 0.0));
        let c = synthetic_bars(300, 43);
        assert!(c.iter().zip(a.iter()).any(|(x, y)| x.close != y.close));
    }
}
