//! Real-data regime distribution: `regime_dist <bars.csv>`.
//! Prints per-state bar counts + RSI-14 oversold hits. Read-only diagnostic,
//! no trading, no synthetic data.
use qd_research::data::load_bars_csv;
use qd_research::regime::{detect, RegimeConfig};
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: regime_dist <bars.csv>");
    let text = std::fs::read_to_string(&path).expect("read csv");
    let bars = load_bars_csv(&text).expect("parse csv");
    let n = bars.len();
    let session = vec![1i64; n];
    let regimes = detect(&bars, &session, &RegimeConfig::default());
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in &regimes {
        *counts.entry(r.name().to_string()).or_default() += 1;
    }
    println!("{path}: n={n}");
    for (k, v) in &counts {
        println!("{k}: {v} ({:.1}%)", *v as f64 / n as f64 * 100.0);
    }
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let rsi = qd_engine::indicators::compute_rsi_wilder(&closes, 14);
    let os = rsi.iter().flatten().filter(|v| **v < 30.0).count();
    println!("rsi14<30 bars: {os}");
}
