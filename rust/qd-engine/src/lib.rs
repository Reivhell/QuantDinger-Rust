//! qd-engine — Rust compute body for QuantDinger.
//!
//! Pure-math port of the hot-path Python helpers. The Python backend
//! (`backend_api_python/`) is the main brain and is NOT touched by this crate.
//! Every module documents its exact Python source file; behavior parity is
//! proven by the unit tests, which mirror the existing Python pytest cases.
//!
//! Modules:
//! - `indicators` — `app/utils/technical_indicators.py` (KDJ-CN, Wilder RSI)
//! - `pnl` — `app/utils/pnl.py` (unrealized PnL, notional, margin, PnL %)
//! - `precise` — `app/utils/numeric_precision.py` (step flooring, formatting)
//! - `risk_guard` — `app/utils/risk_guard.py` (fee coercion, trailing exits)
//! - `net_pnl` — `app/utils/trade_net_pnl.py` (+ symbol/exit helpers it uses)
//! - `grid` — `app/services/grid/levels.py` (level/cell generation)
//! - `protection` — `app/services/strategy_v2/protection.py`
//!   (stop-loss / take-profit / trailing-stop / time-limit engine)
//! - `close_reason` — `app/utils/trade_close_reason.py` + `enrich_execution_reference`
//!   from `app/utils/trade_execution.py`
//! - `curve_sampling` — `app/services/strategy_v2/curve_sampling.py`
//!   (equity-curve downsampler preserving risk observations)
//! - `frequencies` — `app/services/strategy_v2/frequencies.py`
//!   (frequency normalisation + annualisation periods)
//! - `instruments` — `app/services/strategy_v2/instruments.py`
//!   (instrument parsing, market inference, index/pool references)
//! - `perf_metrics` — pure-math core of `app/services/backtest/metrics.py`
//!   (information ratio, band classification, level cleaning)

pub mod close_reason;
pub mod curve_sampling;
pub mod frequencies;
pub mod instruments;

pub mod grid;
pub mod indicators;
pub mod net_pnl;
pub mod perf_metrics;
pub mod pnl;
pub mod precise;
pub mod protection;
pub mod risk_guard;
