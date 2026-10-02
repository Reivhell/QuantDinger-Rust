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
//! - `market_visibility` — `app/utils/market_visibility.py`
//!   (operator-controlled market visibility: ENABLED_MARKETS + SHOW_* flags)
//! - `data_portal` — pandas-free logic of
//!   `app/services/strategy_v2/data.py` (point-in-time windowing, key
//!   resolution, frame normalization)
//! - `snapshot` — content-addressing core of
//!   `app/services/strategy_v2/snapshot.py` (canonical snapshot bytes,
//!   Python-float rendering, id validation)
//! - `readiness` — `app/services/strategy_v2/readiness.py`
//!   (universe-history gate, warmup-bar counts, fundamental-field checks)

pub mod close_reason;
pub mod curve_sampling;
pub mod data_portal;
pub mod frequencies;
pub mod instruments;

pub mod grid;
pub mod indicators;
pub mod market_visibility;
pub mod net_pnl;
pub mod perf_metrics;
pub mod pnl;
pub mod precise;
pub mod protection;
pub mod readiness;
pub mod risk_guard;
pub mod snapshot;
