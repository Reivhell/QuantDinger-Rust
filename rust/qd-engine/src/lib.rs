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
//! - `language` — `app/utils/language.py`
//!   (UI-language tag normalization + request-priority detection)
//! - `timeutil` — `to_utc_iso` from `app/utils/timeutil.py`
//!   (epoch/ISO/aware inputs → UTC `Z` strings; naive = UTC wall clock,
//!   non-UTC naive zones stay Python-side)
//! - `manifest` — `app/services/strategy_v2/models.py`
//!   (universe/subscription/schedule specs + manifest derivations: markets,
//!   frequencies, driving frequency, metadata shapes)

pub mod close_reason;
pub mod curve_sampling;
pub mod data_portal;
pub mod frequencies;
pub mod instruments;

pub mod grid;
pub mod indicators;
pub mod language;
pub mod json_helpers;
/// Notification display metadata (`utils/notification_display.py`).
pub mod notification_display;
/// IBKR desktop-broker deployment policy (`utils/local_brokers.py`).
pub mod local_brokers;
pub mod manifest;
/// Market-data failure taxonomy (`data_sources/errors.py`, pure slice).
pub mod market_data_errors;
pub mod market_visibility;
pub mod net_pnl;
pub mod perf_metrics;
pub mod pnl;
pub mod precise;
pub mod protection;
pub mod readiness;
pub mod risk_guard;
pub mod snapshot;
/// Strategy runtime log lines (`utils/strategy_runtime_logs.py`, DB-free slice).
pub mod strategy_runtime_logs;
pub mod timeutil;
