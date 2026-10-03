//! qd-research — statistical trading-research and validation layer.
//!
//! Built for robustness discovery, NOT for impressive backtests. A strategy
//! with high returns that fails the robustness checks is reported as
//! unreliable — see [`report`] for the evidence-first assessment format.
//!
//! Pipeline: market data → [`features`] → [`regime`] → signal (user code or
//! [`strategies`]) → [`risk`] sizing → [`backtest`] → [`metrics`] +
//! [`mae_mfe`] → [`walkforward`] / [`montecarlo`] / [`cpcv`] / [`pbo`] /
//! [`sensitivity`] / [`costs`] → [`report`].
//!
//! Engineering rules enforced by construction:
//! - No look-ahead: every feature/signal consumes only bars `<= i`.
//! - No leakage: walk-forward/CPCV splitters respect temporal order and
//!   purge + embargo overlapping labels.
//! - Reproducibility: all stochastic code takes an explicit `seed`
//!   (SplitMix64, see [`montecarlo`]).
//! - Data snooping: White's Reality Check + Hansen SPA over the grid OOS
//!   matrix ([`snooping`]); p-values approximate under overlapping CPCV
//!   windows, stated as such — never faked.

pub mod backtest;
pub mod autonomy;
pub mod config;
pub mod costs;
pub mod cpcv;
pub mod data;
pub mod features;
pub mod gate;
pub mod jev;
pub mod mae_mfe;
pub mod metrics;
pub mod montecarlo;
pub mod pbo;
pub mod regime;
pub mod report;
pub mod risk;
pub mod run;
pub mod sensitivity;
pub mod shadow;
pub mod snooping;
pub mod strategies;
pub mod walkforward;
