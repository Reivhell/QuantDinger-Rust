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
//! - Data snooping: White's Reality Check / Hansen SPA are reported as
//!   NOT IMPLEMENTED with reasons, never faked ([`report`]).

pub mod backtest;
pub mod costs;
pub mod cpcv;
pub mod features;
pub mod mae_mfe;
pub mod metrics;
pub mod montecarlo;
pub mod pbo;
pub mod regime;
pub mod report;
pub mod risk;
pub mod sensitivity;
pub mod strategies;
pub mod walkforward;
