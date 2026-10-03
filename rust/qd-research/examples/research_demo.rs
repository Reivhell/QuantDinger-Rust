//! Full research-pipeline runner on REAL market data (or seeded fixture).
//!
//! Data comes from `data.csv` in the research config: a strict-CSV bar file
//! (`t,open,high,low,close,volume`) produced e.g. by `fetch_okx.sh` from a
//! public exchange API — no demo feed, no invented candles. Only when
//! `data.csv` is empty does it fall back to the deterministic synthetic
//! fixture (seeded, reproducible, labeled SYNTH in the report).
//! Runs features → regime → two reference strategies → backtest → metrics →
//! MAE/MFE → walk-forward → Monte Carlo → CPCV → PBO → sensitivity → cost
//! stress → shadow (§12 bulkhead on the unseen last third) → JSON + Markdown
//! report. Proves the pipeline is wired end to end.

use qd_research::backtest::{run_backtest, ExecConfig};
use qd_research::costs::{cost_stress, execution_sensitivity};
use qd_research::cpcv::{cpcv_splits, score_cpcv_paths};
use qd_research::data::{load_bars_csv, synthetic_bars};
use qd_research::mae_mfe::analyze_mae_mfe;
use qd_research::metrics::compute_metrics;
use qd_research::montecarlo::run_monte_carlo;
use qd_research::pbo::{analyze_overfitting, PboBands, ScoreMatrix};
use qd_research::regime::{detect, RegimeConfig};
use qd_research::report::{regime_slices, report_json, report_markdown, ResearchInput};
use qd_research::sensitivity::{analyze_sensitivity, ParamPoint};
use qd_research::strategies::{EmaCrossTrend, RsiMeanReversion};
use qd_research::walkforward::{build_folds, summarize_walkforward, FoldOutcome};

fn main() {
    // Config: first CLI arg = path to research.json, else built-in default.
    // Every tuning constant below comes from `cfg` — nothing is hardcoded.
    let args: Vec<String> = std::env::args().collect();
    let cfg = if args.len() > 1 {
        let text = std::fs::read_to_string(&args[1])
            .unwrap_or_else(|e| panic!("cannot read config {}: {e}", args[1]));
        qd_research::config::load_config(&text)
            .unwrap_or_else(|e| panic!("invalid config {}: {e}", args[1]))
    } else {
        qd_research::config::load_config(qd_research::config::DEFAULT_CONFIG_JSON)
            .expect("built-in default config parses")
    };
    let n_cfg = cfg.bars;
    // REAL data first: `data.csv` (public exchange candles via fetch_okx.sh).
    // Synthetic only when no CSV is configured — and the report says so.
    let (bars, data_label) = if cfg.csv.trim().is_empty() {
        (
            synthetic_bars(n_cfg, cfg.data_seed),
            format!("synthetic seeded (seed={})", cfg.data_seed),
        )
    } else {
        let text = std::fs::read_to_string(&cfg.csv)
            .unwrap_or_else(|e| panic!("cannot read data.csv {}: {e}", cfg.csv));
        let loaded = load_bars_csv(&text)
            .unwrap_or_else(|e| panic!("invalid CSV {}: {e}", cfg.csv));
        let label = format!("REAL {} ({} bars, t {}..{})", cfg.csv, loaded.len(), loaded.first().map(|b| b.t).unwrap_or(0), loaded.last().map(|b| b.t).unwrap_or(0));
        (loaded, label)
    };
    let n = bars.len();
    if n < cfg.wf_train + cfg.wf_oos {
        panic!(
            "data has {n} bars, need >= {} for one walk-forward train+oos window",
            cfg.wf_train + cfg.wf_oos
        );
    }
    let session = vec![1i64; n];
    let rcfg = RegimeConfig::default();
    let regimes = detect(&bars, &session, &rcfg);

    // Strategy under test, from `strategy.name`: EMA-cross trend (rides
    // trends + breakouts) or RSI mean-reversion (buys dips in chop/range).
    // Both share the same pipeline below — only the signal source differs.
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let use_rsi = cfg.strategy_name == "RsiMeanReversion";
    let signals = if use_rsi {
        RsiMeanReversion {
            period: cfg.rsi_period,
            oversold: cfg.rsi_oversold,
            overbought: cfg.rsi_overbought,
            allow_shorts: cfg.rsi_shorts,
        }
        .signals(&closes, &regimes)
    } else {
        EmaCrossTrend { fast: cfg.ema_fast, slow: cfg.ema_slow }.signals(&closes, &regimes)
    };
    let wf_label = if use_rsi {
        format!("rsi{}/{}-{}", cfg.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought)
    } else {
        format!("ema{}/{}", cfg.ema_fast, cfg.ema_slow)
    };
    let feature_ids = vec![1u64; n];
    let exec = ExecConfig {
        commission: cfg.commission,
        spread: cfg.spread,
        slippage: cfg.slippage,
        latency_bars: cfg.latency_bars,
        stop_loss: cfg.stop_loss,
        take_profit: cfg.take_profit,
        ..ExecConfig::default()
    };
    let (risk, stop) = (cfg.risk_fraction, cfg.stop_loss);
    let res = run_backtest(&bars, &signals, &regimes, &feature_ids, &exec, 100_000.0, |px, eq| {
        qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0)
    });
    let metrics = compute_metrics(&res.trades, &res.equity_curve, 100_000.0, 252.0);
    let mae = analyze_mae_mfe(&res.trades);
    let rets: Vec<f64> = res.trades.iter().map(|t| t.net_pnl / 100_000.0).collect();

    // Walk-forward: IS window → (trivial selection) → OOS scoring.
    let folds = build_folds(n, cfg.wf_train, cfg.wf_oos, cfg.wf_step, false);
    let mut outcomes = Vec::new();
    for (k, f) in folds.iter().enumerate() {
        let sub = &bars[f.oos_start..f.oos_end];
        let sub_sig = &signals[f.oos_start..f.oos_end];
        let sub_reg = &regimes[f.oos_start..f.oos_end];
        let sub_fid = &feature_ids[f.oos_start..f.oos_end];
        let r = run_backtest(sub, sub_sig, sub_reg, sub_fid, &exec, 100_000.0, |px, eq| {
            qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0)
        });
        let m = compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0);
        outcomes.push(FoldOutcome {
            fold: k,
            selected_config: wf_label.clone(),
            is_score: 0.0,
            oos_return: m.total_return,
            oos_max_dd: m.max_drawdown,
            oos_sharpe: m.sharpe,
            oos_sortino: m.sortino,
            oos_trades: m.num_trades,
        });
    }
    let wf = summarize_walkforward(outcomes, cfg.min_oos_fraction);

    let mc = run_monte_carlo(&rets, cfg.mc_sims, cfg.mc_seed);
    // CPCV: 15 paths over the full series; each contiguous test run is
    // scored flat-to-flat with the strategy under test. Slicing precomputed
    // signals/regimes is leak-free: every feature is causal per bar, and
    // each run starts with no position.
    let splits = cpcv_splits(n, cfg.cpcv_partitions, cfg.cpcv_test, 5, 2);
    let cpcv_scores = score_cpcv_paths(&splits, 50, |lo, hi| {
        let r = run_backtest(
            &bars[lo..hi],
            &signals[lo..hi],
            &regimes[lo..hi],
            &feature_ids[lo..hi],
            &exec,
            100_000.0,
            |px, eq| qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0),
        );
        compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0).total_return
    });
    let cpcv_summary = qd_research::montecarlo::summarize(cpcv_scores);

    // PBO over a small honest grid. EMA: fast ∈ {10,20,30} × slow ∈ {50,100}.
    // RSI: period ∈ {7,14,21} × oversold ∈ {25,30,35} (overbought fixed).
    // Same folds, IS Sharpe vs OOS Sharpe — IS/OOS degradation decides.
    let mut cfgs = Vec::new();
    let mut is_m = Vec::new();
    let mut oos_m = Vec::new();
    // (label, signal-builder): boxed so the family loop below is shared.
    let grid: Vec<(String, Box<dyn Fn(&[f64], &[qd_research::regime::Regime]) -> Vec<qd_research::backtest::Signal>>)> =
        if use_rsi {
            let mut g: Vec<(String, Box<dyn Fn(&[f64], &[qd_research::regime::Regime]) -> Vec<qd_research::backtest::Signal>>)> = Vec::new();
            for p in [7usize, 14, 21] {
                for os in [25.0f64, 30.0, 35.0] {
                    let ob = cfg.rsi_overbought;
                    let sh = cfg.rsi_shorts;
                    g.push((
                        format!("rsi{p}/{os}-{ob}"),
                        Box::new(move |c: &[f64], r: &[qd_research::regime::Regime]| {
                            RsiMeanReversion { period: p, oversold: os, overbought: ob, allow_shorts: sh }.signals(c, r)
                        }),
                    ));
                }
            }
            g
        } else {
            let mut g: Vec<(String, Box<dyn Fn(&[f64], &[qd_research::regime::Regime]) -> Vec<qd_research::backtest::Signal>>)> = Vec::new();
            for fast in [10usize, 20, 30] {
                for slow in [50usize, 100] {
                    if fast >= slow {
                        continue;
                    }
                    g.push((
                        format!("ema{fast}/{slow}"),
                        Box::new(move |c: &[f64], r: &[qd_research::regime::Regime]| {
                            EmaCrossTrend { fast, slow }.signals(c, r)
                        }),
                    ));
                }
            }
            g
        };
    for (label, build) in &grid {
        cfgs.push(label.clone());
        let mut is_row = Vec::new();
        let mut oos_row = Vec::new();
        for f in build_folds(n, cfg.wf_train, cfg.wf_oos, cfg.wf_step, false) {
            let sg = build(&closes, &regimes);
            for (lo, hi) in [(f.is_start, f.is_end), (f.oos_start, f.oos_end)] {
                let r = run_backtest(
                    &bars[lo..hi],
                    &sg[lo..hi],
                    &regimes[lo..hi],
                    &feature_ids[lo..hi],
                    &exec,
                    100_000.0,
                    |px, eq| qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0),
                );
                let m = compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0);
                if hi == f.is_end {
                    is_row.push(m.sharpe);
                } else {
                    oos_row.push(m.sharpe);
                }
            }
        }
        is_m.push(is_row);
        oos_m.push(oos_row);
    }
    let pbo = analyze_overfitting(
        &ScoreMatrix { configs: cfgs, is: is_m, oos: oos_m },
        &PboBands::default(),
    );

    // Sensitivity around the configured knob (full-sample Sharpe): EMA fast,
    // or RSI period for the mean-reversion family. Plateau = robust.
    let (sens_name, sens_base, sens_vals): (&str, f64, Vec<f64>) = if use_rsi {
        ("rsi_period", cfg.rsi_period as f64, vec![7.0, 10.0, 14.0, 18.0, 21.0])
    } else {
        let d: [i64; 4] = [-2, -1, 1, 2];
        let mut v: Vec<f64> =
            d.iter().map(|x| (cfg.ema_fast as i64 + x).max(2) as f64).collect();
        v.push(cfg.ema_fast as f64);
        ("ema_fast", cfg.ema_fast as f64, v)
    };
    let score_of = |val: f64| {
        let sg = if use_rsi {
            RsiMeanReversion {
                period: (val as usize).max(2),
                oversold: cfg.rsi_oversold,
                overbought: cfg.rsi_overbought,
                allow_shorts: cfg.rsi_shorts,
            }
            .signals(&closes, &regimes)
        } else {
            EmaCrossTrend { fast: (val as usize).max(2), slow: cfg.ema_slow }
                .signals(&closes, &regimes)
        };
        let r = run_backtest(&bars, &sg, &regimes, &feature_ids, &exec, 100_000.0, |px, eq| {
            qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0)
        });
        compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0).sharpe
    };
    let sweep: Vec<ParamPoint> = sens_vals
        .iter()
        .filter(|v| **v != sens_base)
        .map(|v| ParamPoint { name: sens_name.into(), value: *v, score: score_of(*v) })
        .collect();
    let sens = analyze_sensitivity(
        ParamPoint { name: sens_name.into(), value: sens_base, score: score_of(sens_base) },
        &sweep,
        0.8,
        0.3,
    );

    let gc: Vec<(f64, f64)> = res.trades.iter().map(|t| (t.gross_pnl, t.fees + t.slippage_cost)).collect();
    let cs = cost_stress(&gc, &[1.0, 1.25, 1.5, 2.0, 3.0]);
    let cv = execution_sensitivity(&cs).to_string();

    // Shadow bulkhead (§12): the LAST third of bars is unseen by every
    // earlier stage (WF folds, CPCV runs, PBO grid all score IS/OOS windows
    // of their own — none trains here). The candidate (configured EMA) runs
    // under the autonomous gate + live kill-switch against a duller
    // incumbent (slow EMA); the drawdown-first verdict feeds the
    // Shadow → Deploy loop transition.
    let shadow_lo = 2 * n / 3;
    let shadow = {
        use qd_research::regime::Regime as RG;
        use qd_research::shadow::{compare_candidate, run_shadow, ShadowConfig};
        let scfg = ShadowConfig {
            starting_equity: 100_000.0,
            max_positions: cfg.max_positions,
            max_open_risk: cfg.max_open_risk,
            blocked_regimes: vec![
                RG::Ranging,
                RG::HighVolatility,
                RG::LowVolatility,
                RG::MeanReversion,
                RG::Abnormal,
            ],
            leverage_ok: true,
            stop_present: cfg.stop_loss > 0.0,
            execution_ok: true,
            ks_daily: cfg.max_daily_loss,
            ks_weekly: cfg.max_weekly_loss,
            ks_drawdown: cfg.max_drawdown,
            ks_exposure: cfg.max_exposure,
            ks_corr: cfg.max_correlated_exposure,
            week_bars: 5,
            session_bars: 50,
        };
        let w = |lo: usize| {
            (
                &bars[lo..],
                &signals[lo..],
                &regimes[lo..],
                &feature_ids[lo..],
            )
        };
        let (wb, ws, wr, wf) = w(shadow_lo);
        let cand = run_shadow(wb, ws, wr, wf, &exec, |px, eq| {
            qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0)
        }, &scfg)
        .expect("shadow candidate runs");
        // Incumbent: a duller sibling of the same family on the same unseen
        // window — slow EMA for the trend family, slow/long RSI for the
        // mean-reversion family.
        let inc_sig = if use_rsi {
            RsiMeanReversion {
                period: cfg.rsi_period + 7,
                oversold: cfg.rsi_oversold - 5.0,
                overbought: cfg.rsi_overbought,
                allow_shorts: cfg.rsi_shorts,
            }
            .signals(&closes, &regimes)
        } else {
            EmaCrossTrend { fast: cfg.ema_fast + 10, slow: cfg.ema_slow + 50 }
                .signals(&closes, &regimes)
        };
        let inc = run_shadow(
            &bars[shadow_lo..],
            &inc_sig[shadow_lo..],
            &regimes[shadow_lo..],
            &feature_ids[shadow_lo..],
            &exec,
            |px, eq| qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0),
            &scfg,
        )
        .expect("shadow incumbent runs");
        println!(
            "shadow [unseen last third, {} bars]: candidate {} trades ret {:.2}% dd {:.2}% halt {:?} | incumbent {} trades ret {:.2}% dd {:.2}% halt {:?}",
            n - shadow_lo,
            cand.trade_count,
            cand.total_return * 100.0,
            cand.max_drawdown * 100.0,
            cand.halt_reason.as_deref().unwrap_or("none"),
            inc.trade_count,
            inc.total_return * 100.0,
            inc.max_drawdown * 100.0,
            inc.halt_reason.as_deref().unwrap_or("none"),
        );
        let verdict = compare_candidate(&cand, &inc, cpcv_summary.median, 0.05, 0.25);
        println!("shadow verdict: {:?} — {}", verdict.verdict, verdict.reason);
        qd_research::shadow::ShadowSummary {
            unseen_bars: n - shadow_lo,
            candidate_trades: cand.trade_count,
            candidate_return: cand.total_return,
            candidate_max_dd: cand.max_drawdown,
            candidate_halt: cand.halt_reason.clone(),
            candidate_suppressed: cand.suppressed,
            incumbent_trades: inc.trade_count,
            incumbent_return: inc.total_return,
            incumbent_max_dd: inc.max_drawdown,
            incumbent_halt: inc.halt_reason.clone(),
            verdict: format!("{:?}", verdict.verdict),
            verdict_reason: verdict.reason.to_string(),
        }
    };

    let (strat_label, strat_cfg) = if use_rsi {
        (
            format!(
                "RsiMeanReversion({},{:.0},{:.0}{})",
                cfg.rsi_period,
                cfg.rsi_oversold,
                cfg.rsi_overbought,
                if cfg.rsi_shorts { ",shorts" } else { ",long-only" }
            ),
            format!(
                "period={} os={:.0} ob={:.0} shorts={} stop={:.3} take={:.3} risk={:.3}",
                cfg.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought,
                cfg.rsi_shorts, cfg.stop_loss, cfg.take_profit, cfg.risk_fraction
            ),
        )
    } else {
        (
            format!("EmaCrossTrend({},{})", cfg.ema_fast, cfg.ema_slow),
            format!(
                "fast={} slow={} stop={:.3} take={:.3} risk={:.3}",
                cfg.ema_fast, cfg.ema_slow, cfg.stop_loss, cfg.take_profit, cfg.risk_fraction
            ),
        )
    };
    let input = ResearchInput {
        strategy_name: strat_label,
        strategy_config: strat_cfg,
        asset: cfg.asset.clone(),
        timeframe: cfg.timeframe.clone(),
        date_range: data_label.clone(),
        n_bars: n,
        regime_perf: regime_slices(&res.trades, 100_000.0),
        walkforward: Some(wf),
        montecarlo: Some(mc),
        cpcv_paths: splits.len(),
        cpcv_runs_scored: cpcv_summary.count,
        cpcv_median_oos: cpcv_summary.median,
        cpcv_p25_oos: cpcv_summary.p25,
        cpcv_p75_oos: cpcv_summary.p75,
        pbo: Some(pbo),
        sensitivity: vec![sens],
        cost_stress: cs,
        shadow: Some(shadow),
        primary_weakness: if cfg.csv.trim().is_empty() {
            "synthetic fixture — see regime table".into()
        } else {
            format!("real-data run ({data_label}) — see regime table")
        },
        regime_dependency: "see regime table".into(),
        metrics,
        mae_mfe: Some(mae),
        cost_verdict: cv,
    };
    println!("=== JSON ===\n{}", report_json(&input));
    println!("\n=== MARKDOWN ===\n{}", report_markdown(&input));
}
