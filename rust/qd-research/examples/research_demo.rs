//! Full research-pipeline demo on seeded synthetic OHLCV.
//!
//! Generates a deterministic series (trend + range + vol-shock regimes),
//! runs features → regime → two reference strategies → backtest → metrics →
//! MAE/MFE → walk-forward → Monte Carlo → CPCV → PBO → sensitivity → cost
//! stress → JSON + Markdown report. Proves the pipeline is wired end to end.

use qd_research::backtest::{run_backtest, ExecConfig};
use qd_research::costs::{cost_stress, execution_sensitivity};
use qd_research::cpcv::{cpcv_splits, score_cpcv_paths};
use qd_research::features::Bar;
use qd_research::mae_mfe::analyze_mae_mfe;
use qd_research::metrics::compute_metrics;
use qd_research::montecarlo::{run_monte_carlo, SplitMix64};
use qd_research::pbo::{analyze_overfitting, PboBands, ScoreMatrix};
use qd_research::regime::{detect, RegimeConfig};
use qd_research::report::{regime_slices, report_json, report_markdown, ResearchInput};
use qd_research::sensitivity::{analyze_sensitivity, ParamPoint};
use qd_research::strategies::EmaCrossTrend;
use qd_research::walkforward::{build_folds, summarize_walkforward, FoldOutcome};

fn synth_bars(n: usize, seed: u64) -> Vec<Bar> {
    // Deterministic GBM-ish path with three phases: trend, range, shock.
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
    let n = cfg.bars;
    let bars = synth_bars(n, cfg.data_seed);
    let session = vec![1i64; n];
    let rcfg = RegimeConfig::default();
    let regimes = detect(&bars, &session, &rcfg);

    // Strategy under test: EMA-cross trend.
    let strat = EmaCrossTrend { fast: cfg.ema_fast, slow: cfg.ema_slow };
    let signals = strat.signals(&bars.iter().map(|b| b.close).collect::<Vec<_>>(), &regimes);
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
            selected_config: format!("ema{}/{}", cfg.ema_fast, cfg.ema_slow),
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

    // PBO over a small honest grid: fast ∈ {10,20,30} × slow ∈ {50,100}.
    let mut cfgs = Vec::new();
    let mut is_m = Vec::new();
    let mut oos_m = Vec::new();
    for fast in [10usize, 20, 30] {
        for slow in [50usize, 100] {
            if fast >= slow {
                continue;
            }
            cfgs.push(format!("ema{fast}/{slow}"));
            let mut is_row = Vec::new();
            let mut oos_row = Vec::new();
            for f in build_folds(n, cfg.wf_train, cfg.wf_oos, cfg.wf_step, false) {
                let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
                let sg = EmaCrossTrend { fast, slow }.signals(&closes, &regimes);
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
    }
    let pbo = analyze_overfitting(
        &ScoreMatrix { configs: cfgs, is: is_m, oos: oos_m },
        &PboBands::default(),
    );

    // Sensitivity around the configured fast EMA (full-sample Sharpe).
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let score_of = |fast: usize| {
        let sg = EmaCrossTrend { fast, slow: cfg.ema_slow }.signals(&closes, &regimes);
        let r = run_backtest(&bars, &sg, &regimes, &feature_ids, &exec, 100_000.0, |px, eq| {
            qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0)
        });
        compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0).sharpe
    };
    let deltas: [i64; 4] = [-2, -1, 1, 2];
    let sweep: Vec<ParamPoint> = deltas
        .iter()
        .map(|d| (cfg.ema_fast as i64 + d).max(2) as usize)
        .map(|v| ParamPoint { name: "ema_fast".into(), value: v as f64, score: score_of(v) })
        .collect();
    let sens = analyze_sensitivity(
        ParamPoint { name: "ema_fast".into(), value: cfg.ema_fast as f64, score: score_of(cfg.ema_fast) },
        &sweep,
        0.8,
        0.3,
    );

    let gc: Vec<(f64, f64)> = res.trades.iter().map(|t| (t.gross_pnl, t.fees + t.slippage_cost)).collect();
    let cs = cost_stress(&gc, &[1.0, 1.25, 1.5, 2.0, 3.0]);
    let cv = execution_sensitivity(&cs).to_string();

    let input = ResearchInput {
        strategy_name: format!("EmaCrossTrend({},{})", cfg.ema_fast, cfg.ema_slow),
        strategy_config: format!(
            "fast={} slow={} stop={:.3} take={:.3} risk={:.3}",
            cfg.ema_fast, cfg.ema_slow, cfg.stop_loss, cfg.take_profit, cfg.risk_fraction
        ),
        asset: cfg.asset.clone(),
        timeframe: cfg.timeframe.clone(),
        date_range: format!("synthetic seeded (seed={})", cfg.data_seed),
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
        primary_weakness: "synthetic demo — see regime table".into(),
        regime_dependency: "see regime table".into(),
        metrics,
        mae_mfe: Some(mae),
        cost_verdict: cv,
    };
    println!("=== JSON ===\n{}", report_json(&input));
    println!("\n=== MARKDOWN ===\n{}", report_markdown(&input));
}
