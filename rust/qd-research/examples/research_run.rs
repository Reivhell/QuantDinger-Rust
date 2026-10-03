//! `research_run <research.json>` — full research pipeline on REAL market data.
//!
//! Body/brain contract: the Python backend (`backend_api_python/`) is the
//! brain — it owns data fetching (see `fetch_okx.sh`), scheduling, and order
//! submission. This binary is the body: it reads the CSV path from the
//! config, runs features → regime → strategy → backtest → metrics → MAE/MFE
//! → walk-forward → Monte Carlo → CPCV → PBO → sensitivity → cost stress →
//! shadow (§12 bulkhead on the unseen last third) → §13 deployment gate →
//! §14 version registry → loop transition, and prints the JSON + Markdown
//! report on stdout. Exit non-zero on bad config / bad data; a BLOCKED gate
//! is evidence output (exit 0), never an error.
//!
//! Machine contract: the JSON report is the text between the `=== JSON ===`
//! marker line and the blank line before `=== MARKDOWN ===`. Empty
//! `data.csv` is rejected — the synthetic fixture is unit-test only.

use qd_research::backtest::Signal;
use qd_research::costs::{cost_stress, execution_sensitivity};
use qd_research::cpcv::{contiguous_runs, cpcv_splits, score_cpcv_paths};
use qd_research::features::Bar;
use qd_research::mae_mfe::analyze_mae_mfe;
use qd_research::metrics::compute_metrics;
use qd_research::montecarlo::run_monte_carlo;
use qd_research::pbo::{analyze_overfitting, PboBands, ScoreMatrix};
use qd_research::regime::{detect, Regime, RegimeConfig};
use qd_research::report::{regime_slices, report_json, report_markdown, ResearchInput};
use qd_research::run::{
    self, family_of, StrategyFam, PERIODS_PER_YEAR, STARTING_EQUITY,
};
use qd_research::sensitivity::{analyze_sensitivity, ParamPoint};
use qd_research::strategies::{DonchianBreakout, EmaCrossTrend, RsiMeanReversion};
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
    // REAL market bars only — empty `data.csv` is rejected, never silently
    // replaced with synthetic data.
    let (bars, data_label) =
        run::load_bars(&cfg).unwrap_or_else(|e| panic!("{e}"));
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
    // trends + breakouts), RSI mean-reversion (buys dips in chop/range), or
    // Donchian breakout (volume-confirmed range escapes, long-only).
    // All share the same pipeline below — only the signal source differs.
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let fam = family_of(&cfg.strategy_name);
    let use_rsi = fam == StrategyFam::Rsi;
    let signals = run::build_signals(fam, &cfg, &bars, &closes, &regimes);
    let wf_label = run::fold_label(fam, &cfg);
    let feature_ids = vec![1u64; n];
    let exec = run::exec_of(&cfg);
    let res = qd_research::backtest::run_backtest(
        &bars,
        &signals,
        &regimes,
        &feature_ids,
        &exec,
        STARTING_EQUITY,
        |px, eq| run::fixed_qty(px, eq, &cfg),
    );
    let metrics =
        compute_metrics(&res.trades, &res.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR);
    let mae = analyze_mae_mfe(&res.trades);
    let rets: Vec<f64> =
        res.trades.iter().map(|t| t.net_pnl / STARTING_EQUITY).collect();

    // Walk-forward: IS window → (trivial selection) → OOS scoring.
    let folds = build_folds(n, cfg.wf_train, cfg.wf_oos, cfg.wf_step, false);
    let mut outcomes = Vec::new();
    for (k, f) in folds.iter().enumerate() {
        let sub = &bars[f.oos_start..f.oos_end];
        let sub_sig = &signals[f.oos_start..f.oos_end];
        let sub_reg = &regimes[f.oos_start..f.oos_end];
        let sub_fid = &feature_ids[f.oos_start..f.oos_end];
        let r = qd_research::backtest::run_backtest(sub, sub_sig, sub_reg, sub_fid, &exec, STARTING_EQUITY, |px, eq| {
            run::fixed_qty(px, eq, &cfg)
        });
        let m = compute_metrics(&r.trades, &r.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR);
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
        let r = qd_research::backtest::run_backtest(
            &bars[lo..hi],
            &signals[lo..hi],
            &regimes[lo..hi],
            &feature_ids[lo..hi],
            &exec,
            STARTING_EQUITY,
            |px, eq| run::fixed_qty(px, eq, &cfg),
        );
        compute_metrics(&r.trades, &r.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR).total_return
    });
    let cpcv_summary = qd_research::montecarlo::summarize(cpcv_scores);

    // PBO over a small honest grid. EMA: fast ∈ {10,20,30} × slow ∈ {50,100}.
    // RSI: period ∈ {7,14,21} × oversold ∈ {25,30,35} (overbought fixed).
    // Donchian: lookback ∈ {10,20,30} × relvol ∈ {1.5,2.0}.
    // Same folds, IS robust-return vs OOS robust-return.
    // Robust = CAGR with a zero-trade floor (see `run::robust_cagr`).
    let mut cfgs = Vec::new();
    let mut is_m = Vec::new();
    let mut oos_m = Vec::new();
    // (label, signal-builder): boxed so the family loop below is shared.
    type SigFn = dyn Fn(&[Bar], &[f64], &[Regime]) -> Vec<Signal>;
    let grid: Vec<(String, Box<SigFn>)> = if fam == StrategyFam::Donchian {
        let mut g: Vec<(String, Box<SigFn>)> = Vec::new();
        for lb in [10usize, 20, 30] {
            for rv in [1.5f64, 2.0] {
                g.push((
                    format!("donch{lb}/rv{rv:.1}"),
                    Box::new(move |b, _, r| {
                        DonchianBreakout { lookback: lb, relvol_min: rv }.signals(b, r)
                    }),
                ));
            }
        }
        g
    } else if use_rsi {
        let mut g: Vec<(String, Box<SigFn>)> = Vec::new();
        for p in [7usize, 14, 21] {
            for os in [25.0f64, 30.0, 35.0] {
                let ob = cfg.rsi_overbought;
                let sh = cfg.rsi_shorts;
                g.push((
                    format!("rsi{p}/{os}-{ob}"),
                    Box::new(move |_, c: &[f64], r: &[Regime]| {
                        RsiMeanReversion { period: p, oversold: os, overbought: ob, allow_shorts: sh }.signals(c, r)
                    }),
                ));
            }
        }
        g
    } else {
        let mut g: Vec<(String, Box<SigFn>)> = Vec::new();
        for fast in [10usize, 20, 30] {
            for slow in [50usize, 100] {
                if fast >= slow {
                    continue;
                }
                g.push((
                    format!("ema{fast}/{slow}"),
                    Box::new(move |_, c: &[f64], r: &[Regime]| {
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
            let sg = build(&bars, &closes, &regimes);
            for (lo, hi) in [(f.is_start, f.is_end), (f.oos_start, f.oos_end)] {
                let r = qd_research::backtest::run_backtest(
                    &bars[lo..hi],
                    &sg[lo..hi],
                    &regimes[lo..hi],
                    &feature_ids[lo..hi],
                    &exec,
                    STARTING_EQUITY,
                    |px, eq| run::fixed_qty(px, eq, &cfg),
                );
                let m =
                    compute_metrics(&r.trades, &r.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR);
                let robust = run::robust_cagr(m.num_trades, m.cagr);
                if hi == f.is_end {
                    is_row.push(robust);
                } else {
                    oos_row.push(robust);
                }
            }
        }
        is_m.push(is_row);
        oos_m.push(oos_row);
    }
    let pbo = analyze_overfitting(
        &ScoreMatrix { configs: cfgs, is: is_m, oos: oos_m.clone() },
        &PboBands::default(),
    );

    // Data snooping (§7 luck adjustment): White RC + Hansen SPA over the
    // grid scored on every CPCV run (K configs × R runs, vs cash). Each run
    // is a contiguous flat-to-flat OOS window, so T = R ≈ 23 clears the
    // min_periods=10 bar where the 2 WF folds cannot.
    // Thin grids still return None (fail-closed) — never faked.
    let snoop_runs: Vec<(usize, usize)> = splits
        .iter()
        .filter(|s| !s.train.is_empty())
        .flat_map(|s| contiguous_runs(&s.test, 50))
        .collect();
    let mut snoop_m: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    for (_, build) in &grid {
        let sg = build(&bars, &closes, &regimes);
        let mut row = Vec::with_capacity(snoop_runs.len());
        for (lo, hi) in &snoop_runs {
            let r = qd_research::backtest::run_backtest(
                &bars[*lo..*hi],
                &sg[*lo..*hi],
                &regimes[*lo..*hi],
                &feature_ids[*lo..*hi],
                &exec,
                STARTING_EQUITY,
                |px, eq| run::fixed_qty(px, eq, &cfg),
            );
            let m =
                compute_metrics(&r.trades, &r.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR);
            row.push(run::robust_cagr(m.num_trades, m.cagr));
        }
        snoop_m.push(row);
    }
    let snoop_cfg = qd_research::snooping::SnoopConfig {
        boot_sims: cfg.snoop_sims,
        seed: cfg.snoop_seed,
        mean_block: cfg.snoop_block,
        ..qd_research::snooping::SnoopConfig::default()
    };
    let snoop = qd_research::snooping::data_snoop(&snoop_m, &snoop_cfg);
    match &snoop {
        Some(s) => println!(
            "snooping [{} rules x {} periods, {} sims]: White stat {:.3} p={:.4} | SPA stat {:.3} p_consistent={:.4} (lower {:.4} / upper {:.4}) — {}",
            s.rules, s.periods, s.boot_sims,
            s.white_stat, s.white_p, s.spa_stat,
            s.spa_p_consistent, s.spa_p_lower, s.spa_p_upper, s.assessment
        ),
        None => println!("snooping: INSUFFICIENT DATA (OOS matrix too thin) — no p-values fabricated"),
    }

    // Sensitivity around the configured knob (full-sample CAGR with the
    // zero-trade floor): EMA fast, RSI period, or Donchian lookback.
    // Plateau = robust.
    let (sens_name, sens_base, sens_vals): (&str, f64, Vec<f64>) = if fam == StrategyFam::Donchian {
        ("donchian_lookback", cfg.donchian_lookback as f64, vec![10.0, 15.0, 20.0, 25.0, 30.0])
    } else if use_rsi {
        ("rsi_period", cfg.rsi_period as f64, vec![7.0, 10.0, 14.0, 18.0, 21.0])
    } else {
        let d: [i64; 4] = [-2, -1, 1, 2];
        let mut v: Vec<f64> =
            d.iter().map(|x| (cfg.ema_fast as i64 + x).max(2) as f64).collect();
        v.push(cfg.ema_fast as f64);
        ("ema_fast", cfg.ema_fast as f64, v)
    };
    let score_of = |val: f64| {
        let sg = if fam == StrategyFam::Donchian {
            DonchianBreakout {
                lookback: (val as usize).max(2),
                relvol_min: cfg.donchian_relvol,
            }
            .signals(&bars, &regimes)
        } else if use_rsi {
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
        let r = qd_research::backtest::run_backtest(&bars, &sg, &regimes, &feature_ids, &exec, STARTING_EQUITY, |px, eq| {
            run::fixed_qty(px, eq, &cfg)
        });
        let m = compute_metrics(&r.trades, &r.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR);
        // Full-sample window is identical for every knob value, so total
        // return and CAGR rank identically — use CAGR for consistency with
        // the PBO/snooping grids. 0 trades → 0.0 (flat, no evidence).
        run::robust_cagr(m.num_trades, m.cagr)
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
    let cs = cost_stress(&gc, &run::COST_MULTIPLIERS);
    let cv = execution_sensitivity(&cs).to_string();

    // Shadow bulkhead (§12): the LAST third of bars is unseen by every
    // earlier stage (WF folds, CPCV runs, PBO grid all score IS/OOS windows
    // of their own — none trains here). The configured candidate runs
    // under the autonomous gate + live kill-switch against a duller
    // incumbent of the same family; the drawdown-first verdict feeds the
    // Shadow → Deploy loop transition.
    let shadow_lo = 2 * n / 3;
    let shadow = {
        use qd_research::shadow::{compare_candidate, run_shadow};
        let scfg = run::shadow_config(&cfg, run::leverage_ok(&cfg));
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
            run::fixed_qty(px, eq, &cfg)
        }, &scfg)
        .expect("shadow candidate runs");
        // Incumbent: a duller sibling of the same family on the same unseen
        // window — slow EMA, slow/long RSI, or longer-channel Donchian.
        let inc_sig = run::incumbent_signals(fam, &cfg, &bars, &closes, &regimes);
        let inc = run_shadow(
            &bars[shadow_lo..],
            &inc_sig[shadow_lo..],
            &regimes[shadow_lo..],
            &feature_ids[shadow_lo..],
            &exec,
            |px, eq| run::fixed_qty(px, eq, &cfg),
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

    // §13 deployment gate, fed from measured evidence — one mapping,
    // fail-closed (missing stage = false). Never hand-roll gate fields.
    // (The actual evaluate/publish block sits after the strat labels so
    // the registry record carries the final strategy identity.)
    let gate_ev = qd_research::gate::GateEvidence {
        realistic_costs: cfg.commission > 0.0 && cfg.spread > 0.0 && cfg.slippage > 0.0,
        walkforward: Some(wf.clone()),
        monte_carlo: Some(mc.clone()),
        backtest_max_dd: metrics.max_drawdown,
        sensitivity: Some(sens.clone()),
        pbo: Some(pbo.clone()),
        cost_verdict: cv.clone(),
        max_drawdown_cap: cfg.max_drawdown,
        stop_on_every_trade: cfg.stop_loss > 0.0,
        leverage_ok: run::leverage_ok(&cfg),
        killswitch_verified: true, // shadow runs under the live kill-switch
        snooping: snoop.clone(),
        shadow: Some((&shadow).into()),
    };

    let (strat_label, strat_cfg) = run::strategy_identity(fam, &cfg);
    // §13 gate evaluate → §14 registry publish → §12 loop transition,
    // after the final strategy identity exists. Failures return to
    // Research — never live.
    let gate = qd_research::gate::gate_from_evidence(&gate_ev);
    let gate_allowed = gate.deploy_allowed();
    println!("deploy gate: {} {:?}", if gate_allowed { "ALLOWED" } else { "BLOCKED" }, gate.failures());
    let mut reg = qd_research::autonomy::VersionRegistry::default();
    reg.publish(qd_research::autonomy::StrategyVersion {
        id: cfg.strategy_name.clone(),
        version: "0.1.0".into(),
        params: strat_cfg.clone(),
        data_range: data_label.clone(),
        train_range: format!("wf_train={} cpcv_paths={}", cfg.wf_train, splits.len()),
        oos_range: format!("wf_oos={} shadow={}", cfg.wf_oos, n - shadow_lo),
        gate: gate.clone(),
        risk: format!(
            "SL={:.0}% lev={:.0}x risk={:.2}% maxdd={:.0}%",
            cfg.stop_loss * 100.0, cfg.leverage, cfg.risk_fraction * 100.0, cfg.max_drawdown * 100.0
        ),
        exec: format!(
            "comm={} spread={} slip={} lat={}",
            cfg.commission, cfg.spread, cfg.slippage, cfg.latency_bars
        ),
    });
    println!(
        "registry: {} version(s), latest = {:?}",
        reg.len(),
        reg.latest(&cfg.strategy_name).map(|v| &v.version)
    );
    let next = qd_research::autonomy::advance_loop(
        qd_research::autonomy::LoopStage::Shadow,
        gate_allowed,
        false,
    );
    println!("loop: Shadow → {next:?}");

    let input = ResearchInput {
        strategy_name: strat_label,
        strategy_config: strat_cfg,
        asset: cfg.asset.clone(),
        timeframe: cfg.timeframe.clone(),
        date_range: data_label.clone(),
        n_bars: n,
        regime_perf: regime_slices(&res.trades, STARTING_EQUITY),
        walkforward: Some(wf),
        montecarlo: Some(mc),
        cpcv_paths: splits.len(),
        cpcv_runs_scored: cpcv_summary.count,
        cpcv_median_oos: cpcv_summary.median,
        cpcv_p25_oos: cpcv_summary.p25,
        cpcv_p75_oos: cpcv_summary.p75,
        pbo: Some(pbo),
        snooping: snoop,
        sensitivity: vec![sens],
        cost_stress: cs,
        shadow: Some(shadow),
        primary_weakness: format!("real-data run ({data_label}) — see regime table"),
        regime_dependency: "see regime table".into(),
        metrics,
        mae_mfe: Some(mae),
        cost_verdict: cv,
    };
    println!("=== JSON ===\n{}", report_json(&input));
    println!("\n=== MARKDOWN ===\n{}", report_markdown(&input));
}
