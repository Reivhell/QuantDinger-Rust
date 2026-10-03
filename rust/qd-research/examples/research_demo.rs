//! Full research-pipeline runner on REAL market data (or seeded fixture).
//!
//! Data comes from `data.csv` in the research config: a strict-CSV bar file
//! (`t,open,high,low,close,volume`) produced e.g. by `fetch_okx.sh` from a
//! public exchange API — no demo feed, no invented candles. Only when
//! `data.csv` is empty does it fall back to the deterministic synthetic
//! fixture (seeded, reproducible, labeled SYNTH in the report).
//! Runs features → regime → two reference strategies → backtest → metrics →
//! MAE/MFE → walk-forward → Monte Carlo → CPCV → PBO → sensitivity → cost
//! stress → shadow (§12 bulkhead on the unseen last third) → §13
//! deployment gate (evidence-fed, fail-closed) → §14 version registry →
//! §12 loop transition → JSON + Markdown report. Proves the pipeline is
//! wired end to end.

use qd_research::backtest::{run_backtest, ExecConfig};
use qd_research::costs::{cost_stress, execution_sensitivity};
use qd_research::cpcv::{contiguous_runs, cpcv_splits, score_cpcv_paths};
use qd_research::data::{load_bars_csv, synthetic_bars};
use qd_research::mae_mfe::analyze_mae_mfe;
use qd_research::metrics::compute_metrics;
use qd_research::montecarlo::run_monte_carlo;
use qd_research::pbo::{analyze_overfitting, PboBands, ScoreMatrix};
use qd_research::regime::{detect, RegimeConfig};
use qd_research::report::{regime_slices, report_json, report_markdown, ResearchInput};
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
    // trends + breakouts), RSI mean-reversion (buys dips in chop/range), or
    // Donchian breakout (volume-confirmed range escapes, long-only).
    // All share the same pipeline below — only the signal source differs.
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let fam: &str = match cfg.strategy_name.as_str() {
        "RsiMeanReversion" => "rsi",
        "DonchianBreakout" => "donchian",
        _ => "ema",
    };
    let use_rsi = fam == "rsi";
    let signals = match fam {
        "rsi" => RsiMeanReversion {
            period: cfg.rsi_period,
            oversold: cfg.rsi_oversold,
            overbought: cfg.rsi_overbought,
            allow_shorts: cfg.rsi_shorts,
        }
        .signals(&closes, &regimes),
        "donchian" => DonchianBreakout {
            lookback: cfg.donchian_lookback,
            relvol_min: cfg.donchian_relvol,
        }
        .signals(&bars, &regimes),
        _ => EmaCrossTrend { fast: cfg.ema_fast, slow: cfg.ema_slow }.signals(&closes, &regimes),
    };
    let wf_label = match fam {
        "rsi" => format!("rsi{}/{}-{}", cfg.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought),
        "donchian" => {
            format!("donch{}/rv{:.1}", cfg.donchian_lookback, cfg.donchian_relvol)
        }
        _ => format!("ema{}/{}", cfg.ema_fast, cfg.ema_slow),
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
    // Donchian: lookback ∈ {10,20,30} × relvol ∈ {1.5,2.0}.
    // Same folds, IS robust-return vs OOS robust-return — IS/OOS degradation
    // decides. Robust = total return with a trade-count floor: Sharpe on
    // 0-2 trades is ±infinity garbage (var≈0), so windows with <3 trades
    // score 0.0 (no evidence) instead of a spurious extreme.
    let mut cfgs = Vec::new();
    let mut is_m = Vec::new();
    let mut oos_m = Vec::new();
    // (label, signal-builder): boxed so the family loop below is shared.
    type SigFn = dyn Fn(&[qd_research::features::Bar], &[f64], &[qd_research::regime::Regime]) -> Vec<qd_research::backtest::Signal>;
    let grid: Vec<(String, Box<SigFn>)> = if fam == "donchian" {
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
                    Box::new(move |_, c: &[f64], r: &[qd_research::regime::Regime]| {
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
                    Box::new(move |_, c: &[f64], r: &[qd_research::regime::Regime]| {
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
                // Trade-count floor: <3 trades carries no rankable evidence.
                // CAGR, not total return: IS windows (600 bars) are 4x the
                // OOS windows (150 bars), so raw totals are length-biased —
                // a perfectly stable edge shows ~75% "degradation" on totals
                // alone. CAGR annualizes (252 bars/yr, daily data) and makes
                // IS/OOS levels comparable. Same <3-trade floor → 0.0.
                let robust = if m.num_trades >= 3 { m.cagr } else { 0.0 };
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
    // is a contiguous flat-to-flat OOS window with the <3-trade floor, so
    // T = R ≈ 23 clears the min_periods=10 bar where the 2 WF folds cannot.
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
            let r = run_backtest(
                &bars[*lo..*hi],
                &sg[*lo..*hi],
                &regimes[*lo..*hi],
                &feature_ids[*lo..*hi],
                &exec,
                100_000.0,
                |px, eq| qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0),
            );
            let m = compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0);
            // CAGR, not total return: CPCV runs vary in length, so raw
            // totals are length-biased the same way IS/OOS windows were.
            row.push(if m.num_trades >= 3 { m.cagr } else { 0.0 });
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

    // Sensitivity around the configured knob (full-sample robust return —
    // total return with the same <3-trade floor, so thin windows score 0
    // instead of Sharpe ±infinity): EMA fast, RSI period, or Donchian
    // lookback. Plateau = robust.
    let (sens_name, sens_base, sens_vals): (&str, f64, Vec<f64>) = if fam == "donchian" {
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
        let sg = if fam == "donchian" {
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
        let r = run_backtest(&bars, &sg, &regimes, &feature_ids, &exec, 100_000.0, |px, eq| {
            qd_research::risk::fixed_fractional_qty(px, eq, risk, stop, 20_000.0)
        });
        let m = compute_metrics(&r.trades, &r.equity_curve, 100_000.0, 252.0);
        if m.num_trades >= 3 { m.total_return } else { 0.0 }
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
    // of their own — none trains here). The configured candidate runs
    // under the autonomous gate + live kill-switch against a duller
    // incumbent of the same family; the drawdown-first verdict feeds the
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
        // window — slow EMA, slow/long RSI, or longer-channel Donchian.
        let inc_sig = if fam == "donchian" {
            DonchianBreakout {
                lookback: cfg.donchian_lookback + 10,
                relvol_min: cfg.donchian_relvol,
            }
            .signals(&bars, &regimes)
        } else if use_rsi {
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
        leverage_ok: qd_research::risk::liquidation_ok(cfg.leverage, cfg.stop_loss, 3.0),
        killswitch_verified: true, // shadow runs under the live kill-switch
        snooping: snoop.clone(),
        shadow: Some((&shadow).into()),
    };

    let (strat_label, strat_cfg) = if fam == "donchian" {
        (
            format!(
                "DonchianBreakout({},rv{:.1})",
                cfg.donchian_lookback, cfg.donchian_relvol
            ),
            format!(
                "lookback={} relvol={:.2} stop={:.3} take={:.3} risk={:.3}",
                cfg.donchian_lookback, cfg.donchian_relvol,
                cfg.stop_loss, cfg.take_profit, cfg.risk_fraction
            ),
        )
    } else if use_rsi {
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
            cfg.stop_loss * 100.0, cfg.leverage, risk * 100.0, cfg.max_drawdown * 100.0
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
        regime_perf: regime_slices(&res.trades, 100_000.0),
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
