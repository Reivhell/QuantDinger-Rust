//! `autonomy_check <research.json>` — autonomous risk-control check on REAL
//! market data.
//!
//! Body/brain contract: the Python backend (`backend_api_python/`) is the
//! brain — it owns scheduling and order submission. This binary is the body:
//! it runs the configured strategy twice over the same REAL bars (once
//! unguarded, once through the full autonomous pipeline: entry gate →
//! kill-switch → emergency halt → monitor), then prints the §13 deployment
//! verdict, a version-registry record, and one research-loop pass.
//! No-trade outcomes are reported as idle evidence, never errors.
//!
//! Scope is the safety stack only (gate, monitor, kill-switch, emergency).
//! The validation stages (WF, MC, PBO, sensitivity, shadow live in
//! `research_run`) are absent here, so their evidence feeds `None` and the
//! gate fails closed — the expected verdict is BLOCKED: proof the gate
//! refuses to bless an unvalidated run.

use qd_research::autonomy::{
    advance_loop, assess_entry, gate_signals, monitor_position, EmergencyState,
    EntryCtx, EntryVerdict, LoopStage, StrategyVersion, VersionRegistry,
};
use qd_research::backtest::{run_backtest, Signal};
use qd_research::costs::{cost_stress, execution_sensitivity};
use qd_research::regime::{detect, Regime, RegimeConfig};
use qd_research::risk::KillSwitch;
use qd_research::run::{self, PERIODS_PER_YEAR, STARTING_EQUITY};

fn main() {
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
    let (bars, data_label) =
        run::load_bars(&cfg).unwrap_or_else(|e| panic!("{e}"));
    let n = bars.len();
    let regimes = detect(&bars, &vec![1i64; n], &RegimeConfig::default());
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    // Strategy proposes: RAW EMA cross with NO regime filter. The strategy
    // layer is deliberately uncooperative — §6 requires the deterministic
    // gate (not strategy goodwill) to enforce regime discipline.
    let ef = qd_research::features::ema(&closes, cfg.ema_fast);
    let es = qd_research::features::ema(&closes, cfg.ema_slow);
    let raw_signals: Vec<Signal> = (0..n)
        .map(|i| match (ef[i], es[i]) {
            (Some(a), Some(b)) if a > b => Signal::Long,
            (Some(a), Some(b)) if a < b => Signal::Short,
            _ => Signal::Flat,
        })
        .collect();
    let feature_ids = vec![1u64; n];
    let exec = run::exec_of(&cfg);
    let (risk, stop) = (cfg.risk_fraction, cfg.stop_loss);
    // Adaptive sizer: fixed-fractional base scaled by trailing vol,
    // realized drawdown, and position crowding — never inflated.
    let vols =
        qd_research::features::realized_volatility(&closes, run::VOL_WINDOW, PERIODS_PER_YEAR);
    let size = |px: f64, eq: f64| {
        // NOTE: the closure sizer sees only (price, equity); the check feeds
        // the latest trailing vol + zero book crowding. The full stateful
        // path (live DD + open count) is exercised in unit tests.
        let rv = vols.last().copied().flatten();
        qd_research::risk::adaptive_qty(
            px,
            eq,
            qd_research::risk::AdaptiveSizeCtx {
                risk_fraction: risk,
                stop_fraction: stop,
                realized_vol_ann: rv,
                target_vol_ann: run::TARGET_VOL_ANN,
                current_dd: 0.0,
                max_dd_allowance: cfg.max_drawdown,
                open_positions: 0,
                max_positions: cfg.max_positions,
                max_position_notional: run::MAX_POSITION_NOTIONAL,
            },
        )
    };

    // Baseline: raw signals, no autonomous control.
    let base = run_backtest(&bars, &raw_signals, &regimes, &feature_ids, &exec, STARTING_EQUITY, size);
    let raw_longs = raw_signals.iter().filter(|s| **s != Signal::Flat).count();

    // Autonomous run: the multi-layer gate filters every bar BEFORE it can
    // become a fill; a kill-switch + emergency halt watch realized equity.
    // Blocked regimes mirror the reference strategy's discipline — now
    // enforced even though the signal layer ignores regime entirely.
    let blocked = run::blocked_regimes();
    let lev_ok = run::leverage_ok(&cfg);
    let gate = EntryCtx {
        data_ok: true,
        emergency_halted: false,
        risk_halted: false,
        blocked_regimes: &blocked,
        open_positions: 0,
        max_positions: cfg.max_positions,
        open_risk: 0.0,
        max_open_risk: cfg.max_open_risk,
        leverage_ok: lev_ok,
        stop_present: cfg.stop_loss > 0.0,
        execution_ok: true,
    };
    let (gated_signals, suppressed) = gate_signals(&raw_signals, &regimes, &gate);
    let auto = run_backtest(&bars, &gated_signals, &regimes, &feature_ids, &exec, STARTING_EQUITY, size);

    // Realized-risk watch: kill-switch over the autonomous equity curve.
    let mut ks = KillSwitch::with_weekly(
        cfg.max_daily_loss,
        cfg.max_weekly_loss,
        cfg.max_drawdown,
        cfg.max_exposure,
        cfg.max_correlated_exposure,
        run::WEEK_BARS,
    );
    let mut ks_trip: Option<String> = None;
    for (i, eq) in auto.equity_curve.iter().enumerate() {
        let new_day = i % run::SESSION_BARS == 0; // paper daily session marker
        if let Some(r) = ks.update(*eq, 0.0, new_day) {
            ks_trip = Some(r);
            break;
        }
    }
    // Emergency snapshot: healthy here → proves the check path runs.
    let mut em = EmergencyState::default();
    let em_reason = em.check(true, true, true, true, true, true, true);

    // Monitor pass over autonomous trades: what the manager would have done.
    let mut holds = 0usize;
    let mut exits = 0usize;
    let mut tightens = 0usize;
    let mut reduces = 0usize;
    for t in &auto.trades {
        let pnl = t.net_pnl / STARTING_EQUITY;
        let dts = cfg.stop_loss + pnl; // adverse drift eats the stop buffer
        // Thesis abort: the position exited into ABNORMAL (integrity
        // uncertain) or a counter-trend stretch (MEAN_REVERSION) — the
        // trend thesis that opened it no longer holds.
        let thesis_aborted = matches!(
            t.exit_regime,
            Some(Regime::Abnormal | Regime::MeanReversion)
        );
        let vol_extreme = t.entry_regime == Regime::HighVolatility
            || matches!(t.exit_regime, Some(Regime::HighVolatility | Regime::Abnormal));
        match monitor_position(pnl, dts, thesis_aborted, false, vol_extreme) {
            qd_research::autonomy::MonitorAction::Hold => holds += 1,
            qd_research::autonomy::MonitorAction::Exit(_) => exits += 1,
            qd_research::autonomy::MonitorAction::TightenStop => tightens += 1,
            qd_research::autonomy::MonitorAction::Reduce => reduces += 1,
        }
    }

    let b_ret = base.equity_curve.last().copied().unwrap_or(STARTING_EQUITY) / STARTING_EQUITY - 1.0;
    let a_ret = auto.equity_curve.last().copied().unwrap_or(STARTING_EQUITY) / STARTING_EQUITY - 1.0;
    println!("=== AUTONOMOUS RISK CHECK [{data_label}] ===");
    println!("baseline: {} signals → {} trades, return {:.2}%", raw_longs, base.trades.len(), b_ret * 100.0);
    println!(
        "gated:    {} trades ({} suppressed as idle), return {:.2}%",
        auto.trades.len(),
        suppressed,
        a_ret * 100.0
    );
    println!("kill-switch: {}", ks_trip.as_deref().unwrap_or("not tripped"));
    println!("emergency: {}", em_reason.as_deref().unwrap_or("healthy"));
    println!("monitor: {holds} hold / {tightens} tighten / {reduces} reduce / {exits} exit");

    // §13 deployment gate via the shared evidence mapping (fail-closed).
    // Honest scope: this check exercises the safety stack (gate, monitor,
    // kill-switch, emergency) — NOT the validation stages (WF, MC, PBO,
    // sensitivity, shadow live in research_run). Missing stages feed None
    // and fail closed, so the expected verdict here is BLOCKED: proof the
    // gate refuses to bless an unvalidated run.
    let m_auto = qd_research::metrics::compute_metrics(&auto.trades, &auto.equity_curve, STARTING_EQUITY, PERIODS_PER_YEAR);
    let gc: Vec<(f64, f64)> =
        auto.trades.iter().map(|t| (t.gross_pnl, t.fees + t.slippage_cost)).collect();
    let cs = cost_stress(&gc, &run::COST_MULTIPLIERS);
    let dd_ok = auto.trades.iter().all(|t| t.mae <= cfg.stop_loss * 2.0);
    let gate13 = qd_research::gate::gate_from_evidence(&qd_research::gate::GateEvidence {
        realistic_costs: cfg.commission > 0.0 && cfg.slippage > 0.0,
        walkforward: None, // not run in this check → fails closed
        monte_carlo: None, // not run in this check → fails closed
        backtest_max_dd: m_auto.max_drawdown,
        sensitivity: None, // not run in this check → fails closed
        pbo: None,         // not run in this check → fails closed
        cost_verdict: execution_sensitivity(&cs).to_string(),
        max_drawdown_cap: cfg.max_drawdown,
        stop_on_every_trade: cfg.stop_loss > 0.0 && dd_ok,
        leverage_ok: lev_ok,
        killswitch_verified: true, // KS watched the full equity curve above
        snooping: None, // not run in this check → fails closed
        shadow: None, // no holdout in this check → fails closed
    });
    println!(
        "deploy: {} {:?}",
        if gate13.deploy_allowed() { "ALLOWED" } else { "BLOCKED" },
        gate13.failures()
    );

    // §14 versioned record (append-only).
    let mut reg = VersionRegistry::default();
    reg.publish(StrategyVersion {
        id: cfg.strategy_name.clone(),
        version: "0.1.0".into(),
        params: format!("fast={} slow={}", cfg.ema_fast, cfg.ema_slow),
        data_range: data_label.clone(),
        train_range: "wf_train".into(),
        oos_range: "wf_oos+cpcv".into(),
        gate: gate13.clone(),
        risk: format!("SL={:.0}% lev={:.0}x risk={:.2}%", cfg.stop_loss * 100.0, cfg.leverage, risk * 100.0),
        exec: format!("comm={} spread={} slip={}", cfg.commission, cfg.spread, cfg.slippage),
    });
    println!("registry: {} version(s), latest = {:?}", reg.len(), reg.latest(&cfg.strategy_name).map(|v| &v.version));

    // §12 one loop pass: validated → shadow.
    let next = advance_loop(LoopStage::Validate, gate13.deploy_allowed(), false);
    println!("loop: Validate → {next:?}");

    // Proof the per-bar verdict path is reachable (not just the batch gate).
    let probe = assess_entry(Signal::Long, Regime::HighVolatility, &gate);
    assert_eq!(probe, EntryVerdict::NoTrade(qd_research::autonomy::NO_TRADE_REGIME));
    println!("probe: Long-in-HIGH_VOL correctly → NoTrade(regime_blocked)");
}
