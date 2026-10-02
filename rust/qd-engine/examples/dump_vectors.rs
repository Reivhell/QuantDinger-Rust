//! Dump deterministic computation vectors as JSON for cross-checking
//! against the Python originals (see `../parity/check.py`).
//!
//! Run: `cargo run --example dump_vectors` (from `rust/`).

use qd_engine::close_reason::{self, ExecRowIn, NumVal, TradeRowIn};
use qd_engine::{
    curve_sampling::{self, CurvePoint},
    frequencies, grid, indicators, instruments, net_pnl, perf_metrics, pnl, precise,
    protection, risk_guard,
};
use qd_engine::net_pnl::TradeRow;

fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else {
        "null".to_string()
    }
}

fn opt(o: Option<f64>) -> String {
    match o {
        Some(v) => num(v),
        None => "null".to_string(),
    }
}

fn opt_list(xs: &[Option<f64>]) -> String {
    let parts: Vec<String> = xs.iter().map(|o| opt(*o)).collect();
    format!("[{}]", parts.join(","))
}

/// JSON-escape a string (UTF-8 passes through; only `"` `\` + controls escaped).
fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jopt_num(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:?}"),
        Some(_) => "\"nonfinite\"".to_string(),
        None => "null".to_string(),
    }
}

fn jopt_str(v: Option<&str>) -> String {
    match v {
        Some(s) => jstr(s),
        None => "null".to_string(),
    }
}

fn main() {
    // Pseudo-random but deterministic OHLC series.
    let mut high = vec![0.0f64; 40];
    let mut low = vec![0.0f64; 40];
    let mut close = vec![0.0f64; 40];
    let mut seed: u64 = 0x1234_5678_9abc_def1;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64)
    };
    let mut px = 100.0;
    for i in 0..40 {
        let drift = (rng() - 0.48) * 4.0;
        px += drift;
        let h = px + rng() * 1.5;
        let l = px - rng() * 1.5;
        high[i] = h;
        low[i] = l;
        close[i] = px;
    }

    let (k, d, j) = indicators::compute_kdj_cn(&high, &low, &close, 9, 3, 3);
    let rsi = indicators::compute_rsi_wilder(&close, 14);
    // Second parameter set to catch smoothing-path bugs.
    let (k2, d2, j2) = indicators::compute_kdj_cn(&high, &low, &close, 14, 2, 4);
    let rsi2 = indicators::compute_rsi_wilder(&close, 7);

    let mut out = String::from("{\n");
    // Exact input series (shortest round-trip repr parses back bit-identically
    // in Python) so check.py can feed the REAL pandas codegen fragments.
    let sh: Vec<String> = high.iter().map(|v| num(*v)).collect();
    let sl: Vec<String> = low.iter().map(|v| num(*v)).collect();
    let sc: Vec<String> = close.iter().map(|v| num(*v)).collect();
    out += &format!("\"series_high\":[{}],\n", sh.join(","));
    out += &format!("\"series_low\":[{}],\n", sl.join(","));
    out += &format!("\"series_close\":[{}],\n", sc.join(","));
    out += &format!("\"kdj_k\":{},\n", opt_list(&k));
    out += &format!("\"kdj_d\":{},\n", opt_list(&d));
    out += &format!("\"kdj_j\":{},\n", opt_list(&j));
    out += &format!("\"rsi\":{},\n", opt_list(&rsi));
    out += &format!("\"kdj_k_14_2_4\":{},\n", opt_list(&k2));
    out += &format!("\"kdj_d_14_2_4\":{},\n", opt_list(&d2));
    out += &format!("\"kdj_j_14_2_4\":{},\n", opt_list(&j2));
    out += &format!("\"rsi_7\":{},\n", opt_list(&rsi2));

    // PnL vectors.
    let cases = [
        ("long", 100.0, 110.0, 2.0),
        ("short", 110.0, 100.0, 2.0),
        ("long", 110.0, 100.0, 2.0),
        ("SHORT", 50.5, 49.25, 10.0),
        ("long", 0.0, 100.0, 1.0),
        ("long", 100.0, 100.0, -1.0),
    ];
    let up: Vec<String> = cases
        .iter()
        .map(|(s, e, c, z)| num(pnl::calc_unrealized_pnl(s, *e, *c, *z)))
        .collect();
    out += &format!("\"unrealized\":[{}],\n", up.join(","));
    let pct: Vec<String> = [
        (100.0, 2.0, 20.0, 1.0, "spot"),
        (100.0, 2.0, 20.0, 10.0, "swap"),
        (100.0, 2.0, -7.5, 5.0, "perp"),
        (100.0, 2.0, 20.0, 10.0, "spot"),
        (0.0, 1.0, 10.0, 1.0, "spot"),
        (100.0, 2.0, 20.0, 0.0, "futures"),
    ]
    .iter()
    .map(|(e, s, p, l, m)| num(pnl::calc_pnl_percent(*e, *s, *p, *l, m)))
    .collect();
    out += &format!("\"pnl_pct\":[{}],\n", pct.join(","));
    let marg: Vec<String> = [(200.0, 10.0, "swap"), (200.0, 10.0, "spot"), (200.0, 0.0, "perp")]
        .iter()
        .map(|(n, l, m)| num(pnl::calc_margin_notional(*n, *l, m)))
        .collect();
    out += &format!("\"margin\":[{}],\n", marg.join(","));

    // precise vectors.
    let floors = [
        ("0.00009999999999999994", "0.0001"),
        ("1.234567", "0.01"),
        ("5", "2"),
        ("0.00009", "0.0001"),
        ("123.456", "0.5"),
        ("0.1", "0.3"),
        ("10", "3"),
        ("1.5", "0"),
        ("-3", "0.5"),
        ("1e-7", "1e-8"),
    ];
    let fl: Vec<String> = floors
        .iter()
        .map(|(v, s)| format!("\"{}\"", precise::floor_to_step_str(v, s)))
        .collect();
    out += &format!("\"floors\":[{}],\n", fl.join(","));
    let cl: Vec<String> = [1.2345678901234567, 2.5, 0.0, -2.5, 100.0]
        .iter()
        .map(|v| num(precise::clean_number(*v, 12)))
        .collect();
    out += &format!("\"clean\":[{}],\n", cl.join(","));
    let fm: Vec<String> = [1.23000, 0.0, 100.0, 1e-7, -0.0]
        .iter()
        .map(|v| format!("\"{}\"", precise::format_decimal(*v, 12)))
        .collect();
    out += &format!("\"format\":[{}],\n", fm.join(","));

    // risk_guard vectors.
    let fees: Vec<String> = [0.001, -0.5, 0.1, 0.02, 5.0]
        .iter()
        .map(|v| num(risk_guard::coerce_fee_rate(Some(*v), 0.001)))
        .collect();
    out += &format!("\"fees\":[{}],\n", fees.join(","));
    out += &format!(
        "\"fee_none\":{},\n",
        num(risk_guard::coerce_fee_rate(None, 0.001))
    );
    let trail: Vec<String> = [
        ("long", 100.0, 100.3, 0.001, 0.0),
        ("long", 100.0, 100.1, 0.001, 0.0),
        ("short", 100.0, 99.7, 0.001, 0.0),
        ("short", 100.0, 99.9, 0.001, 0.0),
        ("long", 100.0, 100.1, 0.001, 0.005),
        ("sideways", 100.0, 200.0, 0.001, 0.0),
    ]
    .iter()
    .map(|(s, e, x, f, b)| {
        if risk_guard::trailing_exit_locks_net_profit(s, *e, *x, *f, *b) {
            "true".to_string()
        } else {
            "false".to_string()
        }
    })
    .collect();
    out += &format!("\"trail\":[{}],\n", trail.join(","));

    // net_pnl vectors.
    let syms = ["btcusdt", "BTC/USDT", "BTC/USDT:USDT", "eth-usd", "SOLUSDT@BINANCE"];
    let ns: Vec<String> = syms
        .iter()
        .map(|s| format!("\"{}\"", net_pnl::normalize_strategy_symbol(s)))
        .collect();
    out += &format!("\"symbols\":[{}],\n", ns.join(","));

    let mut trades = vec![
        TradeRow::new(1, "ETH/USDT", "open_long", 1.0, 10.0, None, 1),
        TradeRow::new(2, "ETH/USDT", "add_long", 1.0, 8.0, None, 2),
        TradeRow::new(3, "ETH/USDT", "reduce_long", 1.5, 2.0, Some(30.0), 3),
        TradeRow::new(4, "BTC/USDT", "open_short", 0.5, 1.0, None, 4),
        TradeRow::new(5, "BTC/USDT", "close_short", 0.5, 0.5, Some(12.0), 5),
    ];
    trades[3].commission_quote = Some(2.0);
    net_pnl::enrich_trades_net_pnl(&mut trades);
    let er: Vec<String> = trades
        .iter()
        .map(|r| {
            format!(
                "{{\"id\":{},\"profit_gross\":{},\"open\":{},\"close_comm\":{},\"total\":{},\"net\":{},\"profit\":{}}}",
                r.id,
                opt(r.profit_gross),
                opt(r.open_commission_allocated),
                opt(r.close_commission),
                opt(r.total_commission),
                opt(r.net_pnl),
                opt(r.profit),
            )
        })
        .collect();
    out += &format!("\"enriched\":[{}],\n", er.join(","));
    let eq: Vec<String> = trades
        .iter()
        .map(|r| num(net_pnl::net_pnl_for_equity_step(r)))
        .collect();
    out += &format!("\"equity\":[{}],\n", eq.join(","));

    // grid vectors.
    let arith = grid::generate_levels(100.0, 200.0, 5, "arithmetic");
    let geo = grid::generate_levels(100.0, 400.0, 3, "geometric");
    let geo5 = grid::generate_levels(50.0, 500.0, 5, "geometric");
    let la: Vec<String> = arith.iter().map(|v| num(*v)).collect();
    let lg: Vec<String> = geo.iter().map(|v| num(*v)).collect();
    let lg5: Vec<String> = geo5.iter().map(|v| num(*v)).collect();
    out += &format!("\"levels_arith\":[{}],\n", la.join(","));
    out += &format!("\"levels_geo\":[{}],\n", lg.join(","));
    out += &format!("\"levels_geo5\":[{}],\n", lg5.join(","));
    let cells = grid::generate_cells(&arith);
    let cs: Vec<String> = cells
        .iter()
        .map(|c| format!("[{},{}]", num(c.lower_price), num(c.upper_price)))
        .collect();
    out += &format!("\"cells\":[{}],\n", cs.join(","));

    // protection vectors: 3 modes x 4 configs x (8 bar probes + 5 tick probes).
    // Timestamps are epoch seconds; T0 = 2026-01-01 00:00:00 UTC.
    const T0: i64 = 1767225600;
    let specs: [(f64, f64, f64, f64, i64, bool); 4] = [
        // (sl, tp, tsl, tact, limit_secs, rebase)
        (0.02, 0.05, 0.0, 0.0, 0, true),
        (0.02, 0.05, 0.0, 0.0, 0, false),
        (0.0, 0.0, 0.003, 0.01, 0, false),
        (0.0, 0.0, 0.0, 0.0, 3600, true),
    ];
    let modes = ["conservative", "aggressive", "balanced"];
    // bar scenario: gap-down stop, inside-bar stop, take-profit bar, quiet bar
    let bars: [(f64, f64, f64); 4] = [
        (95.0, 96.0, 94.0),
        (100.0, 101.0, 97.0),
        (100.0, 106.0, 99.0),
        (100.0, 101.0, 99.0),
    ];
    let ticks: [f64; 5] = [100.5, 101.5, 102.0, 101.6, 97.5];
    let mut prot: Vec<String> = Vec::new();
    for mode in modes {
        for (si, (sl, tp, tsl, tact, lim, rebase)) in specs.iter().copied().enumerate() {
            let spec = protection::ProtectionSpec::new(sl, tp, tsl, tact, lim, rebase);
            let eng = protection::ProtectionEngine::new(mode);
            for (bi, (o, h, l)) in bars.iter().enumerate() {
                let tag = format!("{mode}:s{si}:b{bi}");
                for side in ["long", "short"] {
                    let mut st =
                        protection::ProtectionState::open("S", side, 100.0, spec.clone(), T0);
                    match eng.evaluate_bar(&mut st, T0 + 4 * 3600, *o, *h, *l) {
                        Some(x) => prot.push(format!(
                            "\"{tag}:{side}:{}:{}:{}\"",
                            x.reason, x.price, x.trigger_price
                        )),
                        None => prot.push(format!(
                            "\"{tag}:{side}:none:{}:{}\"",
                            st.highest_price, st.lowest_price
                        )),
                    }
                }
            }
            // tick scenario: state carries across ticks like a live session.
            let mut st =
                protection::ProtectionState::open("S", "long", 100.0, spec.clone(), T0);
            for (ti, px) in ticks.iter().enumerate() {
                match eng.evaluate_price(&mut st, T0 + 60 * (ti as i64 + 1), *px) {
                    Some(x) => prot.push(format!(
                        "\"{mode}:s{si}:t{ti}:long:{}:{}:{}\"",
                        x.reason, x.price, x.trigger_price
                    )),
                    None => prot.push(format!(
                        "\"{mode}:s{si}:t{ti}:long:none:{}:{}\"",
                        st.highest_price, st.lowest_price
                    )),
                }
            }
        }
    }
    out += &format!("\"protection\":[{}],\n", prot.join(","));

    // close_reason vectors: all 26 known codes x zh/en, edge inputs.
    let codes = [
        "long_entry",
        "long_exit",
        "short_entry",
        "short_exit",
        "grid_initial_long",
        "grid_initial_short",
        "grid_reduce_long",
        "grid_reduce_short",
        "grid_close_all",
        "grid_waterfall_close",
        "grid_equity_stop_loss",
        "grid_equity_take_profit",
        "grid_equity_trailing_stop",
        "dca_equity_stop_loss",
        "dca_equity_take_profit",
        "dca_equity_trailing_stop",
        "robot_equity_stop_loss",
        "robot_equity_take_profit",
        "robot_equity_trailing_stop",
        "grid_out_of_bounds_up",
        "grid_out_of_bounds_down",
        "server_stop_loss",
        "server_take_profit",
        "server_trailing_stop",
        "indicator_signal",
        "signal_trigger",
    ];
    let mut labels: Vec<String> = Vec::new();
    for c in codes {
        labels.push(format!(
            "[{},{},{}]",
            jstr(c),
            jstr(&close_reason::label_for_reason(c, "zh")),
            jstr(&close_reason::label_for_reason(c, "en"))
        ));
    }
    out += &format!("\"labels\":[{}],\n", labels.join(","));

    // enrich_trade_row probes: (type, close_reason, bot_type, lang).
    let enr_cases = [
        ("close_long", "server_stop_loss", "grid", "zh"),
        ("close_short", "grid_waterfall_close", "grid", "zh"),
        ("close_long", "", "", "zh"),
        ("close_long", "server_take_profit", "", "zh"),
        ("open_long", "long_entry", "grid", "zh"),
        ("close_long", "", "grid", "zh"),
        ("close_short", "", "dca", "en"),
        ("CLOSE_LONG", "", "GRID", "zh"),
        ("open_short", "", "grid", "zh"),
        ("reduce_long", "custom_reason", "", "en"),
    ];
    let mut enr: Vec<String> = Vec::new();
    for (tt, cr, bt, lang) in enr_cases {
        let row = close_reason::enrich_trade_row(&TradeRowIn {
            trade_type: tt.into(),
            close_reason: cr.into(),
            bot_type: bt.into(),
            lang: lang.into(),
            matched_entry_price: Some(NumVal::Text("12.5".into())),
            grid_matched_profit: Some(NumVal::Text("bad".into())),
        });
        enr.push(format!(
            "[{},{},{},{},{},{},{}]",
            jstr(tt),
            jstr(cr),
            jstr(bt),
            jstr(&row.close_reason),
            jstr(&row.action_note),
            jopt_num(row.matched_entry_price),
            jopt_num(row.grid_matched_profit)
        ));
    }
    out += &format!("\"enrich_row\":[{}],\n", enr.join(","));

    // infer/resolve probes.
    let mut inf: Vec<String> = Vec::new();
    for (tt, bt) in [
        ("close_long", "grid"),
        ("close_short", "grid"),
        ("close_long", "dca"),
        ("open_long", "grid"),
        ("close_long", ""),
    ] {
        inf.push(format!(
            "[{},{},{}]",
            jstr(tt),
            jstr(bt),
            jstr(&close_reason::infer_legacy_close_reason(tt, bt, ""))
        ));
    }
    out += &format!("\"infer\":[{}],\n", inf.join(","));

    // execution_reference probes.
    let exec_cases: Vec<(&str, &str, &str, &str, &str)> = vec![
        ("100", "", "", "101", "open_long"),            // grid limit ref
        ("", r#"{"ref_price": 200.0}"#, "", "198", "open_short"), // signal ref
        (
            "",
            r#"{"client_order_id": "abc", "limit_price": 50.0, "price": 51.0}"#,
            "",
            "50.5",
            "close_long",
        ), // instruction chain
        ("", r#"{"price": "75.25"}"#, "", "75.25", "add_long"), // string payload price
        ("", "not-json{{{", "", "10", "open_long"),      // bad payload
        ("", "", "", "10", "open_long"),                 // unknown ref
        ("", "", "99.5", "100", "close_short"),          // pending grid ref
        ("0", r#"{"ref_price": 0}"#, "", "10", "open_long"), // non-positive refs
    ];
    let mut exr: Vec<String> = Vec::new();
    for (grid_px, payload, pending, px, tt) in exec_cases {
        let row = close_reason::enrich_execution_reference(&ExecRowIn {
            request_payload_json: if payload.is_empty() {
                None
            } else {
                Some(payload.to_string())
            },
            request_price: NumVal::Text(pending.to_string()),
            grid_request_price: NumVal::Text(grid_px.to_string()),
            grid_client_reference: String::new(),
            price: NumVal::Text(px.to_string()),
            trade_type: tt.to_string(),
        });
        exr.push(format!(
            "[{},{},{},{},{}]",
            jstr(tt),
            jstr(&row.grid_client_reference),
            jopt_num(row.reference_price),
            jopt_str(row.reference_kind.as_deref()),
            jopt_num(row.price_deviation_pct)
        ));
    }
    out += &format!("\"exec_ref\":[{}],\n", exr.join(","));

    // frequencies vectors: labels x markets for periods, assorted lookups.
    let freq_labels = [
        "1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "daily", "DAY",
        "m1", "h1", "d1", "monthly", "15分钟", "2小时", "", "tick", "1mo", "2w",
        "1.5h", "0m", "1_0m", " 4H ",
    ];
    let mut fr: Vec<String> = Vec::new();
    for f in freq_labels {
        fr.push(format!(
            "[{},{},{}]",
            jstr(f),
            jstr(&frequencies::normalize_frequency(f, "1d")),
            frequencies::frequency_seconds(f)
        ));
    }
    out += &format!("\"freqs\":[{}],\n", fr.join(","));
    // periods matrices: (frequency, markets) -> value-or-error.
    let period_cases: [(&str, &[&str]); 15] = [
        ("1h", &["Crypto"]),
        ("4h", &["Crypto"]),
        ("15m", &["Crypto"]),
        ("1h", &["Stocks"]),
        ("1d", &["Stocks"]),
        ("1w", &[]),
        ("1mo", &["Crypto"]),
        ("m", &["Crypto"]),
        ("h", &["Stocks"]),
        ("1.5h", &["Stocks"]),
        ("1d", &[]),
        ("1d", &["Crypto", "Stocks"]),
        ("2w", &["Crypto"]),
        ("90m", &["Stocks"]),
        ("24h", &["Crypto"]),
    ];
    let mut pe: Vec<String> = Vec::new();
    for (f, markets) in period_cases {
        let mkt = format!(
            "[{}]",
            markets.iter().map(|m| jstr(m)).collect::<Vec<_>>().join(",")
        );
        match frequencies::periods_per_year(f, markets) {
            Ok(v) => pe.push(format!("[{},{},{}]", jstr(f), mkt, num(v))),
            Err(e) => pe.push(format!("[{},{},\"ERR:{}\"]", jstr(f), mkt, e)),
        }
    }
    out += &format!("\"periods\":[{}],\n", pe.join(","));
    let mut uf: Vec<String> = Vec::new();
    let uniq_cases: [&[&str]; 3] = [&["1d", "4h", "1d", "1h"], &[], &["tick", "tock", "tick"]];
    for case in uniq_cases {
        let u = frequencies::unique_frequencies(case, "1d");
        let d = frequencies::driving_frequency(case, "1d");
        uf.push(format!(
            "[[{}],{}]",
            u.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","),
            jstr(&d)
        ));
    }
    out += &format!("\"uniqfreq\":[{}],\n", uf.join(","));

    // curve_sampling vectors: scenario id -> kept indices.
    // Scenarios: steady rise, rise+crash, insolvent tail, nan gap, stored-dd.
    let scenarios: Vec<(Vec<f64>, Vec<Option<f64>>, f64, usize)> = vec![
        ((0..25).map(|i| 100.0 + i as f64).collect(), vec![None; 25], 100.0, 10),
        (
            vec![
                100.0, 102.0, 105.0, 110.0, 108.0, 120.0, 115.0, 90.0, 80.0,
                85.0, 88.0, 92.0, 95.0, 97.0, 99.0, 101.0, 103.0, 104.0,
                106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
            ],
            vec![None; 25],
            100.0,
            10,
        ),
        (
            vec![
                100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0, 20.0, 5.0, 1.0,
                0.0, -3.0, -3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0,
            ],
            vec![None; 25],
            100.0,
            12,
        ),
        (
            {
                let mut v = vec![100.0; 25];
                v[7] = f64::NAN;
                v[8] = f64::INFINITY;
                v
            },
            vec![None; 25],
            100.0,
            10,
        ),
        (
            vec![100.0; 25],
            {
                let mut dd = vec![None; 25];
                dd[17] = Some(-42.0);
                dd
            },
            100.0,
            10,
        ),
    ];
    let mut cs: Vec<String> = Vec::new();
    for (si, (vals, dds, initial, limit)) in scenarios.iter().enumerate() {
        let items: Vec<CurvePoint> = vals
            .iter()
            .zip(dds.iter())
            .map(|(v, d)| CurvePoint { value: Some(*v), drawdown: *d })
            .collect();
        match curve_sampling::sample_equity_curve_indices(&items, *limit, *initial) {
            Ok(kept) => cs.push(format!(
                "[{si},{}]",
                kept.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
            )),
            Err(e) => cs.push(format!("[{si},\"ERR:{e}\"]")),
        }
    }
    out += &format!("\"curve\":[{}],\n", cs.join(","));

    // instruments vectors: (value, default_market) probes incl. error cases.
    let ins_cases: [(&str, &str); 24] = [
        ("600519.XSHG", ""),
        ("USStock:MSFT", ""),
        ("Crypto:BTCUSDT@okx:swap", ""),
        ("Crypto:BTC/USDT@swap", ""),
        ("Crypto:00700/HKD@gate:spot", ""),
        ("BTCUSDT", ""),
        ("000001", ""),
        ("00700.HK", ""),
        ("MSFT", "USStock"),
        ("MSFT", ""),
        ("Crypto:BTC/USDT", ""),
        ("Crypto:BTC/USDT@binance", ""),
        ("USStock:MSFT@nasdaq:spot", ""),
        ("Crypto:ETHUSDT", ""),
        ("600519.xshg", ""),
        ("crypto:btcusdt@OKX:SWAP", ""),
        ("POOL:MyPool", ""),
        ("000300.XBHS", ""),
        ("INDEX:CSI300", ""),
        ("", ""),
        ("!!!", ""),
        ("Crypto:", ""),
        ("USStock:", "USStock"),
        ("  Crypto:BTC/USDT@swap  ", ""),
    ];
    let mut ins: Vec<String> = Vec::new();
    for (v, dm) in ins_cases {
        match instruments::parse_instrument(v, dm) {
            Ok(s) => ins.push(format!(
                "[{},{},{},{},{},{},{}]",
                jstr(v),
                jstr(dm),
                jstr(&s.market),
                jstr(&s.symbol),
                jstr(&s.exchange_id),
                jstr(&s.market_type),
                jstr(&s.key())
            )),
            Err(e) => ins.push(format!("[{},{},\"ERR:{}\"]", jstr(v), jstr(dm), e)),
        }
    }
    out += &format!("\"instruments\":[{}],\n", ins.join(","));
    // market inference + index/pool probes.
    let mut im: Vec<String> = Vec::new();
    for s in ["BTC/USDT", "USDT", "600519.SH", "SPY", "", "!!!", "00700.HK", "000001"] {
        im.push(format!("[{},{}]", jstr(s), jstr(&instruments::infer_market(s))));
    }
    out += &format!("\"infer_market\":[{}],\n", im.join(","));
    let mut ix: Vec<String> = Vec::new();
    for s in ["INDEX:CSI300", "000300.XBHS", "Crypto:BTC/USDT", "000300.XSHG_INDEX", "POOL:ABC", "abc"] {
        ix.push(format!(
            "[{},{},{},{}]",
            jstr(s),
            if instruments::is_index_reference(s) { "true" } else { "false" },
            jstr(&instruments::normalize_index_reference(s)),
            match instruments::normalize_pool_reference(s) {
                Ok(p) => jstr(&p),
                Err(e) => format!("\"ERR:{e}\""),
            }
        ));
    }
    out += &format!("\"indexpool\":[{}],\n", ix.join(","));

    // perf_metrics vectors: curve scenarios x (frequency, market, factor).
    // Epoch base: 2026-01-05 00:00 UTC Mondays keep weekday equity gaps valid.
    const D: i64 = 86_400;
    const B: i64 = 1_767_225_600; // 2026-01-05 00:00:00 UTC (Monday)
    let mk_pf: Vec<(i64, f64)> =
        (0..10).map(|i| (B + i * D, 100.0 + i as f64 * 1.1 - (i % 3) as f64 * 0.7)).collect();
    let mk_bm: Vec<(i64, f64)> =
        (0..10).map(|i| (B + i * D, 100.0 + i as f64 * 0.6)).collect();
    // gapped + mismatched-timestamp + dupe/zero/negative variants
    let gap_pf: Vec<(i64, f64)> = vec![
        (B, 100.0), (B + D, 101.0), (B + 3 * D, 102.0), (B + 4 * D, 103.0),
    ];
    let gap_bm: Vec<(i64, f64)> = vec![
        (B, 200.0), (B + D, 200.5), (B + 3 * D, 201.0), (B + 4 * D, 202.0),
    ];
    let mm_pf: Vec<(i64, f64)> = vec![(B, 100.0), (B + D, 101.0), (B + 2 * D, 102.0)];
    let mm_bm: Vec<(i64, f64)> = vec![(B + D, 200.0), (B + 2 * D, 202.0), (B + 3 * D, 203.0)];
    let dirty_pf: Vec<(i64, f64)> = vec![
        (B, 100.0), (B, 110.0), (B + D, -5.0), (B + D, 0.0), (B + 2 * D, 112.0),
    ];
    let dirty_bm: Vec<(i64, f64)> = vec![(B, 50.0), (B + D, 51.0), (B + 2 * D, 52.0)];
    let curves: Vec<(&str, Vec<(i64, f64)>, Vec<(i64, f64)>)> = vec![
        ("clean10", mk_pf, mk_bm),
        ("gap", gap_pf, gap_bm),
        ("mismatch", mm_pf, mm_bm),
        ("dirty", dirty_pf, dirty_bm),
    ];
    let mut pm: Vec<String> = Vec::new();
    for (name, pf, bm) in &curves {
        for (freq, market, factor) in
            [("1d", "Crypto", 365.25), ("1d", "USStock", 252.0), ("1w", "Crypto", 52.0)]
        {
            let pf_pts: Vec<perf_metrics::LevelPoint> = pf
                .iter()
                .map(|(t, v)| perf_metrics::LevelPoint { epoch_secs: *t, value: *v })
                .collect();
            let bm_pts: Vec<perf_metrics::LevelPoint> = bm
                .iter()
                .map(|(t, v)| perf_metrics::LevelPoint { epoch_secs: *t, value: *v })
                .collect();
            // calendar path unavailable here (no exchange_calendars) -> false,
            // exactly like Python's ImportError fallback.
            match perf_metrics::calculate_information_ratio(
                &pf_pts, &bm_pts, freq, factor, market, None, &|_, _| false,
            ) {
                Ok(r) => pm.push(format!(
                    "[{},{},{},{},{},{},{},{},{},{},{}]",
                    jstr(name),
                    jstr(freq),
                    jstr(market),
                    num(factor),
                    jstr(r.status),
                    r.observations,
                    jopt_num(r.portfolio_return_annualized),
                    jopt_num(r.benchmark_return_annualized),
                    jopt_num(r.active_return_annualized),
                    jopt_num(r.tracking_error_annualized),
                    jopt_num(r.information_ratio),
                )),
                Err(e) => pm.push(format!(
                    "[{},{},{},{},\"ERR:{}\"]",
                    jstr(name),
                    jstr(freq),
                    jstr(market),
                    num(factor),
                    e
                )),
            }
        }
    }
    out += &format!("\"perf\":[{}],\n", pm.join(","));
    // band + verdict probes
    let mut bv: Vec<String> = Vec::new();
    for v in [0.1, 0.29, 0.3, 0.7, 1.5, 1e30] {
        bv.push(format!(
            "[{},{}]",
            num(v),
            jstr(&perf_metrics::classify_information_ratio(
                v,
                &perf_metrics::validate_bands(perf_metrics::DEFAULT_BANDS).unwrap()
            ))
        ));
    }
    out += &format!("\"bands\":[{}]\n", bv.join(","));
    out += "}\n";
    print!("{out}");
}
