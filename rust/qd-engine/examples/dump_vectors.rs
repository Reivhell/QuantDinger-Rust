//! Dump deterministic computation vectors as JSON for cross-checking
//! against the Python originals (see `../parity/check.py`).
//!
//! Run: `cargo run --example dump_vectors` (from `rust/`).

use qd_engine::close_reason::{self, ExecRowIn, NumVal, TradeRowIn};
use qd_engine::{
    curve_sampling::{self, CurvePoint},
    frequencies, grid, indicators, instruments, market_visibility, net_pnl, perf_metrics,
    pnl, precise, protection, risk_guard,
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
    out += &format!("\"bands\":[{}],\n", bv.join(","));

    // market_visibility vectors: env cases x market probes + filter + hidden.
    let mv_cases: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("default", vec![]),
        ("cn_yes_hk_no", vec![("SHOW_CN_STOCK", "yes"), ("SHOW_HK_STOCK", "0")]),
        ("cn_true_ws", vec![("SHOW_CN_STOCK", "  True ")]),
        (
            "whitelist",
            vec![("ENABLED_MARKETS", "Crypto, USStock"), ("SHOW_CN_STOCK", "true")],
        ),
        ("whitelist_messy", vec![("ENABLED_MARKETS", " Crypto ,,USStock, ")]),
        ("whitelist_empty", vec![("ENABLED_MARKETS", "")]),
        ("whitelist_hk_only", vec![("ENABLED_MARKETS", "HKStock")]),
    ];
    let mv_markets = [
        "Crypto", "USStock", "CNStock", "HKStock", "Forex", "Futures", "MOEX", "",
        "Unknown", " crypto ",
    ];
    let mut mv: Vec<String> = Vec::new();
    for (name, vars) in &mv_cases {
        let mut env = market_visibility::Env::default();
        for (k, v) in vars {
            env = env.with(k, v);
        }
        let vis: Vec<bool> =
            mv_markets.iter().map(|m| market_visibility::is_market_visible(&env, m)).collect();
        let mut hidden: Vec<String> =
            market_visibility::hidden_markets(&env).into_iter().collect();
        hidden.sort();
        let items = vec![
            market_visibility::MarketItem::Str("Crypto".into()),
            market_visibility::MarketItem::Str("CNStock".into()),
            market_visibility::MarketItem::Str("  ".into()),
            market_visibility::MarketItem::Map(
                [("value".to_string(), "USStock".to_string())].into_iter().collect(),
            ),
            market_visibility::MarketItem::Map(
                [("value".to_string(), "CNStock".to_string())].into_iter().collect(),
            ),
            market_visibility::MarketItem::Map(
                [("other".to_string(), "Crypto".to_string())].into_iter().collect(),
            ),
        ];
        let kept = market_visibility::filter_market_items(&env, &items, "value");
        let kept_dbg: Vec<String> = kept
            .iter()
            .map(|it| match it {
                market_visibility::MarketItem::Str(s) => format!("S:{s}"),
                market_visibility::MarketItem::Map(m) => {
                    format!("M:{}", m.get("value").map(String::as_str).unwrap_or(""))
                }
            })
            .collect();
        let vis_s = vis.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",");
        let hid_s = hidden.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(",");
        let kept_s = kept_dbg.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(",");
        mv.push(format!(
            "[{},[{}],[{}],[{}],[{}]]",
            jstr(name),
            vis_s,
            hid_s,
            kept_s,
            vars.iter().map(|(k, v)| format!("[{},{}]", jstr(k), jstr(v))).collect::<Vec<_>>().join(",")
        ));
    }
    out += &format!("\"mvis\":[{}],\n", mv.join(","));

    // data_portal vectors: one 5-bar daily frame, clocks, probes.
    use qd_engine::data_portal as dp;
    const PD: i64 = 86_400;
    const PB: i64 = 1_767_225_600; // Thu 2026-01-01 00:00 UTC
    let prow: Vec<(i64, f64, f64)> =
        (0..5).map(|i| (PB + i * PD, 100.0 + i as f64, 100.5 + i as f64)).collect();
    let praw = dp::RawFrame {
        columns: ["open", "high", "low", "close", "volume"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        rows: prow
            .iter()
            .map(|(t, o, c)| dp::RawBar {
                ts: Some(*t),
                open: *o,
                high: *o + 1.0,
                low: *o - 1.0,
                close: *c,
                volume: (true, Some(1000.0)),
                extras: vec![("suspended".to_string(), 0.0)],
            })
            .collect(),
    };
    let pkey = "Crypto:BTC/USDT@spot";
    let pframe = dp::normalize_frame(pkey, &praw).unwrap();
    let psym = qd_engine::instruments::parse_instrument(pkey, "").unwrap().symbol;
    // clocks: (now, include_current, driving, freq)
    let mut dvec: Vec<String> = Vec::new();
    for (now, incl) in [(PB + 2 * PD + 6 * 3_600, false), (PB + 2 * PD + 6 * 3_600, true), (PB - PD, false)] {
        let cutoff = dp::visible_cutoff(Some(now), incl, PD, PD);
        let end = dp::visible_end(&pframe.ts, cutoff);
        let cur = dp::current_value(&pframe.close, end, -1.0);
        // unknown field → default (mirrors `field not in columns`); the
        // NaN → default line is covered by the `current_nan_and_empty_default`
        // unit test.
        let cur_miss = dp::current_value(&[], end, -1.0);
        dvec.push(format!(
            "[{},{},{},{},{}]",
            jopt_num(cutoff.map(|c| c as f64)),
            end,
            jopt_num(Some(cur)),
            jopt_num(Some(cur_miss)),
            jopt_num(dp::positive_price(cur)),
        ));
    }
    // bar_at exact + missing (flat [ts,o,h,l,c,v])
    for ts in [PB + PD, PB + 10 * PD] {
        match dp::bar_at(&pframe, ts) {
            Some(b) => dvec.push(format!(
                "[{},{},{},{},{},{}]",
                ts, b.open, b.high, b.low, b.close, b.volume
            )),
            None => dvec.push(format!("[{},null]", ts)),
        }
    }
    // resolve_key probes
    let mut frames_map = std::collections::HashMap::new();
    frames_map.insert(pkey.to_string(), pframe);
    let mut aliases = std::collections::HashMap::new();
    aliases.insert(psym.clone(), pkey.to_string());
    for s in [pkey, "BTCUSDT", "Crypto:BTC/USDT", "NOPE"] {
        match dp::resolve_key(&frames_map, &aliases, s) {
            Ok(k) => dvec.push(format!("[{},{}]", jstr(s), jstr(&k))),
            Err(e) => dvec.push(format!("[{},\"ERR:{}\"]", jstr(s), e)),
        }
    }
    // error paths
    let empty = dp::RawFrame { columns: vec!["open".into()], rows: vec![] };
    dvec.push(format!("\"{}\"", dp::normalize_frame("USStock:K", &empty).unwrap_err()));
    let nohl = dp::RawFrame {
        columns: vec!["open".into(), "close".into()],
        rows: vec![dp::RawBar {
            ts: Some(PB), open: 1.0, high: 2.0, low: 0.5, close: 1.5,
            volume: (false, None), extras: vec![],
        }],
    };
    dvec.push(format!("\"{}\"", dp::normalize_frame("USStock:K", &nohl).unwrap_err()));
    let confl = dp::RawFrame {
        columns: ["open", "high", "low", "close"].iter().map(|s| s.to_string()).collect(),
        rows: vec![
            dp::RawBar { ts: Some(PB), open: 1.0, high: 2.0, low: 0.5, close: 1.5, volume: (false, None), extras: vec![] },
            dp::RawBar { ts: Some(PB), open: 1.0, high: 2.0, low: 0.5, close: 9.5, volume: (false, None), extras: vec![] },
        ],
    };
    dvec.push(format!("\"{}\"", dp::normalize_frame("USStock:K", &confl).unwrap_err()));
    dvec.push(format!("\"{}\"", jstr(&psym).replace('"', "")));
    out += &format!("\"dportal\":[{}],\n", dvec.join(","));

    // snapshot vectors: canonical bytes + float rendering + id checks.
    use qd_engine::snapshot as sn;
    let srows: Vec<sn::SnapshotRow> = vec![
        sn::SnapshotRow { ns: 1_767_254_400_000_000_000, values: vec![102.3, 103.3, 101.3, 102.7, 12.0] },
        sn::SnapshotRow { ns: 1_767_225_600_000_000_000, values: vec![100.1, 101.1, 99.1, 100.8, 10.0] },
        sn::SnapshotRow { ns: 1_767_240_000_000_000_000, values: vec![101.2, 102.2, 100.2, 101.9, 11.0] },
        // dupe keeps last
        sn::SnapshotRow { ns: 1_767_225_600_000_000_000, values: vec![100.1, 101.1, 99.1, 100.8, 10.0] },
    ];
    let sn_bytes = sn::canonical_frame_bytes(&["open", "high", "low", "close", "volume"], &srows).unwrap();
    let mut svec: Vec<String> = vec![format!(
        "\"{}\"",
        String::from_utf8(sn_bytes.clone()).unwrap().replace('"', "'")
    )];
    // sha256 of the bytes (compared with hashlib in check.py)
    svec.push(format!("\"{}\"", sn::sha256_hex(&sn_bytes)));
    for v in [100.0, 0.1, 1e16, 1e-7, -2.5e100, 123456789.123456789] {
        svec.push(format!("\"{}\"", sn::py_float_repr(v).unwrap()));
    }
    for id in ["xyz", &"A".repeat(64), &("ab".repeat(32))] {
        match sn::validate_snapshot_id(id) {
            Ok(n) => svec.push(format!("\"ok:{n}\"")),
            Err(e) => svec.push(format!("\"ERR:{e}\"")),
        }
    }
    out += &format!("\"snap\":[{}],\n", svec.join(","));

    // readiness vectors: universe gate + warmup counts + fundamentals.
    use qd_engine::readiness as rd;
    const RB: i64 = 1_767_225_600; // 2026-01-01 00:00 UTC
    const RD: i64 = 86_400;
    let mut rvec: Vec<String> = Vec::new();
    // universe: [ok|ERR] for (snapshot_only, start_offset_days)
    for (only, off) in [(true, -1), (true, 0), (false, -1)] {
        let start = rd::Zoned::utc(RB + off * RD);
        match rd::validate_universe_history(
            Some("U1"), Some(rd::Zoned::utc(RB)), None, only, Some("2026-01-01"), start,
        ) {
            Ok(()) => rvec.push("\"ok\"".to_string()),
            Err(e) => rvec.push(format!("\"ERR:{e}\"")),
        }
    }
    // warmup: 5-bar frame with bar1 zero/None, warmup 4 vs 5, member valid_from
    let wmk = |bad: bool| rd::WarmupFrame {
        ts: (0..5).map(|i| RB + i * RD).collect(),
        open: (0..5).map(|i| if bad && i == 1 { Some(0.0) } else { Some(100.0) }).collect(),
        high: (0..5).map(|_| Some(101.0)).collect(),
        low: (0..5).map(|_| Some(99.0)).collect(),
        close: (0..5).map(|i| if bad && i == 1 { None } else { Some(100.5) }).collect(),
    };
    let wframes = |bad: bool| {
        vec![("1d".to_string(), vec![("S".to_string(), wmk(bad))])]
    };
    for (bad, w, start) in [(true, 5, RB + 5 * RD), (true, 4, RB + 5 * RD), (false, 5, RB + 5 * RD)] {
        match rd::validate_warmup(&wframes(bad), w, start, &[]) {
            Ok(()) => rvec.push("\"ok\"".to_string()),
            Err(e) => rvec.push(format!("\"ERR:{e}\"")),
        }
    }
    let members = vec![rd::Member { key: "S".to_string(), valid_from: Some(RB + 3 * RD) }];
    match rd::validate_warmup(&wframes(false), 5, RB, &members) {
        Ok(()) => rvec.push("\"ok\"".to_string()),
        Err(e) => rvec.push(format!("\"ERR:{e}\"")),
    }
    // fundamentals: pe ok / pb all-None / roe absent; as_of full vs cut
    use std::collections::{HashMap, HashSet};
    let mut cols = HashMap::new();
    cols.insert("pe".to_string(), vec![Some(10.0), Some(f64::INFINITY), None]);
    cols.insert("pb".to_string(), vec![None, None, None]);
    let fframes = vec![(
        "S".to_string(),
        rd::FundFrame { ts: vec![RB, RB + RD, RB + 2 * RD], cols },
    )];
    let req: HashSet<String> = ["pe".into(), "pb".into(), "roe".into()].into_iter().collect();
    for as_of in [None, Some(RB - RD)] {
        match rd::validate_fundamentals(&fframes, &req, as_of) {
            Ok(()) => rvec.push("\"ok\"".to_string()),
            Err(e) => rvec.push(format!("\"ERR:{e}\"")),
        }
    }
    // cal_date probe: 2025-12-31 23:00 UTC +2h offset → 2026-01-01 local
    let (y, m, d) = rd::cal_date(rd::Zoned { epoch: RB - 3_600, offset: 7_200 });
    rvec.push(format!("\"{y:04}-{m:02}-{d:02}\""));
    out += &format!("\"ready\":[{}],\n", rvec.join(","));

    // language vectors: normalize probes + detect-priority probes.
    use qd_engine::language as lang;
    let norm_cases = [
        "en", "en-US,en;q=0.9", " zh-hans ", "zh-Hant", "JA-jp", "fr-FR; q=0.8",
        "", "   ", "xx-YY", "en;", "KO-kr;q=0.5, en;q=0.3", "zh",
    ];
    let mut lvec: Vec<String> = Vec::new();
    for raw in norm_cases {
        match lang::normalize_lang(raw) {
            Some(l) => lvec.push(format!("[{},{}]", jstr(raw), jstr(&l))),
            None => lvec.push(format!("[{},null]", jstr(raw))),
        }
    }
    let det_cases: Vec<(Option<&str>, Option<&str>, Option<&str>, Option<&str>, &str)> = vec![
        (Some("ja-JP"), Some("fr-FR"), Some("de-DE"), Some("ko-KR"), "en-US"),
        (Some("xx"), Some("fr-FR"), None, None, "en-US"),
        (None, None, Some("th-TH"), None, "en-US"),
        (None, None, None, None, "zh-CN"),
        (None, Some("xx"), None, Some("ar-SA"), "en-US"),
    ];
    for (h, b, q, a, dflt) in &det_cases {
        let req = lang::RequestParts {
            header_app_lang: *h,
            body_language: *b,
            query_language: *q,
            header_accept_language: *a,
        };
        lvec.push(format!(
            "\"{}\"",
            lang::detect_request_language(&req, dflt)
        ));
    }
    out += &format!("\"lang\":[{}]\n", lvec.join(","));
    out += "}\n";
    print!("{out}");
}
