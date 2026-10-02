"""Cross-check Rust qd-engine vectors against the Python originals.

READ-ONLY w.r.t. the Python backend: imports the existing helpers and
compares them against `cargo run --example dump_vectors` output.

Run from repo root:
    python3 rust/parity/check.py
Requires: numpy (for the codegen-path check only, optional).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend_api_python"
RUST = ROOT / "rust"

# NOTE: we do NOT `import app.*` normally — `app/__init__.py` pulls in Flask
# and `records.py` pulls in the DB layer. Instead load the needed modules
# straight from their files with lightweight `app.*` stubs, so this script
# stays read-only AND dependency-light (stdlib + numpy only).
import ast
import importlib.util
import types


def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package
    sys.modules[name] = mod
    return mod


for _pkg in ("app", "app.utils", "app.services", "app.services.grid", "app.services.live_trading", "app.services.strategy_v2"):
    _stub(_pkg)


def _load(name: str, relpath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, BACKEND / relpath)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_function_only(name: str, relpath: str, func: str) -> types.ModuleType:
    """Exec only one function def (plus its imports-free body) from a heavy module."""
    src = (BACKEND / relpath).read_text()
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    stub = types.ModuleType(name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relpath, "exec"), stub.__dict__)  # noqa: S102
    sys.modules[name] = stub
    parent, _, _ = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], name.split(".")[-1], stub)
    return stub


_load("app.utils.trade_close_reason", "app/utils/trade_close_reason.py")
_load_function_only(
    "app.services.live_trading.fill_evidence",
    "app/services/live_trading/fill_evidence.py",
    "positive_number",
)
import math as _math  # noqa: E402
sys.modules["app.services.live_trading.fill_evidence"].__dict__.setdefault(
    "isfinite", _math.isfinite)
_load("app.utils.trade_execution", "app/utils/trade_execution.py")
_load_function_only(
    "app.services.live_trading.records",
    "app/services/live_trading/records.py",
    "normalize_strategy_symbol",
)
_load("app.utils.numeric_precision", "app/utils/numeric_precision.py")
_load("app.utils.market_visibility", "app/utils/market_visibility.py")
_load("app.utils.pnl", "app/utils/pnl.py")
_load("app.utils.risk_guard", "app/utils/risk_guard.py")
_load("app.utils.technical_indicators", "app/utils/technical_indicators.py")
_load("app.utils.trade_net_pnl", "app/utils/trade_net_pnl.py")
_load("app.services.grid.levels", "app/services/grid/levels.py")
_load("app.services.strategy_v2.frequencies", "app/services/strategy_v2/frequencies.py")
_load("app.services.strategy_v2.models", "app/services/strategy_v2/models.py")
_load("app.services.market_schedule", "app/services/market_schedule.py")
_load("app.services.backtest_metrics", "app/services/backtest/metrics.py")
_load("app.services.strategy_v2.instruments", "app/services/strategy_v2/instruments.py")
_load("app.services.strategy_v2.curve_sampling", "app/services/strategy_v2/curve_sampling.py")
_prot = _load("app.services.strategy_v2.protection", "app/services/strategy_v2/protection.py")

from app.services.strategy_v2.protection import (  # noqa: E402
    ProtectionEngine as PyProtectionEngine,
    ProtectionSpec as PyProtectionSpec,
    ProtectionState as PyProtectionState,
)
from app.services.strategy_v2.curve_sampling import sample_equity_curve  # noqa: E402
from app.services.backtest_metrics import (  # noqa: E402
    calculate_information_ratio as py_ir,
)
from app.services.strategy_v2.instruments import (  # noqa: E402
    infer_market as py_infer_market,
    is_index_reference as py_is_index,
    normalize_index_reference as py_norm_index,
    normalize_pool_reference as py_norm_pool,
    parse_instrument as py_parse_ins,
)
from app.services.strategy_v2.frequencies import (  # noqa: E402
    driving_frequency as py_driving,
    frequency_seconds as py_freq_secs,
    normalize_frequency as py_norm_freq,
    periods_per_year as py_periods,
    unique_frequencies as py_unique,
)
from app.utils.trade_close_reason import (  # noqa: E402
    enrich_trade_row as py_enrich_trade_row,
    infer_legacy_close_reason as py_infer,
    label_for_reason as py_label,
)
from app.utils.trade_execution import enrich_execution_reference as py_exec_ref  # noqa: E402

T0_STR = "2026-01-01 00:00:00"


def _py_prot_str(decision, state) -> str:
    if decision is None:
        return f"none:{state.highest_price}:{state.lowest_price}"
    return f"{decision.reason}:{decision.price}:{decision.trigger_price}"


from app.services.grid.levels import generate_cells, generate_levels  # noqa: E402
from app.services.live_trading.records import normalize_strategy_symbol  # noqa: E402
from app.utils.numeric_precision import (  # noqa: E402
    clean_generated_number,
    floor_decimal_to_step,
    format_decimal,
)
from app.utils.pnl import (  # noqa: E402
    calc_margin_notional,
    calc_notional_value,
    calc_pnl_percent,
    calc_unrealized_pnl,
)
from app.utils.risk_guard import coerce_fee_rate, trailing_exit_locks_net_profit  # noqa: E402
from app.utils.technical_indicators import (  # noqa: E402
    compute_kdj_cn,
    compute_rsi_wilder,
    kdj_codegen,
    rsi_wilder_codegen,
)
from app.utils.trade_net_pnl import (  # noqa: E402
    enrich_trades_net_pnl,
    net_pnl_for_equity_step,
    net_realized_pnl,
)

FAILURES: list[str] = []


def check(name: str, rust_val, py_val, *, tol: float = 0.0) -> None:
    if isinstance(rust_val, float) and isinstance(py_val, float) and tol:
        ok = abs(rust_val - py_val) <= tol
    else:
        ok = rust_val == py_val
    if not ok:
        FAILURES.append(f"{name}: rust={rust_val!r} python={py_val!r}")
        print(f"FAIL {name}: rust={rust_val!r} python={py_val!r}")


def check_list(name: str, rust_list, py_list, *, tol: float = 0.0) -> None:
    if len(rust_list) != len(py_list):
        FAILURES.append(f"{name}: length rust={len(rust_list)} python={len(py_list)}")
        print(f"FAIL {name}: length rust={len(rust_list)} python={len(py_list)}")
        return
    for i, (r, p) in enumerate(zip(rust_list, py_list)):
        check(f"{name}[{i}]", r, p, tol=tol)


def main() -> int:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "dump_vectors"],
        cwd=RUST,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        return 2
    vec = json.loads(proc.stdout)

    # --- Indicators: Rust vs direct helpers vs pandas codegen fragments ---
    # The strategy compiler emits kdj_codegen()/rsi_wilder_codegen() pandas
    # fragments (rolling-min/max + fillna path), NOT compute_kdj_cn() calls.
    # Feed all three the exact series from dump_vectors and compare.
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402

    high = vec["series_high"]
    low = vec["series_low"]
    close = vec["series_close"]
    k, d, j = compute_kdj_cn(high, low, close, 9, 3, 3)
    check_list("kdj_k", vec["kdj_k"], k)
    check_list("kdj_d", vec["kdj_d"], d)
    check_list("kdj_j", vec["kdj_j"], j)
    k14, d14, j14 = compute_kdj_cn(high, low, close, 14, 2, 4)
    check_list("kdj_k_14_2_4", vec["kdj_k_14_2_4"], k14)
    check_list("kdj_d_14_2_4", vec["kdj_d_14_2_4"], d14)
    check_list("kdj_j_14_2_4", vec["kdj_j_14_2_4"], j14)
    rsi_all = compute_rsi_wilder(close, 14)
    check_list("rsi", vec["rsi"], rsi_all)
    rsi7 = compute_rsi_wilder(close, 7)
    check_list("rsi_7", vec["rsi_7"], rsi7)

    # Codegen path (what compiled strategies actually execute).
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    ns: dict = {}
    exec(kdj_codegen(9, 3, 3, "t"), {"df": df, "pd": pd, "np": np}, ns)
    tol = 5e-4  # codegen rounds K/D inputs differently (raw RSV vs round4 chain)
    ck = [None if pd.isna(v) else round(float(v), 4) for v in df["t_k"]]
    cd = [None if pd.isna(v) else round(float(v), 4) for v in df["t_d"]]
    cj = [None if pd.isna(v) else round(float(v), 4) for v in df["t_j"]]
    # Known divergence: codegen fillna(50.0) seeds warmup bars [0..7] with
    # 50.0 while the direct helper (and Rust) emit None there. Assert the
    # warmup behavior explicitly, compare values from the first valid bar.
    assert ck[:8] == [50.0] * 8 and k[:8] == [None] * 8
    assert cd[:8] == [50.0] * 8 and d[:8] == [None] * 8
    assert cj[:8] == [50.0] * 8 and j[:8] == [None] * 8
    check_list("codegen_kdj_k", vec["kdj_k"][8:], ck[8:], tol=tol)
    check_list("codegen_kdj_d", vec["kdj_d"][8:], cd[8:], tol=tol)
    check_list("codegen_kdj_j", vec["kdj_j"][8:], cj[8:], tol=tol)
    exec(rsi_wilder_codegen(14, "r"), {"df": df, "pd": pd, "np": np}, ns)
    cr = [None if pd.isna(v) else round(float(v), 4) for v in df["r"]]
    check_list("codegen_rsi", vec["rsi"], cr, tol=tol)

    # Committed unit-test behavior still holds.
    rsi = compute_rsi_wilder(
        [100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106], 14
    )
    assert rsi[13] is None and rsi[14] is not None

    # --- PnL ---
    for i, (s, e, c, z) in enumerate(
        [
            ("long", 100.0, 110.0, 2.0),
            ("short", 110.0, 100.0, 2.0),
            ("long", 110.0, 100.0, 2.0),
            ("SHORT", 50.5, 49.25, 10.0),
            ("long", 0.0, 100.0, 1.0),
            ("long", 100.0, 100.0, -1.0),
        ]
    ):
        check(f"unrealized[{i}]", vec["unrealized"][i], calc_unrealized_pnl(s, e, c, z))
    for i, (e, s, p, lv, m) in enumerate(
        [
            (100.0, 2.0, 20.0, 1.0, "spot"),
            (100.0, 2.0, 20.0, 10.0, "swap"),
            (100.0, 2.0, -7.5, 5.0, "perp"),
            (100.0, 2.0, 20.0, 10.0, "spot"),
            (0.0, 1.0, 10.0, 1.0, "spot"),
            (100.0, 2.0, 20.0, 0.0, "futures"),
        ]
    ):
        check(f"pnl_pct[{i}]", vec["pnl_pct"][i], calc_pnl_percent(e, s, p, leverage=lv, market_type=m))
    assert calc_notional_value(100.0, 2.0) == 200.0
    for i, (n, lv, m) in enumerate([(200.0, 10.0, "swap"), (200.0, 10.0, "spot"), (200.0, 0.0, "perp")]):
        check(f"margin[{i}]", vec["margin"][i], calc_margin_notional(n, lv, m))

    # --- precise ---
    for i, (v, s) in enumerate(
        [
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
        ]
    ):
        check(f"floors[{i}]", vec["floors"][i], str(floor_decimal_to_step(v, s)))
    for i, v in enumerate([1.2345678901234567, 2.5, 0.0, -2.5, 100.0]):
        check(f"clean[{i}]", vec["clean"][i], clean_generated_number(v, decimal_places=12))
    for i, v in enumerate([1.23000, 0.0, 100.0, 1e-7, -0.0]):
        check(f"format[{i}]", vec["format"][i], format_decimal(v, decimal_places=12))

    # --- risk_guard ---
    for i, v in enumerate([0.001, -0.5, 0.1, 0.02, 5.0]):
        check(f"fees[{i}]", vec["fees"][i], coerce_fee_rate(v))
    check("fee_none", vec["fee_none"], coerce_fee_rate(None))
    for i, (s, e, x, f, b) in enumerate(
        [
            ("long", 100.0, 100.3, 0.001, 0.0),
            ("long", 100.0, 100.1, 0.001, 0.0),
            ("short", 100.0, 99.7, 0.001, 0.0),
            ("short", 100.0, 99.9, 0.001, 0.0),
            ("long", 100.0, 100.1, 0.001, 0.005),
            ("sideways", 100.0, 200.0, 0.001, 0.0),
        ]
    ):
        check(
            f"trail[{i}]",
            vec["trail"][i],
            trailing_exit_locks_net_profit(s, entry_price=e, exit_price=x, fee_rate=f, extra_buffer=b),
        )

    # --- net_pnl ---
    for i, s in enumerate(["btcusdt", "BTC/USDT", "BTC/USDT:USDT", "eth-usd", "SOLUSDT@BINANCE"]):
        check(f"symbols[{i}]", vec["symbols"][i], normalize_strategy_symbol(s))

    trades = [
        {"id": 1, "symbol": "ETH/USDT", "type": "open_long", "amount": 1.0, "commission": 10.0, "profit": None, "created_at": 1},
        {"id": 2, "symbol": "ETH/USDT", "type": "add_long", "amount": 1.0, "commission": 8.0, "profit": None, "created_at": 2},
        {"id": 3, "symbol": "ETH/USDT", "type": "reduce_long", "amount": 1.5, "commission": 2.0, "profit": 30.0, "created_at": 3},
        {"id": 4, "symbol": "BTC/USDT", "type": "open_short", "amount": 0.5, "commission": 1.0, "commission_quote": 2.0, "profit": None, "created_at": 4},
        {"id": 5, "symbol": "BTC/USDT", "type": "close_short", "amount": 0.5, "commission": 0.5, "profit": 12.0, "created_at": 5},
    ]
    enrich_trades_net_pnl(trades)
    for i, row in enumerate(trades):
        r = vec["enriched"][i]
        assert r["id"] == row["id"], (r, row)
        for k in ("profit_gross", "open", "close_comm", "total", "net", "profit"):
            pass
        check(f"enriched[{i}].profit_gross", r["profit_gross"], row.get("profit_gross"))
        check(f"enriched[{i}].open", r["open"], row.get("open_commission_allocated"))
        check(f"enriched[{i}].close_comm", r["close_comm"], row.get("close_commission"))
        check(f"enriched[{i}].total", r["total"], row.get("total_commission"))
        check(f"enriched[{i}].net", r["net"], row.get("net_pnl"))
        check(f"enriched[{i}].profit", r["profit"], row.get("profit"))
    for i, row in enumerate(trades):
        check(f"equity[{i}]", vec["equity"][i], net_pnl_for_equity_step(row))

    # net_realized_pnl None-guard parity
    assert net_realized_pnl({"profit": None}, open_commission=5.0) is None

    # --- grid ---
    check_list("levels_arith", vec["levels_arith"], generate_levels(100.0, 200.0, 5, "arithmetic"))
    check_list("levels_geo", vec["levels_geo"], generate_levels(100.0, 400.0, 3, "geometric"), tol=1e-9)
    check_list("levels_geo5", vec["levels_geo5"], generate_levels(50.0, 500.0, 5, "geometric"), tol=1e-9)
    cells = generate_cells(generate_levels(100.0, 200.0, 5, "arithmetic"))
    assert len(cells) == len(vec["cells"]) == 4
    for i, c in enumerate(cells):
        check(f"cells[{i}]", vec["cells"][i], [c.lower_price, c.upper_price])

    # --- protection: differential replay against the real Python engine ---
    # Drive Python with the exact probe matrix dump_vectors used:
    # 3 modes x 4 configs x (8 single-bar probes + 5 chained tick probes).
    import pandas as pd  # noqa: E402  (already imported above; idempotent)

    _specs = [
        dict(stop_loss_pct=0.02, take_profit_pct=0.05),
        dict(stop_loss_pct=0.02, take_profit_pct=0.05),
        dict(trailing_stop_pct=0.003, trailing_activation_pct=0.01,
             trailing_rebase_on_scale_in=False),
        dict(time_limit_seconds=3600),
    ]
    _bars = [(95.0, 96.0, 94.0), (100.0, 101.0, 97.0),
             (100.0, 106.0, 99.0), (100.0, 101.0, 99.0)]
    _ticks = [100.5, 101.5, 102.0, 101.6, 97.5]
    _ts_bar = pd.Timestamp("2026-01-01 04:00:00")
    expected_prot: list[str] = []
    for mode in ("conservative", "aggressive", "balanced"):
        eng = PyProtectionEngine(intrabar_mode=mode)
        for si, kw in enumerate(_specs):
            spec = PyProtectionSpec(**kw)
            for bi, (o, h, l) in enumerate(_bars):
                for side in ("long", "short"):
                    st = PyProtectionState.open(
                        symbol="S", side=side, entry_price=100.0,
                        spec=spec, opened_at=T0_STR)
                    d = eng.evaluate_bar(
                        st, timestamp=_ts_bar,
                        open_price=o, high_price=h, low_price=l)
                    expected_prot.append(f"{mode}:s{si}:b{bi}:{side}:{_py_prot_str(d, st)}")
            st = PyProtectionState.open(
                symbol="S", side="long", entry_price=100.0,
                spec=spec, opened_at=T0_STR)
            for ti, px in enumerate(_ticks):
                d = eng.evaluate_price(
                    st, timestamp=pd.Timestamp(T0_STR) + pd.Timedelta(seconds=60 * (ti + 1)),
                    price=px)
                expected_prot.append(f"{mode}:s{si}:t{ti}:long:{_py_prot_str(d, st)}")
    assert len(expected_prot) == len(vec["protection"]) == 156, (
        len(expected_prot), len(vec["protection"]))
    # Floats render via different str() paths (Rust {:?} vs Python repr);
    # compare numerically part-by-part.
    for i, (r, p) in enumerate(zip(vec["protection"], expected_prot)):
        rs, ps = r.split(":"), p.split(":")
        check(f"protection[{i}].tag", ":".join(rs[:5]), ":".join(ps[:5]))
        for j, (rv, pv) in enumerate(zip(rs[5:], ps[5:])):
            check(f"protection[{i}].num[{j}]", float(rv), float(pv), tol=1e-9)

    # --- close_reason: drive real Python through identical probes ---
    _codes = [
        "long_entry", "long_exit", "short_entry", "short_exit",
        "grid_initial_long", "grid_initial_short", "grid_reduce_long",
        "grid_reduce_short", "grid_close_all", "grid_waterfall_close",
        "grid_equity_stop_loss", "grid_equity_take_profit",
        "grid_equity_trailing_stop", "dca_equity_stop_loss",
        "dca_equity_take_profit", "dca_equity_trailing_stop",
        "robot_equity_stop_loss", "robot_equity_take_profit",
        "robot_equity_trailing_stop", "grid_out_of_bounds_up",
        "grid_out_of_bounds_down", "server_stop_loss", "server_take_profit",
        "server_trailing_stop", "indicator_signal", "signal_trigger",
    ]
    assert len(vec["labels"]) == len(_codes) == 26
    for i, c in enumerate(_codes):
        check(f"labels[{i}]", vec["labels"][i],
              [c, py_label(c, lang="zh"), py_label(c, lang="en")])
    check("label_empty", py_label("", lang="zh"), "")
    check("label_unknown", py_label("custom_x", lang="zh"), "custom_x")

    _enr_cases = [
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
    ]
    assert len(vec["enrich_row"]) == len(_enr_cases)
    for i, (tt, cr, bt, lang) in enumerate(_enr_cases):
        py = py_enrich_trade_row(
            {"type": tt, "close_reason": cr,
             "matched_entry_price": "12.5", "grid_matched_profit": "bad"},
            bot_type=bt, lang=lang)
        check(f"enrich_row[{i}]", vec["enrich_row"][i],
              [tt, cr, bt, py["close_reason"], py["action_note"],
               float(py["matched_entry_price"]),
               float(py["grid_matched_profit"])])

    _inf_cases = [("close_long", "grid"), ("close_short", "grid"),
                  ("close_long", "dca"), ("open_long", "grid"), ("close_long", "")]
    assert len(vec["infer"]) == len(_inf_cases)
    for i, (tt, bt) in enumerate(_inf_cases):
        check(f"infer[{i}]", vec["infer"][i],
              [tt, bt, py_infer(tt, bot_type=bt, stored_reason="")])

    _exec_cases = [
        ("100", None, "", "101", "open_long"),
        ("", '{"ref_price": 200.0}', "", "198", "open_short"),
        ("", '{"client_order_id": "abc", "limit_price": 50.0, "price": 51.0}',
         "", "50.5", "close_long"),
        ("", '{"price": "75.25"}', "", "75.25", "add_long"),
        ("", "not-json{{{", "", "10", "open_long"),
        ("", None, "", "10", "open_long"),
        ("", None, "99.5", "100", "close_short"),
        ("0", '{"ref_price": 0}', "", "10", "open_long"),
    ]
    assert len(vec["exec_ref"]) == len(_exec_cases)
    for i, (grid_px, payload, pending, px, tt) in enumerate(_exec_cases):
        py = py_exec_ref({
            "request_payload": payload,
            "request_price": pending or None,
            "grid_request_price": grid_px or None,
            "grid_client_reference": "",
            "price": px,
            "type": tt,
        })
        check(f"exec_ref[{i}]", vec["exec_ref"][i],
              [tt, py["grid_client_reference"], py["reference_price"],
               py["reference_kind"], py["price_deviation_pct"]])

    # --- frequencies: identical labels through real Python ---
    _freq_labels = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w",
                    "daily", "DAY", "m1", "h1", "d1", "monthly", "15分钟",
                    "2小时", "", "tick", "1mo", "2w", "1.5h", "0m", "1_0m",
                    " 4H "]
    assert len(vec["freqs"]) == len(_freq_labels)
    for i, f in enumerate(_freq_labels):
        check(f"freqs[{i}]", vec["freqs"][i],
              [f, py_norm_freq(f), py_freq_secs(f)])
    _period_cases = [("1h", ["Crypto"]), ("4h", ["Crypto"]), ("15m", ["Crypto"]),
                     ("1h", ["Stocks"]), ("1d", ["Stocks"]), ("1w", []),
                     ("1mo", ["Crypto"]), ("m", ["Crypto"]), ("h", ["Stocks"]),
                     ("1.5h", ["Stocks"]), ("1d", []), ("1d", ["Crypto", "Stocks"]),
                     ("2w", ["Crypto"]), ("90m", ["Stocks"]), ("24h", ["Crypto"])]
    assert len(vec["periods"]) == len(_period_cases)
    for i, (f, mk) in enumerate(_period_cases):
        try:
            py_v: object = py_periods(f, mk)
        except ValueError:
            py_v = "ERR"
        r = vec["periods"][i]
        check(f"periods[{i}].freq", r[0], f)
        check(f"periods[{i}].markets", r[1], mk)
        if isinstance(py_v, str):
            check(f"periods[{i}].val", str(r[2]).startswith("ERR"), True)
        else:
            check(f"periods[{i}].val", r[2], py_v, tol=1e-9)
    _uniq_cases = [["1d", "4h", "1d", "1h"], [], ["tick", "tock", "tick"]]
    assert len(vec["uniqfreq"]) == len(_uniq_cases)
    for i, case in enumerate(_uniq_cases):
        check(f"uniqfreq[{i}]", [list(vec["uniqfreq"][i][0]), vec["uniqfreq"][i][1]],
              [list(py_unique(case)), py_driving(case)])

    # --- curve_sampling: identical scenarios through real Python ---
    _scenarios: list[tuple[list, list, float, int]] = [
        ([100.0 + i for i in range(25)], [None] * 25, 100.0, 10),
        ([100.0, 102.0, 105.0, 110.0, 108.0, 120.0, 115.0, 90.0, 80.0,
          85.0, 88.0, 92.0, 95.0, 97.0, 99.0, 101.0, 103.0, 104.0,
          106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0], [None] * 25,
         100.0, 10),
        ([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0, 20.0, 5.0, 1.0,
          0.0, -3.0, -3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0,
          5.0, 6.0, 7.0, 8.0, 9.0], [None] * 25, 100.0, 12),
        ([100.0] * 7 + [float("nan"), float("inf")] + [100.0] * 16,
         [None] * 25, 100.0, 10),
        ([100.0] * 25,
         [None] * 17 + [-42.0] + [None] * 7, 100.0, 10),
    ]
    assert len(vec["curve"]) == len(_scenarios)
    for si, (vals, dds, initial, limit) in enumerate(_scenarios):
        items = [{"value": v, **({"drawdown": d} if d is not None else {})}
                 for v, d in zip(vals, dds)]
        py_kept = sample_equity_curve(items, limit, initial)
        # map kept item identities back to indices (items are unique dicts)
        ids = [id(it) for it in items]
        py_ids = {id(it) for it in py_kept}
        py_idx = [n for n, ident in enumerate(ids) if ident in py_ids]
        check(f"curve[{si}]", list(vec["curve"][si][1:]), py_idx)

    # --- instruments: identical probes through real Python ---
    _ins_cases = [
        ("600519.XSHG", ""), ("USStock:MSFT", ""),
        ("Crypto:BTCUSDT@okx:swap", ""), ("Crypto:BTC/USDT@swap", ""),
        ("Crypto:00700/HKD@gate:spot", ""), ("BTCUSDT", ""), ("000001", ""),
        ("00700.HK", ""), ("MSFT", "USStock"), ("MSFT", ""),
        ("Crypto:BTC/USDT", ""), ("Crypto:BTC/USDT@binance", ""),
        ("USStock:MSFT@nasdaq:spot", ""), ("Crypto:ETHUSDT", ""),
        ("600519.xshg", ""), ("crypto:btcusdt@OKX:SWAP", ""),
        ("POOL:MyPool", ""), ("000300.XBHS", ""), ("INDEX:CSI300", ""),
        ("", ""), ("!!!", ""), ("Crypto:", ""), ("USStock:", "USStock"),
        ("  Crypto:BTC/USDT@swap  ", ""),
    ]
    assert len(vec["instruments"]) == len(_ins_cases)
    for i, (v, dm) in enumerate(_ins_cases):
        try:
            s = py_parse_ins(v, default_market=dm)
            expected: list = [v, dm, s.market, s.symbol, s.exchange_id,
                              s.market_type, s.key]
        except Exception as e:  # noqa: BLE001 - error message is the probe
            expected = [v, dm, f"ERR:{e}"]
        check(f"instruments[{i}]", vec["instruments"][i], expected)
    _im_cases = ["BTC/USDT", "USDT", "600519.SH", "SPY", "", "!!!",
                 "00700.HK", "000001"]
    assert len(vec["infer_market"]) == len(_im_cases)
    for i, s in enumerate(_im_cases):
        check(f"infer_market[{i}]", vec["infer_market"][i], [s, py_infer_market(s)])
    _ix_cases = ["INDEX:CSI300", "000300.XBHS", "Crypto:BTC/USDT",
                 "000300.XSHG_INDEX", "POOL:ABC", "abc"]
    assert len(vec["indexpool"]) == len(_ix_cases)
    for i, s in enumerate(_ix_cases):
        try:
            pool: object = py_norm_pool(s)
        except Exception as e:  # noqa: BLE001
            pool = f"ERR:{e}"
        check(f"indexpool[{i}]", vec["indexpool"][i],
              [s, py_is_index(s), py_norm_index(s), pool])

    # --- perf_metrics: identical curve scenarios through real Python ---
    # exchange_calendars is NOT installed -> Python takes the ImportError
    # fallback (False), same as Rust's `calendar_ok = false`.
    import pandas as pd  # noqa: E402
    _D = 86_400
    _B = 1_767_225_600  # 2026-01-05 00:00:00 UTC (Monday)
    _mk_pf = [( _B + i * _D, 100.0 + i * 1.1 - (i % 3) * 0.7) for i in range(10)]
    _mk_bm = [(_B + i * _D, 100.0 + i * 0.6) for i in range(10)]
    _gap_pf = [(_B, 100.0), (_B + _D, 101.0), (_B + 3 * _D, 102.0), (_B + 4 * _D, 103.0)]
    _gap_bm = [(_B, 200.0), (_B + _D, 200.5), (_B + 3 * _D, 201.0), (_B + 4 * _D, 202.0)]
    _mm_pf = [(_B, 100.0), (_B + _D, 101.0), (_B + 2 * _D, 102.0)]
    _mm_bm = [(_B + _D, 200.0), (_B + 2 * _D, 202.0), (_B + 3 * _D, 203.0)]
    _dirty_pf = [(_B, 100.0), (_B, 110.0), (_B + _D, -5.0), (_B + _D, 0.0),
                 (_B + 2 * _D, 112.0)]
    _dirty_bm = [(_B, 50.0), (_B + _D, 51.0), (_B + 2 * _D, 52.0)]
    _curves = [("clean10", _mk_pf, _mk_bm), ("gap", _gap_pf, _gap_bm),
               ("mismatch", _mm_pf, _mm_bm), ("dirty", _dirty_pf, _dirty_bm)]
    _perf_cases = [(n, pf, bm, f, m, fac)
                   for (n, pf, bm) in _curves
                   for (f, m, fac) in [("1d", "Crypto", 365.25),
                                       ("1d", "USStock", 252.0),
                                       ("1w", "Crypto", 52.0)]]
    assert len(vec["perf"]) == len(_perf_cases) == 12
    for i, (name, pf, bm, freq, market, factor) in enumerate(_perf_cases):
        pf_curve = [{"time": pd.Timestamp(t, unit="s", tz="UTC").isoformat(),
                     "value": v} for (t, v) in pf]
        bm_curve = [{"time": pd.Timestamp(t, unit="s", tz="UTC").isoformat(),
                     "value": v} for (t, v) in bm]
        py = py_ir(pf_curve, bm_curve, benchmark="BENCH", frequency=freq,
                   annualization_factor=factor, market=market)
        r = vec["perf"][i]
        check(f"perf[{i}].meta", r[:4], [name, freq, market, factor])
        check(f"perf[{i}].status", r[4], py["status"])
        check(f"perf[{i}].obs", r[5], py["observations"])
        for j, k in enumerate(("portfolioReturnAnnualized",
                               "benchmarkReturnAnnualized",
                               "activeReturnAnnualized",
                               "trackingErrorAnnualized",
                               "informationRatio")):
            check(f"perf[{i}].{k}", r[6 + j], py[k], tol=1e-9)
    _band_vals = [0.1, 0.29, 0.3, 0.7, 1.5, 1e30]
    assert len(vec["bands"]) == len(_band_vals)
    from app.services.backtest_metrics import (  # noqa: E402
        DEFAULT_INFORMATION_RATIO_BANDS as py_default_bands,
        _classify_information_ratio as py_classify,
        _validate_classification_bands as py_bands,
    )
    _py_bands = py_bands(list(py_default_bands))
    for i, v in enumerate(_band_vals):
        check(f"bands[{i}]", vec["bands"][i], [v, py_classify(v, _py_bands)])

    # --- market_visibility: env matrix through real Python via monkeypatched getenv ---
    import os as _os  # noqa: E402
    from app.utils.market_visibility import (  # noqa: E402
        filter_market_items as py_filter,
        hidden_markets as py_hidden,
        is_market_visible as py_visible,
    )
    _mv_markets = ["Crypto", "USStock", "CNStock", "HKStock", "Forex", "Futures",
                   "MOEX", "", "Unknown", " crypto "]
    _mv_items = ["Crypto", "CNStock", "  ",
                 {"value": "USStock"}, {"value": "CNStock"}, {"other": "Crypto"}]
    assert len(vec["mvis"]) == 7
    _real_getenv = _os.getenv
    for i, row in enumerate(vec["mvis"]):
        name, vis, hidden, kept, varlist = row
        saved: dict = {}
        try:
            for k in ("ENABLED_MARKETS", "SHOW_CN_STOCK", "SHOW_HK_STOCK"):
                if k in _os.environ:
                    saved[k] = _os.environ[k]
                    del _os.environ[k]
            for k, v in varlist:
                _os.environ[k] = v
            py_vis = [py_visible(m) for m in _mv_markets]
            py_hidden_sorted = sorted(py_hidden())
            py_kept = []
            for it in py_filter(_mv_items):
                py_kept.append(f"S:{it}" if isinstance(it, str)
                               else f"M:{it.get('value', '')}")
        finally:
            for k in ("ENABLED_MARKETS", "SHOW_CN_STOCK", "SHOW_HK_STOCK"):
                _os.environ.pop(k, None)
            _os.environ.update(saved)
        check(f"mvis[{i}].name", name, vec["mvis"][i][0])
        check(f"mvis[{i}].vis", vis, py_vis)
        check(f"mvis[{i}].hidden", hidden, py_hidden_sorted)
        check(f"mvis[{i}].kept", kept, py_kept)

    if FAILURES:
        print(f"\n{len(FAILURES)} parity FAILURES")
        return 1
    print("parity OK: all Rust vectors match Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
