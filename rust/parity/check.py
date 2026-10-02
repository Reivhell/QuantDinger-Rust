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
_load("app.utils.language", "app/utils/language.py")
_load("app.utils.timeutil", "app/utils/timeutil.py")
_load("app.utils.risk_guard", "app/utils/risk_guard.py")
_load("app.utils.technical_indicators", "app/utils/technical_indicators.py")
_load("app.utils.trade_net_pnl", "app/utils/trade_net_pnl.py")
_load("app.services.grid.levels", "app/services/grid/levels.py")
_load("app.services.strategy_v2.frequencies", "app/services/strategy_v2/frequencies.py")
_load("app.services.strategy_v2.models", "app/services/strategy_v2/models.py")
_load("app.services.market_schedule", "app/services/market_schedule.py")
_load("app.services.backtest_metrics", "app/services/backtest/metrics.py")
_load("app.services.strategy_v2.instruments", "app/services/strategy_v2/instruments.py")
# readiness.py needs only StrategyV2ContractError (a ValueError with .code)
# from the heavy contract module — stub it so the real readiness loads.
_contract_stub = _stub("app.services.strategy_v2.contract")
exec(  # noqa: S102
    "class StrategyV2ContractError(ValueError):\n"
    "    def __init__(self, code):\n"
    "        super().__init__(code)\n"
    "        self.code = code\n",
    _contract_stub.__dict__,
)
setattr(sys.modules["app.services.strategy_v2"], "contract", _contract_stub)
_load("app.services.strategy_v2.readiness", "app/services/strategy_v2/readiness.py")
_load("app.services.strategy_v2.snapshot_mod", "app/services/strategy_v2/snapshot.py")
_load("app.services.strategy_v2.data_portal", "app/services/strategy_v2/data.py")
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

    # --- data_portal: same 5-bar frame through the real portal ---
    from app.services.strategy_v2.data_portal import (  # noqa: E402
        MultiAssetDataPortal as PyPortal,
    )
    _PD = 86_400
    _PB = 1_767_225_600
    _idx = pd.date_range(pd.Timestamp(_PB, unit="s"), periods=5, freq="D")
    _py_frame = pd.DataFrame(
        {"open": [100.0 + i for i in range(5)],
         "high": [101.0 + i for i in range(5)],
         "low": [99.0 + i for i in range(5)],
         "close": [100.5 + i for i in range(5)],
         "volume": [1000.0] * 5,
         "suspended": [0.0] * 5},
        index=_idx,
    )
    _portal = PyPortal({"Crypto:BTC/USDT@spot": _py_frame})
    _dp = vec["dportal"]
    assert len(_dp) == 13, len(_dp)
    for i, (now, incl) in enumerate([(_PB + 2 * _PD + 6 * 3_600, False),
                                     (_PB + 2 * _PD + 6 * 3_600, True),
                                     (_PB - _PD, False)]):
        _portal.set_clock(pd.Timestamp(now, unit="s"), include_current=incl)
        _end = len(_portal.visible_frame("BTCUSDT"))
        _cur = _portal.current("BTCUSDT", default=-1.0)
        _cur_miss = _portal.current("BTCUSDT", field="nope", default=-1.0)
        _row = _dp[i]
        check(f"dportal[{i}].cutoff", _row[0],
              _PB + 2 * _PD + 6 * 3_600 - _PD + (_PD if incl else 0)
              if i < 2 else _PB - 2 * _PD)
        check(f"dportal[{i}].end", _row[1], _end)
        check(f"dportal[{i}].cur", _row[2], _cur, tol=1e-12)
        check(f"dportal[{i}].cur_nan", _row[3], _cur_miss, tol=1e-12)
        _pp = _cur if _cur > 0 else None
        check(f"dportal[{i}].pprice", _row[4], _pp)
    _portal.set_clock(_idx[2], include_current=True)
    _bar = _portal.bar_at("BTCUSDT", _idx[1])
    check("dportal.bar.hit", _dp[3],
          [(_PB + _PD), _bar["open"], _bar["high"], _bar["low"], _bar["close"],
           _bar["volume"]])
    check("dportal.bar.miss", _dp[4], [(_PB + 10 * _PD), None])
    check("dportal.resolve.direct", _dp[5],
          ["Crypto:BTC/USDT@spot", _portal.resolve_key("Crypto:BTC/USDT@spot")])
    check("dportal.resolve.alias", _dp[6], ["BTCUSDT", _portal.resolve_key("BTCUSDT")])
    check("dportal.resolve.parsed", _dp[7],
          ["Crypto:BTC/USDT", _portal.resolve_key("Crypto:BTC/USDT")])
    try:
        _portal.resolve_key("NOPE")
        _nope: object = "NO-ERR"
    except Exception as e:  # noqa: BLE001
        _nope = f"ERR:{e}"
    check("dportal.resolve.miss", _dp[8], ["NOPE", _nope])
    try:
        PyPortal({"USStock:K": pd.DataFrame()})
        _e1: object = "NO-ERR"
    except Exception as e:  # noqa: BLE001
        _e1 = str(e)
    check("dportal.err.empty", _dp[9], _e1)
    try:
        PyPortal({"USStock:K": pd.DataFrame({"open": [1.0], "close": [1.5]}, index=[_idx[0]])})
        _e2: object = "NO-ERR"
    except Exception as e:  # noqa: BLE001
        _e2 = str(e)
    check("dportal.err.nohl", _dp[10], _e2)
    try:
        _dup = pd.DataFrame(
            {"open": [1.0, 1.0], "high": [2.0, 2.0], "low": [0.5, 0.5],
             "close": [1.5, 9.5]},
            index=[_idx[0], _idx[0]],
        )
        PyPortal({"USStock:K": _dup})
        _e3: object = "NO-ERR"
    except Exception as e:  # noqa: BLE001
        _e3 = str(e)
    check("dportal.err.confl", _dp[11], _e3)
    check("dportal.symbol", _dp[12], "BTC/USDT")

    # --- snapshot: canonical bytes + float rendering + id checks ---
    import hashlib as _hl  # noqa: E402
    from app.services.strategy_v2.snapshot_mod import (  # noqa: E402
        canonical_frame_bytes as py_canon,
    )
    _sidx = pd.date_range("2026-01-01", periods=3, freq="4h")
    _sframe = pd.DataFrame(
        {"open": [100.1, 101.2, 102.3], "high": [101.1, 102.2, 103.3],
         "low": [99.1, 100.2, 101.3], "close": [100.8, 101.9, 102.7],
         "volume": [10.0, 11.0, 12.0]},
        index=_sidx,
    )
    _py_bytes = py_canon(_sframe)
    _py_hex = _hl.sha256(_py_bytes).hexdigest()
    _sn = vec["snap"]
    assert len(_sn) == 11, len(_sn)
    check("snap.bytes", _sn[0].replace("'", '"'), _py_bytes.decode("utf-8"))
    check("snap.sha256", _sn[1], _py_hex)
    for i, v in enumerate([100.0, 0.1, 1e16, 1e-7, -2.5e100, 123456789.123456789]):
        check(f"snap.repr[{i}]", _sn[2 + i], repr(v))
    check("snap.id.bad", _sn[8], "ERR:strategyV2.snapshotIdInvalid")
    check("snap.id.upper", _sn[9], "ok:" + "a" * 64)
    check("snap.id.ok", _sn[10], f"ok:{'ab' * 32}")

    # --- readiness: same gates through the real validators ---
    from app.services.strategy_v2.readiness import (  # noqa: E402
        validate_fundamentals as py_fund,
        validate_universe_history as py_uni,
        validate_warmup as py_warmup,
    )
    _RB = 1_767_225_600
    _RD = 86_400

    def _rd_call(fn, *args):
        try:
            fn(*args)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERR:{e}"

    _rdy = vec["ready"]
    assert len(_rdy) == 10, len(_rdy)
    _uni_base = {"code": "U1", "history_from": "2026-01-01",
                 "metadata": {"snapshot_only": True}}
    _uni_off = dict(_uni_base, metadata={"snapshot_only": False})
    check("ready.uni.before", _rdy[0],
          _rd_call(py_uni, _uni_base, pd.Timestamp(_RB - _RD, unit="s")))
    check("ready.uni.onday", _rdy[1],
          _rd_call(py_uni, _uni_base, pd.Timestamp(_RB, unit="s")))
    check("ready.uni.optout", _rdy[2],
          _rd_call(py_uni, _uni_off, pd.Timestamp(_RB - _RD, unit="s")))
    _widx = pd.date_range(pd.Timestamp(_RB, unit="s"), periods=5, freq="D")

    def _wframe(bad: bool):
        d = {"open": [100.0] * 5, "high": [101.0] * 5,
             "low": [99.0] * 5, "close": [100.5] * 5}
        if bad:
            d["open"][1] = 0.0
            d["close"][1] = None
        return pd.DataFrame(d, index=_widx)

    _wstart = pd.Timestamp(_RB + 5 * _RD, unit="s")
    check("ready.warm.5bad", _rdy[3],
          _rd_call(py_warmup, {"1d": {"S": _wframe(True)}}, 5, _wstart))
    check("ready.warm.4bad", _rdy[4],
          _rd_call(py_warmup, {"1d": {"S": _wframe(True)}}, 4, _wstart))
    check("ready.warm.5ok", _rdy[5],
          _rd_call(py_warmup, {"1d": {"S": _wframe(False)}}, 5, _wstart))
    _members = ({"key": "S", "valid_from": pd.Timestamp(_RB + 3 * _RD, unit="s")},)
    check("ready.warm.member", _rdy[6],
          _rd_call(py_warmup, {"1d": {"S": _wframe(False)}}, 5,
                   pd.Timestamp(_RB, unit="s"), _members))
    _fframe = pd.DataFrame(
        {"pe": [10.0, float("inf"), None], "pb": [None, None, None]},
        index=pd.date_range(pd.Timestamp(_RB, unit="s"), periods=3, freq="D"),
    )
    _req = {"pe", "pb", "roe"}
    check("ready.fund.full", _rdy[7],
          _rd_call(py_fund, {"S": _fframe}, _req))
    check("ready.fund.cut", _rdy[8],
          _rd_call(py_fund, {"S": _fframe}, _req,
                   pd.Timestamp(_RB - _RD, unit="s")))
    import datetime as _dt  # noqa: E402
    check("ready.caldate", _rdy[9],
          str((_dt.datetime(2025, 12, 31, 23, tzinfo=_dt.timezone.utc)
               + _dt.timedelta(hours=2)).date()))

    # --- language: normalize + detect through the real helpers ---
    from app.utils.language import (  # noqa: E402
        _normalize_lang as py_norm_lang,
        detect_request_language as py_detect,
    )

    class _Headers(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class _Req:
        def __init__(self, headers=None, args=None):
            self.headers = _Headers(headers or {})
            self.args = _Headers(args or {})

    _norm_cases = ["en", "en-US,en;q=0.9", " zh-hans ", "zh-Hant", "JA-jp",
                   "fr-FR; q=0.8", "", "   ", "xx-YY", "en;",
                   "KO-kr;q=0.5, en;q=0.3", "zh"]
    assert len(vec["lang"]) == len(_norm_cases) + 5
    for i, raw in enumerate(_norm_cases):
        check(f"lang.norm[{i}]", vec["lang"][i], [raw, py_norm_lang(raw)])
    _det_cases = [
        ({"X-App-Lang": "ja-JP", "Accept-Language": "ko-KR"},
         {"language": "fr-FR"}, {"language": "de-DE"}, "en-US"),
        ({"X-App-Lang": "xx"}, {"language": "fr-FR"}, {}, "en-US"),
        ({}, {}, {"language": "th-TH"}, "en-US"),
        ({}, {}, {}, "zh-CN"),
        ({}, {"language": "xx"}, {}, "en-US"),
    ]
    # 5th case also exercises Accept-Language fallback via header
    _det_cases[4][0]["Accept-Language"] = "ar-SA"
    for i, (h, body, q, dflt) in enumerate(_det_cases):
        req = _Req(headers=h, args=q)
        check(f"lang.detect[{i}]", vec["lang"][len(_norm_cases) + i],
              py_detect(req, body if body else None, dflt))

    # --- timeutil: same inputs through the real to_utc_iso ---
    import datetime as _dtime  # noqa: E402
    from app.utils.timeutil import to_utc_iso as py_to_utc  # noqa: E402
    _num_cases = [0.0, 1_767_225_600.0, 1_767_225_600_123.0, 1.0, -1.0]
    assert len(vec["tutil"]) == len(_num_cases) + 14 + 2, len(vec["tutil"])
    for i, ts in enumerate(_num_cases):
        check(f"tutil.num[{i}]", vec["tutil"][i], py_to_utc(ts))
    _str_cases = [
        "2026-01-01T00:00:00Z",
        "2026-01-01 00:00:00",
        "2026-01-01",
        "2026-01-01T08:00:00+08:00",
        "2026-01-01T08:00:00+0800",
        "2026-01-01T00:00:00.123456",
        "  2026-01-01T00:00:00Z  ",
        "2026-01-01T00:00:00.5+02:00",
        "",
        "   ",
        "not-a-date",
        "2026-13-01",
        "2026-01-01T25:00:00",
        "2026-01-01T00:00:00+99:99",
    ]
    for i, s in enumerate(_str_cases):
        check(f"tutil.str[{i}]", vec["tutil"][len(_num_cases) + i],
              [s, py_to_utc(s)])
    _aware = _dtime.datetime(2026, 1, 1, tzinfo=_dtime.timezone(_dtime.timedelta(hours=8)))
    check("tutil.aware", vec["tutil"][len(_num_cases) + 14], py_to_utc(_aware))
    check("tutil.other", vec["tutil"][len(_num_cases) + 15], py_to_utc(object()))

    # --- manifest: same manifest shape through the real dataclasses ---
    import json as _json  # noqa: E402
    from app.services.strategy_v2.models import (  # noqa: E402
        InstrumentSpec as PyInst,
        ScheduleSpec as PySched,
        StrategyManifest as PyMani,
        SubscriptionSpec as PySub,
        UniverseSpec as PyUni,
    )
    _mk = lambda m, s: PyInst(market=m, symbol=s, instrument_id=f"{m}:{s}")
    _py_mani = PyMani(
        api_version=2, code_hash="abc", strategy_type="test",
        universe=PyUni(kind="static",
                       instruments=(_mk("USStock", "AAPL"), _mk("Crypto", "BTC/USDT"))),
        subscriptions=(
            PySub(instruments=(_mk("USStock", "AAPL"),), frequency="1h"),
            PySub(instruments=(_mk("Crypto", "BTC/USDT"),), frequency="1d"),
        ),
        schedules=(PySched(frequency="1d", callback="on_open", time="09:30"),),
        benchmark=_mk("USStock", "SPY"),
        handlers=("on_bar",),
        fundamental_dependencies=("pe",),
        warmup_bars=50, direction_mode="long",
    )
    _mv = vec["mani"]
    assert len(_mv) == 8, len(_mv)
    check("mani.markets", _mv[0], ",".join(_py_mani.markets))
    check("mani.primary", _mv[1], _py_mani.primary_frequency)
    check("mani.freqs", _mv[2], ",".join(_py_mani.frequencies))
    check("mani.driving", _mv[3], _py_mani.driving_frequency)
    check("mani.metadata", _json.loads(_mv[4].replace("'", '"')),
          _json.loads(_json.dumps(_py_mani.metadata(), sort_keys=True,
                                  default=str)))
    for i, f in enumerate([1.0, 2.5, 0.1]):
        check(f"mani.float[{i}]", _mv[5 + i], repr(f))

    # --- json_helpers / notification_display / local_brokers ---
    import json as _json2  # noqa: E402
    _jh = _load("app.utils.json_helpers", "app/utils/json_helpers.py")
    _nd = _load("app.utils.notification_display", "app/utils/notification_display.py")
    _lb = _load("app.utils.local_brokers", "app/utils/local_brokers.py")
    _py_default = {"d": 1}
    _jdocs = [
        '{"a":1,"b":[1.5,-2e-3,true,null,"x"]}',
        '{"caf\\u00e9":"A","emoji":"\\ud83d\\ude00","esc":"A\\n\\t\\"q\\"\\\\"}',
        "[1,2,3]",
        '"s"',
        "123",
        "-0.5",
        "1E+16",
        '{"big":123456789123456789123456789}',
        '{"x":1.5e-7}',
        '  { "w" : [ true , false ] }  ',
        '"a\\ud83d\\ude00b"',
        "",
        "   ",
        "{bad",
        "[1,]",
        "01",
        '{"a":1} x',
        "123abc",
        "nul",
    ]
    assert len(vec["jres"]) == len(_jdocs) + 2, len(vec["jres"])
    for i, d in enumerate(_jdocs):
        check(f"jres[{i}]", vec["jres"][i],
              _jh.safe_json_loads(d, dict(_py_default)))
    check("jres.passthrough.dict", vec["jres"][len(_jdocs)], {"k": 1})
    check("jres.passthrough.list", vec["jres"][len(_jdocs) + 1], [True])
    # non-str / non-container inputs yield default
    for other in (None, 5, 1.5, True, (1, 2)):
        check(f"jres.other.{type(other).__name__}",
              _jh.safe_json_loads(other, dict(_py_default)), _py_default)
    _py_notif = [
        _nd.with_display({"title": "T", "body": "B"}, "fill",
                         {"symbol": "BTC", "n": 2, "px": 100.5}),
        _nd.with_display({"a": 1, "display": "stale"}, "t", {}),
        _nd.with_display(None, "x", {}),
    ]
    assert len(vec["notif"]) == 3, len(vec["notif"])
    for i, p in enumerate(_py_notif):
        check(f"notif[{i}]", vec["notif"][i], p)
    _lvals = [None, "1", "true", "YES", " On ", "0", "false", "", "  ", "no", "off"]
    assert len(vec["brokers"]) == len(_lvals) + 1, len(vec["brokers"])
    for i, v in enumerate(_lvals):
        saved_lb = _os.environ.get("ALLOW_LOCAL_DESKTOP_BROKERS")
        try:
            _os.environ.pop("ALLOW_LOCAL_DESKTOP_BROKERS", None)
            if v is not None:
                _os.environ["ALLOW_LOCAL_DESKTOP_BROKERS"] = v
            py_allowed = _lb.local_desktop_brokers_allowed()
            try:
                _lb.require_local_desktop_brokers_allowed()
                py_ok, py_msg = True, ""
            except PermissionError as e:
                py_ok, py_msg = False, str(e)
        finally:
            _os.environ.pop("ALLOW_LOCAL_DESKTOP_BROKERS", None)
            if saved_lb is not None:
                _os.environ["ALLOW_LOCAL_DESKTOP_BROKERS"] = saved_lb
        rust_v, rust_flags = vec["brokers"][i]
        check(f"brokers[{i}].env", rust_v, v)
        check(f"brokers[{i}].flags", rust_flags, [py_allowed, py_ok])
        if not py_ok:
            check(f"brokers[{i}].msg", vec["brokers"][len(_lvals)], py_msg)
    check("brokers.msg", vec["brokers"][len(_lvals)],
          _lb.desktop_broker_cloud_reject_message())

    # --- market_data_errors: classify through the real module ---
    _stub("app.data_sources")
    _mde = _load("app.data_sources.errors", "app/data_sources/errors.py")
    _cerrs = [
        "451 restricted location",
        "ProxyError: tunnel connection failed",
        "symbol not found on market",
        "429 too many requests",
        "incomplete kline coverage",
        "connection refused by peer",
        "timeframe unsupported here",
        "weird thing",
        "",
        "451 proxyerror tunnel",
        "get https://user:pass@api.x.com/a and HTTP://a:b@h.io/b failed: timeout",
        "https://host/path no creds",
        "https://u@host/",
        "caf\u00e9 timeout \u2603",
    ]
    assert len(vec["cerr"]) == len(_cerrs), len(vec["cerr"])
    for i, e in enumerate(_cerrs):
        f = _mde.classify_market_data_failure(e, exchange_id=" Binance ",
                                              market_type="SPOT", symbol="BTC/USDT",
                                              timeframe="1h")
        d = f.as_dict()
        check(f"cerr[{i}]", vec["cerr"][i],
              [d["code"], d["message"], d["technical_detail"], d["retryable"],
               d["exchange_id"], d["market_type"], d["symbol"], d["timeframe"]])
    # from_mapping through the real dataclass
    _mmap = [
        (None, None, "", None),
        ("rate_limited", "M", "d", True),
        ("", "", "", False),
        ("x", "y", "\u00e9" * 600, 0),
        (None, None, "", 1),
        (None, None, "", ""),
        (None, None, "", "x"),
        (None, None, "", []),
        (None, None, "", {}),
    ]
    assert len(vec["cmap"]) == len(_mmap), len(vec["cmap"])
    for i, (c, m, d, r) in enumerate(_mmap):
        fields = {"technical_detail": d}
        if c is not None:
            fields["code"] = c
        if m is not None:
            fields["message"] = m
        if r is not None:
            fields["retryable"] = r
        f = _mde.MarketDataFailure.from_mapping(fields)
        check(f"cmap[{i}]", vec["cmap"][i], [f.code, f.message, f.technical_detail, f.retryable])

    # --- strategy_runtime_logs: stub db/logger, load real module ---
    import logging as _logging  # noqa: E402
    _db_stub = _stub("app.utils.db")
    _db_stub.get_db_connection = lambda: (_ for _ in ()).throw(RuntimeError("no db in parity"))
    _log_stub = _stub("app.utils.logger")
    _log_stub.get_logger = _logging.getLogger
    _srl = _load("app.utils.strategy_runtime_logs", "app/utils/strategy_runtime_logs.py")
    check("cfmt.len", len(vec["cfmt"]), 3)
    for i, e in enumerate(["connection refused by peer", "caf\u00e9 timeout \u2603", ""]):
        f = _mde.classify_market_data_failure(e, exchange_id="binance",
                                              market_type="spot", symbol="BTC",
                                              timeframe="1h")
        check(f"cfmt[{i}]", vec["cfmt"][i], _srl.format_market_data_log(f))
    # parse probes: compare decoded dicts (None for rejects)
    _parse_in = [
        _srl.format_market_data_log(_mde.classify_market_data_failure(
            "timeout", exchange_id="binance", market_type="spot",
            symbol="BTC", timeframe="1h")),
        "plain line",
        "market-data|[1,2]",
        "market-data|{bad",
        "market-data|",
        "market-data|5",
        'market-data|  {"k" : 1 }  ',
    ]
    assert len(vec["cparse"]) == len(_parse_in), len(vec["cparse"])
    for i, line in enumerate(_parse_in):
        check(f"cparse[{i}]", vec["cparse"][i], [line, _srl.parse_market_data_log(line)])
    # normalize probes = append_strategy_log minus DB: replicate pure prelude
    def _py_norm(sid, ok, lv, mg):
        if not ok:
            return None
        l = (lv or "info").strip().lower()[:20]
        m = (mg or "").strip()
        if not m:
            return None
        return [sid, l, m[:8000]]
    _norm_in = [
        (7, True, " WARNING ", "  hi  "),
        (7, True, None, "m"),
        (7, False, None, "m"),
        (7, True, None, "   "),
        (7, True, None, None),
        (1, True, "x" * 40, "m"),
        (1, True, None, "y" * 9000),
        (1, True, None, "\U0001F600" * 9000),
        (1, True, "\u00e9X " * 10, "m"),
    ]
    assert len(vec["cnorm"]) == len(_norm_in), len(vec["cnorm"])
    for i, (sid, ok, lv, mg) in enumerate(_norm_in):
        check(f"cnorm[{i}]", vec["cnorm"][i], _py_norm(sid, ok, lv, mg))

    # --- thread_capacity: same cgroup files through the real module ---
    import re as _re  # noqa: E402
    _tc = _load("app.utils.thread_capacity", "app/utils/thread_capacity.py")
    _py_snap = _tc.thread_capacity_snapshot()
    assert len(vec["tcap"]) == 5, len(vec["tcap"])
    assert isinstance(vec["tcap"][0], int) and vec["tcap"][0] >= 1, vec["tcap"][0]
    assert isinstance(_py_snap["python_threads"], int) and _py_snap["python_threads"] >= 1
    for i, k in enumerate(["pids_current", "pids_max", "memory_current", "memory_max"]):
        rust_v, py_v = vec["tcap"][1 + i], _py_snap[k]
        if py_v is None:
            check(f"tcap.{k}", rust_v, "none")
        else:
            # cgroup-wide counters: stable limits exact, live counters shape-checked
            if k.endswith("_max"):
                check(f"tcap.{k}", rust_v, f"v:{py_v}")
            else:
                check(f"tcap.{k}.present", rust_v.startswith("v:"), True)
                check(f"tcap.{k}.numeric", bool(_re.fullmatch(r"v:\d+", rust_v)), True)
                check(f"tcap.{k}.pynumeric", str(py_v).isdigit(), True)
    _norm = lambda s: _re.sub(r"python_threads=\d+", "python_threads=N", s)
    check("tcap_fmt", _norm(vec["tcap_fmt"]), _norm(_tc.format_thread_capacity()))

    # --- credential_crypto: cross-implementation Fernet vectors ---
    import base64 as _b64  # noqa: E402
    import hashlib as _hl  # noqa: E402
    import hmac as _hmac  # noqa: E402
    import os as _os2  # noqa: E402
    import time as _time  # noqa: E402

    # Pure-Python Fernet stand-in for the `cryptography` package (unavailable
    # offline). SBOX is *derived* via GF(2^8) math, and both AES directions are
    # anchored to the FIPS-197 Appendix B vector below — not to the Rust code.
    def _gf_mul(a: int, b: int) -> int:
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return p

    def _gf_pow(a: int, e: int) -> int:
        r = 1
        while e:
            if e & 1:
                r = _gf_mul(r, a)
            a = _gf_mul(a, a)
            e >>= 1
        return r

    def _rot(b: int, n: int) -> int:
        return ((b << n) | (b >> (8 - n))) & 0xFF

    _SBOX = []
    for _i in range(256):
        _inv = _gf_pow(_i, 254) if _i else 0
        _SBOX.append(_inv ^ _rot(_inv, 1) ^ _rot(_inv, 2) ^ _rot(_inv, 3) ^ _rot(_inv, 4) ^ 0x63)
    _INV_SBOX = [0] * 256
    for _i, _s in enumerate(_SBOX):
        _INV_SBOX[_s] = _i
    _RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

    def _expand(_key: bytes):
        _rk = [bytearray(_key)]
        for _i in range(1, 11):
            _t = bytearray(_SBOX[b] for b in (_rk[-1][13], _rk[-1][14], _rk[-1][15], _rk[-1][12]))
            _t[0] ^= _RCON[_i - 1]
            _n = bytearray(16)
            for _j in range(4):
                _n[_j] = _rk[-1][_j] ^ _t[_j]
            for _j in range(4, 16):
                _n[_j] = _rk[-1][_j] ^ _n[_j - 4]
            _rk.append(_n)
        return _rk

    # (full block encrypt built explicitly to keep shift/mix readable)
    def _shift(_s: bytearray) -> bytearray:
        return bytearray([_s[0], _s[5], _s[10], _s[15], _s[4], _s[9], _s[14], _s[3],
                          _s[8], _s[13], _s[2], _s[7], _s[12], _s[1], _s[6], _s[11]])

    def _inv_shift(_s: bytearray) -> bytearray:
        return bytearray([_s[0], _s[13], _s[10], _s[7], _s[4], _s[1], _s[14], _s[11],
                          _s[8], _s[5], _s[2], _s[15], _s[12], _s[9], _s[6], _s[3]])

    def _mix(_s: bytearray) -> bytearray:
        def _xt(_x):
            return ((_x << 1) & 0xFF) ^ (0x1B if _x & 0x80 else 0)
        _o = bytearray(16)
        for _c in range(4):
            _a0, _a1, _a2, _a3 = _s[4*_c:4*_c+4]
            _o[4*_c] = _xt(_a0) ^ (_xt(_a1) ^ _a1) ^ _a2 ^ _a3
            _o[4*_c+1] = _a0 ^ _xt(_a1) ^ (_xt(_a2) ^ _a2) ^ _a3
            _o[4*_c+2] = _a0 ^ _a1 ^ _xt(_a2) ^ (_xt(_a3) ^ _a3)
            _o[4*_c+3] = (_xt(_a0) ^ _a0) ^ _a1 ^ _a2 ^ _xt(_a3)
        return _o

    def _inv_mix(_s: bytearray) -> bytearray:
        _o = bytearray(16)
        for _c in range(4):
            _a0, _a1, _a2, _a3 = _s[4*_c:4*_c+4]
            _o[4*_c] = _gf_mul(_a0, 0x0E) ^ _gf_mul(_a1, 0x0B) ^ _gf_mul(_a2, 0x0D) ^ _gf_mul(_a3, 0x09)
            _o[4*_c+1] = _gf_mul(_a0, 0x09) ^ _gf_mul(_a1, 0x0E) ^ _gf_mul(_a2, 0x0B) ^ _gf_mul(_a3, 0x0D)
            _o[4*_c+2] = _gf_mul(_a0, 0x0D) ^ _gf_mul(_a1, 0x09) ^ _gf_mul(_a2, 0x0E) ^ _gf_mul(_a3, 0x0B)
            _o[4*_c+3] = _gf_mul(_a0, 0x0B) ^ _gf_mul(_a1, 0x0D) ^ _gf_mul(_a2, 0x09) ^ _gf_mul(_a3, 0x0E)
        return _o

    def _block_enc(_key: bytes, _blk: bytes) -> bytes:
        _rk, _s = _expand(_key), bytearray(_blk)
        for _i in range(16):
            _s[_i] ^= _rk[0][_i]
        for _r in range(1, 10):
            _s = bytearray(_SBOX[b] for b in _s)
            _s = _shift(_s)
            _s = _mix(_s)
            for _i in range(16):
                _s[_i] ^= _rk[_r][_i]
        _s = bytearray(_SBOX[b] for b in _s)
        _s = _shift(_s)
        for _i in range(16):
            _s[_i] ^= _rk[10][_i]
        return bytes(_s)

    def _block_dec(_key: bytes, _blk: bytes) -> bytes:
        _rk, _s = _expand(_key), bytearray(_blk)
        for _i in range(16):
            _s[_i] ^= _rk[10][_i]
        for _r in range(9, 0, -1):
            _s = _inv_shift(_s)
            _s = bytearray(_INV_SBOX[b] for b in _s)
            for _i in range(16):
                _s[_i] ^= _rk[_r][_i]
            _s = _inv_mix(_s)
        _s = _inv_shift(_s)
        _s = bytearray(_INV_SBOX[b] for b in _s)
        for _i in range(16):
            _s[_i] ^= _rk[0][_i]
        return bytes(_s)

    # Anchor the oracle to FIPS-197 Appendix B before trusting it.
    assert _block_enc(bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
                      bytes.fromhex("3243f6a8885a308d313198a2e0370734")).hex() == \
        "3925841d02dc09fbdc118597196a0b32", "oracle AES self-check failed"
    assert _block_dec(bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
                      bytes.fromhex("3925841d02dc09fbdc118597196a0b32")).hex() == \
        "3243f6a8885a308d313198a2e0370734", "oracle AES self-check failed"

    class _StubInvalidToken(Exception):
        pass

    class _StubFernet:
        def __init__(self, key):
            raw = _b64.urlsafe_b64decode(key)
            self._sign, self._enc = raw[:16], raw[16:]

        def encrypt(self, data: bytes) -> bytes:
            iv = _os2.urandom(16)
            pad = 16 - len(data) % 16
            pt = data + bytes([pad]) * pad
            prev, ct = iv, b""
            for i in range(0, len(pt), 16):
                blk = bytes(a ^ b for a, b in zip(pt[i:i+16], prev))
                blk = _block_enc(self._enc, blk)
                ct += blk
                prev = blk
            body = b"\x80" + int(_time.time()).to_bytes(8, "big") + iv + ct
            return _b64.urlsafe_b64encode(body + _hmac.new(self._sign, body, _hl.sha256).digest())

        def decrypt(self, token) -> bytes:
            try:
                data = _b64.urlsafe_b64decode(token)
            except Exception:
                raise _StubInvalidToken
            if len(data) < 57 or data[0] != 0x80:
                raise _StubInvalidToken
            if not _hmac.compare_digest(
                    _hmac.new(self._sign, data[:-32], _hl.sha256).digest(), data[-32:]):
                raise _StubInvalidToken
            prev, pt = data[9:25], b""
            for i in range(25, len(data) - 32, 16):
                dec = _block_dec(self._enc, data[i:i+16])
                pt += bytes(a ^ b for a, b in zip(dec, prev))
                prev = data[i:i+16]
            pad = pt[-1]
            if pad < 1 or pad > 16 or pt[-pad:] != bytes([pad]) * pad:
                raise _StubInvalidToken
            return pt[:-pad]

    _fernet_pkg = types.ModuleType("cryptography")
    _fernet_sub = types.ModuleType("cryptography.fernet")
    _fernet_sub.Fernet, _fernet_sub.InvalidToken = _StubFernet, _StubInvalidToken
    sys.modules["cryptography"], sys.modules["cryptography.fernet"] = _fernet_pkg, _fernet_sub
    _cc = _load("app.utils.credential_crypto", "app/utils/credential_crypto.py")
    _PyFernet = _StubFernet
    _CSEC = "qd-parity-secret"
    _CPLAIN = ["", '{"api_key":"ABC123"}', "caf\u00e9 \U0001F600", "x"]
    assert len(vec["cenc"]) == len(_CPLAIN) + 1, len(vec["cenc"])
    if True:  # stub oracle always available (cryptography missing offline)
        _pyf = _PyFernet(_b64.urlsafe_b64encode(_hl.sha256(_CSEC.encode()).digest()))
        # Rust-encrypted → Python decrypts (incl. deterministic fixed vector)
        for i, tok in enumerate(vec["cenc"]):
            want = _CPLAIN[i] if i < len(_CPLAIN) else "fixed-vector"
            check(f"cenc[{i}].pydec", _pyf.decrypt(tok.encode()).decode(), want)
        # Python-encrypted → Rust decrypt_probe decrypts
        _py_toks = [
            _pyf.encrypt(p.encode()).decode() for p in (["hello", ""] + _CPLAIN)
        ]
        _py_toks += [
            _pyf.encrypt(b"legacy").decode(),  # for key-fallback ordering
            _py_toks[0][:-4] + "AAAA",  # tampered → InvalidToken both sides
            "not-a-token!!",
            "",
            "caf\u00e9-token",
        ]
        _probe = subprocess.run(
            ["cargo", "run", "--quiet", "--example", "decrypt_probe", "--",
             _CSEC, *_py_toks],
            cwd=RUST, capture_output=True, text=True,
        )
        assert _probe.returncode == 0, _probe.stderr[-2000:]
        _rust_out = _json.loads(_probe.stdout)
        # Oracle = the REAL decrypt_credential_blob (env keys pointed at _CSEC).
        _saved_cred2 = _os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
        _saved_secret = _os.environ.get("SECRET_KEY")
        try:
            _os.environ["CREDENTIAL_ENCRYPTION_KEY"] = _CSEC
            _os.environ.pop("SECRET_KEY", None)
            _py_expect = []
            for tok in _py_toks:
                try:
                    _py_expect.append({"ok": _cc.decrypt_credential_blob(tok)})
                except UnicodeEncodeError:
                    _py_expect.append({"err": "NonAsciiInput"})
                except ValueError:
                    _py_expect.append({"err": "CannotDecrypt"})
        finally:
            _os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)
            if _saved_cred2 is not None:
                _os.environ["CREDENTIAL_ENCRYPTION_KEY"] = _saved_cred2
            if _saved_secret is not None:
                _os.environ["SECRET_KEY"] = _saved_secret
        assert len(_rust_out) == len(_py_expect), len(_rust_out)
        for i, (r, p) in enumerate(zip(_rust_out, _py_expect)):
            check(f"cdec[{i}]", r, p)
        # legacy fallback: token under old secret, new secret first
        _oldf = _PyFernet(_b64.urlsafe_b64encode(_hl.sha256(b"old-secret").digest()))
        _legacy_tok = _oldf.encrypt(b"legacy-data").decode()
        _probe2 = subprocess.run(
            ["cargo", "run", "--quiet", "--example", "decrypt_probe", "--",
             "new-secret,old-secret", _legacy_tok, _py_toks[0]],
            cwd=RUST, capture_output=True, text=True,
        )
        assert _probe2.returncode == 0, _probe2.stderr[-2000:]
        _r2 = _json.loads(_probe2.stdout)
        check("cdec.legacy", _r2[0], {"ok": "legacy-data"})
        check("cdec.legacy.py",
              _cc._fernet("old-secret").decrypt(_legacy_tok.encode()).decode(), "legacy-data")
        check("cdec.newkey_miss", _r2[1], {"err": "CannotDecrypt"})
        # key derivation identical
        check("ckey", vec["ckey"], _b64.urlsafe_b64encode(
            _hl.sha256(_CSEC.encode()).digest()).decode())
        # encrypt(None) → encrypts "" → decrypts to "" (with a key configured)
        _saved_cred = _os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
        try:
            _os.environ["CREDENTIAL_ENCRYPTION_KEY"] = "probe-key"
            _none_tok = _cc.encrypt_credential_blob(None)
        finally:
            _os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)
            if _saved_cred is not None:
                _os.environ["CREDENTIAL_ENCRYPTION_KEY"] = _saved_cred
        _probef = _PyFernet(_b64.urlsafe_b64encode(_hl.sha256(b"probe-key").digest()))
        check("cenc.none", _probef.decrypt(_none_tok.encode()).decode(), "")

    # --- direction + contract_builders: AST-free slice vs real modules ---
    _stub("app.services.factors")
    _load("app.services.factors.registry", "app/services/factors/registry.py")
    _load("app.services.factors.talib_adapter", "app/services/factors/talib_adapter.py")
    _load("app.services.factors", "app/services/factors/__init__.py")
    _sd = _load("app.services.strategy_direction", "app/services/strategy_direction.py")
    _load("app.utils.safe_exec", "app/utils/safe_exec.py")
    _contract = _load("app.services.strategy_v2.contract", "app/services/strategy_v2/contract.py")
    _instr = sys.modules["app.services.strategy_v2.instruments"]

    for i, _row in enumerate(vec["dvec"]):
        _inp = _row[0]
        check(f"dvec[{i}].norm", _row[1], _sd.normalize_direction_mode(_inp))
        check(f"dvec[{i}].side", _row[2], _sd.direction_mode_position_side(_inp))
        check(f"dvec[{i}].legs", _row[3], ",".join(sorted(_sd.direction_mode_owned_legs(_inp))))
        check(f"dvec[{i}].long", _row[4], _sd.direction_mode_allows(_inp, "long"))
        check(f"dvec[{i}].short", _row[5], _sd.direction_mode_allows(_inp, "short"))
    for i, _row in enumerate(vec["dman"]):
        check(f"dman[{i}]", _row[1], _sd.direction_mode_from_manifest(_row[0]))
    for i, _row in enumerate(vec["chash"]):
        check(f"chash[{i}]", _row[1], _contract.strategy_source_code_hash(_row[0]))
    for i, _row in enumerate(vec["cdetect"]):
        check(f"cdetect[{i}]", _row[1], _contract.is_strategy_v2_code(_row[0]))

    def _py_many(_js):
        try:
            return ",".join(_it.key for _it in _contract._parse_many(_js))
        except _contract.StrategyV2ContractError as _e:
            return f"ERR:C:{_e.code}"
        except _instr.InstrumentParseError as _e:
            return f"ERR:C:{_e}"
        except ValueError:
            return "ERR:V"
        except TypeError as _e:
            return f"ERR:T:{_e}"

    for i, _row in enumerate(vec["pmany"]):
        check(f"pmany[{i}]", _row[1], _py_many(_row[0]))

    def _on_open():
        pass

    def _kw_cb():
        pass

    _on_open.__name__ = "on_open"
    _kw_cb.__name__ = "kw_cb"

    class _Anon:
        def __call__(self, *a, **k):
            pass

    def _py_state(_ctx):
        _subs = ";".join(
            f"{_s.frequency}|{','.join(_s.fields)}|{_s.universe_reference}|"
            f"{','.join(_it.key for _it in _s.instruments)}"
            for _s in _ctx.subscriptions
        )
        _scheds = ";".join(
            f"{_s.frequency}|{_s.callback}|{_s.time}|"
            f"{'' if _s.weekday is None else _s.weekday}|"
            f"{'' if _s.monthday is None else _s.monthday}"
            for _s in _ctx.schedules
        )
        _ml = _ctx.max_leverage
        return [
            _ctx.universe_reference,
            ",".join(_it.key for _it in _ctx.instruments),
            _subs, _scheds,
            _ctx.benchmark.key if _ctx.benchmark is not None else None,
            _ctx.warmup_bars, _ctx.leverage_allowed,
            "nan" if _ml != _ml else repr(float(_ml)),
            dict(_ctx.metadata),
        ]

    _SCEN = [
        ("u_pool", [lambda c: c.set_universe(pool="my-pool")]),
        ("u_list", [lambda c: c.set_universe(["USStock:AAPL", "USStock:AAPL", "INDEX:HS300"])]),
        ("u_str", [lambda c: c.set_universe("BTCUSDT")]),
        ("u_absent", [lambda c: c.set_universe()]),
        ("u_pool_empty", [lambda c: c.set_universe(pool="")]),
        ("u_pool_null", [lambda c: c.set_universe(pool=None)]),
        ("u_index", [lambda c: c.set_universe(index="INDEX:HS300")]),
        ("u_dict", [lambda c: c.set_universe({"USStock:AAPL": 1, "MSFT": 2})]),
        ("u_badsym", [lambda c: c.set_universe("!!!")]),
        ("bench", [lambda c: c.set_benchmark("MSFT")]),
        ("bench_bad", [lambda c: c.set_benchmark("")]),
        ("sub_defaults", [lambda c: c.set_universe(["USStock:AAPL"]),
                           lambda c: c.subscribe()]),
        ("sub_custom", [lambda c: c.subscribe(["600519.XSHG"], frequency="1H",
                                              fields=["Close", " VOL "])]),
        ("sub_nouniv", [lambda c: c.subscribe(frequency="1d")]),
        ("warmup_seq", [lambda c: c.set_warmup("20"), lambda c: c.set_warmup("-5"),
                        lambda c: c.set_warmup(None)]),
        ("warmup_bad", [lambda c: c.set_warmup("x")]),
        ("lev_seq", [lambda c: c.allow_leverage("3"), lambda c: c.allow_leverage("0.5"),
                     lambda c: c.allow_leverage(None)]),
        ("lev_bad", [lambda c: c.allow_leverage("x")]),
        ("meta_seq", [lambda c: c.set_metadata({"a": "1"}, b="2"),
                      lambda c: c.set_metadata("k", "v")]),
        ("meta_1arg", [lambda c: c.set_metadata("only")]),
        ("meta_3args", [lambda c: c.set_metadata("a", "b", "c")]),
        ("sched_daily", [lambda c: c.daily(_on_open, time="09:30")]),
        ("sched_pos_beats_kw", [lambda c: c.weekly(_Anon(), weekday="3", callback=_kw_cb)]),
        ("sched_nocb", [lambda c: c.monthly(1)]),
        ("sched_monthly_def", [lambda c: c.monthly(_on_open)]),
        ("full", [lambda c: c.set_universe(["USStock:AAPL"]),
                  lambda c: c.subscribe(),
                  lambda c: c.set_warmup("50"),
                  lambda c: c.allow_leverage(2),
                  lambda c: c.set_metadata(frequency="1d"),
                  lambda c: c.daily(_on_open, time="09:30")]),
    ]
    # bind schedule calls (daily/weekly/monthly) through the real
    # _ScheduleBindings wrapper; set_*/subscribe go to the context.
    class _Redirect:
        """Dispatch set_*/subscribe to ctx, daily/weekly/monthly to bindings."""

        def __init__(self, ctx, sched):
            object.__setattr__(self, "_ctx", ctx)
            object.__setattr__(self, "_sched", sched)

        def __getattr__(self, name):
            if name in ("daily", "weekly", "monthly"):
                return getattr(object.__getattribute__(self, "_sched"), name)
            return getattr(object.__getattribute__(self, "_ctx"), name)

    def _py_run_bound(_ops):
        _ctx = _contract.DiscoveryContext()
        _both = _Redirect(_ctx, _contract._ScheduleBindings(_ctx))
        for _op in _ops:
            try:
                _op(_both)
            except _contract.StrategyV2ContractError as _e:
                return f"ERR:C:{_e.code}"
            except _instr.InstrumentParseError as _e:
                return f"ERR:C:{_e}"
            except TypeError as _e:
                return f"ERR:T:{_e}"
            except ValueError:
                return "ERR:V"
        return _py_state(_ctx)

    _rust_cvec = {r[0]: r[1] for r in vec["cvec"]}
    for _name, _ops in _SCEN:
        check(f"cvec[{_name}]", _rust_cvec[_name], _py_run_bound(_ops))

    if FAILURES:
        print(f"\n{len(FAILURES)} parity FAILURES")
        return 1
    print("parity OK: all Rust vectors match Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
