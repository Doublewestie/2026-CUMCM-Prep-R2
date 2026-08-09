"""Q3 守卫（spec_M4_Q3 D-1/D-2 优化题验收）：
锚点一致 / 全局弃电口径 / LP 平衡式（I2）/ SOC 终态与界内 / LP 交叉一致."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import load_step_module
from step0_config import CLEAN, OUTPUT, REGIONS

s32 = load_step_module("step3.2_q3_indicators.py")
OUT_Q3 = OUTPUT / "q3"

# 手算锚点（§1.3 独立核算，与 step3.2 不同代码路径复算核对）
ANCHOR_BASE = {
    "RegionA": {"cost_wan": 62302.2579, "carbon_t": 581269.7001,
                "peak_net_MW": 497.0016, "vol_std_MW": 96.7995,
                "max_ramp_MW": 385.4110},
    "RegionD": {"cost_wan": 8592.6414, "carbon_t": 262960.6536,
                "peak_net_MW": 282.5815, "vol_std_MW": 135.1128,
                "max_ramp_MW": 209.8956},
    "RegionE": {"cost_wan": -2589.2946, "carbon_t": 84555.5905,
                "peak_net_MW": 166.0604, "vol_std_MW": 116.2215,
                "max_ramp_MW": 257.2579},
}


def _report() -> dict:
    return json.loads((OUT_Q3 / "q3_indicators.json").read_text(encoding="utf-8"))


def test_products_exist():
    assert (OUT_Q3 / "q3_indicators.json").exists()
    assert (OUT_Q3 / "q3_baseline_vs_lp.csv").exists()
    df = pd.read_csv(OUT_Q3 / "q3_baseline_vs_lp.csv")
    assert set(df.columns) == {"Region", "metric", "baseline", "lp", "change_pct"}
    assert len(df) == len(REGIONS) * 7


def test_baseline_anchors():
    """基准四指标与手算锚点一致（rel < 1e-6，口径干净验收）。"""
    rep = _report()
    for r, exp in ANCHOR_BASE.items():
        for m, v in exp.items():
            got = rep["baseline"][r][m]
            assert abs(got - v) / max(abs(v), 1e-9) < 1e-6, f"{r}.{m} {got} vs {v}"


def test_global_anchor():
    """全局锚点：弃电 775.5 万 MWh / NU 32.9%（D4 口径）。"""
    ga = _report()["global_anchor"]
    assert abs(ga["curtail_MWh_global"] - 7754926.0) < 100.0
    assert abs(ga["nu_pct_global"] - 32.86) < 0.1


def test_lp_power_balance_identity():
    """LP 逐时平衡式 G + W + Pd = D + Pc + S + Q（I2，rel < 1e-6）。"""
    rt = s32.load_rt()
    for r in REGIONS:
        lp = pd.read_csv(OUT_Q3 / f"lp_baseline_{r}.csv")
        vec = s32._rt_vectors(rt, r)
        lhs = lp["G"].to_numpy() + vec["W"] + lp["Pd"].to_numpy()
        rhs = vec["D"] + lp["Pc"].to_numpy() + lp["S"].to_numpy() \
            + lp["Q"].to_numpy()
        resid = np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1.0)
        assert resid.max() < 1e-6, f"{r} 平衡式残差 {resid.max()}"


def test_lp_soc_bounds_and_terminal():
    """SOC 界内 + SOC(2406) ≥ InitialSOC（附件说明5 终态约束）。"""
    sp = pd.read_csv(CLEAN / "storage_params.csv").set_index("Region")
    for r in REGIONS:
        lp = pd.read_csv(OUT_Q3 / f"lp_baseline_{r}.csv")
        soc = lp["SOC"].to_numpy()
        lo, hi = sp.loc[r, "MinSOC_MWh"], sp.loc[r, "StorageCapacity_MWh"]
        assert soc.min() >= lo - 1e-4 and soc.max() <= hi + 1e-4, r
        final = soc[-1]
        assert final >= sp.loc[r, "InitialSOC_MWh"] - 1e-4, \
            f"{r} SOC(2406)={final} < InitialSOC"


def test_lp_indicators_match_json():
    """LP csv 重算四指标与 q3_indicators.json 一致（交叉验证，rel < 1e-9）。"""
    rep = _report()
    rt = s32.load_rt()
    for r in REGIONS:
        lp = pd.read_csv(OUT_Q3 / f"lp_baseline_{r}.csv")
        vec = s32._rt_vectors(rt, r)
        got = s32.evaluate_storage(lp["G"].to_numpy(), lp["S"].to_numpy(),
                                   vec["price"], vec["sellp"], vec["carbon"],
                                   Q=lp["Q"].to_numpy(), W=vec["W"])
        for m in ["cost_wan", "carbon_t", "peak_net_MW", "vol_std_MW",
                  "max_ramp_MW", "nu_pct"]:
            a, b = got[m], rep["lp"][r][m]
            assert abs(a - b) / max(abs(b), 1e-9) < 1e-9, f"{r}.{m}"


def test_ramp_worsening_documented():
    """M0 自由度红利副作用守卫：D 区 M0 爬坡 380 > M1 258 > 基准 210。"""
    rep = _report()
    d = rep["comparison"]
    d_row = [x for x in d if x["Region"] == "RegionD"
             and x["metric"] == "max_ramp_MW"][0]
    assert d_row["change_pct"] > 0, "D 区爬坡应恶化（若非，需更新叙事与多目标设计）"
    assert d_row["lp"] > d_row["baseline"]


def _model_report() -> dict:
    return json.loads((OUT_Q3 / "q3_model_evolve.json").read_text(
        encoding="utf-8"))


def test_m1_ramp_below_m0():
    """时段状态机约束（M1）应收敛 M0 的自由度红利爬坡：D 区 380→258。"""
    rep = _model_report()
    e2b = rep["e2b"]
    assert e2b["RegionD"]["M1"]["max"] < e2b["RegionD"]["M0"]["max"]
    assert e2b["RegionD"]["M0"]["max"] > e2b["RegionD"]["base"]["max"]


def test_m1_charge_all_renewable():
    """M1 充电来源守恒：六区 Pc_grid=0（充电时段内弃电优先，I7 口径归因）。

    若未来建模引入购电充电通道（M0 自由度），此守卫预警叙事变更。
    """
    import pandas as pd
    for r in REGIONS:
        src = pd.read_csv(OUT_Q3 / f"q3_lp_M1_{r}_charge_src.csv")
        assert src["Pc_grid"].abs().max() < 1e-6, f"{r} M1 出现购电充电"
        assert src["Pc"].sum() > 0


def test_e2a_structural_ramp():
    """E2a：M1 最优面内 min ramp == 原解（结构性爬坡，非顶点任意性）。"""
    rep = _model_report()
    e2a = rep["e2a"]
    assert e2a["status"] == 0
    m1_max = rep["e2b"]["RegionD"]["M1"]["max"]
    assert abs(e2a["min_ramp_MW"] - m1_max) < 1.0
    assert e2a["min_ramp_cost_wan"] <= e2a["cost_cap_wan"] * (1 + 1e-6) + 1e-6


def test_m1_cost_between():
    """三对照序：M1 成本 ∈ (M0 红利下界, 基准) —— 价值分解自洽。"""
    rep = _model_report()
    for r in REGIONS:
        rows = {(x["Region"], x["model"]): x for x in rep["comparison"]}
        m0 = rows[(r, "M0")]
        m1 = rows[(r, "M1")]
        base = rows[(r, "base")]
        assert m1["cost_wan"] >= m0["cost_wan"] - 1e-6, f"{r} M1<M0 异常"
        assert m1["cost_wan"] <= base["cost_wan"] * 1.0 + 1e-6 or \
            m1["cost_wan"] < 0, f"{r} M1 应不劣于基准（或双负）"


def _read(name: str):
    import json
    return json.loads((OUT_Q3 / name).read_text(encoding="utf-8"))


def test_dr_template_products():
    """轨A 产物：DR 逆向模板存在 + 充放时段非空 + 互斥。"""
    rep = _read("q3_dr_reverse.json")
    for r in REGIONS:
        t = rep["templates"][r]
        assert t["charge_hours"] and t["discharge_hours"]
        assert t["mutual_exclusive"] is True
        assert t["max_charge_MW"] < 300 and t["max_discharge_MW"] < 300


def test_pareto_collinearity_and_flat():
    """E1 共线 + 权衡面坍缩（M1 下成本-爬坡几乎无权衡）守卫。"""
    rep = _read("q3_pareto.json")
    for r in ["RegionA", "RegionD", "RegionE"]:
        e1 = rep["E1_collinearity"][r]
        assert e1["cost_carbon_corr"] > 0.9, f"{r} 共线实证缺失"
        fs = rep["E6_shape"][r]["frontier_cost_span_wan"]
        base_cost = abs(rep["frontier"][r]["eps_points"][0]["cost_wan"])
        assert fs / max(base_cost, 1e-9) < 0.05, f"{r} 前沿跨度异常大"


def test_ablation_products():
    """判别实验群：E5 西区外送饱和 + E7 外送价值 + E9 上限非瓶颈。"""
    rep = _read("q3_ablation.json")
    for r in ("RegionD", "RegionE", "RegionF"):
        assert rep[r]["E5_active_constraints"]["sell_saturation"] > 0.9
    e7 = rep["E7_sell_removal"]
    assert e7["RegionD"]["sell_value_wan"] > 5000
    assert e7["RegionE"]["sell_value_wan"] > 5000
    assert rep["E9_charge_cap"]["RegionD"]["gain_wan"] == 0.0


def test_rules_simulation_loss_documented():
    """规则化代价入册：CART 模拟损失存在且 T8 锚点复现。"""
    rep = _read("q3_storage_rules.json")
    for r in ("RegionD", "RegionA"):
        loss = rep["simulation_loss"][r]["cost_gap_pct"]
        assert loss is not None and loss > 0
        assert rep["anchors"][r]["T8_charge_above_q60_hours"] == 0


def test_mpc_and_sobol_products():
    """MPC 窗口效应 + Sobol V2 过门 + 主导参数稳定性。"""
    mpc = _read("q3_mpc.json")
    assert "RegionF" in mpc["window_effect"]
    sob = _read("q3_sobol.json")
    assert sob["V2_estimator"]["verdict"].startswith("【实证】估计器合格")
    d = sob["sobol"]["RegionD"]["cost_wan"]
    assert not d.get("degenerate")
    dom = d["S1"].index(max(d["S1"]))
    assert dom == 4, "D 区成本主导参数应为 sell_scale（与 E7 交叉验证）"


def test_mutex_conveyor_illusion():
    """传送带假象守卫：M0 同时充放 >80%；M0x 互斥=0；M0x < M1（时段红利>0）。"""
    rep = _read("q3_mutex.json")
    for r in REGIONS:
        m0 = rep["M0_free"][r]
        m0x = rep["M0x_mutex"][r]
        m1 = rep["M1_timed"][r]
        assert m0["simul_h"] / 2407 > 0.8, f"{r} M0 应大量同时充放（传送带）"
        assert m0x["simul_h"] == 0, f"{r} M0x 互斥应完全成立"
        assert m0x["cost_wan"] is not None and m0["cost_wan"] < m0x["cost_wan"] \
            < m1["cost_wan"], f"{r} 序应 M0<M0x<M1（传送带假象+时段红利）"
        dec = rep["decomposition"][r]
        assert dec["conveyor_illusion_wan"] < -3000, \
            f"{r} 传送带假象应显著（>3000 万）"
        assert dec["timing_freedom_value_wan"] < -3000, \
            f"{r} 时段红利应显著（>3000 万）"


def test_cross_experiment_verdicts():
    """交叉实验裁决守卫：斜坡免费（D 区）、结算段禁充免费、主时段指标干净。"""
    rep = _read("q3_cross_experiments.json")
    x0 = rep["X0_m1_baseline"]
    x1 = rep["experiments"]["X1_ramp_slope120"]["results"]["RegionD"]
    assert abs(x1["cost"] - x0["RegionD"]["cost"]) / x0["RegionD"]["cost"] < 0.01
    x7 = rep["experiments"]["X7_no_closure_charge_RegionD"]["results"]["RegionD"]
    assert abs(x7["cost"] - x0["RegionD"]["cost"]) / x0["RegionD"]["cost"] < 0.01
    # 主时段爬坡（东区）应显著低于全时段（Closure 边界污染）
    for r in ("RegionA", "RegionB", "RegionC"):
        main_r = rep["X13_main_period"][r]["ramp_main"]
        full_r = x0[r]["ramp"]
        assert main_r < full_r * 0.75, f"{r} 主时段爬坡应排除边界污染"


def test_extras_sigma_discrimination():
    """X15 σ 区分度：场景期望成本随 σ 单调上升；X16 sell 主导稳健。"""
    rep = _read("q3_extras.json")
    x15 = rep["X15_scenario_mpc"]
    c1 = x15["sigma_10"]["scenario_expected_cost_wan"]
    c2 = x15["sigma_20"]["scenario_expected_cost_wan"]
    c3 = x15["sigma_30"]["scenario_expected_cost_wan"]
    assert c1 is not None and c3 > c1, "σ↑ 期望成本应上升（波动成本实证）"
    s1 = rep["X16_sobol_sym"]["S1_sym_sell"]
    assert max(s1) == s1[4], "sell_scale 应仍主导（范围对称化后）"


def test_m3_final_main_caliber():
    """M3_final 主口径守卫：主时段指标存在 + 斜坡免费 + 上界序。"""
    ind = _read("q3_indicators.json")
    for r in REGIONS:
        m3 = ind["m3_final"][r]
        assert "ramp_main_MW" in m3 and m3["ramp_main_MW"] is not None
        assert m3["cost_wan"] < ind["baseline"][r]["cost_wan"] * 1.0 + 1e-6 or \
            m3["cost_wan"] < 0
    me = _read("q3_model_evolve.json")
    for r in REGIONS:
        rows = {(x["Region"], x["model"]): x for x in me["comparison"]}
        m3 = rows[(r, "M3")]
        m0x = rows[(r, "M0x")]
        base = rows[(r, "base")]
        # 斜坡/结算段/终态修正免费：M3 与 M1 成本差 <0.5%
        m1 = rows[(r, "M1")]
        assert abs(m3["cost_wan"] - m1["cost_wan"]) / abs(m1["cost_wan"]) < 0.005
        # 上界序：M3 <= 基准；M0x <= M3（物理上界更优）
        assert m3["cost_wan"] <= base["cost_wan"] * 1.0 + 1e-6 or m3["cost_wan"] < 0
        assert m0x["cost_wan"] <= m3["cost_wan"] + 1e-6


def test_rules_hourly_loss_reduced():
    """规则逐时损失 < 聚合版（sum_10 修正：聚合高估；逐时版 2.93/0.78%）。"""
    rep = _read("q3_storage_rules.json")
    d_loss = rep["simulation_loss"]["RegionD"]["cost_gap_pct"]
    a_loss = rep["simulation_loss"]["RegionA"]["cost_gap_pct"]
    assert d_loss is not None and d_loss < 6.0
    assert a_loss is not None and a_loss < 4.0
    assert d_loss < 5.76, "逐时版应显著低于聚合版 5.76%"
    for r in ("RegionD", "RegionA"):
        assert rep["anchors"][r]["T8_charge_above_q60_hours"] == 0


def test_sobol_unreliable_flag():
    """Sobol S1>1 归因不可靠标记（E carbon/A ramp_p95）。"""
    rep = _read("q3_sobol.json")
    assert rep["sobol"]["RegionE"]["carbon_t"]["unreliable"] is True
    assert rep["sobol"]["RegionA"]["peak_net_MW"]["degenerate"] is True
