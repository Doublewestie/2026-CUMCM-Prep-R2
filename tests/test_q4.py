"""test_q4 — Q4 三层验收断言（spec_M4_Q4 验收章节落地）.

口径层:  六指标手算锚点 rel<1e-9；Baseline 负荷下 solve_m3 == q3 m3_final
         （组合重现对账）；QOS∈[0,1]；峰谷比重构保均值
求解层:  LP 约束残差（功率平衡/SOC/弃电下界）<1e-6；M0x 充放互斥 Pc×Pd≈0；
         白名单零违约；SOC(2406)≥Init；碳限额收紧成本单调不减；可复现逐位一致
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_step(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".", "_"), ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ctx():
    s41 = load_step("step4.1_q4_indicators")
    return {**s41.load_ctx(), "s41": s41}


@pytest.fixture(scope="module")
def s40():
    return load_step("step4.0_q4_bilevel")


# ---------- 口径层 ----------

def test_hand_anchor_six_metrics(ctx):
    """手算锚点：Cost/Carbon/P_peak/Ramp 与手工核算 rel<1e-9（Q3 标准移植）。"""
    s32 = ctx["s32"]
    price = np.array([0.5, 0.6])
    sellp = np.array([0.0, 0.1])
    carbon = np.array([0.3, 0.4])
    G = np.array([10.0, 20.0])
    S = np.array([1.0, 2.0])
    Q = np.array([0.0, 0.0])
    W = np.array([100.0, 100.0])
    ev = s32.evaluate_storage(G, S, price, sellp, carbon, Q=Q, W=W)
    cost_expected = (0.5 * 10 - 0.0 * 1 + 0.6 * 20 - 0.1 * 2) / 1e4
    carbon_expected = 0.3 * 10 + 0.4 * 20
    assert abs(ev["cost_wan"] - cost_expected) / cost_expected < 1e-9
    assert abs(ev["carbon_t"] - carbon_expected) / carbon_expected < 1e-9
    assert abs(ev["peak_net_MW"] - 18.0) < 1e-9
    assert abs(ev["max_ramp_MW"] - 9.0) < 1e-9
    assert abs(ev["nu_pct"] - 100.0) < 1e-9


def test_baseline_reproduction_m3(ctx):
    """组合重现：Baseline 负荷下 solve_m3 与 q3_indicators m3_final 对账（产物 round 精度）。"""
    q3 = json.loads((ROOT / "output" / "q3" / "q3_indicators.json").read_text(
        encoding="utf-8"))["m3_final"]
    consume = ctx["consume"]
    tpl = ctx["tpl"]
    for r in ["RegionA", "RegionD"]:
        d = ctx["s33"]._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        m = ctx["s33"].solve_m3(d, ch, dh, region=r)
        rel_c = abs(m["cost_wan"] - q3[r]["cost_wan"]) / abs(q3[r]["cost_wan"])
        assert rel_c < 1e-3, f"{r} cost 对账失败 rel={rel_c}"


def test_qos_in_unit_interval(ctx):
    """QOS 定义域 [0,1]；完成率/裕度各自 ∈[0,1]。"""
    pol = ctx["s41"].q2_compromise_policy(ctx)
    sched = ctx["s20"].schedule_constructive(
        ctx["wt"], ctx["rt"], ctx["s10"], ctx["params"], tuple(pol))
    q = ctx["s41"].qos_metrics(ctx, sched)
    assert 0.0 <= q["completion_rate"] <= 1.0
    assert 0.0 <= q["margin_mean"] <= 1.0


def test_price_reconstruct_keeps_mean(ctx):
    """峰谷比重构保均值（均值漂移 <1e-6），放大峰谷差。"""
    rt = ctx["rt"]
    p = rt["ElectricityPrice_CNY_per_MWh"].to_numpy()
    mu = rt.groupby("PricePeriod")["ElectricityPrice_CNY_per_MWh"].transform(
        "mean").to_numpy()
    for k in (1.5, 2.0):
        pk = mu + k * (p - mu)
        assert abs(pk.mean() - p.mean()) / p.mean() < 1e-6
        assert pk.max() - pk.min() > p.max() - p.min()


# ---------- 求解层 ----------

def _small_d(ctx, T=48):
    d = ctx["s33"]._load_region_data("RegionD")
    d["D"] = d["D"][:T]
    d["W"] = d["W"][:T]
    d["price"] = d["price"][:T]
    d["sellp"] = d["sellp"][:T]
    d["carbon"] = d["carbon"][:T]
    d["c_h"] = np.asarray(ctx["consume"]["RegionD"], dtype=float)
    return d


def test_lower_lp_constraint_residuals(ctx, s40):
    """LP 约束残差：功率平衡 / SOC 递推 / 弃电下界 <1e-6（I2 恒等式守卫）。"""
    d = _small_d(ctx)
    ch, dh = ctx["tpl"]["RegionD"]["charge_hours"], \
        ctx["tpl"]["RegionD"]["discharge_hours"]
    m = s40.solve_lower_constrained(d, ch, dh, T=48)
    assert m["cost_wan"] is not None, m["message"]
    rows = m["rows"]
    T = len(rows)
    for t in range(T):
        rec = rows[t]
        soc_prev = d["init_soc"] if t == 0 else rows[t - 1]["SOC"]
        bal = (rec["G"] + d["W"][t] + rec["Pd"] - d["D"][t]
               - rec["Pc"] - rec["S"] - rec["Q"])
        soc = soc_prev + d["eta_c"] * rec["Pc"] - rec["Pd"] / d["eta_d"]
        assert abs(bal) < 1e-6, f"t={t} 功率平衡残差 {bal}"
        assert abs(rec["SOC"] - soc) < 1e-6, f"t={t} SOC 递推残差"
        c_h = min(d["W"][t], d["c_h"][t % 24] * d["D"][t])
        assert rec["Q"] >= d["W"][t] - c_h - rec["Pc"] - rec["S"] - 1e-6, \
            f"t={t} 弃电下界违例"


def test_mutex_no_simultaneous(ctx):
    """M0x 充放互斥：任何时刻不同时充放（传送带假象红线）。"""
    s37 = load_step("step3.7+_q3_mutex")
    d = _small_d(ctx)
    m = s37.solve_mutex(d, T=48)
    assert m["cost_wan"] is not None
    both = sum(1 for x in m["rows"] if x["Pc"] > 0.01 and x["Pd"] > 0.01)
    assert both == 0, f"同时充放 {both} 小时（互斥失效）"


def test_whitelist_zero_violation(ctx):
    """白名单零违约：Q4 基线调度所有任务 ω_i ≤ m_i（时延硬约束前置）。"""
    pol = ctx["s41"].q2_compromise_policy(ctx)
    sched = ctx["s20"].schedule_constructive(
        ctx["wt"], ctx["rt"], ctx["s10"], ctx["params"], tuple(pol))
    lm, src_idx, dst_idx = ctx["s20"]._latency_matrix()
    m = ctx["wt"].merge(sched, on="TaskID")
    MAX_LATENCY = {"RealTimeInference": 20, "BatchInference": 80,
                   "AITraining": 150}
    REG = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
    for rec in m.itertuples(index=False):
        src = REG.index(rec.SourceRegion)
        dst = REG.index(rec.Region)
        assert lm[src_idx[src], dst_idx[dst]] <= MAX_LATENCY[rec.TaskType]


def test_soc_terminal_constraint(ctx, s40):
    """SOC(2406) ≥ InitialSOC（说明 5）。"""
    d = _small_d(ctx)
    ch, dh = ctx["tpl"]["RegionD"]["charge_hours"], \
        ctx["tpl"]["RegionD"]["discharge_hours"]
    m = s40.solve_lower_constrained(d, ch, dh, T=48)
    assert m["rows"][-1]["SOC"] >= d["init_soc"] - 1e-6


def test_monotonicity_sell(ctx, s40):
    """单调性守卫：约束 RHS 收紧 → 成本单调不减（ε-约束验收三件套）。

    载体=SellLimit 收紧（step4.2 实证：碳限额在可行域下界，任何收紧 infeasible
    ——消纳模板锁定 G≥D−cap_h；故用始终可行且单调的卖电上限作验收）。
    """
    d = _small_d(ctx, T=168)
    ch, dh = ctx["tpl"]["RegionD"]["charge_hours"], \
        ctx["tpl"]["RegionD"]["discharge_hours"]
    costs = []
    for sell_f in (1.0, 0.9, 0.8):
        d2 = dict(d)
        d2["sell_lim"] = d["sell_lim"] * sell_f
        m = s40.solve_lower_constrained(d2, ch, dh, T=168)
        assert m["cost_wan"] is not None, f"sell×{sell_f} 不可行"
        costs.append(m["cost_wan"])
    assert costs[0] <= costs[1] + 1e-6 <= costs[2] + 1e-6, costs


def test_reproducibility(ctx):
    """可复现性：同一策略两次评估逐位一致（#42 纪律）。"""
    pol = ctx["s41"].q2_compromise_policy(ctx)
    sched = ctx["s20"].schedule_constructive(
        ctx["wt"], ctx["rt"], ctx["s10"], ctx["params"], tuple(pol))
    e1 = ctx["s41"].evaluate_q4_six(ctx, sched)
    e2 = ctx["s41"].evaluate_q4_six(ctx, sched)
    assert np.array_equal(e1["obj"], e2["obj"])
