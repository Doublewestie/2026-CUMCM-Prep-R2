"""test_q3_mechanism — Q3 反思 P2 机理五件套守卫（step3.9）.

弃电边际价值: sellp 缩放成本单调（外送饱和下只改收入）；充电量 Cap 锁死
SOC 裕度: 贴 Cap 占比 >30%（Cap 活跃约束分布证据，E5 三证）
物理一致性: 充电双通道残差 ≈0；SellPrice>0 → GridSell>0 占比 >99%
CVaR: F 区 64/64 场景 infeasible 如实报告（购电上限尾部瓶颈）
循环寿命: M3 口径等效循环 ~90/100 天（与 sum_10 一致）
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mech():
    p = ROOT / "output" / "q3" / "q3_mechanism.json"
    if not p.exists():
        pytest.skip("q3_mechanism.json 缺失——先跑 step3.9_q3_mechanism.py")
    return json.loads(p.read_text(encoding="utf-8"))


def test_sellp_curve_cost_monotone(mech):
    """弃电边际价值：D 区 sellp 缩放成本单调递减（外送饱和只改收入）。"""
    cur = mech["regions"]["RegionD"]["sellp_curve"]
    costs = [c["cost_wan"] for c in cur]
    assert costs == sorted(costs, reverse=True), "成本未随 sellp 单调降"
    # 充电量 Cap 锁死（波动 <0.5%）
    charges = [c["charge_MWh"] for c in cur]
    assert max(charges) - min(charges) < 0.01 * max(charges), "充电量异常波动"
    # 外送饱和（≈SellLimit×2407）
    assert cur[0]["sell_MWh"] / 2407 > 170, "D 区外送未饱和"


def test_soc_touches_cap(mech):
    """SOC 贴 Cap >30%——Cap 活跃约束（E5 分布证据）。"""
    for r in ("RegionA", "RegionD"):
        sm = mech["regions"][r]["soc_margin"]
        assert sm["touch_cap_pct"] > 30, f"{r} 贴 Cap 占比过低"
        assert sm["dist_to_cap_pct"]["p50"] < 5, f"{r} 中位距 Cap 过大"


def test_physical_consistency(mech):
    """数据列自洽：充电双通道残差≈0；SellPrice>0 → GridSell>0 占比>99%。"""
    for r in ("RegionA", "RegionD", "RegionF"):
        pc = mech["regions"][r]["physical_consistency"]
        assert abs(pc["charge_dual_channel_resid"]["max_rel"]) < 1e-6, \
            f"{r} 充电双通道不自洽"
    f_ = mech["regions"]["RegionF"]["physical_consistency"]
    assert f_["sell_positive_when_sellprice"] > 0.99, "F 区外送占比异常"


def test_cvar_e_f_honest(mech):
    """CVaR 如实：F 区 64/64 场景 infeasible（购电上限尾部瓶颈）——如实标记。"""
    f_ = mech["regions"]["RegionF"]["cvar"]
    assert f_["n_infeasible"] == 64 and f_["n_scenarios"] == 0, \
        "F 区 CVaR 应如实报告全 infeasible"
    d_ = mech["regions"]["RegionD"]["cvar"]
    assert d_["n_infeasible"] == 0 and d_["cvar90_wan"] is not None, \
        "D 区 CVaR 应正常计算"


def test_cycle_life_scale(mech):
    """循环寿命：M3 口径等效循环 ~90/100 天（sum_10 一致）。"""
    d_ = mech["regions"]["RegionD"]["cycle_life"]
    assert 60 < d_["equivalent_cycles"] < 120, \
        f"等效循环异常: {d_['equivalent_cycles']}"
