"""test_q4_v2 — Q4 定稿守卫（T0-T5 修复与发现固化）.

口径层:  α 固定 0.5（评估器参数不可优化——α=1 目标游戏化，总账 +2）
求解层:  波动场景 peak 双口径（期望/最坏，基线同量级）；碳杠杆三族字段
         完整；lever_floor 结构性结论（任务层碳杠杆 <1%）
工具层:  对偶验证门（dual_verify 存在——HiGHS marginals 退化裁决，总账 +7）
验收层:  正式预算产物（收敛曲线长度/种子方差 accept）；前沿坍缩判别产物
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


def load_json(rel):
    return json.loads((ROOT / "output" / "q4" / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ctx():
    s41 = load_step("step4.1_q4_indicators")
    return {**s41.load_ctx(), "s41": s41}


# ---------- 口径层 ----------

def test_alpha_fixed_half(ctx):
    """α 维固定 0.5：评估器参数不可被优化（α=1 使 1−QOS≡0 目标游戏化）。"""
    s40 = load_step("step4.0_q4_bilevel")
    assert s40.LB[5] == 0.5 and s40.UB[5] == 0.5
    assert s40.CONSTRUCT_SEED[5] == 0.5


def test_bilevel_formal_budget_and_convergence(ctx):
    """正式预算产物：n_pop/n_gen/n_seed ≥ 40/30/3；收敛曲线长度 = n_gen+1；
    种子方差 accept（rel_std < 0.1%）。"""
    r = load_json("q4_bilevel.json")
    assert r["n_pop"] >= 40 and r["n_gen"] >= 30 and r["n_seed"] >= 3, \
        "正式预算未执行（smoke 产物残留）"
    hv = r["convergence"]["hv_median"]
    assert len(hv) == r["n_gen"] + 1, "收敛曲线长度与代数不符"
    assert r["seed_variance"]["accept"] is True, "种子方差超限"
    assert r["convergence"]["verdict"].startswith("收敛"), \
        f"收敛判定异常: {r['convergence']['verdict']}"
    # 折中解 QOS 不得为游戏化 0（α 修复验证）
    assert r["compromise"]["one_minus_qos"] > 0.05, "1−QOS=0 系 α 游戏化残留"


# ---------- 求解层 ----------

def test_volatility_peak_dual_caliber(ctx):
    """波动场景 peak 双口径：期望/最坏峰值存在且与基线同量级（478.5±30%）。"""
    p = load_json("q4_pressure.json")
    baseline_peak = load_json("q4_indicators.json")["baseline"]["peak_net_MW"]
    for sigma in ("0.1", "0.2", "0.3"):
        for pol in ("construct", "q2_compromise", "q4_compromise"):
            cl = p["grid"]["volatility"][sigma][pol]
            assert not cl.get("infeasible"), f"vol {sigma}/{pol} infeasible"
            pe = cl.get("peak_expect_MW")
            pw = cl.get("peak_worst_MW")
            assert pe is not None and pe > 0, f"{sigma}/{pol} 期望峰值缺失"
            assert pw is not None and pw >= pe, f"{sigma}/{pol} 最坏<期望"
            assert 0.7 * baseline_peak <= pe <= 1.3 * baseline_peak, \
                f"{sigma}/{pol} 期望峰值异常 {pe}"


def test_carbon_lever_variants_and_floor(ctx):
    """碳杠杆三族字段完整 + lever_floor 结构性结论（任务层碳杠杆 <1%）。"""
    p = load_json("q4_pressure.json")
    for tau in ("0.1", "0.2", "0.3"):
        cl = p["grid"]["carbon"][tau]["q4_compromise"]
        assert cl["lever_needed"] is True
        assert cl["lever_ok"] is False, f"τ={tau} 杠杆意外成功"
        assert cl["lever_variants"] is not None, f"τ={tau} 杠杆族字段缺失"
        for name in ("A_shift24", "B_migrate", "C_joint"):
            assert name in cl["lever_variants"], f"τ={tau} 缺杠杆族 {name}"
    floor = p.get("carbon_lever_floor")
    assert floor is not None and floor.get("max_lever_pct") is not None, \
        "carbon_lever_floor 缺失"
    assert 0.0 < floor["max_lever_pct"] < 1.0, \
        f"任务层碳杠杆异常: {floor['max_lever_pct']}%（应 <1%，NonAI 稀释）"


def test_shadow_dual_verify_present(ctx):
    """对偶验证门产物存在（T5：HiGHS marginals 退化裁决）。"""
    s = load_json("q4_shadow.json")
    assert "dual_verify" in s, "dual_verify 缺失"
    assert "degradation" in s["dual_verify"], "退化诊断缺失"
    assert "回退数值差分" in s["dual_verify"]["verdict"], \
        f"对偶裁决异常: {s['dual_verify']['verdict']}"
    # π_sell 唯一可行载体：SellLimit 收紧成本单调
    costs = [r["total_cost_wan"] for r in s["pi_sell"]["rows"]]
    assert costs == sorted(costs), "π_sell 成本非单调"


def test_collapse_discriminant(ctx):
    """前沿坍缩判别产物：三证据裁决落盘（储能结构锁定/吸收）。"""
    c = load_json("q4_collapse.json")
    assert "discriminant_A" in c and "discriminant_B" in c
    assert c["verdict"] and c["verdict"].startswith("【实证】")
    assert c["discriminant_A"]["cost_span_pct"] < 1.0, \
        "极端策略跨度异常大（搜索不足未排除）"


# ---------- 单调性/物理合理性 ----------

def test_sell_monotonicity_ablation(ctx):
    """消融单调性：SellLimit 收紧成本单调不减（ε-约束验收三件套）。"""
    a = load_json("q4_ablation.json")
    assert a["monotonicity"]["monotone"] is True
    costs = a["monotonicity"]["costs_wan"]
    assert costs == sorted(costs)


def test_ablation_component_order(ctx):
    """组件消融排序：无储能 > 无迁移 > 无错峰 ≈ 基线（储能 2 亿/迁移 1.2 亿/错峰≈0）。"""
    a = load_json("q4_ablation.json")
    comp = a["component_ablation"]
    assert comp["no_storage"]["cost_wan"] > comp["no_migration"]["cost_wan"] \
        > comp["no_shift"]["cost_wan"]
    gap = (comp["no_storage"]["cost_wan"] - comp["no_shift"]["cost_wan"]) / 1e4
    assert 1.5 < gap < 2.5, f"储能贡献 {gap} 亿（预期 ~2 亿）"


def test_rules_q4_compromise_cheap(ctx):
    """规则层：q4 折中规则化代价近乎无损（|gap| < 0.5%）。"""
    r = load_json("q4_rules.json")
    q4 = r["rules"]["q4_compromise"]
    assert q4["rule_infeasible"] is False
    assert q4["rule_cost_gap_pct"] is not None
    assert abs(q4["rule_cost_gap_pct"]) < 0.5, \
        f"规则化代价异常: {q4['rule_cost_gap_pct']}%"
    assert r["rules"]["q2_compromise"]["rule_infeasible"] is True, \
        "q2 规则模拟应如实标记 infeasible"


def test_reproducibility_indicators(ctx):
    """可复现性（#42 纪律）：同一基线策略两次评估逐位一致。"""
    pol = ctx["s41"].q2_compromise_policy(ctx)
    sched = ctx["s20"].schedule_constructive(
        ctx["wt"], ctx["rt"], ctx["s10"], ctx["params"], tuple(pol))
    e1 = ctx["s41"].evaluate_q4_six(ctx, sched)
    e2 = ctx["s41"].evaluate_q4_six(ctx, sched)
    assert np.array_equal(e1["obj"], e2["obj"])
