"""test_phase4 — 阶段 1-3 修正与发现守卫（证伪批次 + 三遗留研究）.

口径修正: 双锚 LP 对照（D −38.3%）；波动 64 场景（infeasible_pct 字段）；
         c_h 扰动非对称（−10% E/F infeasible / +10% 降本）
证伪批次: oracle 迁移轨迹单调恶化（非过冲）；E5 顶点稳定；碳构造上界 <2%；
         对偶退化普遍（扩展验证）
三遗留:   #17 MPC 结构性保守（确定性 sanity 5.4%）；#5 全网峰值伴随口径；
         #18 Sobol 交互（非单调/交互字段）
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_json(rel):
    p = ROOT / "output" / rel
    if not p.exists():
        pytest.skip(f"{rel} 缺失——先跑对应 step")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- 口径修正 ----------

def test_double_anchor_lp_purity():
    """1-1 双锚 LP 对照：D 区储能价值 −38%（LP 自由度 1.43 亿被分离）。"""
    r = load_json("q3/q3_rigor.json")
    d = r["regions"]["RegionD"]["decomposition"]
    assert -45 < d["storage_value_pct"] < -30
    assert d["lp_freedom_value_wan"] < -10000
    assert r["regions"]["RegionD"]["no_storage_lp"]["cost_wan"] < \
        r["regions"]["RegionD"]["no_storage_template"]["cost_wan"]


def test_volatility_64_scenarios():
    """1-2 波动 64 原始场景：infeasible_pct 随 σ 上升（尾部瓶颈量化）。"""
    p = load_json("q4/q4_pressure.json")
    pcts = []
    for s in ("0.1", "0.2", "0.3"):
        cl = p["grid"]["volatility"][s]["q4_compromise"]
        ip = cl.get("infeasible_pct")
        assert ip is not None and 0 <= ip <= 100, f"σ={s} infeasible_pct 异常"
        pcts.append(ip)
    assert pcts[0] < pcts[1] < pcts[2], "infeasible_pct 应随 σ 单调上升"
    # 波动成本应显著高于确定性（截断修正后）
    base = p["grid"]["price"]["1.2"]["q4_compromise"]["cost_wan"]
    vol = p["grid"]["volatility"]["0.3"]["q4_compromise"]["cost_wan"]
    assert vol > base * 1.01, "64 场景波动成本应 >1%（K-means 截断已修正）"


def test_ch_sensitivity_asymmetric():
    """1-3 c_h 扰动非对称：−10% E/F infeasible；+10% 降本 >4%。"""
    r = load_json("q4/q4_ch_sensitivity.json")
    scan = r["scan"]
    assert "RegionE" in scan["0.9"]["infeasible_regions"], "c_h×0.9 E 区应 infeasible"
    assert "RegionF" in scan["0.9"]["infeasible_regions"], "c_h×0.9 F 区应 infeasible"
    assert scan["1.1"]["d_cost_vs_base_pct"] < -4.0, "c_h×1.1 应显著降本"


# ---------- 证伪批次 ----------

def test_oracle_trajectory_monotone_worse():
    """2-2 oracle 迁移轨迹：成本单调恶化（非过冲——方向性错误）。"""
    c = load_json("q4/q4_collapse.json")
    traj = c["oracle"].get("trajectory")
    assert traj and len(traj) >= 5, "轨迹缺失"
    costs = [t["cost_wan"] for t in traj]
    assert costs == sorted(costs), "迁移轨迹应单调恶化（从第一个迁移起）"
    # 第一个非零迁移点即恶化 >1%
    first = next(t for t in traj if t["n_moved"] > 0)
    assert first["cost_wan"] > traj[0]["cost_wan"] * 1.01, "首个迁移即应恶化"


def test_e5_vertex_stable():
    """2-3 E5 饱和率顶点稳定（扰动重解逐位一致——约束绑定结构）。"""
    r = load_json("q3/q3_e5_vertex_robust.json")
    for region in ("RegionD", "RegionE"):
        v = r[region]
        assert v["stable_sell"] and v["stable_soc"]
        assert v["base"]["sell_sat_pct"] > 95, f"{region} SellLimit 未饱和"


def test_carbon_constructive_upper_bound():
    """2-4 碳排构造上界 <2%（AI 迁 E/F 扫描——结构性结论）。"""
    r = load_json("q3/q3_rigor.json")  # 占位——实际用 q4 压力 lever_floor
    p = load_json("q4/q4_pressure.json")
    floor = p.get("carbon_lever_floor", {})
    assert floor.get("max_lever_pct") is not None
    assert 0 < floor["max_lever_pct"] < 2.0, "6 旋钮杠杆 <2%（构造上界 1.4%）"


def test_dual_degradation_extended():
    """2-5 对偶退化扩展验证：随机 10 小时 ≥7/10 不一致。"""
    r = load_json("q4/q4_dual_extended.json")
    assert r["n_inconsistent_gt5pct"] >= 7, "退化应普遍"


# ---------- 三遗留研究 ----------

def test_mpc_structural_conservative():
    """#17 MPC：结构性保守（反馈负值）；确定性 sanity 仅 5.4% 截断。"""
    r = load_json("q4/q4_mpc.json")
    assert r["n_fallback"] == 0, "窗口联合 LP 应全成功"
    assert "结构性保守" in r["verdict"], "MPC 结论应标注结构性保守"
    # 确定性 sanity：MPC ≈ 全知 + 5.4%（框架正确）
    sanity = r["mpc_deterministic_sanity_wan"]
    full = r["full_knowledge_wan"]
    assert abs(sanity / full - 1) < 0.1, "确定性 MPC 应接近全知（框架 sanity）"
    # 波动 MPC 显著劣于 OL（保守交集）
    assert r["mpc_wan"] > r["open_loop_expected_wan"]


def test_system_peak_structural():
    """#5 全网峰值：削峰约束 infeasible（储能层降峰空间=0 全网版）。"""
    r = load_json("q4/q4_system_peak.json")
    ind = r["independent"]
    assert ind["sys_peak_main_MW"] > ind["region_peak_max_MW"], "全网>区域最大"
    assert 0 < ind["simultaneity_normalized"] < 1, "同时性归一化 ∈(0,1)"
    assert all(s["infeasible"] for s in r["joint_sweep"]), "削峰 5% 应 infeasible"


def test_sobol_interaction_mechanism():
    """#18 Sobol 交互：S1>1 的机理=非单调或交互（字段完整）。"""
    r = load_json("q3/q3_sobol_interaction.json")
    slices = r["slices"]
    assert len(slices) == 3
    has_nonmono = any(s["nonmonotonic_a"] or s["nonmonotonic_b"]
                      for s in slices)
    assert has_nonmono, "应有非单调切片（S1>1 机理）"
    # S1 vs ST 对照存在且 E|carbon S1>1 被解释
    assert "RegionE|carbon_t" in r["S1_vs_ST"]
    assert r["S1_vs_ST"]["RegionE|carbon_t"]["unreliable"] is True
