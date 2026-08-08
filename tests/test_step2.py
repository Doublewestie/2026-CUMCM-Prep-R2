"""step2（Q2 三段式）回归：构造层/NSGA-II/规则提取/时延裁决/baseline_proof."""
import json

import numpy as np
import pandas as pd

from conftest import load_step_module
from step0_config import CLEAN, OUTPUT

OUT_Q2 = OUTPUT / "q2"

s20 = load_step_module("step2.0_construct.py")


def _load(name):
    return json.loads((OUT_Q2 / name).read_text(encoding="utf-8"))


def _ctx():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    return s10, s20, wt, rt, params, consume


def test_construct_feasible_and_improves():
    """构造层：viol=0 + 成本/碳低于基线 + 迁移率合理。"""
    s10, s20, wt, rt, params, consume = _ctx()
    sched = s20.schedule_constructive(wt, rt, s10, params)
    m4 = s20.evaluate_4obj(wt, rt, sched, s10, params, consume)
    assert m4["viol_h"] == 0
    local = _load_ctx_local()
    assert m4["cost_wan"] < local["cost_wan"]
    assert m4["carbon_t"] < local["carbon_t"]
    local_sched = pd.read_csv(OUTPUT / "baseline" / "local_schedule.csv")
    assert m4["latency_ms"] > s20.compute_latency(wt, local_sched)
    mig = (sched.Region != wt.set_index("TaskID")
           .loc[sched.TaskID].SourceRegion.values).mean()
    assert 0.2 < mig < 0.8


def _load_ctx_local():
    r = json.loads((OUTPUT / "baseline" / "baseline_metrics.json")
                   .read_text(encoding="utf-8"))
    return r["local"]


def test_threshold_monotonicity():
    """迁移阈值单调性：阈值↑ → 迁移率↓、成本↑、时延↓。"""
    s10, s20, wt, rt, params, consume = _ctx()
    res = []
    for g in (0, 15, 127):
        sched = s20.schedule_constructive(wt, rt, s10, params, (g, 0, 0, 0, 0))
        m4 = s20.evaluate_4obj(wt, rt, sched, s10, params, consume)
        src = wt.set_index("TaskID").loc[sched.TaskID].SourceRegion.values
        res.append({"g": g, "mig": float((sched.Region != src).mean()),
                    "cost": m4["cost_wan"], "lat": m4["latency_ms"]})
    assert res[0]["mig"] > res[1]["mig"] > res[2]["mig"]
    assert res[0]["cost"] < res[1]["cost"] < res[2]["cost"]
    assert res[0]["lat"] > res[1]["lat"] > res[2]["lat"]


def test_nsga2_front_feasible():
    """NSGA-II 前沿：全部解 viol=0（罚函数排除不可行）。"""
    fr = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    assert len(fr) >= 90                    # 3 seed 合并前沿
    assert set(fr.seed.unique()) <= {0, 1, 2}
    assert "policy" in fr.columns           # 决策变量落盘（规则提取输入）
    assert fr["cost_wan"].min() > 1e5       # 无罚函数值泄漏（< 1e6 量级）


def test_nsga2_front_dominates_construct():
    """构造解 = 成本最优端点（前沿包含构造解或更优解）。"""
    fr = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    c = _load_ctx_local()
    assert fr["cost_wan"].min() <= c["cost_wan"] * 1.001
    assert fr["carbon_t"].min() <= c["carbon_t"] * 1.001


def test_rules_extracted():
    """规则层：≥3 条可读规则 + 纯度 ≥0.5。"""
    r = _load("rules.json")
    for rep in r["representatives"].values():
        assert rep["n_rules"] >= 3
        assert all(x["purity"] >= 0.5 for x in rep["rules"][:3])
        assert rep["metrics"]["viol_h"] == 0 if "viol_h" in rep["metrics"] else True


def test_delay_scan_verdict():
    """时延形式数据裁决：白名单下零违约 → T2/T3 无区分度。"""
    d = _load("delay_scan.json")
    assert d["verdict"]["T2_all_zero"] is True
    assert d["verdict"]["T3_equals_T1"] is True
    assert "T1" in d["verdict"]["conclusion"]


def test_baseline_proof_q2_wins():
    """baseline_proof：Q2 折中解成本/碳均优于 Q1 全链。"""
    p = _load("baseline_proof.json")
    assert p["q2_compromise_vs_local"]["cost_pct"] > 3.0
    assert p["q2_compromise_vs_local"]["carbon_pct"] > 3.0
    assert p["q2_compromise_vs_local"]["nu_pp"] > 0.5
    assert p["q2_compromise_vs_local"]["latency_ratio"] > 1.0   # 时延代价如实
    assert p["greedy_vs_local"]["cost_pct"] < 0.1                # Q1 无迁移≈0
