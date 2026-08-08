"""step1.0 核心回归：评估器锚点 / 贪心可行化 / 乱序对齐（防错位复发）."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import load_step_module
from step0_config import CLEAN, REGIONS

s10 = load_step_module("step1.0_baseline_schedule.py")


@pytest.fixture(scope="module")
def metrics():
    return json.loads(
        (s10.OUT_BASE / "baseline_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    return wt, rt


def test_ai_load_sum_anchor(metrics):
    a = metrics["anchor"]
    assert a["ai_load_sum_rel_err"] < 1e-6
    assert a["ai_load_sum_ours"] == pytest.approx(a["ai_load_sum_baseline"], rel=1e-6)


def test_nu_anchor(metrics):
    nu = metrics["local"]["nu_pct"]
    assert 25.0 < nu < 40.0, f"nu_pct={nu} 偏离题目基线 32.9%"


def test_curtail_anchor(metrics):
    c = metrics["local"]["curtail_MWh"]
    assert 700e4 < c < 900e4, f"curtail={c} 偏离题目基线 775.5 万 MWh"


def test_viol_anchor(metrics):
    v = metrics["local"]["viol_by_region"]
    assert v["RegionE"] == 30
    assert v["RegionF"] == 67


def test_greedy_no_violation(metrics):
    assert metrics["greedy"]["viol_h"] == 0
    assert metrics["greedy"]["fail_tasks"] == 0


def test_rt_fixed_at_arrival(data):
    wt, _ = data
    gd = pd.read_csv(s10.OUT_BASE / "greedy_schedule.csv")
    rt = wt[wt.TaskType == "RealTimeInference"].merge(gd, on="TaskID")
    assert (rt["StartHour"] == rt["ArrivalHour"]).all()


def test_greedy_lower_cost_than_local(metrics):
    assert metrics["greedy"]["cost_wan"] <= metrics["local"]["cost_wan"] * 1.01


def test_consume_ratio_closure(metrics):
    assert metrics["consume_fit"]["closure_minWD_hit"] == 1.0


def test_consume_ratio_template(metrics):
    """R1 升级：c_r 为日内模板（24 值），构造性命中率 1.0（形式修正实证）。"""
    c = metrics["consume_fit"]["consume_ratio"]
    assert set(c) == set(REGIONS)
    for r in REGIONS:
        v = np.asarray(c[r], dtype=float)
        assert v.shape == (24,), f"{r} 模板应为 24 值"
        assert np.all((0.0 <= v) & (v <= 1.0))
    stats = {s["Region"]: s for s in metrics["consume_fit"]["fit_stats"]}
    for r in REGIONS:
        assert stats[r]["template_hit_rate"] == 1.0
        assert stats[r]["max_std_within_hour"] < 1e-6


def test_occupancy_row_order_independent(data):
    """回归：build_occupancy 必须与调度表行序无关（错位 bug 防护）。"""
    wt, _ = data
    sched = pd.read_csv(s10.OUT_BASE / "local_schedule.csv")
    shuffled = sched.sample(frac=1.0, random_state=42).reset_index(drop=True)
    p1, _ = s10.build_occupancy(
        wt, sched.set_index("TaskID")["Region"], sched.set_index("TaskID")["StartHour"])
    p2, _ = s10.build_occupancy(
        wt, shuffled.set_index("TaskID")["Region"],
        shuffled.set_index("TaskID")["StartHour"])
    assert np.allclose(p1, p2)


def test_power_balance_self_consistent(data):
    """自洽口径：G + U = Total_Load 每小时成立（无储能 Q1）。"""
    wt, rt = data
    lh = pd.read_csv(s10.OUT_BASE / "local_hourly.csv")
    bal = lh["GridPurchase_MW"] + lh["Renewable_Used_MW"] - lh["Total_Load_MW"]
    assert bal.abs().max() < 1e-6
