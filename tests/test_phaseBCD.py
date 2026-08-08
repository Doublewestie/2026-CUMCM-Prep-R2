"""Phase B/C/D 守卫：类选v4/充电口径/滚动κ/素材/Q3铺路."""
import json

import numpy as np
import pandas as pd

from conftest import load_step_module
from step0_config import CLEAN, OUTPUT

OUT_R = OUTPUT / "robust"
OUT_Q2 = OUTPUT / "q2"
OUT_Q3 = OUTPUT / "q3"
OUT_F = OUTPUT / "forecast"

s10 = load_step_module("step1.0_baseline_schedule.py")


def test_b7_v4_no_task_false_positive():
    """类选 v4：任务侧部署入选 0（白噪声假阳性消除）+ 集合级 Jaccard。"""
    r = json.loads((OUT_F / "deploy_gate.json").read_text(encoding="utf-8"))
    assert r["v4"]["task_deploy_pass"] == 0
    assert r["v4"]["jaccard_set_consistency"]["energy"] >= 0.7
    assert len(r["v4"]["cv_noisy_sequences"]) >= 5


def test_b9_charge_caliber():
    """充电口径：含充电后弃电偏差 ≈ 外送口径差（−11.4 万，非充电项）。"""
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    local = pd.read_csv(OUTPUT / "baseline" / "local_schedule.csv")
    m, _ = s10.evaluate_schedule(local, wt, rt, params, consume,
                                 include_baseline_charge=True)
    diff = m["curtail_MWh"] - rt["Curtailment_MW"].sum()
    assert abs(diff - (-116111)) < 15000, f"充电项后弃电偏差 {diff} 应≈外送口径差"


def test_b10_rolling_kappa():
    """滚动 κ：冻结段覆盖率 >0.94（F2 闭环）。"""
    r = json.loads((OUT_R / "rolling_kappa.json").read_text(encoding="utf-8"))
    assert r["verdict"]["meets_target"] is True
    assert r["rolling_calibration"]["a=0.95"]["frozen_cov"] > 0.94


def test_c_materials():
    """Phase C 素材落盘。"""
    r = json.loads((OUT_Q2 / "materials.json").read_text(encoding="utf-8"))
    assert r["c6_marginal"]["median_exchange_t_per_wan"] > 0
    assert len(r["c7_pruning"]) >= 5
    assert "c8_sobol" in r


def test_d_q3_exploration():
    """Q3 探路：六区域 LP 全量 + 场景框架 + 窗口效应。"""
    allr = json.loads((OUT_Q3 / "lp_all_regions.json").read_text(encoding="utf-8"))
    assert all(allr[r]["status"] == 0 for r in allr)
    assert allr["RegionE"]["nu"] > 0.7          # 西区弃电充电大幅提升利用率
    scen = json.loads((OUT_Q3 / "scenario_report.json").read_text(encoding="utf-8"))
    assert scen["scenarios"]["RegionE"]["quality_rel_err"] < 0.05
    assert scen["window_effect"]["RegionE"]["window_effect_pct"] < 10
