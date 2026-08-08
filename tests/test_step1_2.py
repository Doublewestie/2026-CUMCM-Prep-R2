"""step1.2 系列回归：κ_ε 校准 / E1 判据 / 实验群产物完整性."""
import json

import numpy as np
import pandas as pd
import pytest

from step0_config import HOURS_TOTAL, REGIONS

OUT_R = "output/robust"


def load(name):
    return json.loads((__import__("pathlib").Path(OUT_R) / name)
                      .read_text(encoding="utf-8"))


def test_kappa_selection_in_passed_set():
    kf = load("kappa_fit.json")
    assert kf["eps_selected"] in [float(e) for e in kf["passed_set"]]
    assert kf["passed_set"]


def test_kappa_coverage_monotonic():
    kf = load("kappa_fit.json")
    covs = [kf["calibration"][e]["cov"] for e in sorted(kf["calibration"])]
    assert all(covs[i] >= covs[i + 1] - 1e-9 for i in range(len(covs) - 1))


def test_kappa_pseudo_consistent():
    kf = load("kappa_fit.json")
    e_sel = f"{kf['eps_selected']:.2f}"
    pseudo = kf["pseudo_calibration"][e_sel]["mean"]
    real = kf["calibration"][e_sel]["cov"]
    assert abs(pseudo - real) < 0.06, f"选中 ε={e_sel} 伪/真校准偏差过大"


def test_kappa_calibration_segment_fixed():
    """R0 修正守卫：校准段必须为 2352-2375（原实现误用冻结段 2376-2399）。

    修正后 ε=0.10 伪校准(0.939) 与真校准(0.938) 偏差 <1pp —— "漂移"假象解除。
    """
    kf = load("kappa_fit.json")
    assert kf.get("calibration_fix"), "校准段修正记录缺失"
    drift = abs(kf["pseudo_calibration"]["0.10"]["mean"]
                - kf["calibration"]["0.10"]["cov"])
    assert drift < 0.01, f"修正后伪/真校准仍偏差 {drift:.3f}（应≈0.2pp）"


def test_e1_criterion_holds():
    e1 = load("e1_three_way.json")
    assert e1["gap_pp"] < 5, "E1 判据不成立（>5pp）"
    assert e1["conclusion"] == "结构套利主导"
    assert e1["viol_h"]["local"] > e1["viol_h"]["perfect"]


def test_kappa_bounds():
    kf = load("kappa_fit.json")
    for v in kf["kappa_means"].values():
        assert 0.0 <= v <= 0.95


def test_quantile_schedule_shape():
    sc = pd.read_csv(f"{OUT_R}/quantile_schedule.csv")
    assert set(sc.columns) == {"TaskID", "Region", "StartHour"}
    assert len(sc) == 50000


def test_region_quantiles_shape():
    rq = pd.read_csv(f"{OUT_R}/region_quantiles.csv")
    assert len(rq) == HOURS_TOTAL


def test_e1a_migration_beats_shift():
    e = load("e1_mechanism.json")
    assert e["e1a"]["migrate_only_cost_wan"] < e["e1a"]["shift_only_cost_wan"]


def test_e1d_curve_saturation():
    e = load("e1_mechanism.json")
    d = {r["sigma"]: r for r in e["e1d"]}
    cost_drift = (d[1.0]["cost_wan"] - d[0.0]["cost_wan"]) / d[0.0]["cost_wan"]
    assert cost_drift < 0.01, f"E1d 成本漂移 {cost_drift:.4f} 未饱和"


def test_fusion_ablation_no_nan():
    fa = load("fusion_ablation.json")
    assert not np.isnan(fa["e2b"]["width_inverse"])
    assert fa["e2b"]["rf_metalearner"] < fa["e2b"]["equal_weight"]


def test_priceperiod_ablation_value():
    fa = load("fusion_ablation.json")
    assert fa["ablation"]["with_priceperiod"] < fa["ablation"]["without_priceperiod"]


def test_hurdle_no_gain():
    fa = load("fusion_ablation.json")
    assert fa["hurdle"]["hurdle_cov"] < fa["hurdle"]["baseline_cov"]


def test_e2e_tabpfn_dominates_structure():
    fa = load("fusion_ablation.json")
    wins = fa["e2e"]["wins"]
    struct = [w for w in wins if w["bucket"] == "结构类"]
    assert len(struct) == 18
    assert all(w["winner"] == "tabpfn" for w in struct)
