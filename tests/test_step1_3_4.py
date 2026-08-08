"""step1.3/1.4 回归：η 档案 / oracle 上界 / 敏感性 / 融合范式 / 基函数 / 画像."""
import json
from pathlib import Path

import numpy as np
import pytest

OUT_R = "output/robust"


def load(name):
    return json.loads((Path(OUT_R) / name).read_text(encoding="utf-8"))


def test_eta_profile_two_layer():
    r = load("rigor_pack.json")["a1_eta"]
    assert r["task"]["eta_mean"] < 0.01, "任务侧 η 应≈0"
    assert r["energy"]["eta_mean"] > 0.9, "能源侧 η 应≈1"
    assert r["energy"]["by_type"]["renewable"] == 1.0


def test_arrival_dispersion_poisson():
    r = load("rigor_pack.json")["a2_arrival_gof"]["per_series"]
    ds = [v["dispersion"] for v in r.values()]
    assert np.mean(ds) < 1.3, "dispersion 应接近 1（泊松主结构）"


def test_oracle_upper_bound_range():
    r = load("rigor_pack.json")["a3_oracle"]
    assert 0.0 < r["C4_oracle_price_imp_pct"] < r["C2_migrate_imp_pct"] + 0.5


def test_threshold_robust():
    r = load("rigor_pack.json")["a4_threshold"]
    assert r["energy_gate_5pct"] >= r["energy_gate_7pct"]
    assert r["e1_criteria_3pp"] and r["e1_criteria_8pp"]


def test_consume_sensitivity_stable():
    r = load("rigor_pack.json")["a5_consume"]
    gaps = [v["gap_pp"] for v in r.values()]
    assert max(gaps) < 0.5, f"E1 结论对消纳模板 ±10% 扰动应仍远小于 5pp 判据"


def test_gate_v2_recorded():
    r = load("rigor_pack.json")["a6_gate_v2"]
    assert r["n_changed"] > 0


def test_b1_oracle_below_best():
    r = load("fusion_research.json")["b1"]
    for b in r:
        assert b["oracle_routing_mape"] <= b["best_single_mape"] + 1e-6


def test_b1_drift_detected():
    r = load("fusion_research.json")["b1"][0]
    assert len(set(r["drift_level"].values())) >= 2, "需求水平维度应有漂移"


def test_b2_domain_basis_nonsignificant():
    r = load("fusion_research.json")["b2"]["basis"]
    assert all(v["sig_folds"] == 0 for v in r.values())


def test_b3_deep_hetero_low_corr():
    r = load("fusion_research.json")["b3"]["energy|RegionA|price"]["pairs"]
    assert r["lgbm_point|deep_tcn"] < 0.3, "深度与树类应低相关（异构互补）"


def test_b4_rf_global_no_explosion():
    r = load("fusion_research.json")["b4"]["configs"]
    assert all(v < 20 for v in r.values()), "RF global 不应量纲爆炸"


def test_b5_tabpfn_best_median():
    r = load("fusion_research.json")["b5"]["profile_table"]
    for typ in ("price", "carbon", "nonai", "renewable"):
        fam_map = r[typ]
        best = min(fam_map.values())
        assert abs(fam_map["基础模型"] - best) < 1e-6, f"{typ} TabPFN 应最优"
