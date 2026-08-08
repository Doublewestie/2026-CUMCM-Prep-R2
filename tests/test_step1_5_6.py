"""step1.5/1.6 系列回归：冻结段 / 融合 v2 / 反证包 / 生成器指纹."""
import json
from pathlib import Path

import numpy as np
import pytest

OUT_R = Path("output/robust")


def load(name):
    return json.loads((OUT_R / name).read_text(encoding="utf-8"))


def test_frozen_consistency_noise_benchmark():
    f = load("frozen_test.json")
    assert "window_noise_benchmark" in f
    assert f["consistency"]["match_rate"] <= 0.5
    assert "interpretation" in f


def test_frozen_no_catastrophic_failure():
    f = load("frozen_test.json")
    for name, fd in f["per_series"].items():
        for model, sc in fd.items():
            if "mape" in sc and np.isfinite(sc["mape"]):
                assert sc["mape"] < 1e4, f"{name}/{model} 冻结段灾难性失败"


def test_e2g_domain_basis_gain():
    """翻案回归：领域基函数在正确载体上必须显著（B2 OLS 载体错误的修正）。"""
    r = load("fusion_v2.json")["e2g"]
    for name, v in r.items():
        assert v["carrier2_lgbm"]["gain_pct"] > 5, f"{name} 基函数增益缺失"
        assert v["carrier3_fusion"]["gain_pct"] > 0


def test_e2i_hetero_gain():
    r = load("fusion_v2.json")["e2i"]
    carbon = [e for e in r if "carbon" in e["series"]][0]
    assert carbon["hetero_gain"] > 10, "carbon 异构融合应有显著增益"


def test_e2b_rf_dominates_multi():
    r = load("fusion_v2.json")["e2b_multi"]
    assert r["winner_counts"].get("rf", 0) >= 5, "RF 元学习器多序列统治"


def test_e2c_std_no_explosion():
    r = load("fusion_v2.json")["e2c_std"]
    assert all(v < 10 for v in r["results"].values()), "标准化后不应爆炸"


def test_counter_nonlinear_no_gain():
    r = load("counter_evidence.json")["c1_nonlinear"]
    for v in r.values():
        assert abs(v["nonlinear_cov"] - v["base_cov"]) < 0.05


def test_generator_sensitivity_stable():
    r = load("generator_fingerprint.json")["f2"]
    for k, v in r.items():
        if k.startswith("resample"):
            assert v["stat_still_best"], "重采样下统计基线应仍最优"


def test_generator_fingerprints_present():
    r = load("generator_fingerprint.json")["f1"]
    assert r["price_seg_residual"]
    assert r["zero_inflation_task"]
    assert "latency_asymmetry" in r
