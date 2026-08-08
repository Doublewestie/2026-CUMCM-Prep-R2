"""B1/B5 守卫：部署口径类选 v3 + E1 v2 干净对照."""
import json

import pandas as pd

from step0_config import OUTPUT

OUT_F = OUTPUT / "forecast"
OUT_R = OUTPUT / "robust"


def test_deploy_gate_products():
    d = pd.read_csv(OUT_F / "deploy_arena.csv")
    assert {"series", "model", "frozen_mape", "frozen_imp", "deploy_pass"} \
        <= set(d.columns)
    r = json.loads((OUT_F / "deploy_gate.json").read_text(encoding="utf-8"))
    assert r["cv_deploy_mismatch_share"] > 0.5      # 错位实锤
    assert r["n_series_with_deploy_best"] >= 20


def test_deploy_best_tree_dominant():
    """B1：部署口径下树类统治（能源侧），tabpfn 不再居首。"""
    r = json.loads((OUT_F / "deploy_gate.json").read_text(encoding="utf-8"))
    per = [x for x in r["per_series"] if x["series"].startswith("energy")]
    dep = [x["deploy_best"] for x in per if x["deploy_best"]]
    tree = sum(1 for m in dep if m.startswith("lgbm") or m.startswith("xgb")
               or m.startswith("gbm") or m.startswith("qrf"))
    tab = sum(1 for m in dep if m == "tabpfn")
    assert tree > tab * 3, f"树类 {tree} 应显著多于 tabpfn {tab}"


def test_e1_v2_clean_measurement():
    """B5：干净框架下预测精度边际价值≈0（预留代价 <1pp）。"""
    e = json.loads((OUT_R / "e1_v2.json").read_text(encoding="utf-8"))
    assert e["decomposition"]["spatial_arbitrage_pp"] > 3.0   # 套利真实
    assert e["decomposition"]["reserve_cost_pp"] < 1.0        # 预留代价小
    assert e["gap_pp"] < 5
    assert e["conclusion"].startswith("预测精度边际价值≈0")
