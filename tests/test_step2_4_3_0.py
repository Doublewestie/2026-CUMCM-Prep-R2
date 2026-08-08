"""B6/C4 守卫：五方法裁决 / LP 铺路口径."""
import json

import pandas as pd

from step0_config import OUTPUT

OUT_Q2 = OUTPUT / "q2"
OUT_Q3 = OUTPUT / "q3"


def test_method_verdict_cost_endpoint_consensus():
    """五方法成本端点一致（=构造解）——多方法交叉验证。"""
    r = json.loads((OUT_Q2 / "method_verdict.json").read_text(encoding="utf-8"))
    gaps = [v["cost_gap_vs_construct_pct"] for v in r["per_method"].values()]
    assert all(abs(g) < 0.01 for g in gaps), f"成本端点分歧: {gaps}"


def test_method_front_files_exist():
    import os
    for m in ["nsga2", "nsga3", "moead", "alns", "lagrangian"]:
        assert (OUT_Q2 / f"method_front_{m}.csv").exists()


def test_lp_calibration_status():
    """Q3 铺路：LP 全时段求解成功 + 弃电下界口径生效（curtail>0）。"""
    r = json.loads((OUT_Q3 / "lp_calibration.json").read_text(encoding="utf-8"))
    c = r["calibration"]["2407h"]
    assert c["status"] == 0
    assert c["solve_s"] < 60
    assert c["curtail_MWh"] > 1e3, "弃电下界口径应产生可观弃电（利用率不虚高）"


def test_lp_negative_cost_is_arbitrage():
    """负成本 = 弃电充电-外送合法套利（SellPrice > 免费弃电边际成本）。"""
    r = json.loads((OUT_Q3 / "lp_calibration.json").read_text(encoding="utf-8"))
    c = r["calibration"]["2407h"]
    assert c["cost_wan"] < 0, "D 区弃电充电-外送应产生净收益（负成本）"
