"""step1.7 冻结段结构研究守卫：判别实验 / 滚动重估修复 / ε 验证 / 价格机理."""
import json

import numpy as np

from step0_config import OUTPUT

OUT_R = OUTPUT / "robust"


def load() -> dict:
    return json.loads(
        (OUT_R / "frozen_structure.json").read_text(encoding="utf-8"))


def test_discrimination_frozen_outside_train():
    """冻结段需求抬升为生成器收尾结构（z>2，训练分布之外）。"""
    s1 = load()["s1_discrimination"]
    assert s1["z_score"] > 2.0
    assert s1["percentile"] >= 99.0


def test_rolling_requantile_improves():
    """滚动重估修复静态分位数外推失效（336h > 静态）。"""
    s3 = load()["s3_rolling_requantile"]
    assert s3["rolling_336h"] > s3["static_all_train"] + 0.01
    assert s3["rolling_336h"] >= 0.94


def test_frozen_viol_rate_near_epsilon():
    """冻结段最终验证：预留调度超容率 ≈ ε=5% 设计。"""
    s4 = load()["s4_kappa_final"]
    assert 2.0 < s4["viol_rate_frozen_pct"] < 7.0


def test_conditional_coverage_direction():
    """覆盖风险集中在高需求窗口（高需求箱覆盖更低）。"""
    s2 = load()["s2_conditional_cov"]
    assert s2["high_cov"] < s2["low_cov"]


def test_price_template_structure():
    """冻结段价格存在日内结构：日模板优于段标签均值（0.19% 机理线索）。"""
    s5 = load()["s5_price_mystery"]
    assert s5["median_hour_template_mape_pct"] < \
        s5["median_period_template_mape_pct"]
