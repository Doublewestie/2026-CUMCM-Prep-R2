"""step1.1 竞技榜回归：QRF 均值化陷阱 / TabPFN 形状与性能配置 / 门判据 / 特征集."""
import importlib.util
import time

import numpy as np
import pandas as pd
import pytest

from conftest import load_step_module
from step0_config import CLEAN

s11 = load_step_module("step1.1_forecast_arena.py")


@pytest.fixture(scope="module")
def task_series():
    return s11.make_series_dict()["RegionA|RealTimeInference"]


@pytest.fixture(scope="module")
def price_series():
    return s11.make_series_dict()["energy|RegionA|price"]


def test_model_pool_interface_ok():
    for layer in ("task", "energy"):
        pool = s11.build_model_pool(layer)
        for m in pool:
            assert hasattr(m, "fit") and hasattr(m, "predict_point")
            assert hasattr(m, "predict_quantile")


def test_priceperiod_features_present():
    y = np.zeros(s11.HOURS_TOTAL)
    X = s11.build_features(y, "energy", "RegionA")
    for col in ("pp_valley", "pp_flat", "pp_peak", "seg_price_level"):
        assert col in X.columns
    task_X = s11.build_features(y, "task")
    assert "pp_valley" not in task_X.columns


def test_qrf_meinshausen_no_mean_trap(task_series):
    """回归：QRF 必须用叶内样本分位数，任务侧 cov 不得退化到 ~0.08。"""
    y = task_series["y"]
    X = s11.build_features(y, "task")
    pool = s11.build_model_pool("task")
    qrf = [m for m in pool if m.name == "qrf_quantile"][0]
    from sklearn.model_selection import TimeSeriesSplit
    tr, va = next(iter(TimeSeriesSplit(n_splits=5).split(np.arange(s11.SEG_TRAIN))))
    qrf.fit(X.iloc[tr], y[tr])
    q = qrf.predict_quantile(X.iloc[va], 0)
    point = qrf.predict_point(X.iloc[va], 0)
    sc = s11.evaluate_fold("task", y[va], point, q)
    assert sc["cov"] > 0.5, f"QRF cov={sc['cov']} 均值化陷阱复发"


def test_tabpfn_predict_shapes(price_series):
    y = price_series["y"]
    X = s11.build_features(y, "energy", "RegionA")
    pool = s11.build_model_pool("energy")
    tab = [m for m in pool if m.name == "tabpfn"][0]
    tab.fit(X.iloc[:s11.SEG_TRAIN], y[:s11.SEG_TRAIN])
    p = tab.predict_point(X, 0)
    q = tab.predict_quantile(X, 0)
    assert len(p) == s11.HOURS_TOTAL
    for a, v in q.items():
        assert len(v) == s11.HOURS_TOTAL


def test_tabpfn_no_state_leak(price_series):
    """回归：多次 fit 后 predict 不得灾难性退化（445s 事件）。"""
    y = price_series["y"]
    X = s11.build_features(y, "energy", "RegionA")
    pool = s11.build_model_pool("energy")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    t_start = time.time()
    for tr, va in tscv.split(np.arange(s11.SEG_TRAIN)):
        tab = s11.make_tabpfn_models("energy")[0]
        tab.fit(X.iloc[tr], y[tr])
        tab.predict_quantile(X.iloc[va], 0)
    tab = s11.make_tabpfn_models("energy")[0]
    tab.fit(X.iloc[:s11.SEG_TRAIN], y[:s11.SEG_TRAIN])
    tab.predict_quantile(X, 0)
    assert time.time() - t_start < 300, "TabPFN 多次 fit 后 predict 退化"


def test_gate_significance_requirement():
    base = {"cov_mean": 0.895, "cov_std": 0.021, "width_mean": 102.3,
            "pinball_mean": 27.6, "mape_mean": 2.8, "layer": "task"}
    marginal = {"n_folds": 5, "layer": "task", "cov_mean": 0.910,
                "width_mean": 112.0, "pinball_mean": 28.3}
    assert s11.apply_gate(marginal, base).startswith("拒绝")
    significant = {"n_folds": 5, "layer": "task", "cov_mean": 0.930,
                   "width_mean": 105.0, "pinball_mean": 26.0}
    assert s11.apply_gate(significant, base).startswith("通过")
    energy_gain = {"n_folds": 5, "layer": "energy", "mape_mean": 2.72}
    assert s11.apply_gate(energy_gain, base).startswith("拒绝")
    energy_win = {"n_folds": 5, "layer": "energy", "mape_mean": 2.0}
    assert s11.apply_gate(energy_win, base).startswith("通过")


def test_gate_v2_pinball_guard():
    """v2 守卫：cov 通道通过但 pinball 显著恶化 → 拒绝（宽度换覆盖防御）。"""
    base = {"cov_mean": 0.895, "cov_std": 0.019, "width_mean": 102.3,
            "pinball_mean": 27.6, "mape_mean": 2.8, "layer": "task"}
    edge = {"n_folds": 5, "layer": "task", "cov_mean": 0.923,
            "width_mean": 116.0, "pinball_mean": 28.7}   # pinball 恶化 -4.0%
    assert s11.apply_gate(edge, base).startswith("拒绝")
    ok = {"n_folds": 5, "layer": "task", "cov_mean": 0.923,
          "width_mean": 116.0, "pinball_mean": 27.0}    # pinball 持平
    assert s11.apply_gate(ok, base).startswith("通过")


def test_fuse_task_quantiles_all(task_series):
    y = task_series["y"]
    pool = s11.build_model_pool("task")
    q, meta = s11.fuse_task(y, pool, [])
    assert meta["fuse"] == "统计基线(无入选)"
    assert set(s11.KAPPA_QUANTILES) <= set(q.keys())
    for a in s11.ALL_QUANTILES:
        assert len(q[a]) == s11.HOURS_TOTAL
        assert not np.isnan(q[a]).any()
