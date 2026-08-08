"""step0.5 SOC 递推重建回归：分区效率吻合 / E 区偏移标记 / 产物完整性."""
import json

import numpy as np
import pandas as pd

from conftest import load_step_module
from step0_config import CLEAN, REGIONS

s05 = load_step_module("step0.5_soc_rebuild.py")


def _load_summary() -> dict:
    return json.loads(
        (CLEAN / "soc_rebuild.json").read_text(encoding="utf-8"))


def _load_csv() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "soc_rebuilt.csv")


def test_products_exist():
    assert (CLEAN / "soc_rebuilt.csv").exists()
    assert (CLEAN / "soc_rebuild.json").exists()
    df = _load_csv()
    assert len(df) == len(REGIONS) * 2407
    assert set(df.columns) == {"Region", "Hour", "SOC_data_MWh",
                               "SOC_rebuilt_MWh", "diff_MWh"}


def test_nonE_regions_match_data_table():
    """分区效率递推应与数据表吻合（diff <= 0.01 MWh）。"""
    df = _load_csv()
    for r in REGIONS:
        if r == "RegionE":
            continue
        sub = df[df.Region == r]
        assert np.abs(sub["diff_MWh"]).max() < 0.01, f"{r} 递推与数据表不符"


def test_regionE_constant_offset():
    """RegionE 数据表 = 递推 − 1.0（全程恒定偏移，官方确认录入误差）。"""
    df = _load_csv()
    sub = df[df.Region == "RegionE"]
    assert np.abs(sub["diff_MWh"] - 1.0).max() < 0.01
    assert np.abs(sub["diff_MWh"].std() - 0.0) < 0.01


def test_eta_partition():
    """效率分区制（A/B/C 与 D/E/F 不同）—— 早期全 0.93 文档错误的守卫。"""
    assert s05.ETA_C["RegionA"] == 0.93
    assert s05.ETA_C["RegionE"] == 0.94
    assert s05.ETA_D["RegionE"] == 0.93


def test_soc_within_bounds():
    """递推 SOC 应在 [MinSOC, Cap] 内（容差 0.01：数据表=round(递推,3位)）。

    RegionE 例外：偏移 +1.0 使递推真值可超 Cap 至 821（生成器口径松弛，
    官方定性"录入误差"），容差 1.02；数据表本身恒在界内。
    """
    df = _load_csv()
    sp = pd.read_csv(CLEAN / "storage_params.csv").set_index("Region")
    for r in REGIONS:
        lo = sp.loc[r, "MinSOC_MWh"]
        hi = sp.loc[r, "StorageCapacity_MWh"]
        tol = 1.02 if r == "RegionE" else 0.01
        sub = df[df.Region == r]["SOC_rebuilt_MWh"]
        assert sub.min() >= lo - tol
        assert sub.max() <= hi + tol
        d = df[df.Region == r]["SOC_data_MWh"]
        assert d.min() >= lo - 0.01 and d.max() <= hi + 0.01, \
            f"{r} 数据表本身应在界内"


def test_terminal_vs_initial_recorded():
    """终态特征记录：题目基线 D/E/F 区 SOC(2406) < InitialSOC ——
    Q3 终态约束 SOC(2406) ≥ InitialSOC 的优化张力来源（基线不满足，需优化回升）。"""
    df = _load_csv()
    shortfall = {}
    for r in REGIONS:
        v = df[(df.Region == r) & (df.Hour == 2406)]["SOC_rebuilt_MWh"].iloc[0]
        init = s05.INIT_SOC[r]
        shortfall[r] = float(v - init)
    assert shortfall["RegionD"] < 0 and shortfall["RegionE"] < 0 \
        and shortfall["RegionF"] < 0, "D/E/F 基线终态应低于 InitialSOC（Q3 张力）"
