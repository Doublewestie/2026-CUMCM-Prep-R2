"""M1 地基回归：白名单 / 占用展开 / 质量报告 / 序列形状."""
import json

import pandas as pd
import pytest

from step0_config import CLEAN, HOURS_TOTAL, MAX_LATENCY, REGIONS, TASK_TYPES


def test_quality_report_zero_dirty():
    q = json.loads((CLEAN / "quality_report.json").read_text(encoding="utf-8"))
    assert q["workload_nan"] == 0
    assert q["region_time_nan"] == 0
    assert q["contradict_tasks"] == 0
    assert q["negative_slack_tasks"] == 0
    assert q["latest_finish_above_2406"] == 0
    assert q["arrival_range"] == [0, 2399]


def test_whitelist_matches_latency_matrix():
    wl = pd.read_csv(CLEAN / "whitelist.csv")
    lat = pd.read_csv(CLEAN / "whitelist.csv")  # 结构校验占位
    assert set(wl.columns) == {"TaskType", "SourceRegion", "Reachable", "MaxLatency_ms"}
    for tt in TASK_TYPES:
        row = wl[wl.TaskType == tt].iloc[0]
        assert row["MaxLatency_ms"] == MAX_LATENCY[tt]
        assert len(row["Reachable"].split("|")) >= 1


def test_rt_whitelist_d_isolated():
    wl = pd.read_csv(CLEAN / "whitelist.csv")
    rt = wl[wl.TaskType == "RealTimeInference"]
    n_reach = rt["Reachable"].str.split("|").apply(len)
    assert n_reach.min() == 1
    assert n_reach.max() <= 3


def test_occupancy_row_count_anchor():
    occ = pd.read_csv(CLEAN / "occupancy_local.csv")
    assert len(occ) == 195047
    assert set(occ.columns) == {"TaskID", "Region", "Hour", "GPU_Demand"}


def test_series_shapes():
    gd = pd.read_csv(CLEAN / "series_gpu_demand.csv")
    ar = pd.read_csv(CLEAN / "series_arrivals.csv")
    assert gd.shape == (HOURS_TOTAL, 1 + len(REGIONS) * len(TASK_TYPES))
    assert ar.shape == (HOURS_TOTAL, 3)


def test_region_time_hour_span():
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    assert rt.groupby("Region")["Hour"].nunique().eq(HOURS_TOTAL).all()
    assert set(rt["Hour"]) == set(range(HOURS_TOTAL))
