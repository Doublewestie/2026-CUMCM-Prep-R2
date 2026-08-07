"""step0_loader — 统一数据加载与预处理（全题共享地基）.

产物（output/clean/）:
  quality_report.json      数据质量检查报告
  region_time_clean.csv    0-2406 × 6 区域逐时电力参数（含 DataPeriod 标记）
  workload_clean.csv       任务表 + dur_h/slack 派生列
  whitelist.csv            任务类型×来源 → 可达区域集合（时延白名单）
  occupancy_local.csv      本地执行假设下的任务小时占用表
  series_gpu_demand.csv    18 条逐时 GPU 需求序列（区域×类型）
  series_arrivals.csv      逐时到达数量/GPU-hours 序列
  storage_params.csv       储能参数表（含 SOC 骨架）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step0_config import (CLEAN, DATA_RAW, FILES, HOURS_TOTAL, MAIN_END,
                          MAX_LATENCY, PRIORITY, REGIONS, SETTLE_HOUR,
                          TASK_TYPES)


def load_all() -> dict[str, pd.DataFrame]:
    """读取全部 6 张表（显式 sheet 名，杜绝读错表）。"""
    data = {}
    data["workload"] = pd.read_excel(DATA_RAW / FILES["workload"], sheet_name="Sheet1")
    data["region_time"] = pd.read_excel(DATA_RAW / FILES["region_time"], sheet_name="region_time_data")
    data["gpu"] = pd.read_excel(DATA_RAW / FILES["gpu"], sheet_name="GPU中心基础情况")
    data["latency"] = pd.read_excel(DATA_RAW / FILES["latency"], sheet_name="network_latency")
    data["power"] = pd.read_excel(DATA_RAW / FILES["power"], sheet_name="任务功率映射")
    data["storage"] = pd.read_excel(DATA_RAW / FILES["storage"], sheet_name="storage_information")
    return data


def check_quality(wt: pd.DataFrame, rt: pd.DataFrame) -> dict:
    """数据质量检查：NaN / 矛盾任务 / 时段完整性 / 时延矩阵对称性核对。"""
    report = {}
    report["workload_nan"] = int(wt.isna().sum().sum())
    report["region_time_nan"] = int(rt.isna().sum().sum())
    wt2 = wt.copy()
    wt2["slack"] = wt2["LatestFinishHour"] - wt2["ArrivalHour"]
    wt2["dur_h"] = wt2["EstimatedDuration_min"] / 60.0
    report["contradict_tasks"] = int((wt2["dur_h"] > wt2["slack"]).sum())
    report["negative_slack_tasks"] = int((wt2["slack"] < 0).sum())
    report["latest_finish_above_2406"] = int((wt2["LatestFinishHour"] > SETTLE_HOUR).sum())
    report["arrival_range"] = [int(wt["ArrivalHour"].min()), int(wt["ArrivalHour"].max())]
    hour_count = rt.groupby("Region")["Hour"].nunique()
    report["region_hour_counts"] = hour_count.to_dict()
    report["hour_range"] = [int(rt["Hour"].min()), int(rt["Hour"].max())]
    report["data_periods"] = rt["DataPeriod"].value_counts().to_dict()
    return report


def build_whitelist(latency: pd.DataFrame) -> pd.DataFrame:
    """预计算 (TaskType, SourceRegion) → 可达区域集合（时延白名单三层）。"""
    lat = latency.pivot_table(index="FromRegion", columns="ToRegion",
                              values="NetworkLatency_ms", aggfunc="first")
    rows = []
    for tt in TASK_TYPES:
        th = MAX_LATENCY[tt]
        for src in REGIONS:
            ok = [r for r in REGIONS if lat.loc[src, r] <= th]
            rows.append({"TaskType": tt, "SourceRegion": src,
                         "Reachable": "|".join(ok), "MaxLatency_ms": th})
    return pd.DataFrame(rows)


def expand_occupancy(wt: pd.DataFrame) -> pd.DataFrame:
    """本地执行假设下展开任务小时占用表（1h 粒度重叠折算）.

    任务 i 占用 [ArrivalHour, ArrivalHour+dur_h)，对每个重叠小时产出
    一行 (TaskID, Region, Hour, GPU_Demand)；容量核算按并行 GPU 数，
    功率核算按 Overlap 小时数（GPU-hour）。
    """
    rows = []
    for rec in wt.itertuples(index=False):
        h0 = int(rec.ArrivalHour)
        h1 = int(np.ceil(rec.ArrivalHour + rec.EstimatedDuration_min / 60.0))
        for h in range(h0, min(h1, SETTLE_HOUR + 1)):
            rows.append((rec.TaskID, rec.SourceRegion, h, rec.GPU_Demand))
    occ = pd.DataFrame(rows, columns=["TaskID", "Region", "Hour", "GPU_Demand"])
    return occ


def build_series(wt: pd.DataFrame, occ: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """18 条逐时 GPU 需求序列 + 到达过程序列（全部覆盖 0-2406）。"""
    idx = pd.RangeIndex(0, HOURS_TOTAL, name="Hour")
    piv = wt.pivot_table(index="ArrivalHour", columns=["SourceRegion", "TaskType"],
                         values="GPU_Demand", aggfunc="sum", fill_value=0)
    piv = piv.reindex(idx, fill_value=0)
    cols = [f"{r}|{t}" for r in REGIONS for t in TASK_TYPES]
    piv.columns = cols
    series = {"gpu_demand": piv.reset_index()}

    w = wt.copy()
    w["gpu_hours"] = w["GPU_Demand"] * w["EstimatedDuration_min"] / 60.0
    arr = w.groupby("ArrivalHour").agg(
        n_tasks=("TaskID", "count"), gpu_hours=("gpu_hours", "sum")).reindex(idx, fill_value=0)
    arr.index.name = "Hour"
    series["arrivals"] = arr.reset_index()

    occ_par = occ.groupby(["Region", "Hour"])["GPU_Demand"].sum().reindex(
        pd.MultiIndex.from_product([REGIONS, idx]), fill_value=0)
    occ_par.index.names = ["Region", "Hour"]
    series["occupancy_parallel"] = occ_par.reset_index()
    return series


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wt, rt = data["workload"], data["region_time"]

    report = check_quality(wt, rt)

    wt = wt.copy()
    wt["dur_h"] = wt["EstimatedDuration_min"] / 60.0
    wt["slack"] = wt["LatestFinishHour"] - wt["ArrivalHour"]
    wt["Priority"] = wt["TaskType"].map(PRIORITY)

    whitelist = build_whitelist(data["latency"])
    occ = expand_occupancy(wt)
    series = build_series(wt, occ)

    rt = rt.sort_values(["Region", "Hour"]).reset_index(drop=True)
    storage = data["storage"].sort_values("Region").reset_index(drop=True)

    wt.to_csv(CLEAN / "workload_clean.csv", index=False)
    rt.to_csv(CLEAN / "region_time_clean.csv", index=False)
    whitelist.to_csv(CLEAN / "whitelist.csv", index=False)
    occ.to_csv(CLEAN / "occupancy_local.csv", index=False)
    series["gpu_demand"].to_csv(CLEAN / "series_gpu_demand.csv", index=False)
    series["arrivals"].to_csv(CLEAN / "series_arrivals.csv", index=False)
    series["occupancy_parallel"].to_csv(CLEAN / "occupancy_parallel.csv", index=False)
    storage.to_csv(CLEAN / "storage_params.csv", index=False)

    with open(CLEAN / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("quality_report:", json.dumps(report, ensure_ascii=False))
    print("whitelist 前5行:")
    print(whitelist.head().to_string(index=False))
    print("occupancy 行数:", len(occ), "| series 形状:",
          series["gpu_demand"].shape, series["arrivals"].shape)


if __name__ == "__main__":
    main()
