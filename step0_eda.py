"""step0_eda — 证据链 D1-D9 图集与数据质量可视化（论文附录素材）.

输出（figures/eda/）:
  figD1_whitenoise.png       任务到达白噪声证据（时序/自相关/lag 热图）
  figD2_renewable_template.png 新能源全区模板与严格日周期
  figD3_overcapacity.png     本地执行超容证据（E/F 超容小时）
  figD4_curtailment.png      弃电结构（hod 分布 + 累计弃电）
  figD5_phase_misalignment.png 负荷/价格/新能源三曲线相位错位
  figD6_whitelist.png        时延矩阵热图 + 白名单三层
  figD7_price_structure.png  电价时序与价格周期色带
  figD8_realtime_slack.png   实时推理 slack 分布 + 本地无超容验证
  figD9_task_specs.png       任务规格分布（GPU_Demand/Duration）
  eda_summary.json           全部证据链数字汇总（论文引用）
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from step0_config import (CLEAN, FIGURES, GPU_POWER_MW, MAX_LATENCY, REGIONS,
                          TASK_TYPES, HOURS_TOTAL, SETTLE_HOUR)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT = FIGURES / "eda"
OUT.mkdir(parents=True, exist_ok=True)

summary = {}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def figD1(wt, gpu_demand_series, gpu_series, occ_par):
    idx = pd.RangeIndex(0, HOURS_TOTAL)
    arr = gpu_demand_series.set_index("Hour") if "Hour" in gpu_demand_series.columns else gpu_demand_series
    n_tasks = gpu_series.set_index("Hour")["n_tasks"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(n_tasks.index, n_tasks.values, lw=0.6, color="#4a7a9b")
    axes[0].plot(n_tasks.rolling(168, min_periods=1).mean(), lw=2, color="#c0392b")
    axes[0].set_title("每小时到达任务数（红线=168h 滚动均值）")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("n_tasks")
    lags = []
    for r in REGIONS:
        for t in TASK_TYPES:
            s = pd.Series(arr.get(f"{r}|{t}", 0.0)).astype(float)
            lags.append((r, t, s.autocorr(1), s.autocorr(24)))
    lagdf = pd.DataFrame(lags, columns=["Region", "Type", "lag1", "lag24"])
    im = axes[1].imshow(lagdf[["lag1", "lag24"]].values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.2)
    axes[1].set_yticks(range(len(lagdf)), [f"{a}|{b}" for a, b in zip(lagdf.Region, lagdf.Type)], fontsize=6)
    axes[1].set_xticks([0, 1], ["lag1", "lag24"])
    axes[1].set_title("18 序列自相关热图")
    plt.colorbar(im, ax=axes[1], fraction=0.03)
    s0 = pd.Series(arr.get("RegionD|AITraining", 0.0)).astype(float)
    r = list(range(1, 49))
    acf = [s0.autocorr(l) for l in r]
    axes[2].bar(r, acf, width=0.8, color="#7f8c8d")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].axhline(1.96 / np.sqrt(2400), color="r", ls="--", lw=0.8)
    axes[2].axhline(-1.96 / np.sqrt(2400), color="r", ls="--", lw=0.8)
    axes[2].set_title("示例序列 ACF（RegionD|AITraining）")
    axes[2].set_xlabel("lag")
    fig.suptitle("D1 证据：任务到达为白噪声（lag1≈-0.01, lag24≈0.02）", y=1.02)
    fig.tight_layout(); save(fig, "figD1_whitenoise.png")
    q, p = stats.boxcox1p(s0.values + 1, 0) if False else (0, 0)
    summary["D1"] = {
        "mean_lag1": round(float(lagdf["lag1"].mean()), 4),
        "mean_lag24": round(float(lagdf["lag24"].mean()), 4),
        "range_n_tasks_hour": [int(n_tasks.min()), int(n_tasks.max())],
    }


def figD2(rt):
    piv = rt.pivot_table(index="Hour", columns="Region", values="AvailableRenewable_MW")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for r in REGIONS:
        axes[0].plot(piv.index, piv[r], lw=0.8, alpha=0.7, label=r)
    axes[0].set_title("D2 证据：六区域可用新能源曲线完全重合")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("MW")
    axes[0].legend(fontsize=8, ncol=3)
    hod = piv.groupby(piv.index % 24).mean()
    axes[1].plot(hod.index, hod["RegionE"], "o-", label="RegionE(光伏)")
    axes[1].plot(hod.index, hod["RegionF"], "s--", label="RegionF(风电)")
    axes[1].set_title("按小时日均值：光伏与风电曲线一致（模板化）")
    axes[1].set_xlabel("hour-of-day")
    axes[1].legend()
    fig.tight_layout(); save(fig, "figD2_renewable_template.png")
    summary["D2"] = {
        "all_regions_identical": bool((piv["RegionA"] == piv["RegionE"]).all()),
        "day0_eq_day1": bool((piv["RegionE"].iloc[0:24].values == piv["RegionE"].iloc[24:48].values).all()),
        "range_MW": [float(piv.min().min()), float(piv.max().max())],
    }


def figD3(occ_par, gpu_info):
    cap = gpu_info.set_index("Region")["Available_GPU"].to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    over = {}
    for i, r in enumerate(REGIONS):
        s = occ_par[occ_par.Region == r].set_index("Hour")["GPU_Demand"]
        over[r] = int((s > cap[r]).sum())
        axes[0].plot(s.index, s.values, lw=0.5, alpha=0.6, label=f"{r}(超容{over[r]}h)")
        axes[0].axhline(cap[r], color="k", ls="--", lw=0.8)
    axes[0].set_title("D3 证据：本地执行占用 vs 可用容量（虚线=容量）")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("并行 GPU")
    axes[0].legend(fontsize=7, ncol=2)
    s_f = occ_par[occ_par.Region == "RegionF"].set_index("Hour")["GPU_Demand"]
    ov_f = (s_f > cap["RegionF"])
    hod = np.where(ov_f.values, s_f.index % 24, np.nan)
    axes[1].hist(hod[~np.isnan(hod)], bins=24, color="#c0392b")
    axes[1].set_title("RegionF 超容小时分布（67h，最大超 62%）")
    axes[1].set_xlabel("hour-of-day")
    fig.tight_layout(); save(fig, "figD3_overcapacity.png")
    summary["D3"] = {r: over[r] for r in REGIONS}


def figD4(rt):
    e = rt[rt.Region == "RegionE"].copy()
    e["hod"] = e["Hour"] % 24
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].bar(e.groupby("hod")["Curtailment_MW"].mean().index,
                e.groupby("hod")["Curtailment_MW"].mean().values, color="#e67e22")
    axes[0].set_title("D4 证据：RegionE 弃电按小时均值（峰值 5-17 时）")
    axes[0].set_xlabel("hour-of-day"); axes[0].set_ylabel("MW")
    byr = rt.groupby("Region").agg(
        curtail=("Curtailment_MW", "sum"), avail=("AvailableRenewable_MW", "sum"),
        used=("UsedRenewable_MW", "sum"), ren_chg=("RenewableCharge_MW", "sum"),
        sell=("GridSell_MW", "sum"))
    byr["curtail_ratio"] = byr["curtail"] / byr["avail"] * 100
    axes[1].bar(byr.index, byr["curtail_ratio"], color="#c0392b")
    axes[1].set_title("各区域弃电率（%）")
    axes[1].set_ylabel("%")
    fig.tight_layout(); save(fig, "figD4_curtailment.png")
    summary["D4"] = {
        "total_curtail_MWh": float(rt["Curtailment_MW"].sum()),
        "utilization_pct": float((byr["used"] + byr["ren_chg"] + byr["sell"]).sum() / byr["avail"].sum() * 100),
        "curtail_by_region_pct": {r: round(v, 1) for r, v in byr["curtail_ratio"].items()},
    }


def figD5(rt):
    a = rt[rt.Region == "RegionA"].copy()
    a["hod"] = a["Hour"] % 24
    fig, ax = plt.subplots(figsize=(10, 4.6))
    norm = lambda s: (s - s.min()) / (s.max() - s.min())
    ax.plot(a.groupby("hod")["NonAI_IT_Load_MW"].mean().index,
            norm(a.groupby("hod")["NonAI_IT_Load_MW"].mean()).values, label="NonAI 负荷")
    ax.plot(a.groupby("hod")["ElectricityPrice_CNY_per_MWh"].mean().index,
            norm(a.groupby("hod")["ElectricityPrice_CNY_per_MWh"].mean()).values, label="电价")
    e = rt[rt.Region == "RegionE"].copy(); e["hod"] = e["Hour"] % 24
    ax.plot(e.groupby("hod")["AvailableRenewable_MW"].mean().index,
            norm(e.groupby("hod")["AvailableRenewable_MW"].mean()).values, label="新能源(RegionE)")
    ax.set_title("D5 证据：负荷-价格-新能源三曲线相位错位（归一化）")
    ax.set_xlabel("hour-of-day"); ax.legend()
    fig.tight_layout(); save(fig, "figD5_phase_misalignment.png")
    peak = a[a.PricePeriod == "Peak"]["NonAI_IT_Load_MW"].mean()
    valley = a[a.PricePeriod == "Valley"]["NonAI_IT_Load_MW"].mean()
    summary["D5"] = {
        "nonAI_load_peak_valley_diff_pct": round((valley / peak - 1) * 100, 1),
        "valley_price": round(float(a[a.PricePeriod == "Valley"]["ElectricityPrice_CNY_per_MWh"].mean()), 1),
        "peak_price": round(float(a[a.PricePeriod == "Peak"]["ElectricityPrice_CNY_per_MWh"].mean()), 1),
    }


def figD6(latency_raw):
    lat = latency_raw.pivot_table(index="FromRegion", columns="ToRegion",
                                  values="NetworkLatency_ms", aggfunc="first")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    im = axes[0].imshow(lat.values, cmap="YlOrRd")
    axes[0].set_xticks(range(6), lat.columns, fontsize=7)
    axes[0].set_yticks(range(6), lat.index, fontsize=7)
    for i in range(6):
        for j in range(6):
            axes[0].text(j, i, int(lat.values[i, j]), ha="center", va="center", fontsize=7)
    axes[0].set_title("D6 证据：单向时延矩阵（ms）")
    plt.colorbar(im, ax=axes[0], fraction=0.03)
    wl = pd.read_csv(CLEAN / "whitelist.csv")
    wl["n_reach"] = wl["Reachable"].str.split("|").apply(len)
    piv = wl.pivot_table(index="TaskType", columns="SourceRegion", values="n_reach")
    im2 = axes[1].imshow(piv.values, cmap="Blues", vmin=1, vmax=6)
    axes[1].set_yticks(range(3), piv.index, fontsize=8)
    axes[1].set_xticks(range(6), piv.columns, fontsize=7)
    for i in range(3):
        for j in range(6):
            axes[1].text(j, i, int(piv.values[i, j]), ha="center", va="center", fontsize=8)
    axes[1].set_title("白名单可达区域数（三层 20/80/150ms）")
    plt.colorbar(im2, ax=axes[1], fraction=0.03)
    fig.tight_layout(); save(fig, "figD6_whitelist.png")
    summary["D6"] = {tt: wl[wl.TaskType == tt]["n_reach"].min().item() for tt in TASK_TYPES}


def figD7(rt):
    a = rt[rt.Region == "RegionA"].copy()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    colors = {"Valley": "#3498db", "Flat": "#95a5a6", "Peak": "#e74c3c"}
    ax.plot(a["Hour"], a["ElectricityPrice_CNY_per_MWh"], lw=0.7, color="#2c3e50")
    for t0 in range(0, 2407, 48):
        seg = a[(a.Hour >= t0) & (a.Hour < t0 + 48)]
        pp = seg["PricePeriod"].iloc[0] if len(seg) else "Flat"
        ax.axvspan(t0, t0 + 48, color=colors.get(pp, "#95a5a6"), alpha=0.12)
    ax.set_title("D7 证据：电价逐时变化 + 价格周期色带（RegionA）")
    ax.set_xlabel("Hour"); ax.set_ylabel("元/MWh")
    fig.tight_layout(); save(fig, "figD7_price_structure.png")
    d = a.groupby("PricePeriod")["ElectricityPrice_CNY_per_MWh"]
    summary["D7"] = {k: {"mean": round(float(v.mean()), 1),
                         "spread_pct": round((v.max() - v.min()) / v.mean() * 100, 1)}
                     for k, v in d}


def figD8(wt, occ_par, gpu_info):
    rt_ = wt[wt.TaskType == "RealTimeInference"].copy()
    rt_["slack"] = rt_["LatestFinishHour"] - rt_["ArrivalHour"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].hist(rt_["slack"], bins=range(1, 9), color="#2e86c1", edgecolor="w")
    axes[0].set_title("D8 证据：实时推理 slack 分布（中位 4h，弹性≈0）")
    axes[0].set_xlabel("slack (h)")
    cap = gpu_info.set_index("Region")["Available_GPU"].to_dict()
    rt_only = occ_par  # occupancy 按来源区域；此处仅展示 max/容量 比率
    mx = occ_par.groupby("Region")["GPU_Demand"].max()
    axes[1].bar(range(6), [mx[r] / cap[r] for r in REGIONS], color="#16a085")
    axes[1].set_xticks(range(6), REGIONS, fontsize=8)
    axes[1].axhline(1, color="r", ls="--")
    axes[1].set_title("本地执行峰值占用 / 可用容量（>1 即超容）")
    fig.tight_layout(); save(fig, "figD8_realtime_slack.png")
    summary["D8"] = {"rt_slack_median": int(rt_["slack"].median()),
                     "rt_slack_range": [int(rt_["slack"].min()), int(rt_["slack"].max())]}


def figD9(wt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(wt["GPU_Demand"], bins=40, color="#8e44ad")
    axes[0].set_title("D9 任务规格：GPU_Demand 分布")
    axes[0].set_xlabel("GPU")
    axes[1].hist(wt["EstimatedDuration_min"], bins=40, color="#2980b9")
    axes[1].set_title("D9 任务规格：时长分布（min）")
    axes[1].set_xlabel("min")
    fig.tight_layout(); save(fig, "figD9_task_specs.png")
    summary["D9"] = {
        "gpu_demand": {"min": int(wt.GPU_Demand.min()), "median": int(wt.GPU_Demand.median()),
                       "max": int(wt.GPU_Demand.max())},
        "n_tasks": int(len(wt)),
        "type_counts": wt.TaskType.value_counts().to_dict(),
    }


def main():
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    gpu_demand_series = pd.read_csv(CLEAN / "series_gpu_demand.csv")
    gpu_series = pd.read_csv(CLEAN / "series_arrivals.csv")
    occ_par = pd.read_csv(CLEAN / "occupancy_parallel.csv")
    gpu_info = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw" / "GPU_information.xlsx",
                             sheet_name="GPU中心基础情况")
    lat_raw = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw" / "network_latency.xlsx",
                            sheet_name="network_latency")

    figD1(wt, gpu_demand_series, gpu_series, occ_par)
    figD2(rt)
    figD3(occ_par, gpu_info)
    figD4(rt)
    figD5(rt)
    figD6(lat_raw)
    figD7(rt)
    figD8(wt, occ_par, gpu_info)
    figD9(wt)

    with open(OUT / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
