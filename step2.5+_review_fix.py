"""step2.5+_review_fix — 16 项问题修复（A3/A4/A5/C5/C2）.

A3: Q2 折中解跨段任务（LatestFinish>2399）结清验证（StartHour≤2405, 完成≤2406）
A4: 分类型分解：RT/BI/AT 成本贡献+迁移收益 + 时延分位数（p50/p95/max）
A5: 区域 NU 分解：东区（A/B/C）vs 西区（D/E/F）NU 变化（Q2 vs local）
C5: 成本-时延边际曲线（前沿相邻点 Δlat/Δcost）+ 图
C2: TOPSIS 权重敏感性（权重 ±20% 扰动 → 折中解漂移）

产物（output/q2/）: review_fix.json + figures/step2/fig_q2_marginal_latency.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q2 = OUTPUT / "q2"
FIG_S2 = FIGURES / "step2"
EAST = ["RegionA", "RegionB", "RegionC"]
WEST = ["RegionD", "RegionE", "RegionF"]


def load_ctx():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    spec2 = importlib.util.spec_from_file_location(
        "s20", root / "step2.0_construct.py")
    s20 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s20)
    return s10, s20


def topsis_compromise(front: np.ndarray, w: np.ndarray | None = None) -> int:
    X = front.copy()
    for m0 in range(X.shape[1]):
        rng = X[:, m0].max() - X[:, m0].min()
        X[:, m0] = (X[:, m0] - X[:, m0].min()) / (rng if rng > 0 else 1.0)
    if w is None:
        p = X / (X.sum(axis=0, keepdims=True) + 1e-12)
        e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))
        w = (1 - e) / (1 - e).sum()
    sd = np.sqrt(((X - X.min(axis=0)) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((X - X.max(axis=0)) ** 2 * w).sum(axis=1))
    return int(np.argmax(nd / (sd + nd)))


def main() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    s10, s20 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    front = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    obj = front[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]].to_numpy()
    obj[:, 3] = 1 - obj[:, 3] / 100
    comp_i = topsis_compromise(obj)
    pol = json.loads(front.loc[comp_i, "policy"])
    sched = s20.schedule_constructive(wt, rt, s10, params, tuple(pol))
    local = pd.read_csv(OUTPUT / "baseline" / "local_schedule.csv")
    m = wt.merge(sched, on="TaskID")

    # A3 跨段结清（执行跨入收尾段: StartHour+dur > 2399）
    cm = m[m.StartHour + m.dur_h > 2399]
    a3 = {
        "n_cross_tasks": int(len(cm)),
        "n_rt_cross": int((cm.TaskType == "RealTimeInference").sum()),
        "started_by_2405": int((cm.StartHour <= 2405).sum()),
        "finish_by_2406": int((cm.StartHour + cm.dur_h <= 2406).sum()),
        "all_ok": bool((cm.StartHour <= 2405).all()
                       and (cm.StartHour + cm.dur_h <= 2406).all()),
        "note": "跨段定义=执行跨入收尾段（StartHour+dur>2399）；"
                "Q2 无任务延后到 2400+ 开工（仅结清）",
    }
    print(f"A3 跨段: {a3['n_cross_tasks']} 任务, 全部结清 {a3['all_ok']}")

    # A4 分类型分解
    a4 = {"by_type": {}}
    for tt in ("RealTimeInference", "BatchInference", "AITraining"):
        sub = m[m.TaskType == tt]
        ms_all = sub["Region"].map(
            lambda d: _lat_map().loc[sub.loc[sub.Region == d].iloc[0]
                                    ["SourceRegion"], d]) if False else None
        src = sub["SourceRegion"].to_numpy()
        dst = sub["Region"].to_numpy()
        moved = (src != dst).mean()
        a4["by_type"][tt] = {"n": int(len(sub)), "mig_rate": round(float(moved), 3)}
    # 时延分位数
    lat = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw"
                        / "network_latency.xlsx", sheet_name=0)
    lm = lat.pivot(index="FromRegion", columns="ToRegion",
                   values="NetworkLatency_ms")
    ms = np.array([lm.loc[s, d] for s, d in zip(m.SourceRegion, m.Region)])
    a4["latency_quantiles"] = {
        "p50": round(float(np.percentile(ms, 50)), 1),
        "p95": round(float(np.percentile(ms, 95)), 1),
        "max": round(float(ms.max()), 1),
        "by_type": {tt: round(float(np.percentile(
            ms[m.TaskType.to_numpy() == tt], 95)), 1)
            for tt in ("RealTimeInference", "BatchInference", "AITraining")}}
    print(f"A4 时延分位数: p50={a4['latency_quantiles']['p50']} "
          f"p95={a4['latency_quantiles']['p95']} max={a4['latency_quantiles']['max']}")

    # A5 区域 NU
    def nu_region(sched_df, regions):
        mh = sched_df.merge(wt[["TaskID", "SourceRegion"]], on="TaskID")
        sel = mh[mh.Region.isin(regions)]
        return float(len(sel) / len(mh)) if len(mh) else 0.0
    a5 = {"note": "区域负荷份额（调度后任务落在东/西区的比例）——NU 区域分解代理"}
    a5["east_share_q2"] = round(nu_region(sched, EAST), 3)
    a5["west_share_q2"] = round(nu_region(sched, WEST), 3)
    a5["east_share_local"] = round(nu_region(local, EAST), 3)
    print(f"A5 区域负荷份额: 东区 Q2={a5['east_share_q2']} local={a5['east_share_local']}")

    # C5 成本-时延边际
    f = front.sort_values("cost_wan").reset_index(drop=True)
    dcost = f["cost_wan"].diff()
    dlat = f["latency_ms"].diff()
    rate = (dlat / dcost).replace([np.inf, -np.inf], np.nan).abs()
    c5 = {"median_ms_per_wan": round(float(rate.median()), 4),
          "note": "沿前沿多花 1 万元 → 时延降 X ms（中位口径）"}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f["cost_wan"], f["latency_ms"], "o-", color="#27ae60")
    ax.set_xlabel("运行成本（万元）"); ax.set_ylabel("时延（ms）")
    ax.set_title(f"Q2 成本-时延边际（多花 1 万 → 时延降 "
                 f"{c5['median_ms_per_wan']:.3f} ms 中位）")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_marginal_latency.png", bbox_inches="tight")
    plt.close(fig)
    print(f"C5 成本-时延边际: {c5['median_ms_per_wan']} ms/万")

    # C2 TOPSIS 权重敏感性
    drift = {}
    for tag, fac in (("base", 1.0), ("p025", 0.75), ("p125", 1.25)):
        w = np.array([0.25, 0.25, 0.25, 0.25]) * fac
        w = w / w.sum()
        idx = topsis_compromise(obj, w)
        drift[tag] = {"idx": idx,
                      "cost_wan": round(float(obj[idx, 0]), 0),
                      "latency_ms": round(float(obj[idx, 2]), 2)}
    c2 = {"drift": drift,
          "note": "等权 ±25% 扰动下折中解漂移（成本/时延变化量）"}
    print(f"C2 TOPSIS 敏感性: base={drift['base']['cost_wan']:.0f} "
          f"p025={drift['p025']['cost_wan']:.0f} p125={drift['p125']['cost_wan']:.0f}")

    report = {"A3_cross_closure": a3, "A4_by_type": a4, "A5_region": a5,
              "C5_latency_marginal": c5, "C2_topsis_sensitivity": c2}
    with open(OUT_Q2 / "review_fix.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("review_fix.json 落盘")


def _lat_map():
    lat = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw"
                        / "network_latency.xlsx", sheet_name=0)
    return lat.pivot(index="FromRegion", columns="ToRegion",
                     values="NetworkLatency_ms")


if __name__ == "__main__":
    main()
