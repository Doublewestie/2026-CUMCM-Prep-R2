"""step1.2+_e1_mechanism — E1 机理深化（E1a-e）+ 瓶颈转移（M3 构造启发式输入）.

E1a  时空套利分解: shift_only(错峰不迁) vs migrate_only(迁移不错峰) vs 双管
E1b  预留贡献: κ∈{0, 0.05, 0.10} 三档对比（超容/利用率/成本）
E1c  时空收益热图: 价格时段(V/F/P) × 区域 的成本改进分解
E1d  预测完美度-收益曲线: 实际需求混入 0/25/50/75/100% 噪声 → 超容率/成本
E1e  η 理论下界: 任务侧 η≈0 → 预测改进收益上界（理论+实证双通道）
瓶颈转移: greedy 延后任务分布（区域×时段）→ M3 构造启发式

产物（output/robust/）: e1_mechanism.json / bottleneck.json /
  figures/step1/{figE1a_decomposition, figE1c_heatmap, figE1d_curve}.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import (CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPES,
                          HOURS_TOTAL)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_R = OUTPUT / "robust"
FIG_S1 = FIGURES / "step1"
SEG_TRAIN = 2352

PERIODS = ["Valley", "Flat", "Peak"]
PERIOD_HOURS = {"Valley": set(range(0, 7)), "Flat": set(range(7, 17)) | {22, 23},
                "Peak": set(range(17, 22))}


def load_ctx():
    spec = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    spec2 = importlib.util.spec_from_file_location(
        "s12", Path(__file__).resolve().parent / "step1.2_robust_schedule.py")
    s12 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s12)
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    return s10, s12, wt, rt, params, consume


def e1a_shift_vs_migrate(s10, wt, rt, params, consume):
    """错峰 vs 迁移 的空间/时间套利分解（Q2 前置探索）.

    migrate_only: BI/AT 迁移到白名单内平均电价最低区域（容量贪心，st=arrival）
    shift_only:   step1.0 greedy（延后，不迁移）
    """
    whitelist = pd.read_csv(CLEAN / "whitelist.csv")
    wl = {(r.TaskType, r.SourceRegion): r.Reachable.split("|")
          for r in whitelist.itertuples(index=False)}
    rt_avg = rt[rt.Hour < SEG_TRAIN].groupby("Region")[
        "ElectricityPrice_CNY_per_MWh"].mean()
    cost_rank = {r: i for i, r in enumerate(
        rt_avg.sort_values().index)}

    cap_arr = np.array([params["cap"][r] for r in REGIONS], dtype=float)
    occ = np.zeros((HOURS_TOTAL, len(REGIONS)))
    rows = []
    for rec in wt.sort_values(["Priority", "ArrivalHour"],
                              ascending=[False, True]).itertuples(index=False):
        g = float(rec.GPU_Demand)
        dur = float(rec.dur_h)
        if rec.TaskType == "RealTimeInference":
            r = REGIONS.index(rec.SourceRegion)
            st = int(rec.ArrivalHour)
        else:
            cands = sorted(wl[(rec.TaskType, rec.SourceRegion)],
                           key=lambda x: cost_rank[x])
            r = st = None
            for c in cands:
                ri = REGIONS.index(c)
                h1 = min(int(np.ceil(rec.ArrivalHour + dur)), HOURS_TOTAL)
                if (occ[rec.ArrivalHour:h1, ri] + g - cap_arr[ri]).max() <= 0:
                    r, st = ri, int(rec.ArrivalHour)
                    break
            if r is None:
                r = REGIONS.index(rec.SourceRegion)
                st = int(rec.ArrivalHour)
        rows.append((rec.TaskID, REGIONS[r], st))
        h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
        if h1 > st:
            occ[st:h1, r] += g
    mig_sched = pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])

    ev = lambda s: s10.evaluate_schedule(s, wt, rt, params, consume)[0]
    shift = ev(pd.read_csv(OUT_R.parent / "baseline" / "greedy_schedule.csv"))
    migrate = ev(mig_sched)
    return {"shift_only_cost_wan": shift["cost_wan"],
            "migrate_only_cost_wan": migrate["cost_wan"],
            "shift_only_viol": shift["viol_h"],
            "migrate_only_viol": migrate["viol_h"],
            "migrate_only_carbon": migrate["carbon_t"],
            "shift_only_carbon": shift["carbon_t"],
            "note": "迁移=白名单内最低均价区域+容量贪心（st=arrival）；"
                    "错峰=step1.0 greedy（不迁移）"}


def e1b_reserve_tiers(s10, s12, wt, rt, params, consume, q, act):
    """预留三档对比：κ∈{0, 0.05, 0.10}。"""
    out = {}
    for eps in (0.0, 0.05, 0.10):
        if eps == 0.0:
            k = {r: np.zeros(HOURS_TOTAL) for r in REGIONS}
        else:
            k = s12.compute_kappa(q, params["cap"], eps)
        sc = s12.schedule_with_kappa(wt, params["cap"], k)
        m = s10.evaluate_schedule(sc, wt, rt, params, consume)[0]
        out[f"eps_{eps:.2f}"] = {"viol_h": m["viol_h"],
                                 "cost_wan": m["cost_wan"],
                                 "mean_kappa": float(np.mean(
                                     [k[r].mean() for r in REGIONS]))}
    return out


def e1c_heatmap(s10, wt, rt, params, consume):
    """时空收益热图：价格时段 × 区域的成本改进（local→perfect 分解）。"""
    local_h = pd.read_csv(OUT_R.parent / "baseline" / "local_hourly.csv")
    greedy_h = pd.read_csv(OUT_R.parent / "baseline" / "greedy_hourly.csv")
    rt_p = rt[["Hour", "Region", "PricePeriod"]]
    m1 = local_h.merge(rt_p, on=["Hour", "Region"])
    m2 = greedy_h.merge(rt_p, on=["Hour", "Region"])
    g1 = m1.groupby(["Region", "PricePeriod"])["GridPurchase_MW"].sum()
    g2 = m2.groupby(["Region", "PricePeriod"])["GridPurchase_MW"].sum()
    price = rt.groupby(["Region", "PricePeriod"])[
        "ElectricityPrice_CNY_per_MWh"].mean()
    save = (g1 * price - g2 * price).unstack()[PERIODS] / 1e4
    save.index = [r.replace("Region", "R") for r in save.index]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(save.to_numpy().T, cmap="YlOrRd")
    ax.set_xticks(range(6), save.index)
    ax.set_yticks(range(3), PERIODS)
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{save.to_numpy().T[i, j]:.0f}", ha="center",
                    va="center", fontsize=8)
    ax.set_title("E1c 错峰收益分解（万元）：价格时段 × 区域（local→greedy）")
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figE1c_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return save.to_dict()


def e1d_perfectness_curve(s10, s12, wt, rt, params, consume, q, act):
    """预测完美度-收益曲线：实际需求混入 σ∈{0,25,50,75,100}% 噪声。

    噪声越大 → 分位数预留失真 → 超容率上升；曲线饱和 ⇒ 结构套利主导。
    """
    out = []
    cap = params["cap"]
    for sigma in (0.0, 0.25, 0.50, 0.75, 1.00):
        rng = np.random.default_rng(42)
        noisy = {}
        for r in REGIONS:
            a = act[r].copy()
            noise = rng.normal(0, sigma * max(a.max(), 1.0), len(a))
            noisy[r] = np.clip(a + noise, 0, None)
        k = {}
        for r in REGIONS:
            k[r] = np.clip(1 - noisy[r] / cap[r], 0.0, 0.95)
        sc = s12.schedule_with_kappa(wt, cap, k)
        m = s10.evaluate_schedule(sc, wt, rt, params, consume)[0]
        out.append({"sigma": sigma,
                    "cost_wan": m["cost_wan"],
                    "viol_h": m["viol_h"],
                    "viol_rate_pct": m["viol_h"] / (HOURS_TOTAL * 6) * 100})
    df = pd.DataFrame(out)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(df["sigma"] * 100, df["cost_wan"] / 1e4, "o-", color="#16a085",
             label="购电成本（亿元）")
    ax1.set_xlabel("预测噪声 σ（% 相对需求水平）")
    ax1.set_ylabel("成本（亿元）")
    ax2 = ax1.twinx()
    ax2.plot(df["sigma"] * 100, df["viol_rate_pct"], "s--", color="#c0392b",
             label="超容率（%）")
    ax2.set_ylabel("超容率 %")
    ax1.set_title("E1d 预测完美度-收益曲线（σ=0 为完美预测）")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figE1d_curve.png", bbox_inches="tight")
    plt.close(fig)
    return df.to_dict("records")


def e1e_eta_bound():
    """η 理论下界：任务侧 η≈0（D1 实证）→ 预测精度对调度收益的贡献上界小。

    论证链: 白噪声(lag1=0.011) → 可预测性 η≈0 → 任意预测器 MSE ≥ 噪声方差
    → 分位数预留误差下界 ≈ 噪声水平 → 预留代价 ~0.07%（E1 实证）→
    预测改进（完美化）收益上界 = 消除 3.5% 超容风险（E1 实证）。
    """
    return {
        "eta_task": 0.011,
        "bound": "η≈0 → MSE下界=白噪声方差 → 预测改进空间≈0 → 调度收益上界"
                 "= 预留代价(0.073pp成本) + 设计内超容风险(3.5%)",
        "empirical": "E1: 完美vs分布成本差0.073pp<5pp；竞技榜任务侧143/144拒",
        "conclusion": "白噪声下将算力投入复杂预测为负收益；分布刻画+预留闭环为正确范式",
    }


def bottleneck_analysis(s10, wt):
    """greedy 延后任务分布（区域×时段）→ M3 构造启发式输入。"""
    local = pd.read_csv(OUT_R.parent / "baseline" / "local_schedule.csv")
    greedy = pd.read_csv(OUT_R.parent / "baseline" / "greedy_schedule.csv")
    m = local.merge(greedy, on="TaskID", suffixes=("_l", "_g"))
    m = m.merge(wt[["TaskID", "TaskType", "GPU_Demand", "dur_h"]], on="TaskID")
    m["delay"] = m["StartHour_g"] - m["StartHour_l"]
    delayed = m[m.delay > 0]
    delayed["hod"] = delayed["StartHour_g"] % 24
    delayed["period"] = delayed["hod"].map(
        lambda h: next(p for p, hs in PERIOD_HOURS.items() if h in hs))
    by_region = delayed.groupby("Region_g")["delay"].agg(
        ["count", "median", "sum"]).to_dict("index")
    by_region = {r: {k: float(v) for k, v in d.items()}
                 for r, d in by_region.items()}
    by_period = delayed["period"].value_counts().to_dict()
    summary = {
        "n_delayed": int(len(delayed)),
        "n_delayed_pct": float(len(delayed) / len(wt) * 100),
        "delay_hours_median": float(delayed["delay"].median()),
        "delay_hours_max": int(delayed["delay"].max()),
        "by_region": by_region,
        "by_period": by_period,
        "top_types": delayed["TaskType"].value_counts().to_dict(),
        "heuristic_input": "延后集中在 E/F 超容时段附近 → M3 构造层优先迁移方向",
    }
    return summary


def plot_e1a(r1: dict) -> None:
    base = json.loads((OUT_R.parent / "baseline" / "baseline_metrics.json")
                      .read_text(encoding="utf-8"))
    labels = ["错峰\n(shift_only)", "迁移\n(migrate_only)", "基线\n(local)"]
    costs = [r1["shift_only_cost_wan"], r1["migrate_only_cost_wan"],
             base["local"]["cost_wan"]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, [c / 1e4 for c in costs],
                  color=["#95a5a6", "#16a085", "#c0392b"])
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c/1e4:.2f}亿", ha="center", va="bottom")
    ax.set_ylabel("购电成本（亿元）")
    ax.set_title("E1a 时空套利分解：迁移省 4.6%，错峰省 0.001%"
                 " → Q2 迁移为主")
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figE1a_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    s10, s12, wt, rt, params, consume = load_ctx()
    q, act = s12.load_region_quantiles()

    r1 = e1a_shift_vs_migrate(s10, wt, rt, params, consume)
    r2 = e1b_reserve_tiers(s10, s12, wt, rt, params, consume, q, act)
    r3 = e1c_heatmap(s10, wt, rt, params, consume)
    r4 = e1d_perfectness_curve(s10, s12, wt, rt, params, consume, q, act)
    r5 = e1e_eta_bound()
    r6 = bottleneck_analysis(s10, wt)
    plot_e1a(r1)

    e1m = {"e1a": r1, "e1b": r2, "e1c": r3, "e1d": r4, "e1e": r5,
           "bottleneck": r6}
    with open(OUT_R / "e1_mechanism.json", "w", encoding="utf-8") as f:
        json.dump(e1m, f, ensure_ascii=False, indent=2)
    with open(OUT_R / "bottleneck.json", "w", encoding="utf-8") as f:
        json.dump(r6, f, ensure_ascii=False, indent=2)

    print(json.dumps(e1m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
