"""step2.0_construct — Q2 构造层：白名单 + 价差/碳差双排序强启发式初始可行解.

方法（PLAN_details §7.1 构造层）:
  迁移决策 = 白名单可达集 × 区域评分（均价排序 + 碳强度排序 加权 λ=0.5）
    × 容量贪心（可用即放）；RT 到达即开工固定；st=Arrival（错峰留给 NSGA-II）
  评估四目标（模板口径，与 Q1 全链可比）:
    Cost₂  运行成本（万元）       CE₂   碳排放（tCO2）
    Lat    GPU-hours 加权时延(ms)  NU    新能源利用率（%）
  口径: 消纳 U=c_r(h)·D 模板（基线口径）；S=min(SellLimit, W−U)；G=D−U

产物（output/q2/）: construct_schedule.csv + construct_metrics.json +
  figures/step2/fig_q2_construct.png（四目标 vs 基线对照）
"""
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPE_SHORT, \
    HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q2 = OUTPUT / "q2"
FIG_S2 = FIGURES / "step2"
SEG_TRAIN = 2352
LAMBDA = 0.5          # 价差 vs 碳差 权重（NSGA-II 进化对象之一）


def load_ctx():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    return s10


def region_rank(rt: pd.DataFrame, wl: dict, s10, params: dict,
                lam: float = LAMBDA) -> dict:
    """区域评分表：均价/碳强度训练段均值 → 排序（低价低碳靠前）。

    返回 {task_type: {region: rank}}，rank 由 λ·价差排序 + (1−λ)·碳差排序 合并。
    """
    tr = rt[rt.Hour < SEG_TRAIN]
    price = tr.groupby("Region")["ElectricityPrice_CNY_per_MWh"].mean()
    carbon = tr.groupby("Region")["CarbonIntensity_tCO2_per_MWh"].mean()
    ranks = {}
    for tt in ("RealTimeInference", "BatchInference", "AITraining"):
        p_rank = {r: i for i, r in enumerate(price.sort_values().index)}
        c_rank = {r: i for i, r in enumerate(carbon.sort_values().index)}
        rk = {}
        for r in REGIONS:
            if r in wl.get((tt, r), [r]):
                rk[r] = lam * p_rank[r] + (1 - lam) * c_rank[r]
        ranks[tt] = dict(sorted(rk.items(), key=lambda kv: kv[1]))
    return ranks


_HP_CACHE = None


def _hourly_price(rt: pd.DataFrame) -> pd.DataFrame:
    """区域逐时电价（错峰价格感知用，模块级缓存）。"""
    global _HP_CACHE
    if _HP_CACHE is None:
        _HP_CACHE = rt.pivot_table(index="Hour", columns="Region",
                                   values="ElectricityPrice_CNY_per_MWh",
                                   aggfunc="first").reindex(range(HOURS_TOTAL))
    return _HP_CACHE


def schedule_constructive(wt: pd.DataFrame, rt: pd.DataFrame, s10,
                          params: dict, policy: tuple | None = None,
                          whitelist_override: pd.DataFrame | None = None,
                          capacity_factor: np.ndarray | None = None
                          ) -> pd.DataFrame:
    """构造性调度：RT 固定到达即开工；BI/AT 按白名单+区域评分迁移（容量贪心）.

    policy = (mig_gpu_min, mig_dur_min, shift_BI, shift_AT, headroom)
      mig_gpu_min  迁移最小 GPU 规格阈值（0=全迁；大值=少迁，时延-成本权衡）
      mig_dur_min  迁移最小时长阈值（h）
      shift_BI     BI 最大错峰小时（0=不偏移；>0 时窗口内价格感知选点）
      shift_AT     AT 最大错峰小时
      headroom     容量松弛比例（占用 ≤ (1−headroom)·cap，默认 0）
    默认 (0,0,0,0,0) = 全迁移构造解（E1a migrate_only 口径）。
    """
    mig_gpu, mig_dur, sh_bi, sh_at, headroom = (policy if policy is not None
                                                else (0.0, 0.0, 0.0, 0.0, 0.0))
    wl = {(r.TaskType, r.SourceRegion): r.Reachable.split("|")
          for r in (whitelist_override
                    if whitelist_override is not None
                    else pd.read_csv(CLEAN / "whitelist.csv"))
          .itertuples(index=False)}
    ranks = region_rank(rt, wl, s10, params, LAMBDA)
    price = _hourly_price(rt).to_numpy(dtype=float)   # (2407, 6) numpy
    cap_arr = np.array([params["cap"][r] for r in REGIONS], dtype=float)
    cap_arr = cap_arr * (1 - headroom)
    if capacity_factor is not None:
        # 逐时容量因子（κ 预留）: (2407,6) → 有效容量矩阵（仅弹性任务生效）
        cap_eff = cap_arr[None, :] * capacity_factor
    else:
        cap_eff = np.broadcast_to(cap_arr, (HOURS_TOTAL, len(REGIONS)))
    cap_rt = cap_arr                                # RT 刚性：不受预留压缩
    occ = np.zeros((HOURS_TOTAL, len(REGIONS)))
    shift_max = {"RealTimeInference": 0.0,
                 "BatchInference": sh_bi, "AITraining": sh_at}
    rows = []
    for rec in wt.sort_values(["Priority", "ArrivalHour"],
                              ascending=[False, True]).itertuples(index=False):
        g = float(rec.GPU_Demand)
        dur = float(rec.dur_h)
        if rec.TaskType == "RealTimeInference":
            r = REGIONS.index(rec.SourceRegion)
            st = int(rec.ArrivalHour)
        elif g < mig_gpu or dur < mig_dur:
            r = REGIONS.index(rec.SourceRegion)      # 阈值外：本地执行（时延友好）
            st = int(rec.ArrivalHour)
        else:
            cands = ranks[rec.TaskType]
            smax = shift_max[rec.TaskType]
            last = min(int(rec.ArrivalHour + smax), HOURS_TOTAL - 1)
            r = st = None
            for c in cands:
                ri = REGIONS.index(c)
                lo = int(rec.ArrivalHour)
                hi = min(int(rec.LatestFinishHour - dur), last)
                if hi < lo:
                    continue
                if occ[lo:hi + 1, ri].max() + g > cap_eff[lo:hi + 1, ri].max():
                    continue                    # 区域级：窗口内放不下
                for s in range(lo, hi + 1):
                    h1 = min(int(np.ceil(s + dur)), HOURS_TOTAL)
                    if h1 <= s:
                        break
                    if (occ[s:h1, ri] + g - cap_eff[s:h1, ri].max()).max() <= 0:
                        r, st = ri, s
                        break
                if r is not None:
                    break
            if r is None:
                # 容量修复：错峰窗口内放不下 → 全窗口延后扫描（viol=0 硬底线）
                for c in cands:
                    ri = REGIONS.index(c)
                    lo = int(rec.ArrivalHour)
                    hi = min(int(rec.LatestFinishHour - dur), 2405)
                    if hi < lo:
                        continue
                    if occ[lo:hi + 1, ri].max() + g > cap_eff[lo:hi + 1, ri].max():
                        continue
                    for s in range(lo, hi + 1):
                        h1 = min(int(np.ceil(s + dur)), HOURS_TOTAL)
                        if (occ[s:h1, ri] + g - cap_eff[s:h1, ri].max()).max() <= 0:
                            r, st = ri, s
                            break
                    if r is not None:
                        break
            if r is None:
                r = REGIONS.index(rec.SourceRegion)
                st = int(rec.ArrivalHour)
        rows.append((rec.TaskID, REGIONS[r], st))
        h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
        if h1 > st:
            occ[st:h1, r] += g
    return pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])


@lru_cache(maxsize=1)
def _latency_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """时延矩阵 numpy 版（模块级缓存，NSGA-II 每次评估复用）。"""
    lat = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw"
                        / "network_latency.xlsx", sheet_name=0)
    lm = lat.pivot(index="FromRegion", columns="ToRegion",
                   values="NetworkLatency_ms")
    src_idx = np.array([REGIONS.index(r) for r in lm.index])
    dst_idx = np.array([REGIONS.index(r) for r in lm.columns])
    return lm.to_numpy(dtype=float), src_idx, dst_idx


def compute_latency(wt: pd.DataFrame, sched: pd.DataFrame) -> float:
    """GPU-hours 加权时延（ω_i = L(s_i, r_i)，numpy 向量化）。"""
    lm, src_idx, dst_idx = _latency_matrix()
    m = wt.merge(sched, on="TaskID")
    w = m["GPU_Demand"].to_numpy(dtype=float)
    dur = m["dur_h"].to_numpy(dtype=float)
    si = src_idx[m["SourceRegion"].map(lambda x: REGIONS.index(x))]
    di = dst_idx[m["Region"].map(lambda x: REGIONS.index(x))]
    ms = lm[si, di]
    num = (ms * w * dur).sum()
    den = (w * dur).sum()
    return float(num / den)


def evaluate_4obj(wt: pd.DataFrame, rt: pd.DataFrame, sched: pd.DataFrame,
                  s10, params: dict, consume: dict,
                  include_baseline_charge: bool = False) -> dict:
    """四目标评估（模板口径）：Cost₂/CE₂/NU 复用评估器 + Lat 单独计算.

    include_baseline_charge=True（B2）: 对照题目基线口径（充电项计入 NU/弃电）。
    """
    m, _ = s10.evaluate_schedule(sched, wt, rt, params, consume,
                                 include_baseline_charge=include_baseline_charge)
    lat = compute_latency(wt, sched)
    return {"cost_wan": m["cost_wan"], "carbon_t": m["carbon_t"],
            "nu_pct": m["nu_pct"], "latency_ms": lat,
            "curtail_MWh": m["curtail_MWh"], "viol_h": m["viol_h"],
            "caliber": "模板消纳口径（U=c_r(h)·D）；Lat=GPU-hours 加权；"
                       "Q2 无储能"}


def main() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    s10 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    sched = schedule_constructive(wt, rt, s10, params)
    sched.to_csv(OUT_Q2 / "construct_schedule.csv", index=False)
    m4 = evaluate_4obj(wt, rt, sched, s10, params, consume)

    local = json.loads((OUTPUT / "baseline" / "baseline_metrics.json")
                       .read_text(encoding="utf-8"))["local"]
    base_4obj = {"cost_wan": local["cost_wan"], "carbon_t": local["carbon_t"],
                 "nu_pct": local["nu_pct"],
                 "latency_ms": compute_latency(wt, pd.read_csv(
                     OUTPUT / "baseline" / "local_schedule.csv"))}
    report = {"construct": m4, "baseline_local": base_4obj,
              "lambda": LAMBDA,
              "improve": {
                  "cost_pct": (base_4obj["cost_wan"] - m4["cost_wan"])
                              / base_4obj["cost_wan"] * 100,
                  "carbon_pct": (base_4obj["carbon_t"] - m4["carbon_t"])
                                / base_4obj["carbon_t"] * 100,
                  "nu_pp": m4["nu_pct"] - base_4obj["nu_pct"],
                  "latency_ratio": m4["latency_ms"]
                                   / max(base_4obj["latency_ms"], 1e-9)},
              "note": "构造层=白名单+价差/碳差双排序（λ=0.5）+容量贪心；"
                      "RT 固定；错峰留待 NSGA-II；迁移收益预期落在 "
                      "E1a 区间（2-4.75%）内"}
    with open(OUT_Q2 / "construct_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    labels = ["附件基线", "构造解"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, fmt, title in [
            (axes[0], "cost_wan", "%.0f", "运行成本(万元)"),
            (axes[1], "carbon_t", "%.0f", "碳排放(tCO2)"),
            (axes[2], "nu_pct", "%.1f%%", "新能源利用率(%)")]:
        vals = [base_4obj[key], m4[key]]
        ax.bar(labels, vals, color=["#95a5a6", "#e67e22"])
        for b, v in zip(ax.patches, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt % v, ha="center", va="bottom")
        ax.set_title(title)
    fig.suptitle(f"Q2 构造层: 成本 {report['improve']['cost_pct']:.2f}%↓  "
                 f"碳 {report['improve']['carbon_pct']:.2f}%↓  "
                 f"时延 {report['improve']['latency_ratio']:.3f}×")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_construct.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
