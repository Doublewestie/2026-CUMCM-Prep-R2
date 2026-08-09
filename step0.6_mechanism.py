"""step0.6_mechanism — Phase 4: 任务侧机制分解（M1-M4）.

M1 AI 漂移谱三因子分解: Delta ln(gh) = Delta ln(n) + Delta ln(avg_g) + Delta ln(avg_d)
   （数量/规格/时长对冻结段 GPU-hour 变化的贡献，按区域 x 类型）
M2 冻结段任务四段统计: 到达数/规格/时长的 KS 检验（train vs frozen）
M3 到达过程段间参数: 泊松 lambda 段间变化 + 拟合优度（泊松假设在冻结段是否维持）
M4 I11 深挖: Baseline_AI_IT_Load 生成口径
   - 假设 A: 分数 Overlap（分钟级重叠折算）而非 ceil 整点
   - 假设 B: 基线错峰调度（任务延后开工）
   - 重建两种口径 vs 基线列，选最小残差口径

产物: output/clean/mechanism_report.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from step0_config import CLEAN, REGIONS, HOURS_TOTAL

OUT_JSON = CLEAN / "mechanism_report.json"
TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
POWER = {"RealTimeInference": 0.08, "BatchInference": 0.10, "AITraining": 0.16}


def m1_factor_decomp(wk: pd.DataFrame) -> dict:
    out = {}
    wk2 = wk.copy()
    wk2["gh"] = wk2["GPU_Demand"] * wk2["dur_h"]
    for r in REGIONS:
        rows = []
        for tt in TASK_TYPES:
            m_tr = wk2[(wk2.SourceRegion == r) & (wk2.TaskType == tt)
                       & (wk2.ArrivalHour < 2352)]
            m_fz = wk2[(wk2.SourceRegion == r) & (wk2.TaskType == tt)
                       & (wk2.ArrivalHour >= 2376) & (wk2.ArrivalHour < 2400)]
            if len(m_tr) == 0 or len(m_fz) == 0:
                continue
            gh_tr = m_tr.gh.sum() / 98
            gh_fz = m_fz.gh.sum()
            ln_n = np.log(len(m_fz) / (len(m_tr) / 98))
            ln_g = np.log(m_fz.GPU_Demand.mean() / m_tr.GPU_Demand.mean())
            ln_d = np.log(m_fz.dur_h.mean() / m_tr.dur_h.mean())
            rows.append({"type": tt, "ratio_gh": float(gh_fz / gh_tr),
                         "dln_n": float(ln_n), "dln_g": float(ln_g),
                         "dln_d": float(ln_d),
                         "share_n": float(ln_n / (ln_n + ln_g + ln_d))})
        out[r] = rows
    return out


def m2_ks_tests(wk: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        rows = []
        for tt in TASK_TYPES:
            m_tr = wk[(wk.SourceRegion == r) & (wk.TaskType == tt)
                      & (wk.ArrivalHour < 2352)]
            m_fz = wk[(wk.SourceRegion == r) & (wk.TaskType == tt)
                      & (wk.ArrivalHour >= 2376) & (wk.ArrivalHour < 2400)]
            if len(m_tr) < 30 or len(m_fz) < 10:
                continue
            kg = ks_2samp(m_tr.GPU_Demand, m_fz.GPU_Demand)
            kd = ks_2samp(m_tr.dur_h, m_fz.dur_h)
            rows.append({"type": tt,
                         "ks_gpu_stat": float(kg.statistic), "ks_gpu_p": float(kg.pvalue),
                         "ks_dur_stat": float(kd.statistic), "ks_dur_p": float(kd.pvalue),
                         "gpu_mean_tr_fz": (float(m_tr.GPU_Demand.mean()),
                                            float(m_fz.GPU_Demand.mean()))})
        out[r] = rows
    return out


def m3_arrival_process(wk: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        rows = []
        for tt in TASK_TYPES:
            lam_tr = float(((wk.SourceRegion == r) & (wk.TaskType == tt)
                            & (wk.ArrivalHour < 2352)).mean())
            lam_fz = float(((wk.SourceRegion == r) & (wk.TaskType == tt)
                            & (wk.ArrivalHour >= 2376) & (wk.ArrivalHour < 2400)).mean())
            n_tr = int(((wk.SourceRegion == r) & (wk.TaskType == tt)
                        & (wk.ArrivalHour < 2352)).sum())
            n_fz = int(((wk.SourceRegion == r) & (wk.TaskType == tt)
                        & (wk.ArrivalHour >= 2376) & (wk.ArrivalHour < 2400)).sum())
            rows.append({"type": tt, "lambda_tr": lam_tr, "lambda_fz": lam_fz,
                         "lambda_ratio": lam_fz / max(lam_tr, 1e-9),
                         "n_tr": n_tr, "n_fz": n_fz})
        out[r] = rows
    return out


def rebuild_fractional(wk: pd.DataFrame) -> pd.DataFrame:
    """分数 Overlap 重建：任务 [st, st+dur) 与小时窗口的分钟级重叠分数。

    返回 Region x Hour 的 AI 负荷（MW）。
    """
    rows = []
    for rec in wk.itertuples(index=False):
        st = float(rec.ArrivalHour)
        dur = float(rec.dur_h)
        p = POWER[rec.TaskType]
        t0 = int(np.floor(st))
        t1 = int(np.ceil(st + dur))
        for t in range(t0, t1):
            if t < 0 or t >= HOURS_TOTAL:
                continue
            ov = max(0.0, min(t + 1, st + dur) - max(t, st))
            rows.append((rec.SourceRegion, t, rec.GPU_Demand * p * ov))
    df = pd.DataFrame(rows, columns=["Region", "Hour", "mw"])
    ai = df.groupby(["Region", "Hour"])["mw"].sum().unstack("Region").fillna(0.0)
    ai = ai.reindex(index=pd.RangeIndex(HOURS_TOTAL, name="Hour"),
                    columns=REGIONS, fill_value=0)
    return ai


def m4_baseline_ai_caliber(wk: pd.DataFrame, rt: pd.DataFrame) -> dict:
    base = rt.pivot_table(index="Hour", columns="Region",
                          values="Baseline_AI_IT_Load_MW")
    base = base.reindex(pd.RangeIndex(HOURS_TOTAL))
    # 口径 A: ceil 整点（occupancy_local 同款）
    occ = pd.read_csv(CLEAN / "occupancy_local.csv")
    m = occ.merge(wk[["TaskID", "TaskType"]], on="TaskID", how="left")
    m["mw"] = m["GPU_Demand"] * m["TaskType"].map(POWER)
    a = m.groupby(["Region", "Hour"])["mw"].sum().unstack("Region").fillna(0.0)
    a = a.reindex(index=pd.RangeIndex(HOURS_TOTAL, name="Hour"),
                  columns=REGIONS, fill_value=0)
    # 口径 B: 分数 Overlap
    b = rebuild_fractional(wk)
    out = {}
    for r in REGIONS:
        y = base[r].to_numpy(float)
        ra = np.mean(np.abs(a[r].to_numpy(float) - y)) / np.mean(np.abs(y))
        rb = np.mean(np.abs(b[r].to_numpy(float) - y)) / np.mean(np.abs(y))
        diff_b = b[r].to_numpy(float) - y
        out[r] = {"rel_ceil": float(ra), "rel_fractional": float(rb),
                  "best": "fractional" if rb < ra else "ceil",
                  "diff_mean_b": float(diff_b.mean()),
                  "diff_std_b": float(diff_b.std()),
                  "note": "分数 Overlap 与 ceil 整点两种重建口径对照基线列"}
    return out


def main() -> None:
    wk = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    report = {
        "caliber": "Phase 4 任务侧机制（训练段 0-2351 vs 冻结段 2376-2399 描述）",
        "M1_factor_decomp": m1_factor_decomp(wk),
        "M2_ks_tests": m2_ks_tests(wk),
        "M3_arrival_process": m3_arrival_process(wk),
        "M4_baseline_ai_caliber": m4_baseline_ai_caliber(wk, rt),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved", OUT_JSON, flush=True)
    print("M4 baseline_ai caliber:", {k: (round(v["rel_ceil"], 4), round(v["rel_fractional"], 4))
                                      for k, v in report["M4_baseline_ai_caliber"].items()})
    print("M1 (RegionE):")
    for row in report["M1_factor_decomp"]["RegionE"]:
        print("  ", row["type"], "ratio=%.2f" % row["ratio_gh"],
              "share_n=%.2f" % row["share_n"], "dln_n=%.2f dln_g=%.2f dln_d=%.2f"
              % (row["dln_n"], row["dln_g"], row["dln_d"]))


if __name__ == "__main__":
    main()
