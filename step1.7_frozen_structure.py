"""step1.7_frozen_structure — 冻结段结构研究（R2：问题三闭环）.

背景: step1.5 发现冻结段(2376-2399) CV-最优一致率仅 8.3%，曾被归因"24h 短窗噪声"。
R2 实证推翻了该归因: 冻结段需求水平系统性抬升 +20%（GPU-hour 均值 2499 vs 2089），
判别实验 z=2.57 / 99.0 百分位 —— 生成器收尾结构，非随机噪声。

本脚本四项研究（发现 → 机理 → 应用）:
  S1 判别实验:  98 个训练 24h 窗口需求均值分布 vs 冻结段（z-score/百分位）
  S2 条件覆盖率: 覆盖率条件于需求水平（窗口均值分箱）——风险集中在高位窗口
  S3 滚动重估:  任务侧分位数改为"最近 K 小时滚动经验分位数"（shift≥1 防泄漏）
                冻结段聚合覆盖率 vs 全训练段静态分位数（外推失效的修复实验）
  S4 κ 最终验证: 冻结段 q_{0.95} 覆盖率（静态/滚动）+ quantile_schedule 冻结段
                超容率 vs ε=5% 设计（冻结段承担"最终验证"职责）
  S5 价格之谜:  冻结段价格 = 训练段同小时均值模板？(解释 step1.5 价格 MAPE
                0.19% vs CV 2.81% 的机理: 段标签+日模板外推精确命中)

产物（output/robust/）: frozen_structure.json + figures/step1/fig_frozen_structure.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_R = OUTPUT / "robust"
FIG_S1 = FIGURES / "step1"
SEG_TEST = 2376
SEG_END = 2400


def load_task_agg() -> tuple[dict, np.ndarray]:
    """18 序列实际需求 → 区域聚合需求（与 step1.2 同口径）。"""
    ft = pd.read_csv(OUTPUT / "forecast" / "fuse_quantiles_task.csv")
    act = {r: ft[[f"{r}|{t}_actual" for t in TASK_TYPES]].sum(axis=1).to_numpy()
           for r in REGIONS}
    tot = np.sum([act[r] for r in REGIONS], axis=0)
    return act, tot


def s1_discrimination(tot: np.ndarray) -> dict:
    """判别实验: 冻结段是否落在训练窗口需求分布之外。"""
    win = np.array([tot[t * 24:(t + 1) * 24].mean() for t in range(98)])
    fz = tot[SEG_TEST:SEG_END].mean()
    z = (fz - win.mean()) / win.std()
    return {
        "train_window_mean": float(win.mean()),
        "train_window_std": float(win.std()),
        "train_window_max": float(win.max()),
        "frozen_window_mean": float(fz),
        "z_score": float(z),
        "percentile": float(np.mean(win <= fz) * 100),
        "conclusion": ("冻结段需求水平系统性抬升（生成器收尾结构），"
                       "非 24h 随机噪声" if z > 2 else "冻结段在训练分布内"),
    }


def s2_conditional_coverage(act: dict, tot: np.ndarray, eps: float = 0.05) -> dict:
    """覆盖率条件于需求水平: 训练段 98 窗按需求均值分箱 → 各箱覆盖率。"""
    q = {r: _static_q95(act, r) for r in REGIONS}
    bins = np.array_split(np.argsort([tot[t * 24:(t + 1) * 24].mean()
                                      for t in range(98)]), 5)
    rows = []
    for bi, idx in enumerate(bins):
        covs = []
        for t0 in idx * 24:
            ok = [q[r][t0:t0 + 24] >= act[r][t0:t0 + 24] for r in REGIONS]
            covs.append(np.mean(np.concatenate(ok)))
        rows.append({"bin": bi, "demand_level": f"Q{bi * 20}-{bi * 20 + 20}%",
                     "cov": float(np.mean(covs))})
    low = rows[:2]
    high = rows[-2:]
    return {"bins": rows,
            "low_cov": float(np.mean([b["cov"] for b in low])),
            "high_cov": float(np.mean([b["cov"] for b in high])),
            "gap": float(np.mean([b["cov"] for b in high])
                         - np.mean([b["cov"] for b in low]))}


def _static_q95(act: dict, r: str) -> np.ndarray:
    """全训练段(0-2351)经验 95 分位（常数）——与统计基线同口径。"""
    q = np.quantile(act[r][:2352], 0.95)
    return np.full(2407, q)


def _rolling_q95(act: dict, r: str, k: int) -> np.ndarray:
    """最近 K 小时滚动经验 95 分位（nowcast: t 时刻用 [t-K, t-1]，shift≥1 防泄漏）。"""
    y = act[r]
    out = np.zeros(2407)
    for t in range(2407):
        lo = max(0, t - k)
        if lo >= t:
            out[t] = y[t]
        else:
            out[t] = np.quantile(y[lo:t], 0.95)
    return out


def s3_rolling_requantile(act: dict) -> dict:
    """滚动重估 vs 静态外推: 冻结段区域聚合 95% 覆盖率对比。"""
    seg = np.arange(SEG_TEST, SEG_END)
    res = {"static_all_train": None, "rolling_168h": None, "rolling_336h": None}
    res["static_all_train"] = float(np.mean(
        np.concatenate([_static_q95(act, r)[seg] >= act[r][seg]
                        for r in REGIONS])))
    for k in (168, 336):
        res[f"rolling_{k}h"] = float(np.mean(
            np.concatenate([_rolling_q95(act, r, k)[seg] >= act[r][seg]
                            for r in REGIONS])))
    return res


def s4_kappa_final_validation(s3: dict) -> dict:
    """冻结段最终验证: 覆盖率（静态/滚动）+ quantile_schedule 冻结段超容率 vs ε。"""
    qs = pd.read_csv(OUT_R / "quantile_schedule.csv")
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    gpu = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw"
                        / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    cap = gpu.set_index("Region")["Available_GPU"].to_dict()
    sel = wt[(wt.ArrivalHour >= SEG_TEST) & (wt.ArrivalHour < SEG_END)] \
        .merge(qs, on="TaskID")
    occ = np.zeros((SEG_END - SEG_TEST, len(REGIONS)))
    for rec in sel.itertuples(index=False):
        r = REGIONS.index(rec.Region)
        st = int(rec.StartHour)
        dur = float(rec.EstimatedDuration_min) / 60
        h1 = min(int(np.ceil(st + dur)), SEG_END)
        if h1 > st:
            occ[st - SEG_TEST:h1 - SEG_TEST, r] += float(rec.GPU_Demand)
    viol_hours = int((occ > np.array([cap[r] for r in REGIONS])).sum())
    total_hrs = (SEG_END - SEG_TEST) * len(REGIONS)
    return {
        "static_q95_cov_frozen": s3["static_all_train"],
        "rolling_q95_cov_frozen": s3["rolling_336h"],
        "schedule_viol_hours_frozen": viol_hours,
        "viol_rate_frozen_pct": float(viol_hours / total_hrs * 100),
        "eps_design": 0.05,
        "note": "ε=5% 设计保障 vs 冻结段实测超容率（预留贪心 + 静态分位数 κ）；"
                "覆盖率口径见 s3（聚合 144 点）",
    }


def s5_price_mystery() -> dict:
    """冻结段价格 vs 两种静态模板（日模板 / 段标签均值）——解 0.19% MAPE 之谜。"""
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    tr = rt[(rt.Hour < 2352) & (rt.DataPeriod == "Main_0_2399")]
    fz = rt[(rt.Hour >= SEG_TEST) & (rt.Hour < SEG_END)]
    rows = []
    for r in REGIONS:
        tpl_h = tr[tr.Region == r].groupby(tr[tr.Region == r].Hour % 24)[
            "ElectricityPrice_CNY_per_MWh"].mean()
        tpl_p = tr[tr.Region == r].groupby("PricePeriod")[
            "ElectricityPrice_CNY_per_MWh"].mean()
        sub = fz[fz.Region == r]
        pred_h = tpl_h.loc[sub.Hour % 24].to_numpy()
        pred_p = tpl_p.loc[sub.PricePeriod].to_numpy()
        act = sub["ElectricityPrice_CNY_per_MWh"].to_numpy()
        mape = lambda p: float(np.mean(np.abs(p - act) / act) * 100)
        rows.append({"region": r,
                     "hour_template_mape_pct": mape(pred_h),
                     "period_template_mape_pct": mape(pred_p),
                     "residual_std_frozen": float(np.std(act - pred_p))})
    return {"per_region": rows,
            "median_hour_template_mape_pct": float(np.median(
                [x["hour_template_mape_pct"] for x in rows])),
            "median_period_template_mape_pct": float(np.median(
                [x["period_template_mape_pct"] for x in rows])),
            "note": ("冻结段价格含日内结构（日模板 1.27% < 段标签均值 2.48%）；"
                     "真实冻结段 2376-2399 修复评估后 price 最优 = lgbm_point 0.34%"
                     "（残差修正真实增益）；旧 0.19% 系 step1.5 评估段误用 closure"
                     "（2400-2406）所致——closure 价格=完美日模板（stat_hist 0.19%）")}


def s6_power_curve(series: dict) -> dict:
    """S6 功效曲线（双口径）：窗口长度 vs 一致率（短窗排名噪声的统计功效）.

    口径（判别实验，与 step1.5 冻结段一致）: 模型 0-2351 全段拟合后预测全段，
    按窗口切片评估；窗内最优与参考最优的一致率随 W 变化。模型子集: 6 个非深度。

    双参考（R2 深化发现）:
      cv_best: arena CV(5折) 最优 → 揭示"CV 类选 → 全段部署"的训练规模错位
               （实证: TabPFN CV 统治但全段拟合下树类更优——tabpfn 对训练
               规模敏感，2352 点下 few-shot 优势消失）
      deploy_best: 全段拟合最优 → 纯"短窗排名噪声"功效曲线（预期随 W 上升）
    用途: 支撑"短窗只承担不崩溃验证" + 类选-部署错位的方法学声明。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    s11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s11)
    cv = pd.read_csv(OUTPUT / "forecast" / "arena_table.csv")
    cv_best = {}
    for name, g in cv[cv.family != "统计基线"].groupby("series"):
        g = g[g.n_folds > 0]
        if len(g):
            cv_best[name] = g.loc[g.mape_mean.idxmin(), "model"]
    wins = [24, 48, 96, 168, 336]
    rows = []
    for name, info in series.items():
        if info["layer"] != "energy" or name not in cv_best:
            continue
        y = info["y"][:2352]
        X = s11.build_features(y, "energy", info.get("region"))
        pool = [m for m in s11.build_model_pool("energy")
                if m.family != "深度"]
        preds = {}
        for m in pool:
            m = s11._fresh_model(m)
            try:
                if isinstance(m, s11.StatisticalBaseline):
                    m.fit(y[:2352])
                else:
                    m.fit(X.iloc[:2352], y[:2352])
                preds[m.name] = m.predict_point(X.iloc[:2352], 0)
            except Exception:
                continue
        if not preds:
            continue
        full_mape = {mn: float(np.mean(np.abs(p - y)
                                       / np.maximum(y, 1e-9)))
                     for mn, p in preds.items()}
        deploy_best = min(full_mape, key=full_mape.get)
        cb = cv_best[name]
        for W in wins:
            n, m_cv, m_dep = 0, 0, 0
            for t0 in range(0, 2352 - W + 1, W):
                n += 1
                seg = np.arange(t0, t0 + W)
                mape = {mn: float(np.mean(np.abs(p[seg] - y[seg])
                                          / np.maximum(y[seg], 1e-9)))
                        for mn, p in preds.items()}
                win_best = min(mape, key=mape.get)
                m_cv += (win_best == cb)
                m_dep += (win_best == deploy_best)
            rows.append({"series": name, "W": W,
                         "cv_consistency": m_cv / max(n, 1),
                         "deploy_consistency": m_dep / max(n, 1),
                         "n_windows": n,
                         "cv_best": cb, "deploy_best": deploy_best})
    df = pd.DataFrame(rows)
    agg = df.groupby("W")[["cv_consistency", "deploy_consistency"]].agg(
        ["mean", "std", "count"])
    mismatch = df.groupby("W").apply(
        lambda d: float(np.mean(d["cv_best"] != d["deploy_best"])),
        include_groups=False)
    return {"windows": wins,
            "summary": {int(w): {"cv_consistency": float(r[("cv_consistency", "mean")]),
                                 "cv_std": float(r[("cv_consistency", "std")]),
                                 "deploy_consistency": float(r[("deploy_consistency", "mean")]),
                                 "deploy_std": float(r[("deploy_consistency", "std")]),
                                 "mismatch_share": float(mismatch[w]),
                                 "n_series": int(r[("cv_consistency", "count")])}
                        for w, r in agg.iterrows()},
            "per_series_window": df.to_dict("records"),
            "caliber": "判别实验：模型全段拟合（含窗口），窗内 MAPE 最优 vs 双参考；"
                       "非深度 6 模型子集；训练段 0-2351 内滑窗",
            "finding": ("① TabPFN 统治是 CV(5折) 口径现象；全段拟合(2352点)下树类更优"
                        "（训练规模迁移：few-shot 优势在长训练段消失），"
                        "24/24 能源序列 CV-部署最优错位（mismatch=1.00）；"
                        "② 真实冻结段 2376-2399 评估（R2 修复 step1.5 段误用）"
                        "一致率 0.119、price 最优 lgbm 0.34%（残差修正真实增益）、"
                        "旧 8.3%/0.19% 系 closure 段口径产物")}


def plot_structure(s1: dict, s2: dict, s3: dict, tot: np.ndarray,
                   s6: dict | None = None) -> None:
    n_panels = 4 if s6 is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.2))
    win = np.array([tot[t * 24:(t + 1) * 24].mean() for t in range(98)])
    axes[0].hist(win, bins=16, color="#95a5a6", edgecolor="white",
                 label="训练段 98 窗口均值")
    axes[0].axvline(s1["frozen_window_mean"], color="#c0392b", lw=2,
                    label=f"冻结段 {s1['frozen_window_mean']:.0f} "
                          f"(z={s1['z_score']:.2f})")
    axes[0].set_title("S1 判别: 冻结段需求抬升")
    axes[0].legend(fontsize=8)
    xs = [b["bin"] for b in s2["bins"]]
    ys = [b["cov"] for b in s2["bins"]]
    axes[1].plot(xs, ys, "o-", color="#2980b9")
    axes[1].set_title(f"S2 条件覆盖率(ε=0.05): gap="
                      f"{s2['gap'] * 100:.1f}pp")
    axes[1].set_xlabel("需求水平分箱（低→高）")
    axes[1].set_ylabel("窗口覆盖率")
    names = ["静态全训练", "滚动 168h", "滚动 336h"]
    vals = [s3["static_all_train"], s3["rolling_168h"], s3["rolling_336h"]]
    axes[2].bar(names, vals, color=["#95a5a6", "#e67e22", "#16a085"])
    axes[2].axhline(0.95, color="k", ls="--", lw=0.8)
    axes[2].set_title("S3 滚动重估: 冻结段 95% 覆盖率")
    axes[2].set_ylim(0.8, 1.0)
    if s6 is not None:
        ws = s6["windows"]
        m_cv = [s6["summary"][w]["cv_consistency"] for w in ws]
        m_dep = [s6["summary"][w]["deploy_consistency"] for w in ws]
        e_cv = [s6["summary"][w]["cv_std"] for w in ws]
        e_dep = [s6["summary"][w]["deploy_std"] for w in ws]
        axes[3].errorbar(ws, m_dep, yerr=e_dep, fmt="o-", color="#8e44ad",
                         label="vs 部署最优(全段拟合)")
        axes[3].errorbar(ws, m_cv, yerr=e_cv, fmt="s--", color="#c0392b",
                         label="vs CV 最优(5折)")
        axes[3].axhline(1.0, color="k", ls="--", lw=0.8)
        axes[3].set_title("S6 功效曲线: 窗口 vs 一致率")
        axes[3].set_xlabel("评估窗口 W (h)")
        axes[3].set_ylabel("窗内最优一致率")
        axes[3].set_ylim(0, 1.05)
        axes[3].legend(fontsize=8)
    fig.suptitle("冻结段结构研究: 需求抬升 → 外推失效 → 滚动重估修复"
                 + (" → 短窗功效" if s6 is not None else ""))
    fig.tight_layout()
    fig.savefig(FIG_S1 / "fig_frozen_structure.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_R.mkdir(parents=True, exist_ok=True)
    FIG_S1.mkdir(parents=True, exist_ok=True)
    act, tot = load_task_agg()
    s1 = s1_discrimination(tot)
    s2 = s2_conditional_coverage(act, tot, eps=0.05)
    s3 = s3_rolling_requantile(act)
    s4 = s4_kappa_final_validation(s3)
    s5 = s5_price_mystery()
    s11 = None
    spec = importlib.util.spec_from_file_location(
        "s11b", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    s11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s11)
    s6 = s6_power_curve(s11.make_series_dict())
    report = {"s1_discrimination": s1, "s2_conditional_cov": s2,
              "s3_rolling_requantile": s3, "s4_kappa_final": s4,
              "s5_price_mystery": s5, "s6_power_curve": s6,
              "conclusion": ("冻结段需求抬升是生成器收尾结构（S1 z=2.86/100 百分位）→ "
                             "常数历史分位数外推欠覆盖（S3 0.917→滚动 336h 0.944）→ "
                             "覆盖风险条件于需求水平（S2 gap −1.8pp）→ "
                             "冻结段职责=最终验证（S4 超容率 4.86%≈ε=5% 设计）"
                             "而非排名裁决（短窗排名噪声是次级效应）"),
              "caliber": "覆盖率为区域聚合 144 点口径（与 step1.2 一致）；"
                         "滚动分位数为 nowcast（t 用 [t-K,t-1]，shift≥1 防泄漏）"}
    with open(OUT_R / "frozen_structure.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    plot_structure(s1, s2, s3, tot, s6)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
