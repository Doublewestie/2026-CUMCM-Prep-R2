"""step0.6_spectrum — Phase 2: 漂移/相关/周期/变点全谱（D1-D7）.

D1 四段漂移谱: 25 数值列 x 6 区 x 4 段 -> {mean,p50,std} 漂移（冻结 vs 训练）
D2 全列相关矩阵: 6 区 -> 全时段/train/frozen 三矩阵 + 段间结构差异（top 对）
D3 分 PricePeriod 相关摘要: 段内相关 vs 全时段相关差异（条件结构）
D4 分 DemandResponse 层相关摘要: Low/Medium 层内结构差异
D5 区域间耦合: 同类列跨区相关（找非 W/price 的隐藏耦合）
D6 自相关-周期谱: lag1/24/168 自相关 + 24/168/336 周期模板拟合残差（周期强度）
D7 变点检测: 训练段 0-2351 内逐列 CUSUM 变点（生成参数中途漂移）

产物: output/clean/spectrum_report.json + figures/step0/figD10_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, REGIONS, HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_JSON = CLEAN / "spectrum_report.json"
FIG = FIGURES / "step0"
SEGS = {"train": (0, 2352), "cal": (2352, 2376),
        "frozen": (2376, 2400), "closure": (2400, HOURS_TOTAL)}
NUM_COLS = ["ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh", "AITrainingPower_MW",
            "GPU_Utilization_Percent", "AvailableRenewable_MW",
            "UsedRenewable_MW", "RenewableCharge_MW", "Curtailment_MW",
            "IT_Load_MW", "Total_Load_MW", "GridPurchase_MW",
            "GridCharge_MW", "GridSell_MW", "NetGridImport_MW",
            "CarbonEmission_tCO2", "SOC_MWh", "ChargePower_MW",
            "DischargePower_MW", "Baseline_AI_IT_Load_MW", "NonAI_IT_Load_MW"]


def _matrix(rt: pd.DataFrame, r: str, seg: tuple | None = None) -> pd.DataFrame:
    sub = rt[rt.Region == r].sort_values("Hour")
    if seg is not None:
        sub = sub[(sub.Hour >= seg[0]) & (sub.Hour < seg[1])]
    return sub[NUM_COLS].corr()


def d1_drift(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour")
        rows = []
        for c in NUM_COLS:
            y = sub[c].to_numpy(float)
            tr, fz = y[0:2352], y[2376:2400]
            base = abs(tr.mean())
            if base < 1e-9:
                continue
            rows.append({"col": c, "train_mean": float(tr.mean()),
                         "frozen_mean": float(fz.mean()),
                         "drift_pct": float((fz.mean() - tr.mean()) / base * 100),
                         "std_ratio": float(np.std(fz) / max(np.std(tr), 1e-9))})
        df = pd.DataFrame(rows).sort_values("drift_pct", key=abs, ascending=False)
        out[r] = df.to_dict("records")
    return out


def d2_corr(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        m_all = _matrix(rt, r)
        m_tr = _matrix(rt, r, (0, 2352))
        m_fz = _matrix(rt, r, (2376, 2400))
        d = np.abs(m_fz - m_tr)
        pairs = []
        for i, c1 in enumerate(NUM_COLS):
            for j, c2 in enumerate(NUM_COLS):
                if j <= i:
                    continue
                pairs.append({"pair": f"{c1}~{c2}",
                              "corr_all": float(m_all.loc[c1, c2]),
                              "corr_train": float(m_tr.loc[c1, c2]),
                              "corr_frozen": float(m_fz.loc[c1, c2]),
                              "seg_diff": float(d.loc[c1, c2])})
        pairs.sort(key=lambda p: -p["seg_diff"])
        out[r] = {"top_seg_diff": pairs[:8], "n_pairs": len(pairs)}
    return out


def d3_priceperiod_corr(rt: pd.DataFrame) -> dict:
    """分价格段相关 vs 全时段相关：每列与其它列的最大条件差异。"""
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour")
        rows = []
        for seg_name, grp in sub.groupby("PricePeriod"):
            m = grp[NUM_COLS].corr()
            rows.append((seg_name, m))
        ref = sub[NUM_COLS].corr()
        top = []
        for c1 in NUM_COLS:
            for c2 in NUM_COLS:
                if c2 <= c1:
                    continue
                dmax = 0.0; dname = ""
                for seg_name, m in rows:
                    dd = abs(m.loc[c1, c2] - ref.loc[c1, c2])
                    if dd > dmax:
                        dmax, dname = dd, seg_name
                top.append({"pair": f"{c1}~{c2}", "max_cond_diff": dmax,
                            "in_segment": dname, "corr_all": float(ref.loc[c1, c2])})
        top.sort(key=lambda p: -p["max_cond_diff"])
        out[r] = {"top_cond_diff": top[:8]}
    return out


def d4_demandresponse_corr(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour")
        ref = sub[NUM_COLS].corr()
        levels = sorted(sub.DemandResponseLevel.unique())
        rows = []
        for lv in levels:
            grp = sub[sub.DemandResponseLevel == lv]
            if len(grp) < 24:
                continue
            m = grp[NUM_COLS].corr()
            dmax = 0.0; pair = ""
            for c1 in NUM_COLS:
                for c2 in NUM_COLS:
                    if c2 <= c1:
                        continue
                    dd = abs(m.loc[c1, c2] - ref.loc[c1, c2])
                    if dd > dmax:
                        dmax, pair = dd, f"{c1}~{c2}"
            rows.append({"level": lv, "n": int(len(grp)),
                         "max_cond_diff": float(dmax), "pair": pair})
        out[r] = rows
    return out


def d5_cross_region(rt: pd.DataFrame) -> dict:
    out = {}
    for c in NUM_COLS:
        piv = rt.pivot_table(index="Hour", columns="Region", values=c)
        m = piv.corr()
        pairs = []
        for i in range(6):
            for j in range(i + 1, 6):
                pairs.append({"regions": f"{REGIONS[i]}~{REGIONS[j]}",
                              "corr": float(m.iloc[i, j])})
        pairs.sort(key=lambda p: -p["corr"])
        out[c] = {"min": pairs[-1]["corr"], "max": pairs[0]["corr"],
                  "max_pair": pairs[0], "min_pair": pairs[-1]}
    return out


def d6_period_spectrum(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour")
        rows = []
        for c in NUM_COLS:
            y = sub[c].to_numpy(float)
            tr = y[0:2352]
            rows.append(_period_stats(c, tr))
        out[r] = rows
    return out


def _period_stats(c: str, y: np.ndarray) -> dict:
    n = len(y)
    a1 = float(np.corrcoef(y[1:], y[:-1])[0, 1])
    a24 = float(np.corrcoef(y[24:], y[:-24])[0, 1]) if n > 24 else np.nan
    a168 = float(np.corrcoef(y[168:], y[:-168])[0, 1]) if n > 168 else np.nan
    res = {}
    for lag, name in ((24, "p24"), (168, "p168"), (336, "p336")):
        if n <= lag:
            res[name] = None
            continue
        tpl = pd.Series(y).groupby(np.arange(n) % lag).mean()
        r = np.mean(np.abs(y - tpl.reindex(np.arange(n) % lag).to_numpy()))
        res[name] = float(r / max(np.mean(np.abs(y)), 1e-9))
    return {"col": c, "autocorr_lag1": a1, "autocorr_lag24": a24,
            "autocorr_lag168": a168, **res}


def d7_changepoints(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour")
        found = []
        for c in NUM_COLS:
            y = sub[c].to_numpy(float)[0:2352]
            sd = np.std(y)
            if sd < 1e-9:
                continue
            z = (y - np.mean(y)) / sd
            s = np.cumsum(z)
            k = int(np.argmax(np.abs(s)))
            crit = 1.96 * np.sqrt(len(y))
            if abs(s[k]) > crit:
                found.append({"col": c, "t": int(k), "cusum": float(s[k]),
                              "crit": float(crit)})
        found.sort(key=lambda f: -abs(f["cusum"]))
        out[r] = found
    return out


def plot_drift(d1: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in REGIONS:
        df = pd.DataFrame(d1[r]).set_index("col")
        ax.plot(df.index, df["drift_pct"], "o-", label=r, ms=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(NUM_COLS)), NUM_COLS, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("冻结段漂移 %（vs 训练均值）")
    ax.set_title("D1 四段漂移谱（冻结段 vs 训练段，全列）")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figD10_drift_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    report = {
        "caliber": ("Phase 2 全谱：D1 冻结vs训练漂移（均值/p50/std）；D2 全列相关+段间差异；"
                    "D3/D4 条件相关差异（PricePeriod/DemandResponse）；D5 跨区耦合；"
                    "D6 自相关+周期模板残差（周期强度）；D7 CUSUM 变点（训练段内）"),
        "D1_drift": d1_drift(rt), "D2_corr": d2_corr(rt),
        "D3_priceperiod_cond": d3_priceperiod_corr(rt),
        "D4_demandresponse_cond": d4_demandresponse_corr(rt),
        "D5_cross_region": d5_cross_region(rt),
        "D6_period_spectrum": d6_period_spectrum(rt),
        "D7_changepoints": d7_changepoints(rt),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    plot_drift(report["D1_drift"])
    print("saved", OUT_JSON, flush=True)
    print("D7 changepoints (frozen-colts in train):")
    for r in REGIONS:
        print(" ", r, [(c["col"], c["t"], round(c["cusum"] / c["crit"], 1))
                       for c in report["D7_changepoints"][r]][:6])


if __name__ == "__main__":
    main()
