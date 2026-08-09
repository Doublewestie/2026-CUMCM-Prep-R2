"""step0.6_thresholds — Phase 3: 临界点/阈值扫描（T1-T9）.

T1 W 阶梯模板精确公式（800+300*sin 量化）残差验证
T2 SellLimit 饱和: 触发条件（时段/W/弃电特征）
T3 SOC 边界触碰: MinSOC/Cap 触碰时段 + 触碰时充放行为
T4 MaxGridImport 顶格检测（时段分布）
T5 MaxCharge/MaxDischarge 顶格检测（基线）
T6 消纳能力激活率: min(W, c(h)*D) 中 W 瓶颈小时比例（应≈0，R1 复检）
T7 时段标签切换时刻全列跳变（PricePeriod/DemandResponse 切换前后 1h 各列均值差异）
T8 分箱条件均值断点: 关键列对的分位数分箱 -> 其它列条件均值跳变检测
T9 互信息矩阵: 25 列 x 25 列 MI（离散化，非线性关联补充）

产物: output/clean/threshold_report.json + figures/step0/figD11_*.png
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from step0_config import CLEAN, FIGURES, REGIONS, HOURS_TOTAL

OUT_JSON = CLEAN / "threshold_report.json"
FIG = FIGURES / "step0"
NUM_COLS = ["ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh", "AITrainingPower_MW",
            "GPU_Utilization_Percent", "AvailableRenewable_MW",
            "UsedRenewable_MW", "RenewableCharge_MW", "Curtailment_MW",
            "IT_Load_MW", "Total_Load_MW", "GridPurchase_MW",
            "GridCharge_MW", "GridSell_MW", "NetGridImport_MW",
            "CarbonEmission_tCO2", "SOC_MWh", "ChargePower_MW",
            "DischargePower_MW", "Baseline_AI_IT_Load_MW", "NonAI_IT_Load_MW"]
TR = np.arange(0, 2352)


def t1_w_template(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[(rt.Region == r) & (rt.Hour < 2352)].sort_values("Hour")
        w = sub["AvailableRenewable_MW"].to_numpy()
        h = np.arange(len(w)) % 24
        fit = 800 + 300 * np.sin(2 * np.pi * (h - 4) / 24)
        resid = w - fit
        out[r] = {"max_abs_resid": float(np.max(np.abs(resid))),
                  "rel_mean": float(np.mean(np.abs(resid)) / 800),
                  "unique_values": int(len(np.unique(w))),
                  "formula": "W = 800 + 300*sin(2*pi*(h-4)/24)"}
    return out


def t2_sell_saturation(rt: pd.DataFrame, sp: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        lim = float(sp.set_index("Region").loc[r, "SellLimit_MW"])
        sub = rt[(rt.Region == r) & (rt.Hour < 2352)].sort_values("Hour")
        sell = sub["GridSell_MW"].to_numpy()
        if lim <= 0:
            out[r] = {"sell_limit": lim, "saturation_share": 0.0,
                      "note": "SellLimit=0（无外送能力）"}
            continue
        sat = sell >= lim - 0.5
        if sat.sum() == 0:
            out[r] = {"saturation_share": 0.0, "note": "SellLimit=0 或无饱和"}
            continue
        cond = sub[sat]
        out[r] = {
            "sell_limit": float(lim),
            "saturation_share": float(sat.mean()),
            "sat_peak_share": float((cond.PricePeriod == "Peak").mean()),
            "sat_valley_share": float((cond.PricePeriod == "Valley").mean()),
            "sat_W_mean": float(cond.AvailableRenewable_MW.mean()),
            "sat_curtail_mean": float(cond.Curtailment_MW.mean()),
            "unsat_W_mean": float(sub[~sat].AvailableRenewable_MW.mean()),
            "unsat_curtail_mean": float(sub[~sat].Curtailment_MW.mean()),
            "sat_hour_of_day": sorted((cond.Hour % 24).unique().tolist()),
        }
    return out


def t3_soc_bounds(rt: pd.DataFrame, sp: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        row = sp.set_index("Region").loc[r]
        mn, cap = float(row["MinSOC_MWh"]), float(row["StorageCapacity_MWh"])
        sub = rt[rt.Region == r].sort_values("Hour")
        soc = sub["SOC_MWh"].to_numpy()
        out[r] = {
            "min_touch_h": int(np.sum(soc < mn + 1e-6)),
            "cap_touch_h": int(np.sum(soc > cap - 1e-6)),
            "soc_min": float(soc.min()), "soc_max": float(soc.max()),
            "at_min_pct": float(np.mean(soc < mn + 1e-6)) * 100,
            "at_cap_pct": float(np.mean(soc > cap - 1e-6)) * 100,
        }
    return out


def t4_max_import(rt: pd.DataFrame, sp: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        lim = float(sp.set_index("Region").loc[r, "MaxGridImport_MW"])
        sub = rt[rt.Region == r].sort_values("Hour")
        g = sub["GridPurchase_MW"].to_numpy()
        out[r] = {"limit": lim, "at_limit_pct": float(np.mean(g >= lim - 0.5)) * 100,
                  "max": float(g.max())}
    return out


def t5_power_limits(rt: pd.DataFrame, sp: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        row = sp.set_index("Region").loc[r]
        mc, md = float(row["MaxChargePower_MW"]), float(row["MaxDischargePower_MW"])
        sub = rt[rt.Region == r].sort_values("Hour")
        pc, pd_ = sub["ChargePower_MW"].to_numpy(), sub["DischargePower_MW"].to_numpy()
        out[r] = {"charge_at_limit_pct": float(np.mean(pc >= mc - 0.5)) * 100,
                  "discharge_at_limit_pct": float(np.mean(pd_ >= md - 0.5)) * 100,
                  "charge_max": float(pc.max()), "discharge_max": float(pd_.max())}
    return out


def t6_consume_bottleneck(rt: pd.DataFrame) -> dict:
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    out = {}
    for r in REGIONS:
        sub = rt[(rt.Region == r) & (rt.Hour < 2352)].sort_values("Hour")
        w = sub["AvailableRenewable_MW"].to_numpy()
        d = sub["Total_Load_MW"].to_numpy()
        c_h = np.array(consume[r], dtype=float)
        cap = c_h[np.arange(len(w)) % 24] * d
        out[r] = {"w_bottleneck_share": float(np.mean(w < cap)),
                  "note": "W < c(h)*D 时消纳受 W 限制（R1 预测≈0）"}
    return out


def t7_label_switch_jump(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[(rt.Region == r) & (rt.Hour < 2352)].sort_values("Hour")
        pp = sub["PricePeriod"].to_numpy()
        dr = sub["DemandResponseLevel"].to_numpy()
        rows = []
        for label, arr in (("PricePeriod", pp), ("DemandResponse", dr)):
            idx = np.where(arr[1:] != arr[:-1])[0]
            jumps = []
            for i in idx[:200]:
                t = i + 1
                before, after = arr[t - 1], arr[t]
                a = sub.iloc[t - 1]; b = sub.iloc[t]
                j = {}
                for c in NUM_COLS[:12]:
                    v0, v1 = float(a[c]), float(b[c])
                    base = max(abs(v0), abs(v1), 1e-9)
                    j[c] = round((v1 - v0) / base * 100, 1)
                jumps.append({"t": int(t), "from": before, "to": after, "jump_pct": j})
            if jumps:
                rows.append({"label": label, "n_transitions": len(idx),
                             "samples": jumps[:5]})
        out[r] = rows
    return out


def t8_binned_jump(rt: pd.DataFrame) -> dict:
    """关键列对分箱条件均值跳变（断点候选）。"""
    pairs = [("GridPurchase_MW", "CarbonEmission_tCO2"),
             ("AvailableRenewable_MW", "Curtailment_MW"),
             ("GridSell_MW", "Curtailment_MW"),
             ("ElectricityPrice_CNY_per_MWh", "ChargePower_MW"),
             ("SOC_MWh", "DischargePower_MW"),
             ("GridPurchase_MW", "ElectricityPrice_CNY_per_MWh")]
    out = {}
    for r in REGIONS:
        sub = rt[(rt.Region == r) & (rt.Hour < 2352)].sort_values("Hour")
        res = []
        for xc, yc in pairs:
            x = sub[xc].to_numpy(float); y = sub[yc].to_numpy(float)
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(x, qs)
            means = [float(y[bins == k].mean()) for k in range(5)]
            deltas = [abs(means[k + 1] - means[k]) for k in range(4)]
            res.append({"x": xc, "y": yc, "bin_means": means,
                        "max_delta_rel": float(max(deltas) / max(np.std(y), 1e-9)),
                        "at_bin": int(np.argmax(deltas))})
        res.sort(key=lambda v: -v["max_delta_rel"])
        out[r] = res[:6]
    return out


def _mi(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    c = np.histogram2d(x, y, bins=bins)[0]
    c = c / c.sum()
    cx, cy = c.sum(1), c.sum(0)
    ent = 0.0
    for i in range(bins):
        for j in range(bins):
            if c[i, j] > 0:
                ent += c[i, j] * np.log(c[i, j] / (cx[i] * cy[j] + 1e-12))
    return float(ent)


def t9_mi_matrix(rt: pd.DataFrame) -> dict:
    out = {}
    for r in REGIONS:
        sub = rt[(rt.Region == r) & (rt.Hour < 2352)].sort_values("Hour")
        arr = sub[NUM_COLS].to_numpy(float)
        pairs = []
        for i, j in combinations(range(len(NUM_COLS)), 2):
            mi = _mi(arr[:, i], arr[:, j])
            if mi > 0.2:
                pairs.append({"pair": f"{NUM_COLS[i]}~{NUM_COLS[j]}", "mi": mi})
        pairs.sort(key=lambda p: -p["mi"])
        out[r] = pairs[:10]
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    sp = pd.read_csv(CLEAN / "storage_params.csv")
    report = {
        "caliber": "Phase 3 临界点扫描（训练段 0-2351 为主，冻结段描述性）",
        "T1_w_template": t1_w_template(rt),
        "T2_sell_saturation": t2_sell_saturation(rt, sp),
        "T3_soc_bounds": t3_soc_bounds(rt, sp),
        "T4_max_import": t4_max_import(rt, sp),
        "T5_power_limits": t5_power_limits(rt, sp),
        "T6_consume_bottleneck": t6_consume_bottleneck(rt),
        "T7_label_switch_jump": t7_label_switch_jump(rt),
        "T8_binned_jump": t8_binned_jump(rt),
        "T9_mi_matrix": t9_mi_matrix(rt),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved", OUT_JSON, flush=True)
    print("T1:", {k: round(v["max_abs_resid"], 4) for k, v in report["T1_w_template"].items()})
    print("T2:", {k: (round(v["saturation_share"], 3), v.get("sat_peak_share")) for k, v in report["T2_sell_saturation"].items()})
    print("T3:", {k: (v["at_min_pct"], v["at_cap_pct"]) for k, v in report["T3_soc_bounds"].items()})
    print("T6:", {k: v["w_bottleneck_share"] for k, v in report["T6_consume_bottleneck"].items()})


if __name__ == "__main__":
    main()
