"""step1.3_rigor_analysis — Q1 严谨性补强（自检漏洞 A1-A6 修复）.

A1  η 档案量化: 42 序列 η=1−Var(残差)/Var(Y)（任务侧=均值残差，能源侧=日模板残差）
A2  到达过程拟合优度: 泊松/负二项拟合 + 拟合优度检验（白噪声→计数过程表述）
A3  E1a 机制纯度控制: 同框架自由度对照 + 价格 oracle 贪心上界（套利潜力上限）
A4  门槛敏感性: 验证门(3/5/7%, cov 0.85/0.90/0.95) / κ(90/95/98%) / E1(3/5/8pp)
A5  消纳口径敏感性: c_r ±10% 扰动重算 E1 收益差幅
A6  门判据方差整合: 能源侧门加 mape_std 条件（v2 判据作稳健性验证，不覆盖主榜）

产物（output/robust/rigor_pack.json + figures/step1/fig_eta_profile.png）
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from step0_config import (CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPES,
                          HOURS_TOTAL, TASK_TYPE_SHORT)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_R = OUTPUT / "robust"
FIG_S1 = FIGURES / "step1"
SEG_TRAIN = 2352


def load_ctx():
    spec = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    spec2 = importlib.util.spec_from_file_location(
        "s12", Path(__file__).resolve().parent / "step1.2_robust_schedule.py")
    s12 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s12)
    return s10, s12


def a1_eta_profile() -> dict:
    """η 档案：任务侧=均值残差（白噪声基准），能源侧=日模板残差。"""
    s11 = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    m = importlib.util.module_from_spec(s11)
    s11.loader.exec_module(m)
    series = m.make_series_dict()
    rows = []
    for name, info in series.items():
        y = info["y"][:SEG_TRAIN]
        var_y = float(np.var(y))
        if info["layer"] == "task":
            resid = y - np.mean(y)
        else:
            hod = np.arange(SEG_TRAIN) % 24
            tpl = pd.Series(y).groupby(hod).transform("mean").to_numpy()
            resid = y - tpl
        eta = 1 - float(np.var(resid)) / var_y
        rows.append({"series": name, "layer": info["layer"],
                     "type": name.split("|")[-1], "eta": eta})
    df = pd.DataFrame(rows)
    task = df[df.layer == "task"]
    ene = df[df.layer == "energy"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(-0.1, 1.05, 30)
    ax.hist(task.eta, bins=bins, alpha=0.6, label="任务侧(18)")
    ax.hist(ene.eta, bins=bins, alpha=0.6, label="能源侧(24)")
    ax.axvline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xlabel("η = 1 − Var(残差)/Var(Y)")
    ax.set_ylabel("序列数")
    ax.legend()
    ax.set_title("A1 η 档案：任务侧≈0（白噪声）vs 能源侧≈1（确定性模板）")
    fig.tight_layout()
    fig.savefig(FIG_S1 / "fig_eta_profile.png", bbox_inches="tight")
    plt.close(fig)
    return {"task": {"eta_mean": float(task.eta.mean()),
                     "eta_max": float(task.eta.max()),
                     "eta_dist": task.eta.round(3).tolist()},
            "energy": {"eta_mean": float(ene.eta.mean()),
                       "eta_min": float(ene.eta.min()),
                       "by_type": ene.groupby("type")["eta"].mean()
                       .round(3).to_dict()},
            "note": "任务侧基准=无条件均值（白噪声下最优拟合）；"
                    "能源侧基准=日模板（确定性结构）"}


def a2_arrival_gof() -> dict:
    """到达数序列的泊松/负二项拟合优度（区域×类型级）。"""
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    out = {}
    for r in REGIONS:
        for t in TASK_TYPES:
            sub = wt[(wt.SourceRegion == r) & (wt.TaskType == t)]
            counts = sub.groupby("ArrivalHour").size().reindex(
                range(SEG_TRAIN), fill_value=0).to_numpy()
            mean, var = float(counts.mean()), float(counts.var(ddof=1))
            n = len(counts)
            poisson_p = None
            nb_p = None
            if mean > 0:
                counts_bin = np.bincount(counts)
                if len(counts_bin) > 20:
                    counts_bin[19] += counts_bin[20:].sum()
                    counts_bin = counts_bin[:20]
                else:
                    counts_bin = np.pad(counts_bin, (0, 20 - len(counts_bin)))
                exp_poi = n * stats.poisson.pmf(np.arange(20), mean)
                exp_poi[19] += n - exp_poi.sum()
                with np.errstate(invalid="ignore"):
                    mask = (exp_poi > 5)
                if mask.sum() >= 3:
                    chi2 = np.sum((counts_bin[mask] - exp_poi[mask]) ** 2
                                  / exp_poi[mask])
                    dof = max(int(mask.sum()) - 2, 1)
                    poisson_p = float(stats.chi2.sf(chi2, dof))
                p_nb = 1.0 / max(mean, 1e-9)
                if p_nb > 0:
                    k_est = mean * p_nb / (1 - p_nb)
                    if k_est > 0:
                        exp_nb = n * stats.nbinom.pmf(np.arange(20), k_est, p_nb)
                        exp_nb[19] += n - exp_nb.sum()
                        mask = exp_nb > 5
                        if mask.sum() >= 3:
                            chi2 = np.sum((counts_bin[mask] - exp_nb[mask]) ** 2
                                          / exp_nb[mask])
                            dof = max(int(mask.sum()) - 3, 1)
                            nb_p = float(stats.chi2.sf(chi2, dof))
            out[f"{r}|{TASK_TYPE_SHORT[t]}"] = {
                "mean": mean, "var": var, "dispersion": var / mean,
                "poisson_gof_p": poisson_p, "negbin_gof_p": nb_p}
    return {"per_series": out,
            "summary": "到达过程：dispersion≈1 支持泊松，p 值低=偏离泊松"
                       "（白噪声表述以 ACF 为主，计数分布为补充）"}


def a3_oracle_upper_bound(s10, s12, wt, rt, params, consume) -> dict:
    """机制纯度控制 + 价格 oracle 贪心上界。

    C0 基准 local / C1 延后 greedy（已有产物）
    C2 换区自由度（E1a migrate，已有）
    C3 换区+延后（新） / C4 价格感知 oracle 贪心（新，套利上界）
    """
    whitelist = pd.read_csv(CLEAN / "whitelist.csv")
    wl = {(x.TaskType, x.SourceRegion): x.Reachable.split("|")
          for x in whitelist.itertuples(index=False)}
    rt_p = rt[["Hour", "Region", "ElectricityPrice_CNY_per_MWh"]].pivot(
        index="Hour", columns="Region",
        values="ElectricityPrice_CNY_per_MWh").to_numpy()
    cap_arr = np.array([params["cap"][r] for r in REGIONS], dtype=float)
    power_of = lambda tt: s10.GPU_POWER_MW[tt]

    def price_greedy():
        """价格感知 oracle 贪心：弹性任务在白名单×窗口网格选边际成本最小。"""
        occ = np.zeros((HOURS_TOTAL, len(REGIONS)))
        rows = []
        w = wt.sort_values(["Priority", "ArrivalHour"], ascending=[False, True])
        for rec in w.itertuples(index=False):
            g = float(rec.GPU_Demand)
            dur = float(rec.dur_h)
            r0 = REGIONS.index(rec.SourceRegion)
            if rec.TaskType == "RealTimeInference":
                st = int(rec.ArrivalHour)
                rows.append((rec.TaskID, rec.SourceRegion, st))
                h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
                if h1 > st:
                    occ[st:h1, r0] += g
                continue
            last = min(int(rec.LatestFinishHour - dur), HOURS_TOTAL - 1)
            cands = []
            for rn in wl[(rec.TaskType, rec.SourceRegion)]:
                ri = REGIONS.index(rn)
                for st in range(int(rec.ArrivalHour), last + 1):
                    h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
                    if h1 <= st:
                        continue
                    cost = float(rt_p[st, ri]) * g * power_of(rec.TaskType) * dur
                    cands.append((cost, ri, st, h1))
            cands.sort(key=lambda c: c[0])
            placed = False
            for cost, ri, st, h1 in cands:
                if (occ[st:h1, ri] + g - cap_arr[ri]).max() <= 0:
                    occ[st:h1, ri] += g
                    rows.append((rec.TaskID, REGIONS[ri], st))
                    placed = True
                    break
            if not placed:
                rows.append((rec.TaskID, rec.SourceRegion,
                             int(rec.ArrivalHour)))
        return pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])

    def migrate_shift():
        """C3：换区+延后（白名单最低均价区，容量贪心延后）。"""
        rt_avg = rt[rt.Hour < SEG_TRAIN].groupby("Region")[
            "ElectricityPrice_CNY_per_MWh"].mean()
        cost_rank = {r: i for i, r in enumerate(rt_avg.sort_values().index)}
        occ = np.zeros((HOURS_TOTAL, len(REGIONS)))
        rows = []
        w = wt.sort_values(["Priority", "ArrivalHour"], ascending=[False, True])
        for rec in w.itertuples(index=False):
            g = float(rec.GPU_Demand)
            dur = float(rec.dur_h)
            if rec.TaskType == "RealTimeInference":
                ri = REGIONS.index(rec.SourceRegion)
                st = int(rec.ArrivalHour)
                rows.append((rec.TaskID, rec.SourceRegion, st))
                h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
                if h1 > st:
                    occ[st:h1, ri] += g
                continue
            last = min(int(rec.LatestFinishHour - dur), HOURS_TOTAL - 1)
            cand_regions = sorted(wl[(rec.TaskType, rec.SourceRegion)],
                                  key=lambda x: cost_rank[x])
            placed = False
            for rn in cand_regions:
                ri = REGIONS.index(rn)
                for st in range(int(rec.ArrivalHour), last + 1):
                    h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
                    if h1 <= st:
                        break
                    if (occ[st:h1, ri] + g - cap_arr[ri]).max() <= 0:
                        occ[st:h1, ri] += g
                        rows.append((rec.TaskID, rn, st))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                ri = REGIONS.index(rec.SourceRegion)
                rows.append((rec.TaskID, rec.SourceRegion,
                             int(rec.ArrivalHour)))
        return pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])

    ev = lambda s: s10.evaluate_schedule(s, wt, rt, params, consume)[0]
    local = ev(pd.read_csv(OUT_R.parent / "baseline" / "local_schedule.csv"))
    greedy = ev(pd.read_csv(OUT_R.parent / "baseline" / "greedy_schedule.csv"))
    e1m = json.loads((OUT_R / "e1_mechanism.json").read_text(encoding="utf-8"))
    mig_cost = e1m["e1a"].get("migrate_only_cost_wan")
    c3 = ev(migrate_shift())
    c4 = ev(price_greedy())
    imp = lambda m: (local["cost_wan"] - m["cost_wan"]) / local["cost_wan"] * 100
    return {
        "C0_local_cost": local["cost_wan"],
        "C1_greedy_imp_pct": float(imp(greedy)),
        "C2_migrate_imp_pct": float((local["cost_wan"] - mig_cost)
                                    / local["cost_wan"] * 100) if mig_cost else None,
        "C3_migrate_shift_imp_pct": float(imp(c3)),
        "C4_oracle_price_imp_pct": float(imp(c4)),
        "oracle_upper_bound_note": "价格感知贪心=套利潜力的启发式上界"
                                   "（非全局最优，标注为 oracle 贪心）",
        "criterion_note": "E1a 的 4.6% 为 C2 启发式；oracle 上界给出套利潜力上限",
    }


def a4_threshold_sensitivity() -> dict:
    """验证门/κ/E1 门槛敏感性（对已有产物重判，秒级）。"""
    t = pd.read_csv(OUTPUT / "forecast" / "arena_table.csv")
    base_rows = t[t.family == "统计基线"].copy()
    out = {}
    for mape_th in (0.03, 0.05, 0.07):
        ok = 0
        for _, r in t[t.family != "统计基线"].iterrows():
            bm = base_rows[base_rows.series == r.series]
            b = bm.iloc[0].to_dict() if len(bm) else None
            if b is None:
                continue
            if r.layer == "energy":
                imp = (b["mape_mean"] - r["mape_mean"]) / max(b["mape_mean"], 1e-9)
                ok += int(imp >= mape_th)
            else:
                ok += int(r["gate"].startswith("通过"))
        out[f"energy_gate_{int(mape_th*100)}pct"] = ok
    for cov_th in (0.85, 0.90, 0.95):
        ok = 0
        for _, r in t[t.family != "统计基线"].iterrows():
            if r.layer != "task":
                continue
            bm = base_rows[base_rows.series == r.series]
            b = bm.iloc[0].to_dict() if len(bm) else None
            if b is None:
                continue
            cov_sig = (r["cov_mean"] - b["cov_mean"] >= b["cov_std"]
                       and r["cov_mean"] >= cov_th)
            ok += int(cov_sig or r["gate"].startswith("通过"))
        out[f"task_cov_{int(cov_th*100)}"] = ok
    kf = json.loads((OUT_R / "kappa_fit.json").read_text(encoding="utf-8"))
    for ct in (0.90, 0.95, 0.98):
        passed = [e for e in kf["eps_grid"]
                  if kf["calibration"][f"{float(e):.2f}"]["cov"] >= ct]
        out[f"kappa_target_{int(ct*100)}"] = {
            "passed": passed,
            "selected": float(max(passed)) if passed else 0.02}
    e1 = json.loads((OUT_R / "e1_three_way.json").read_text(encoding="utf-8"))
    out["e1_gap_pp"] = e1["gap_pp"]
    out["e1_criteria_3pp"] = e1["gap_pp"] < 3
    out["e1_criteria_8pp"] = e1["gap_pp"] < 8
    return out


def a5_consume_sensitivity(s10, wt, rt, params, consume) -> dict:
    """c_r(h) 模板 ±10% 扰动下 E1 收益差幅的稳定性（R1 模板口径适配）。"""
    qs = pd.read_csv(OUT_R / "quantile_schedule.csv")
    gs = pd.read_csv(OUT_R.parent / "baseline" / "greedy_schedule.csv")
    ls = pd.read_csv(OUT_R.parent / "baseline" / "local_schedule.csv")
    out = {}
    for tag, fac in (("minus10", 0.9), ("base", 1.0), ("plus10", 1.1)):
        cr = {r: np.clip(np.asarray(c, dtype=float) * fac, 0.0, 1.0)
              for r, c in consume.items()}
        ev = lambda s: s10.evaluate_schedule(s, wt, rt, params, cr)[0]
        base = ev(ls)
        perfect = ev(gs)
        quant = ev(qs)
        imp_p = (base["cost_wan"] - perfect["cost_wan"]) / base["cost_wan"]
        imp_q = (base["cost_wan"] - quant["cost_wan"]) / base["cost_wan"]
        out[tag] = {"gap_pp": abs(imp_p - imp_q) * 100,
                    "imp_perfect_pct": imp_p * 100,
                    "imp_quantile_pct": imp_q * 100}
    return out


def a6_gate_v2() -> dict:
    """门判据 v2：能源侧加方差条件（降幅 > max(5%, 2×基线CV)），作稳健性验证。"""
    t = pd.read_csv(OUTPUT / "forecast" / "arena_table.csv")
    base_rows = t[t.family == "统计基线"].copy()
    changed, rejected = [], []
    for _, r in t[t.family != "统计基线"].iterrows():
        bm = base_rows[base_rows.series == r.series]
        if not len(bm):
            continue
        b = bm.iloc[0]
        if r.layer != "energy":
            continue
        imp = (b["mape_mean"] - r["mape_mean"]) / max(b["mape_mean"], 1e-9)
        cv = b["mape_std"] / max(b["mape_mean"], 1e-9)
        v2_ok = imp >= max(0.05, 2 * cv)
        v1_ok = r["gate"].startswith("通过")
        if v1_ok and not v2_ok:
            rejected.append({"series": r["series"], "model": r["model"],
                             "imp": round(imp * 100, 1),
                             "base_cv": round(cv * 100, 1)})
        if v1_ok != v2_ok:
            changed.append(r["series"])
    return {"v2_rejects": rejected, "n_changed": len(changed),
            "note": "v2 判据仅作稳健性验证，主榜结论保持 v1 判据不变"}


def main() -> None:
    s10, s12 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    r = {}
    r["a1_eta"] = a1_eta_profile()
    r["a2_arrival_gof"] = a2_arrival_gof()
    r["a3_oracle"] = a3_oracle_upper_bound(s10, s12, wt, rt, params, consume)
    r["a4_threshold"] = a4_threshold_sensitivity()
    r["a5_consume"] = a5_consume_sensitivity(s10, wt, rt, params, consume)
    r["a6_gate_v2"] = a6_gate_v2()

    with open(OUT_R / "rigor_pack.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: (v if k not in ("a2_arrival_gof",) else
                          {kk: vv for kk, vv in v.items()
                           if kk != "per_series"})
                      for k, v in r.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
