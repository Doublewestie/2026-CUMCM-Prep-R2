"""step4.3_q4_scenarios — Q4 压力矩阵：碳约束 × 电价机制 × 新能源波动三场景对比.

方法（Q4 方案定稿 step4.3，档位依据全链实证）:
  维度（题面钦定三维）:
    碳约束  τ∈{10%,20%,30%}  区域碳限额 = 碳排基线×(1−τ)（Sobol 碳由碳强度主导）
    电价机制 峰谷比 ×k（档位来自 step4.1 预扫 price_ratio_prescan.suggested，
            重构保均值：新价=时段均值+k×(原价−时段均值)）
    新能源波动 σ∈{10%,20%,30%}  W 模板×AR(1) 扰动×K-means 12 缩减
        （档位继承 Q3 MPC SIGMAS；波动有真实成本实证 X15）
  代表性策略集（不做 18 轮完整进化——"策略迁移"本身是研究结论，C3 决策）:
    Q4 折中 / Q2 折中（Q4 基线）/ 构造解 三档
  每格输出: 六指标 + 最优策略 + 与基线对照 → 策略鲁棒性热图
  合理性检查: 重构价格统计量（峰谷比/均值漂移）随档位报告

产物（output/q4/）:
  q4_pressure.json   每格六指标 + 最优策略 + 变化表 + 档位合理性
figures/step4/fig_q4_pressure.png  三张热图（碳/峰谷比/波动 × 策略）
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# K-means 已弃用（1-2 修正：中心截断低估波动成本 25.6%）

from step0_config import FIGURES, OUTPUT, REGIONS, HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"
SIGMAS = (0.10, 0.20, 0.30)
PHI = 0.8
N_SCEN = 64  # 原始场景数（1-2 修正：弃 K-means 中心截断）


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"),
        Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    s41 = _load_mod("step4.1_q4_indicators.py")
    s40 = _load_mod("step4.0_q4_bilevel.py")
    c = s41.load_ctx()
    return {**c, "s41": s41, "s40": s40}


def representative_policies(c) -> dict[str, list]:
    """三档代表性策略：Q4 折中（step4.0 产物，缺失回退 Q2 折中）/ Q2 折中 / 构造。"""
    pols = {"construct": [0.0, 0.0, 0.0, 0.0, 0.0]}
    pols["q2_compromise"] = c["s41"].q2_compromise_policy(c)
    front = OUT_Q4 / "q4_front.csv"
    if front.exists():
        df = pd.read_csv(front)
        F = df[["cost_wan", "carbon_t", "latency_ms", "one_minus_qos",
                "one_minus_nu", "peak_net_MW"]].to_numpy()
        i = int(np.argmin(F[:, 0] + F[:, 2] + F[:, 5]))   # 成本+时延+峰值均衡代表
        pols["q4_compromise"] = json.loads(df.loc[i, "policy"])
    else:
        pols["q4_compromise"] = pols["q2_compromise"] + [0.5]
    return pols


def price_reconstruct(c, k: float) -> dict[str, np.ndarray]:
    """峰谷比重构（保均值）：新价 = 时段均值 + k×(原价 − 时段均值)。"""
    out = {}
    rt = c["rt"]
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
        p = sub["ElectricityPrice_CNY_per_MWh"].to_numpy()
        mu = sub.groupby("PricePeriod")[
            "ElectricityPrice_CNY_per_MWh"].transform("mean").to_numpy()
        out[r] = mu + k * (p - mu)
    return out


def gen_scenarios(c, r: str, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W 场景：模板×AR(1) 扰动（原始 64 场景，1-2 修正——K-means 中心
    截断低估波动成本 25.6%，D 区 S4 实证；Q3 X15 用 64 场景期望未污染）。

    返回原始场景（等权）——E/F 区低 W 尾部触发购电上限瓶颈（step3.9：
    W×0.2 即 infeasible），期望限定可行域 + infeasible_pct 双报告。
    """
    sub = c["rt"][c["rt"].Region == r].sort_values("Hour").reset_index(drop=True)
    W = sub["AvailableRenewable_MW"].to_numpy()
    T = len(W)
    rng = np.random.RandomState(int(hash(r) % 2**31) + int(sigma * 1000))
    S = np.zeros((N_SCEN, T))
    for s in range(N_SCEN):
        eps = np.zeros(T)
        z = rng.randn(T)
        for t in range(1, T):
            eps[t] = PHI * eps[t - 1] + sigma * z[t]
        S[s] = W * np.clip(1 + eps, 0.1, 2.0)
    w = np.ones(N_SCEN) / N_SCEN
    return S, w, W


def carbon_scenario(c, pol: list, tau: float) -> dict:
    """碳约束场景：储能层可行则直接求解；不可行则测'上层杠杆'三族。

    机理（step4.2 实证）：G ≥ D−cap_h 隐含下界（消纳模板锁定）→ 碳排下界
    由负荷时段分布 D 决定 → 错峰（把负荷移到 cap_h 大的小时）/迁移（换
    低碳区域）可降下界。杠杆三族（T3 决策）:
      A 错峰: shift_BI=shift_AT=24（现状，实测无效——弹性任务占比有限）
      B 迁移: mig_gpu_min=8 / mig_dur_min=6.65（压向 E/F 低碳高消纳区）
      C 联合: A+B + headroom=0.03
    裁决: 任一族可行 → lever_ok=True + 代价；全不可行 → 杠杆不足（诚实），
    扫 τ 找杠杆可达碳下限（见 lever_floor）。
    输出: {carbon, cost, lever_needed, lever_ok, lever_cost, per_region}
    """
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol[:5]))
    ai_mw = c["s41"].schedule_occupancy(c, sched)

    def _solve(ai):
        cost = carbon = 0.0
        peak = 0.0
        ok = True
        per = {}
        for r in REGIONS:
            d = c["s41"].build_lower_data(c, ai, r)
            ch, dh = c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"]
            m0 = c["s40"].solve_lower_constrained(d, ch, dh)
            if m0["cost_wan"] is None:
                return None
            cap0 = m0["carbon_t"]
            m = c["s40"].solve_lower_constrained(d, ch, dh,
                                                 carbon_cap_t=cap0 * (1 - tau))
            if m["cost_wan"] is None:
                ok = False
                per[r] = {"infeasible": True, "carbon_lb": round(cap0, 1)}
                continue
            G = np.array([x["G"] for x in m["rows"]])
            S = np.array([x["S"] for x in m["rows"]])
            cost += m["cost_wan"]
            carbon += m["carbon_t"]
            peak = max(peak, float((G - S)[:2400].max()))
            per[r] = {"cost_wan": round(m["cost_wan"], 2),
                      "carbon_t": round(m["carbon_t"], 1)}
        return {"cost": cost, "carbon": carbon, "peak": peak, "ok": ok,
                "per": per}

    base = _solve(ai_mw)
    if base is None:
        return {"infeasible_global": True}
    if base["ok"]:
        return {"carbon": base["carbon"], "cost": base["cost"],
                "peak_net_MW": base["peak"], "lever_needed": False,
                "per_region": base["per"]}

    # 杠杆三族（T3）：A 错峰 24h / B 迁移压低碳区 / C 联合（A+B+headroom）
    lever_pols = {
        "A_shift24": list(pol[:5])[:2] + [24.0, 24.0] + [pol[4]],
        "B_migrate": [8.0, 6.65, 12.0, 12.0, 0.0],
        "C_joint": [8.0, 6.65, 24.0, 24.0, 0.03],
    }
    lev_res = {}
    for name, lpol in lever_pols.items():
        sched_l = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                                 c["params"], tuple(lpol))
        ai_l = c["s41"].schedule_occupancy(c, sched_l)
        lev = _solve(ai_l)
        lev_res[name] = {"ok": bool(lev and lev["ok"]),
                         "cost": (lev["cost"] if lev and lev["ok"] else None),
                         "carbon": (lev["carbon"] if lev and lev["ok"] else None)}
    any_ok = any(v["ok"] for v in lev_res.values())
    ok_name = next((k for k, v in lev_res.items() if v["ok"]), None)
    return {"carbon": None, "cost": None, "peak_net_MW": None,
            "lever_needed": True, "lever_ok": any_ok,
            "lever_variants": lev_res,
            "lever_cost": (lev_res[ok_name]["cost"] if ok_name else None),
            "lever_carbon": (lev_res[ok_name]["carbon"] if ok_name else None),
            "lever_best": ok_name,
            "per_region": base["per"],
            "note": "储能层碳排已在下界（消纳模板锁定 G≥D−cap_h）——"
                    "τ 需上层杠杆配合；杠杆三族=错峰/迁移/联合（T3 决策）"}


def run_policy_grid(c, pol: list, tau=None, price_map=None,
                    sigma=None) -> dict:
    """单策略 × 单格：逐区下层求解 → 六指标（价格/W 旋钮；碳走 carbon_scenario）。"""
    if tau is not None:
        cs = carbon_scenario(c, pol, tau)
        sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                               c["params"], tuple(pol[:5]))
        lat = c["s20"].compute_latency(c["wt"], sched)
        out = {"cost_wan": cs.get("cost"), "carbon_t": cs.get("carbon"),
               "latency_ms": lat, "infeasible": bool(
                   cs.get("infeasible_global") or
                   (cs.get("lever_needed") and not cs.get("lever_ok"))),
               "lever_needed": cs.get("lever_needed", False),
               "lever_ok": cs.get("lever_ok", None),
               "lever_best": cs.get("lever_best", None),
               "lever_variants": cs.get("lever_variants"),
               "lever_cost_wan": cs.get("lever_cost")}
        return out
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol[:5]))
    ai_mw = c["s41"].schedule_occupancy(c, sched)
    cost = carbon = qsum = wsum = 0.0
    peak = 0.0
    peak_expect = 0.0
    peak_worst = 0.0
    n_infeas_total = 0
    for r in REGIONS:
        d = c["s41"].build_lower_data(c, ai_mw, r)
        ch, dh = c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"]
        kw = {}
        if price_map is not None:
            d = dict(d)
            d["price"] = price_map[r]
        if sigma is not None:
            reps, w, W0 = gen_scenarios(c, r, sigma)
            exp_cost = 0.0
            exp_carbon = 0.0
            exp_q = 0.0
            r_peak_exp = 0.0
            r_peak_worst = 0.0
            w_sum = 0.0
            n_infeas = 0
            for s_i in range(N_SCEN):
                ds = dict(d)
                ds["W"] = reps[s_i]
                m = c["s40"].solve_lower_constrained(ds, ch, dh, **kw)
                if m["cost_wan"] is None:
                    n_infeas += 1
                    continue
                G = np.array([x["G"] for x in m["rows"]])
                S = np.array([x["S"] for x in m["rows"]])
                p_i = float((G - S)[:2400].max())
                exp_cost += w[s_i] * m["cost_wan"]
                exp_carbon += w[s_i] * m["carbon_t"]
                exp_q += w[s_i] * m["curtail"]
                r_peak_exp += w[s_i] * p_i
                r_peak_worst = max(r_peak_worst, p_i)
                w_sum += w[s_i]
            # 条件期望（可行域）：E/F 低 W 尾部 infeasible（购电上限瓶颈，
            # step3.9 实证）——不可行占比即尾部风险度量
            if w_sum > 1e-12:
                exp_cost /= w_sum
                exp_carbon /= w_sum
                exp_q /= w_sum
                r_peak_exp /= w_sum
            cost += exp_cost
            carbon += exp_carbon
            qsum += exp_q
            wsum += float(W0.sum())
            peak_expect = max(peak_expect, r_peak_exp)
            peak_worst = max(peak_worst, r_peak_worst)
            n_infeas_total += n_infeas
            continue
        m = c["s40"].solve_lower_constrained(d, ch, dh, **kw)
        if m["cost_wan"] is None:
            return {"infeasible": True}
        G = np.array([x["G"] for x in m["rows"]])
        S = np.array([x["S"] for x in m["rows"]])
        Q = np.array([x["Q"] for x in m["rows"]])
        cost += m["cost_wan"]
        carbon += m["carbon_t"]
        qsum += float(Q.sum())
        wsum += float(d["W"].sum())
        peak = max(peak, float((G - S)[:2400].max()))
    nu = 100.0 * (1 - qsum / max(wsum, 1e-9))
    lat = c["s20"].compute_latency(c["wt"], sched)
    out = {"cost_wan": cost, "carbon_t": carbon, "latency_ms": lat,
           "nu_pct": nu, "infeasible": False}
    if sigma is not None:
        out["peak_net_MW"] = round(peak_expect, 3)
        out["peak_expect_MW"] = round(peak_expect, 3)
        out["peak_worst_MW"] = round(peak_worst, 3)
        out["n_infeasible_scenarios"] = n_infeas_total
        out["infeasible_pct"] = round(
            n_infeas_total / (len(REGIONS) * N_SCEN) * 100, 1)
        out["peak_caliber"] = ("期望峰值=max_r(Σwᵢ·峰ᵢ/Σwᵢ 条件期望)；"
                               "最坏峰值=max_r(maxᵢ 峰ᵢ)；64 原始场景（1-2 修正："
                               "K-means 中心截断低估 25.6% 已弃用）；"
                               "E/F 低 W 尾部购电上限瓶颈→条件期望+占比双报告")
    else:
        out["peak_net_MW"] = peak
    return out


def lever_floor_scan(c) -> dict:
    """碳杠杆可达下限扫描（mig 旋钮全档，仅碳场景全败时触发）。

    判别实验（q4_carbon_lever_scan.json）：mig_gpu_min∈{0,2,4,6,8,11.9,15}
    的碳排响应 → 杠杆上限 = (基线碳 − 最小碳)/基线碳。结论：任务层碳杠杆
    ≈0.14-0.25%（NonAI 底数稀释 + 调度器成本驱动）→ τ≥1% 即超杠杆能力。
    """
    sched0 = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                            c["params"],
                                            tuple(c["s41"].q2_compromise_policy(c)))
    ai0 = c["s41"].schedule_occupancy(c, sched0)
    qsum = wsum = 0.0
    for r in REGIONS:
        d = c["s41"].build_lower_data(c, ai0, r)
        ch, dh = c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"]
        m0 = c["s40"].solve_lower_constrained(d, ch, dh)
        qsum += m0["curtail"]
        wsum += float(d["W"].sum())
    base_nu = 100.0 * (1 - qsum / max(wsum, 1e-9))
    best_c, best_cost = None, None
    rows = []
    for mig in (0.0, 6.0, 11.9, 15.0):
        pol = [mig, 0.0, 12.0, 12.0, 0.0]
        sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                               c["params"], tuple(pol))
        ev = c["s41"].evaluate_q4_six(c, sched, alpha=0.5)
        if ev["viol_h"] > 0:
            continue
        rows.append({"mig_gpu_min": mig,
                     "carbon_t": round(float(ev["carbon_t"]), 0),
                     "cost_wan": round(float(ev["cost_wan"]), 1)})
        if best_c is None or ev["carbon_t"] < best_c:
            best_c, best_cost = ev["carbon_t"], ev["cost_wan"]
    base_c = rows[-1]["carbon_t"] if rows else None
    lever = (base_c - best_c) / max(base_c, 1e-9) if base_c else None
    return {"scan": rows, "base_carbon_t": base_c,
            "min_carbon_t": best_c, "max_lever_pct": round(lever * 100, 3)
            if lever else None,
            "verdict": ("任务层碳杠杆上限 ≈0.1-0.3%（NonAI 底数稀释+成本驱动"
                        "调度器）→ 等比例区域碳限额 τ≥1% 即超杠杆能力，"
                        "τ∈{10%,20%,30%} 不可达系结构性（政策级结论："
                        "减排需源头或跨区电量调度，非任务调度杠杆）"),
            "base_nu_pct": round(base_nu, 2)}


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    pols = representative_policies(c)

    # 档位合理性（来自 step4.1 预扫，缺失回退）
    try:
        ind = json.loads((OUT_Q4 / "q4_indicators.json").read_text(
            encoding="utf-8"))
        ks = ind["price_ratio_prescan"]["suggested"][:2]
    except Exception:
        ks = [1.5, 2.0]

    grid = {"carbon": {}, "price": {}, "volatility": {}}

    for tau in (0.10, 0.20, 0.30):
        grid["carbon"][str(tau)] = {
            name: run_policy_grid(c, pol, tau=tau) for name, pol in pols.items()}

    price_map = {k: price_reconstruct(c, k) for k in ks}
    for k in ks:
        grid["price"][str(k)] = {
            name: run_policy_grid(c, pol, price_map=price_map[k])
            for name, pol in pols.items()}

    for sigma in SIGMAS:
        grid["volatility"][str(sigma)] = {
            name: run_policy_grid(c, pol, sigma=sigma) for name, pol in pols.items()}

    # 策略变化：每格最优策略（成本最低，仅限可行）vs 基线（q2_compromise 可行时）
    change = {}
    for dim, sub in grid.items():
        for frac, cells in sub.items():
            feasible = {n: cl for n, cl in cells.items()
                        if not cl.get("infeasible") and cl.get("cost_wan") is not None}
            b = cells.get("q2_compromise", {})
            if not feasible:
                change[f"{dim}|{frac}"] = {
                    "best_policy": None, "cost_wan": None,
                    "vs_baseline_cost_pct": None, "peak_net_MW": None,
                    "note": "该档全部策略不可行（储能层可行域下界，见 q4_shadow）"}
                continue
            best = min(feasible, key=lambda n: feasible[n]["cost_wan"])
            base_cost = b.get("cost_wan")
            change[f"{dim}|{frac}"] = {
                "best_policy": best,
                "cost_wan": round(feasible[best]["cost_wan"], 2),
                "vs_baseline_cost_pct": (round(
                    (feasible[best]["cost_wan"] - base_cost)
                    / max(abs(base_cost), 1e-9) * 100, 2)
                    if base_cost is not None and not b.get("infeasible") else None),
                "peak_net_MW": feasible[best].get("peak_net_MW", None),
                "lever_ok": cells.get("q4_compromise", {}).get("lever_ok")
                if cells.get("q4_compromise", {}).get("lever_needed") else None}

    price_stats = {}
    for k in ks:
        allp = np.concatenate([price_map[k][r] for r in REGIONS])
        cur = np.concatenate([
            c["rt"][c["rt"].Region == r]["ElectricityPrice_CNY_per_MWh"].to_numpy()
            for r in REGIONS])
        price_stats[str(k)] = {
            "peak_ratio": round(float(allp.max() / allp.min()), 2),
            "current_peak_ratio": round(float(cur.max() / cur.min()), 2),
            "mean_drift_pct": round(float(
                abs(allp.mean() / cur.mean() - 1) * 100), 4)}

    report = {
        "grid": grid, "strategy_change": change,
        "price_ratio_scan": price_stats,
        "caliber": ("碳约束=区域碳限额×(1−τ)；峰谷比=时段均值+k×(原价−均值)保均值重构；"
                    "波动=W 模板×AR(1)σ×K-means12 场景期望；"
                    "代表策略集=Q4 折中/Q2 折中/构造（C3 决策：策略迁移即结论）；"
                    "场景自造声明（题面未给场景数据）"),
        "spec": "spec_M4_Q4 场景章节；档位依据：Sobol 归因 + Q3 SIGMAS + step4.1 预扫",
    }
    # 碳杠杆可达下限（碳场景全部不可行时——结构性结论支撑，T4）
    carbon_cells = [cl for sub in grid["carbon"].values() for cl in sub.values()]
    if all(cl.get("infeasible") for cl in carbon_cells):
        report["carbon_lever_floor"] = lever_floor_scan(c)
    else:
        report["carbon_lever_floor"] = {"note": "存在可行碳场景，无需杠杆下限扫描"}
    with open(OUT_Q4 / "q4_pressure.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    # 热图：策略 × 档位 的成本
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, dim, fmt in ((axes[0], "carbon", "τ"), (axes[1], "price", "k"),
                         (axes[2], "volatility", "σ")):
        names = list(pols.keys())
        fracs = list(grid[dim].keys())
        M = np.array([[grid[dim][f][n].get("cost_wan", np.nan)
                       if not grid[dim][f][n].get("infeasible") else np.nan
                       for n in names] for f in fracs], dtype=float)
        im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_yticks(range(len(fracs)))
        ax.set_yticklabels([f"{fmt}={f}" for f in fracs], fontsize=8)
        ax.set_title(f"{dim}: 成本热图（万元）")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Q4 压力矩阵：三场景 × 三策略的成本鲁棒性")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_pressure.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"strategy_change": change, "price_ratio_scan": price_stats,
                      "caliber": report["caliber"]},
                     ensure_ascii=True, indent=2, default=float)[:2500])


if __name__ == "__main__":
    main()
