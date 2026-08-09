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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from step0_config import FIGURES, OUTPUT, REGIONS, HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"
SIGMAS = (0.10, 0.20, 0.30)
PHI = 0.8
N_SCEN, K = 64, 12


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
    """W 场景：模板×AR(1) 扰动 → K-means 缩减（Q3 MPC 同款）。"""
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
    km = KMeans(n_clusters=K, random_state=0, n_init=5).fit(S)
    w = np.bincount(km.labels_, minlength=K) / len(km.labels_)
    return km.cluster_centers_, w, W


def carbon_scenario(c, pol: list, tau: float) -> dict:
    """碳约束场景：储能层可行则直接求解；不可行则测'上层错峰杠杆'。

    机理（step4.2 实证）：G ≥ D−cap_h 隐含下界（消纳模板锁定）→ 碳排下界
    由负荷时段分布 D 决定 → 错峰（把负荷移到 cap_h 大的小时）可降下界。
    输出: {carbon, cost, lever_needed, lever_cost, per_region}
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
    # 上层杠杆：错峰增强变体（shift_BI/AT → 24h 全窗口价格感知）
    lever_pol = list(pol[:5])
    lever_pol[2] = lever_pol[3] = 24.0
    sched_l = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                             c["params"], tuple(lever_pol))
    ai_l = c["s41"].schedule_occupancy(c, sched_l)
    lev = _solve(ai_l)
    return {"carbon": base["carbon"] if not base["ok"] else None,
            "cost": base["cost"] if not base["ok"] else None,
            "peak_net_MW": base["peak"] if base["ok"] else None,
            "lever_needed": True, "lever_ok": bool(lev and lev["ok"]),
            "lever_cost": (lev["cost"] if lev and lev["ok"] else None),
            "lever_carbon": (lev["carbon"] if lev and lev["ok"] else None),
            "per_region": base["per"],
            "note": "储能层碳排已在下界（消纳模板锁定 G≥D−cap_h）——"
                    "τ 需上层错峰配合；错峰增强变体（shift=24h）为杠杆演示"}


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
               "lever_cost_wan": cs.get("lever_cost")}
        return out
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol[:5]))
    ai_mw = c["s41"].schedule_occupancy(c, sched)
    cost = carbon = qsum = wsum = 0.0
    peak = 0.0
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
            for s_i in range(K):
                ds = dict(d)
                ds["W"] = reps[s_i]
                m = c["s40"].solve_lower_constrained(ds, ch, dh, **kw)
                if m["cost_wan"] is None:
                    continue
                exp_cost += w[s_i] * m["cost_wan"]
                exp_carbon += w[s_i] * m["carbon_t"]
                exp_q += w[s_i] * m["curtail"]
            cost += exp_cost
            carbon += exp_carbon
            qsum += exp_q
            wsum += float(W0.sum())
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
    return {"cost_wan": cost, "carbon_t": carbon, "latency_ms": lat,
            "nu_pct": nu, "peak_net_MW": peak, "infeasible": False}


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

    # 策略变化：每格最优策略（成本最低）vs 基线（q2_compromise）
    change = {}
    for dim, sub in grid.items():
        for frac, cells in sub.items():
            best = min(cells, key=lambda n: (cells[n].get("cost_wan")
                                             if not cells[n].get("infeasible")
                                             else 1e18))
            base = cells.get("q2_compromise", {}).get("cost_wan")
            b = cells.get("q2_compromise", {})
            change[f"{dim}|{frac}"] = {
                "best_policy": best,
                "cost_wan": round(cells[best].get("cost_wan", None), 2)
                if cells[best].get("cost_wan") is not None else None,
                "vs_baseline_cost_pct": round(
                    (cells[best].get("cost_wan", 0)
                     - (b.get("cost_wan", 0) or 0))
                    / max(abs(b.get("cost_wan", 1)), 1e-9) * 100, 2),
                "peak_net_MW": cells[best].get("peak_net_MW", None)}

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
                     ensure_ascii=False, indent=2, default=float)[:2500])


if __name__ == "__main__":
    main()
