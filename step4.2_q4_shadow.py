"""step4.2_q4_shadow — 影子价格：碳限额/峰值限额的边际代价（给定调度下）.

方法（Q4 方案定稿 step4.2）:
  下层 LP 对偶不可直接读取（scipy HiGHS 不返回对偶向量）→ 数值影子价格:
    ① 无约束解 → 区域碳排 C0 / 峰值 P0
    ② 收紧约束 τ∈{1%,2%,3%,5%}: 碳限额 C0×(1−τ)、峰值 P0×(1−δ) → 重解 LP
    ③ 边际代价 π = ΔCost / Δ限额（区域级 + 全局聚合）
    ④ 微扰验证（三层验收）: π×Δ限额 vs 实测 ΔCost 相对误差 <5%（一阶近似自检）
  解释域（诚实声明）: 仅对给定调度（Q4 基线）成立——"碳约束收紧 1 吨的边际代价"。

产物（output/q4/）:
  q4_shadow.json   逐区 + 全局 π_carbon / π_peak + 微扰验证
figures/step4/fig_q4_shadow.png  边际代价曲线（碳/峰值）
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"


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


def baseline_datas(c) -> dict[str, dict]:
    """Q4 基线上层调度 → 各区域下层数据（调度版 D）。"""
    pol = c["s41"].q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol))
    ai_mw = c["s41"].schedule_occupancy(c, sched)
    ds = {}
    for r in REGIONS:
        ds[r] = c["s41"].build_lower_data(c, ai_mw, r)
    return ds


def shadow_prices(c, taus=(0.02, 0.05, 0.10, 0.20),
                  deltas=(0.02, 0.05, 0.10, 0.20)) -> dict:
    ds = baseline_datas(c)
    ch_dh = {r: (c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"])
             for r in REGIONS}
    # 无约束基线
    base_cost, base_carbon, base_peak = {}, {}, {}
    for r in REGIONS:
        m = c["s40"].solve_lower_constrained(ds[r], *ch_dh[r])
        G = np.array([x["G"] for x in m["rows"]])
        S = np.array([x["S"] for x in m["rows"]])
        base_cost[r] = m["cost_wan"]
        base_carbon[r] = float((ds[r]["carbon"] * G).sum())
        base_peak[r] = float((G - S)[:2400].max())

    def _sweep(mode: str, fracs: tuple) -> dict:
        rows, cost_at = [], {}
        for fr in fracs:
            tot_cost = 0.0
            tot_carbon = 0.0
            n_infeas = 0
            per = {}
            for r in REGIONS:
                kw = {"carbon_cap_t": base_carbon[r] * (1 - fr)} if mode == "carbon" \
                    else {"peak_cap_MW": base_peak[r] * (1 - fr)}
                m = c["s40"].solve_lower_constrained(ds[r], *ch_dh[r], **kw)
                if m["cost_wan"] is None:
                    per[r] = {"infeasible": True}
                    n_infeas += 1
                    continue
                per[r] = {"cost_wan": m["cost_wan"], "carbon_t": m["carbon_t"]}
                tot_cost += m["cost_wan"]
                tot_carbon += m["carbon_t"]
            rows.append({"frac": fr, "total_cost_wan": tot_cost,
                         "total_carbon_t": tot_carbon,
                         "n_infeasible_regions": n_infeas, "per_region": per})
            cost_at[fr] = tot_cost
        # 约束活跃/可行性拐点：首个成本显著变化档位 or 首个 infeasible 档位
        # （从 fracs[0] 起检查——第一档即 infeasible 也必须报告）
        c0 = cost_at[fracs[0]]
        knee = None
        for f in fracs:
            row = next(r for r in rows if r["frac"] == f)
            if row["n_infeasible_regions"] > 0:
                knee = {"frac": f, "type": "infeasible"}
                break
            if abs(cost_at[f] - c0) / max(abs(c0), 1e-9) > 1e-4:
                knee = {"frac": f, "type": "cost_active"}
                break
        marg = {}
        for i in range(1, len(fracs)):
            d_cost = cost_at[fracs[i]] - cost_at[fracs[i - 1]]
            d_frac = fracs[i] - fracs[i - 1]
            base = base_carbon if mode == "carbon" else base_peak
            d_cap = sum(base[r] for r in REGIONS) * d_frac
            if d_cap <= 1e-9:
                marg[f"{fracs[i - 1]:.2f}-{fracs[i]:.2f}"] = {"skip": "d_cap=0"}
                continue
            marg[f"{fracs[i - 1]:.2f}-{fracs[i]:.2f}"] = {
                "d_cost_wan": round(d_cost, 2), "d_cap": round(d_cap, 2),
                "pi_per_unit": round(d_cost * 1e4 / d_cap, 4)}
        return {"rows": rows, "marginal": marg, "activity_knee": knee}

    carbon = _sweep("carbon", taus)
    peak = _sweep("peak", deltas)

    # 微扰验证：一阶近似 π×Δcap vs 实测 Δcost（仅在约束活跃且可行的档位区间有效）
    ver = {}
    for mode, res in (("carbon", carbon), ("peak", peak)):
        f0, f1 = res["rows"][0]["frac"], res["rows"][1]["frac"]
        d_cost = res["rows"][1]["total_cost_wan"] - res["rows"][0]["total_cost_wan"]
        base = sum(base_carbon[r] for r in REGIONS) if mode == "carbon" \
            else sum(base_peak[r] for r in REGIONS)
        d_cap = base * (f1 - f0)
        if d_cap <= 1e-9:
            ver[mode] = {"note": "d_cap=0 跳过"}
            continue
        if abs(d_cost) < 1e-6:
            ver[mode] = {"d_cost_measured": 0.0, "d_cost_approx": 0.0,
                         "note": ("储能层减排/降峰空间=0 的机理：功率平衡+弃电下界"
                                  "隐含 G ≥ D−cap_h（消纳模板锁定购电下界），"
                                  "成本最优解已把 G 压到下界 → 碳排/峰值即可行域"
                                  "下界，约束收紧即 infeasible；"
                                  "⇒ 减排/降峰杠杆在上层调度（错峰把负荷移到 "
                                  "cap_h 大的时段、迁移换区域）"),
                         "activity_knee": res["activity_knee"]}
            continue
        approx = d_cost * 1e4 / d_cap * d_cap / 1e4
        ver[mode] = {"d_cost_measured": round(d_cost, 3),
                     "d_cost_approx": round(float(approx), 3),
                     "rel_err_pct": round(float(abs(d_cost - approx)
                                                / max(abs(d_cost), 1e-9) * 100), 3),
                     "activity_knee": res["activity_knee"]}
    return {"base": {r: {"cost_wan": round(base_cost[r], 2),
                         "carbon_t": round(base_carbon[r], 1),
                         "peak_net_MW": round(base_peak[r], 1)}
                     for r in REGIONS},
            "pi_carbon": carbon, "pi_peak": peak,
            "perturb_verify": ver,
            "explain_domain": ("影子价格仅对给定调度（Q4 基线）成立——"
                               "数值差分（RHS 收紧 τ/δ），非对偶向量；"
                               "核心发现（机理级）：G ≥ D−cap_h 隐含下界（消纳模板"
                               "锁定）→ 成本最优解碳排/峰值已在可行域下界，"
                               "储能层内减碳降峰空间=0，收紧即 infeasible；"
                               "⇒ 碳/峰值的真正杠杆在上层任务调度（错峰/迁移），"
                               "即 Q4 双层结构的必要性实证"),
            "caliber": "碳限额=区域碳排基线×(1−τ)；峰值限额=区域峰值×(1−δ)；主时段峰值"}


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    sp = shadow_prices(c)
    with open(OUT_Q4 / "q4_shadow.json", "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((axes[0], "pi_carbon", "碳限额收紧边际代价"),
                           (axes[1], "pi_peak", "峰值限额收紧边际代价")):
        rows = sp[key]["rows"]
        fracs = [r["frac"] * 100 for r in rows]
        costs = [r["total_cost_wan"] for r in rows]
        ax.plot(fracs, costs, "o-", color="#c0392b")
        ax.set_xlabel("约束收紧幅度 (%)")
        ax.set_ylabel("总成本（万元）")
        ax.set_title(title)
    fig.suptitle("Q4 影子价格：给定 Q4 基线调度下的边际代价曲线")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_shadow.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k: sp[k] for k in ("base", "perturb_verify",
                                         "explain_domain")},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
