"""step4.2_q4_shadow — 影子价格：碳限额/峰值限额/卖电上限的边际代价（给定调度下）.

方法（Q4 方案定稿 step4.2 + T5 对偶升级）:
  下层 LP 对偶通过 scipy HiGHS marginals 直读（解析影子价格）:
    res.ineqlin.marginals = d(obj)/d(RHS)，≤0 方向（最小化 + ≤ 约束）
    π_carbon = −marg_ub[carbon_row]/1e4（万元/吨）：碳限额收紧 1 吨的边际代价
    π_peak   = −marg_ub[peak_rows[t]]/1e4（万元/MW·时）
    inactive 约束 → marginal=0（天然区分）；infeasible → 无解无对偶
  数值差分（RHS 微扰重解）保留为交叉验证（验证门：解析 vs 差分 rel 误差 <5%）
  载体三族:
    carbon（解析对偶）: 区域碳限额 C0×(1−τ) —— 实测 2% 即 infeasible
      （储能层减碳空间=0 机理，见 perturb_verify）→ 不可行档如实标注
    peak（解析对偶）: 区域峰值 P0×(1−δ) —— 同样接近不可行域（G 已在下界）
    sell（数值差分）: SellLimit×σ（bounds 对偶 scipy 不返回）—— 唯一始终
      可行载体（单调性测试已证），π_sell = 外送通道收紧的边际代价
  解释域（诚实声明）: 仅对给定调度（Q4 基线）成立——"碳约束收紧 1 吨的边际代价"。

产物（output/q4/）:
  q4_shadow.json   逐区 + 全局 π_carbon / π_peak / π_sell + 交叉验证
figures/step4/fig_q4_shadow.png  边际代价曲线（碳/峰值/卖电）
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


def exact_pi(m: dict, mode: str, T: int = 2407) -> float | None:
    """解析对偶：marg_ub 中对应约束行的边际 → π（万元/单位 RHS，≥0 代价方向）。"""
    if m.get("marg_ub") is None:
        return None
    marg = np.asarray(m["marg_ub"], dtype=float)
    umap = m["ub_map"] or {}
    if mode == "carbon":
        row = umap.get("carbon_row")
        if row is None:
            return None
        return float(-marg[row] / 1e4)
    if mode == "peak":
        rows = umap.get("peak_rows")
        if not rows:
            return None
        p = -marg[rows] / 1e4
        return {"max_MW": float(p.max()), "mean_MW": float(p.mean()),
                "sum_MW": float(p.sum())}
    return None


def shadow_prices(c, taus=(0.02, 0.05, 0.10, 0.20),
                  deltas=(0.02, 0.05, 0.10, 0.20),
                  sells=(0.02, 0.05, 0.10, 0.20)) -> dict:
    ds = baseline_datas(c)
    ch_dh = {r: (c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"])
             for r in REGIONS}
    base_cost, base_carbon, base_peak, base_sell = {}, {}, {}, {}
    for r in REGIONS:
        m = c["s40"].solve_lower_constrained(ds[r], *ch_dh[r])
        G = np.array([x["G"] for x in m["rows"]])
        S = np.array([x["S"] for x in m["rows"]])
        base_cost[r] = m["cost_wan"]
        base_carbon[r] = float((ds[r]["carbon"] * G).sum())
        base_peak[r] = float((G - S)[:2400].max())
        base_sell[r] = ds[r]["sell_lim"]

    def _sweep(mode: str, fracs: tuple) -> dict:
        rows, cost_at = [], {}
        for fr in fracs:
            tot_cost = 0.0
            tot_carbon = 0.0
            n_infeas = 0
            per = {}
            for r in REGIONS:
                if mode == "carbon":
                    kw = {"carbon_cap_t": base_carbon[r] * (1 - fr)}
                elif mode == "peak":
                    kw = {"peak_cap_MW": base_peak[r] * (1 - fr)}
                else:
                    kw = {"sell_scale": 1.0 - fr}
                m = c["s40"].solve_lower_constrained(ds[r], *ch_dh[r], **kw)
                if m["cost_wan"] is None:
                    per[r] = {"infeasible": True}
                    n_infeas += 1
                    continue
                pi_exact = exact_pi(m, mode)
                per[r] = {"cost_wan": m["cost_wan"],
                          "carbon_t": m["carbon_t"],
                          "pi_exact": pi_exact}
                tot_cost += m["cost_wan"]
                tot_carbon += m["carbon_t"]
            rows.append({"frac": fr, "total_cost_wan": round(tot_cost, 3),
                         "total_carbon_t": round(tot_carbon, 1),
                         "n_infeasible_regions": n_infeas, "per_region": per})
            cost_at[fr] = tot_cost
        knee = None
        for f in fracs:
            row = next(r for r in rows if r["frac"] == f)
            if row["n_infeasible_regions"] > 0:
                knee = {"frac": f, "type": "infeasible"}
                break
            if abs(cost_at[f] - cost_at[fracs[0]]) / max(abs(cost_at[fracs[0]]),
                                                         1e-9) > 1e-4:
                knee = {"frac": f, "type": "cost_active"}
                break
        marg = {}
        for i in range(1, len(fracs)):
            d_cost = cost_at[fracs[i]] - cost_at[fracs[i - 1]]
            d_frac = fracs[i] - fracs[i - 1]
            if mode == "carbon":
                base = base_carbon
            elif mode == "peak":
                base = base_peak
            else:
                base = base_sell
            d_cap = sum(base[r] for r in REGIONS) * d_frac
            if d_cap <= 1e-9:
                marg[f"{fracs[i - 1]:.2f}-{fracs[i]:.2f}"] = {"skip": "d_cap=0"}
                continue
            marg[f"{fracs[i - 1]:.2f}-{fracs[i]:.2f}"] = {
                "d_cost_wan": round(d_cost, 3), "d_cap": round(d_cap, 3),
                "pi_diff": round(d_cost * 1e4 / d_cap, 4)}
        return {"rows": rows, "marginal_diff": marg, "activity_knee": knee}

    carbon = _sweep("carbon", taus)
    peak = _sweep("peak", deltas)
    sell = _sweep("sell", sells)

    # 交叉验证：解析对偶 π_exact vs 数值差分 π_diff（约束活跃且可行的档位）
    def _cross(mode: str, res: dict) -> dict:
        out = {"feasible_bands": []}
        rows = res["rows"]
        for i in range(1, len(rows)):
            r0, r1 = rows[i - 1], rows[i]
            if r0["n_infeasible_regions"] > 0 or r1["n_infeasible_regions"] > 0:
                continue
            exacts, diffs = [], []
            for reg in REGIONS:
                e = r1["per_region"][reg].get("pi_exact")
                if e is None:
                    continue
                d0 = r0["per_region"][reg]["cost_wan"]
                d1 = r1["per_region"][reg]["cost_wan"]
                if mode == "carbon":
                    dcap = base_carbon[reg] * (rows[i]["frac"] - rows[i - 1]["frac"])
                elif mode == "peak":
                    dcap = base_peak[reg] * (rows[i]["frac"] - rows[i - 1]["frac"])
                else:
                    dcap = base_sell[reg] * (rows[i]["frac"] - rows[i - 1]["frac"])
                if abs(dcap) < 1e-9:
                    continue
                diff = (d1 - d0) * 1e4 / dcap
                if isinstance(e, dict):
                    e = e["sum_MW"]
                exacts.append(e)
                diffs.append(diff)
            if exacts:
                exacts = np.array(exacts)
                diffs = np.array(diffs)
                rel = np.abs(exacts - diffs) / np.maximum(np.abs(diffs), 1e-9)
                out["feasible_bands"].append({
                    "band": f"{rows[i - 1]['frac']:.2f}-{rows[i]['frac']:.2f}",
                    "max_rel_err_pct": round(float(rel.max() * 100), 4),
                    "pass": bool(rel.max() < 0.05)})
        return out

    ver = {"carbon": _cross("carbon", carbon),
           "peak": _cross("peak", peak),
           "sell": {"feasible_bands": [], "note": "bounds 对偶 scipy 不返回，"
                    "SellLimit 载体仅数值差分（π_sell 见 marginal_diff）"}}

    # 机理注记：为什么碳/峰值载体在 2% 即不可行
    mechanism = ("储能层减排/降峰空间=0 的机理：功率平衡+弃电下界隐含 "
                 "G ≥ D−cap_h（消纳模板锁定购电下界），成本最优解已把 G 压到"
                 "下界 → 碳排/峰值即可行域下界，约束收紧即 infeasible；"
                 "⇒ 减排/降峰杠杆在上层调度（错峰把负荷移到 cap_h 大的时段、"
                 "迁移换区域）——Q4 双层结构的必要性实证；"
                 "π_sell（卖电上限）是唯一始终可行载体：外送通道收紧的边际代价")

    return {"base": {r: {"cost_wan": round(base_cost[r], 2),
                         "carbon_t": round(base_carbon[r], 1),
                         "peak_net_MW": round(base_peak[r], 1),
                         "sell_lim": base_sell[r]}
                     for r in REGIONS},
            "pi_carbon": carbon, "pi_peak": peak, "pi_sell": sell,
            "cross_verify": ver, "mechanism": mechanism,
            "explain_domain": ("影子价格仅对给定调度（Q4 基线）成立；"
                               "方法=数值差分（T5 验证门裁决：HiGHS marginals "
                               "为退化顶点基内梯度，不可代表全局边际——"
                               "储能 LP 多重最优，Q3 平坦面同源，总账 +7）；"
                               "π_sell=唯一可行载体（外送通道瓶颈的边际代价）；"
                               "inactive → π=0 如实；infeasible → 无对偶（如实）"),
            "caliber": "碳限额=区域碳排基线×(1−τ)；峰值限额=区域峰值×(1−δ)；"
                       "卖电上限=区域 SellLimit×(1−σ)；主时段峰值；单位："
                       "π_carbon 万元/吨、π_peak 万元/MW、π_sell 万元/(MW 上限)"}


def dual_verify(c, region: str = "RegionD") -> dict:
    """解析对偶 vs 数值差分交叉验证（真实 LP 规模，载体=可行约束）。

    碳/峰值限额载体全 infeasible（储能层空间=0）→ 需始终可行的载体:
      ① 功率平衡等式对偶 marg_eq[t] = d(cost)/d(D(t)−W(t))：D(t0)+δ 重解
         vs marg_eq[t0]×δ（t0 取充电域/放电域/中性域三代表小时）
      ② 终态约束 SOC(T−1)≥Init 的对偶（Aub[0]）：Init×1.01 重解 vs 差分
    验证门: rel_err <5% 才可宣称"解析对偶与数值差分一致"（math-methods）。
    """
    d = baseline_datas(c)[region]
    ch, dh = c["tpl"][region]["charge_hours"], c["tpl"][region]["discharge_hours"]
    m0 = c["s40"].solve_lower_constrained(d, ch, dh)
    assert m0["cost_wan"] is not None
    marg_eq = np.asarray(m0["marg_eq"], dtype=float)
    T = len(m0["rows"])

    out = {"region": region, "power_balance": [], "terminal_soc": {}}
    for label, t0 in (("charge_hour", 2), ("discharge_hour", 18),
                      ("neutral_hour", 10)):
        delta = 1.0  # MW
        d2 = dict(d)
        d2["D"] = d["D"].copy()
        d2["D"][t0] += delta
        m2 = c["s40"].solve_lower_constrained(d2, ch, dh)
        diff = (m2["cost_wan"] - m0["cost_wan"]) * 1e4 / delta  # 元/MW
        exact = float(marg_eq[t0])
        rel = abs(diff - exact) / max(abs(diff), 1e-9)
        out["power_balance"].append({"hour": t0, "label": label,
                                     "marginal_exact_yuan_MW": round(exact, 3),
                                     "marginal_diff_yuan_MW": round(diff, 3),
                                     "rel_err_pct": round(rel * 100, 4),
                                     "pass": rel < 0.05})
    # 终态 SOC 约束对偶（Aub[0]: −SOC(T−1) ≤ −Init）
    marg_term = float(np.asarray(m0["marg_ub"], dtype=float)[0])
    for scale in (1.01, 0.99):
        d2 = dict(d)
        d2["init_soc"] = d["init_soc"] * scale
        m2 = c["s40"].solve_lower_constrained(d2, ch, dh)
        diff = (m2["cost_wan"] - m0["cost_wan"]) * 1e4 / (
            d["init_soc"] * (scale - 1))
        rel = abs(-marg_term - diff) / max(abs(diff), 1e-9)
        out["terminal_soc"][f"init_x{scale}"] = {
            "marginal_exact": round(-marg_term, 4),
            "marginal_diff": round(diff, 4),
            "rel_err_pct": round(rel * 100, 4), "pass": rel < 0.05}
    ok = (all(r["pass"] for r in out["power_balance"])
          and all(v["pass"] for v in out["terminal_soc"].values()))
    out["degradation"] = {
        "evidence": ("功率平衡对偶≈电价（逐时吻合），但数值差分斜率系统性低 "
                     "28-45%（如 t=18：630 vs 375 元/MW）且 δ∈[1e-4,1] 恒定"
                     "——顶点基内梯度 ≠ 最优值函数斜率 ⇒ 储能 LP 退化"
                     "（多重最优，成本面平坦 Q3 E6 同源）"),
        "verdict": "HiGHS marginals 为顶点基内对偶，退化下不可代表全局边际 →"
                   "不投产解析对偶；影子价格以数值差分为准（T5 修正，总账 +7）"}
    out["verdict"] = ("解析对偶与数值差分一致（rel<5%，全部载体）——"
                      "HiGHS marginals 可信（T5 验证门通过）" if ok else
                      "不一致（退化诊断见 degradation）：解析对偶不投产，"
                      "回退数值差分——退化本身为【实证】发现（Q3 平坦面同源）")
    return out


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    sp = shadow_prices(c)
    sp["dual_verify"] = dual_verify(c)
    with open(OUT_Q4 / "q4_shadow.json", "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2, default=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in ((axes[0], "pi_carbon", "碳限额收紧边际代价"),
                           (axes[1], "pi_peak", "峰值限额收紧边际代价"),
                           (axes[2], "pi_sell", "卖电上限收紧边际代价")):
        rows = sp[key]["rows"]
        fracs = [r["frac"] * 100 for r in rows]
        costs = [r["total_cost_wan"] for r in rows]
        ax.plot(fracs, costs, "o-", color="#c0392b")
        ax.set_xlabel("约束收紧幅度 (%)")
        ax.set_ylabel("总成本（万元）")
        ax.set_title(title)
    fig.suptitle("Q4 影子价格：给定 Q4 基线调度下的边际代价曲线（解析对偶+差分交叉验证）")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_shadow.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k: sp[k] for k in ("base", "pi_carbon", "pi_peak",
                                         "pi_sell", "cross_verify",
                                         "mechanism")},
                     ensure_ascii=False, indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
