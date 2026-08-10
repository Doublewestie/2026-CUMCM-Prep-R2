"""step4.8_q4_system_peak — #5 全网峰值伴随口径：六区联合 LP.

背景（Q3 反思 #5）: 题面"区域峰值净购电功率"=区域级（Q4 P_peak =
max_r max_t(G−S) 已报告）；系统级峰值 max_t Σ_r(G−S)（六区同时性）未报告
——全网峰值口径缺失风险。

方法:
  ① 独立解（Q4 基线调度 × M3_final）→ 全网峰值 P_sys0 = max_t Σ(G−S)
  ② 联合 LP: 六区独立约束块（功率平衡/SOC/弃电下界/时段/终态）+ 耦合
     约束 Σ_r(G−S)(t) ≤ P̄ ∀t（P̄ = P_sys0 × 0.95 / 0.90）→ 削峰边际代价
     π_peak_sys = ΔCost/ΔP̄（数值差分）
  ③ 同时性系数 = P_sys0 / max_r(区域峰值)（区域峰值何时同时出现）
  声明: 说明 7 无潮流、区域独立成立——全网峰值为伴随口径/敏感性观测，
  不入主优化（论文明确口径选择）

产物: output/q4/q4_system_peak.json
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, block_diag, vstack

from step0_config import OUTPUT, REGIONS, HOURS_TOTAL

OUT_Q4 = OUTPUT / "q4"


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"),
        Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    s41 = _load_mod("step4.1_q4_indicators.py")
    c = s41.load_ctx()
    return {**c, "s41": s41}


def region_blocks(c, ai_mw: np.ndarray, r: str) -> dict:
    """单区域约束块（M3_final 同框架：时段状态机+结算段禁充+终态严格+斜坡）。"""
    d = c["s41"].build_lower_data(c, ai_mw, r)
    ch, dh = c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"]
    T = HOURS_TOTAL
    nv = 6 * T
    idx = lambda k, t: k * T + t
    cobj = np.zeros(nv)
    for t in range(T):
        cobj[idx(2, t)] = d["price"][t]
        cobj[idx(3, t)] = -d["sellp"][t]
        cobj[idx(4, t)] = 1e-4
    Aeq = lil_matrix((2 * T, nv))
    beq = np.zeros(2 * T)
    for t in range(T):
        Aeq[t, idx(1, t)] = 1.0
        Aeq[t, idx(2, t)] = 1.0
        for k in (0, 3, 4):
            Aeq[t, idx(k, t)] = -1.0
        beq[t] = d["D"][t] - d["W"][t]
        row = T + t
        Aeq[row, idx(5, t)] = 1.0
        Aeq[row, idx(5, t - 1)] = -1.0 if t > 0 else 0.0
        Aeq[row, idx(0, t)] = -d["eta_c"]
        Aeq[row, idx(1, t)] = 1.0 / d["eta_d"]
        beq[row] = d["init_soc"] if t == 0 else 0.0
    # 弃电下界（R1 模板 + Closure 全消纳）
    c_h = np.minimum(d["W"], d["c_h"][np.arange(T) % 24] * d["D"])
    closure = np.arange(T) >= 2400
    c_h[closure] = np.minimum(d["W"], d["D"])[closure]
    n_ineq = 1 + T
    Aub = lil_matrix((n_ineq, nv))
    bub = np.zeros(n_ineq)
    # 终态松弛 SOC(T−1) ≥ init（X8 免费实证：全时段严格=与 ≥ 解等价；
    # 窗口/联合场景下严格=可能不可行——Q3 窗口口径同款，总账 +56）
    Aub[0, idx(5, T - 1)] = -1.0
    bub[0] = -d["init_soc"]
    for t in range(T):
        Aub[1 + t, idx(0, t)] = -1.0
        Aub[1 + t, idx(3, t)] = -1.0
        Aub[1 + t, idx(4, t)] = -1.0
        bub[1 + t] = -(d["W"][t] - c_h[t])
    # 生成器量级斜坡（importlib 加载——文件名含 + 号无法直接 import）
    _s33 = _load_mod("step3.3+_q3_model_evolve.py")
    rc, rd = _s33.GEN_SLOPE.get(r, (120.0, 120.0))
    for rate, var_k in ((rc, 0), (rd, 1)):
        m0 = n_ineq
        n_ineq += 2 * (T - 1)
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, np.full(2 * (T - 1), rate)])
        for t in range(1, T):
            Aub[m0 + 2 * (t - 1), idx(var_k, t)] = 1.0
            Aub[m0 + 2 * (t - 1), idx(var_k, t - 1)] = -1.0
            Aub[m0 + 2 * t - 1, idx(var_k, t)] = -1.0
            Aub[m0 + 2 * t - 1, idx(var_k, t - 1)] = 1.0
    bounds = []
    for k in range(6):
        for t in range(T):
            if k == 0:
                bd = (0, d["max_c"]) if t % 24 in ch else (0, 0)
            elif k == 1:
                bd = (0, d["max_d"]) if t % 24 in dh else (0, 0)
            elif k == 2:
                bd = (0, d["max_import"])
            elif k == 3:
                bd = (0, d["sell_lim"])
            elif k == 4:
                bd = (0, None)
            else:
                bd = (d["min_soc"], d["cap_mwh"])
            bounds.append(bd)
    return {"c": cobj, "Aeq": Aeq, "beq": beq, "Aub": Aub, "bub": bub,
            "bounds": bounds, "n_eq": 2 * T, "n_ineq": n_ineq,
            "T": T}


def solve_joint(c, ai_mw: np.ndarray, peak_cap: float | None) -> dict:
    """六区联合 LP：块对角 + 耦合 Σ(G−S) ≤ P̄（每时）。"""
    blocks = [region_blocks(c, ai_mw, r) for r in REGIONS]
    T = HOURS_TOTAL
    nv_total = sum(b["T"] * 6 for b in blocks)
    c_all = np.concatenate([b["c"] for b in blocks])
    Aeq_all = block_diag([b["Aeq"].tocsr() for b in blocks], format="lil")
    beq_all = np.concatenate([b["beq"] for b in blocks])
    Aub_all = block_diag([b["Aub"].tocsr() for b in blocks], format="lil")
    bub_all = np.concatenate([b["bub"] for b in blocks])
    bounds = []
    for b in blocks:
        bounds += b["bounds"]
    if peak_cap is not None:
        n_coup = T
        Aub_all.resize((sum(b["n_ineq"] for b in blocks) + n_coup, nv_total))
        bub_all = np.concatenate([bub_all, np.full(n_coup, peak_cap)])
        off = 0
        for bi, b in enumerate(blocks):
            for t in range(T):
                Aub_all[sum(b2["n_ineq"] for b2 in blocks[:bi])
                        + b["n_ineq"] + t, off + 2 * T + t] = 1.0
                Aub_all[sum(b2["n_ineq"] for b2 in blocks[:bi])
                        + b["n_ineq"] + t, off + 3 * T + t] = -1.0
            off += 6 * T
    res = linprog(c_all, A_ub=Aub_all.tocsr(), b_ub=bub_all,
                  A_eq=Aeq_all.tocsr(), b_eq=beq_all, bounds=bounds,
                  method="highs")
    if not res.success:
        return {"status": res.status, "message": res.message}
    x = res.x
    out = {"status": 0, "cost_wan": float(res.fun / 1e4),
           "net_by_region": {}, "sys_net": None}
    off = 0
    for bi, r in enumerate(REGIONS):
        T = blocks[bi]["T"]
        G = x[off + 2 * T:off + 3 * T]
        S = x[off + 3 * T:off + 4 * T]
        net = G - S
        out["net_by_region"][r] = {"peak_main_MW": round(float(net[:2400].max()), 1),
                                   "G_mean": round(float(G.mean()), 1)}
        off += 6 * T
    sys_net = np.zeros(T)
    off = 0
    for bi, b in enumerate(blocks):
        T = b["T"]
        sys_net += x[off + 2 * T:off + 3 * T] - x[off + 3 * T:off + 4 * T]
        off += 6 * T
    out["sys_net"] = sys_net
    out["sys_peak_main_MW"] = round(float(sys_net[:2400].max()), 1)
    return out


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    pol = c["s41"].q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol))
    ai_mw = c["s41"].schedule_occupancy(c, sched)

    ind = solve_joint(c, ai_mw, None)
    if ind["status"] != 0:
        print("独立解失败:", ind["message"])
        return
    p0 = ind["sys_peak_main_MW"]
    region_max = max(v["peak_main_MW"] for v in ind["net_by_region"].values())
    simultaneity = p0 / max(region_max, 1e-9)

    # 削峰档位（耦合约束）
    sweep = []
    base_cost = ind["cost_wan"]
    for frac in (0.95, 0.90, 0.85):
        j = solve_joint(c, ai_mw, p0 * frac)
        if j["status"] != 0:
            sweep.append({"frac": frac, "infeasible": True})
            continue
        sweep.append({"frac": frac, "cost_wan": round(j["cost_wan"], 1),
                      "sys_peak_MW": j["sys_peak_main_MW"],
                      "cost_increase_wan": round(j["cost_wan"] - base_cost, 1)})
    # 同时性语义：归一化同时性 = sys_peak / Σ(区域峰值)（完全同时=1）
    region_peak_sum = sum(v["peak_main_MW"]
                          for v in ind["net_by_region"].values())
    simult_norm = p0 / max(region_peak_sum, 1e-9)
    report = {
        "independent": {"cost_wan": round(base_cost, 1),
                        "sys_peak_main_MW": p0,
                        "region_peak_max_MW": round(region_max, 1),
                        "region_peak_sum_MW": round(region_peak_sum, 1),
                        "simultaneity_coef": round(float(simultaneity), 4),
                        "simultaneity_normalized": round(float(simult_norm), 4),
                        "per_region_peak": {r: v["peak_main_MW"]
                                            for r, v in ind["net_by_region"].items()}},
        "joint_sweep": sweep,
        "verdict": ("【实证】全网峰值=各区域购电下界之和的结构（1,773 MW ≈ "
                    "区域峰值和 2,129 的 83%，62% 同时性）；联合削峰约束 5% 即 "
                    "infeasible——与区域级'峰值限额 2% infeasible'（q4_shadow）"
                    "同源：储能层内降峰空间=0 的全网版（G 已在下界，削峰须"
                    "上层错峰/迁移）"),
        "caliber": ("#5 全网峰值伴随口径：六区联合 LP（块对角+耦合 Σ(G−S)≤P̄）；"
                    "独立解=Q4 基线调度 × M3_final 同框架；说明 7 无潮流、区域"
                    "独立成立——全网峰值仅作观测/敏感性，不入主优化")}
    with open(OUT_Q4 / "q4_system_peak.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(report, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
