"""step3.0_lp_baseline — Q3 铺路：储能协同确定性 LP 基准（框架 + 标定）.

范围（M4 前铺路，不求解全量）: 1 区域 × 全时段标定 + 求解器计时外推；
全量 6 区域求解留给 M4（step3.1 场景 MPC 的前置）。

口径（附件1 + 说明4）:
  给定负荷 = Baseline_AI_IT_Load + NonAI_IT_Load（任务调度固定，来自题目基线）
  设施负荷 D = IT_Load × PUE(r)
  功率平衡 G + W_used + Pd = D + Pc + S + Q（Q=弃电松弛，目标惩罚）
  SOC(t) = SOC(t−1) + ηc·Pc(t) − Pd(t)/ηd；SOC(2406) ≥ InitialSOC
  效率分区制（step0.5: A/B/C 0.93/0.92; D/E/F 0.94/0.93）
  消纳 U = min(W, c_r(h)·D) 模板口径（R1）——Q3 基线对照口径
  目标 min Cost₃ = Σ λ·G − Σ λs·S（碳排/峰值作为评估指标事后计算）

产物（output/q3/）: lp_calibration.json（1 区域标定 + 计时外推）+
  lp_baseline_1region.csv（1 区域逐时决策明细）
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

from step0_config import CLEAN, DATA_RAW, FILES, OUTPUT, REGIONS, SETTLE_HOUR

OUT_Q3 = OUTPUT / "q3"

ETA_C = {"RegionA": 0.93, "RegionB": 0.93, "RegionC": 0.93,
         "RegionD": 0.94, "RegionE": 0.94, "RegionF": 0.94}
ETA_D = {"RegionA": 0.92, "RegionB": 0.92, "RegionC": 0.92,
         "RegionD": 0.93, "RegionE": 0.93, "RegionF": 0.93}
HOURS = 2407


def load_region_data(r: str) -> dict:
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    sp = pd.read_csv(CLEAN / "storage_params.csv").set_index("Region").loc[r]
    sub = rt[rt.Region == r].sort_values("Hour")
    it = (sub["Baseline_AI_IT_Load_MW"] + sub["NonAI_IT_Load_MW"]).to_numpy()
    pue = float(pd.read_excel(DATA_RAW / FILES["gpu"],
                              sheet_name="GPU中心基础情况")
                .set_index("Region").loc[r, "PUE"])
    return {
        "D": it * pue,
        "W": sub["AvailableRenewable_MW"].to_numpy(),
        "c_h": None,   # 由调用方注入（fit_consume_ratio 模板）
        "price": sub["ElectricityPrice_CNY_per_MWh"].to_numpy(),
        "sellp": sub["SellPrice_CNY_per_MWh"].to_numpy(),
        "carbon": sub["CarbonIntensity_tCO2_per_MWh"].to_numpy(),
        "cap_mwh": float(sp["StorageCapacity_MWh"]),
        "min_soc": float(sp["MinSOC_MWh"]),
        "init_soc": float(sp["InitialSOC_MWh"]),
        "max_c": float(sp["MaxChargePower_MW"]),
        "max_d": float(sp["MaxDischargePower_MW"]),
        "sell_lim": float(sp["SellLimit_MW"]),
        "max_import": float(sp["MaxGridImport_MW"]),
        "eta_c": ETA_C[r], "eta_d": ETA_D[r],
    }


def solve_region(d: dict, n_hours: int = HOURS) -> dict:
    """单区域 LP（scipy HiGHS）——附件1 口径终版.

    ⚠️ DEPRECATED：Q3 主流程已迁移至 step3.3+_q3_model_evolve.solve_region_timed
    （含 Closure 口径修复/时段约束/斜坡/互斥等全部参数化能力）。
    本函数保留用于 lp_baseline 历史产物追溯，不再作为新实验入口。

    平衡式: G + W + Pd = D + Pc + S + Q（W=可用新能源全进，Q=弃电松弛）
    利用率 = 1 − Q/W（(消纳+充电+外送)/W 等价）
    消纳能力（R1 实证）: 直接消纳 ≤ min(W, c(h)·D) → 弃电下界
      Q ≥ W − min(W, c(h)·D) − Pc − S（充电/外送可吃弃电=弃电充电抓手）
    变量: Pc, Pd, G, S, Q, SOC（6T）
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    T = n_hours
    nv = 6 * T
    idx = lambda k, t: k * T + t
    c = np.zeros(nv)
    for t in range(T):
        c[idx(2, t)] = d["price"][t]
        c[idx(3, t)] = -d["sellp"][t]
        c[idx(4, t)] = 1e-4
    Aeq = lil_matrix((2 * T, nv))
    beq = np.zeros(2 * T)
    for t in range(T):
        row = t
        Aeq[row, idx(1, t)] = 1.0
        Aeq[row, idx(2, t)] = 1.0
        for k in (0, 3, 4):
            Aeq[row, idx(k, t)] = -1.0
        beq[row] = d["D"][t] - d["W"][t]
        row = T + t
        Aeq[row, idx(5, t)] = 1.0
        Aeq[row, idx(5, t - 1)] = -1.0 if t > 0 else 0.0
        Aeq[row, idx(0, t)] = -d["eta_c"]
        Aeq[row, idx(1, t)] = 1.0 / d["eta_d"]
        beq[row] = d["init_soc"] if t == 0 else 0.0
    # 不等式: SOC(T-1) ≥ init + 弃电下界（消纳能力）
    n_ineq = 1 + T
    Aub = lil_matrix((n_ineq, nv))
    bub = np.zeros(n_ineq)
    Aub[0, idx(5, T - 1)] = -1.0
    bub[0] = -d["init_soc"]
    # 口径（sum_10 修复 #1 对齐 solve_region_timed）：主时段消纳=R1 模板 c_h·D；
    # Closure 段 2400-2406 = min(W, D)（R1 Closure 口径，全消纳）
    cap_h = np.minimum(d["W"][:T], d["c_h"][np.arange(T) % 24] * d["D"][:T])
    closure = np.arange(T) >= 2400
    cap_h[closure] = np.minimum(d["W"][:T], d["D"][:T])[closure]
    for t in range(T):
        Aub[1 + t, idx(0, t)] = -1.0          # -Pc
        Aub[1 + t, idx(3, t)] = -1.0          # -S
        Aub[1 + t, idx(4, t)] = -1.0          # -Q
        bub[1 + t] = -(d["W"][t] - cap_h[t])  # -(W - 消纳能力)
    bounds = [(0, d["max_c"])] * T + [(0, d["max_d"])] * T \
        + [(0, d["max_import"])] * T + [(0, d["sell_lim"])] * T \
        + [(0, None)] * T + [(d["min_soc"], d["cap_mwh"])] * T
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"status": res.status, "cost_wan": None, "carbon_t": None,
                "curtail": None, "rows": [], "solve_s": None,
                "message": res.message}
    x = res.x
    rows = [{"Hour": t, "G": float(x[idx(2, t)]), "Pc": float(x[idx(0, t)]),
             "Pd": float(x[idx(1, t)]), "S": float(x[idx(3, t)]),
             "Q": float(x[idx(4, t)]), "SOC": float(x[idx(5, t)])}
            for t in range(T)]
    G = x[[idx(2, t) for t in range(T)]]
    cost = float((d["price"][:T] * G
                  - d["sellp"][:T] * x[[idx(3, t) for t in range(T)]]).sum())
    carbon = float((d["carbon"][:T] * G).sum())
    curtail = float(x[[idx(4, t) for t in range(T)]].sum())
    nu = 1 - curtail / max(d["W"][:T].sum(), 1e-9)
    return {"status": res.status, "cost_wan": cost / 1e4,
            "carbon_t": carbon, "curtail": curtail, "nu": float(nu),
            "rows": rows, "solve_s": None, "message": ""}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    r = "RegionD"                       # 标定区域（西部算力中心，弃电充电代表性）
    d = load_region_data(r)
    # R1 消纳能力模板（c(h)·D）注入（口径修复：防 LP 自由消纳虚高利用率）
    import importlib.util
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    d["c_h"] = np.array(consume[r], dtype=float)
    # 标定：24h → 168h → 全时段（计时外推）
    calib = {}
    for n in (24, 168, HOURS):
        t0 = time.time()
        res = solve_region(d, n)
        dt = time.time() - t0
        calib[f"{n}h"] = {"status": res["status"], "solve_s": round(dt, 1),
                          "cost_wan": round(res["cost_wan"], 2),
                          "curtail_MWh": round(res["curtail"], 2)}
        print(f"[calib] {r} {n}h: {dt:.1f}s cost={res['cost_wan']:.0f}万 "
              f"status={res['status']}", flush=True)
        if n == HOURS:
            pd.DataFrame(res["rows"]).to_csv(
                OUT_Q3 / "lp_baseline_1region.csv", index=False)
            full = res
    report = {
        "region": r, "calibration": calib,
        "extrapolate_6regions_min": round(
            calib[f"{HOURS}h"]["solve_s"] * 6 / 60, 1),
        "caliber": ("给定负荷=Baseline_AI+NonAI；消纳模板口径（R1）；"
                    "效率分区制（step0.5）；SOC(2406)≥Initial；"
                    "弃电松弛 Q 目标惩罚 1e-4；pulp/CBC"),
        "note": ("Q3 铺路（M4 前置）：LP 模型构建+标定+计时外推；"
                 "全量 6 区域与 MPC/Sobol 留 M4"),
    }
    with open(OUT_Q3 / "lp_calibration.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def solve_all_regions() -> dict:
    """六区域 LP 全量（探路扩大，用户批准：前期经验，后续可删）。"""
    import importlib.util
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    out = {}
    for r in REGIONS:
        d = load_region_data(r)
        d["c_h"] = np.array(consume[r], dtype=float)
        res = solve_region(d)
        out[r] = {"status": res["status"], "cost_wan": res["cost_wan"],
                  "carbon_t": res["carbon_t"], "curtail_MWh": res["curtail"],
                  "nu": res["nu"]}
        if res["rows"]:
            pd.DataFrame(res["rows"]).to_csv(
                OUT_Q3 / f"lp_baseline_{r}.csv", index=False)
        print(f"[LP] {r}: cost={res['cost_wan']:.0f}万 "
              f"curtail={res['curtail']:.0f}MWh nu={res['nu']:.2f}",
              flush=True)
    with open(OUT_Q3 / "lp_all_regions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        solve_all_regions()
    else:
        main()
