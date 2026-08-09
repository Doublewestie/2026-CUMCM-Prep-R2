"""step3.7+_q3_mutex.py — 充放互斥 MILP（M0x）：储能传送带假象的物理性修复.

实证（step3.7 交叉实验深化发现）:
  M0（全时段自由 LP）解中 96.7%（D 区）/91%（A 区）小时同时充放——
  "充电吃弃电（绕过消纳模板）+ 放电供负荷"同步进行 → 吞吐突破 SOC 容量限制
  （每天充 6,008 MWh vs 物理每日净容量 810）——同一电池物理上不可能同时充放。
  此"传送带"制造了 M0−M1 的 1.72 亿红利中的大部分（不物理）。

修复: scipy.optimize.milp（HiGHS 混合整数）——二进制 z_t 强制充放互斥:
  Pc(t) ≤ MaxC·z_t,  Pd(t) ≤ MaxD·(1−z_t),  z_t∈{0,1}
  → M0x（物理上界）vs M0（不物理上界）vs M1（时段约束）三分:
    M0x − M1 = 真实时段自由度红利（互斥下）
    M0 − M0x = 传送带假象量

产物: output/q3/q3_mutex.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

from step0_config import OUTPUT, REGIONS

OUT_Q3 = OUTPUT / "q3"


def _setup():
    root = Path(__file__).resolve().parent
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s33p", root / "step3.3+_q3_model_evolve.py")
    s33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s33)
    spec2 = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s10)
    rt = s33._import_step32().load_rt()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    tpl = json.loads((OUT_Q3 / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"]
    return s33, rt, consume, tpl


def solve_mutex(d, charge_hours=None, discharge_hours=None, T: int = 2407) -> dict:
    """充放互斥 LP/MILP（z_t 二进制；charge_hours=None 全自由=M0x 物理上界）。"""
    nv = 7 * T  # 6T 连续 + T 二进制 z
    idx = lambda k, t: k * T + t
    c = np.zeros(nv)
    for t in range(T):
        c[idx(2, t)] = d["price"][t]
        c[idx(3, t)] = -d["sellp"][t]
        c[idx(4, t)] = 1e-4
    # 等式: 功率平衡 + SOC 递推
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
    # 不等式: 终态 + 弃电下界 + 互斥（big-M）
    n_ineq = 1 + T + 2 * T
    Aub = lil_matrix((n_ineq, nv))
    bub = np.zeros(n_ineq)
    Aub[0, idx(5, T - 1)] = -1.0
    bub[0] = -d["init_soc"]
    cap_h = np.minimum(d["W"][:T], d["c_h"][np.arange(T) % 24] * d["D"][:T])
    closure = np.arange(T) >= 2400
    cap_h[closure] = np.minimum(d["W"][:T], d["D"][:T])[closure]
    for t in range(T):
        Aub[1 + t, idx(0, t)] = -1.0
        Aub[1 + t, idx(3, t)] = -1.0
        Aub[1 + t, idx(4, t)] = -1.0
        bub[1 + t] = -(d["W"][t] - cap_h[t])
        # 互斥: Pc ≤ MaxC·z ; Pd ≤ MaxD·(1−z)
        Aub[1 + T + 2 * t, idx(0, t)] = 1.0
        Aub[1 + T + 2 * t, idx(6, t)] = -d["max_c"]
        bub[1 + T + 2 * t] = 0.0
        Aub[1 + T + 2 * t + 1, idx(1, t)] = 1.0
        Aub[1 + T + 2 * t + 1, idx(6, t)] = d["max_d"]
        bub[1 + T + 2 * t + 1] = d["max_d"]
    # 时段约束（可选：M1 式时段限制）；Q 上界用 np.inf（None 会进 Bounds → HiGHS model error）
    bounds = [(0, d["max_c"]) if (charge_hours is None or t % 24 in charge_hours)
              else (0, 0) for t in range(T)] \
        + [(0, d["max_d"]) if (discharge_hours is None or t % 24 in discharge_hours)
           else (0, 0) for t in range(T)] \
        + [(0, d["max_import"])] * T + [(0, d["sell_lim"])] * T \
        + [(0, np.inf)] * T + [(d["min_soc"], d["cap_mwh"])] * T \
        + [(0, 1)] * T
    lc = LinearConstraint(Aeq.tocsr(), beq, beq)
    uc = LinearConstraint(Aub.tocsr(), -np.inf, bub)
    integrality = np.zeros(nv, dtype=int)
    integrality[6 * T:] = 1
    t0 = time.time()
    res = milp(c, integrality=integrality, bounds=Bounds(
        np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])),
        constraints=[lc, uc], options={"time_limit": 600, "mip_rel_gap": 0.01})
    dt = time.time() - t0
    if not res.success and res.status == 1:  # 时间限制内未证明最优——近似解可用
        pass
    elif not res.success:
        return {"status": res.status, "cost_wan": None, "rows": [],
                "solve_s": round(dt, 2), "message": res.message}
    x = res.x
    rows = [{"Hour": t, "G": float(x[idx(2, t)]), "Pc": float(x[idx(0, t)]),
             "Pd": float(x[idx(1, t)]), "S": float(x[idx(3, t)]),
             "Q": float(x[idx(4, t)]), "SOC": float(x[idx(5, t)]),
             "z": int(x[idx(6, t)])} for t in range(T)]
    G = x[[idx(2, t) for t in range(T)]]
    S = x[[idx(3, t) for t in range(T)]]
    net = G - S
    pc = x[[idx(0, t) for t in range(T)]]
    pd = x[[idx(1, t) for t in range(T)]]
    both = (pc > 0.01) & (pd > 0.01)
    return {"status": 0, "cost_wan": float((d["price"] * G - d["sellp"] * S)
                                           .sum() / 1e4),
            "carbon_t": float((d["carbon"] * G).sum()),
            "nu": float(1 - x[[idx(4, t) for t in range(T)]].sum()
                        / max(d["W"][:T].sum(), 1e-9)),
            "peak_net_MW": float(net.max()),
            "vol_std_MW": float(net.std()),
            "max_ramp_MW": float(np.abs(np.diff(net)).max()),
            "simultaneous_charge_discharge_h": int(both.sum()),
            "rows": rows, "solve_s": round(dt, 2), "message": ""}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl = _setup()
    import joblib
    results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(_run_region)(r, s33, consume, tpl) for r in REGIONS)
    report = {k: {} for k in ("M0_free", "M0x_mutex", "M1_timed")}
    decomposition = {}
    for r, (m0, m0x, m1) in results:
        for name, m in (("M0_free", m0), ("M0x_mutex", m0x),
                        ("M1_timed", m1)):
            report[name][r] = {"cost_wan": round(m["cost_wan"], 2)
                               if m["cost_wan"] is not None else None,
                               "ramp": round(m["max_ramp_MW"], 1)
                               if m["cost_wan"] is not None else None,
                               "nu_pct": round(m["nu"] * 100, 2)
                               if m["cost_wan"] is not None else None,
                               "simul_h": m.get("simultaneous_charge_discharge_h"),
                               "solve_s": m.get("solve_s")}
        decomposition[r] = {
            "conveyor_illusion_wan": round(m0["cost_wan"] - m0x["cost_wan"], 2),
            "timing_freedom_value_wan": round(m0x["cost_wan"] - m1["cost_wan"], 2),
            "structure": m0x.get("structure")}
        print(f"[{r}] M0={m0['cost_wan']:.0f} M0x={m0x['cost_wan']:.0f} "
              f"M1={m1['cost_wan']:.0f} | "
              f"传送带假象={decomposition[r]['conveyor_illusion_wan']:.0f}万 "
              f"时段自由红利={decomposition[r]['timing_freedom_value_wan']:.0f}万",
              flush=True)
    report["decomposition"] = decomposition
    report["caliber"] = ("M0x=充放互斥 MILP（scipy.optimize.milp/HiGHS，z_t 二进制，"
                         "time_limit=600s/区，mip_rel_gap=0.01——近似最优需声明）；"
                         "M0=无互斥 LP（传送带假象）；M1=时段状态机（生成器同构）；"
                         "M1 时段分离天然互斥（M1x=M1 数学成立，无需 MILP）；"
                         "传送带假象=M0−M0x；时段自由度红利=M0x−M1")
    with open(OUT_Q3 / "q3_mutex.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["decomposition"], ensure_ascii=False, indent=1))


def _run_region(r: str, s33, consume, tpl) -> tuple:
    """单区三解：M0（LP 松弛）/ M0x（互斥 MILP）/ M1（时段约束）。"""
    d = s33._load_region_data(r)
    d["c_h"] = np.asarray(consume[r], dtype=float)
    ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
    m0 = s33.solve_region_timed(d)
    pc = np.array([x["Pc"] for x in m0["rows"]])
    pd_ = np.array([x["Pd"] for x in m0["rows"]])
    m0["simultaneous_charge_discharge_h"] = int(
        ((pc > 0.01) & (pd_ > 0.01)).sum())
    m0x = solve_mutex(d)
    m1 = s33.solve_region_timed(d, charge_hours=ch, discharge_hours=dh)
    if m0x.get("rows"):
        pc = np.array([x["Pc"] for x in m0x["rows"]])
        pd_ = np.array([x["Pd"] for x in m0x["rows"]])
        h = np.arange(len(pc)) % 24
        ch_h = sorted(int(hh) for hh in range(24) if pc[h == hh].max() > 0.01)
        dh_h = sorted(int(hh) for hh in range(24) if pd_[h == hh].max() > 0.01)
        m0x["structure"] = {"charge_hours": ch_h, "discharge_hours": dh_h}
    return r, (m0, m0x, m1)


if __name__ == "__main__":
    main()
