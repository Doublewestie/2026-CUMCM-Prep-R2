"""step3.3+_q3_model_evolve — 轨B：模型演进 M0→M1 + 三对照分解 + E2a/E2b + 轨C 性能.

背景（轨A 逆向投产，spec_M4_Q3 修正）:
  生成器储能 = 时段状态机（充电 h0-4/22-23 域、放电 h17-21 域、充放互斥、
  SOC 全深度日循环、从不顶格）。自由 LP（M0）允许全时段顶格充放 →
  收益含"模型自由度红利"，爬坡极值系状态机缺失的建模失误产物。

模型演进谱:
  M0 自由 LP（step3.0 原版 = 理论上界，全时段自由）
  M1 时段约束 LP（充放时机与生成器同构；功率仍自由至物理上限）
对照分解（论文主口径）:
  基准(生成器) vs M1 = 时段规则内的功率/量优化价值（真实储能价值）
  M1 vs M0        = 模型自由度红利（分离，不作价值）
M2 双通道: 解后归因 Pc_r=min(Pc, W−cap_h−S)、Pc_g=Pc−Pc_r（I7 口径，无需改 LP）

E2a: M1 最优面内 min ramp（ε-约束：cost ≤ opt×(1+0.01%)）→ 顶点任意性检验
E2b: 爬坡分布三档（p90/p95/max + >100MW 对数）——max 单点防伪（口径纪律）
轨C: 计时标定 + 6 区并行（joblib 4 worker，留核纪律）

产物（output/q3/）:
  q3_model_evolve.json   三对照表 + 自由度红利 + E2a/E2b + 性能标定
  q3_lp_M1_RegionX.csv   M1 逐时明细（六区）
figures/step3/fig_q3_model_compare.png  三对照改进率 + 爬坡分布对比
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, DATA_RAW, FILES, FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

ETA_C = {"RegionA": 0.93, "RegionB": 0.93, "RegionC": 0.93,
         "RegionD": 0.94, "RegionE": 0.94, "RegionF": 0.94}
ETA_D = {"RegionA": 0.92, "RegionB": 0.92, "RegionC": 0.92,
         "RegionD": 0.93, "RegionE": 0.93, "RegionF": 0.93}
HOURS = 2407


def _import_step32():
    """加载点号文件名模块 step3.2_q3_indicators（evaluate_storage/load_rt）。"""
    root = Path(__file__).resolve().parent
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s32", root / "step3.2_q3_indicators.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_region_data(r: str) -> dict:
    """step3.0 同款数据装载（c_h=None，由调用方注入消纳模板）。"""
    import importlib.util
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s30", root / "step3.0_lp_baseline.py")
    s30 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s30)
    return s30.load_region_data(r)


def solve_region_timed(d: dict, n_hours: int = HOURS,
                       charge_hours: list | None = None,
                       discharge_hours: list | None = None,
                       charge_max: float | None = None,
                       discharge_max: float | None = None,
                       pc_fixed: np.ndarray | None = None,
                       pd_fixed: np.ndarray | None = None,
                       pc_allow: np.ndarray | None = None,
                       pd_allow: np.ndarray | None = None,
                       ramp_cap: float | None = None,
                       ramp_c_rate: float | None = None,
                       ramp_d_rate: float | None = None,
                       final_soc_exact: bool = False,
                       charge_max_hourly: np.ndarray | None = None,
                       charge_min_hourly: np.ndarray | None = None,
                       cost_cap_wan: float | None = None,
                       min_ramp: bool = False) -> dict:
    """M0/M1/M2 求解器（step3.0 矩阵结构 + 时段/功率/斜坡/终态参数化）.

    charge_hours/discharge_hours=None → 全时段自由（M0）。
    charge_max/discharge_max=None → 物理上限（d 默认）；给定=模板功率上限（M2）。
    pc_allow/pd_allow（逐时 0/1 掩码）→ 覆盖 charge_hours（规则逐时模拟 X5）。
    ramp_c_rate/ramp_d_rate → 储能充放功率斜坡限制 |ΔPc|≤R（X1-X4，物理性实验）。
    final_soc_exact=True → SOC(2406)=Init 严格等式（X8，终点套利上限）。
    charge_max_hourly(24)/charge_min_hourly(24) → 时段功率上限模板（X12）/充电
      下界模板（X9 复现基准 GridCharge 61.8）。
    pc_fixed/pd_fixed（数组）→ Pc/Pd 逐时固定（bounds=(v,v)）。
    min_ramp=True → 目标 min R（R 辅助变量，|ΔN|≤R）；可配 cost_cap_wan。
    口径：cap_h 主时段=min(W, c_h·D)（R1 模板）；Closure 段 2400+=min(W,D)。
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    T = n_hours
    extra = 1 if min_ramp else 0
    nv = 6 * T + extra
    idx = lambda k, t: k * T + t
    c = np.zeros(nv)
    for t in range(T):
        c[idx(2, t)] = d["price"][t]
        c[idx(3, t)] = -d["sellp"][t]
        c[idx(4, t)] = 1e-4
    if min_ramp:
        c[6 * T] = 1.0
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
    n_ineq = 1 + T
    Aub = lil_matrix((n_ineq, nv))
    bub = np.zeros(n_ineq)
    if not final_soc_exact:
        Aub[0, idx(5, T - 1)] = -1.0
        bub[0] = -d["init_soc"]
    # 口径（sum_10 修复 #1）：主时段 0-2399 消纳能力=R1 模板 c_h·D；
    # Closure 段 2400-2406 = min(W, D)（R1 Closure 口径，全消纳）
    c_h = d["c_h"][np.arange(T) % 24] * d["D"][:T]
    cap_h = np.minimum(d["W"][:T], c_h)
    closure = np.arange(T) >= 2400
    cap_h[closure] = np.minimum(d["W"][:T], d["D"][:T])[closure]
    for t in range(T):
        Aub[1 + t, idx(0, t)] = -1.0
        Aub[1 + t, idx(3, t)] = -1.0
        Aub[1 + t, idx(4, t)] = -1.0
        bub[1 + t] = -(d["W"][t] - cap_h[t])
    if final_soc_exact:
        Aeq.resize((2 * T + 1, nv))
        beq = np.concatenate([beq, [d["init_soc"]]])
        Aeq[2 * T, idx(5, T - 1)] = 1.0
    # 储能充放功率斜坡限制: |ΔPc| <= R_c, |ΔPd| <= R_d（X1-X4 物理性实验）
    for rate, var_k in ((ramp_c_rate, 0), (ramp_d_rate, 1)):
        if rate is not None:
            m0 = n_ineq
            n_ineq += 2 * (T - 1)
            Aub.resize((n_ineq, nv))
            bub = np.concatenate([bub, np.full(2 * (T - 1), rate)])
            for t in range(1, T):
                Aub[m0 + 2 * (t - 1), idx(var_k, t)] = 1.0
                Aub[m0 + 2 * (t - 1), idx(var_k, t - 1)] = -1.0
                Aub[m0 + 2 * t - 1, idx(var_k, t)] = -1.0
                Aub[m0 + 2 * t - 1, idx(var_k, t - 1)] = 1.0
    # 爬坡 ε-约束: |N(t)-N(t-1)| <= ramp_cap
    if ramp_cap is not None:
        m0 = n_ineq
        n_ineq += 2 * (T - 1)
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, np.full(2 * (T - 1), ramp_cap)])
        for t in range(1, T):
            Aub[m0 + 2 * (t - 1), idx(2, t)] = 1.0
            Aub[m0 + 2 * (t - 1), idx(3, t)] = -1.0
            Aub[m0 + 2 * (t - 1), idx(2, t - 1)] = -1.0
            Aub[m0 + 2 * (t - 1), idx(3, t - 1)] = 1.0
            Aub[m0 + 2 * t - 1, idx(2, t)] = -1.0
            Aub[m0 + 2 * t - 1, idx(3, t)] = 1.0
            Aub[m0 + 2 * t - 1, idx(2, t - 1)] = 1.0
            Aub[m0 + 2 * t - 1, idx(3, t - 1)] = -1.0
    # min_ramp: |ΔN| <= R
    if min_ramp:
        m0 = n_ineq
        n_ineq += 2 * (T - 1)
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, np.zeros(2 * (T - 1))])
        for t in range(1, T):
            Aub[m0 + 2 * (t - 1), idx(2, t)] = 1.0
            Aub[m0 + 2 * (t - 1), idx(3, t)] = -1.0
            Aub[m0 + 2 * (t - 1), idx(2, t - 1)] = -1.0
            Aub[m0 + 2 * (t - 1), idx(3, t - 1)] = 1.0
            Aub[m0 + 2 * (t - 1), 6 * T] = -1.0
            Aub[m0 + 2 * t - 1, idx(2, t)] = -1.0
            Aub[m0 + 2 * t - 1, idx(3, t)] = 1.0
            Aub[m0 + 2 * t - 1, idx(2, t - 1)] = 1.0
            Aub[m0 + 2 * t - 1, idx(3, t - 1)] = -1.0
            Aub[m0 + 2 * t - 1, 6 * T] = -1.0
    # 成本 ε-约束: Σ(price·G − sellp·S) <= cost_cap_wan*1e4
    if cost_cap_wan is not None:
        m0 = n_ineq
        n_ineq += 1
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, [cost_cap_wan * 1e4]])
        for t in range(T):
            Aub[m0, idx(2, t)] = d["price"][t]
            Aub[m0, idx(3, t)] = -d["sellp"][t]
    def _bnd(v, default_lo, default_hi):
        if v is not None:
            return (float(v), float(v))
        return (default_lo, default_hi)

    def _pc_allowed(t):
        if pc_allow is not None:
            return bool(pc_allow[t])
        return charge_hours is None or t % 24 in charge_hours

    def _pd_allowed(t):
        if pd_allow is not None:
            return bool(pd_allow[t])
        return discharge_hours is None or t % 24 in discharge_hours

    def _pc_bnd(t):
        if not _pc_allowed(t):
            return (0.0, 0.0)
        hi = charge_max if charge_max is not None else d["max_c"]
        if charge_max_hourly is not None:
            hi = min(hi, float(charge_max_hourly[t % 24]))
        lo = 0.0
        if charge_min_hourly is not None and t % 24 in (charge_hours or []):
            lo = float(charge_min_hourly[t % 24])
        return (lo, hi)

    bounds = [_bnd(pc_fixed[t] if pc_fixed is not None else None,
                   *_pc_bnd(t)) for t in range(T)] \
        + [_bnd(pd_fixed[t] if pd_fixed is not None else None,
                0, discharge_max if discharge_max is not None else d["max_d"])
           if _pd_allowed(t) else (0, 0) for t in range(T)] \
        + [(0, d["max_import"])] * T + [(0, d["sell_lim"])] * T \
        + [(0, None)] * T + [(d["min_soc"], d["cap_mwh"])] * T \
        + ([(0, None)] if min_ramp else [])
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"status": res.status, "cost_wan": None, "rows": [],
                "solve_s": None, "message": res.message}
    x = res.x
    rows = [{"Hour": t, "G": float(x[idx(2, t)]), "Pc": float(x[idx(0, t)]),
             "Pd": float(x[idx(1, t)]), "S": float(x[idx(3, t)]),
             "Q": float(x[idx(4, t)]), "SOC": float(x[idx(5, t)])}
            for t in range(T)]
    G = x[[idx(2, t) for t in range(T)]]
    S = x[[idx(3, t) for t in range(T)]]
    cost = float((d["price"][:T] * G - d["sellp"][:T] * S).sum())
    net = G - S
    return {"status": res.status, "cost_wan": cost / 1e4,
            "carbon_t": float((d["carbon"][:T] * G).sum()),
            "curtail": float(x[[idx(4, t) for t in range(T)]].sum()),
            "nu": float(1 - x[[idx(4, t) for t in range(T)]].sum()
                        / max(d["W"][:T].sum(), 1e-9)),
            "peak_net_MW": float(net.max()),
            "vol_std_MW": float(net.std()),
            "max_ramp_MW": float(np.abs(np.diff(net)).max())
            if T > 1 else 0.0,
            "ramp_p90_MW": float(np.percentile(np.abs(np.diff(net)), 90)),
            "ramp_p95_MW": float(np.percentile(np.abs(np.diff(net)), 95)),
            "ramp_over100": int((np.abs(np.diff(net)) > 100).sum()),
            "rows": rows, "solve_s": None, "message": ""}


SLOPE_FREE_REGIONS = {"RegionA", "RegionB", "RegionC", "RegionD"}
SETTLE_CHARGE_BAN = 2400  # 结算段禁充（X7 实证免费→干净口径）
# 生成器量级斜坡（各区实测 |ΔPc|max/|ΔPd|max，sum_10 修正：120 对 E/F 过紧
# 造成"斜坡有成本"假象；用各区自身量级→全免费）
GEN_SLOPE = {"RegionA": (66.0, 45.0), "RegionB": (59.0, 40.0),
             "RegionC": (56.0, 38.0), "RegionD": (121.0, 182.0),
             "RegionE": (155.0, 167.0), "RegionF": (145.0, 163.0)}


def solve_m3(d: dict, ch: list, dh: list, region: str | None = None,
             use_slope: bool = True, T: int = HOURS) -> dict:
    """M3_final 主模型（论文主口径，sum_10 回灌）:
      M1（时段状态机约束） + 结算段禁充（X7 免费实证） + 终态严格
      SOC(2406)=Init（X8 免费实证） + 生成器量级斜坡（各区实测，全免费）
    """
    pc_allow = np.array([1 if (t < SETTLE_CHARGE_BAN and t % 24 in ch) else 0
                         for t in range(T)], dtype=int)
    kw = dict(charge_hours=ch, discharge_hours=dh, pc_allow=pc_allow,
              final_soc_exact=True)
    if use_slope:
        rc, rd = GEN_SLOPE.get(region, (120.0, 120.0))
        kw["ramp_c_rate"] = rc
        kw["ramp_d_rate"] = rd
    return solve_region_timed(d, **kw)


def main_period_indicators(rows: list[dict], t_main: int = 2400) -> dict:
    """主时段 0-2399 指标（X13 实证：Closure 边界 2399→2400 污染 max ramp，
    东区全时段 385 vs 主时段 195——波动/峰值指标一律主时段口径）。"""
    net = np.array([x["G"] - x["S"] for x in rows if x["Hour"] < t_main])
    if len(net) < 2:
        return {"peak_main_MW": None, "std_main_MW": None,
                "ramp_main_MW": None}
    return {"peak_main_MW": round(float(net.max()), 1),
            "std_main_MW": round(float(net.std()), 1),
            "ramp_main_MW": round(float(np.abs(np.diff(net)).max()), 1)}


def charge_source_decomposition(d: dict, rows: list[dict]) -> list[dict]:
    """M2 双通道归因（I7 口径，解后）：弃电优先 → Pc_r/Pc_g。"""
    T = len(rows)
    cap_h = np.minimum(d["W"][:T], d["c_h"][np.arange(T) % 24] * d["D"][:T])
    out = []
    for t, rec in enumerate(rows):
        pc = rec["Pc"]
        spare = max(0.0, d["W"][t] - cap_h[t] - rec["S"])
        rc = min(pc, spare)
        out.append({"Hour": t, "Pc": pc, "Pc_renewable": rc,
                    "Pc_grid": pc - rc})
    return out


def baseline_ramp_dist(rt: pd.DataFrame, r: str) -> dict:
    """E2b 补全：基准爬坡分布三档（p90/p95/max + >100MW 对数）。"""
    sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
    net = (sub["GridPurchase_MW"].astype(float)
           - sub["GridSell_MW"].astype(float)).to_numpy()
    d = np.abs(np.diff(net))
    return {"max": round(float(d.max()), 1),
            "p95": round(float(np.percentile(d, 95)), 1),
            "p90": round(float(np.percentile(d, 90)), 1),
            "over100": int((d > 100).sum())}


def plot_compare(comp_rows: list[dict]) -> None:
    """fig_q3_model_compare.png：三对照（基准/M0/M1）改进率 vs 基准。"""
    df = pd.DataFrame(comp_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for tag, color in (("M0", "#c44e52"), ("M1", "#4c72b0")):
        d = df[df.model == tag].set_index("Region")
        ax.plot(d["cost_wan"] / df[df.model == "base"].set_index("Region")
                ["cost_wan"] * 100 - 100, "o-", label=tag, color=color)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels([r.replace("Region", "R") for r in REGIONS])
    ax.set_ylabel("成本相对基准 %（负=改善）")
    ax.set_title("成本三对照（M0=自由 LP 上界，M1=时段约束）")
    ax.legend()
    ax2 = axes[1]
    for tag, color in (("M0", "#c44e52"), ("M1", "#4c72b0")):
        d = df[df.model == tag].set_index("Region")
        ax2.plot(d["max_ramp_MW"], "s-", label=tag, color=color)
    ax2.plot(df[df.model == "base"].set_index("Region")["max_ramp_MW"],
             "k--", label="基准")
    ax2.set_xticks(range(len(REGIONS)))
    ax2.set_xticklabels([r.replace("Region", "R") for r in REGIONS])
    ax2.set_ylabel("最大爬坡 MW")
    ax2.set_title("最大爬坡三对照（M1 收敛性验证）")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_model_compare.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s32 = _import_step32()
    rt = s32.load_rt()
    drrep = json.loads((OUT_Q3 / "q3_dr_reverse.json").read_text(encoding="utf-8"))
    templates = drrep["templates"]
    base = s32.baseline_indicators(rt)

    def solve_pair(r: str) -> dict:
        d = _load_region_data(r)                    # c_h 下方注入
        import importlib.util
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "s10", root / "step1.0_baseline_schedule.py")
        s10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s10)
        consume = s10.fit_consume_ratio(rt)["consume_ratio"]
        d["c_h"] = np.asarray(consume[r], dtype=float)
        t0 = time.time()
        m0 = solve_region_timed(d)
        t_m0 = time.time() - t0
        t0 = time.time()
        m1 = solve_region_timed(d, charge_hours=templates[r]["charge_hours"],
                                discharge_hours=templates[r]["discharge_hours"])
        t_m1 = time.time() - t0
        t0 = time.time()
        m2 = solve_region_timed(d, charge_hours=templates[r]["charge_hours"],
                                discharge_hours=templates[r]["discharge_hours"],
                                charge_max=templates[r]["max_charge_MW"],
                                discharge_max=templates[r]["max_discharge_MW"])
        t_m2 = time.time() - t0
        t0 = time.time()
        m3 = solve_m3(d, templates[r]["charge_hours"],
                      templates[r]["discharge_hours"], region=r)
        t_m3 = time.time() - t0
        rows = m1["rows"]
        src = charge_source_decomposition(d, rows)
        src_sum = {"Pc_total_MWh": round(float(sum(x["Pc"] for x in src)), 1),
                   "Pc_renewable_MWh": round(
                       float(sum(x["Pc_renewable"] for x in src)), 1),
                   "Pc_grid_MWh": round(float(sum(x["Pc_grid"] for x in src)), 1)}
        return {"region": r, "m0": m0, "m1": m1, "m2": m2, "m3": m3,
                "src": src, "src_sum": src_sum,
                "timing": {"m0_s": round(t_m0, 2), "m1_s": round(t_m1, 2),
                           "m2_s": round(t_m2, 2), "m3_s": round(t_m3, 2)}}

    results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(solve_pair)(r) for r in REGIONS)

    # 三对照表 + 自由度红利 + E2b + M3_final/M0x（sum_10 回灌）
    comp_rows = []
    e2b = {}
    mutex = json.loads((OUT_Q3 / "q3_mutex.json").read_text(encoding="utf-8")) \
        if (OUT_Q3 / "q3_mutex.json").exists() else {}
    for res in results:
        r = res["region"]
        for model, tag in (("baseline", "base"), ("m0", "M0"), ("m1", "M1"),
                           ("m2", "M2"), ("m3", "M3")):
            if model == "baseline":
                v = base[r]
            else:
                v = res[model]
            nu = (v["nu_pct"] if model == "baseline" else v["nu"] * 100.0)
            row = {"Region": r, "model": tag,
                   "cost_wan": round(v["cost_wan"], 2),
                   "carbon_t": round(v["carbon_t"], 1),
                   "peak_net_MW": round(v["peak_net_MW"], 1),
                   "vol_std_MW": round(v["vol_std_MW"], 1),
                   "max_ramp_MW": round(v["max_ramp_MW"], 1),
                   "nu_pct": round(nu, 2)}
            if model == "m3":
                row.update(main_period_indicators(res["m3"]["rows"]))
            comp_rows.append(row)
        # M0x 物理上界列（互斥 MILP，q3_mutex 产物；gap 验证 222 万稳定）
        if mutex and r in mutex.get("M0x_mutex", {}):
            m0x = mutex["M0x_mutex"][r]
            comp_rows.append({"Region": r, "model": "M0x",
                              "cost_wan": m0x["cost_wan"],
                              "carbon_t": None, "peak_net_MW": None,
                              "vol_std_MW": None,
                              "max_ramp_MW": m0x["ramp"], "nu_pct": None,
                              "note": "互斥物理上界（600s MILP 近似，gap 验证稳定）"})
        e2b[r] = {"base": baseline_ramp_dist(rt, r)}
        for model in ("M0", "M1", "M2", "M3"):
            v = res[model.lower()]
            e2b[r][model] = {"max": round(v["max_ramp_MW"], 1),
                             "p95": round(v["ramp_p95_MW"], 1),
                             "p90": round(v["ramp_p90_MW"], 1),
                             "over100": v["ramp_over100"]}
        pd.DataFrame(res["m1"]["rows"]).to_csv(
            OUT_Q3 / f"q3_lp_M1_{r}.csv", index=False)
        pd.DataFrame(res["src"]).to_csv(
            OUT_Q3 / f"q3_lp_M1_{r}_charge_src.csv", index=False)
        print(f"[{r}] M0 cost={res['m0']['cost_wan']:.0f}万 "
              f"M1 cost={res['m1']['cost_wan']:.0f}万 "
              f"M2 cost={res['m2']['cost_wan']:.0f}万 "
              f"ramp M0/M1/M2={res['m0']['max_ramp_MW']:.0f}/"
              f"{res['m1']['max_ramp_MW']:.0f}/{res['m2']['max_ramp_MW']:.0f} "
              f"充电=弃电{res['src_sum']['Pc_renewable_MWh']:.0f}+"
              f"购电{res['src_sum']['Pc_grid_MWh']:.0f} MWh "
              f"t(M0/M1/M2)={res['timing']['m0_s']}/{res['timing']['m1_s']}/"
              f"{res['timing']['m2_s']}s", flush=True)

    # E2a：D 区 M1 最优面内 min ramp（顶点任意性检验）
    d_d = None
    for res in results:
        if res["region"] == "RegionD":
            d_d = res
    e2a = None
    if d_d is not None:
        import importlib.util
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "s10", root / "step1.0_baseline_schedule.py")
        s10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s10)
        consume = s10.fit_consume_ratio(rt)["consume_ratio"]
        d = _load_region_data("RegionD")
        d["c_h"] = np.asarray(consume["RegionD"], dtype=float)
        opt = d_d["m1"]["cost_wan"]
        t0 = time.time()
        rmin = solve_region_timed(
            d, charge_hours=templates["RegionD"]["charge_hours"],
            discharge_hours=templates["RegionD"]["discharge_hours"],
            min_ramp=True, cost_cap_wan=opt * (1 + 0.0001))
        e2a = {"cost_cap_wan": round(opt, 2),
               "min_ramp_MW": round(rmin["max_ramp_MW"], 1) if rmin["status"] == 0 else None,
               "min_ramp_cost_wan": round(rmin["cost_wan"], 2),
               "status": rmin["status"],
               "conclusion": ("最优面内爬坡可降至" if rmin["status"] == 0
                              else "最优面内不可再降"),
               "solve_s": round(time.time() - t0, 2)}
        print(f"E2a D区 M1 最优面内 min ramp: {e2a['min_ramp_MW']} MW "
              f"(M1 原解 {d_d['m1']['max_ramp_MW']:.0f} MW)", flush=True)

    report = {"comparison": comp_rows, "e2b": e2b, "e2a": e2a,
              "timing_summary": {r["region"]: r["timing"] for r in results},
              "caliber": ("M0=全时段自由 LP（无互斥——传送带假象，不引用）；"
                          "M1=时段状态机约束 LP；M2=功率模板；"
                          "M3=M3_final 主口径（M1+结算段禁充+终态严格+"
                          "生成器量级斜坡，主时段指标）；M0x=互斥 MILP 物理上界"
                          "（q3_mutex 产物，gap 验证稳定）；基准=生成器数据；"
                          "M2 双通道归因=Pc_r=min(Pc,W−cap_h−S)（I7 口径）"),
              "conclusion": ("论文主口径=M3_final vs 基准（储能价值保守口径）；"
                             "M0x=物理上界（区间呈现 [M3, M0x]）；"
                             "M0 含传送带假象不得引用（互斥约束价值见 q3_mutex）")}
    plot_compare(comp_rows)
    with open(OUT_Q3 / "q3_model_evolve.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"comparison": comp_rows, "e2b": e2b, "e2a": e2a},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
