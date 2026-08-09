"""step4.0_q4_bilevel — Q4 双层主框架：NSGA-II（上层策略）× M3_final LP（下层储能）.

结构（Q4 方案定稿 step4.0，Stackelberg 主从）:
  上层: 6 维策略 = Q2 五旋钮（mig_gpu_min/mig_dur_min/shift_BI/shift_AT/headroom）
        + α 固定 0.5（评估器参数不可优化——α=1 目标游戏化，总账 +2；PLAN §9.5）→ schedule_constructive → 逐时 AI 负荷
  下层: 每区域 M3_final LP（solve_m3：时段状态机+结算段禁充+终态严格+生成器斜坡）
        + 可选碳限额/峰值限额约束（solve_lower_constrained，场景/影子价格复用）
  评价: 六目标最小化 [Cost, Carbon, Lat, 1−QOS, 1−NU, P_peak]（step4.1 评估器）
  进化: 自研 NSGA-II（step2.1 算子复用）+ 构造解/折中解精英注入 + 多进程池
  预算: 默认正式 40pop×30gen×3seed（T0 标定 2.44s/eval → ~19min/8worker）；
        --smoke 12/4/1 供 CI/快速验证（q4_bilevel 产物含 n_pop/n_gen/n_seed 自证）
  验收（三层量化）: 约束残差 <1e-6 / 单调性 / 可复现逐位一致 / 基线组合重现
        + 收敛验收（MC 超体积末 5 代范围 <5×MC 噪声 σ）+ 种子方差（折中解 rel std <0.1%）

产物（output/q4/）:
  q4_bilevel.json          双层结果（折中解/前沿统计/收敛曲线/种子方差/基线对照/口径声明）
  q4_front.csv             联合帕累托前沿（policy + 六目标）
  figures/step4/fig_q4_pareto.png  成本×时延 + 成本×峰值 2D 投影 + 折中标记
  figures/step4/fig_q4_convergence.png  超体积收敛曲线（3 seed 中位数）
"""
import importlib.util
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import FIGURES, OUTPUT, REGIONS, HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"

# 正式预算（论文口径，T0 标定 2.44s/eval → 40×30×3≈19min/8worker）；
# --smoke 覆盖为 12/4/1 供 CI/快速验证
N_POP, N_GEN, N_SEED = 40, 30, 3
N_WORKERS = 12   # 16 核 × OMP=1（单线程 LP），留 4 核余量
# 边界 = Q2 可行区实证内（mig_gpu≤8 / headroom≤0.03 提高初始种群可行率，
# 避免 1e12 支配排挤；α 维固定 0.5——评估器参数不可被优化（α=1 使 1−QOS≡0
# 目标游戏化，错误-修复总账 +2；α 敏感性由 step4.1 alpha_scan 单独报告））
LB = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
UB = np.array([8.0, 6.65, 24.0, 24.0, 0.03, 0.5])
CONSTRUCT_SEED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])   # 构造解 + α=0.5 精英

_CTX = {}


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"),
        Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    """worker 进程一次性加载：step4.1 评估器上下文 + NSGA-II 算子。"""
    global _CTX
    if _CTX:
        return _CTX
    s41 = _load_mod("step4.1_q4_indicators.py")
    c = s41.load_ctx()
    s21 = _load_mod("step2.1_nsga2.py")
    _CTX.update(c, s41=s41, s21=s21)
    return _CTX


# ---------- 下层带约束求解器（碳限额/峰值限额，场景与影子价格共用） ----------

def solve_lower_constrained(d: dict, ch: list, dh: list,
                            carbon_cap_t: float | None = None,
                            peak_cap_MW: float | None = None,
                            sell_scale: float = 1.0,
                            T: int = HOURS_TOTAL) -> dict:
    """M1 时段约束 LP + 可选碳限额/峰值限额/卖电上限缩放（Q4 场景与影子价格共用）。

    carbon_cap_t: Σ carb·G ≤ cap（碳限额场景 τ）
    peak_cap_MW:  G−S ≤ P̄ 每时（峰值定价/场景）
    sell_scale:   SellLimit × 缩放（影子价格 SellLimit 载体；bounds 对偶
                  scipy 不返回 → 该载体只能用数值差分）
    矩阵结构与 solve_region_timed 同源（功率平衡 + SOC 递推 + 弃电下界 + 终态）。
    返回含 marg_ub/marg_eq（HiGHS 对偶边际：d(obj)/d(RHS)，≤0 方向）+
    ub_map（碳限额/峰值行号）——解析影子价格（T5 对偶升级）。
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

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
    n_ineq = 1 + T
    Aub = lil_matrix((n_ineq, nv))
    bub = np.zeros(n_ineq)
    Aub[0, idx(5, T - 1)] = -1.0
    bub[0] = -d["init_soc"]
    c_h = np.minimum(d["W"][:T], d["c_h"][np.arange(T) % 24] * d["D"][:T])
    closure = np.arange(T) >= 2400
    c_h[closure] = np.minimum(d["W"][:T], d["D"][:T])[closure]
    for t in range(T):
        Aub[1 + t, idx(0, t)] = -1.0
        Aub[1 + t, idx(3, t)] = -1.0
        Aub[1 + t, idx(4, t)] = -1.0
        bub[1 + t] = -(d["W"][t] - c_h[t])
    carbon_row = None
    peak_rows = None
    if carbon_cap_t is not None:
        carbon_row = n_ineq
        n_ineq += 1
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, [carbon_cap_t]])
        for t in range(T):
            Aub[carbon_row, idx(2, t)] = d["carbon"][t]
    if peak_cap_MW is not None:
        peak_rows = list(range(n_ineq, n_ineq + T))
        m0 = n_ineq
        n_ineq += T
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, np.full(T, peak_cap_MW)])
        for t in range(T):
            Aub[m0 + t, idx(2, t)] = 1.0
            Aub[m0 + t, idx(3, t)] = -1.0
    bounds = [(0, d["max_c"]) if t % 24 in ch else (0, 0) for t in range(T)] \
        + [(0, d["max_d"]) if t % 24 in dh else (0, 0) for t in range(T)] \
        + [(0, d["max_import"])] * T \
        + [(0, d["sell_lim"] * sell_scale)] * T \
        + [(0, None)] * T + [(d["min_soc"], d["cap_mwh"])] * T
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"status": res.status, "cost_wan": None, "rows": [],
                "message": res.message}
    x = res.x
    rows = [{"Hour": t, "G": float(x[idx(2, t)]), "Pc": float(x[idx(0, t)]),
             "Pd": float(x[idx(1, t)]), "S": float(x[idx(3, t)]),
             "Q": float(x[idx(4, t)]), "SOC": float(x[idx(5, t)])}
            for t in range(T)]
    marg_ub = np.asarray(res.ineqlin.marginals, dtype=float).tolist()
    marg_eq = np.asarray(res.eqlin.marginals, dtype=float).tolist()
    out = {"status": 0, "cost_wan": float(
        (d["price"][:T] * x[[idx(2, t) for t in range(T)]]
         - d["sellp"][:T] * x[[idx(3, t) for t in range(T)]]).sum() / 1e4),
        "carbon_t": float((d["carbon"][:T] * x[[idx(2, t) for t in range(T)]]
                           ).sum()),
        "curtail": float(x[[idx(4, t) for t in range(T)]].sum()),
        "rows": rows,
        "marg_ub": marg_ub, "marg_eq": marg_eq,
        "ub_map": {"n_ineq": int(n_ineq), "T": int(T),
                   "carbon_row": carbon_row, "peak_rows": peak_rows}}
    return out


# ---------- 双层评价（worker） ----------

def evaluate_policy(policy: np.ndarray) -> tuple[np.ndarray, dict]:
    """6 旋钮策略（α 维固定 0.5）→ 六目标（最小化）。上层 viol>0 → 全 1e12（Deb 约束处理）。"""
    c = load_ctx()
    sched = c["s20"].schedule_constructive(
        c["wt"], c["rt"], c["s10"], c["params"], tuple(policy[:5]))
    ev = c["s41"].evaluate_q4_six(c, sched, alpha=float(policy[5]))
    return ev["obj"], {"policy": policy.tolist(), "viol_h": ev["viol_h"]}


def eval_batch(pool, policies: list[np.ndarray]) -> tuple[np.ndarray, list]:
    res = pool.map(evaluate_policy, policies)
    return np.array([r[0] for r in res]), [r[1] for r in res]


# ---------- NSGA-II（step2.1 算子复用：非支配/拥挤度；SBX/变异为 6 维版） ----------

def sbx_q4(p1: np.ndarray, p2: np.ndarray, eta: float = 15.0):
    """模拟二进制交叉（6 维，LB/UB 为本模块 Q4 边界——step2.1 算子仅 5 维）。"""
    c1, c2 = p1.copy(), p2.copy()
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5, np.power(2 * u, 1 / (eta + 1)),
                    np.power(1 / (2 * (1 - u)), 1 / (eta + 1)))
    for i in range(len(p1)):
        if np.random.rand() < 0.5:
            c1[i] = 0.5 * ((1 + beta[i]) * p1[i] + (1 - beta[i]) * p2[i])
            c2[i] = 0.5 * ((1 - beta[i]) * p1[i] + (1 + beta[i]) * p2[i])
    return np.clip(c1, LB, UB), np.clip(c2, LB, UB)


def polynomial_mut_q4(x: np.ndarray, eta: float = 20.0) -> np.ndarray:
    """多项式变异（6 维版，α 维度同样受 [0,1] 边界约束）。"""
    out = x.copy()
    for i in range(len(x)):
        if np.random.rand() < 1.0 / len(x):
            u = np.random.rand()
            delta = (np.power(2 * u, 1 / (eta + 1)) - 1 if u < 0.5
                     else 1 - np.power(2 * (1 - u), 1 / (eta + 1)))
            out[i] = x[i] + delta * (UB[i] - LB[i])
    return np.clip(out, LB, UB)


def hv_mc(F: np.ndarray, lo: np.ndarray, hi: np.ndarray,
          n_samples: int = 20000, seed: int = 42) -> float:
    """MC 超体积（最小化方向）：[lo,hi] 归一化立方体中被前沿支配的体积分数。

    参考点固定为第 0 代（含精英注入）min/max → 跨代可比；退化维（span≈0）
    用 ±10% 扩展（标记 degenerate 由调用方判断）。估计量 = 支配率（无偏 MC）。
    """
    n, d = F.shape
    if n == 0:
        return 0.0
    span = hi - lo
    degen = span < 1e-12
    lo2 = lo - 0.05 * np.maximum(span, 1e-9)
    hi2 = hi + 0.05 * np.maximum(span, 1e-9)
    if degen.any():
        lo2[degen] = lo[degen] * 0.9
        hi2[degen] = lo[degen] * 1.1
        if (hi2[degen] - lo2[degen] < 1e-9).any():
            lo2[degen], hi2[degen] = lo[degen] - 1.0, lo[degen] + 1.0
    Fn = (F - lo2) / (hi2 - lo2)
    rng = np.random.RandomState(seed)
    Y = rng.uniform(0.0, 1.0, size=(n_samples, d))
    dom = np.zeros(n_samples, dtype=bool)
    for i in range(n):
        dom |= (Fn[i] <= Y).all(axis=1)
    return float(dom.mean())


def run_seed(seed: int, pool) -> dict:
    c = load_ctx()
    s21 = c["s21"]
    t0 = time.time()
    rng = np.random.RandomState(seed)
    pop = rng.rand(N_POP, 6) * (UB - LB) + LB
    pop[0] = CONSTRUCT_SEED
    F, details = eval_batch(pool, list(pop))
    feas_mask = (F < 1e10).all(axis=1)   # viol>0 → 1e12 标记（Deb 约束处理）
    ref_lo = F[feas_mask].min(axis=0)
    ref_hi = F[feas_mask].max(axis=0)
    fe0 = F[feas_mask]
    hv0 = hv_mc(fe0, ref_lo, ref_hi)
    history = [{"gen": 0, "n_front": len(s21.fast_non_dominated_sort(F)[0]),
                "hv": hv0, "min_cost": float(fe0[:, 0].min())
                if len(fe0) else None}]
    for _g in range(N_GEN):
        fronts = s21.fast_non_dominated_sort(F)
        dist = {}
        for fr in fronts:
            dist.update(s21.crowding_distance(F, fr))
        rank_of = {i: k for k, fr in enumerate(fronts) for i in fr}

        def better(i: int, j: int) -> int:
            if rank_of[i] != rank_of[j]:
                return i if rank_of[i] < rank_of[j] else j
            return i if dist[i] > dist[j] else j

        sel = [better(rng.randint(N_POP), rng.randint(N_POP))
               for _ in range(N_POP)]
        children = []
        for k in range(0, N_POP, 2):
            c1, c2 = sbx_q4(pop[sel[k]], pop[sel[k + 1]])
            children += [polynomial_mut_q4(c1), polynomial_mut_q4(c2)]
        cF, cdet = eval_batch(pool, children)
        combo = np.vstack([F, cF])
        combo_pop = np.vstack([pop, children])
        combo_det = details + cdet
        fronts = s21.fast_non_dominated_sort(combo)
        new_pop, new_F, new_det = [], [], []
        for fr in fronts:
            if len(new_pop) + len(fr) <= N_POP:
                new_pop += [combo_pop[i] for i in fr]
                new_F += [combo[i] for i in fr]
                new_det += [combo_det[i] for i in fr]
            else:
                d = s21.crowding_distance(combo, fr)
                order = sorted(fr, key=lambda i: -d[i])
                take = N_POP - len(new_pop)
                new_pop += [combo_pop[i] for i in order[:take]]
                new_F += [combo[i] for i in order[:take]]
                new_det += [combo_det[i] for i in order[:take]]
                break
        pop, F, details = np.array(new_pop), np.array(new_F), new_det
        front_idx = s21.fast_non_dominated_sort(F)[0]
        feas = F[(F < 1e10).all(axis=1)]
        hv = hv_mc(feas, ref_lo, ref_hi)
        history.append({"gen": _g + 1, "n_front": len(front_idx),
                        "hv": hv, "min_cost": float(feas[:, 0].min())
                        if len(feas) else None})
        if (_g + 1) % 10 == 0:
            print(f"  [seed {seed}] gen {_g + 1}/{N_GEN} hv={hv:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    fronts = s21.fast_non_dominated_sort(F)
    idx = fronts[0]
    return {"seed": seed, "front_idx": idx, "front": F[idx].tolist(),
            "policies": pop[idx].tolist(), "history": history}


def topsis_compromise(c, F: np.ndarray) -> int:
    """熵权 TOPSIS 折中（六目标，step2.1 同算法）。单点/退化输入守卫。"""
    if len(F) <= 1:
        return 0
    X = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    p = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))
    w = (1 - e) / (1 - e).sum()
    sd = np.sqrt(((X - X.min(axis=0)) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((X - X.max(axis=0)) ** 2 * w).sum(axis=1))
    score = nd / (sd + nd + 1e-12)
    if not np.isfinite(score).all():
        return int(np.argmin(F[:, 0]))
    return int(np.argmax(score))


def main() -> None:
    import sys
    global N_POP, N_GEN, N_SEED
    if "--smoke" in sys.argv:
        N_POP, N_GEN, N_SEED = 12, 4, 1
        print("[smoke] N_POP=12 N_GEN=4 N_SEED=1", flush=True)
    elif "--pop" in sys.argv:
        N_POP = int(sys.argv[sys.argv.index("--pop") + 1])
        N_GEN = int(sys.argv[sys.argv.index("--gen") + 1])
        N_SEED = int(sys.argv[sys.argv.index("--seed") + 1])
        print(f"[custom] N_POP={N_POP} N_GEN={N_GEN} N_SEED={N_SEED}",
              flush=True)
    else:
        print(f"[full] N_POP={N_POP} N_GEN={N_GEN} N_SEED={N_SEED}", flush=True)
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)
    # worker 线程纪律：8 worker × 单线程 LP（16 核留余量，spawn 子进程继承环境）
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    results = []
    with mp.Pool(N_WORKERS, initializer=load_ctx) as pool:
        for s in range(N_SEED):
            t0 = time.time()
            results.append(run_seed(s, pool))
            print(f"[seed {s}] done in {time.time()-t0:.1f}s", flush=True)

    all_front = []
    for r in results:
        for i, f in enumerate(r["front"]):
            all_front.append({"seed": r["seed"], "cost_wan": f[0],
                              "carbon_t": f[1], "latency_ms": f[2],
                              "one_minus_qos": f[3],
                              "one_minus_nu": f[4], "peak_net_MW": f[5],
                              "policy": json.dumps(r["policies"][i])})
    front_df = pd.DataFrame(all_front)
    front_df.to_csv(OUT_Q4 / "q4_front.csv", index=False)

    merged = np.array([[r["cost_wan"], r["carbon_t"], r["latency_ms"],
                        r["one_minus_qos"], r["one_minus_nu"],
                        r["peak_net_MW"]] for r in all_front])
    c = load_ctx()
    s21 = c["s21"]
    dom = s21.fast_non_dominated_sort(merged)[0]
    mfront = merged[dom]
    comp_i = topsis_compromise(c, mfront)

    # 收敛验收：每 seed 超体积曲线（中位数），末 5 代范围 < 5×MC 噪声 σ
    # （p≈0.03 时 2 万采样 σ≈0.0011——相对变化判据会被噪声误判为未收敛，
    #   故用绝对范围阈值；总账 +3）
    hv_series = {str(r["seed"]): [h["hv"] for h in r["history"]]
                 for r in results}
    hv_median = np.median(np.array(list(hv_series.values())), axis=0)
    last5 = hv_median[-5:]
    p_med = float(np.median(last5))
    sigma_mc = float(np.sqrt(max(p_med * (1 - p_med), 1e-12) / 20000))
    hv_range = float(np.max(last5) - np.min(last5))
    converged = hv_range < max(5 * sigma_mc, 1e-4)
    last5_rel = hv_range / max(p_med, 1e-12)

    # 种子方差验收：每 seed 独立折中解六目标 rel std（合并前沿 re-TOPSIS 前）
    seed_comp = {}
    for r in results:
        Fs = np.array(r["front"])
        if len(Fs) > 1:
            Fs = Fs[s21.fast_non_dominated_sort(Fs)[0]]
        is_ = topsis_compromise(c, Fs)
        seed_comp[str(r["seed"])] = Fs[is_].tolist()
    sc_arr = np.array(list(seed_comp.values()))
    rel_std = (sc_arr.std(axis=0) / np.abs(sc_arr.mean(axis=0) + 1e-12)).tolist()
    SIX = ["cost_wan", "carbon_t", "latency_ms", "one_minus_qos",
           "one_minus_nu", "peak_net_MW"]
    seed_variance = {SIX[i]: {"rel_std": round(float(rel_std[i]), 6),
                              "values": [round(float(v), 4) for v in sc_arr[:, i]]}
                     for i in range(6)}
    seed_ok = float(np.max(rel_std)) < 0.001

    report = {
        "n_pop": N_POP, "n_gen": N_GEN, "n_seed": N_SEED,
        "n_front_total": len(merged), "n_front_merged": len(mfront),
        "compromise_idx": int(comp_i),
        "compromise": {"cost_wan": float(mfront[comp_i, 0]),
                       "carbon_t": float(mfront[comp_i, 1]),
                       "latency_ms": float(mfront[comp_i, 2]),
                       "one_minus_qos": float(mfront[comp_i, 3]),
                       "one_minus_nu": float(mfront[comp_i, 4]),
                       "peak_net_MW": float(mfront[comp_i, 5])},
        "front_range": {"cost_wan": [float(mfront[:, 0].min()),
                                     float(mfront[:, 0].max())],
                        "peak_net_MW": [float(mfront[:, 5].min()),
                                        float(mfront[:, 5].max())]},
        "convergence": {"hv_series": hv_series,
                        "hv_median": [round(float(x), 6) for x in hv_median],
                        "last5_range_pct": round(last5_rel * 100, 4),
                        "sigma_mc": round(sigma_mc, 6),
                        "verdict": ("收敛（末 5 代 HV 范围 <5×MC 噪声 σ）"
                                    if converged else
                                    "未收敛（预算不足，需扩代）")},
        "seed_variance": {"rel_std_max": round(float(np.max(rel_std)), 6),
                          "accept": seed_ok, "per_metric": seed_variance},
        "caliber": ("双层 Stackelberg：上层 NSGA-II 6 旋钮（Q2 五旋钮+α 固定 0.5）"
                    "× 下层 M3_final LP；六目标 [Cost, Carbon, Lat, 1−QOS, 1−NU, "
                    "P_peak]；viol>0 → 1e12 绝对不可行；构造解精英注入；"
                    "收敛=MC 超体积末 5 代范围 <5×噪声 σ；种子方差=3 seed 折中解 "
                    "rel std <0.1%；验收：约束残差/单调性/可复现见 tests/test_q4.py"),
    }
    with open(OUT_Q4 / "q4_bilevel.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(mfront[:, 0], mfront[:, 2], s=18, color="#2980b9")
    axes[0].scatter(*mfront[comp_i, [0, 2]], s=80, marker="*", color="#c0392b",
                    label="TOPSIS 折中")
    axes[0].set_xlabel("运行成本(万元)")
    axes[0].set_ylabel("加权时延(ms)")
    axes[0].legend()
    axes[1].scatter(mfront[:, 0], mfront[:, 5], s=18, color="#27ae60")
    axes[1].scatter(*mfront[comp_i, [0, 5]], s=80, marker="*", color="#c0392b")
    axes[1].set_xlabel("运行成本(万元)")
    axes[1].set_ylabel("峰值净购电(MW, 主时段)")
    fig.suptitle(f"Q4 双层联合帕累托（{len(mfront)} 解，成本×时延 / 成本×峰值）")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_pareto.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for sd, hv in hv_series.items():
        ax2.plot(range(len(hv)), hv, "-", linewidth=1.2, alpha=0.6,
                 label=f"seed {sd}")
    ax2.plot(range(len(hv_median)), hv_median, "o-", color="#c0392b",
             linewidth=2, label="中位数")
    ax2.set_xlabel("进化代数")
    ax2.set_ylabel("MC 超体积（归一化支配体积分数）")
    ax2.set_title(f"Q4 收敛曲线（末 5 代范围 {last5_rel * 100:.2f}%"
                  f"{'，收敛' if converged else '，未收敛'}）")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_Q4 / "fig_q4_convergence.png", bbox_inches="tight")
    plt.close(fig2)
    print(json.dumps({k: v for k, v in report.items() if k != "caliber"},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
