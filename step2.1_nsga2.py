"""step2.1_nsga2 — Q2 进化层：自研 NSGA-II（策略参数进化，numpy 无依赖）.

设计（决策记录 spec_R0-R3 后批准）:
  染色体 = 5 维策略向量（连续，v2 阈值版——v1 的 λ 为死参数导致前沿塌缩）:
    p0 mig_gpu_min  迁移最小 GPU 规格阈值   [0, 127]
    p1 mig_dur_min  迁移最小时长阈值(h)     [0, 6.65]
    p2 shift_BI     BI 最大错峰小时         [0, 24]
    p3 shift_AT     AT 最大错峰小时         [0, 24]
    p4 headroom     容量松弛比例            [0, 0.08]
  迁移量由阈值控制 → 打开"成本/碳 vs 时延"真实帕累托面（迁多省钱、迁少低时延）
  目标 = 4 维最小化: [Cost₂, CE₂, Lat, 1−NU]
  评估 = 策略 → schedule_constructive（全任务贪心）→ evaluate_4obj（模板口径）
  NSGA-II = Deb 2002: 快速非支配排序 + 拥挤度 + SBX(η=15)/PM(η=20)
            + 二元锦标赛 + 精英保留（父代∪子代 → 非支配 → 填满）
  约束 = 超容罚函数（viol_h × 1e6 加到成本/碳）
  可复现 = 3 seed × 40 pop × 60 gen，mean±std 报告；构造解(0,0,0,0,0)精英注入
  并行 = 12 worker 评估池（3 seed 串行 × 代内并行；纯 CPU——贪心调度是串行
         状态依赖过程，无法 GPU 化，见 sum_4 决策记录）

产物（output/q2/）: nsga2_front.csv（合并前沿+seed 标注）+ nsga2_result.json +
  figures/step2/fig_q2_pareto.png（成本×碳 + 成本×时延 二维投影 + 折中解标记）
"""
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import FIGURES, OUTPUT

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q2 = OUTPUT / "q2"
FIG_S2 = FIGURES / "step2"

N_POP = 40
N_GEN = 60
N_SEED = 3
N_WORKERS = 12        # 机器 16 核，留 4 核给其他任务（用户指定）
ETA_C, ETA_M = 15.0, 20.0
LB = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
UB = np.array([15.0, 6.65, 24.0, 24.0, 0.08])   # mig_gpu≤15: 可行区（viol=0 扫描实证）
CONSTRUCT_SEED = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # 全迁移构造解
PENALTY = 1e6

_CTX = {}


def _load_ctx():
    """worker 进程内一次性加载上下文（schedule_constructive/evaluate 依赖）。"""
    global _CTX
    if _CTX:
        return _CTX
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    spec2 = importlib.util.spec_from_file_location(
        "s20", root / "step2.0_construct.py")
    s20 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s20)
    wt = pd.read_csv(root / "output" / "clean" / "workload_clean.csv")
    rt = pd.read_csv(root / "output" / "clean" / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    _CTX.update(s10=s10, s20=s20, wt=wt, rt=rt, params=params,
                consume=consume)
    return _CTX


def evaluate_policy(policy: np.ndarray) -> tuple[np.ndarray, dict]:
    """策略 → 四目标（最小化）+ 明细。超容解 = 绝对不可行（全目标 1e12）。

    v2 修正：罚函数（viol×1e6 只加 cost/carbon）会让 viol 解在 lat 维度保持
    优势而留在前沿——改为不可行标记（任何可行解支配之，Deb 约束处理）。
    """
    c = _load_ctx()
    sched = c["s20"].schedule_constructive(
        c["wt"], c["rt"], c["s10"], c["params"], tuple(policy))
    m4 = c["s20"].evaluate_4obj(c["wt"], c["rt"], sched, c["s10"],
                                c["params"], c["consume"])
    if m4["viol_h"] > 0:
        return np.full(4, 1e12), m4
    cost = m4["cost_wan"]
    carbon = m4["carbon_t"]
    lat = m4["latency_ms"]
    nu = 1.0 - m4["nu_pct"] / 100.0
    return np.array([cost, carbon, lat, nu]), m4


def eval_batch(pool, policies: list[np.ndarray]) -> tuple[np.ndarray, list]:
    """批量评估（评估池并行）。"""
    res = pool.map(evaluate_policy, policies)
    F = np.array([r[0] for r in res])
    det = [r[1] for r in res]
    return F, det


# ---------- NSGA-II 核心 ----------

def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    """Deb 2002 快速非支配排序。F: (N, M) 最小化目标。"""
    n = len(F)
    dom_count = np.zeros(n, dtype=int)
    dom_list = [[] for _ in range(n)]
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                dom_list[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dom_list[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F: np.ndarray, front: list[int]) -> dict:
    """拥挤度（边界个体无穷大）。"""
    dist = {i: 0.0 for i in front}
    nf = len(front)
    if nf <= 2:
        return {i: float("inf") for i in front}
    for m in range(F.shape[1]):
        order = sorted(front, key=lambda i: F[i, m])
        dist[order[0]] = dist[order[-1]] = float("inf")
        fmin, fmax = F[order[0], m], F[order[-1], m]
        rng = fmax - fmin if fmax > fmin else 1.0
        for k in range(1, nf - 1):
            dist[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / rng
    return dist


def sbx(p1: np.ndarray, p2: np.ndarray, eta: float = ETA_C) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（实数编码）。"""
    c1, c2 = p1.copy(), p2.copy()
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5,
                    np.power(2 * u, 1 / (eta + 1)),
                    np.power(1 / (2 * (1 - u)), 1 / (eta + 1)))
    for i in range(len(p1)):
        if np.random.rand() < 0.5:
            c1[i] = 0.5 * ((1 + beta[i]) * p1[i] + (1 - beta[i]) * p2[i])
            c2[i] = 0.5 * ((1 - beta[i]) * p1[i] + (1 + beta[i]) * p2[i])
    return np.clip(c1, LB, UB), np.clip(c2, LB, UB)


def polynomial_mut(x: np.ndarray, eta: float = ETA_M) -> np.ndarray:
    """多项式变异。"""
    out = x.copy()
    for i in range(len(x)):
        if np.random.rand() < 1.0 / len(x):
            u = np.random.rand()
            delta = (np.power(2 * u, 1 / (eta + 1)) - 1 if u < 0.5
                     else 1 - np.power(2 * (1 - u), 1 / (eta + 1)))
            out[i] = x[i] + delta * (UB[i] - LB[i])
    return np.clip(out, LB, UB)


def run_seed(seed: int, pool) -> dict:
    """单 seed 完整 NSGA-II 进化（评估走共享池），返回前沿 + 明细。"""
    rng = np.random.RandomState(seed)
    pop = rng.rand(N_POP, len(LB)) * (UB - LB) + LB
    pop[0] = CONSTRUCT_SEED                      # 精英注入（全迁移构造解）
    pop[1] = np.array([8.0, 0.0, 12.0, 12.0, 0.0])   # 部分迁移+错峰变体
    F, details = eval_batch(pool, list(pop))
    gen_history = []
    for _g in range(N_GEN):
        fronts = fast_non_dominated_sort(F)
        dist = {}
        for fr in fronts:
            dist.update(crowding_distance(F, fr))
        rank_of = {i: k for k, fr in enumerate(fronts) for i in fr}

        def better(i: int, j: int) -> int:
            """二元锦标赛：支配/拥挤度比较算子。"""
            if rank_of[i] != rank_of[j]:
                return i if rank_of[i] < rank_of[j] else j
            return i if dist[i] > dist[j] else j

        pool_sel = [better(rng.randint(N_POP), rng.randint(N_POP))
                    for _ in range(N_POP)]
        children = []
        for k in range(0, N_POP, 2):
            c1, c2 = sbx(pop[pool_sel[k]], pop[pool_sel[k + 1]])
            children += [polynomial_mut(c1), polynomial_mut(c2)]
        child_F, child_det = eval_batch(pool, children)
        combo = np.vstack([F, child_F])
        combo_pop = np.vstack([pop, children])
        combo_det = details + child_det
        fronts = fast_non_dominated_sort(combo)
        new_pop, new_F, new_det = [], [], []
        for fr in fronts:
            if len(new_pop) + len(fr) <= N_POP:
                new_pop += [combo_pop[i] for i in fr]
                new_F += [combo[i] for i in fr]
                new_det += [combo_det[i] for i in fr]
            else:
                d = crowding_distance(combo, fr)
                order = sorted(fr, key=lambda i: -d[i])
                take = N_POP - len(new_pop)
                new_pop += [combo_pop[i] for i in order[:take]]
                new_F += [combo[i] for i in order[:take]]
                new_det += [combo_det[i] for i in order[:take]]
                break
        pop, F, details = np.array(new_pop), np.array(new_F), new_det
        front0 = fast_non_dominated_sort(F)[0]
        gen_history.append({"gen": _g + 1,
                            "front0_size": len(front0),
                            "front0": F[front0].tolist()})
    fronts = fast_non_dominated_sort(F)
    idx = fronts[0]
    return {"seed": seed, "front_idx": idx,
            "front": F[idx].tolist(),
            "policies": pop[idx].tolist(),
            "details": [details[i] for i in idx],
            "history": gen_history}


def topsi_compromise(front: np.ndarray) -> dict:
    """熵权 TOPSIS 折中解（四目标等权熵权）。"""
    X = front.copy()
    for m in range(X.shape[1]):
        rng = X[:, m].max() - X[:, m].min()
        X[:, m] = (X[:, m] - X[:, m].min()) / (rng if rng > 0 else 1.0)
    p = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))
    w = (1 - e) / (1 - e).sum()
    ideal = X.min(axis=0)
    nadir = X.max(axis=0)
    sd = np.sqrt(((X - ideal) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((X - nadir) ** 2 * w).sum(axis=1))
    score = nd / (sd + nd)
    k = int(np.argmax(score))
    return {"idx": k, "weights": w.tolist(), "score": float(score[k])}


def main() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)
    with mp.Pool(N_WORKERS, initializer=_load_ctx) as pool:
        results = [run_seed(seed, pool) for seed in range(N_SEED)]

    all_front = []
    for r in results:
        for i, f in enumerate(r["front"]):
            all_front.append({"seed": r["seed"], "cost_wan": f[0],
                              "carbon_t": f[1], "latency_ms": f[2],
                              "nu_pct": 100 * (1 - f[3]),
                              "policy": json.dumps(r["policies"][i])})
    front_df = pd.DataFrame(all_front)
    front_df.to_csv(OUT_Q2 / "nsga2_front.csv", index=False)

    merged = np.array([[r["cost_wan"], r["carbon_t"], r["latency_ms"],
                        1 - r["nu_pct"] / 100]
                       for r in all_front])
    dom = fast_non_dominated_sort(merged)[0]
    mfront = merged[dom]
    comp = topsi_compromise(mfront)

    n = len(mfront)
    report = {
        "n_pop": N_POP, "n_gen": N_GEN, "n_seed": N_SEED,
        "n_front_total": len(merged), "n_front_merged": n,
        "compromise_idx": comp["idx"],
        "compromise": {"cost_wan": float(mfront[comp["idx"], 0]),
                       "carbon_t": float(mfront[comp["idx"], 1]),
                       "latency_ms": float(mfront[comp["idx"], 2]),
                       "nu_pct": float(100 * (1 - mfront[comp["idx"], 3]))},
        "topsis_weights": comp["weights"],
        "front_range": {"cost_wan": [float(mfront[:, 0].min()),
                                     float(mfront[:, 0].max())],
                        "carbon_t": [float(mfront[:, 1].min()),
                                     float(mfront[:, 1].max())],
                        "latency_ms": [float(mfront[:, 2].min()),
                                       float(mfront[:, 2].max())],
                        "nu_pct": [float(100 * (1 - mfront[:, 3].max())),
                                   float(100 * (1 - mfront[:, 3].min()))]},
        "per_seed": [{"seed": r["seed"], "n_front": len(r["front_idx"])}
                     for r in results],
        "caliber": "策略参数进化（λ/错峰/松弛）；模板消纳口径；超容罚函数 1e6；"
                   "SBX η=15/PM η=20；构造解精英注入",
    }
    with open(OUT_Q2 / "nsga2_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(mfront[:, 0], mfront[:, 1], s=18, color="#2980b9",
                    label="帕累托前沿")
    axes[0].scatter(*mfront[comp["idx"], :2], s=80, marker="*",
                    color="#c0392b", label="TOPSIS 折中")
    axes[0].set_xlabel("运行成本(万元)"); axes[0].set_ylabel("碳排放(t)")
    axes[0].legend()
    axes[1].scatter(mfront[:, 0], mfront[:, 2], s=18, color="#27ae60")
    axes[1].scatter(mfront[comp["idx"], 0], mfront[comp["idx"], 2],
                    s=80, marker="*", color="#c0392b")
    axes[1].set_xlabel("运行成本(万元)"); axes[1].set_ylabel("时延(ms)")
    fig.suptitle(f"Q2 NSGA-II 帕累托前沿（3 seed 合并 {n} 解）")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_pareto.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
