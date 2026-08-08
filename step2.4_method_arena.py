"""step2.4_method_arena — Q2 五方法判别实验（B6，模型选择的数据裁决）.

对照纯度纪律（math-methods）: 同空间内对比——
  阶段 A（策略空间 5 维，40pop×60gen×3seed）: NSGA-II(现状) vs NSGA-III-lite
    （参考方向引导） vs MOEA/D（Tchebycheff 分解）——进化族内部选型
  阶段 B（任务级空间）: ALNS（破坏-修复） vs 拉格朗日对偶引导贪心
    （容量影子价格 π） vs 阶段 A 赢家——任务级 vs 代理空间的支配对比
  统一评估器（模板消纳口径）+ 统一容量约束 + 统一白名单。
  裁决: 帕累托支配关系（谁覆盖谁）+ 超体积（HV）+ 计算时间 + 对偶 gap。

产物（output/q2/）: method_arena.json + figures/step2/fig_method_arena.png
"""
import importlib.util
import json
import multiprocessing as mp
import time
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q2 = OUTPUT / "q2"
FIG_S2 = FIGURES / "step2"

N_POP, N_GEN, N_SEED, N_WORKERS = 40, 30, 3, 8
ETA_C, ETA_M = 15.0, 20.0
LB = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
UB = np.array([15.0, 6.65, 24.0, 24.0, 0.08])

_CTX = {}


def _load_ctx():
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
    _CTX.update(s10=s10, s20=s20, wt=wt, rt=rt, params=params, consume=consume)
    return _CTX


def eval_policy(p: np.ndarray) -> np.ndarray:
    """策略 → 4 目标最小化（viol>0 → 1e12 不可行标记）。"""
    c = _load_ctx()
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(p))
    m4 = c["s20"].evaluate_4obj(c["wt"], c["rt"], sched, c["s10"],
                                c["params"], c["consume"])
    if m4["viol_h"] > 0:
        return np.full(4, 1e12)
    return np.array([m4["cost_wan"], m4["carbon_t"], m4["latency_ms"],
                     1 - m4["nu_pct"] / 100.0])


def eval_sched_df(sched: pd.DataFrame) -> np.ndarray:
    """任务级调度 → 4 目标。"""
    c = _load_ctx()
    m4 = c["s20"].evaluate_4obj(c["wt"], c["rt"], sched, c["s10"],
                                c["params"], c["consume"])
    if m4["viol_h"] > 0:
        return np.full(4, 1e12)
    return np.array([m4["cost_wan"], m4["carbon_t"], m4["latency_ms"],
                     1 - m4["nu_pct"] / 100.0])


def non_dom_front(F: np.ndarray) -> np.ndarray:
    """返回前沿掩码（Deb 快速非支配第 0 层）。"""
    n = len(F)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if F[i, 0] > 1e11:
            continue
        dominated = False
        for j in range(n):
            if i != j and F[j, 0] <= 1e11 \
                    and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated = True
                break
        if not dominated:
            keep[i] = True
    return keep


def fast_nondom(F: np.ndarray) -> list[list[int]]:
    n = len(F)
    dc = np.zeros(n, dtype=int)
    dl = [[] for _ in range(n)]
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                dl[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                dc[p] += 1
        if dc[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dl[p]:
                dc[q] -= 1
                if dc[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F: np.ndarray, fr: list[int]) -> dict:
    dist = {i: 0.0 for i in fr}
    nf = len(fr)
    if nf <= 2:
        return {i: float("inf") for i in fr}
    for m in range(F.shape[1]):
        order = sorted(fr, key=lambda i: F[i, m])
        dist[order[0]] = dist[order[-1]] = float("inf")
        rng = F[order[-1], m] - F[order[0], m]
        rng = rng if rng > 0 else 1.0
        for k in range(1, nf - 1):
            dist[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / rng
    return dist


def sbx_pm(p1: np.ndarray, p2: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.rand() < 0.5:
            u = rng.rand()
            beta = (2 * u) ** (1 / (ETA_C + 1)) if u < 0.5 \
                else (1 / (2 * (1 - u))) ** (1 / (ETA_C + 1))
            c1[i] = np.clip(0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i]), LB[i], UB[i])
            c2[i] = np.clip(0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i]), LB[i], UB[i])
    for x in (c1, c2):
        for i in range(len(x)):
            if rng.rand() < 1 / len(x):
                u = rng.rand()
                d = (2 * u) ** (1 / (ETA_M + 1)) - 1 if u < 0.5 \
                    else 1 - (2 * (1 - u)) ** (1 / (ETA_M + 1))
                x[i] = np.clip(x[i] + d * (UB[i] - LB[i]), LB[i], UB[i])
    return c1, c2


def run_nsga2(seed: int, pool) -> np.ndarray:
    """NSGA-II（与 step2.1 同算法，自包含实现）。"""
    rng = np.random.RandomState(seed)
    pop = rng.rand(N_POP, 5) * (UB - LB) + LB
    pop[0] = np.zeros(5)
    F = np.array(pool.map(eval_policy, list(pop)))
    for _g in range(N_GEN):
        fronts = fast_nondom(F)
        dist = {}
        for fr in fronts:
            dist.update(crowding(F, fr))
        rank = {i: k for k, fr in enumerate(fronts) for i in fr}
        sel = []
        for _ in range(N_POP):
            a, b = rng.randint(N_POP), rng.randint(N_POP)
            while b == a:
                b = rng.randint(N_POP)
            if rank[a] != rank[b]:
                sel.append(a if rank[a] < rank[b] else b)
            else:
                sel.append(a if dist[a] > dist[b] else b)
        kids = []
        for k in range(0, N_POP, 2):
            kids += list(sbx_pm(pop[sel[k]], pop[sel[(k + 1) % N_POP]], rng))
        kF = np.array(pool.map(eval_policy, kids))
        combo = np.vstack([F, kF])
        cpop = np.vstack([pop, kids])
        fronts = fast_nondom(combo)
        np_, nF = [], []
        for fr in fronts:
            if len(np_) + len(fr) <= N_POP:
                np_ += [cpop[i] for i in fr]
                nF += [combo[i] for i in fr]
            else:
                d = crowding(combo, fr)
                order = sorted(fr, key=lambda i: -d[i])
                take = N_POP - len(np_)
                np_ += [cpop[i] for i in order[:take]]
                nF += [combo[i] for i in order[:take]]
                break
        pop, F = np.array(np_), np.array(nF)
    keep = non_dom_front(F)
    return F[keep]


# ---------- 阶段 A: 进化族（策略空间） ----------

def run_nsga3_lite(seed: int, pool) -> np.ndarray:
    """NSGA-III-lite：非支配排序 + 参考方向多样性（归一化简化为理想点距离）。"""
    rng = np.random.RandomState(seed + 100)
    ref = np.random.dirichlet(np.ones(4), N_POP)          # 参考方向（4 目标单纯形）
    pop = rng.rand(N_POP, 5) * (UB - LB) + LB
    pop[0] = np.zeros(5)
    F = np.array(pool.map(eval_policy, list(pop)))
    for _g in range(N_GEN):
        # 参考方向关联：每个个体归属最近参考方向（归一化后夹角）
        z = F.min(axis=0)
        rng_ = F - z
        nrm = np.linalg.norm(rng_, axis=1, keepdims=True) + 1e-12
        u = rng_ / nrm
        refn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        # 选择：按参考方向聚合（每方向取归一化范数最小者入池）
        pool_idx = []
        for rd in refn:
            sim = u @ rd
            k = int(np.argmax(sim))
            pool_idx.append(k)
        kids = []
        for k in range(0, N_POP, 2):
            p1, p2 = pop[pool_idx[k]], pop[pool_idx[(k + 1) % N_POP]]
            c1 = np.clip(p1 + 0.5 * (p2 - p1) * (rng.rand() - 0.5) * 2, LB, UB)
            c2 = np.clip(p2 + 0.5 * (p1 - p2) * (rng.rand() - 0.5) * 2, LB, UB)
            kids += [c1, c2]
        kF = np.array(pool.map(eval_policy, kids))
        combo = np.vstack([F, kF])
        cpop = np.vstack([pop, kids])
        keep = non_dom_front(combo)
        if keep.sum() >= N_POP:
            pop, F = cpop[keep][:N_POP], combo[keep][:N_POP]
        else:
            # 补充：从剩余解按参考方向距离选
            rest = ~keep
            dist = ((combo[rest][:, None, :] - z[None, None, :]) ** 2).sum(-1)
            pick = rest.nonzero()[0][np.argsort(dist.sum(axis=1))][:N_POP - keep.sum()]
            pop = np.vstack([cpop[keep], cpop[pick]])
            F = np.vstack([combo[keep], combo[pick]])
    return combo[keep][:N_POP] if keep.sum() >= N_POP else F[:N_POP]


def run_moead(seed: int, pool) -> np.ndarray:
    """MOEA/D：Tchebycheff 分解，邻居交配 + 邻居更新。"""
    rng = np.random.RandomState(seed + 200)
    W = rng.dirichlet(np.ones(4), N_POP)                  # 权重向量
    nbr = 10
    dist = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    nb_idx = np.argsort(dist, axis=1)[:, 1:nbr + 1]
    pop = rng.rand(N_POP, 5) * (UB - LB) + LB
    pop[0] = np.zeros(5)
    F = np.array(pool.map(eval_policy, list(pop)))
    z = F.min(axis=0)
    for _g in range(N_GEN):
        order = rng.permutation(N_POP)
        ys = [np.clip(pop[rng.choice(nb_idx[i], 2, replace=False)[0]]
                      + rng.rand(5) * (pop[rng.choice(nb_idx[i], 2, replace=False)[1]]
                                       - pop[rng.choice(nb_idx[i], 2, replace=False)[0]]),
                      LB, UB)
              for i in order]
        ysF = np.array(pool.map(eval_policy, ys))
        for k, i in enumerate(order):
            y, yF = ys[k], ysF[k]
            if yF[0] > 1e11:
                continue
            z = np.minimum(z, yF)
            for j in nb_idx[i]:
                gx = np.max(W[j] * np.abs(F[j] - z))
                gy = np.max(W[j] * np.abs(yF - z))
                if gy <= gx:
                    pop[j], F[j] = y, yF
    keep = non_dom_front(F)
    return F[keep]


# ---------- 阶段 B: 任务级方法 ----------

def sched_to_df(wt: pd.DataFrame, region_map: dict, start_map: dict) -> pd.DataFrame:
    return pd.DataFrame({"TaskID": list(region_map.keys()),
                         "Region": list(region_map.values()),
                         "StartHour": list(start_map.values())})


def run_alns(seed: int, wt, rt, s10, s20, params, n_iter: int = 12) -> np.ndarray:
    """ALNS ????occ ???? + ???????????? set_index?46k ? = ????"""
    rng = np.random.RandomState(seed + 300)
    wl = {(r.TaskType, r.SourceRegion): r.Reachable.split("|")
          for r in pd.read_csv(CLEAN / "whitelist.csv").itertuples(index=False)}
    cap = np.array([params["cap"][r] for r in REGIONS], dtype=float)
    ws = wt.set_index("TaskID")
    gpu_d = ws["GPU_Demand"].to_dict()
    dur_d = ws["dur_h"].to_dict()
    typ_d = ws["TaskType"].to_dict()
    src_d = ws["SourceRegion"].to_dict()
    last_d = ws["LatestFinishHour"].to_dict()
    arr_d = ws["ArrivalHour"].to_dict()
    cur = s20.schedule_constructive(wt, rt, s10, params, (0, 0, 0, 0, 0))
    bestF = eval_sched_df(cur)
    best = cur.set_index("TaskID")
    rmap = {tid: REGIONS.index(rec.Region) for tid, rec in best.iterrows()}
    smap = {tid: int(rec.StartHour) for tid, rec in best.iterrows()}
    occ = np.zeros((2407, 6))
    for tid, ri in rmap.items():
        st = smap[tid]
        h1 = min(int(np.ceil(st + dur_d[tid])), 2407)
        occ[st:h1, ri] += gpu_d[tid]
    all_tids = list(best.index)
    T = 1e3
    for it in range(n_iter):
        n_rm = max(100, int(len(all_tids) * 0.04))
        rm = rng.choice(all_tids, n_rm, replace=False)
        for tid in rm:
            ri = rmap[tid]
            st = smap[tid]
            h1 = min(int(np.ceil(st + dur_d[tid])), 2407)
            occ[st:h1, ri] -= gpu_d[tid]
        ok = True
        for tid in rm:
            g = gpu_d[tid]
            dur = dur_d[tid]
            if typ_d[tid] == "RealTimeInference":
                ri = REGIONS.index(src_d[tid])
                st = int(arr_d[tid])
                rmap[tid], smap[tid] = ri, st
                h1 = min(int(np.ceil(st + dur)), 2407)
                occ[st:h1, ri] += g
                continue
            cands = wl[(typ_d[tid], src_d[tid])]
            placed = False
            for c in cands:
                ri = REGIONS.index(c)
                lo = int(arr_d[tid])
                hi = min(int(last_d[tid] - dur), 2406)
                if occ[lo:hi + 1, ri].max() + g > cap[ri]:
                    continue
                for s in range(lo, hi + 1):
                    h1 = min(int(np.ceil(s + dur)), 2407)
                    if (occ[s:h1, ri] + g <= cap[ri]).all():
                        rmap[tid], smap[tid] = ri, s
                        occ[s:h1, ri] += g
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                ri = REGIONS.index(src_d[tid])
                st = int(arr_d[tid])
                rmap[tid], smap[tid] = ri, st
                h1 = min(int(np.ceil(st + dur)), 2407)
                occ[st:h1, ri] += g
                ok = False
        if not ok:
            continue
        new = pd.DataFrame({"TaskID": list(rmap.keys()), "Region": [REGIONS[ri] for ri in rmap.values()],
                            "StartHour": list(smap.values())})
        nF = eval_sched_df(new)
        if nF[0] <= 1e11 and (nF <= bestF).all():
            best, bestF = new, nF
        elif nF[0] <= 1e11 and rng.rand() < np.exp(-(nF[0] - bestF[0]) / T):
            best = new
        T *= 0.995
    return bestF[None, :]


def run_lagrangian(seed: int, wt, rt, s10, s20, params, n_iter: int = 5) -> np.ndarray:
    """拉格朗日对偶引导贪心：容量影子价格 π 迭代 + 任务边际排序放置。"""
    rng = np.random.RandomState(seed + 400)
    wl = {(r.TaskType, r.SourceRegion): r.Reachable.split("|")
          for r in pd.read_csv(CLEAN / "whitelist.csv").itertuples(index=False)}
    cap = np.array([params["cap"][r] for r in REGIONS], dtype=float)
    price = s20._hourly_price(rt).to_numpy(dtype=float)
    pi = np.zeros((2407, 6))
    bestF = None
    for it in range(n_iter):
        # 任务独立子问题：边际 = 电价成本 + π·占用；白名单内选最小可行
        occ = np.zeros((2407, 6))
        rows = []
        for rec in wt.sort_values(["Priority", "ArrivalHour"],
                                  ascending=[False, True]).itertuples(index=False):
            g = float(rec.GPU_Demand)
            dur = float(rec.dur_h)
            if rec.TaskType == "RealTimeInference":
                r = REGIONS.index(rec.SourceRegion)
                st = int(rec.ArrivalHour)
            else:
                cands = sorted(wl[(rec.TaskType, rec.SourceRegion)],
                               key=lambda c: price[rec.ArrivalHour,
                                                   REGIONS.index(c)])
                r = st = None
                for c in cands:
                    ri = REGIONS.index(c)
                    lo = int(rec.ArrivalHour)
                    hi = min(int(rec.LatestFinishHour - dur), 2406)
                    if occ[lo:hi + 1, ri].max() + g > cap[ri]:
                        continue
                    for s in range(lo, hi + 1):
                        h1 = min(int(np.ceil(s + dur)), 2407)
                        if (occ[s:h1, ri] + g <= cap[ri]).all():
                            r, st = ri, s
                            break
                    if r is not None:
                        break
                if r is None:
                    r, st = REGIONS.index(rec.SourceRegion), int(rec.ArrivalHour)
            rows.append((rec.TaskID, REGIONS[r], st))
            h1 = min(int(np.ceil(st + dur)), 2407)
            occ[st:h1, r] += g
        sched = pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])
        F = eval_sched_df(sched)
        if bestF is None or (F < bestF).all():
            bestF = F
        # 子梯度更新 π（超容区升、富余区降）
        viol = occ - cap
        pi += 1.5 * np.clip(viol, -50, 50) / max(it + 1, 1)
        pi = np.clip(pi, 0, None)
    return bestF[None, :]


# ---------- 对比与裁决 ----------

def hypervolume(F: np.ndarray, ref: np.ndarray, n_samples: int = 20000) -> float:
    """蒙特卡洛超体积近似（归一化空间）。"""
    lo = F.min(axis=0) - 1e-6
    hi = ref.copy()
    rng = np.random.RandomState(0)
    pts = rng.rand(n_samples, 4) * (hi - lo) + lo
    dominated = np.zeros(n_samples, dtype=bool)
    for f in F:
        dominated |= np.all(pts <= f, axis=1)
    return float(dominated.mean() * np.prod(hi - lo))


def main() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)
    c = _load_ctx()
    with mp.Pool(N_WORKERS, initializer=_load_ctx) as pool:
        fronts = {}
        for seed in range(N_SEED):
            for mname, fn in [("nsga2", run_nsga2), ("nsga3", run_nsga3_lite),
                              ("moead", run_moead)]:
                t0 = time.time()
                fronts.setdefault(mname, []).extend(fn(seed, pool).tolist())
                print(f"[progress] {mname} seed{seed} done {time.time()-t0:.0f}s",
                      flush=True)
        for seed in range(N_SEED):
            for mname, fn in [("alns", run_alns), ("lagrangian", run_lagrangian)]:
                t0 = time.time()
                fronts.setdefault(mname, []).extend(
                    fn(seed, c["wt"], c["rt"], c["s10"], c["s20"],
                       c["params"]).tolist())
                print(f"[progress] {mname} seed{seed} done {time.time()-t0:.0f}s",
                      flush=True)

    merged = {}
    for m, lst in fronts.items():
        F = np.array(lst)
        if len(F):
            merged[m] = F[non_dom_front(F)]
    for m, F in merged.items():
        pd.DataFrame(F, columns=["cost_wan", "carbon_t", "latency_ms", "nu_neg"])\
            .to_csv(OUT_Q2 / f"method_front_{m}.csv", index=False)
    allF = np.vstack(list(merged.values()))
    ref = allF.max(axis=0) * 1.05
    hv = {m: hypervolume(F, ref, n_samples=200000) for m, F in merged.items()}
    # 2D ???????????? + ???? gap
    c_construct = 183706.1
    best_cost = {m: float(F[:, 0].min()) for m, F in merged.items()}
    gap_construct = {m: (best_cost[m] - c_construct) / c_construct * 100
                     for m, F in merged.items()}
    # 覆盖分析：每方法前沿被其他方法前沿支配的比例
    dom_report = {}
    for m, F in merged.items():
        others = np.vstack([v for k, v in merged.items() if k != m])
        surv = np.zeros(len(F), dtype=bool)
        for i, f in enumerate(F):
            d = np.all(others <= f, axis=1) & np.any(others < f, axis=1)
            surv[i] = not d.any()
        dom_report[m] = {"n_front": int(len(F)),
                         "not_dominated_share": float(surv.mean()),
                         "hv": hv[m]}
    report = {"per_method": dom_report,
              "best_cost": best_cost, "gap_vs_construct_pct": gap_construct,
              "winner": max(dom_report, key=lambda k: dom_report[k]["hv"]),
              "ref_point": ref.tolist(),
              "caliber": ("阶段A 策略空间(NSGA-II/III-lite/MOEA-D, 40×60×3seed)；"
                          "阶段B 任务级(ALNS 300轮, 拉格朗日 30轮, 3seed)；"
                          "统一模板口径评估器；HV 蒙特卡洛 2e4 样本")}
    with open(OUT_Q2 / "method_arena.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(merged.keys())
    hvs = [hv[m] for m in names]
    bars = ax.bar(names, hvs, color=["#95a5a6", "#2980b9", "#8e44ad",
                                     "#27ae60", "#c0392b"])
    for b, v in zip(bars, hvs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("超体积 HV（越大越好）")
    ax.set_title(f"Q2 五方法判别实验: 赢家 = {report['winner']}")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_method_arena.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
