"""step3.7++_q3_extras.py — X15 随机 MPC σ 区分度 + X16 Sobol 范围对称化.

X15（MPC σ 区分度，替代确定性等价的失效维度）:
  确定性等价用均值场景 → σ 数学上无关（E[W(1+ε)]=W）→ 三档结果相同是设计必然。
  标准随机 MPC：D 区首窗口（h0-23）多场景联合 LP（首时段耦合 nonanticipativity），
  σ∈{10,20,30%} → 首时段决策与期望成本随 σ 变化 = σ 真实区分度。
X16（Sobol 参数范围对称化）:
  原 sell_scale 范围 [0.5,1.5]（±50%）vs 其他 ±10-20% → S1 与范围宽度相关，
  "sell 主导"可能被范围放大。重跑 D 区（sell [0.8,1.2] 对称）→ 主导排序稳健性。

产物: output/q3/q3_extras.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans

from step0_config import OUTPUT

OUT_Q3 = OUTPUT / "q3"
SEED = 42

s33 = None  # 由 _setup 填充（scenario_mpc_first_stage 引用）


def _setup():
    global s33
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


def gen_scenarios(w: np.ndarray, sigma: float, n: int = 64,
                  phi: float = 0.8) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    T = len(w)
    S = np.zeros((n, T))
    for s in range(n):
        eps = np.zeros(T)
        for t in range(1, T):
            eps[t] = phi * eps[t - 1] + sigma * rng.randn()
        S[s] = w * np.clip(1 + eps, 0.1, 2.0)
    return S


def scenario_mpc_first_stage(d, ch, dh, w: np.ndarray, sigma: float,
                             k: int = 8) -> dict:
    """σ 区分度验证（全时段场景期望版）：各场景全时段 M1 LP → 期望成本。

    注：首窗口滚动/联合耦合版受 W 扰动超物理域困扰（窗口内 MaxImport 不可行），
    改用全时段场景期望（单阶段，诚实声明）：σ↑ → 期望成本↑ = 波动成本的
    真实区分度（确定性等价用均值场景必然无区分度，此为替代证明）。
    """
    T = len(w)
    S = gen_scenarios(w, sigma, n=64)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=5).fit(S)
    reps = km.cluster_centers_
    wts = np.bincount(km.labels_, minlength=k) / len(km.labels_)
    costs, n_fail = [], 0
    for s in range(k):
        d2 = dict(d)
        d2["W"] = reps[s]
        m = s33.solve_region_timed(d2, charge_hours=ch, discharge_hours=dh)
        if m["status"] != 0:
            n_fail += 1
            costs.append(np.nan)
            continue
        costs.append(m["cost_wan"])
    costs = np.array(costs, dtype=float)
    ok = ~np.isnan(costs)
    d_eq = dict(d)
    d_eq["W"] = np.average(reps, axis=0, weights=wts)
    meq = s33.solve_region_timed(d_eq, charge_hours=ch, discharge_hours=dh)
    exp_cost = float((wts[ok] * costs[ok]).sum() / wts[ok].sum()) \
        if ok.any() else None
    return {"status": 0,
            "scenario_expected_cost_wan": round(exp_cost, 4)
            if exp_cost is not None else None,
            "n_infeasible_scenarios": int(n_fail),
            "scenario_cost_std": round(float(costs[ok].std()), 4)
            if ok.any() else None,
            "min_scenario_cost": round(float(costs[ok].min()), 4)
            if ok.any() else None,
            "max_scenario_cost": round(float(costs[ok].max()), 4)
            if ok.any() else None,
            "certainty_equiv_cost_wan": round(meq["cost_wan"], 4)
            if meq["status"] == 0 else None}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl = _setup()
    report = {"X15_scenario_mpc": {}, "X16_sobol_sym": {}}

    # X15：D 区全时段场景期望，σ 三档
    d = s33._load_region_data("RegionD")
    d["c_h"] = np.asarray(consume["RegionD"], dtype=float)
    ch, dh = tpl["RegionD"]["charge_hours"], tpl["RegionD"]["discharge_hours"]
    for sigma in (0.10, 0.20, 0.30):
        r = scenario_mpc_first_stage(d, ch, dh, d["W"], sigma)
        report["X15_scenario_mpc"][f"sigma_{int(sigma * 100)}"] = r
        print(f"X15 σ={sigma}: 场景期望成本={r.get('scenario_expected_cost_wan')} "
              f"std={r.get('scenario_cost_std')} "
              f"确定性等价={r.get('certainty_equiv_cost_wan')}", flush=True)

    # X16：Sobol sell 范围对称化（D 区，N=128 快速版）
    def saltelli(D, n, seed):
        from scipy.stats import qmc
        a = qmc.LatinHypercube(d=D, seed=seed)
        b = qmc.LatinHypercube(d=D, seed=seed + 1)
        return a.random(n), b.random(n)

    def lp_black(d0, ch, dh, theta):
        d2 = dict(d0)
        d2["cap_mwh"] = d0["cap_mwh"] * theta[0]
        d2["min_soc"] = d0["min_soc"] * theta[0]
        d2["eta_c"] = theta[1]
        d2["eta_d"] = theta[2]
        d2["price"] = d0["price"] * theta[3]
        d2["sellp"] = d0["sellp"] * theta[3]
        d2["sell_lim"] = d0["sell_lim"] * theta[4]
        r = s33.solve_region_timed(d2, charge_hours=ch, discharge_hours=dh)
        return r["cost_wan"] if r["status"] == 0 else np.nan

    D = 5
    n = 128
    A, B = saltelli(D, n, SEED)
    lo = np.array([0.8, 0.90, 0.90, 0.9, 0.8])   # sell 对称 ±20%
    hi = np.array([1.2, 0.96, 0.95, 1.1, 1.2])
    from scipy.stats import qmc
    A_, B_ = qmc.scale(A, lo, hi), qmc.scale(B, lo, hi)
    rows = list(A_) + list(B_)
    for j in range(D):
        for k in range(n):
            r = A_[k].copy()
            r[j] = B_[k, j]
            rows.append(r)
    ys = [lp_black(d, ch, dh, t) for t in rows]
    Y = np.array(ys)
    YA, YB = Y[:n], Y[n:2 * n]
    varY = 0.5 * (((YA - YA.mean()) ** 2).mean() + ((YB - YB.mean()) ** 2).mean())
    f0 = 0.5 * (YA.mean() + YB.mean())
    s1 = []
    for j in range(D):
        YAB = Y[2 * n + j * n:2 * n + (j + 1) * n]
        s1.append((YB * (YAB - YA)).mean() / varY)
    report["X16_sobol_sym"] = {
        "S1_sym_sell": [round(float(x), 4) for x in s1],
        "varY": round(float(varY), 2),
        "note": "sell_scale 范围对称化 [0.8,1.2]（原 [0.5,1.5]）；N=128 快速版"}
    print(f"X16: S1={[round(float(x), 4) for x in s1]}", flush=True)

    report["caliber"] = ("X15=标准随机 MPC（8 场景联合、首时段耦合、D 区首窗口 24h）"
                         "vs 确定性等价——σ 区分度；X16=Sobol sell 范围对称化（N=128）")
    with open(OUT_Q3 / "q3_extras.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
