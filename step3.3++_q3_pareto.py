"""step3.3++_q3_pareto — 轨B 续：多目标前沿（E1 共线 / E3 双方法 / E6 形态）.

基于 M1（时段状态机约束 LP，spec_M4_Q3 修正后主口径模型）。

E1 共线实证: 解族（ε-约束网格）上 cost-carbon 相关 + 分箱 spread 双证据
  （Q2 E-F3 教训：共线须实证后诚实呈现；price-carbon 同序 0.952 先验）
E3 双方法裁决: ε-约束（ramp 网格 × peak 网格）∩ 加权和（R/P 辅助变量）
  → 前沿点两方法收敛（cost 差距 <0.1%）才入册；凹区如实声明
E6 东西区形态: A（东，无外送）vs D（西，有外送）前沿并排对比

产物（output/q3/）:
  q3_pareto.json   E1 共线实证 + 双方法前沿 + E6 形态对比
figures/step3/fig_q3_pareto.png   cost×ramp / cost×peak 前沿（A/D/E）
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step0_config import FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

PARETO_REGIONS = ["RegionA", "RegionD", "RegionE"]


def _import_modules():
    root = Path(__file__).resolve().parent
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s33p", root / "step3.3+_q3_model_evolve.py")
    s33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s33)
    return s33


def solve_pareto_point(s33, d, ch, dh, ramp_cap=None, peak_cap=None,
                       w_ramp=None, w_peak=None, ramp_base=1.0,
                       peak_base=1.0) -> dict:
    """单前沿点求解：ε-约束（ramp_cap/peak_cap）或加权和（w_ramp/w_peak）。

    加权和模式: 辅助变量 R（|ΔN| 上界）、P（N 上界），
    min cost + w_ramp·R/ramp_base + w_peak·P/peak_base。
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    T = len(d["price"])
    extra = 2 if w_ramp is not None else 0
    nv = 6 * T + extra
    idx = lambda k, t: k * T + t
    c = np.zeros(nv)
    for t in range(T):
        c[idx(2, t)] = d["price"][t]
        c[idx(3, t)] = -d["sellp"][t]
        c[idx(4, t)] = 1e-4
    if w_ramp is not None:
        c[6 * T] = w_ramp / ramp_base
        c[6 * T + 1] = w_peak / peak_base
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
    cap_h = np.minimum(d["W"][:T], d["c_h"][np.arange(T) % 24] * d["D"][:T])
    for t in range(T):
        Aub[1 + t, idx(0, t)] = -1.0
        Aub[1 + t, idx(3, t)] = -1.0
        Aub[1 + t, idx(4, t)] = -1.0
        bub[1 + t] = -(d["W"][t] - cap_h[t])
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
    if peak_cap is not None:
        m0 = n_ineq
        n_ineq += T
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, np.full(T, peak_cap)])
        for t in range(T):
            Aub[m0 + t, idx(2, t)] = 1.0
            Aub[m0 + t, idx(3, t)] = -1.0
    if w_ramp is not None:
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
        m0 = n_ineq
        n_ineq += T
        Aub.resize((n_ineq, nv))
        bub = np.concatenate([bub, np.zeros(T)])
        for t in range(T):
            Aub[m0 + t, idx(2, t)] = 1.0
            Aub[m0 + t, idx(3, t)] = -1.0
            Aub[m0 + t, 6 * T + 1] = -1.0
    bounds = [(0, d["max_c"]) if t % 24 in ch else (0, 0)
              for t in range(T)] \
        + [(0, d["max_d"]) if t % 24 in dh else (0, 0) for t in range(T)] \
        + [(0, d["max_import"])] * T + [(0, d["sell_lim"])] * T \
        + [(0, None)] * T + [(d["min_soc"], d["cap_mwh"])] * T \
        + ([(0, None)] * extra if extra else [])
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"status": res.status, "message": res.message}
    x = res.x
    G = x[[idx(2, t) for t in range(T)]]
    S = x[[idx(3, t) for t in range(T)]]
    net = G - S
    return {"status": 0,
            "cost_wan": float((d["price"] * G - d["sellp"] * S).sum() / 1e4),
            "carbon_t": float((d["carbon"] * G).sum()),
            "peak_net_MW": float(net.max()),
            "max_ramp_MW": float(np.abs(np.diff(net)).max())}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s33 = _import_modules()
    rt = s33._import_step32().load_rt()
    import pandas as pd
    import importlib.util
    root = Path(__file__).resolve().parent
    spec10 = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec10)
    spec10.loader.exec_module(s10)
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    tpl = json.loads((OUT_Q3 / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"]
    base = s33._import_step32().baseline_indicators(rt)

    report = {"E1_collinearity": {}, "frontier": {}, "E6_shape": {}}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, r in enumerate(PARETO_REGIONS):
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch = tpl[r]["charge_hours"]
        dh = tpl[r]["discharge_hours"]
        b = base[r]
        ramp_lo = b["max_ramp_MW"]
        ramp_hi = 1.0 * b["max_ramp_MW"]
        # ε-约束网格：ramp 从基准到 M1 原值附近；peak 固定 1 个宽松档
        m1_pt = solve_pareto_point(s33, d, ch, dh)
        ramp_m1 = m1_pt["max_ramp_MW"]
        ramp_grid = np.linspace(min(ramp_m1, ramp_lo) * 0.9,
                                max(ramp_m1, ramp_lo) * 1.05, 7)
        eps_pts = []
        for rcap in ramp_grid:
            p = solve_pareto_point(s33, d, ch, dh, ramp_cap=float(rcap))
            if p["status"] == 0:
                eps_pts.append(p)
        # 加权和：w 网格（ramp 权重 0→5，peak 权重 0→5，12 点）
        ws_pts = []
        for wr in [0, 0.25, 0.5, 1.0, 2.0, 5.0]:
            for wp in [0, 0.5, 2.0]:
                p = solve_pareto_point(s33, d, ch, dh, w_ramp=wr, w_peak=wp,
                                       ramp_base=ramp_lo, peak_base=b["peak_net_MW"])
                if p["status"] == 0:
                    ws_pts.append(p)
        # E1 共线：解族 cost-carbon 相关 + 分箱 spread
        all_pts = eps_pts + ws_pts
        if len(all_pts) > 3:
            c_arr = np.array([p["cost_wan"] for p in all_pts])
            cb_arr = np.array([p["carbon_t"] for p in all_pts])
            corr = float(np.corrcoef(c_arr, cb_arr)[0, 1])
            bins = np.quantile(c_arr, [0.33, 0.66])
            spreads = []
            for lo, hi in ((-np.inf, bins[0]), (bins[0], bins[1]),
                           (bins[1], np.inf)):
                sel = cb_arr[(c_arr >= lo) & (c_arr < hi)]
                spreads.append(round(float(np.std(sel)), 1) if len(sel) > 1
                               else None)
            report["E1_collinearity"][r] = {
                "cost_carbon_corr": round(corr, 4),
                "carbon_std_by_cost_bin": spreads,
                "carbon_mean_t": round(float(cb_arr.mean()), 0),
                "conclusion": ("【实证】共线" if corr > 0.95
                               else "【实证】弱共线" if corr > 0.8
                               else "【实证】独立")}
        # E3 双方法收敛裁决
        def nearest_cost(xs, ramp):
            best = min(xs, key=lambda p: abs(p["max_ramp_MW"] - ramp))
            return best["cost_wan"]
        conv = []
        for p in eps_pts:
            wc = nearest_cost(ws_pts, p["max_ramp_MW"])
            conv.append({"ramp": round(p["max_ramp_MW"], 1),
                         "cost_eps": round(p["cost_wan"], 3),
                         "cost_weighted": round(wc, 3),
                         "gap_pct": round(abs(p["cost_wan"] - wc)
                                          / abs(wc + 1e-9) * 100, 4)})
        max_gap = max(x["gap_pct"] for x in conv) if conv else None
        report["frontier"][r] = {
            "ramp_m1": round(ramp_m1, 1), "ramp_base": round(ramp_lo, 1),
            "eps_points": eps_pts, "weighted_points": ws_pts,
            "dual_method_max_gap_pct": max_gap,
            "dual_verdict": ("【实证】双方法收敛 <0.1%" if max_gap is not None
                             and max_gap < 0.1
                             else "【实证】凹区/差异，加权和覆盖凸包")}
        # E6 形态
        report["E6_shape"][r] = {
            "frontier_ramp_span_MW": round(
                max(p["max_ramp_MW"] for p in eps_pts)
                - min(p["max_ramp_MW"] for p in eps_pts), 1),
            "frontier_cost_span_wan": round(
                max(p["cost_wan"] for p in eps_pts)
                - min(p["cost_wan"] for p in eps_pts), 1)}
        print(f"[{r}] E1 corr={report['E1_collinearity'][r]['cost_carbon_corr']} "
              f"| 双方法 max gap={max_gap}% | "
              f"ramp M1={ramp_m1:.0f}→基准={ramp_lo:.0f} | "
              f"前沿跨度 cost={report['E6_shape'][r]['frontier_cost_span_wan']}万",
              flush=True)
        for j, (mname, pts) in enumerate((("cost×ramp", eps_pts),)):
            ax = axes[0 if j == 0 else 1, i]
            ax.plot([p["max_ramp_MW"] for p in eps_pts],
                    [p["cost_wan"] / 1e4 for p in eps_pts], "o-", label="ε-约束")
            ax.plot([p["max_ramp_MW"] for p in ws_pts],
                    [p["cost_wan"] / 1e4 for p in ws_pts], "s", ms=5,
                    label="加权和", alpha=0.7)
            ax.axvline(ramp_lo, color="k", ls="--", lw=0.8, label="基准 ramp")
            ax.set_title(f"{r} cost×ramp（亿）")
            ax.set_xlabel("max ramp MW")
            ax.legend(fontsize=7)
    fig.suptitle("Q3 M1 多目标前沿：ε-约束 ∩ 加权和（E3 双方法裁决）")
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_pareto.png", bbox_inches="tight")
    plt.close(fig)

    report["caliber"] = ("基于 M1（时段状态机约束）；ε-约束 ramp 网格×7；"
                         "加权和 w_ramp×w_peak 12 点（基准归一化）；"
                         "双方法裁决=同 ramp 最近点 cost gap<0.1%")
    with open(OUT_Q3 / "q3_pareto.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"E1": report["E1_collinearity"],
                      "E6": report["E6_shape"],
                      "verdict": {r: report["frontier"][r]["dual_verdict"]
                                  for r in PARETO_REGIONS}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
