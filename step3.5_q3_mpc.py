"""step3.5_q3_mpc — Q3 随机扩展：MPC 滚动闭环（M1 时段约束场景化）.

口径（spec_M4_Q3 D-5，诚实声明：数据 W 为确定性模板，扰动自造）:
  场景: W 公式模板 × AR(1) 扰动（σ∈{10%,20%,30%}）× K-means k=12 缩减
  主实现: 确定性等价 MPC——每 24h 窗口用场景加权均值 W̄ 求解 M1 约束 LP，
          执行首时段 → SOC 递推 → 滚动（预算纪律；反馈价值=滚动重估收益）
  对照纯度: open-loop（均值场景全时段一次求解）vs closed-loop（滚动）
  标准形式演示: D 区单窗口多场景联合 LP（首时段耦合 nonanticipativity）
  ——与确定性等价对比（方法学差异量化）
窗口效应基线（探路已量化）: D 0.06% / E 2.17%（全知 vs 滚动）

产物: output/q3/q3_mpc.json + figures/step3/fig_q3_mpc_robust.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from step0_config import FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

MPC_REGIONS = ["RegionA", "RegionD", "RegionE", "RegionF"]
SIGMAS = [0.10, 0.20, 0.30]
WINDOW = 24
N_SCEN = 64
K = 12
SEED = 42


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


def gen_scenarios(w_template: np.ndarray, sigma: float, phi: float = 0.8,
                  n_scen: int = N_SCEN, seed: int = SEED) -> np.ndarray:
    """AR(1) 扰动场景（step3.1 同款，seed 固定可复现）。"""
    rng = np.random.RandomState(seed)
    T = len(w_template)
    S = np.zeros((n_scen, T))
    for s in range(n_scen):
        z = rng.randn(T)
        eps = np.zeros(T)
        for t in range(1, T):
            eps[t] = phi * eps[t - 1] + sigma * z[t]
        S[s] = w_template * np.clip(1 + eps, 0.1, 2.0)
    return S


def reduce_scenarios(S: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """K-means 缩减 → (代表场景, 权重)。"""
    km = KMeans(n_clusters=k, random_state=SEED, n_init=5).fit(S)
    reps = km.cluster_centers_
    w = np.bincount(km.labels_, minlength=k) / len(km.labels_)
    return reps, w


def rolling_mpc(s33, d, ch, dh, w_mean: np.ndarray, T: int = 2407,
                w: int = WINDOW) -> dict:
    """确定性等价 MPC：24h 窗口滚动，执行首时段，SOC 递推衔接。"""
    soc = d["init_soc"]
    rows = []
    for t0 in range(0, T, 1):
        h = t0 % 24
        # 每窗口在 t0 处求解（窗口起点 = t0，长度 w）
        if t0 % w == 0:
            tw = min(w, T - t0)
            d_w = dict(d)
            d_w["W"] = w_mean[t0:t0 + tw]
            d_w["D"] = d["D"][t0:t0 + tw]
            d_w["price"] = d["price"][t0:t0 + tw]
            d_w["sellp"] = d["sellp"][t0:t0 + tw]
            d_w["carbon"] = d["carbon"][t0:t0 + tw]
            d_w["init_soc"] = soc
            res = s33.solve_region_timed(d_w, n_hours=tw, charge_hours=ch,
                                         discharge_hours=dh)
            if res["status"] != 0:
                return {"status": res["status"], "rows": [],
                        "message": res["message"]}
            win_rows = res["rows"]
        rec = win_rows[min(h, len(win_rows) - 1)]
        # 执行首决策 → SOC 递推
        soc = (soc + d["eta_c"] * rec["Pc"] - rec["Pd"] / d["eta_d"])
        rows.append({"Hour": t0, "G": rec["G"], "S": rec["S"],
                     "Pc": rec["Pc"], "Pd": rec["Pd"], "Q": rec["Q"],
                     "SOC": soc})
    # 指标重算
    G = np.array([x["G"] for x in rows])
    S = np.array([x["S"] for x in rows])
    net = G - S
    return {"status": 0,
            "cost_wan": float((d["price"] * G - d["sellp"] * S).sum() / 1e4),
            "carbon_t": float((d["carbon"] * G).sum()),
            "peak_net_MW": float(net.max()),
            "vol_std_MW": float(net.std()),
            "max_ramp_MW": float(np.abs(np.diff(net)).max()),
            "rows": rows}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl = _setup()
    report = {"mpc": {}, "open_loop": {}, "window_effect": {}}

    for r in MPC_REGIONS:
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        w_template = d["W"]
        # 全知 M1（确定性基线）
        full = s33.solve_region_timed(d, charge_hours=ch, discharge_hours=dh)
        report["mpc"][r] = {"full_knowledge_cost_wan": round(full["cost_wan"], 2)}
        for sigma in SIGMAS:
            S = gen_scenarios(w_template, sigma)
            reps, wts = reduce_scenarios(S)
            w_mean = (reps * wts[:, None]).sum(axis=0)
            # 确定性等价 MPC（滚动）
            mpc = rolling_mpc(s33, d, ch, dh, w_mean)
            # open-loop（均值场景一次求解全时段）
            d_ol = dict(d)
            d_ol["W"] = w_mean
            ol = s33.solve_region_timed(d_ol, charge_hours=ch,
                                        discharge_hours=dh)
            key = f"sigma_{int(sigma * 100)}"
            report["mpc"][r][key] = {
                "mpc_cost_wan": round(mpc["cost_wan"], 2)
                if mpc["status"] == 0 else None,
                "openloop_cost_wan": round(ol["cost_wan"], 2),
                "feedback_value_wan": round(
                    (ol["cost_wan"] - mpc["cost_wan"]) / 1e4, 2)  # 滚动收益（万元级）
                if mpc["status"] == 0 else None,
                "mpc_ramp_MW": round(mpc["max_ramp_MW"], 1)
                if mpc["status"] == 0 else None,
                "mpc_peak_MW": round(mpc["peak_net_MW"], 1)
                if mpc["status"] == 0 else None,
            }
            print(f"[{r}] σ={sigma}: MPC={report['mpc'][r][key]['mpc_cost_wan']} "
                  f"OL={ol['cost_wan']:.2f} | "
                  f"反馈价值={report['mpc'][r][key]['feedback_value_wan']}万",
                  flush=True)
        # 窗口效应（滚动 vs 全知，σ=0.2 均值场景）
        S = gen_scenarios(w_template, 0.2)
        reps, wts = reduce_scenarios(S)
        w_mean = (reps * wts[:, None]).sum(axis=0)
        mpc = rolling_mpc(s33, d, ch, dh, w_mean)
        if mpc["status"] == 0:
            we = (mpc["cost_wan"] - full["cost_wan"]) \
                / abs(full["cost_wan"]) * 100
            report["window_effect"][r] = {
                "rolling_cost_wan": round(mpc["cost_wan"], 2),
                "full_cost_wan": round(full["cost_wan"], 2),
                "window_effect_pct": round(we, 3)}
            print(f"[{r}] 窗口效应 {we:.3f}%", flush=True)

    # 标准随机 MPC 演示（D 区单窗口，场景联合 LP + 首时段耦合）
    r = "RegionD"
    d = s33._load_region_data(r)
    d["c_h"] = np.asarray(consume[r], dtype=float)
    ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
    S = gen_scenarios(d["W"][:WINDOW], 0.2)
    reps, wts = reduce_scenarios(S, k=8)
    # 联合 LP：min Σ p_ω·cost_ω，首时段（t=0）决策跨场景一致
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    tw = WINDOW
    K_ = 8
    nv = 6 * tw * K_
    idx = lambda k, t, s: (k * tw + t) * K_ + s
    c = np.zeros(nv)
    for s in range(K_):
        for t in range(tw):
            c[idx(2, t, s)] = wts[s] * d["price"][t]
            c[idx(3, t, s)] = -wts[s] * d["sellp"][t]
            c[idx(4, t, s)] = 1e-4 * wts[s]
    n_ineq = K_ * (1 + tw) + 4 * tw * (K_ - 1)
    Aub = lil_matrix((n_ineq, nv))
    bub = np.zeros(n_ineq)
    row = 0
    for s in range(K_):
        Aub[row, idx(5, tw - 1, s)] = -1.0
        bub[row] = -d["init_soc"]
        row += 1
        cap_h = np.minimum(reps[s], d["c_h"][np.arange(tw) % 24] * d["D"][:tw])
        for t in range(tw):
            Aub[row, idx(0, t, s)] = -1.0
            Aub[row, idx(3, t, s)] = -1.0
            Aub[row, idx(4, t, s)] = -1.0
            bub[row] = -(reps[s][t] - cap_h[t])
            row += 1
    Aeq = lil_matrix((2 * tw * K_ + 4 * (K_ - 1), nv))
    beq = np.zeros(2 * tw * K_ + 4 * (K_ - 1))
    erow = 0
    for s in range(K_):
        for t in range(tw):
            Aeq[erow, idx(1, t, s)] = 1.0
            Aeq[erow, idx(2, t, s)] = 1.0
            for k in (0, 3, 4):
                Aeq[erow, idx(k, t, s)] = -1.0
            beq[erow] = d["D"][t] - reps[s][t]
            erow += 1
            Aeq[erow, idx(5, t, s)] = 1.0
            Aeq[erow, idx(5, t - 1, s)] = -1.0 if t > 0 else 0.0
            Aeq[erow, idx(0, t, s)] = -d["eta_c"]
            Aeq[erow, idx(1, t, s)] = 1.0 / d["eta_d"]
            beq[erow] = d["init_soc"] if t == 0 else 0.0
            erow += 1
    # 首时段耦合（nonanticipativity）：跨场景 Pc/Pd/G/S 一致
    for k in (0, 1, 2, 3):
        for s in range(1, K_):
            Aeq[erow, idx(k, 0, s)] = 1.0
            Aeq[erow, idx(k, 0, 0)] = -1.0
            beq[erow] = 0.0
            erow += 1
    bounds = []
    for k in range(6 * tw):
        pass
    bounds = [(0, d["max_c"]) if k % tw in ch else (0, 0)
              for k in range(6 * tw) for _ in range(K_)]
    bounds = []
    for t in range(tw):
        for _ in range(K_):
            bounds += [(0, d["max_c"]) if t % 24 in ch else (0, 0),
                       (0, d["max_d"]) if t % 24 in dh else (0, 0),
                       (0, d["max_import"]), (0, d["sell_lim"]),
                       (0, None), (d["min_soc"], d["cap_mwh"])]
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(),
                  b_eq=beq, bounds=bounds, method="highs")
    report["standard_scenario_mpc_demo"] = {
        "region": r, "status": res.status,
        "n_scenarios": K_, "window": tw,
        "first_stage_decision": {
            "Pc": round(float(res.x[idx(0, 0, 0)]), 2) if res.success else None,
            "Pd": round(float(res.x[idx(1, 0, 0)]), 2) if res.success else None,
            "G": round(float(res.x[idx(2, 0, 0)]), 2) if res.success else None},
        "note": ("标准随机 MPC 形式演示（首时段耦合）；全量用确定性等价"
                 "（预算纪律+诚实声明）")}
    print(f"标准场景 MPC 演示 status={res.status}", flush=True)

    # 图：σ 三档 MPC vs OL 成本（D/E）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r, color in (("RegionD", "#4c72b0"), ("RegionE", "#dd8452"),
                     ("RegionF", "#c44e52")):
        xs, ys_mpc, ys_ol = [], [], []
        for sigma in SIGMAS:
            key = f"sigma_{int(sigma * 100)}"
            v = report["mpc"][r][key]
            xs.append(sigma * 100)
            ys_mpc.append(v["mpc_cost_wan"])
            ys_ol.append(v["openloop_cost_wan"])
        ax.plot(xs, ys_mpc, "o-", label=f"{r} MPC", color=color)
        ax.plot(xs, ys_ol, "s--", label=f"{r} open-loop", color=color, alpha=0.6)
    ax.set_xlabel("扰动 σ（%）")
    ax.set_ylabel("成本（万元）")
    ax.set_title("MPC 滚动 vs open-loop（自造 AR(1) 扰动，方法学演示）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_mpc_robust.png", bbox_inches="tight")
    plt.close(fig)

    report["caliber"] = ("确定性等价 MPC（场景加权均值，24h 滚动，首时段执行，"
                         "SOC 递推衔接）+ open-loop 对照 + D 区标准联合场景演示；"
                         "扰动自造（数据 W 为 T1 确定性模板），σ∈{10,20,30%}，"
                         "K-means k=12（质量<5%）；MPC 价值=方法论演示+鲁棒对比；"
                         "【sum_10 修正】确定性等价用均值场景→σ 数学上无区分度"
                         "（反馈价值≈0 系方法局限）；σ 真实区分度见 step3.7++ X15："
                         "全时段场景期望成本随 σ 上升（7,560→7,582→7,622，+0.8%）"
                         "——波动有真实成本（场景生成 clip(0.1,2.0) 截断效应需声明）")
    with open(OUT_Q3 / "q3_mpc.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"mpc": {r: report["mpc"][r] for r in MPC_REGIONS},
                      "window_effect": report["window_effect"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
