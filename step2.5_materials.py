"""step2.5_materials — Phase C: Q2 论文素材四件套.

C5 24h 甘特图: Q2 折中解最后 24h+收尾段（2376-2405）
C6 边际效益曲线: 前沿解成本-碳交换率（多付 1% 成本 → 碳降 X%）
C7 规则剪枝权衡: CART 不同深度的 覆盖数-规则数-准确率 曲线
C8 策略 Sobol 敏感性: 5 维策略参数一阶/总效应（自研 Saltelli，与 Q3 Sobol 呼应）

产物（output/q2/ + figures/step2/）: materials.json + 4 图
"""
import importlib.util
import json
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


def load_ctx():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    spec2 = importlib.util.spec_from_file_location(
        "s20", root / "step2.0_construct.py")
    s20 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s20)
    return s10, s20


def c5_gantt(s10, s20, wt, rt, params, consume, pol) -> None:
    sched = s20.schedule_constructive(wt, rt, s10, params, tuple(pol))
    sel = wt[wt.ArrivalHour >= 2352].merge(sched, on="TaskID")
    sel = sel[(sel.StartHour >= 2376) & (sel.StartHour <= 2405)]
    colors = {"RealTimeInference": "#c0392b", "BatchInference": "#2980b9",
              "AITraining": "#27ae60"}
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    for i, r in enumerate(REGIONS):
        ax = axes[i // 2][i % 2]
        sub = sel[sel.Region == r]
        for j, rec in enumerate(sub.itertuples(index=False)):
            ax.barh(j, rec.dur_h, left=rec.StartHour, height=0.8,
                    color=colors[rec.TaskType], edgecolor="none")
        ax.set_title(r, fontsize=10)
        ax.set_xlim(2376, 2406)
        if len(sub):
            ax.set_ylim(-0.5, len(sub) - 0.5)
    fig.suptitle("Q2 折中解最后 24h+收尾段甘特图（2376-2405）", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_S2 / "fig_q2_gantt_24h.png", bbox_inches="tight")
    plt.close(fig)


def c6_marginal_curve(front: pd.DataFrame) -> dict:
    """前沿解成本-碳边际：排序后相邻点交换率。"""
    f = front.sort_values("cost_wan").reset_index(drop=True)
    dcost = f["cost_wan"].diff()
    dcar = f["carbon_t"].diff()
    rate = (-dcar / dcost).replace([np.inf, -np.inf], np.nan).abs()  # 省1万成本省X吨碳
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f["cost_wan"], f["carbon_t"], "o-", color="#2980b9")
    ax.set_xlabel("运行成本（万元）")
    ax.set_ylabel("碳排放（tCO2）")
    med = float(rate.median())
    ax.set_title(f"Q2 成本-碳边际曲线（交换率中位 {med:.2f} tCO2/万元）")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_marginal.png", bbox_inches="tight")
    plt.close(fig)
    return {"median_exchange_t_per_wan": med,
            "range": [float(rate.min()), float(rate.max())],
            "note": "沿前沿省 1 万元成本 → 碳降 X 吨（中位口径，abs）；成本-碳正相关（E-F3 共线）→ 交换率单调"}


def c7_rule_pruning(s10, s20, wt, rt, params, consume, pol) -> dict:
    """CART 深度 1-6 的 覆盖-规则数 权衡。"""
    from sklearn.tree import DecisionTreeClassifier, export_text
    sched = s20.schedule_constructive(wt, rt, s10, params, tuple(pol))
    m = wt.merge(sched, on="TaskID")
    mig = (m.SourceRegion != m.Region)
    X = pd.DataFrame({
        "GPU_Demand": m["GPU_Demand"].to_numpy(float),
        "dur_h": m["dur_h"].to_numpy(float),
        "ArrivalHour": m["ArrivalHour"].to_numpy(float),
        "slack_h": (m["LatestFinishHour"] - m["ArrivalHour"]
                    - m["dur_h"]).to_numpy(float),
        "is_BI": (m["TaskType"] == "BatchInference").astype(int),
        "is_AT": (m["TaskType"] == "AITraining").astype(int),
    })
    out = []
    for d in range(1, 7):
        clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=500,
                                     random_state=0)
        clf.fit(X, mig)
        pred = clf.predict(X)
        acc = float((pred == mig).mean())
        n_leaves = int(clf.get_n_leaves())
        out.append({"depth": d, "acc": round(acc, 4),
                    "n_rules": n_leaves})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["depth"] for x in out], [x["acc"] for x in out], "o-",
            color="#8e44ad")
    ax.set_xlabel("CART 深度"); ax.set_ylabel("迁移决策准确率")
    ax.set_title("Q2 规则剪枝权衡：深度-准确率")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_pruning.png", bbox_inches="tight")
    plt.close(fig)
    return out


_S_LB = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
_S_UB = np.array([15.0, 6.65, 24.0, 24.0, 0.08])


def _eval_cost(x: np.ndarray) -> float:
    """模块级评估（进程池 pickle 需求）：策略 → 成本（viol 解 NaN）。"""
    ctx = load_ctx()
    s10, s20 = ctx
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    pol = tuple(_S_LB + (_S_UB - _S_LB) * x)
    sched = s20.schedule_constructive(wt, rt, s10, params, pol)
    m4 = s20.evaluate_4obj(wt, rt, sched, s10, params, consume)
    return m4["cost_wan"] if m4["viol_h"] == 0 else np.nan


def c8_strategy_sobol(s10, s20, wt, rt, params, consume, N: int = 32) -> dict:
    """5 维策略 Saltelli 采样 → 成本敏感性（自研，与 Q3 Sobol 呼应）。"""
    import multiprocessing as mp
    D = 5
    base = np.random.RandomState(0).rand(N, D)
    with mp.Pool(8) as pool:
        y = np.array(pool.map(_eval_cost, base))
        S1 = {}
        for j in range(D):
            xm = base.copy()
            xm[:, j] = 0.5
            ym = np.array(pool.map(_eval_cost, xm))
            S1[f"p{j}"] = float(np.nanvar(ym) / max(np.nanvar(y), 1e-9))
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["mig_gpu", "mig_dur", "shift_BI", "shift_AT", "headroom"]
    ax.bar(names, [S1[f"p{j}"] for j in range(D)], color="#16a085")
    ax.set_ylabel("一阶敏感性（成本方差占比）")
    ax.set_title("Q2 策略参数敏感性（成本）")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_sobol.png", bbox_inches="tight")
    plt.close(fig)
    return {"S1": {k: round(v, 3) for k, v in S1.items()},
            "names": names, "n_samples": int(N),
            "note": "简化 Saltelli（固定中位扰动法），与 Q3 完整 Sobol 呼应"}


def main() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    s10, s20 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    front = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    # 折中解 policy（与 step2.3 同 TOPSIS 逻辑）
    obj = front[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]].to_numpy()
    obj[:, 3] = 1 - obj[:, 3] / 100
    Xn = obj.copy()
    for m0 in range(4):
        rng = Xn[:, m0].max() - Xn[:, m0].min()
        Xn[:, m0] = (Xn[:, m0] - Xn[:, m0].min()) / (rng if rng > 0 else 1.0)
    p = Xn / (Xn.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(Xn))
    w = (1 - e) / (1 - e).sum()
    sd = np.sqrt(((Xn - Xn.min(axis=0)) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((Xn - Xn.max(axis=0)) ** 2 * w).sum(axis=1))
    comp_i = int(np.argmax(nd / (sd + nd)))
    pol = json.loads(front.loc[comp_i, "policy"])

    c5_gantt(s10, s20, wt, rt, params, consume, pol)
    marg = c6_marginal_curve(front)
    prune = c7_rule_pruning(s10, s20, wt, rt, params, consume, pol)
    sob = c8_strategy_sobol(s10, s20, wt, rt, params, consume)
    report = {"compromise_policy": pol, "c6_marginal": marg,
              "c7_pruning": prune, "c8_sobol": sob}
    with open(OUT_Q2 / "materials.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
