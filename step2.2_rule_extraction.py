"""step2.2_rule_extraction — Q2 规则层：从帕累托前沿提取可解释调度策略.

方法（PLAN_details §7.7）:
  1. 读 nsga2_front.csv（含 policy）→ 选代表性解（TOPSIS 折中 + 成本/时延端点）
  2. 每代表解 → schedule_constructive → 任务级迁移决策
  3. CART（sklearn.tree，max_depth=3-4）: X = 任务属性（规格/时长/类型/源区/
     时延裕度/到达小时），y = 是否迁移
  4. 决策树路径 → 人读运营规则表（前件 → 决策，含覆盖数与纯度）

产出（output/q2/）: rules.json（每代表解规则表 + 迁移统计）+
  figures/step2/fig_q2_rules.png（规则覆盖与目标对比）
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q2 = OUTPUT / "q2"
FIG_S2 = FIGURES / "step2"


def load_ctx():
    spec = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s10)
    spec2 = importlib.util.spec_from_file_location(
        "s20", Path(__file__).resolve().parent / "step2.0_construct.py")
    s20 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s20)
    return s10, s20


def topsis_index(front: np.ndarray) -> int:
    """熵权 TOPSIS（与 step2.1 同算法，保证折中解一致）。"""
    X = front.copy()
    for m in range(X.shape[1]):
        rng = X[:, m].max() - X[:, m].min()
        X[:, m] = (X[:, m] - X[:, m].min()) / (rng if rng > 0 else 1.0)
    p = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))
    w = (1 - e) / (1 - e).sum()
    ideal, nadir = X.min(axis=0), X.max(axis=0)
    sd = np.sqrt(((X - ideal) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((X - nadir) ** 2 * w).sum(axis=1))
    return int(np.argmax(nd / (sd + nd)))


def extract_rules(tree: DecisionTreeClassifier, X: pd.DataFrame,
                  feat_names: list[str]) -> list[dict]:
    """决策树 DFS → 根到叶路径规则表（前件特征/阈值 + 覆盖 + 纯度）。"""
    kids = tree.tree_.children_left
    rights = tree.tree_.children_right
    feat = tree.tree_.feature
    thr = tree.tree_.threshold
    classes = list(tree.classes_)
    rules = []
    stack = [(0, [])]
    while stack:
        node, conds = stack.pop()
        if kids[node] == rights[node]:
            n_samples = int(tree.tree_.n_node_samples[node])
            v = tree.tree_.value[node][0]
            pred = classes[int(np.argmax(v))]
            purity = float(v.max() / v.sum())
            rules.append({
                "node": int(node), "conditions": conds,
                "pred": str(pred),
                "n_samples": n_samples,
                "purity": round(purity, 3),
                "n_mig": int(v[1]) if len(v) > 1 else int(v[0])})
        else:
            c_feat = feat_names[feat[node]]
            c_thr = round(thr[node], 2)
            stack.append((rights[node],
                          conds + [f"{c_feat} >= {c_thr}"]))
            stack.append((kids[node],
                          conds + [f"{c_feat} < {c_thr}"]))
    rules.sort(key=lambda r: -r["n_samples"])
    return rules


def run() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    s10, s20 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    front = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    obj = front[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]].to_numpy()
    obj[:, 3] = 1 - obj[:, 3] / 100
    comp_i = topsis_index(obj)
    # 代表性解：TOPSIS 折中 + 成本最优端点 + 时延最优端点
    reps = {
        "compromise": comp_i,
        "min_cost": int(np.argmin(front["cost_wan"].to_numpy())),
        "min_latency": int(np.argmin(front["latency_ms"].to_numpy())),
    }
    labels = {"compromise": "TOPSIS 折中解", "min_cost": "成本最优端点",
              "min_latency": "时延最优端点"}

    feat_cols = ["GPU_Demand", "dur_h", "ArrivalHour", "slack_h"]
    for tt in TASK_TYPES:
        feat_cols.append(f"type_{tt}")
    for r in REGIONS:
        feat_cols.append(f"src_{r}")

    out = {"representatives": {}, "caliber": "CART max_depth=4; "
           "y=是否迁移（代表性解调度）；X=任务属性"}
    rule_stats = []
    for key, idx in reps.items():
        pol = json.loads(front.loc[idx, "policy"])
        sched = s20.schedule_constructive(wt, rt, s10, params, tuple(pol))
        m4 = s20.evaluate_4obj(wt, rt, sched, s10, params, consume)
        m = wt.merge(sched, on="TaskID")
        src = m["SourceRegion"].to_numpy()
        dst = m["Region"].to_numpy()
        mig = (src != dst)
        X = pd.DataFrame({
            "GPU_Demand": m["GPU_Demand"].to_numpy(float),
            "dur_h": m["dur_h"].to_numpy(float),
            "ArrivalHour": m["ArrivalHour"].to_numpy(float),
            "slack_h": (m["LatestFinishHour"] - m["ArrivalHour"]
                        - m["dur_h"]).to_numpy(float),
        })
        for tt in TASK_TYPES:
            X[f"type_{tt}"] = (m["TaskType"] == tt).astype(int)
        for r in REGIONS:
            X[f"src_{r}"] = (m["SourceRegion"] == r).astype(int)
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=500,
                                     random_state=0)
        clf.fit(X, mig)
        rules = extract_rules(clf, X, list(X.columns))
        n_leaf_rules = [r for r in rules if r["pred"] is not None
                        and r["n_samples"] >= 500]
        out["representatives"][key] = {
            "label": labels[key], "policy": pol,
            "metrics": {k: m4[k] for k in
                        ("cost_wan", "carbon_t", "latency_ms", "nu_pct")},
            "mig_rate": float(mig.mean()),
            "n_rules": len(n_leaf_rules[:8]),
            "rules": n_leaf_rules[:8],
            "tree_text": export_text(clf, feature_names=list(X.columns),
                                     max_depth=4)[:2000],
        }
        rule_stats.append({"rep": key, "mig_rate": float(mig.mean()),
                           "latency": m4["latency_ms"]})
        print(f"[{key}] {labels[key]}: mig={mig.mean():.3f} "
              f"cost={m4['cost_wan']:.0f} lat={m4['latency_ms']:.2f}")

    with open(OUT_Q2 / "rules.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(len(rule_stats)))
    ax.bar(xs, [r["mig_rate"] for r in rule_stats], color="#8e44ad")
    for i, r in enumerate(rule_stats):
        ax.text(i, r["mig_rate"] + 0.01, f"{r['mig_rate']:.1%}",
                ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([out["representatives"][r["rep"]]["label"]
                        for r in rule_stats])
    ax.set_ylabel("迁移率")
    ax.set_title("Q2 代表性解的迁移规模（规则提取对象）")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_rules.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"representatives": {k: v["label"] for k, v in
                                          out["representatives"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    run()
