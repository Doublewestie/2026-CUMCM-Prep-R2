"""step4.5_q4_rules — Q4 策略规则层：CART 规则提取 + 规则化代价 + 三档策略对比.

方法（Q4 方案定稿 step4.5，Q2/Q3 规则层同法）:
  ① 对 Q4 折中策略调度 → 任务级 (特征, 是否迁移) → CART（max_depth=4,
     min_samples_leaf=500）→ 人读规则表（前件/覆盖/纯度）
  ② 规则模拟（诚实代价）：按 CART 预测决定迁移与否的重调度（rule_schedule，
     容量贪心同构）→ 重算六指标 → 规则化代价 = 与最优解的成本差
     （Q3 教训：逐时/任务级核算，规则化代价 0.78-2.93% 量级）
  ③ 三档策略规则对比：Q4 折中 / Q2 折中（Q4 基线）/ 构造 —— 迁移率 + 规则表
     （"策略变化"呈现：规则差异 + step4.3 最优策略切换结论交叉引用）

产物（output/q4/）:
  q4_rules.json   规则表 + 规则化代价 + 三档对比
figures/step4/fig_q4_rules.png  三档迁移率 + 规则化代价柱状
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

from step0_config import FIGURES, OUTPUT, REGIONS, TASK_TYPES, HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"),
        Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    s41 = _load_mod("step4.1_q4_indicators.py")
    c = s41.load_ctx()
    return {**c, "s41": s41}


def build_features(c, m: pd.DataFrame) -> pd.DataFrame:
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
    return X


def extract_rules(tree, X, feat_names):
    kids, rights = tree.tree_.children_left, tree.tree_.children_right
    feat, thr = tree.tree_.feature, tree.tree_.threshold
    classes = list(tree.classes_)
    rules = []
    stack = [(0, [])]
    while stack:
        node, conds = stack.pop()
        if kids[node] == rights[node]:
            n = int(tree.tree_.n_node_samples[node])
            v = tree.tree_.value[node][0]
            pred = classes[int(np.argmax(v))]
            rules.append({"conditions": conds, "pred": str(pred),
                          "n_samples": n,
                          "purity": round(float(v.max() / v.sum()), 3),
                          "n_mig": int(v[1]) if len(v) > 1 else 0})
        else:
            cn = feat_names[feat[node]]
            ct = round(thr[node], 2)
            stack.append((rights[node], conds + [f"{cn} >= {ct}"]))
            stack.append((kids[node], conds + [f"{cn} < {ct}"]))
    rules.sort(key=lambda r: -r["n_samples"])
    return rules


def rule_schedule(c, clf, policy: list) -> pd.DataFrame:
    """规则模拟调度：迁移与否由 CART 预测决定（复制构造层容量贪心逻辑）。

    与 Q2 schedule_constructive 两段式对齐：第一轮迁移窗口限 [Arrival,
    Arrival+shift_max]（错峰上限，防止弹性任务拖入冻结段挤压 RT 容量——
    冻结段需求抬升 z=2.86 是 RT 超容的诱因）；失败进全窗口修复扫描。
    """
    ranks = c["s20"].region_rank(c["rt"],
                                 {(r.TaskType, r.SourceRegion):
                                  r.Reachable.split("|")
                                  for r in pd.read_csv(
                                      OUTPUT / "clean" / "whitelist.csv"
                                  ).itertuples(index=False)},
                                 c["s10"], c["params"])
    cap_arr = np.array([c["params"]["cap"][r] for r in REGIONS], dtype=float)
    occ = np.zeros((HOURS_TOTAL, len(REGIONS)))
    shift_max = {"RealTimeInference": 0.0,
                 "BatchInference": float(policy[2]),
                 "AITraining": float(policy[3])}
    m = c["wt"].merge(pd.DataFrame({"TaskID": c["wt"]["TaskID"],
                                    "Region": c["wt"]["SourceRegion"],
                                    "StartHour": c["wt"]["ArrivalHour"]}),
                      on="TaskID", how="left")
    X = build_features(c, m)
    mig = clf.predict(X)
    rows = []
    # 与 Q2 同款处理顺序：优先级高者先占位（RT 最先，防弹性任务挤压实时容量）
    wt2 = c["wt"].sort_values(["Priority", "ArrivalHour"],
                              ascending=[False, True]).reset_index(drop=True)
    mig_map = dict(zip(m["TaskID"], mig))
    for rec in wt2.itertuples(index=False):
        g = float(rec.GPU_Demand)
        dur = float(rec.dur_h)
        r_src = REGIONS.index(rec.SourceRegion)
        st0 = int(rec.ArrivalHour)
        h1_0 = min(int(np.ceil(st0 + dur)), HOURS_TOTAL)
        local_ok = (h1_0 <= st0
                    or (occ[st0:h1_0, r_src] + g - cap_arr[r_src]).max() <= 0)
        should_mig = bool(mig_map.get(rec.TaskID)) \
            and rec.TaskType != "RealTimeInference"
        if (not should_mig) and local_ok:
            r, st = r_src, st0
        else:
            # 迁移（CART 决策）或本地放不下降级：shift 窗口 → 全窗口修复 → 本地兜底
            r = st = None
            smax = shift_max[rec.TaskType]
            for cand in ranks[rec.TaskType]:
                ri = REGIONS.index(cand)
                lo = int(rec.ArrivalHour)
                hi = min(int(rec.ArrivalHour + smax),
                         int(rec.LatestFinishHour - dur), 2405)
                if hi < lo:
                    continue
                for s in range(lo, hi + 1):
                    h1 = min(int(np.ceil(s + dur)), HOURS_TOTAL)
                    if h1 <= s:
                        break
                    if (occ[s:h1, ri] + g - cap_arr[ri]).max() <= 0:
                        r, st = ri, s
                        break
                if r is not None:
                    break
            if r is None:
                # 容量修复（Q2 同款）：全区域全窗口延后扫描，viol=0 硬底线
                for ri in range(6):
                    lo = int(rec.ArrivalHour)
                    hi = min(int(rec.LatestFinishHour - dur), 2405)
                    if hi < lo:
                        continue
                    for s in range(lo, hi + 1):
                        h1 = min(int(np.ceil(s + dur)), HOURS_TOTAL)
                        if h1 <= s:
                            break
                        if (occ[s:h1, ri] + g - cap_arr[ri]).max() <= 0:
                            r, st = ri, s
                            break
                    if r is not None:
                        break
            if r is None:
                r, st = r_src, st0   # 兜底本地（RT 语义：到达即开工）
        rows.append((rec.TaskID, REGIONS[r], st))
        h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
        if h1 > st:
            occ[st:h1, r] += g
    return pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])


def rules_for(c, name: str, policy: list) -> dict:
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(policy[:5]))
    ev = c["s41"].evaluate_q4_six(c, sched,
                                  alpha=float(policy[5] if len(policy) > 5 else 0.5))
    m = c["wt"].merge(sched, on="TaskID")
    mig = (m["SourceRegion"] != m["Region"]).to_numpy()
    X = build_features(c, m)
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=500,
                                 random_state=0)
    clf.fit(X, mig)
    rules = extract_rules(clf, X, list(X.columns))
    # 规则模拟代价（可行性双检查：上层容量 viol + 下层 LP 可行性）
    sched_rule = rule_schedule(c, clf, policy)
    ev_rule = c["s41"].evaluate_q4_six(c, sched_rule,
                                       alpha=float(policy[5] if len(policy) > 5 else 0.5))
    if ev_rule.get("viol_h", 0) > 0 or ev_rule.get("cost_wan") is None:
        cost_gap = None
        rule_infeasible = True
    else:
        cost_gap = (ev_rule["cost_wan"] - ev["cost_wan"]) \
            / max(abs(ev["cost_wan"]), 1e-9)
        rule_infeasible = False
    return {"policy": policy, "mig_rate": float(mig.mean()),
            "opt_cost_wan": round(ev["cost_wan"], 2),
            "rule_cost_wan": round(ev_rule["cost_wan"], 2)
            if not rule_infeasible else None,
            "rule_cost_gap_pct": round(float(cost_gap * 100), 3)
            if cost_gap is not None else None,
            "rule_infeasible": rule_infeasible,
            "n_rules": len(rules[:8]), "rules": rules[:8],
            "tree_text": export_text(clf, feature_names=list(X.columns),
                                     max_depth=4)[:1500],
            "latency_ms": round(ev["latency_ms"], 2)}


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    pols = {}
    pols["construct"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    pols["q2_compromise"] = c["s41"].q2_compromise_policy(c) + [0.5]
    front = OUT_Q4 / "q4_front.csv"
    if front.exists():
        df = pd.read_csv(front)
        F = df[["cost_wan", "carbon_t", "latency_ms", "one_minus_qos",
                "one_minus_nu", "peak_net_MW"]].to_numpy()
        i = int(np.argmin(F[:, 0] + F[:, 2] + F[:, 5]))
        pols["q4_compromise"] = json.loads(df.loc[i, "policy"])
    else:
        pols["q4_compromise"] = pols["q2_compromise"]

    out = {name: rules_for(c, name, pol) for name, pol in pols.items()}
    report = {"rules": out,
              "caliber": ("CART max_depth=4, min_samples_leaf=500；"
                          "y=是否迁移；规则化代价=规则模拟调度 vs 最优解成本差；"
                          "策略变化呈现=三档迁移率/规则差异 + step4.3 最优策略切换"),
              "note": ("规则化代价预期 0.8-2.9% 量级（Q3 逐时版口径）；"
                       "若 >5% 定位'策略示意'并诚实声明")}
    with open(OUT_Q4 / "q4_rules.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    names = list(out.keys())
    labels = ["构造", "Q2 折中", "Q4 折中"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, [out[n]["mig_rate"] for n in names],
                color=["#95a5a6", "#e67e22", "#2980b9"])
    for b, v in zip(axes[0].patches, [out[n]["mig_rate"] for n in names]):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{v:.1%}", ha="center", va="bottom")
    axes[0].set_ylabel("迁移率")
    axes[0].set_title("三档策略迁移规模")
    axes[1].bar(labels, [out[n]["rule_cost_gap_pct"] or 0.0 for n in names],
                color="#8e44ad")
    for b, v in zip(axes[1].patches,
                    [out[n]["rule_cost_gap_pct"] for n in names]):
        if v is None:
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                         "不可行", ha="center", va="bottom", fontsize=8)
        else:
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                         f"{v:.2f}%", ha="center", va="bottom")
    axes[1].set_ylabel("规则化代价（成本%）")
    axes[1].set_title("可解释性的代价")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_rules.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("mig_rate", "rule_cost_gap_pct",
                                    "opt_cost_wan", "rules")}
                      for k, v in out.items()},
                     ensure_ascii=False, indent=2, default=float)[:2500])


if __name__ == "__main__":
    main()
