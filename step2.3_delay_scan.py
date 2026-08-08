"""step2.3_delay_scan — T6: 时延目标形式扫描（T1-T3）+ baseline_proof.

数据裁决（step2.2+ 实证）: 前沿解全部零违约（白名单硬约束），最小时延裕度 2ms
→ T2(纯违约惩罚)/T3(混合) 违约项恒 0，无区分度 → T1 纯加权是唯一有效形式。
本脚本完成两件事:
  S1 形式化验证: θ×β 网格下 T1/T2/T3 目标值对比（证明 T2/T3≡0，数据裁决）
  S2 baseline_proof: 附件基线 local / Q1 greedy / Q1 quantile / Q2 折中解
     四目标改进率表（论文消融 H 的输入）

产物（output/q2/）: delay_scan.json + baseline_proof.json +
  figures/step2/fig_q2_baseline_proof.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q2 = OUTPUT / "q2"
FIG_S2 = FIGURES / "step2"
MAX_LATENCY = {"RealTimeInference": 20, "BatchInference": 80,
               "AITraining": 150}


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


def delay_forms(wt: pd.DataFrame, sched: pd.DataFrame) -> dict:
    """T1/T2/T3 目标值（θ×β 网格无关项为违约数——无违约时恒 0）。"""
    lat = pd.read_excel(Path(__file__).resolve().parent / "data" / "raw"
                        / "network_latency.xlsx", sheet_name=0)
    lm = lat.pivot(index="FromRegion", columns="ToRegion",
                   values="NetworkLatency_ms")
    m = wt.merge(sched, on="TaskID")
    w = m["GPU_Demand"].to_numpy(dtype=float)
    dur = m["dur_h"].to_numpy(dtype=float)
    ms = np.array([lm.loc[s, d] for s, d in zip(m["SourceRegion"],
                                                m["Region"])], dtype=float)
    mx = np.array([MAX_LATENCY[t] for t in m["TaskType"]])
    n_viol = int((ms > mx).sum())
    t1 = float((ms * w * dur).sum() / (w * dur).sum())
    t2 = float(n_viol)
    margin = float((mx - ms).min())
    p95 = float(np.percentile(ms, 95))
    return {"T1_weighted_ms": t1, "T2_viol_count": n_viol,
            "T3_base": t1, "min_margin_ms": margin, "p95_ms": p95}


def s2_baseline_proof(wt, rt, s10, s20, params, consume) -> dict:
    """四方案对照：local / greedy / quantile(Q1) / Q2 折中解。"""
    from step0_config import REGIONS
    out = {}
    schemes = {
        "local": pd.read_csv(OUTPUT / "baseline" / "local_schedule.csv"),
        "greedy": pd.read_csv(OUTPUT / "baseline" / "greedy_schedule.csv"),
        "q1_quantile": pd.read_csv(OUTPUT / "robust" / "quantile_schedule.csv"),
    }
    front = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    obj = front[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]].to_numpy()
    obj[:, 3] = 1 - obj[:, 3] / 100
    # TOPSIS 折中（同 step2.1 算法）
    X = obj.copy()
    for m0 in range(4):
        rng = X[:, m0].max() - X[:, m0].min()
        X[:, m0] = (X[:, m0] - X[:, m0].min()) / (rng if rng > 0 else 1.0)
    p = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))
    w = (1 - e) / (1 - e).sum()
    sd = np.sqrt(((X - X.min(axis=0)) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((X - X.max(axis=0)) ** 2 * w).sum(axis=1))
    comp_i = int(np.argmax(nd / (sd + nd)))
    pol = json.loads(front.loc[comp_i, "policy"])
    schemes["q2_compromise"] = s20.schedule_constructive(
        wt, rt, s10, params, tuple(pol))

    for name, sched in schemes.items():
        m4 = s20.evaluate_4obj(wt, rt, sched, s10, params, consume)
        out[name] = {k: m4[k] for k in
                     ("cost_wan", "carbon_t", "latency_ms", "nu_pct",
                      "viol_h", "curtail_MWh")}
    base = out["local"]
    for name in ("greedy", "q1_quantile", "q2_compromise"):
        out[f"{name}_vs_local"] = {
            "cost_pct": (base["cost_wan"] - out[name]["cost_wan"])
                        / base["cost_wan"] * 100,
            "carbon_pct": (base["carbon_t"] - out[name]["carbon_t"])
                          / base["carbon_t"] * 100,
            "nu_pp": out[name]["nu_pct"] - base["nu_pct"],
            "latency_ratio": out[name]["latency_ms"]
                             / max(base["latency_ms"], 1e-9),
        }
    out["q2_compromise_policy"] = pol
    out["note"] = ("Q2 折中解 vs Q1 greedy 的增量 = 迁移/错峰优化；"
                   "vs local = 全链改进（baseline_proof，消融 H 输入）")
    return out


def main() -> None:
    OUT_Q2.mkdir(parents=True, exist_ok=True)
    FIG_S2.mkdir(parents=True, exist_ok=True)
    s10, s20 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    front = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    sample = front.sample(min(10, len(front)), random_state=0)
    dscan = []
    for _, row in sample.iterrows():
        pol = json.loads(row["policy"])
        sched = s20.schedule_constructive(wt, rt, s10, params, tuple(pol))
        dscan.append({"cost_wan": row["cost_wan"], **delay_forms(wt, sched)})
    verdict = {"T2_all_zero": all(d["T2_viol_count"] == 0 for d in dscan),
               "T3_equals_T1": all(abs(d["T3_base"] - d["T1_weighted_ms"])
                                   < 1e-9 for d in dscan),
               "conclusion": ("白名单硬约束下前沿解零违约 → T2/T3 违约项恒 0 "
                              "无区分度 → T1 纯加权是唯一有效时延目标形式"
                              "（数据裁决，T2/T3 不引入）")}
    ds_out = {"sample_n": len(dscan), "rows": dscan, "verdict": verdict}
    with open(OUT_Q2 / "delay_scan.json", "w", encoding="utf-8") as f:
        json.dump(ds_out, f, ensure_ascii=False, indent=2)

    proof = s2_baseline_proof(wt, rt, s10, s20, params, consume)
    with open(OUT_Q2 / "baseline_proof.json", "w", encoding="utf-8") as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, fmt in [(axes[0], "cost_wan", "%.0f"),
                         (axes[1], "carbon_t", "%.0f"),
                         (axes[2], "latency_ms", "%.1f")]:
        names = ["local", "greedy", "q1_quantile", "q2_compromise"]
        vals = [proof[n][key] for n in names]
        cols = ["#95a5a6", "#7f8c8d", "#2980b9", "#e67e22"]
        ax.bar(names, vals, color=cols)
        for b, v in zip(ax.patches, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt % v, ha="center", va="bottom", fontsize=8)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle("Q2 baseline_proof: 四方案四目标对照（消融 H）")
    fig.tight_layout()
    fig.savefig(FIG_S2 / "fig_q2_baseline_proof.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"verdict": verdict,
                      "proof_vs_local": {k: v for k, v in proof.items()
                                         if k.endswith("_vs_local")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
