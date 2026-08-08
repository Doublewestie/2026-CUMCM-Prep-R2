"""step1.9_e1_v2 — E1 对照重设计（B5，修复对照纯度缺陷）.

缺陷（A6 审查）: 旧 E1 的 perfect 组 = 无预留贪心（无价格感知）、quantile 组 =
预留贪心（同样无价格感知）→ 两者差异 = 预留容量成本（0.08pp），未测预测价值。

重设计（统一价格感知框架，对照纯度纪律）:
  三方 = 附件基线 local / 完美预测（实际任务+价格感知迁移，无预留）/
         分布调度（实际任务+价格感知迁移+5% 容量预留）
  三者差异分解:
    local vs perfect  = 确定性空间套利收益（迁移）
    perfect vs quantile = 预留松弛成本 = 预测不确定性代价（干净测量）
  gap = |imp_perfect − imp_quantile|（相对 local 成本改进差），判据 <5pp

产物（output/robust/）: e1_v2.json + figures/step1/figE1_v2.png
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

OUT_R = OUTPUT / "robust"
FIG_S1 = FIGURES / "step1"


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


def main() -> None:
    OUT_R.mkdir(parents=True, exist_ok=True)
    FIG_S1.mkdir(parents=True, exist_ok=True)
    s10, s20 = load_ctx()
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]

    ev = lambda s: s20.evaluate_4obj(wt, rt, s, s10, params, consume)
    local_sched = pd.read_csv(OUTPUT / "baseline" / "local_schedule.csv")
    perfect = s20.schedule_constructive(wt, rt, s10, params,
                                        (0.0, 0.0, 0.0, 0.0, 0.0))
    quantile = s20.schedule_constructive(wt, rt, s10, params,
                                         (0.0, 0.0, 0.0, 0.0, 0.05))

    base = ev(local_sched)
    p = ev(perfect)
    q = ev(quantile)
    imp_p = (base["cost_wan"] - p["cost_wan"]) / base["cost_wan"] * 100
    imp_q = (base["cost_wan"] - q["cost_wan"]) / base["cost_wan"] * 100
    gap = abs(imp_p - imp_q)

    out = {
        "baseline_local": {k: base[k] for k in
                           ("cost_wan", "carbon_t", "viol_h")},
        "perfect": {k: p[k] for k in ("cost_wan", "carbon_t", "viol_h")},
        "quantile": {k: q[k] for k in ("cost_wan", "carbon_t", "viol_h")},
        "improve_perfect_cost_pct": imp_p,
        "improve_quantile_cost_pct": imp_q,
        "gap_pp": gap,
        "decomposition": {
            "spatial_arbitrage_pp": imp_p,          # local -> perfect
            "reserve_cost_pp": imp_p - imp_q,        # perfect -> quantile
        },
        "criterion": "gap < 5pp（统一价格感知框架内，预留=预测不确定性代价）",
        "conclusion": ("预测精度边际价值≈0（预留成本极小）" if gap < 5
                       else "预测精度有显著价值（判据不成立）"),
        "caliber": ("三方同框架：附件基线 / 实际任务+价格感知迁移（无预留）/ "
                    "实际任务+价格感知迁移+5%容量预留；模板消纳口径"),
        "note": ("B5 修复对照纯度：旧 E1 两组均无价格感知，测的是预留容量成本；"
                 "新版三方共享迁移套利框架，差异纯在预留（=预测不确定性的真实代价）"),
    }
    with open(OUT_R / "e1_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["附件基线", "完美预测\n(实际任务+迁移)", "分布调度\n(迁移+5%预留)"]
    vals = [base["cost_wan"], p["cost_wan"], q["cost_wan"]]
    bars = ax.bar(labels, vals, color=["#95a5a6", "#16a085", "#e67e22"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}",
                ha="center", va="bottom")
    ax.set_ylabel("成本(万元)")
    ax.set_title(f"E1 v2 对照（干净框架）: 套利 {imp_p:.2f}pp / "
                 f"预留代价 {imp_p - imp_q:.2f}pp / gap {gap:.2f}pp")
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figE1_v2.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
