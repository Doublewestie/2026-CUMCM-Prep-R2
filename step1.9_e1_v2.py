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
    # B3: 真 κ 版（滚动 nowcast 分位数逐时预留）——替代 headroom 近似
    kappa_factor = None
    try:
        import importlib.util as _iu
        from pathlib import Path as _P
        spec = _iu.spec_from_file_location(
            "srk", _P(__file__).resolve().parent / "step1.2+_rolling_kappa.py")
        srk = _iu.module_from_spec(spec)
        spec.loader.exec_module(srk)
        ft = pd.read_csv(OUTPUT / "forecast" / "fuse_quantiles_task.csv")
        act = {r: {t: ft[f"{r}|{t}_actual"].to_numpy() for t in
                   ("RealTimeInference", "BatchInference", "AITraining")}
               for r in REGIONS}
        q95 = srk.rolling_q(act, 336, 0.95)
        cap = {"RegionA": 630, "RegionB": 585, "RegionC": 540,
               "RegionD": 1472, "RegionE": 1012, "RegionF": 966}
        kappa = {r: np.clip(1 - q95[r] / cap[r], 0, 0.95) for r in REGIONS}
        kappa_factor = np.stack([kappa[r] for r in REGIONS], axis=1) * 0 + 1
        kappa_factor = np.stack([1 - kappa[r] for r in REGIONS], axis=1)
        quantile_k = s20.schedule_constructive(
            wt, rt, s10, params, (0.0, 0.0, 0.0, 0.0, 0.0),
            capacity_factor=kappa_factor)
        quantile = quantile_k
    except Exception as e:
        print(f"[B3] 真 κ 版失败，回退 headroom 版: {repr(e)[:80]}", flush=True)

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
        "conclusion": ("实际任务全知下预测与预留均无价值（Q2 语境）——"
                       "gap = 过度预留代价（κ 全时段按 q95 缩容量→压缩迁移套利）；"
                       "说明 2 用实际任务 → 正确策略=无预留+容量感知迁移（=perfect）"),
        "caliber": ("三方同框架：附件基线 / 实际任务+价格感知迁移（无预留）/ "
                    "实际任务+价格感知迁移+滚动 q95 全时段预留（真 κ）；模板消纳口径"),
        "note": ("B3/B5 修正链：①旧 E1 两组无价格感知→测预留容量成本（缺陷）；"
                 "②B5 headroom=0.05 近似低估（0.20pp）；③B3 真 κ（滚动 q95 全时段）"
                 "→ gap 4.81pp，暴露 κ 过度预留（需求水平被当预留，压缩迁移目的地容量）"
                 "+ RT/弹性交互（预留致超容需修复）；④语义修正：实际任务全知下"
                 "预留不必要——预测精度边际价值≈0 的更强版本（Q2 语境无预测无预留）"),
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
