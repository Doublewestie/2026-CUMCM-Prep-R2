"""step1.2+_rolling_kappa — B10: 滚动分位数接入 κ 闭环（F2 修复实验）.

背景（E-F2）: 静态外推分位数在冻结段覆盖率 0.917（需求抬升欠覆盖）；
滚动 336h nowcast 分位数 0.944（+2.8pp）。F2 问题 = 预测→调度桥
（κ_ε）建立在静态外推上。

本实验（轻量验证，不动主流程）:
  1. 滚动 336h 分位数（shift≥1 防泄漏）→ 区域聚合
  2. 校准段 2352-2375 覆盖率 → ε 选择（≥95% 达标集取最大 ε）
  3. 冻结段 2376-2399 覆盖率 + 预留超容率 vs 静态版本对比
裁决: 滚动 κ 是否使冻结段覆盖率 ≥0.94 且超容率 ≈ε（B10 收益量化）

产物（output/robust/）: rolling_kappa.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step0_config import CLEAN, OUTPUT, REGIONS, TASK_TYPES

OUT_R = OUTPUT / "robust"
SEG_CAL_S, SEG_CAL_E = 2352, 2376
SEG_TE_S, SEG_TE_E = 2376, 2400


def rolling_q(act: dict, k: int = 336, a: float = 0.95) -> dict:
    """每序列最近 k 小时滚动经验分位数（nowcast，shift≥1）。"""
    out = {}
    for r in REGIONS:
        cols = []
        for t in TASK_TYPES:
            y = act[r][t]
            q = np.zeros(len(y))
            for t0 in range(len(y)):
                lo = max(0, t0 - k)
                q[t0] = np.quantile(y[lo:t0], a) if t0 > lo else y[t0]
            cols.append(q)
        out[r] = np.sum(cols, axis=0)
    return out


def main() -> None:
    ft = pd.read_csv(OUTPUT / "forecast" / "fuse_quantiles_task.csv")
    act = {r: {t: ft[f"{r}|{t}_actual"].to_numpy() for t in TASK_TYPES}
           for r in REGIONS}
    stat_agg = {r: np.sum([act[r][t] for t in TASK_TYPES], axis=0)
                for r in REGIONS}
    q_roll = {a: rolling_q(act, 336, a) for a in (0.95, 0.90, 0.97)}
    seg_cal = np.arange(SEG_CAL_S, SEG_CAL_E)
    seg_te = np.arange(SEG_TE_S, SEG_TE_E)
    cap = {"RegionA": 630, "RegionB": 585, "RegionC": 540,
           "RegionD": 1472, "RegionE": 1012, "RegionF": 966}

    def cov(q, seg):
        return float(np.mean(np.concatenate(
            [q[r][seg] >= stat_agg[r][seg] for r in REGIONS])))

    res = {}
    for a in (0.97, 0.95, 0.90):
        c_cal = cov(q_roll[a], seg_cal)
        c_te = cov(q_roll[a], seg_te)
        k = {r: np.clip(1 - q_roll[a][r] / cap[r], 0, 0.95) for r in REGIONS}
        res[f"a={a}"] = {"cal_cov": round(c_cal, 4), "frozen_cov": round(c_te, 4),
                         "kappa_mean": float(np.mean([k[r].mean() for r in REGIONS]))}
    # 静态对照（现有 κ 流程数值）
    res["static_ref"] = {"frozen_cov_q95": 0.917, "frozen_cov_q90": None,
                         "note": "S3 实证：静态 q95 冻结段 0.917"}
    # ε 选择（校准段 ≥0.95 达标集取最大 ε）
    passed = [a for a in (0.97, 0.95, 0.90) if res[f"a={a}"]["cal_cov"] >= 0.95]
    eps_sel = 1 - max(passed) if passed else None
    report = {
        "rolling_calibration": res,
        "eps_selected": eps_sel,
        "verdict": {
            "frozen_cov_improved": res["a=0.95"]["frozen_cov"] > 0.917,
            "meets_target": res["a=0.95"]["frozen_cov"] >= 0.94,
            "note": ("滚动 336h 分位数接入 κ：冻结段覆盖率 0.917→"
                     f"{res['a=0.95']['frozen_cov']:.3f}；"
                     "若仍 <0.95 目标 → 需 ε 更保守或滚动+余量（B10 结论）"),
        },
        "caliber": "nowcast 滚动 336h 经验分位数（shift≥1）；区域聚合 Σ类型；"
                   "校准段 2352-2375 / 冻结段 2376-2399",
    }
    with open(OUT_R / "rolling_kappa.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
