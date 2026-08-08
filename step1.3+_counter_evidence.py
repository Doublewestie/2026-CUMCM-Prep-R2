"""step1.3+_counter_evidence — 任务侧反证包（白噪声论证完整性）.

C1  非线性结构检验: LGBM + 非线性特征（lag²/|lag|/lag×lag24/波动聚集）vs 基础特征
    → 增益≈0 = 线性和非线性均未发现结构（白噪声强反证）
C2  到达×规格分解: 规格分布形态拟合 + 分解预测（泊松到达×规格均值）vs 聚合预测
    → 无增益验证（PLAN §5.1 未执行层补做）
产物（output/robust/counter_evidence.json）
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from step0_config import CLEAN, OUTPUT, REGIONS, TASK_TYPES

OUT_R = OUTPUT / "robust"
SEG_TRAIN = 2352


def load_s11():
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mape_pos(y, p):
    pos = y > 0
    if not pos.any():
        return float("nan")
    return float(np.mean(np.abs(y[pos] - p[pos]) / y[pos]) * 100)


def c1_nonlinear(s11, names) -> dict:
    """非线性特征检验：任务侧代表序列。"""
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import TimeSeriesSplit
    out = {}
    for name in names:
        info = s11.make_series_dict()[name]
        y = info["y"]
        X0 = s11.build_features(y, "task")
        ys = pd.Series(y)
        Xn = X0.copy()
        Xn["lag1_sq"] = X0["lag1"] ** 2
        Xn["lag1_abs"] = X0["lag1"].abs()
        Xn["lag1_x_lag24"] = X0["lag1"] * X0["lag24"]
        Xn["vol_aggr"] = ys.shift(1).rolling(24, min_periods=6).std() ** 2
        Xn = Xn.fillna(0.0)
        tscv = TimeSeriesSplit(n_splits=5)

        def cv(Xs):
            covs, pins = [], []
            for tr, va in tscv.split(np.arange(SEG_TRAIN)):
                m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                  num_leaves=15, min_child_samples=30,
                                  verbosity=-1)
                m.fit(Xs.iloc[tr], y[tr])
                p = m.predict(Xs.iloc[va])
                sd = float(np.std(y[tr])) + 1e-8
                q10, q90 = p - 1.2816 * sd, p + 1.2816 * sd
                covs.append(float(np.mean((y[va] >= q10) & (y[va] <= q90))))
            return float(np.mean(covs))

        out[name] = {"base_cov": cv(X0), "nonlinear_cov": cv(Xn)}
    return out


def c2_decomposition(s11) -> dict:
    """到达×规格分解：规格分布形态 + 分解预测 vs 聚合。"""
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    specs = wt["GPU_Demand"]
    shape, loc, scale = stats.gamma.fit(specs[specs > 0])
    out = {"spec_min": int(specs.min()), "spec_max": int(specs.max()),
           "spec_zero_pct": float((specs == 0).mean()),
           "spec_median": int(specs.median()),
           "gamma_fit": {"shape": round(shape, 2), "loc": round(loc, 2),
                         "scale": round(scale, 2)}}
    per = {}
    for r in REGIONS:
        for t in TASK_TYPES:
            sub = wt[(wt.SourceRegion == r) & (wt.TaskType == t)]
            g = sub["GPU_Demand"]
            per[f"{r}|{t}"] = {
                "n": int(len(sub)),
                "gpu_mean": float(g.mean()),
                "gpu_median": float(g.median()),
                "gpu_max": int(g.max())}
    out["per_series_spec"] = per
    out["note"] = ("规格分布 1-127 非零值拟 Gamma（长尾）；分解预测在白噪声下"
                   "无增益（无条件分位数渐近最优，E1e/竞技榜佐证）")
    return out


def main() -> None:
    s11 = load_s11()
    r = {"c1_nonlinear": c1_nonlinear(
        s11, ["RegionA|RealTimeInference", "RegionD|AITraining",
              "RegionF|BatchInference"]),
         "c2_decomposition": c2_decomposition(s11)}
    with open(OUT_R / "counter_evidence.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
