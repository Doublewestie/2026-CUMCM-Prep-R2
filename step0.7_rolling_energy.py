"""step0.7_rolling_energy — Phase 5-L4: 滚动重估推广（价格变点自适应）.

背景（D7 实证）: 价格在 t=1245（第 52 天）六区同步等比缩放 -4.4%（三时段同幅）
→ 全段日模板被两段均值拉偏；滚动 336h 模板（nowcast）应自动跟随水平。
对照（冻结段 2376-2399 MAPE）:
  P0 全段日模板（stat_hist 现状）
  P1 滚动 336h 日模板（近 336h 重估，shift>=1 防泄漏）
  P2 lgbm 残差修正（竞技榜部署最优，参照）
产物: output/robust/rolling_energy_price.json
"""
import json
import numpy as np
import pandas as pd

from step0_config import CLEAN, REGIONS, OUTPUT

OUT = OUTPUT / "robust" / "rolling_energy_price.json"
SEG_TRAIN, SEG_CAL, SEG_FZ = 2352, 2376, 2400


def hod_tpl(y: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """逐时段均值模板。键=绝对小时 hod（seg 可能不从 hod 0 开始，
    用 len(seg)%24 会错位——rolling 窗口 bug 教训）。"""
    t = pd.Series(y[seg])
    return t.groupby(seg % 24).mean().reindex(
        np.arange(len(y)) % 24).to_numpy()


def rolling_tpl(y: np.ndarray, w: int = 336) -> np.ndarray:
    """滚动 336h nowcast 模板：t 时刻用 [t-w, t) 的逐时段均值（shift>=1）。"""
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(w, n):
        seg = np.arange(t - w, t)
        out[t] = hod_tpl(y, seg)[t]
    out[:w] = hod_tpl(y, np.arange(w))[:w]
    return out


def eval_mape(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred) / (np.abs(y) + 1e-9)) * 100)


def run() -> dict:
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    s11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s11)
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour")
        y = sub["ElectricityPrice_CNY_per_MWh"].to_numpy(float)
        p0 = hod_tpl(y, np.arange(SEG_TRAIN))
        p1 = rolling_tpl(y, 336)
        X = s11.build_features(y, "energy", r)
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(**s11.MODEL_PARAMS["lgbm"])
        m.fit(X.iloc[:SEG_TRAIN], y[:SEG_TRAIN])
        p2 = m.predict(X.iloc[SEG_CAL:SEG_FZ])
        seg = np.arange(SEG_CAL, SEG_FZ)
        out[r] = {
            "frozen_mape": {
                "p0_full_tpl": round(eval_mape(y[seg], p0[seg]), 4),
                "p1_rolling336": round(eval_mape(y[seg], p1[seg]), 4),
                "p2_lgbm": round(eval_mape(y[seg], p2), 4)},
            "changepoint": {"t": 1245,
                            "scale": float(np.mean(y[1245:SEG_TRAIN])
                                           / np.mean(y[:1245]))}}
        o = out[r]
        print(f"{r}: p0={o['frozen_mape']['p0_full_tpl']:.3f} "
              f"p1={o['frozen_mape']['p1_rolling336']:.3f} "
              f"p2={o['frozen_mape']['p2_lgbm']:.3f} "
              f"(cp_scale={o['changepoint']['scale']:.4f})", flush=True)
    out["caliber"] = ("冻结段 2376-2399；P0 全段日模板（0-2351 拟合）；"
                      "P1 滚动 336h 模板（nowcast，shift>=1）；P2 lgbm（部署最优参照）；"
                      "变点：t=1245 价格等比缩放（D7 实证）")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    run()
