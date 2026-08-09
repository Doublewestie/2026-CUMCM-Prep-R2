"""step0.7_nonai_layered — Phase 5-L1: NonAI 恒等式分层实验.

命题（I1 实证支撑）: IT_Load = NonAI + AI（六区逐点精确）且 IT_Load 是稳定日模板
（冻结段漂移<1%）→ NonAI 的"难"是 AI 镜像污染；分层预测:
  S0 现状: NonAI 直接日模板（stat_hist）
  S1 分层-实际口径: IT_Load 模板 - AI 实际（说明2: 2376-2399 调度输入=实际任务）
  S2 分层-均值口径: IT_Load 模板 - AI 逐时段均值模板（纯预测口径，AI 段间不确定不可知）
  S3 现状树类: lgbm（竞技榜冻结段 E/F 13-16% 参照）
协议: 5 折 TimeSeriesSplit CV（0-2351，模板用折内 tr 段拟合）+ 冻结段 2376-2399 验证
裁决: 冻结段 MAPE（主），CV MAPE（稳定性）
产物: output/robust/nonai_layered.json
"""
import json
import numpy as np
import pandas as pd

from step0_config import CLEAN, REGIONS, OUTPUT

OUT = OUTPUT / "robust" / "nonai_layered.json"
SEG_TRAIN, SEG_CAL, SEG_FZ = 2352, 2376, 2400


def series(r: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    sub = rt[rt.Region == r].sort_values("Hour")
    na = sub["NonAI_IT_Load_MW"].to_numpy(float)
    ai = sub["Baseline_AI_IT_Load_MW"].to_numpy(float)
    it = sub["IT_Load_MW"].to_numpy(float)
    return na, ai, it


def hod_template(y: np.ndarray, tr: np.ndarray) -> np.ndarray:
    t = pd.Series(y[tr])
    return t.groupby(tr % 24).mean().reindex(
        np.arange(len(y)) % 24).to_numpy()


def eval_mape(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred) / (np.abs(y) + 1e-9)) * 100)


def run() -> dict:
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    out = {}
    for r in REGIONS:
        na, ai, it = series(r)
        cv = {"s0_direct": [], "s1_layered_actual": [], "s2_layered_mean": [],
              "s3_tree": []}
        ai_tpl = hod_template(ai, np.arange(SEG_TRAIN))
        for tr, va in tscv.split(np.arange(SEG_TRAIN)):
            t0 = int(tr[-1]) + 1
            it_tpl = hod_template(it, tr)
            # S0: NonAI 直接模板
            na_tpl = hod_template(na, tr)
            cv["s0_direct"].append(eval_mape(na[va], na_tpl[va]))
            # S1: IT_Load 模板 - AI 实际（折内 AI 已知）
            cv["s1_layered_actual"].append(eval_mape(na[va], it_tpl[va] - ai[va]))
            # S2: IT_Load 模板 - AI 均值模板（纯预测）
            cv["s2_layered_mean"].append(eval_mape(na[va], it_tpl[va] - ai_tpl[va]))
        # S3: lgbm 参照（能量层特征，折内 fit）
        from lightgbm import LGBMRegressor
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
        s11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s11)
        X = s11.build_features(na, "energy", r)
        for tr, va in tscv.split(np.arange(SEG_TRAIN)):
            m = LGBMRegressor(**s11.MODEL_PARAMS["lgbm"])
            m.fit(X.iloc[tr], na[tr])
            cv["s3_tree"].append(eval_mape(na[va], m.predict(X.iloc[va])))
        # 冻结段（全段 fit 0-2351）
        tr_all = np.arange(SEG_TRAIN)
        it_tpl_f = hod_template(it, tr_all)
        na_tpl_f = hod_template(na, tr_all)
        m = LGBMRegressor(**s11.MODEL_PARAMS["lgbm"])
        m.fit(X.iloc[:SEG_TRAIN], na[:SEG_TRAIN])
        out[r] = {
            "cv": {k: {"mean": round(float(np.mean(v)), 3),
                       "std": round(float(np.std(v)), 3)} for k, v in cv.items()},
            "frozen_mape": {
                "s0_direct": round(eval_mape(na[SEG_CAL:SEG_FZ], na_tpl_f[SEG_CAL:SEG_FZ]), 3),
                "s1_layered_actual": round(
                    eval_mape(na[SEG_CAL:SEG_FZ], it_tpl_f[SEG_CAL:SEG_FZ] - ai[SEG_CAL:SEG_FZ]), 3),
                "s2_layered_mean": round(
                    eval_mape(na[SEG_CAL:SEG_FZ], it_tpl_f[SEG_CAL:SEG_FZ] - ai_tpl[SEG_CAL:SEG_FZ]), 3),
                "s3_tree": round(eval_mape(na[SEG_CAL:SEG_FZ], m.predict(X.iloc[SEG_CAL:SEG_FZ])), 3)},
        }
        o = out[r]
        print(f"{r}: CV s0={o['cv']['s0_direct']['mean']:.2f} "
              f"s1={o['cv']['s1_layered_actual']['mean']:.2f} "
              f"s2={o['cv']['s2_layered_mean']['mean']:.2f} "
              f"s3={o['cv']['s3_tree']['mean']:.2f} | "
              f"FZ s0={o['frozen_mape']['s0_direct']:.2f} "
              f"s1={o['frozen_mape']['s1_layered_actual']:.2f} "
              f"s2={o['frozen_mape']['s2_layered_mean']:.2f} "
              f"s3={o['frozen_mape']['s3_tree']:.2f}", flush=True)
    out["caliber"] = ("S0 NonAI 直接模板（现状）；S1 IT_Load模板-AI实际（说明2 实际任务口径）；"
                      "S2 IT_Load模板-AI均值模板（纯预测口径）；S3 lgbm（竞技榜参照）；"
                      "CV=5折TSCV（折内模板拟合）；冻结段=0-2351全段拟合后评估 2376-2399")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    run()
