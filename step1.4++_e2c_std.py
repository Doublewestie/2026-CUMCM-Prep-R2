"""E2c'' — 跨序列融合标准化修正（per-sequence z-score，修量纲迁移缺陷）.

训练: 每序列 OOF 特征/目标 z-score（用该序列训练段统计量）→ 元模型全局学习
评估: GroupKFold(5) by series，反标准化后 MAPE
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

OUT_R = Path("output") / "robust"
SEG_TRAIN = 2352

s11s = importlib.util.spec_from_file_location(
    "s11", "step1.1_forecast_arena.py")
s11 = importlib.util.module_from_spec(s11s)
s11s.loader.exec_module(s11)
s14s = importlib.util.spec_from_file_location(
    "s14", "step1.4_fusion_research.py")
s14 = importlib.util.module_from_spec(s14s)
s14s.loader.exec_module(s14)

structure = [n for n, i in s11.make_series_dict().items()
             if i["layer"] == "energy" and "renewable" not in n]


def mape(y, p):
    return float(np.mean(np.abs(y - p) / (np.abs(y) + 1e-9)) * 100)


def main() -> None:
    Xm, ym, gm, stat = [], [], [], {}
    for name in structure:
        info = s11.make_series_dict()[name]
        y = info["y"]
        X = s11.build_features(y, "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in
                  ("lgbm_point", "xgboost_point", "qrf_quantile")]
        oofdf, yv = s14.get_oof(s11, y, X, passed)
        names = list(oofdf.columns)
        ytr = y[:SEG_TRAIN]
        mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-9
        stat[name] = (mu, sd)
        for k, (i, row) in enumerate(oofdf.iterrows()):
            Xm.append([(row[n] - mu) / sd for n in names])
            ym.append((yv[k] - mu) / sd)
            gm.append(name)
    Xm, ym, gm = map(np.array, (Xm, ym, gm))
    out = {}
    for tag, model in (("ridge", Ridge(alpha=1.0)),
                       ("rf", RandomForestRegressor(
                           n_estimators=200, max_depth=8, min_samples_leaf=10,
                           random_state=42, n_jobs=-1))):
        scores = []
        for tr, va in GroupKFold(n_splits=5).split(Xm, ym, groups=gm):
            model.fit(Xm[tr], ym[tr])
            pred = model.predict(Xm[va])
            yt = np.array([ym[va[k]] * stat[gm[va[k]]][1]
                           + stat[gm[va[k]]][0] for k in range(len(va))])
            pp = np.array([pred[k] * stat[gm[va[k]]][1]
                           + stat[gm[va[k]]][0] for k in range(len(va))])
            scores.append(mape(yt, pp))
        out[tag] = float(np.mean(scores))
    result = {"n_rows": len(ym), "n_series": len(structure),
              "results": out,
              "note": "E2c'' 标准化修正：per-sequence z-score 特征与目标，"
                      "修跨序列量纲迁移缺陷"}
    path = OUT_R / "fusion_v2.json"
    if path.exists():
        merged = json.loads(path.read_text(encoding="utf-8"))
    else:
        merged = {}
    merged["e2c_std"] = result
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
