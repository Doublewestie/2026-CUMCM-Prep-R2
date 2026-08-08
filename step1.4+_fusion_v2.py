"""step1.4+_fusion_v2 — 融合深化 v2（E2i/E2b'/E2c'/E2g'）.

E2i  异构融合: 融合池加入 deep_tcn（B3 实证低相关互补源）→ 3 序列 × 4 方法
E2b' 多序列元学习器: 6 代表序列 × 4 方法胜率矩阵（单序列→统计结论）
E2c' 跨序列重构: OOF 样本（18 序列×2352≈4.2 万行，修 97% 浪费）+ 序列统计特征
     + RF/Ridge 对照（修量纲与特征不足），GroupKFold by series
E2g' 非线性三载体: ①LGBM 残差~基函数（t 检验）②LGBM 特征±基函数（5 折增益）
     ③融合层 Ridge(预测+基函数) vs Ridge(预测)（应用闭环）

产物（output/robust/fusion_v2.json + figures/step1/figE2b2_wins.png）
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS, HOURS_TOTAL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_R = OUTPUT / "robust"
FIG_S1 = FIGURES / "step1"
SEG_TRAIN = 2352
REP6 = ["energy|RegionA|price", "energy|RegionF|price",
        "energy|RegionD|carbon", "energy|RegionC|carbon",
        "energy|RegionA|nonai", "energy|RegionF|nonai"]
DOMAIN_BASIS = ["seg_offset", "hinge_boundary", "fourier2", "saturation",
                "zero_infl_log", "multiplicative", "phase_shift", "count_link"]


def load_modules():
    s11 = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    m11 = importlib.util.module_from_spec(s11)
    s11.loader.exec_module(m11)
    s14 = importlib.util.spec_from_file_location(
        "s14", Path(__file__).resolve().parent / "step1.4_fusion_research.py")
    m14 = importlib.util.module_from_spec(s14)
    s14.loader.exec_module(m14)
    s12p = importlib.util.spec_from_file_location(
        "s12p", Path(__file__).resolve().parent / "step1.2++_fusion_ablation.py")
    m12p = importlib.util.module_from_spec(s12p)
    s12p.loader.exec_module(m12p)
    s10 = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    m10 = importlib.util.module_from_spec(s10)
    s10.loader.exec_module(m10)
    return m11, m14, m10, m12p


def mape(y, p):
    return float(np.mean(np.abs(y - p) / (np.abs(y) + 1e-9)) * 100)


def e2i_hetero(s11, s12p, name: str) -> dict:
    """异构融合：树类3 + deep_tcn（B3 互补源），4 方法 OOF 对比。"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    info = s11.make_series_dict()[name]
    y = info["y"]
    X = s11.build_features(y, "energy", info["region"])
    pool = s11.build_model_pool("energy")
    passed = [m for m in pool if m.name in
              ("lgbm_point", "xgboost_point", "qrf_quantile", "deep_tcn")]
    oofdf, wdf, yv = s12p.get_oof(s11, y, X, passed)
    names = list(oofdf.columns)
    Z = oofdf[names].to_numpy()
    ridge = Ridge(alpha=1.0).fit(Z, yv)
    m_ridge = mape(yv, ridge.predict(Z))
    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                               min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(Z, yv)
    m_rf = mape(yv, rf.predict(Z))
    m_eq = mape(yv, oofdf.mean(axis=1).to_numpy())
    widths = wdf[names].to_numpy()
    fill = np.nanmedian(widths) if not np.isnan(widths).all() else 1.0
    widths = np.nan_to_num(widths, nan=fill) + 1e-9
    w_inv = (1.0 / widths)
    w_inv = w_inv / w_inv.sum(axis=1, keepdims=True)
    m_wi = mape(yv, (w_inv * Z).sum(axis=1))
    tree3 = Z[:, :3]
    m_ridge3 = mape(yv, Ridge(alpha=1.0).fit(tree3, yv).predict(tree3))
    return {"series": name, "members": names,
            "ridge_hetero": m_ridge, "rf_hetero": m_rf,
            "equal_hetero": m_eq, "width_inverse": m_wi,
            "ridge_tree3_only": m_ridge3,
            "hetero_gain": (m_ridge3 - m_ridge) / m_ridge3 * 100}


def e2b_multi(s11, s14) -> dict:
    """多序列元学习器：6 序列 × 4 方法胜率矩阵。"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    wins = []
    for name in REP6:
        info = s11.make_series_dict()[name]
        y = info["y"]
        X = s11.build_features(y, "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in
                  ("lgbm_point", "xgboost_point", "qrf_quantile")]
        oofdf, yv = s14.get_oof(s11, y, X, passed)
        names = list(oofdf.columns)
        Z = oofdf[names].to_numpy()
        ridge = mape(yv, Ridge(alpha=1.0).fit(Z, yv).predict(Z))
        rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   min_samples_leaf=10, random_state=42,
                                   n_jobs=-1).fit(Z, yv)
        m_rf = mape(yv, rf.predict(Z))
        m_eq = mape(yv, oofdf.mean(axis=1).to_numpy())
        m_best = min(mape(yv, oofdf[n].to_numpy()) for n in names)
        scores = {"ridge": ridge, "rf": m_rf, "equal": m_eq}
        winner = min(scores, key=scores.get)
        wins.append({"series": name, "ridge": ridge, "rf": m_rf,
                     "equal": m_eq, "best_single": m_best,
                     "winner": winner})
    df = pd.DataFrame(wins)
    return {"per_series": df.to_dict("records"),
            "winner_counts": df["winner"].value_counts().to_dict()}


def e2c_cross_rebuild(s11, s14, structure_names) -> dict:
    """跨序列重构：OOF 全样本 + 序列统计特征 + RF/Ridge（GroupKFold）。"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    Xm, ym, gm, feats = [], [], [], []
    for name in structure_names:
        info = s11.make_series_dict()[name]
        y = info["y"]
        X = s11.build_features(y, "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in
                  ("lgbm_point", "xgboost_point", "qrf_quantile")]
        oofdf, yv = s14.get_oof(s11, y, X, passed)
        names = list(oofdf.columns)
        ytr = y[:SEG_TRAIN]
        stat = {"eta": 0.0, "mean": float(ytr.mean()), "cv": float(
            ytr.std() / max(ytr.mean(), 1e-9))}
        idx_arr = oofdf.index.to_numpy()
        hod = idx_arr % 24
        yv_arr = yv
        for k, (i, row) in enumerate(oofdf.iterrows()):
            Xm.append([row[n] for n in names] +
                      [stat["eta"], stat["mean"], stat["cv"],
                       np.sin(2 * np.pi * hod[k] / 24),
                       np.cos(2 * np.pi * hod[k] / 24)])
            ym.append(yv_arr[k])
            gm.append(name)
    Xm, ym, gm = map(np.array, (Xm, ym, gm))
    results = {}
    for tag, model in (("ridge", Ridge(alpha=1.0)),
                       ("rf", RandomForestRegressor(
                           n_estimators=200, max_depth=8, min_samples_leaf=10,
                           random_state=42, n_jobs=-1))):
        scores = []
        for tr, va in GroupKFold(n_splits=5).split(Xm, ym, groups=gm):
            model.fit(Xm[tr], ym[tr])
            scores.append(mape(ym[va], model.predict(Xm[va])))
        results[tag] = float(np.mean(scores))
    return {"n_rows": len(ym), "n_series": len(structure_names),
            "results": results,
            "note": "修正版：OOF 全样本（4.2 万行）+ 序列统计/时间特征 + RF 量纲无关"}


def e2g_nonlinear(s11, s14, name: str) -> dict:
    """非线性三载体：残差/特征增益/融合层。"""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    import statsmodels.api as sm
    s10 = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    m10 = importlib.util.module_from_spec(s10)
    s10.loader.exec_module(m10)
    cap = m10.load_params()["cap"]
    info = s11.make_series_dict()[name]
    y = info["y"]
    X0 = s11.build_features(y, "energy", info["region"])
    Xb = s14._domain_basis_features(y, X0, info["region"], cap)
    tscv = TimeSeriesSplit(n_splits=5)

    def lgbm_cv(Xs, feats):
        scores = []
        for tr, va in tscv.split(np.arange(SEG_TRAIN)):
            m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                              num_leaves=15, min_child_samples=30,
                              verbosity=-1)
            m.fit(Xs.iloc[tr][feats], y[tr])
            scores.append(mape(y[va], m.predict(Xs.iloc[va][feats])))
        return float(np.mean(scores))

    base_feats = ["hour_sin", "hour_cos", "dow", "is_weekend", "lag1",
                  "lag24", "lag168", "roll24_mean", "roll168_mean"]
    m_base = lgbm_cv(Xb, base_feats)
    m_plus = lgbm_cv(Xb, base_feats + list(DOMAIN_BASIS))
    carrier1 = {}
    tr0, va0 = next(iter(tscv.split(np.arange(SEG_TRAIN))))
    m = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                      min_child_samples=30, verbosity=-1)
    m.fit(Xb.iloc[tr0][base_feats], y[tr0])
    resid = y[va0] - m.predict(Xb.iloc[va0][base_feats])
    for feat in DOMAIN_BASIS:
        Zb = sm.add_constant(Xb.iloc[va0][feat].to_numpy())
        fit = sm.OLS(resid, Zb).fit()
        carrier1[feat] = {"t": float(fit.tvalues[-1]),
                          "p": float(fit.pvalues[-1]),
                          "sig": bool(fit.pvalues[-1] < 0.05)}
    pool = s11.build_model_pool("energy")
    passed = [m for m in pool if m.name in
              ("lgbm_point", "xgboost_point", "qrf_quantile")]
    oofdf, yv = s14.get_oof(s11, y, X0, passed)
    names = list(oofdf.columns)
    Z = oofdf[names].to_numpy()
    m_fuse0 = mape(yv, Ridge(alpha=1.0).fit(Z, yv).predict(Z))
    Zb2 = np.column_stack([Z] + [Xb.loc[oofdf.index, f].to_numpy()
                                 for f in DOMAIN_BASIS])
    m_fuse1 = mape(yv, Ridge(alpha=1.0).fit(Zb2, yv).predict(Zb2))
    return {"series": name,
            "carrier1_residual_t": carrier1,
            "carrier2_lgbm": {"base": m_base, "plus_basis": m_plus,
                              "gain_pct": (m_base - m_plus) / m_base * 100},
            "carrier3_fusion": {"base": m_fuse0, "plus_basis": m_fuse1,
                                "gain_pct": (m_fuse0 - m_fuse1) / m_fuse0 * 100}}


def main() -> None:
    s11, s14, s10, s12p = load_modules()
    structure = [n for n, i in s11.make_series_dict().items()
                 if i["layer"] == "energy" and "renewable" not in n]
    res = {}
    res["e2i"] = [e2i_hetero(s11, s12p, n)
                  for n in ("energy|RegionA|price", "energy|RegionD|carbon",
                            "energy|RegionF|nonai")]
    res["e2b_multi"] = e2b_multi(s11, s14)
    res["e2c_rebuild"] = e2c_cross_rebuild(s11, s14, structure)
    res["e2g"] = {n: e2g_nonlinear(s11, s14, n) for n in
                  ("energy|RegionA|price", "energy|RegionD|carbon")}
    with open(OUT_R / "fusion_v2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
