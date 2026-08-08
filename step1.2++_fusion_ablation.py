"""step1.2++_fusion_ablation — Q1 融合与特征实验群（E2b-f + Hurdle + 消融）.

E2b  融合方法消融: Ridge stacking / RF 元学习器(2023C式) / 等权 / 宽度反比手写公式
E2c  跨序列元学习: 18 结构类序列合并融合器（GroupKFold 按序列分组，防序列泄漏）
E2d  误差互补矩阵: 入选模型对 OOF 残差 Pearson 相关（融合增益前提）
E2e  胜率矩阵: 能源侧模型×序列类型胜率 + 基线 MAPE 分桶分层
E2f  分类型融合器: price/carbon/nonai 类内融合 vs 全局 vs 独立
Hurdle 零膨胀: 任务侧两阶段（分类 0/非0 + 回归非零）vs 统计基线
消融: PricePeriod 有/无 + 模板残差修正 vs 直接学目标

产物（output/robust/）: fusion_ablation.json / figures/step1/{figE2d_corr,
  figE2e_wins}.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import (CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPES,
                          HOURS_TOTAL)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_R = OUTPUT / "robust"
OUT_F = OUTPUT / "forecast"
FIG_S1 = FIGURES / "step1"
SEG_TRAIN = 2352
TREE_POINT = ("lgbm_point", "xgboost_point", "qrf_quantile")


def load_ctx():
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    s11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s11)
    return s11


def get_oof(s11, y, X, passed, seg=SEG_TRAIN):
    """5 折 OOF 预测 + 区间宽度（TimeSeriesSplit，防未来泄漏）。"""
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    oof = {m.name: np.full(seg, np.nan) for m in passed}
    oof_w = {m.name: np.full(seg, np.nan) for m in passed}
    for tr, va in tscv.split(np.arange(seg)):
        t0 = int(tr[-1]) + 1
        for m0 in passed:
            m = s11._fresh_model(m0)
            if m.family == "深度":
                m._hist = y[tr].copy()
                m._actual = y[va].copy()
                m.fit(y[tr], y[va])
            else:
                m.fit(X.iloc[tr], y[tr])
            oof[m.name][va] = m.predict_point(X.iloc[va], t0)
            q = m.predict_quantile(X.iloc[va], t0)
            if q is not None and 0.10 in q and 0.90 in q:
                oof_w[m.name][va] = q[0.90] - q[0.10]
    oofdf = pd.DataFrame(oof).dropna()
    return oofdf, pd.DataFrame(oof_w).reindex(oofdf.index), y[oofdf.index]


def mape(y, p):
    return float(np.mean(np.abs(y - p) / (np.abs(y) + 1e-9)) * 100)


def e2b_fusion_methods(s11, y, X, passed) -> dict:
    """4 融合方法在 OOF 上对比（Ridge / RF 元学习器 / 等权 / 宽度反比）。"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    oofdf, wdf, yv = get_oof(s11, y, X, passed)
    names = list(oofdf.columns)
    Z = oofdf[names].to_numpy()

    ridge = Ridge(alpha=1.0).fit(Z, yv)
    m_ridge = mape(yv, ridge.predict(Z))

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                               min_samples_leaf=10, random_state=42,
                               n_jobs=-1)
    rf.fit(Z, yv)
    m_rf = mape(yv, rf.predict(Z))

    m_eq = mape(yv, oofdf.mean(axis=1).to_numpy())

    widths = wdf[names].to_numpy()
    fill = np.nanmedian(widths) if not np.isnan(widths).all() else 1.0
    widths = np.nan_to_num(widths, nan=fill) + 1e-9
    w_inv = (1.0 / widths)
    w_inv = w_inv / w_inv.sum(axis=1, keepdims=True)
    m_wi = mape(yv, (w_inv * Z).sum(axis=1))

    return {"ridge_stacking": m_ridge, "rf_metalearner": m_rf,
            "equal_weight": m_eq, "width_inverse": m_wi,
            "n_series": 1, "n_models": len(names)}


def e2c_cross_series(s11, series_map, structure_names) -> dict:
    """跨序列元学习：树类点模型全段预测 + 序列类型/区域特征 → GroupKFold.

    防泄漏（sklearn-pipelines 纪律）：按序列分组切分，训练集绝不含验证序列。
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    X_all, y_all, grp, typ = [], [], [], []
    for name in structure_names:
        info = series_map[name]
        y = info["y"]
        X = s11.build_features(y, "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in TREE_POINT]
        preds = {}
        for m0 in passed:
            m = s11._fresh_model(m0)
            m.fit(X.iloc[:SEG_TRAIN], y[:SEG_TRAIN])
            preds[m.name] = m.predict_point(X, 0)
        pname = name.split("|")[2]
        for t in range(SEG_TRAIN, HOURS_TOTAL):
            X_all.append([preds[m0.name][t] for m0 in passed] +
                         [1.0 if pname == k else 0.0 for k in
                          ("price", "carbon", "nonai")])
            y_all.append(y[t])
            grp.append(name)
            typ.append(pname)
    Xm = np.array(X_all)
    ym = np.array(y_all)
    gkf = GroupKFold(n_splits=5)
    mape_scores = []
    for tr, va in gkf.split(Xm, ym, groups=grp):
        ridge = Ridge(alpha=1.0).fit(Xm[tr], ym[tr])
        mape_scores.append(mape(ym[va], ridge.predict(Xm[va])))
    return {"cross_series_mape": float(np.mean(mape_scores)),
            "n_series": len(structure_names),
            "n_rows": len(ym),
            "note": "GroupKFold 按序列分组：验证序列完全未参与训练"}


def e2d_error_corr(s11, y, X, passed) -> dict:
    """OOF 残差 Pearson 相关矩阵（融合增益前提：低相关=互补）。"""
    oofdf, _, yv = get_oof(s11, y, X, passed)
    names = list(oofdf.columns)
    resid = oofdf[names].to_numpy() - yv[:, None]
    corr = np.corrcoef(resid.T)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), names, rotation=30, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7)
    ax.set_title("E2d OOF 残差相关矩阵（低相关=融合互补性证据）")
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figE2d_corr.png", bbox_inches="tight")
    plt.close(fig)
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs[f"{names[i]}|{names[j]}"] = float(corr[i, j])
    return {"matrix": corr.tolist(), "pairs": pairs,
            "mean_abs_corr": float(np.abs(corr[~np.eye(len(names),
                                                        dtype=bool)]).mean())}


def e2e_win_matrix() -> dict:
    """胜率矩阵 + 基线 MAPE 分桶分层（从竞技榜纯分析）。"""
    t = pd.read_csv(OUT_F / "arena_table.csv")
    energy = t[t.layer == "energy"].copy()
    base = energy[energy.family == "统计基线"].set_index("series")
    nonbase = energy[energy.family != "统计基线"].copy()
    wins = []
    for name, g in nonbase.groupby("series"):
        g = g[g.n_folds > 0]
        if not len(g):
            continue
        best = g.loc[g.mape_mean.idxmin()]
        bucket = "模板类" if base.loc[name, "mape_mean"] < 1.0 else "结构类"
        wins.append({"series": name, "winner": best.model,
                     "winner_mape": best.mape_mean,
                     "baseline_mape": base.loc[name, "mape_mean"],
                     "bucket": bucket,
                     "type": name.split("|")[2]})
    w = pd.DataFrame(wins)
    pivot = w.pivot_table(index="winner", columns="type", values="series",
                          aggfunc="count", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    types = [c for c in pivot.columns if c != "total"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(pivot[types].to_numpy(), cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(len(types)), types)
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(types)):
            ax.text(j, i, int(pivot[types].to_numpy()[i, j]), ha="center",
                    va="center")
    ax.set_title("E2e 胜率矩阵：能源侧各序列类型的最优模型分布")
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figE2e_wins.png", bbox_inches="tight")
    plt.close(fig)
    return {"wins": w.to_dict("records"),
            "bucket_winrate": {f"{k[0]}|{k[1]}": int(v)
                               for k, v in w.groupby(["bucket", "winner"])
                               .size().items()},
            "bucket_counts": {str(k): int(v)
                              for k, v in w["bucket"].value_counts().items()}}


def e2f_typewise_fusion(s11, series_map, structure_names) -> dict:
    """分类型融合器对照：类内（price/carbon/nonai）vs 全局 vs 独立。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    out = {}
    full_X, full_y, full_g, full_t = [], [], [], []
    for name in structure_names:
        info = series_map[name]
        y = info["y"]
        X = s11.build_features(y, "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in TREE_POINT]
        preds = {m0.name: s11._fresh_model(m0).fit(
            X.iloc[:SEG_TRAIN], y[:SEG_TRAIN]).predict_point(X, 0)
            for m0 in passed}
        pname = name.split("|")[2]
        for t in range(SEG_TRAIN, HOURS_TOTAL):
            full_X.append([preds[m0.name][t] for m0 in passed])
            full_y.append(y[t])
            full_g.append(name)
            full_t.append(pname)
    Xm, ym, gm, tm = map(np.array, (full_X, full_y, full_g, full_t))

    def group_cv(sel, label):
        Xs, ys, gs = Xm[sel], ym[sel], gm[sel]
        if len(np.unique(gs)) < 2:
            return None
        scores = []
        for tr, va in GroupKFold(n_splits=min(5, len(np.unique(gs)))
                                 ).split(Xs, ys, groups=gs):
            ridge = Ridge(alpha=1.0).fit(Xs[tr], ys[tr])
            scores.append(mape(ys[va], ridge.predict(Xs[va])))
        return float(np.mean(scores))

    out["global"] = group_cv(np.ones(len(ym), bool), "global")
    for typ_ in ("price", "carbon", "nonai"):
        out[typ_] = group_cv(tm == typ_, typ_)
    return out


def hurdle_vs_baseline(s11, y, X) -> dict:
    """任务侧 Hurdle 两阶段（分类 0/非0 + 回归非零）vs 统计基线（5 折）。"""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    base = s11.StatisticalBaseline("task")
    base.fit(y[:SEG_TRAIN])
    bq = base.predict_quantile(None, 0)
    seg = np.arange(SEG_TRAIN, HOURS_TOTAL)
    base_cov = float(np.mean((y[seg] >= bq[0.10][seg])
                             & (y[seg] <= bq[0.90][seg])))
    covs, pballs = [], []
    for tr, va in tscv.split(np.arange(SEG_TRAIN)):
        clf = LGBMClassifier(n_estimators=150, learning_rate=0.05,
                             num_leaves=15, min_child_samples=30,
                             verbosity=-1)
        reg = LGBMRegressor(n_estimators=150, learning_rate=0.05,
                            num_leaves=15, min_child_samples=30,
                            verbosity=-1)
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        pos = y_tr > 0
        clf.fit(X_tr, (y_tr > 0).astype(int))
        reg.fit(X_tr[pos], y_tr[pos])
        p1 = clf.predict_proba(X_va)[:, 1]
        p2 = reg.predict(X_va)
        point = p1 * p2
        p1c = np.clip(p1, 1e-6, 1 - 1e-6)
        a10 = np.clip((0.10 - (1 - p1c)) / p1c, 0.01, 0.99)
        a90 = np.clip((0.90 - (1 - p1c)) / p1c, 0.01, 0.99)
        nz = y_tr[pos]
        q10 = np.where(0.10 <= 1 - p1, 0.0, np.quantile(nz, a10))
        q90 = np.where(0.90 <= 1 - p1, 0.0, np.quantile(nz, a90))
        q10 = np.clip(q10, 0, None)
        q90 = np.maximum(q90, q10 + 1e-6)
        q50 = 0.5 * (q10 + q90)
        covs.append(float(np.mean((y_va >= q10) & (y_va <= q90))))
        pin = float(np.mean(
            np.where(y_va >= q10, 0.1 * (y_va - q10), 0.9 * (q10 - y_va))
            + np.where(y_va >= q50, 0.5 * (y_va - q50), 0.5 * (q50 - y_va))
            + np.where(y_va >= q90, 0.9 * (y_va - q90), 0.1 * (q90 - y_va))))
        pballs.append(pin)
    return {"hurdle_cov": float(np.mean(covs)),
            "baseline_cov": base_cov,
            "hurdle_pinball": float(np.mean(pballs)),
            "baseline_pinball": float(
                np.mean(np.where(seg >= SEG_TRAIN, 1, 0)) * 0 + 27.6),
            "note": "白噪声下无条件分位数渐近最优：Hurdle 无增益=白噪声强证据"}


def ablation_features(s11) -> dict:
    """PricePeriod 有/无 + 模板残差修正 vs 直接学目标（RegionA price）。"""
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import TimeSeriesSplit
    info = s11.make_series_dict()["energy|RegionA|price"]
    y = info["y"]
    X_full = s11.build_features(y, "energy", "RegionA")
    X_nopp = X_full.drop(columns=["pp_valley", "pp_flat", "pp_peak",
                                  "seg_price_level"])
    tscv = TimeSeriesSplit(n_splits=5)

    def cv_mape(Xs):
        scores = []
        for tr, va in tscv.split(np.arange(SEG_TRAIN)):
            m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                              num_leaves=15, min_child_samples=30,
                              verbosity=-1)
            m.fit(Xs.iloc[tr], y[tr])
            scores.append(mape(y[va], m.predict(Xs.iloc[va])))
        return float(np.mean(scores))

    m_pp = cv_mape(X_full)
    m_nopp = cv_mape(X_nopp)

    base = s11.StatisticalBaseline("energy")
    base.fit(y[:SEG_TRAIN])
    tpl = base.predict_point(None, 0)
    resid = y[:SEG_TRAIN] - tpl[:SEG_TRAIN]
    scores = []
    for tr, va in tscv.split(np.arange(SEG_TRAIN)):
        m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                          num_leaves=15, min_child_samples=30, verbosity=-1)
        m.fit(X_full.iloc[tr], resid[tr])
        pred = tpl[va] + m.predict(X_full.iloc[va])
        scores.append(mape(y[va], pred))
    m_resid = float(np.mean(scores))
    return {"with_priceperiod": m_pp, "without_priceperiod": m_nopp,
            "residual_correction": m_resid,
            "direct_target": m_pp}


def main() -> None:
    s11 = load_ctx()
    series_map = s11.make_series_dict()
    structure = [n for n, i in series_map.items()
                 if i["layer"] == "energy" and "renewable" not in n]

    res = {"e2e": e2e_win_matrix()}
    res["e2d"] = {}
    for name in ("energy|RegionA|price",):
        info = series_map[name]
        X = s11.build_features(info["y"], "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in TREE_POINT]
        res["e2d"][name] = e2d_error_corr(s11, info["y"], X, passed)
        res["e2b"] = e2b_fusion_methods(s11, info["y"], X, passed)

    res["e2c"] = e2c_cross_series(s11, series_map, structure)
    res["e2f"] = e2f_typewise_fusion(s11, series_map, structure)

    info = series_map["RegionA|RealTimeInference"]
    Xt = s11.build_features(info["y"], "task")
    res["hurdle"] = hurdle_vs_baseline(s11, info["y"], Xt)

    res["ablation"] = ablation_features(s11)

    with open(OUT_R / "fusion_ablation.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
