"""step1.4_fusion_research — 融合研究深化（B1-B5）.

B1  融合范式判别: 条件残差画像（时段×需求水平×类型）→ 最优模型漂移检测
    + oracle routing 上界 + 分箱权重 vs 固定加权 vs 选择 三范式
B2  领域基函数检验: 8 族预定基函数（阶梯/铰链/傅里叶/饱和/零膨胀/乘法/时移/计数）
    + F 检验逐项裁决 + 领域 vs 通用基函数对照
B3  异构互补补测: tabpfn/deep vs 树类残差相关（融合池成员补充）
B4  RF 元模型重测: global 融合换 RF（尺度不变）+ 轻量调参
B5  四家族条件擅长画像: TabPFN 稳健性 + 条件擅长表（建模铺路+论文素材）

产物（output/robust/fusion_research.json + figures/step1/figB1_*.png 等）
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
PERIODS = ["Valley", "Flat", "Peak"]
PERIOD_HOURS = {"Valley": set(range(0, 7)), "Flat": set(range(7, 17)) | {22, 23},
                "Peak": set(range(17, 22))}
REP_SERIES = ["energy|RegionA|price", "energy|RegionF|nonai",
              "energy|RegionD|carbon"]
DOMAIN_BASIS = ["seg_offset", "hinge_boundary", "fourier2", "saturation",
                "zero_infl_log", "multiplicative", "phase_shift", "count_link"]


def load_s11():
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def get_oof(s11, y, X, passed, seg=SEG_TRAIN):
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    oof = {m.name: np.full(seg, np.nan) for m in passed}
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
    oofdf = pd.DataFrame(oof).dropna()
    return oofdf, y[oofdf.index]


def mape(y, p):
    return float(np.mean(np.abs(y - p) / (np.abs(y) + 1e-9)) * 100)


def b1_paradigm(s11, name: str) -> dict:
    """融合范式判别：条件残差画像 + oracle routing 上界 + 三范式对比。"""
    from sklearn.linear_model import Ridge
    info = s11.make_series_dict()[name]
    y = info["y"]
    X = s11.build_features(y, "energy", info["region"])
    pool = s11.build_model_pool("energy")
    passed = [m for m in pool if m.name in
              ("lgbm_point", "xgboost_point", "qrf_quantile")]
    oofdf, yv = get_oof(s11, y, X, passed)
    names = list(oofdf.columns)
    idx = oofdf.index.to_numpy()
    resid = oofdf[names].to_numpy() - yv[:, None]
    abs_resid = np.abs(resid)

    hod = idx % 24
    period = np.array([next(p for p, hs in PERIOD_HOURS.items() if h in hs)
                       for h in hod])
    q_level = pd.qcut(yv, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)

    drift = {}
    for cond_name, cond in (("period", period), ("level", q_level)):
        best_per_cond = {}
        for c in np.unique(cond):
            m_ = abs_resid[cond == c].mean(axis=0)
            best_per_cond[str(c)] = names[int(np.argmin(m_))]
        drift[cond_name] = best_per_cond
    drift_ratio = len(set(drift["period"].values())) / len(names)

    oracle_best = abs_resid.min(axis=1)
    oracle_mape = mape(yv, oofdf.to_numpy()[np.arange(len(yv)),
                                            abs_resid.argmin(axis=1)])
    best_single = mape(yv, oofdf.mean(axis=1).to_numpy())
    best_model_mape = min(mape(yv, oofdf[n].to_numpy()) for n in names)

    ridge = Ridge(alpha=1.0).fit(oofdf[names].to_numpy(), yv)
    ridge_mape = mape(yv, ridge.predict(oofdf[names].to_numpy()))

    weights = np.zeros((len(yv), len(names)))
    for c in np.unique(period):
        m_ = abs_resid[period == c].mean(axis=0)
        w = 1.0 / (m_ + 1e-9)
        weights[period == c] = w / w.sum()
    cond_mape = mape(yv, (weights * oofdf[names].to_numpy()).sum(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    table = np.stack([abs_resid[period == p].mean(axis=0)
                      for p in PERIODS])
    im = axes[0].imshow(table, cmap="YlOrRd")
    axes[0].set_xticks(range(len(names)), names, rotation=30, fontsize=7)
    axes[0].set_yticks(range(3), PERIODS)
    for i in range(3):
        for j in range(len(names)):
            axes[0].text(j, i, f"{table[i, j]:.1f}", ha="center", va="center",
                         fontsize=7)
    axes[0].set_title("时段×模型 平均|残差|（最优=暗）")
    plt.colorbar(im, ax=axes[0], fraction=0.04)
    lv = np.stack([abs_resid[q_level == q].mean(axis=0)
                   for q in ["Q1", "Q2", "Q3", "Q4"]])
    im2 = axes[1].imshow(lv, cmap="YlOrRd")
    axes[1].set_xticks(range(len(names)), names, rotation=30, fontsize=7)
    axes[1].set_yticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    axes[1].set_title("需求水平×模型 平均|残差|")
    plt.colorbar(im2, ax=axes[1], fraction=0.04)
    fig.suptitle(f"B1 条件残差画像（{name.split('|')[1]}）")
    fig.tight_layout()
    fig.savefig(FIG_S1 / "figB1_profile.png", bbox_inches="tight")
    plt.close(fig)

    return {"series": name,
            "drift_period": drift["period"],
            "drift_level": drift["level"],
            "drift_fraction_period": drift_ratio,
            "oracle_routing_mape": oracle_mape,
            "best_single_mape": best_model_mape,
            "equal_weight_mape": best_single,
            "ridge_mape": ridge_mape,
            "conditional_weight_mape": cond_mape,
            "note": "oracle routing=事后真实最优模型选择（融合潜力理论上界）；"
                    "条件权重=按时段分箱的最优权重"}


def _domain_basis_features(y, X, region, cap: dict):
    """8 族预定领域基函数（机理推导，非数据挖掘；容量从数据读）。"""
    df = X.copy()
    hod = np.arange(len(y)) % 24
    cap_r = cap[region]
    seg_means = {}
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    seg_means = rt[rt.Region == region].groupby("PricePeriod")[
        "ElectricityPrice_CNY_per_MWh"].mean().to_dict()
    period = np.array([next(p for p, hs in PERIOD_HOURS.items() if h in hs)
                       for h in hod])
    df["seg_offset"] = df["lag24"] - df["lag24"].groupby(
        pd.Series(period)).transform("mean")
    df["hinge_boundary"] = np.maximum(0, hod - 17) + np.maximum(0, 6 - hod)
    df["fourier2"] = np.sin(4 * np.pi * hod / 24)
    df["saturation"] = np.minimum(df["lag24"], cap_r)
    df["zero_infl_log"] = np.log1p(np.maximum(df["lag1"], 0))
    df["multiplicative"] = df["lag1"] * df["lag24"] / 1e3
    df["phase_shift"] = df["lag24"] * df["lag1"]
    df["count_link"] = np.exp(-df["lag1"] / (df["lag1"].mean() + 1e-9))
    return df.fillna(0.0)


def b2_domain_basis(s11, name: str) -> dict:
    """领域基函数检验：线性元模型 + 基函数，F 检验逐项裁决。"""
    import statsmodels.api as sm
    from sklearn.model_selection import TimeSeriesSplit
    s10 = importlib.util.spec_from_file_location(
        "s10", Path(__file__).resolve().parent / "step1.0_baseline_schedule.py")
    m10 = importlib.util.module_from_spec(s10)
    s10.loader.exec_module(m10)
    cap = m10.load_params()["cap"]
    info = s11.make_series_dict()[name]
    y = info["y"]
    X0 = s11.build_features(y, "energy", info["region"])
    X = _domain_basis_features(y, X0, info["region"], cap)
    tscv = TimeSeriesSplit(n_splits=5)
    base_feats = ["hour_sin", "hour_cos", "dow", "is_weekend", "lag1",
                  "lag24", "lag168", "roll24_mean", "roll168_mean"]
    rows = []
    for tr, va in tscv.split(np.arange(SEG_TRAIN)):
        t0 = int(tr[-1]) + 1
        sub = y[va]
        for feat in DOMAIN_BASIS:
            cols = base_feats + [feat]
            Zb = sm.add_constant(X.iloc[va][cols].to_numpy())
            model = sm.OLS(sub, Zb).fit()
            rows.append({"fold": tr[0], "feat": feat,
                         "coef": float(model.params[-1]),
                         "t": float(model.tvalues[-1]),
                         "p": float(model.pvalues[-1])})
    df = pd.DataFrame(rows)
    summary = {}
    for feat in DOMAIN_BASIS:
        sub = df[df.feat == feat]
        sig = int((sub.p < 0.05).sum())
        summary[feat] = {"sig_folds": sig, "n_folds": 5,
                         "median_t": float(sub.t.median()),
                         "median_coef": float(sub.coef.median()),
                         "significant": sig >= 3}
    return {"series": name, "basis": summary,
            "note": "F/t 检验裁决领域基函数是否显著；显著≥3 折视为机理存在"}


def b3_hetero_corr(s11, name: str) -> dict:
    """异构互补补测：tabpfn/deep vs 树类残差相关。"""
    info = s11.make_series_dict()[name]
    y = info["y"]
    X = s11.build_features(y, "energy", info["region"])
    pool = s11.build_model_pool("energy")
    passed = [m for m in pool if m.name in
              ("lgbm_point", "xgboost_point", "qrf_quantile", "tabpfn",
               "deep_tcn")]
    oofdf, yv = get_oof(s11, y, X, passed)
    names = list(oofdf.columns)
    resid = oofdf[names].to_numpy() - yv[:, None]
    corr = np.corrcoef(resid.T)
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs[f"{names[i]}|{names[j]}"] = float(corr[i, j])
    return {"series": name, "pairs": pairs}


def b4_rf_global(s11, structure_names: list) -> dict:
    """RF 元模型 global 重测（尺度不变）+ 轻量调参。"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    Xm, ym, gm = [], [], []
    for name in structure_names:
        info = s11.make_series_dict()[name]
        y = info["y"]
        X = s11.build_features(y, "energy", info["region"])
        pool = s11.build_model_pool("energy")
        passed = [m for m in pool if m.name in
                  ("lgbm_point", "xgboost_point", "qrf_quantile")]
        preds = {m0.name: s11._fresh_model(m0).fit(
            X.iloc[:SEG_TRAIN], y[:SEG_TRAIN]).predict_point(X, 0)
            for m0 in passed}
        for t in range(SEG_TRAIN, HOURS_TOTAL):
            Xm.append([preds[m0.name][t] for m0 in passed])
            ym.append(y[t])
            gm.append(name)
    Xm, ym, gm = map(np.array, (Xm, ym, gm))
    results = {}
    for cfg in ({"n_estimators": 200, "max_depth": 6},
                {"n_estimators": 400, "max_depth": 10},
                {"n_estimators": 200, "max_depth": None}):
        scores = []
        for tr, va in GroupKFold(n_splits=5).split(Xm, ym, groups=gm):
            rf = RandomForestRegressor(random_state=42, n_jobs=-1, **cfg)
            rf.fit(Xm[tr], ym[tr])
            scores.append(mape(ym[va], rf.predict(Xm[va])))
        results[str(cfg)] = float(np.mean(scores))
    return {"n_series": len(structure_names), "n_rows": len(ym),
            "configs": results,
            "note": "RF 尺度不变：global 融合不再因量纲爆炸"}


def b5_profile(s11) -> dict:
    """四家族条件擅长画像：按序列类型×模型族的最佳者 + TabPFN 稳健性。"""
    t = pd.read_csv(OUTPUT / "forecast" / "arena_table.csv")
    e = t[(t.layer == "energy") & (t.family != "统计基线")].copy()
    e["type"] = e["series"].str.split("|").str[2]
    rows = []
    for (typ, fam), g in e.groupby(["type", "family"]):
        rows.append({"type": typ, "family": fam,
                     "best_mape": float(g.mape_mean.min()),
                     "median_mape": float(g.mape_mean.median())})
    tab = pd.DataFrame(rows).pivot_table(index="family", columns="type",
                                         values="median_mape")
    tab = tab.round(3)
    tabpfn_rows = e[e.model == "tabpfn"]
    stability = {"mean_cov_std": float(tabpfn_rows.cov_std.mean()),
                 "mean_mape_std": float(tabpfn_rows.mape_std.mean()),
                 "best_series": int((tabpfn_rows.groupby("series")
                                     .mape_mean.idxmin().count())),
                 "n_series": tabpfn_rows["series"].nunique()}
    return {"profile_table": tab.to_dict(), "tabpfn_stability": stability}


def main() -> None:
    s11 = load_s11()
    structure = [n for n, i in s11.make_series_dict().items()
                 if i["layer"] == "energy" and "renewable" not in n]

    res = {}
    res["b1"] = [b1_paradigm(s11, n) for n in REP_SERIES]
    res["b2"] = b2_domain_basis(s11, REP_SERIES[0])
    res["b3"] = {n: b3_hetero_corr(s11, n) for n in
                 ("energy|RegionA|price", "energy|RegionD|carbon")}
    res["b4"] = b4_rf_global(s11, structure)
    res["b5"] = b5_profile(s11)

    with open(OUT_R / "fusion_research.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
