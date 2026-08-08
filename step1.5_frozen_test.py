"""step1.5_frozen_test — 冻结段最终评估（防泄露三段协议闭环）+ CV-冻结段一致性.

协议: 训练 0-2351（全段）→ 评估 2376-2399（一次性，不回头调参）。
对象: 42 序列 × 全池 8 模型（用户指示：评估对象全面）。
指标: 能源侧 MAPE/RMSE；任务侧 覆盖率(90% 区间)/宽度/pinball。
一致性研究: CV 类选结论（arena_table）vs 冻结段实测——验证门是否预测了
  冻结段相对表现（GRADE 式"结论可泛化"证据）。

产物（output/robust/frozen_test.json + figures/step1/fig_frozen_consistency.png）
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
SEG_TRAIN, SEG_TEST = 2352, 2400


def load_s11():
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parent / "step1.1_forecast_arena.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def eval_frozen(s11, name, info):
    """单序列冻结段评估（全池 8 模型）。"""
    y = info["y"]
    X = s11.build_features(y, info["layer"], info.get("region"))
    pool = s11.build_model_pool(info["layer"])
    seg = np.arange(SEG_TEST, HOURS_TOTAL)
    y_true = y[seg]
    rows = []
    for m in pool:
        m = s11._fresh_model(m)
        try:
            if isinstance(m, s11.StatisticalBaseline):
                m.fit(y[:SEG_TRAIN])
                point = m.predict_point(X.iloc[seg], 0)
                q = m.predict_quantile(X.iloc[seg], 0)
            elif m.family == "深度":
                m._hist = y[:24].copy()
                m._actual = y[24:].copy()
                m.fit(y[:SEG_TRAIN], y[2352 - 100:SEG_TRAIN])
                point = m.predict_point(X.iloc[seg], 0)
                q = m.predict_quantile(X.iloc[seg], 0)
            else:
                m.fit(X.iloc[:SEG_TRAIN], y[:SEG_TRAIN])
                point = m.predict_point(X.iloc[seg], 0)
                q = m.predict_quantile(X.iloc[seg], 0)
                if q is None:
                    sd = float(np.std(y[:SEG_TRAIN])) + 1e-8
                    z = np.array([-1.2816, 0.0, 1.2816])
                    q = {a: point + z[i] * sd
                         for i, a in enumerate((0.10, 0.50, 0.90))}
            sc = s11.evaluate_fold(info["layer"], y_true, point, q)
            rows.append({"model": m.name, "family": m.family,
                         "mape": sc["mape"], "rmse": sc["rmse"],
                         "cov": sc["cov"], "width": sc["width"],
                         "pinball": sc["pinball"]})
        except Exception as e:
            rows.append({"model": m.name, "family": m.family,
                         "error": repr(e)[:80]})
    return pd.DataFrame(rows)


def consistency(series: dict, frozen: dict) -> dict:
    """CV-冻结段一致性：每序列 CV 最优模型 vs 冻结段最优模型。"""
    cv = pd.read_csv(OUTPUT / "forecast" / "arena_table.csv")
    cv_best = {}
    for name, g in cv[cv.family != "统计基线"].groupby("series"):
        g = g[g.n_folds > 0]
        if len(g):
            cv_best[name] = g.loc[g.mape_mean.idxmin(), "model"]
    match = 0
    total = 0
    for name, fd0 in frozen.items():
        fd = pd.DataFrame(fd0)
        if name not in cv_best:
            continue
        fb = fd[fd.mape.notna()].sort_values("mape")
        if not len(fb):
            continue
        total += 1
        if fb.iloc[0]["model"] == cv_best[name]:
            match += 1
    return {"n_series": total, "cv_frozen_best_match": match,
            "match_rate": round(match / max(total, 1), 3)}


def main() -> None:
    s11 = load_s11()
    series = s11.make_series_dict()
    frozen = {}
    for name, info in series.items():
        fd = eval_frozen(s11, name, info)
        frozen[name] = fd.to_dict("records")
        if len(frozen) % 10 == 0:
            print(f"进度 {len(frozen)}/42", flush=True)

    cons = consistency(series, frozen)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    layers = ["task", "energy"]
    for i, lay in enumerate(layers):
        sub = {n: pd.DataFrame(fd) for n, fd in frozen.items()
               if series[n]["layer"] == lay}
        mape_med = []
        for fd in sub.values():
            fb = fd[fd.mape.notna()]
            if len(fb):
                mape_med.append(fb.sort_values("mape").iloc[0]["mape"])
        if mape_med:
            ax.bar(i, np.median(mape_med), color=["#e67e22", "#16a085"][i],
                   label=f"{lay} 冻结段最优 MAPE 中位数")
    ax.set_xticks(range(2), ["任务侧", "能源侧"])
    ax.set_ylabel("最优 MAPE")
    ax.legend()
    ax.set_title(f"冻结段评估：CV-冻结段最优模型一致率 {cons['match_rate']:.0%}")
    fig.tight_layout()
    fig.savefig(FIG_S1 / "fig_frozen_consistency.png", bbox_inches="tight")
    plt.close(fig)

    out = {"consistency": cons,
           "per_series": {n: {r["model"]: {k: r[k] for k in
                          ("mape", "rmse", "cov", "width", "pinball")
                          if k in r} for r in fd}
                          for n, fd in frozen.items()},
           "note": "冻结段 2376-2399 一次性评估（三段协议闭环）；"
                   "训练 0-2351 全段；深度早停用 2352-2375（调参段，合法）"}
    with open(OUT_R / "frozen_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"consistency": cons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
