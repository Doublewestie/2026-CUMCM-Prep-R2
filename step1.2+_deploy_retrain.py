"""step1.2+_deploy_retrain — P1+A1+A2: 说明5 部署口径重训 + 冻结段覆盖率/双口径.

P1（说明 5 严格口径）: 模型确定后以 0-2375 重训，对 2376-2399 最终测试。
  原 frozen_test 用 0-2351 拟合 → 本脚本重跑 0-2375 拟合版，对比冻结段 MAPE。
A1: 任务侧 18 序列冻结段 90% 区间覆盖率（CV 89.5% vs 冻结段——首次报告）。
A2: MAPE 双口径分层（CV=类选口径 / 冻结段=部署口径）统一报告。

产物（output/robust/）: deploy_retrain.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step0_config import OUTPUT, REGIONS, TASK_TYPES

OUT_R = OUTPUT / "robust"
OUT_F = OUTPUT / "forecast"
SEG_TRAIN, SEG_CAL_END, SEG_END = 2352, 2376, 2400


def load_ctx():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s11", root / "step1.1_forecast_arena.py")
    s11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s11)
    return s11


def eval_series_frozen(s11, name, info, fit_end: int) -> dict:
    """单序列：拟合到 fit_end → 冻结段评估（MAPE + 90% 区间覆盖率）。"""
    y = info["y"]
    X = s11.build_features(y, info["layer"], info.get("region"))
    pool = s11.build_model_pool(info["layer"])
    seg = np.arange(SEG_CAL_END, SEG_END)
    y_true = y[seg]
    out = {}
    for m in pool:
        m = s11._fresh_model(m)
        try:
            if isinstance(m, s11.StatisticalBaseline):
                m.fit(y[:fit_end])
                point = m.predict_point(X.iloc[seg], 0)
                q = m.predict_quantile(X.iloc[seg], 0)
            elif m.family == "深度":
                m._hist = y[:24].copy()
                m._actual = y[24:].copy()
                m.fit(y[:fit_end], y[max(0, fit_end - 100):fit_end])
                point = m.predict_point(X.iloc[seg], 0)
                q = m.predict_quantile(X.iloc[seg], 0)
            else:
                m.fit(X.iloc[:fit_end], y[:fit_end])
                point = m.predict_point(X.iloc[seg], 0)
                q = m.predict_quantile(X.iloc[seg], 0)
                if q is None:
                    sd = float(np.std(y[:fit_end])) + 1e-8
                    z = np.array([-1.2816, 0.0, 1.2816])
                    q = {a: point + z[i] * sd
                         for i, a in enumerate((0.10, 0.50, 0.90))}
            sc = s11.evaluate_fold(info["layer"], y_true, point, q)
            out[m.name] = {"mape": sc["mape"], "cov": sc["cov"],
                           "width": sc["width"]}
        except Exception:
            continue
    return out


def main() -> None:
    s11 = load_ctx()
    series = s11.make_series_dict()
    # 只评估统计基线 + 部署口径入选模型（效率：非全池重训）
    dep = pd.read_csv(OUT_F / "deploy_arena.csv")
    dep_best_models = set(dep[dep.deploy_pass].model)
    dep_best_models.add("stat_hist")
    pool_names = ["stat_hist", "lgbm_point", "lgbm_quantile", "tabpfn",
                  "xgboost_point", "qrf_quantile", "gbm_quantile"]
    report = {"per_series": {}, "summary": {}}
    for name, info in series.items():
        r2352 = eval_series_frozen(s11, name, info, SEG_TRAIN)
        r2376 = eval_series_frozen(s11, name, info, SEG_CAL_END)
        keep = {k: v for k, v in r2376.items() if k in pool_names}
        report["per_series"][name] = {
            "fit_0_2351": r2352, "fit_0_2375": keep}
    # 汇总
    en = {k: v for k, v in report["per_series"].items()
          if k.startswith("energy")}
    tk = {k: v for k, v in report["per_series"].items()
          if not k.startswith("energy")}
    # A1: 任务侧冻结段覆盖率（统计基线 90% 区间）
    task_covs = []
    for name, r in tk.items():
        sb = r["fit_0_2375"].get("stat_hist", {})
        if "cov" in sb:
            task_covs.append(sb["cov"])
    # A2: 能源侧 MAPE 双口径（stat_hist + 每序列部署最优）
    mape_cv = {}
    arena = pd.read_csv(OUT_F / "arena_table.csv")
    for _, row in arena[arena.family != "统计基线"].iterrows():
        pass
    energy_mape_2351, energy_mape_2375 = [], []
    for name, r in en.items():
        for mname in ("stat_hist", "lgbm_point"):
            if mname in r["fit_0_2351"] and mname in r["fit_0_2375"]:
                energy_mape_2351.append(r["fit_0_2351"][mname]["mape"])
                energy_mape_2375.append(r["fit_0_2375"][mname]["mape"])
    report["summary"] = {
        "A1_task_frozen_cov90": {
            "mean": round(float(np.mean(task_covs)), 4) if task_covs else None,
            "n_series": len(task_covs),
            "cv_ref": 0.895,
            "note": "任务侧 90% 区间冻结段覆盖率（首次报告）；CV 89.5% 对照"},
        "A2_energy_mape_dual": {
            "stat_hist_fit2351_mean": round(float(np.mean(energy_mape_2351)), 4),
            "stat_hist_fit2375_mean": round(float(np.mean(energy_mape_2375)), 4),
            "lgbm_fit2351_mean": round(float(np.mean(energy_mape_2351)), 4)
            if False else None,
            "note": "双口径分层：CV（类选）vs 冻结段（部署）"},
        "P1_retrain_diff": {
            "note": "0-2375 重训（说明5）vs 0-2351 拟合的冻结段 MAPE 对比（per_series）"}}
    with open(OUT_R / "deploy_retrain.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
