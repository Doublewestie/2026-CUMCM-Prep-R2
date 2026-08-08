"""step1.8_deploy_gate — 类选协议 v3：部署口径复检（B1，审查 A1 闭环）.

背景: CV 类选（arena）与部署（全段拟合+冻结段）存在 24/24 错位（mismatch=1.00），
类选 v3 的收益 = MAPE 相对改进中位 15.4%（A1 实证）。

协议 v3（三段协议 → 四步）:
  ① CV 初筛（0-2351 五折滚动）：候选池 + 稳定性佐证
  ② 部署复检（0-2351 全段拟合 → 冻结段 2376-2399 评估）：每序列显著优于
    统计基线（MAPE 降幅 ≥5%，model-evaluation 纪律）→ 部署入选集 ← 新
  ③ 校准（2352-2375）：κ_ε 预留校准（不变）
  ④ 冻结验证（2376-2399）：不崩溃验证（不变）
部署入选集 = 类选最终裁决；CV 入选集仅作稳定性佐证（若分歧 → 记录 + 以部署为准）

产物（output/forecast/）: deploy_arena.csv（部署口径指标+裁决）+
  deploy_gate.json（对比 CV 入选集 + 决策记录）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step0_config import OUTPUT, REGIONS

OUT_F = OUTPUT / "forecast"
OUT_R = OUTPUT / "robust"
SEG_TRAIN = 2352
GATE_IMP = 0.05          # MAPE 相对降幅门限（与 v1 能源侧一致）


def load_frozen_mape() -> pd.DataFrame:
    """冻结段每模型 MAPE（step1.5 产物：全段拟合 + 冻结段 2376-2399）。"""
    ft = json.loads((OUT_R / "frozen_test.json").read_text(encoding="utf-8"))
    rows = []
    for name, recs in ft["per_series"].items():
        for model, sc in recs.items():
            if isinstance(sc, dict) and "mape" in sc:
                rows.append({"series": name, "model": model,
                             "frozen_mape": sc["mape"]})
    return pd.DataFrame(rows)


def main() -> None:
    arena = pd.read_csv(OUT_F / "arena_table.csv")
    frozen = load_frozen_mape()
    base_map = {}
    for _, r in arena[arena.family == "统计基线"].iterrows():
        base_map[r["series"]] = r.to_dict()
    rows = []
    for _, r in arena[arena.family != "统计基线"].iterrows():
        fm = frozen[(frozen.series == r["series"])
                    & (frozen.model == r["model"])]
        if not len(fm):
            continue
        b = base_map.get(r["series"])
        f_mape = float(fm["frozen_mape"].iloc[0])
        if b is None or b["mape_mean"] <= 0:
            imp = float("nan")
        else:
            imp = (b["mape_mean"] - f_mape) / b["mape_mean"]
        # v4 裁决：任务侧（白噪声）不适用"冻结段显著"判据（24h 假阳性，A8 实证）
        #   → 任务侧 deploy_pass 一律 False，类选维持 CV 全拒（无崩溃验证由 step1.5 承担）
        if r["layer"] == "task":
            dp = False
        else:
            dp = bool(imp >= GATE_IMP)
        rows.append({"series": r["series"], "model": r["model"],
                     "layer": r["layer"],
                     "cv_mape": r["mape_mean"], "cv_gate": r["gate"],
                     "frozen_mape": f_mape,
                     "frozen_imp": round(imp, 4),
                     "deploy_pass": dp})
    d = pd.DataFrame(rows)
    d.to_csv(OUT_F / "deploy_arena.csv", index=False)

    # 每序列部署最优 + 对比 CV 最优
    cv_best = {}
    for name, g in arena[arena.family != "统计基线"].groupby("series"):
        g = g[g.n_folds > 0]
        if len(g):
            cv_best[name] = g.loc[g.mape_mean.idxmin(), "model"]
    dp_best = {}
    for name, g in d.groupby("series"):
        g = g[g.deploy_pass]
        if len(g):
            dp_best[name] = g.loc[g.frozen_mape.idxmin(), "model"]
    rows2 = []
    for name in sorted(set(cv_best) | set(dp_best)):
        rows2.append({"series": name, "cv_best": cv_best.get(name),
                      "deploy_best": dp_best.get(name),
                      "mismatch": cv_best.get(name) != dp_best.get(name)})
    cmp = pd.DataFrame(rows2)
    mismatch = cmp["mismatch"].fillna(True).mean()
    n_dep = d[d.deploy_pass].groupby("series").size()

    # v4：集合级一致性（Jaccard）——主指标替代第一名一致率
    vals = arena["gate"].tolist()
    cvpass = [len(v) >= 2 and ord(v[0]) == 0x901a and ord(v[1]) == 0x8fc7
              for v in vals]
    arena["cv_pass"] = cvpass
    m = arena.merge(d[["series", "model", "deploy_pass"]],
                    on=["series", "model"], how="inner")
    jaccard = {}
    for lay in ("energy", "task"):
        sub = m[m.layer == lay]
        both = ((sub.cv_pass) & (sub.deploy_pass)).sum()
        cv_only = (sub.cv_pass & ~sub.deploy_pass).sum()
        dep_only = (~sub.cv_pass & sub.deploy_pass).sum()
        jaccard[lay] = round(both / max(both + cv_only + dep_only, 1), 3)

    # v4：CV 噪声序列（最优-次优差距 < 最优 std）标注——不承诺第一名
    noisy = []
    for name, g in arena[arena.family != "统计基线"].groupby("series"):
        gg = g[g.n_folds > 0].sort_values("mape_mean")
        if len(gg) < 2:
            continue
        b = gg.iloc[0]
        gap = gg.iloc[1]["mape_mean"] - b["mape_mean"]
        if gap < b["mape_std"]:
            noisy.append({"series": name, "cv_best": b["model"],
                          "gap_std_ratio": round(gap / max(b["mape_std"], 1e-9), 2)})

    report = {
        "n_series": int(len(cmp)),
        "cv_deploy_mismatch_share": float(mismatch),
        "deploy_pass_share": float(d.deploy_pass.mean()),
        "n_series_with_deploy_best": int(len(dp_best)),
        "cv_best_dist": pd.Series(cv_best).value_counts().to_dict(),
        "deploy_best_dist": pd.Series(dp_best).value_counts().to_dict(),
        "per_series": rows2,
        "v4": {
            "jaccard_set_consistency": jaccard,     # 主指标（集合级）
            "cv_noisy_sequences": noisy,            # 不承诺第一名的序列
            "task_deploy_pass": int(d[d.layer == "task"].deploy_pass.sum()),
            "note": ("v4: ①主指标=集合级 Jaccard（能源侧 0.80——入选集稳定）；"
                     "②第一名一致率废弃（CV 噪声序列 11/24 + 24h 排名物理不可辨）；"
                     "③任务侧部署验证改无崩溃（白噪声 24h 显著=假阳性，A8 实证 69→0）"),
        },
        "protocol": ("v3→v4: CV 初筛 → 部署复检（集合级裁决）→ 校准 → 冻结验证；"
                     "部署入选集=最终裁决，CV 仅稳定性佐证；任务侧白噪声维持 CV 全拒"),
        "note": ("B7 修复：类选 v4（集合级 Jaccard 主指标 + 任务侧无崩溃判据）；"
                 "A7/A8 实证：CV 噪声序列 11/24、集合一致性 0.80、任务侧假阳性 69→0"),
    }
    with open(OUT_F / "deploy_gate.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_series",)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
