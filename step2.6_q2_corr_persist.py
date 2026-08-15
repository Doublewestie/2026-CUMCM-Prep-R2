"""step2.6_q2_corr_persist — E-F3 目标退化实证补算落盘（素材核查 #3 修复）.

背景: sum_6 E-F3 的"价格-碳同序 0.952 / 前沿相关矩阵 / 箱内 spread"
     原始分析未落盘（素材 3-3 声称映射 review_fix.json 但无此数据）。
     本脚本按原口径复算并追加写入 review_fix.json["B11_corr_matrix"]，
     不修改既有字段（A3/A4/A5/C5/C2）。

口径（与 sum_6 E-F3 一致）:
- region_price_carbon: 六区域 ElectricityPrice/CarbonIntensity 均值 → Pearson
  （区域级"价格-碳同序"；复算 0.9524 ≈ 0.952 ✓）
- corr_matrix: output/q2/nsga2_front.csv 四目标 Pearson
  （cost-carbon 0.974 / NU-carbon -0.987 / NU-cost -0.926 ✓）
- bin_spread: 按成本分 10 箱，箱内 carbon_t std 中位数 / 全局 std
  （"相关性高≠确定性共线"的判别——箱内零 spread 则共线实锤）

产物: output/q2/review_fix.json（追加 B11_corr_matrix 字段）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step0_config import CLEAN, OUTPUT

OUT_Q2 = OUTPUT / "q2"
N_BINS = 10


def main() -> None:
    # 1) 前沿解相关矩阵（Q2 四目标，nsga2_front.csv）
    front = pd.read_csv(OUT_Q2 / "nsga2_front.csv")
    obj = front[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]]
    corr = obj.corr().round(3)

    # 2) 区域级价格-碳同序（数据层，region_time_clean.csv）
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    g = rt.groupby("Region")[
        ["ElectricityPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh"]].mean()
    price_carbon_r = round(
        float(g["ElectricityPrice_CNY_per_MWh"].corr(
            g["CarbonIntensity_tCO2_per_MWh"])), 4)

    # 3) 箱内 spread（成本分箱 → 箱内 carbon_t std 比值）
    x = obj["cost_wan"].to_numpy()
    y = obj["carbon_t"].to_numpy()
    bins = pd.qcut(x, N_BINS, duplicates="drop")
    bin_std = pd.Series(y).groupby(bins).std().dropna()
    global_std = float(y.std())
    bin_ratio = round(float(bin_std.median() / global_std), 4)

    new = {
        "region_price_carbon_pearson": price_carbon_r,
        "region_means": {
            r: {"price": round(float(row["ElectricityPrice_CNY_per_MWh"]), 1),
                "carbon": round(float(row["CarbonIntensity_tCO2_per_MWh"]), 3)}
            for r, row in g.iterrows()},
        "corr_matrix": corr.to_dict(),
        "bin_spread": {
            "n_bins": int(len(bin_std)),
            "median_bin_carbon_std_ratio": bin_ratio,
            "note": "成本分箱内碳排 std 中位 / 全局 std——"
                    "箱内零 spread 证明相关性=共线而非相关漂移"},
        "note": "E-F3 补算落盘（素材 3-3 核查 #3 修复）："
                "区域级 price-carbon Pearson 0.9524≈0.952；"
                "前沿相关 cost-carbon 0.974/NU-carbon -0.987/NU-cost -0.926——"
                "四目标实际 2 独立维度（成本-碳-利用率共线 + 时延）",
    }

    path = OUT_Q2 / "review_fix.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["B11_corr_matrix"] = new
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"review_fix.json += B11_corr_matrix（price-carbon r={price_carbon_r}, "
          f"bin_std_ratio={bin_ratio}, corr 见文件）")


if __name__ == "__main__":
    main()
