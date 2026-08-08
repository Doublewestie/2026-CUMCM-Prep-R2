"""step0.5_soc_rebuild — 储能 SOC 递推重建（Q3 地基，R3 口径修正）.

背景（官方反馈 + 实证）:
  充放效率为分区制: A/B/C 用 0.93/0.92，D/E/F 用 0.94/0.93
  （storage_information.xlsx 原始口径；CONSTITUTION 早期"0.93/0.92 全区域"为文档错误）。
  递推验证（R3 实证）: 用分区效率按附件公式递推，A/B/C/D/F 与数据表完全吻合
  （mean|diff|<=0.005 MWh）；唯一 RegionE 数据表 = 递推 − 1.0（全程恒定偏移，
  不止 hour0）—— 官方确认为"基准数据局部录入误差/口径残差"，处理口径 =
  "以储能递推公式重新计算 SOC 状态"（官方推荐措辞，Q3 采用）。

口径: SOC(0) 前 = InitialSOC；SOC(t) = SOC(t−1) + ηc·ChargePower(t) − DischargePower(t)/ηd
      SOC_MWh 为时段末状态（storage 表 SOC_State_Convention 字段）。

产物（output/clean/）: soc_rebuilt.csv（逐区逐时 数据表/递推/差值）+
  soc_rebuild.json（验证摘要 + E 区偏移标记 + 口径声明）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step0_config import CLEAN, HOURS_TOTAL, REGIONS, SETTLE_HOUR

ETA_C = {"RegionA": 0.93, "RegionB": 0.93, "RegionC": 0.93,
         "RegionD": 0.94, "RegionE": 0.94, "RegionF": 0.94}
ETA_D = {"RegionA": 0.92, "RegionB": 0.92, "RegionC": 0.92,
         "RegionD": 0.93, "RegionE": 0.93, "RegionF": 0.93}
INIT_SOC = {"RegionA": 157.5, "RegionB": 144.0, "RegionC": 135.0,
            "RegionD": 405.0, "RegionE": 370.0, "RegionF": 382.5}
OFFSET_E = 1.0  # RegionE 数据表全程偏移（数据表 = 递推 − 1.0）


def load_storage_params() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "storage_params.csv")


def rebuild_region(rt: pd.DataFrame, r: str) -> pd.DataFrame:
    """分区效率递推重建单区 SOC（附件公式，0..2406 全时域）。"""
    sub = rt[rt["Region"] == r].sort_values("Hour")
    eta_c, eta_d = ETA_C[r], ETA_D[r]
    soc = np.zeros(len(sub))
    prev = INIT_SOC[r]
    for t in range(len(sub)):
        prev = (prev + eta_c * sub["ChargePower_MW"].values[t]
                - sub["DischargePower_MW"].values[t] / eta_d)
        soc[t] = prev
    out = pd.DataFrame({
        "Region": r,
        "Hour": sub["Hour"].values,
        "SOC_data_MWh": sub["SOC_MWh"].values,
        "SOC_rebuilt_MWh": soc,
    })
    out["diff_MWh"] = out["SOC_rebuilt_MWh"] - out["SOC_data_MWh"]
    return out


def verify_all(rt: pd.DataFrame) -> dict:
    """全区域递推-数据表一致性摘要 + E 区偏移标记。"""
    summary = {}
    for r in REGIONS:
        d = rebuild_region(rt, r)
        diff = d["diff_MWh"]
        summary[r] = {
            "mean_abs_diff": float(np.abs(diff).mean()),
            "max_abs_diff": float(np.abs(diff).max()),
            "diff_at_hour0": float(diff.iloc[0]),
            "n_hours": int(len(d)),
        }
    summary["note"] = (
        "分区效率递推（A/B/C: 0.93/0.92; D/E/F: 0.94/0.93）与数据表吻合："
        "A/B/C/D/F mean|diff|<=0.005；RegionE 全程恒定偏移 +1.0 MWh"
        "（数据表 = 递推 − 1.0，官方确认录入误差，Q3 以递推为准）")
    return summary


def main() -> None:
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    frames = [rebuild_region(rt, r) for r in REGIONS]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CLEAN / "soc_rebuilt.csv", index=False)

    summary = verify_all(rt)
    report = {
        "eta_c": ETA_C, "eta_d": ETA_D, "initial_soc": INIT_SOC,
        "per_region": summary,
        "caliber": "SOC(t)=SOC(t-1)+eta_c*ChargePower(t)-DischargePower(t)/eta_d; "
                   "SOC_MWh 为时段末；InitialSOC 为 Hour0 前（附件口径）",
        "official_feedback": "主办方确认 RegionE SOC(0) 数据表与递推差 ~1MWh 为"
                             "局部录入误差/口径残差；建议'以储能递推公式重新计算'",
        "q3_usage": "Q3 储能优化输入 = SOC_rebuilt_MWh（递推为准，E 区偏移修正）",
    }
    with open(CLEAN / "soc_rebuild.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["per_region"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
