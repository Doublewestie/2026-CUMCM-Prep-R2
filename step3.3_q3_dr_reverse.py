"""step3.3_q3_dr_reverse — 轨A：DR 储能规则全谱逆向（E-DR1，spec_M4_Q3 D-4+ 修正）.

背景（绝对客观自检发现，sum_9 T7 结案）:
  生成器储能行为 = DR 状态机 × 时段模板 × 来源模板（与 R1 消纳模板同构）:
  - D 区: DR=Medium ⟺ Peak(h17-21) 放电域 500h；DR=Low = Valley+Flat 充电域；
    充电仅 h0-4+h22-23（白天 h5-16 即使 Low 也不充）；充放互斥；从不顶格
  - 区域异构: A/B/C 区 DR=Medium 76%（覆盖全天）；E 区有早峰 h4-9 ——
    状态机约束不能全域统一套用（M4/M6/M7 实证）

本文件逆向全谱并裁决机理问题 M1-M10，产出:
  output/q3/q3_dr_reverse.json   每区机理裁决表 + 充电/放电允许时段模板（M1 建模输入）
  figures/step3/fig_q3_dr_template.png  每区 24h 充放/DR 模板热力对比
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

H = 24
N_DAY = 100  # 2407h = 100×24 + 7


def load_rt() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "region_time_clean.csv")


def _sub(rt: pd.DataFrame, r: str) -> pd.DataFrame:
    return rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)


def dr_template_test(sub: pd.DataFrame) -> dict:
    """M5：DR 是否确定性日模板（前 2400h 逐日重复？）。"""
    dr = sub["DemandResponseLevel"].astype(str).to_numpy()[:2400]
    dr24 = dr.reshape(N_DAY, 24)
    first = dr24[0]
    match = (dr24 == first).mean(axis=0)          # 每小时的跨日一致率
    day_consistent = float((dr24[1:] == dr24[:-1]).all(axis=1).mean())
    return {
        "is_daily_template": float(match.mean()),
        "day_to_day_consistent": day_consistent,
        "hour_consistency_min": float(match.min()),
        "conclusion": ("【实证】确定性日模板" if match.mean() > 0.999
                       else "【实证】非纯日模板，存在跨日变异"),
    }


def charge_discharge_template(sub: pd.DataFrame) -> dict:
    """M1/M8：充放电时段模板（Pc>0 / Pd>0 的小时集合跨天稳定？）+ 功率上限。"""
    pc = sub["ChargePower_MW"].astype(float).to_numpy()[:2400].reshape(N_DAY, 24)
    pdv = sub["DischargePower_MW"].astype(float).to_numpy()[:2400].reshape(N_DAY, 24)
    pc_on = (pc > 0.01).mean(axis=0)              # 每小时充电活跃率
    pd_on = (pdv > 0.01).mean(axis=0)
    charge_hours = sorted(int(h) for h in np.where(pc_on > 0.9)[0])
    discharge_hours = sorted(int(h) for h in np.where(pd_on > 0.9)[0])
    return {
        "charge_hours": charge_hours,
        "charge_on_rate": {int(h): round(float(pc_on[h]), 3) for h in range(24)},
        "discharge_hours": discharge_hours,
        "discharge_on_rate": {int(h): round(float(pd_on[h]), 3) for h in range(24)},
        "max_charge_MW": round(float(pc.max()), 1),
        "max_discharge_MW": round(float(pdv.max()), 1),
        "mutual_exclusive": bool(not ((pc > 0.01) & (pdv > 0.01)).any()),
    }


def rc_template(sub: pd.DataFrame) -> dict:
    """M3/M10：RenewableCharge 固定模板（跨天 std≈0？）+ 弃电充电触发时段。"""
    rc = sub["RenewableCharge_MW"].astype(float).to_numpy()[:2400].reshape(N_DAY, 24)
    gc = sub["GridCharge_MW"].astype(float).to_numpy()[:2400].reshape(N_DAY, 24)
    rc_std = rc.std(axis=0)
    return {
        "rc_hourly_mean": {int(h): round(float(rc[:, h].mean()), 2)
                           for h in range(24)},
        "rc_hourly_std_max": round(float(rc_std.max()), 3),
        "rc_is_template": bool(rc_std.max() < 0.05),
        "gc_hourly_mean": {int(h): round(float(gc[:, h].mean()), 2)
                           for h in range(24)},
    }


def soc_cycle(sub: pd.DataFrame) -> dict:
    """M9：SOC 日循环（每日 min/max/循环深度，跨天稳定？）。"""
    soc = sub["SOC_MWh"].astype(float).to_numpy()[:2400].reshape(N_DAY, 24)
    day_min = soc.min(axis=1)
    day_max = soc.max(axis=1)
    depth = day_max - day_min
    return {
        "daily_min_mean": round(float(day_min.mean()), 1),
        "daily_max_mean": round(float(day_max.mean()), 1),
        "daily_depth_mean": round(float(depth.mean()), 1),
        "depth_std": round(float(depth.std()), 1),
        "cycle_is_template": bool(depth.std() < 1.0),
    }


def region_semantics(rt: pd.DataFrame, r: str) -> dict:
    """M4/M6/M7：DR 状态 × 价格/负荷/充放的区域语义。"""
    sub = _sub(rt, r)
    d = pd.DataFrame({
        "DR": sub["DemandResponseLevel"].astype(str).values,
        "price": sub["ElectricityPrice_CNY_per_MWh"].astype(float).values,
        "load": sub["Total_Load_MW"].astype(float).values,
        "pc": sub["ChargePower_MW"].astype(float).values,
        "pd": sub["DischargePower_MW"].astype(float).values,
    })
    g = d.groupby("DR").agg(price=("price", "mean"), load=("load", "mean"),
                            pc=("pc", "mean"), pd_=("pd", "mean"),
                            n=("price", "size"))
    med_share = float(g.loc["Medium", "n"] / len(d))
    return {
        "medium_share": round(med_share, 4),
        "per_dr": {k: {kk: round(float(vv), 1) for kk, vv in v.items()}
                   for k, v in g.iterrows()},
        "note": ("D 区 Medium=Peak 严格对应；A/B/C 区 Medium 覆盖全天（高负荷域）；"
                 "E 区 Medium 含早峰 h4-9 —— 区域异构，M1 约束按分区实证"),
    }


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    rt = load_rt()
    report = {"mechanism": {}, "templates": {}}
    for r in REGIONS:
        sub = _sub(rt, r)
        m5 = dr_template_test(sub)
        m1 = charge_discharge_template(sub)
        m3 = rc_template(sub)
        m9 = soc_cycle(sub)
        rs = region_semantics(rt, r)
        report["mechanism"][r] = {
            "M5_dr_template": m5,
            "M1_M8_charge_discharge": m1,
            "M3_M10_renewable_charge": m3,
            "M9_soc_cycle": m9,
            "M4_M6_M7_region_semantics": rs,
        }
        report["templates"][r] = {
            "charge_hours": m1["charge_hours"],
            "discharge_hours": m1["discharge_hours"],
            "max_charge_MW": m1["max_charge_MW"],
            "max_discharge_MW": m1["max_discharge_MW"],
            "mutual_exclusive": m1["mutual_exclusive"],
        }
        print(f"[{r}] M5 模板一致率 {m5['is_daily_template']:.3f} | "
              f"充电时段 {m1['charge_hours']} | 放电时段 {m1['discharge_hours']} | "
              f"maxPc {m1['max_charge_MW']} maxPd {m1['max_discharge_MW']} | "
              f"互斥 {m1['mutual_exclusive']}", flush=True)

    # 热力图：每区 24h 充放活跃率 + DR=Medium 占比
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, r in enumerate(REGIONS):
        ax = axes[i // 2, i % 2]
        m1 = report["mechanism"][r]["M1_M8_charge_discharge"]
        m5 = report["mechanism"][r]["M5_dr_template"]
        x = np.arange(24)
        ax.bar(x - 0.25, [m1["charge_on_rate"][h] for h in x], 0.25,
               label="充电活跃率")
        ax.bar(x, [m1["discharge_on_rate"][h] for h in x], 0.25,
               label="放电活跃率")
        dr = np.array([int(v > 0.5) for v in
                       [report["mechanism"][r]["M4_M6_M7_region_semantics"]
                        ["per_dr"].get("Medium", {}).get("n", 0)]])
        ax.set_title(f"{r}（DR 日模板一致率 {m5['is_daily_template']:.2f}）")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=7)
    fig.suptitle("Q3 生成器储能时段模板（24h 活跃率，轨A 逆向）")
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_dr_template.png", bbox_inches="tight")
    plt.close(fig)

    report["caliber"] = ("轨A E-DR1 全谱逆向；充放电时段模板=活跃率>0.9 的小时集；"
                         "M1 建模输入=templates；机理分级见 mechanism")
    with open(OUT_Q3 / "q3_dr_reverse.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({r: report["templates"][r] for r in REGIONS},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
