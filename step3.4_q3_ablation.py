"""step3.4_q3_ablation — 判别实验群 E4/E5/E7/E8/E9（基于 M1 主模型）.

E4 充电来源竞争: M1 充电全弃电（Pc_grid=0 已证）→ 反事实：W×0.5 时
  是否出现购电充电（弃电不足→购电通道激活阈值）
E5 活跃约束钳制: M1 解中 SellLimit 饱和率（T2）/SOC 顶 Cap 率（T3）对偶验证
E7 储能价值反证: SellLimit=0（去外送）→ M1 成本变化 = 外送套利价值
E8 敏感性+扩容: Cap/价格水平 ±10% 门槛扰动；Cap ×1.1/1.2/1.3 扩容价值曲线
E9 充电上限反证: charge_max ×2 → 价值增量（东区充电瓶颈结构性检验）

产物（output/q3/）: q3_ablation.json + figures/step3/fig_q3_cap_scale.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step0_config import FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"


def _setup():
    root = Path(__file__).resolve().parent
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s33p", root / "step3.3+_q3_model_evolve.py")
    s33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s33)
    spec2 = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s10)
    rt = s33._import_step32().load_rt()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    tpl = json.loads((OUT_Q3 / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"]
    base = s33._import_step32().baseline_indicators(rt)
    return s33, rt, consume, tpl, base


def solve_m1(s33, d, ch, dh, **kw) -> dict:
    """M3_final 主模型求解（sum_10 回灌：判别实验基于主口径）。"""
    region = d.get("_region", None)
    m = s33.solve_m3(d, ch, dh, region=region)
    if m["status"] != 0:
        # 回退 M1（若主模型不可行）
        return s33.solve_region_timed(d, charge_hours=ch, discharge_hours=dh,
                                      **kw)
    return m


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl, base = _setup()
    report = {}

    # 各判别实验（全区域 E5/E8；代表性区 A/D/E 做 E4/E7/E9）
    for r in REGIONS:
        d = s33._load_region_data(r)
        d["_region"] = r
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        base_pt = solve_m1(s33, d, ch, dh)
        # E5 活跃约束钳制（M1 解统计）
        rows = base_pt["rows"]
        soc = np.array([x["SOC"] for x in rows])
        s_ = np.array([x["S"] for x in rows])
        sell_lim = d["sell_lim"]
        sat_sell = float((s_ > sell_lim * 0.999).mean()) if sell_lim > 0 else 0.0
        sat_soc = float((soc > d["cap_mwh"] * 0.999).mean())
        touch_min = float((soc < d["min_soc"] * 1.001).mean())
        e5 = {"sell_saturation": round(sat_sell, 4),
              "soc_cap_saturation": round(sat_soc, 4),
              "soc_min_touch": round(touch_min, 4)}
        # E8 敏感性 ±10%（Cap / 价格水平）+ 扩容 ×1.1/1.2/1.3
        sens = {}
        d2 = dict(d)
        d2["cap_mwh"] = d["cap_mwh"] * 0.9
        d2["min_soc"] = d["min_soc"] * 0.9
        p = solve_m1(s33, d2, ch, dh)
        sens["cap_0.9x"] = round(p["cost_wan"], 2)
        d2 = dict(d)
        d2["cap_mwh"] = d["cap_mwh"] * 1.1
        d2["min_soc"] = d["min_soc"] * 1.1
        p = solve_m1(s33, d2, ch, dh)
        sens["cap_1.1x"] = round(p["cost_wan"], 2)
        d2 = dict(d)
        d2["price"] = d["price"] * 1.1
        p = solve_m1(s33, d2, ch, dh)
        sens["price_1.1x"] = round(p["cost_wan"], 2)
        d2 = dict(d)
        d2["price"] = d["price"] * 0.9
        p = solve_m1(s33, d2, ch, dh)
        sens["price_0.9x"] = round(p["cost_wan"], 2)
        scale_curve = {}
        for k in (1.1, 1.2, 1.3):
            d2 = dict(d)
            d2["cap_mwh"] = d["cap_mwh"] * k
            d2["min_soc"] = d["min_soc"] * k
            p = solve_m1(s33, d2, ch, dh)
            scale_curve[f"{k}x"] = round(p["cost_wan"], 2)
        report[r] = {
            "base_cost_wan": round(base_pt["cost_wan"], 2),
            "E5_active_constraints": e5,
            "E8_sensitivity": sens,
            "E8_cap_scale_curve": scale_curve,
            "E8_cap_scale_gain_wan": round(
                base_pt["cost_wan"] - scale_curve["1.3x"], 2),
        }
        print(f"[{r}] E5 售电饱和={sat_sell:.2f} SOC顶Cap={sat_soc:.2f} "
              f"| E8 cap1.3x增益={report[r]['E8_cap_scale_gain_wan']}万",
              flush=True)

    # E4（D 区）：W×0.5 反事实 → 购电充电激活？
    d = s33._load_region_data("RegionD")
    d["_region"] = "RegionD"
    d["c_h"] = np.asarray(consume["RegionD"], dtype=float)
    ch, dh = tpl["RegionD"]["charge_hours"], tpl["RegionD"]["discharge_hours"]
    d2 = dict(d)
    d2["W"] = d["W"] * 0.5
    p = solve_m1(s33, d2, ch, dh)
    src = s33.charge_source_decomposition(d2, p["rows"])
    pg = float(sum(x["Pc_grid"] for x in src))
    pr = float(sum(x["Pc_renewable"] for x in src))
    report["E4_charge_source_competition"] = {
        "normal_W": {"Pc_grid_MWh": 0.0, "note": "弃电优先，购电通道关闭（已证）"},
        "W_half_scenario": {"Pc_grid_MWh": round(pg, 1),
                            "Pc_renewable_MWh": round(pr, 1),
                            "cost_wan": round(p["cost_wan"], 2),
                            "verdict": ("弃电不足→购电充电激活" if pg > 1.0
                                        else "仍全弃电")}}
    print(f"E4 W×0.5: 购电充电 {pg:.0f} MWh（正常=0）", flush=True)

    # E7（D/E 区）：SellLimit=0 反证
    e7 = {}
    for r in ("RegionD", "RegionE"):
        d = s33._load_region_data(r)
        d["_region"] = r
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        base_pt = solve_m1(s33, d, ch, dh)
        d2 = dict(d)
        d2["sell_lim"] = 0.0
        p = solve_m1(s33, d2, ch, dh)
        e7[r] = {"cost_with_sell": round(base_pt["cost_wan"], 2),
                 "cost_no_sell": round(p["cost_wan"], 2),
                 "sell_value_wan": round(p["cost_wan"] - base_pt["cost_wan"], 2)}
        print(f"E7 {r}: 外送价值 {e7[r]['sell_value_wan']} 万", flush=True)
    report["E7_sell_removal"] = e7

    # E9（A/D 区）：充电上限 ×2
    e9 = {}
    for r in ("RegionA", "RegionD"):
        d = s33._load_region_data(r)
        d["_region"] = r
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        base_pt = solve_m1(s33, d, ch, dh)
        p = solve_m1(s33, d, ch, dh, charge_max=2 * d["max_c"])
        e9[r] = {"cost_base": round(base_pt["cost_wan"], 2),
                 "cost_charge2x": round(p["cost_wan"], 2),
                 "gain_wan": round(base_pt["cost_wan"] - p["cost_wan"], 2),
                 "verdict": ("充电上限非瓶颈" if abs(p["cost_wan"]
                                                    - base_pt["cost_wan"]) < 1
                             else "充电上限=结构瓶颈")}
        print(f"E9 {r}: 充电×2 增益 {e9[r]['gain_wan']} 万", flush=True)
    report["E9_charge_cap"] = e9

    # Cap 扩容曲线图（D/A）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, color in (("RegionD", "#4c72b0"), ("RegionA", "#c44e52")):
        c = report[r]["E8_cap_scale_curve"]
        xs = [1.0] + [float(k[:-1]) for k in c]
        ys = [report[r]["base_cost_wan"]] + list(c.values())
        ax.plot(xs, ys, "o-", label=r, color=color)
    ax.set_xlabel("Cap 扩容倍数")
    ax.set_ylabel("M1 成本（万元）")
    ax.set_title("储能扩容价值曲线（E8 应用闭环）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_cap_scale.png", bbox_inches="tight")
    plt.close(fig)

    report["caliber"] = ("判别实验群基于 M1（时段状态机约束）；"
                         "E8 扩容时 MinSOC 同比例缩放；E4 反事实 W×0.5；"
                         "E7 外送归零；E9 充电上限×2")
    with open(OUT_Q3 / "q3_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
