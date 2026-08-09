"""step3.7_q3_cross_experiments — 交叉实验矩阵 X0-X14（深化证伪，sum_10 修正驱动）.

问题驱动的实验设计（每个实验回答一个"建模处理/指标处理"攻击点）:
  X0  M1 基线（Closure 口径修复后）——全部实验的对照
  X1  M1 + 充放功率斜坡 ≤120 MW/h（生成器量级 |ΔPc|max=121）——物理性
  X2  M1 + 仅充电斜坡 ≤120 —— 充/放拆分
  X3  M0 + 斜坡 120/120 —— 自由度红利在物理约束下的真实值
  X4  M1 + 斜坡 120 + ramp_cap=基准 —— 联合（爬坡指标物理化上界）
  X5  树规则逐时状态模拟（pc_allow/pd_allow 掩码，不聚合）—— 规则层真实损失
  X6  X5 + 放电功率 ≤182 —— 规则+功率模板
  X7  M1 + 结算段（2400-2406）禁充 —— 终态套利口径裁决
  X8  M1 + 终态 SOC(2406)=Init 严格 —— 终点套利上限
  X9  M1 + 充电时段购电充电下界=各区基准 Gc 模板 —— 弃电优先贡献量化
  X10 M1 + 放电功率 ≤182（生成器模板）—— 放电形状贡献
  X11 M1 + 午间充电允许（h10-14 扩展）—— 时段规则改进红利（M0−M1 分解关键）
  X12 M1 + 充电功率上限=生成器 24h 均值模板 —— 功率形状自由度
  X13 指标口径：主时段 0-2399（峰值/波动排除结清边界）—— 边界污染量化
  X14 X7 解 + 主时段指标（干净口径组合）

产物: output/q3/q3_cross_experiments.json + figures/step3/fig_q3_cross.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from step0_config import FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

ALL6 = REGIONS
REP3 = ["RegionA", "RegionD", "RegionE"]


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


def main_indicators(rows: list[dict]) -> dict:
    """X13/X14：主时段 0-2399 指标（排除结清边界污染）。"""
    net = np.array([x["G"] - x["S"] for x in rows if x["Hour"] < 2400])
    return {"peak_main": round(float(net.max()), 1),
            "std_main": round(float(net.std()), 1),
            "ramp_main": round(float(np.abs(np.diff(net)).max()), 1)}


def gen_24h_template(rt: pd.DataFrame, r: str, col: str) -> list[float]:
    """各区 24h 均值模板（生成器行为复现用）。"""
    sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
    v = sub[col].astype(float).values
    return [round(float(v[sub.Hour.values % 24 == h].mean()), 1)
            for h in range(24)]


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl, base = _setup()
    res = {"X0_m1_baseline": {}, "X13_main_period": {}, "experiments": {}}

    # 预生成各区模板（X9/X12）
    gc_tpl = {r: gen_24h_template(rt, r, "GridCharge_MW") for r in REGIONS}
    pc_tpl = {r: gen_24h_template(rt, r, "ChargePower_MW") for r in REGIONS}
    pd_tpl = {r: gen_24h_template(rt, r, "DischargePower_MW") for r in REGIONS}

    def solve(r, **kw):
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        return s33.solve_region_timed(d, charge_hours=ch, discharge_hours=dh,
                                      **kw)

    # X0 基线（6 区）
    x0 = {}
    for r in ALL6:
        m = solve(r)
        x0[r] = {"cost": round(m["cost_wan"], 2), "ramp": round(m["max_ramp_MW"], 1),
                 "nu": round(m["nu"] * 100, 2)}
        res["X13_main_period"][r] = main_indicators(m["rows"])
    res["X0_m1_baseline"] = x0
    print("X0 完成", flush=True)

    def run(name, regions, tag, **kw):
        out = {}
        for r in regions:
            m = solve(r, **kw)
            out[r] = {"cost": round(m["cost_wan"], 2),
                      "ramp": round(m["max_ramp_MW"], 1),
                      "nu": round(m["nu"] * 100, 2),
                      "status": m["status"]}
        res["experiments"][name] = {"tag": tag, "results": out}
        print(f"{name} 完成: { {r: out[r]['cost'] for r in out} }", flush=True)

    # X1/X2/X3/X4 斜坡组
    run("X1_ramp_slope120", ALL6, "充电+放电斜坡 ≤120",
        ramp_c_rate=120.0, ramp_d_rate=120.0)
    run("X2_ramp_charge120", REP3, "仅充电斜坡 ≤120", ramp_c_rate=120.0)
    run("X3_M0_slope120", ALL6, "M0 全自由+斜坡 120",
        ramp_c_rate=120.0, ramp_d_rate=120.0)
    run("X4_slope_rampcap", REP3, "斜坡120+爬坡≤基准",
        ramp_c_rate=120.0, ramp_d_rate=120.0,
        ramp_cap=float(base["RegionD"]["max_ramp_MW"]))

    # X5/X6 规则逐时（D/A 区，在 X0 解上训练树）
    for r in ("RegionD", "RegionA"):
        m = solve(r)
        sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
        f = pd.DataFrame({
            "hour": sub.Hour.values % 24,
            "price": sub.ElectricityPrice_CNY_per_MWh.astype(float).values,
            "W": sub.AvailableRenewable_MW.astype(float).values,
            "D": sub.Total_Load_MW.astype(float).values,
            "SOC": np.array([x["SOC"] for x in m["rows"]]),
            "Pc": np.array([x["Pc"] for x in m["rows"]]),
            "Pd": np.array([x["Pd"] for x in m["rows"]]),
        })
        y = np.where(f.Pc.values > 0.01, 1, np.where(f.Pd.values > 0.01, 2, 0))
        X = f[["hour", "price", "W", "D", "SOC"]].values
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                     random_state=42).fit(X, y)
        pred = clf.predict(X)
        pc_allow = (pred == 1).astype(int)
        pd_allow = (pred == 2).astype(int)
        acc = float(clf.score(X, y))
        # X5 逐时规则模拟
        m5 = solve(r, pc_allow=pc_allow, pd_allow=pd_allow)
        res["experiments"][f"X5_rules_hourly_{r}"] = {
            "tag": "树规则逐时状态模拟（不聚合）",
            "tree_acc": round(acc, 4),
            "n_charge_h": int(pc_allow.sum()),
            "n_discharge_h": int(pd_allow.sum()),
            "results": {r: {"cost": round(m5["cost_wan"], 2),
                            "ramp": round(m5["max_ramp_MW"], 1),
                            "nu": round(m5["nu"] * 100, 2),
                            "status": m5["status"]}}}
        # X6 X5+放电≤182
        m6 = solve(r, pc_allow=pc_allow, pd_allow=pd_allow,
                   discharge_max=float(np.max(pd_tpl[r])))
        res["experiments"][f"X6_rules_discharge_{r}"] = {
            "tag": "逐时规则+放电功率模板",
            "results": {r: {"cost": round(m6["cost_wan"], 2),
                            "ramp": round(m6["max_ramp_MW"], 1),
                            "nu": round(m6["nu"] * 100, 2),
                            "status": m6["status"]}}}
        print(f"X5/X6 {r}: 树精度 {acc:.3f} | X5 成本 {m5['cost_wan']:.1f} 万",
              flush=True)

    # X7 结算段禁充（pc_allow = 主时段且充电域）
    for r in ALL6:
        ch = tpl[r]["charge_hours"]
        pc_allow = np.array([1 if (t < 2400 and t % 24 in ch) else 0
                             for t in range(2407)])
        m = solve(r, pc_allow=pc_allow)
        res["experiments"][f"X7_no_closure_charge_{r}"] = {
            "tag": "结算段禁充",
            "results": {r: {"cost": round(m["cost_wan"], 2),
                            "ramp": round(m["max_ramp_MW"], 1),
                            "nu": round(m["nu"] * 100, 2),
                            "status": m["status"]}}}
        # X14 = X7 解 + 主时段指标
        res["experiments"][f"X14_{r}"] = {
            "tag": "结算段禁充+主时段指标",
            "main": main_indicators(m["rows"]),
            "cost": round(m["cost_wan"], 2)}
    print("X7/X14 完成", flush=True)

    # X8 终态严格
    run("X8_final_exact", REP3, "终态 SOC(2406)=Init 严格",
        final_soc_exact=True)
    # X9 购电充电下界（各区 Gc 模板）
    for r in REP3:
        ch = tpl[r]["charge_hours"]
        mn = np.array([gc_tpl[r][h] if h in ch else 0.0 for h in range(24)])
        m = solve(r, charge_min_hourly=mn)
        res["experiments"][f"X9_gridcharge_{r}"] = {
            "tag": "充电下界=基准 Gc 模板（弃电优先贡献）",
            "results": {r: {"cost": round(m["cost_wan"], 2),
                            "ramp": round(m["max_ramp_MW"], 1),
                            "nu": round(m["nu"] * 100, 2),
                            "status": m["status"]}}}
    print("X8/X9 完成", flush=True)
    # X10 放电≤182
    run("X10_discharge182", REP3, "放电功率≤基准模板 max",
        discharge_max=182.0)
    # X11 午间充电扩展
    for r in REP3:
        ch = sorted(set(tpl[r]["charge_hours"]) | {10, 11, 12, 13, 14})
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        m = s33.solve_region_timed(d, charge_hours=ch,
                                   discharge_hours=tpl[r]["discharge_hours"])
        res["experiments"][f"X11_midday_charge_{r}"] = {
            "tag": "时段规则改进：午间充电允许",
            "charge_hours": ch,
            "results": {r: {"cost": round(m["cost_wan"], 2),
                            "ramp": round(m["max_ramp_MW"], 1),
                            "nu": round(m["nu"] * 100, 2),
                            "status": m["status"]}}}
    print("X10/X11 完成", flush=True)
    # X12 充电功率上限=生成器均值模板
    for r in REP3:
        m = solve(r, charge_max_hourly=np.array(pc_tpl[r]))
        res["experiments"][f"X12_pc_template_{r}"] = {
            "tag": "充电功率上限=生成器 24h 均值模板",
            "results": {r: {"cost": round(m["cost_wan"], 2),
                            "ramp": round(m["max_ramp_MW"], 1),
                            "nu": round(m["nu"] * 100, 2),
                            "status": m["status"]}}}
    print("X12 完成", flush=True)

    res["caliber"] = ("交叉实验矩阵（深化证伪）：每个实验一个建模/指标处理变量；"
                      "X0=Closure 口径修复后 M1 基线；斜坡 120=生成器 |ΔPc|max 量级；"
                      "X5 规则逐时=树预测状态掩码（不聚合）；"
                      "X9 下界=各区 GridCharge 24h 均值模板（仅充电域）")
    with open(OUT_Q3 / "q3_cross_experiments.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # 汇总图：成本变化（相对 X0）
    fig, ax = plt.subplots(figsize=(11, 5))
    exp_names = [n for n in res["experiments"] if not n.startswith("X5")
                 and not n.startswith("X6") and not n.startswith("X14")]
    rows = []
    for n in exp_names:
        e = res["experiments"][n]
        for r, v in e["results"].items():
            rows.append({"exp": n, "region": r, "cost": v["cost"],
                         "ramp": v["ramp"], "nu": v["nu"]})
    df = pd.DataFrame(rows)
    for r, color in (("RegionD", "#4c72b0"), ("RegionA", "#c44e52"),
                     ("RegionE", "#dd8452")):
        sub = df[df.region == r]
        ax.plot(sub.exp, sub.cost, "o-", label=r, color=color, ms=4)
    ax.axhline(x0["RegionD"]["cost"], color="#4c72b0", ls="--", lw=0.7)
    ax.set_xticklabels(sub.exp, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("成本（万元）")
    ax.set_title("Q3 交叉实验矩阵：成本对比（虚线=X0 基线）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_cross.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"X13_main_period": res["X13_main_period"],
                      "X0": x0}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
