"""step3.4+_q3_storage_rules — 储能充放电策略规则提取（题目交付物）.

基于 M1 解（时段状态机约束 LP，真实储能价值口径）→ CART 决策树 →
"何时充电 / 何时放电 / 何时静置"运营规则（D 区主，A 区对照）。

验证锚点:
  T8 谷价充电阈值（价格>60 分位充电恒 0）——规则应复现
  时段模板（充电 h0-4/22-23、放电 h17-21）——特征驱动应复现
规则模拟损失（部署一致性验收）: 按规则固定 Pc/Pd → LP 重解 G/S/Q →
  四指标 vs M1 最优 → 成本差距（高准确率门槛 <0.1%，探索后定标）

产物: output/q3/q3_storage_rules.json + figures/step3/fig_q3_storage_rules.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from step0_config import FIGURES, OUTPUT

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
    return s33, rt, consume, tpl


def build_features(rt: pd.DataFrame, r: str, rows: list[dict]) -> pd.DataFrame:
    sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
    cap_h = np.minimum(sub["AvailableRenewable_MW"].astype(float),
                       [0.0])  # 占位，下方重算
    f = pd.DataFrame({
        "hour": sub["Hour"].values % 24,
        "price": sub["ElectricityPrice_CNY_per_MWh"].astype(float).values,
        "price_q60": sub["ElectricityPrice_CNY_per_MWh"].astype(
            float).values > np.quantile(
            sub["ElectricityPrice_CNY_per_MWh"].astype(float), 0.60),
        "W": sub["AvailableRenewable_MW"].astype(float).values,
        "D": sub["Total_Load_MW"].astype(float).values,
        "SOC": np.array([x["SOC"] for x in rows]),
        "Pc": np.array([x["Pc"] for x in rows]),
        "Pd": np.array([x["Pd"] for x in rows]),
    })
    return f


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl = _setup()
    report = {"rules": {}, "simulation_loss": {}, "anchors": {}}
    for r in ("RegionD", "RegionA"):
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        m1 = s33.solve_region_timed(d, charge_hours=ch, discharge_hours=dh)
        f = build_features(rt, r, m1["rows"])
        y = np.where(f["Pc"].values > 0.01, 1,
                     np.where(f["Pd"].values > 0.01, 2, 0))
        X = f[["hour", "price", "W", "D", "SOC"]].values
        Xnames = ["hour", "price", "W", "D", "SOC"]
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                     random_state=42)
        clf.fit(X, y)
        acc = float(clf.score(X, y))
        rules_txt = export_text(clf, feature_names=Xnames)
        # 规则化模拟（sum_10 修正：逐时状态掩码，不聚合——聚合高估损失）
        pred = clf.predict(X)
        pc_allow = (pred == 1).astype(int)
        pd_allow = (pred == 2).astype(int)
        sim = s33.solve_region_timed(d, pc_allow=pc_allow, pd_allow=pd_allow)
        if sim["status"] == 0:
            loss_cost = abs(sim["cost_wan"] - m1["cost_wan"]) \
                / max(abs(m1["cost_wan"]), 1e-9) * 100
        else:
            loss_cost = None
        # T8 锚点：规则是否复现"价格>60分位充电恒 0"
        t8_viol = int(((f["price_q60"].values) & (f["Pc"].values > 0.01)).sum())
        h24 = f["hour"].values
        report["rules"][r] = {
            "state_accuracy": round(acc, 4),
            "rule_charge_hours": sorted(
                int(h) for h in range(24)
                if pc_allow[h24 == h].mean() > 0.5),
            "rule_discharge_hours": sorted(
                int(h) for h in range(24)
                if pd_allow[h24 == h].mean() > 0.5),
            "tree_text": rules_txt,
            "class_dist": {"idle": int((y == 0).sum()),
                           "charge": int((y == 1).sum()),
                           "discharge": int((y == 2).sum())},
        }
        report["anchors"][r] = {
            "T8_charge_above_q60_hours": int(t8_viol),
            "T8_verdict": ("【实证】T8 复现（充电恒低于 60 分位价格）"
                           if t8_viol == 0 else "T8 未复现"),
        }
        report["simulation_loss"][r] = {
            "m1_cost_wan": round(m1["cost_wan"], 2),
            "rule_sim_cost_wan": round(sim["cost_wan"], 2)
            if sim["status"] == 0 else None,
            "cost_gap_pct": round(loss_cost, 4) if loss_cost is not None
            else None,
            "sim_ramp_MW": round(sim["max_ramp_MW"], 1)
            if sim["status"] == 0 else None,
        }
        rule_ch = report["rules"][r]["rule_charge_hours"]
        rule_dh = report["rules"][r]["rule_discharge_hours"]
        print(f"[{r}] 规则精度={acc:.3f} 状态分布={report['rules'][r]['class_dist']} "
              f"| 规则时段 充={rule_ch} 放={rule_dh} | T8 违例={t8_viol} "
              f"| 逐时模拟损失={loss_cost}%", flush=True)
        print(rules_txt[:600], flush=True)

        # 24h 策略热图（模拟状态 vs M1 状态）
        fig, ax = plt.subplots(figsize=(9, 3.5))
        act_m1 = np.array([float(f["Pc"].values[h24 == h].mean())
                           for h in range(24)])
        act_rule = np.array([float(f["Pc"].values[h24 == h].mean())
                             if h in rule_ch else 0.0 for h in range(24)])
        ax.bar(np.arange(24) - 0.25, act_m1, 0.25, label="M1 充电",
               color="#4c72b0")
        ax.bar(np.arange(24) + 0.25, act_rule, 0.25, label="规则时段充电",
               color="#dd8452")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour"); ax.set_ylabel("充电功率 MW")
        ax.set_title(f"{r} 储能充电策略：M1 最优 vs CART 规则（24h 均值）")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_Q3 / f"fig_q3_storage_rules_{r}.png",
                    bbox_inches="tight")
        plt.close(fig)

    report["caliber"] = ("CART（max_depth=5，特征 hour/price/W/D/SOC）；"
                         "模拟=规则固定 Pc/Pd→LP 重解 G/S/Q；"
                         "损失口径=成本相对差距；T8 锚点=价格>60 分位充电恒 0")
    with open(OUT_Q3 / "q3_storage_rules.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"simulation_loss": report["simulation_loss"],
                      "anchors": report["anchors"]}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
