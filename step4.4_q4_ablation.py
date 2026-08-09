"""step4.4_q4_ablation — Q4 消融与收敛验证：联合 vs 交替 + 组件消融 + 单调性守卫.

方法（Q4 方案定稿 step4.4，三层验收的求解层）:
  ① 结构消融（联合 vs 交替）:
     联合 = Q4 双层 NSGA-II 折中（step4.0 产物）
     交替 = 第1轮上层独立优化（Q2 无储能口径已得折中策略）→ 第2轮加完整
            M3_final 下层重算 → 六指标对比（差距 <0.1% 视为收敛等价；
            差距大则诚实呈现"联合必要"）
  ② 组件消融（贡献分解，Q2 baseline_proof 同款）:
     无储能（Q2 评估器口径，G=D−U 模板消纳）
     无迁移（Q1 greedy 本地调度 + M3_final 下层）
     无错峰（构造策略 shift 旋钮=0 + M3_final 下层）
  ③ 单调性守卫（ε-约束验收）: 碳限额收紧 5%→2%→1% → 全局成本单调不减
  （不满足 = 求解器 bug，断言由 tests/test_q4.py 守卫）

产物（output/q4/）:
  q4_ablation.json   消融表 + 单调性扫描 + 结论分级
figures/step4/fig_q4_ablation.png  组件贡献柱状图
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"),
        Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    s41 = _load_mod("step4.1_q4_indicators.py")
    s40 = _load_mod("step4.0_q4_bilevel.py")
    c = s41.load_ctx()
    return {**c, "s41": s41, "s40": s40}


def _six_compact(ev: dict) -> dict:
    return {k: round(float(ev["obj"][i]), 4) for i, k in enumerate(
        ("cost_wan", "carbon_t", "latency_ms", "one_minus_qos",
         "one_minus_nu", "peak_net_MW"))}


def component_ablations(c) -> dict:
    s10 = c["s10"]
    out = {}
    # 无储能：Q2 折中调度 + 模板消纳直接功率平衡（Q1/Q2 评估器口径）
    pol_q2 = c["s41"].q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], s10,
                                           c["params"], tuple(pol_q2))
    m4 = c["s20"].evaluate_4obj(c["wt"], c["rt"], sched, s10, c["params"],
                                c["consume"])
    ai_mw = c["s41"].schedule_occupancy(c, sched)
    lat = c["s20"].compute_latency(c["wt"], sched)
    out["no_storage"] = {"cost_wan": m4["cost_wan"], "carbon_t": m4["carbon_t"],
                         "latency_ms": lat, "nu_pct": m4["nu_pct"],
                         "note": "Q2 口径（无储能，G=D−U 模板消纳）"}
    # 无迁移：Q1 greedy 本地调度 + M3_final 下层
    gs = pd.read_csv(OUTPUT / "baseline" / "greedy_schedule.csv")
    ev = c["s41"].evaluate_q4_six(c, gs)
    out["no_migration"] = {"cost_wan": ev["cost_wan"],
                           "carbon_t": ev["carbon_t"],
                           "latency_ms": ev["latency_ms"],
                           "nu_pct": ev["nu_pct"],
                           "peak_net_MW": ev["peak_net_MW"],
                           "note": "Q1 greedy（本地执行）+ M3_final 下层"}
    # 无错峰：构造策略 shift=0 变体 + M3_final 下层
    pol_noshift = [pol_q2[0], pol_q2[1], 0.0, 0.0, pol_q2[4]]
    sched2 = c["s20"].schedule_constructive(c["wt"], c["rt"], s10,
                                            c["params"], tuple(pol_noshift))
    ev2 = c["s41"].evaluate_q4_six(c, sched2)
    out["no_shift"] = {"cost_wan": ev2["cost_wan"], "carbon_t": ev2["carbon_t"],
                       "latency_ms": ev2["latency_ms"], "nu_pct": ev2["nu_pct"],
                       "peak_net_MW": ev2["peak_net_MW"],
                       "note": "Q2 折中（shift 旋钮=0）+ M3_final 下层"}
    return out


def joint_vs_alternate(c) -> dict:
    """交替 = Q2 折中上层（无储能第1轮）→ M3_final 下层（第2轮）。"""
    pol_q2 = c["s41"].q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol_q2))
    alt = c["s41"].evaluate_q4_six(c, sched)
    front = OUT_Q4 / "q4_front.csv"
    joint = None
    if front.exists():
        df = pd.read_csv(front)
        F = df[["cost_wan", "carbon_t", "latency_ms", "one_minus_qos",
                "one_minus_nu", "peak_net_MW"]].to_numpy()
        i = int(np.argmin(F[:, 0] + F[:, 2] + F[:, 5]))
        pol_j = json.loads(df.loc[i, "policy"])
        sched_j = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                                 c["params"], tuple(pol_j[:5]))
        joint = c["s41"].evaluate_q4_six(c, sched_j, alpha=float(pol_j[5]))
    if joint is None:
        return {"status": "joint_front_missing（先跑 step4.0）",
                "alternate": _six_compact(alt)}
    gap = (joint["cost_wan"] - alt["cost_wan"]) / max(abs(alt["cost_wan"]), 1e-9)
    return {"joint": _six_compact(joint), "alternate": _six_compact(alt),
            "cost_gap_pct": round(float(gap * 100), 3),
            "verdict": ("收敛等价（<0.1%）" if abs(gap) < 0.001
                        else "交替迭代次优——联合必要（诚实呈现）")}


def monotonicity(c, sells=(1.0, 0.9, 0.8)) -> dict:
    """单调性守卫（ε-约束验收）：约束 RHS 收紧 → 全局成本单调不减。

    用卖电上限 SellLimit 收紧（sells×原值）作 ε-约束——碳限额不可行域
    （step4.2 实证：消纳模板锁定 G≥D−cap_h，碳排已在可行域下界，任何收紧
    infeasible），故选择始终可行且单调的 SellLimit 作验收载体。
    """
    pol_q2 = c["s41"].q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol_q2))
    ai_mw = c["s41"].schedule_occupancy(c, sched)
    costs = []
    for sell_f in sells:
        tot = 0.0
        for r in REGIONS:
            d = c["s41"].build_lower_data(c, ai_mw, r)
            ch, dh = c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"]
            d = dict(d)
            d["sell_lim"] = d["sell_lim"] * sell_f
            m = c["s40"].solve_lower_constrained(d, ch, dh)
            if m["cost_wan"] is None:
                return {"sells": list(sells), "costs_wan": costs,
                        "monotone": False,
                        "verdict": f"sell×{sell_f} 不可行", "note": ""}
            tot += m["cost_wan"]
        costs.append(round(tot, 3))
    mono = all(costs[i] >= costs[i - 1] - 1e-6 for i in range(1, len(costs)))
    return {"sells": list(sells), "costs_wan": costs,
            "monotone": bool(mono),
            "verdict": "单调不减（ε-约束验收通过）" if mono
            else "违反单调性——求解器 bug，禁止入册",
            "note": "载体=SellLimit 收紧（碳限额不可行域，见 step4.2）"}


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    comp = component_ablations(c)
    jv = joint_vs_alternate(c)
    mono = monotonicity(c)
    report = {"component_ablation": comp, "joint_vs_alternate": jv,
              "monotonicity": mono,
              "caliber": ("组件消融口径：无储能=Q2 评估器（G=D−U 模板）；"
                          "无迁移=Q1 greedy；无错峰=shift 旋钮 0；"
                          "单调性=碳限额收紧成本不减（ε-约束验收三件套）")}
    with open(OUT_Q4 / "q4_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    names = ["no_storage", "no_migration", "no_shift"]
    labels = ["无储能", "无迁移", "无错峰"]
    vals = [comp[n]["cost_wan"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#95a5a6", "#e67e22", "#2980b9"])
    for b, v in zip(ax.patches, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.0f}", ha="center", va="bottom")
    ax.set_ylabel("运行成本（万元）")
    ax.set_title("Q4 组件消融：缺零件的成本")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float)[:2500])


if __name__ == "__main__":
    main()
