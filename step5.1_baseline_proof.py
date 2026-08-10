"""step5.1_baseline_proof — 全题改进率汇总表（论文核心数字表）.

汇总四问的基线锚与改进率（口径声明见 08_口径声明大全）：
  Q1: 附件基线（弃电/超容/利用率）
  Q2: 附件基线 vs 三段式调度（成本/碳/时延）
  Q3: 生成器基准 vs M3_final + 无储能 LP 双锚（储能价值/生成器次优）
  Q4: Q2 折中×M3_final 基线 vs Q4 折中（六指标）
输出: output/q5/baseline_proof_all.json（或 output/baseline_proof_all.json）
"""
import json
from pathlib import Path

import pandas as pd

from step0_config import OUTPUT

OUT = OUTPUT / "q5"
OUT.mkdir(parents=True, exist_ok=True)


def load(rel):
    p = Path(rel)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    report = {"Q1": {}, "Q2": {}, "Q3": {}, "Q4": {}}

    # Q1 基线（baseline_metrics.local——附件基线本地执行）
    bm = load("output/baseline/baseline_metrics.json")
    if bm and "local" in bm:
        loc = bm["local"]
        report["Q1"] = {
            "cost_wan": loc.get("cost_wan"),
            "carbon_t": loc.get("carbon_t"),
            "curtail_MWh": loc.get("curtail_MWh"),
            "utilization_pct": loc.get("nu_pct"),
            "overcap_hours": loc.get("viol_h"),
            "overcap_by_region": loc.get("viol_by_region"),
            "note": ("附件基线（本地执行，模板消纳口径）——弃电 "
                     f"{loc.get('curtail_MWh', 0)/1e4:.1f} 万 MWh/"
                     f"利用率 {loc.get('nu_pct', 0):.1f}%/超容 "
                     f"{loc.get('viol_h', 0)}h（E 30/F 67）"),
        }

    # Q2（baseline_proof.q2_compromise_vs_local）
    q2 = load("output/q2/baseline_proof.json")
    if q2 and "q2_compromise_vs_local" in q2:
        v = q2["q2_compromise_vs_local"]
        report["Q2"] = {
            "cost_improve_pct": v.get("cost_pct"),
            "carbon_improve_pct": v.get("carbon_pct"),
            "latency_ratio": v.get("latency_ratio"),
            "nu_pp": v.get("nu_pp"),
            "q2_cost_wan": q2.get("q2_compromise", {}).get("cost_wan"),
            "note": "附件基线 vs Q2 三段式（模板消纳口径）；五方法端点共识",
        }

    # Q3（双锚，LP 对照净化口径）
    rigor = load("output/q3/q3_rigor.json")
    q3i = load("output/q3/q3_indicators.json")
    if rigor and q3i:
        regs = {}
        for r in rigor["regions"]:
            dec = rigor["regions"][r]["decomposition"]
            regs[r] = {
                "storage_value_wan": dec["storage_value_wan"],
                "storage_value_pct": dec["storage_value_pct"],
                "gen_subopt_wan": dec["gen_subopt_wan"],
                "gen_subopt_pct": dec["gen_subopt_pct"],
                "m3_cost_wan": rigor["regions"][r]["m3_final"]["cost_wan"],
                "baseline_cost_wan": rigor["regions"][r]["baseline"]["cost_wan"],
            }
        report["Q3"] = {
            "regions": regs,
            "caliber": ("双锚 LP 对照（sum_12 净化口径）：储能价值 vs 无储能 LP；"
                        "生成器次优 vs 基准；百分比对 E/F 负基数失真用绝对值"),
        }

    # Q4（基线 vs 折中）
    q4 = load("output/q4/q4_indicators.json")
    q4b = load("output/q4/q4_bilevel.json")
    q4a = load("output/q4/q4_ablation.json")
    if q4 and q4b:
        base = q4["baseline"]
        comp = q4b["compromise"]
        report["Q4"] = {
            "baseline": {k: base.get(k) for k in
                         ("cost_wan", "carbon_t", "latency_ms", "qos",
                          "nu_pct", "peak_net_MW")},
            "compromise": {k: comp.get(k) for k in
                           ("cost_wan", "carbon_t", "latency_ms",
                            "one_minus_qos", "one_minus_nu", "peak_net_MW")},
            "improve_vs_baseline_pct": round(
                (base["cost_wan"] - comp["cost_wan"]) / base["cost_wan"] * 100, 3),
            "caliber": "Q4 基线=Q2 折中×M3_final；折中=正式前沿 TOPSIS；波动/峰值主时段",
        }

    report["discipline"] = ("论文每个数字可追溯唯一产物文件（数据核查纪律）；"
                            "口径声明见 docs/materials/08_口径声明大全.md；"
                            "修正历史见 docs/sums/sum_12")
    with open(OUT / "baseline_proof_all.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()
