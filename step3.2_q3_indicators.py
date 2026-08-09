"""step3.2_q3_indicators — Q3 四指标评估器 + 基准对照（spec_M4_Q3 D-1/D-2）.

口径（附件1 唯一口径，spec_M4_Q3 D-1）:
  Cost     = Σ(price·G − sellp·S)（万元）
  Carbon   = Σ(carb·G)（吨）
  P_peak   = max_t(G−S)（区域峰值净购电功率）
  Vol_std  = std(G−S)（净购电波动·统计视角）
  Ramp_max = max|Δ(G−S)|（净购电波动·物理视角，波动双指标并列）
  NU       = 1 − Q/W（I5 恒等式）——伴随指标（题目四指标不含，叙事对照用）
时域: 0-2406 全 2407h（说明6：2400-2405 结清计入结算；2406 仅终态结算）。

验收（优化题三件套，严禁预测指标错配）:
  1) 基准四指标与手算锚点一致（rel<1e-9，tests/test_q3.py 守卫）
  2) LP 四指标与 lp_all_regions.json 交叉一致（cost/carbon/nu）
  3) LP 逐时平衡式残差 <1e-6（I2 恒等式守卫，见测试）

产物（output/q3/）:
  q3_indicators.json       基准/LP/对比三表 + 口径声明
  q3_baseline_vs_lp.csv    逐区逐指标对照（改进率表，论文素材）
figures/step3/fig_q3_baseline_vs_opt.png   六区域四指标改进率柱状图
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import CLEAN, DATA_RAW, FILES, FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

METRICS = ["cost_wan", "carbon_t", "peak_net_MW", "vol_std_MW", "max_ramp_MW"]


def evaluate_storage(G: np.ndarray, S: np.ndarray, price: np.ndarray,
                     sellp: np.ndarray, carbon: np.ndarray,
                     Q: np.ndarray | None = None,
                     W: np.ndarray | None = None) -> dict:
    """Q3 四指标评估器（附件1 口径）——G/S/price/sellp/carbon 等长序列。

    Q/W 可选（给定则报告 NU 伴随指标 = 1 − Q/W，I5）。
    """
    G = np.asarray(G, dtype=float)
    S = np.asarray(S, dtype=float)
    net = G - S
    out = {
        "cost_wan": float((price * G - sellp * S).sum() / 1e4),
        "carbon_t": float((carbon * G).sum()),
        "peak_net_MW": float(net.max()),
        "vol_std_MW": float(net.std()),
        "max_ramp_MW": float(np.abs(np.diff(net)).max()) if len(net) > 1 else 0.0,
        "mean_net_MW": float(net.mean()),
    }
    if Q is not None and W is not None:
        wsum = float(np.asarray(W, dtype=float).sum())
        out["nu_pct"] = float(
            100.0 * (1.0 - np.asarray(Q, dtype=float).sum() / max(wsum, 1e-9)))
        out["curtail_MWh"] = float(np.asarray(Q, dtype=float).sum())
    return out


def load_rt() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "region_time_clean.csv")


def _rt_vectors(rt: pd.DataFrame, r: str) -> dict:
    sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
    pue = float(pd.read_excel(DATA_RAW / FILES["gpu"],
                              sheet_name="GPU中心基础情况")
                .set_index("Region").loc[r, "PUE"])
    return {
        "price": sub["ElectricityPrice_CNY_per_MWh"].to_numpy(),
        "sellp": sub["SellPrice_CNY_per_MWh"].to_numpy(),
        "carbon": sub["CarbonIntensity_tCO2_per_MWh"].to_numpy(),
        "W": sub["AvailableRenewable_MW"].to_numpy(),
        "D": ((sub["Baseline_AI_IT_Load_MW"] + sub["NonAI_IT_Load_MW"])
              * pue).to_numpy(),
    }


def baseline_indicators(rt: pd.DataFrame) -> dict:
    """题目基准四指标（附件列直接核算，0-2406 全时域）——改进率对照锚点."""
    out = {}
    for r in REGIONS:
        sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
        vec = _rt_vectors(rt, r)
        G = sub["GridPurchase_MW"].to_numpy()
        S = sub["GridSell_MW"].to_numpy()
        Q = sub["Curtailment_MW"].to_numpy()
        out[r] = evaluate_storage(G, S, vec["price"], vec["sellp"],
                                  vec["carbon"], Q=Q, W=vec["W"])
    return out


def lp_indicators(rt: pd.DataFrame) -> dict:
    """LP 优化解四指标（step3.0 产物 lp_baseline_RegionX.csv，历史 M0 传送带版）."""
    out = {}
    for r in REGIONS:
        lp = pd.read_csv(OUT_Q3 / f"lp_baseline_{r}.csv")
        vec = _rt_vectors(rt, r)
        G = lp["G"].to_numpy()
        S = lp["S"].to_numpy()
        out[r] = evaluate_storage(G, S, vec["price"], vec["sellp"],
                                  vec["carbon"], Q=lp["Q"].to_numpy(),
                                  W=vec["W"])
    return out


def m3_indicators(rt: pd.DataFrame) -> dict:
    """M3_final 主模型四指标（sum_10 回灌：时段约束+结算段禁充+终态严格+
    生成器量级斜坡+主时段指标）——论文主口径。"""
    import importlib.util
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s33p", root / "step3.3+_q3_model_evolve.py")
    s33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s33)
    spec2 = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s10)
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    tpl = json.loads((OUT_Q3 / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"]
    out = {}
    for r in REGIONS:
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        m = s33.solve_m3(d, ch, dh, region=r)
        vec = _rt_vectors(rt, r)
        G = np.array([x["G"] for x in m["rows"]])
        S = np.array([x["S"] for x in m["rows"]])
        out[r] = evaluate_storage(G, S, vec["price"], vec["sellp"],
                                  vec["carbon"], Q=np.array(
            [x["Q"] for x in m["rows"]]), W=vec["W"])
        out[r].update(s33.main_period_indicators(m["rows"]))
        out[r]["max_ramp_full_MW"] = round(m["max_ramp_MW"], 1)
    return out


def build_comparison(base: dict, opt: dict) -> list[dict]:
    """逐区逐指标改进率表（相对基准；NU 用 pp 差）。"""
    rows = []
    for r in REGIONS:
        for m in METRICS:
            b, o = base[r][m], opt[r][m]
            denom = abs(b) if abs(b) > 1e-9 else 1.0
            rows.append({"Region": r, "metric": m,
                         "baseline": round(b, 4), "lp": round(o, 4),
                         "change_pct": round((o - b) / denom * 100.0, 3)})
        for m in ("nu_pct", "curtail_MWh"):
            b, o = base[r][m], opt[r][m]
            rows.append({"Region": r, "metric": m,
                         "baseline": round(b, 4), "lp": round(o, 4),
                         "change_pct": round(o - b, 3)})
    return rows


def plot_improvement(comp: list[dict]) -> None:
    """fig_q3_baseline_vs_opt.png：六区域改进率分组柱状（% 或 pp）。"""
    df = pd.DataFrame(comp)
    main = df[df.metric.isin(METRICS)].pivot(index="Region",
                                             columns="metric",
                                             values="change_pct")
    main = main[METRICS]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(REGIONS))
    width = 0.18
    labels = {"cost_wan": "成本改进%", "carbon_t": "碳排改进%",
              "peak_net_MW": "峰值净购电%", "vol_std_MW": "波动std%",
              "max_ramp_MW": "最大爬坡%"}
    for i, m in enumerate(METRICS):
        ax.bar(x + (i - 2) * width, main[m], width, label=labels[m])
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("Region", "R") for r in REGIONS])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("相对基准变化（%，负=改善；正值=恶化）")
    ax.set_title("Q3 储能优化 vs 附件基准：四指标改进率（LP 成本单目标）")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_baseline_vs_opt.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    rt = load_rt()
    base = baseline_indicators(rt)
    opt = lp_indicators(rt)
    m3 = m3_indicators(rt)
    comp = build_comparison(base, opt)

    # 全局锚点自检（D4 口径：弃电 775.5 万 MWh / NU 32.9%）
    q_global = float(rt["Curtailment_MW"].sum())
    w_global = float(rt["AvailableRenewable_MW"].sum())
    nu_global = 100.0 * (1.0 - q_global / w_global)
    global_anchor = {"curtail_MWh_global": round(q_global, 0),
                     "nu_pct_global": round(nu_global, 2),
                     "note": "题目基准全局口径（D4）：弃电≈775.5 万 MWh、NU≈32.9%"}

    report = {
        "baseline": base, "lp": opt, "m3_final": m3, "comparison": comp,
        "global_anchor": global_anchor,
        "caliber": ("Cost=Σ(price·G−sellp·S)万；Carbon=Σ(carb·G)吨；"
                    "P_peak=max(G−S)；Vol_std=std(G−S)；Ramp_max=max|Δ(G−S)|；"
                    "NU=1−Q/W（I5）；时域 0-2406 全 2407h；"
                    "lp=step3.0 历史 M0（传送带版，不引用）；"
                    "m3_final=主口径（M1+结算段禁充+终态严格+生成器量级斜坡，"
                    "主时段指标 ramp_main/peak_main/std_main——X13 边界污染修复）"),
        "spec": "spec_M4_Q3 D-1/D-2（优化题验收：锚点 rel<1e-9）",
    }
    pd.DataFrame(comp).to_csv(OUT_Q3 / "q3_baseline_vs_lp.csv", index=False)
    with open(OUT_Q3 / "q3_indicators.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    plot_improvement(comp)

    print(json.dumps(global_anchor, ensure_ascii=False, indent=1))
    print(pd.DataFrame(comp).to_string(index=False))
    print("=== M3_final（主口径）===")
    print(json.dumps(m3, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
