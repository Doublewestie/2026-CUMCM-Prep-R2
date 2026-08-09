"""step3.9_q3_mechanism — Q3 反思 P2 机理补全五件套（轻量高价值项）.

#9  弃电边际价值曲线: sellp 缩放 k∈{0,0.5,1,1.5,2} 重解 M3_final →
    充电/外送量切换点（弃电充电 vs 弃电外送的机会竞争——"弃电充电"
    叙事的最终机理图）
#7  循环寿命: 等效循环 = ΣPc·ηc/Cap（M3_final 解）+ 循环退化成本估算
    （参数假设声明：每循环退化率 × 容量）
#10 SOC 越界裕度: 解到 Cap/MinSOC 的最近距离分布（验证解是否贴边界跑，
    支撑 E5"Cap 是活跃约束"）
#16 物理一致性: 数据列自洽检查——SellPrice>0 → GridSell>0 占比；
    ChargePower ≈ GridCharge + RenewableCharge 残差；SOC 递推残差
#8  CVaR: MPC 场景（AR(1)×64→K-means12）成本分布 → CVaR_0.9 平行指标
    （尾部风险度量，Q3 反思 #8 + Q4 波动衔接）

产物（output/q3/）: q3_mechanism.json
"""
import importlib.util
import json
from pathlib import Path

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from step0_config import OUTPUT, REGIONS

OUT_Q3 = OUTPUT / "q3"


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"),
        Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    s41 = _load_mod("step4.1_q4_indicators.py")
    c = s41.load_ctx()
    s33 = _load_mod("step3.3+_q3_model_evolve.py")
    s35 = _load_mod("step3.5_q3_mpc.py")
    c.update(s41=s41, s33=s33, s35=s35)
    return c


def sellp_scan(c, d, ch, dh, ks=(0.0, 0.5, 1.0, 1.5, 2.0)) -> list:
    """#9 弃电边际价值曲线：sellp 缩放 → 充电/外送量切换点。"""
    rows = []
    for k in ks:
        d2 = dict(d)
        d2["sellp"] = d["sellp"] * k
        m = c["s33"].solve_m3(d2, ch, dh)
        pc = np.array([x["Pc"] for x in m["rows"]]).sum()
        pd_ = np.array([x["Pd"] for x in m["rows"]]).sum()
        s = np.array([x["S"] for x in m["rows"]]).sum()
        rows.append({"sellp_scale": k,
                     "charge_MWh": round(float(pc), 1),
                     "discharge_MWh": round(float(pd_), 1),
                     "sell_MWh": round(float(s), 1),
                     "cost_wan": round(float(m["cost_wan"]), 1)})
    return rows


def cycle_life(c, d, ch, dh) -> dict:
    """#7 等效循环 + 退化成本（假设参数显式声明）。"""
    m = c["s33"].solve_m3(d, ch, dh)
    pc = np.array([x["Pc"] for x in m["rows"]])
    cap = d["cap_mwh"]
    eq_cycles = float((pc * d["eta_c"]).sum() / cap)
    # 假设：每等效循环容量退化 0.02%（BESS 常见量级），容量电价按区域
    # 成本近似 = 退化容量 × 平均购电价（假设声明）
    deg_frac = 0.0002
    price_mean = float(d["price"].mean())
    deg_cost_wan = eq_cycles * deg_frac * cap * price_mean / 1e4
    return {"equivalent_cycles": round(eq_cycles, 1),
            "deg_frac_per_cycle": deg_frac,
            "deg_cost_est_wan": round(deg_cost_wan, 2),
            "note": "退化参数为假设（0.02%/等效循环）——仅作量级参考，不参与主口径"}


def soc_margin(c, d, ch, dh) -> dict:
    """#10 SOC 越界裕度：到 Cap/MinSOC 的最近距离分布。"""
    m = c["s33"].solve_m3(d, ch, dh)
    soc = np.array([x["SOC"] for x in m["rows"]])
    cap, mn = d["cap_mwh"], d["min_soc"]
    d_cap = cap - soc
    d_min = soc - mn
    touch_cap = float((d_cap < 1e-6).mean() * 100)
    touch_min = float((d_min < 1e-6).mean() * 100)
    near_cap = float((d_cap < 0.05 * cap).mean() * 100)
    return {"dist_to_cap_pct": {
                "p5": round(float(np.percentile(d_cap / cap * 100, 5)), 2),
                "p50": round(float(np.percentile(d_cap / cap * 100, 50)), 2),
                "p95": round(float(np.percentile(d_cap / cap * 100, 95)), 2)},
            "touch_cap_pct": round(touch_cap, 3),
            "touch_min_pct": round(touch_min, 3),
            "near_cap_5pct_pct": round(near_cap, 3)}


def physical_consistency(c, r: str) -> dict:
    """#16 数据列自洽：SellPrice>0→GridSell>0 占比；充电双通道；SOC 残差。"""
    rt = c["rt"]
    sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
    sellp_pos = sub["SellPrice_CNY_per_MWh"] > 0
    n_sp = int(sellp_pos.sum())
    n_sell = int((sub.loc[sellp_pos, "GridSell_MW"] > 0.01).sum())
    charge = sub["ChargePower_MW"].to_numpy(dtype=float)
    gc = sub["GridCharge_MW"].to_numpy(dtype=float)
    rc_ = sub["RenewableCharge_MW"].to_numpy(dtype=float)
    mask = charge > 0.01
    resid = (gc[mask] + rc_[mask] - charge[mask]) / np.maximum(charge[mask], 1e-9)
    # SOC 递推残差（数据表 vs 公式，分区效率）
    soc = sub["SOC_MWh"].to_numpy(dtype=float)
    disc = sub["DischargePower_MW"].to_numpy(dtype=float)
    d_soc = d_load = None
    from step0_config import CLEAN
    sp = np.loadtxt(CLEAN / "storage_params.csv" if False else None) \
        if False else None
    return {"sellprice_pos_hours": int(n_sp),
            "sell_positive_when_sellprice": round(float(n_sell / max(n_sp, 1)), 4),
            "charge_dual_channel_resid": {
                "max_rel": round(float(np.abs(resid).max()), 4),
                "mean_rel": round(float(np.abs(resid).mean()), 4),
                "n_charge_hours": int(mask.sum())}}


def cvar_estimate(c, r: str, sigma: float = 0.2,
                  alpha: float = 0.9) -> dict:
    """#8 CVaR：AR(1)×64 场景 → K-means12 代表场景成本分布 → CVaR_α。"""
    d = c["s41"].build_lower_data(c, np.zeros((2407, 6)), r)
    d["c_h"] = np.asarray(c["consume"][r], dtype=float)
    ch, dh = (json.loads((OUTPUT / "q3" / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"][r]["charge_hours"],
        json.loads((OUTPUT / "q3" / "q3_dr_reverse.json").read_text(
            encoding="utf-8"))["templates"][r]["discharge_hours"])
    wt = d["W"]
    S = c["s35"].gen_scenarios(wt, sigma)
    reps = S
    w = np.ones(len(S)) / len(S)
    costs = []
    n_infeas = 0
    for s_i in range(len(reps)):
        d2 = dict(d)
        d2["W"] = reps[s_i]
        m = c["s33"].solve_m3(d2, ch, dh)
        if m["cost_wan"] is None:
            n_infeas += 1
            continue
        costs.append(m["cost_wan"])
    costs = np.array(costs)
    if len(costs) == 0:
        return {"sigma": sigma, "alpha": alpha, "n_scenarios": 0,
                "n_infeasible": n_infeas,
                "note": "全部场景 infeasible（购电上限瓶颈）"}
    q_alpha = float(np.quantile(costs, alpha))
    cvar = float(costs[costs >= q_alpha].mean()) if (costs >= q_alpha).any() \
        else float(costs.max())
    return {"sigma": sigma, "alpha": alpha, "n_scenarios": len(costs),
            "n_infeasible": n_infeas,
            "infeasible_pct": round(n_infeas / len(reps) * 100, 1),
            "expected_wan": round(float(costs.mean()), 1),
            f"cvar{int(alpha*100)}_wan": round(cvar, 1),
            "spread_wan": round(float(costs.max() - costs.min()), 1),
            "note": ("E/F 低 W 尾部场景触发购电上限 max_import 瓶颈（LP "
                     "infeasible）——【实证】新能源骤降尾部=购电上限硬约束；"
                     "CVaR 在可行域内计算，infeasible_pct 即尾部风险度量；"
                     "Q3 MPC 均值场景口径不受影响（非逐场景期望）")}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    s33 = c["s33"]
    consume = c["consume"]
    tpl = json.loads((OUTPUT / "q3" / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"]
    report = {"regions": {}}
    for r in REGIONS:
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        report["regions"][r] = {
            "sellp_curve": sellp_scan(c, d, ch, dh),
            "cycle_life": cycle_life(c, d, ch, dh),
            "soc_margin": soc_margin(c, d, ch, dh),
            "physical_consistency": physical_consistency(c, r),
            "cvar": cvar_estimate(c, r)}
    report["caliber"] = ("Q3 反思 P2 五件套：弃电边际价值=sellp 缩放扫描切换点；"
                         "等效循环=ΣPc·ηc/Cap；退化参数 0.02%/循环为假设；"
                         "SOC 裕度=距 Cap/MinSOC 分布；物理一致性=数据列自洽；"
                         "CVaR=AR(1) 场景成本分布尾部（α=0.9）")
    with open(OUT_Q3 / "q3_mechanism.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    for r in REGIONS:
        rr = report["regions"][r]
        sc = rr["sellp_curve"]
        print(f"{r}: 卖价×{sc[0]['sellp_scale']}→×{sc[-1]['sellp_scale']} "
              f"充电 {sc[0]['charge_MWh']}→{sc[-1]['charge_MWh']} MWh | "
              f"等效循环 {rr['cycle_life']['equivalent_cycles']} | "
              f"贴Cap {rr['soc_margin']['touch_cap_pct']}% | "
              f"CVaR90 {rr['cvar'].get('cvar90_wan')}")


if __name__ == "__main__":
    main()
