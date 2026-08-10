"""step3.8_q3_rigor — Q3 反思 P1 严谨性补强四件套（sum_11 §九 遗留项）.

#4 无储能双锚（三档价值分解）: 无储能（模板消纳 G=D−min(W,c_h·D)）/
   生成器基准（数据列 GridPurchase−GridSell）/ M3_final（LP 最优）
   → 储能价值 = M3_final − 无储能；生成器次优代价 = 无储能 − 基准
   （改进率叙事从单锚升为双锚，Q3 反思 #4）
#1 斜坡活跃率: M3_final 解中 |ΔPc|/R_c 与 |ΔPd|/R_d 的 binding 小时数
   （"斜坡全免费"须证明约束活跃性——若不 binding 则物理化修正是装饰，
   Q3 反思 #1）
#3 基准终态对称性: 生成器 SOC(2406) vs Init（若生成器非严格终态，则优化侧
   "终态严格"是单侧约束，X8 免费不等价于口径对称，Q3 反思 #3）+ Closure
   段（2400-2405）基准充放行为（#2）
#6 分窗分布: 0-2399 划 14 周 → 各窗改进率分布（点估计 → 分布，
   替代 bootstrap——数据确定性下分窗比重采样更诚实）

产物（output/q3/）: q3_rigor.json + 无图（表格化）
"""
import importlib.util
import json
from pathlib import Path

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from step0_config import CLEAN, OUTPUT, REGIONS

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
    c.update(s41=s41, s33=s33)
    return c


def baseline_series(c, r: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基准逐时 G/S/Q（数据列，全 2407h）。"""
    rt = c["rt"]
    sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
    G = sub["GridPurchase_MW"].to_numpy(dtype=float)
    S = sub["GridSell_MW"].to_numpy(dtype=float)
    Q = sub["Curtailment_MW"].to_numpy(dtype=float)
    return G, S, Q


def no_storage_series(c, d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """模板无储能口径（参考）：G = D − min(W, c_h·D)，S=0，Q = W − min(W, c_h·D)。

    注：该口径为"模板消纳"参考（无 LP 优化自由度、无外送）——论文主对照
    用 no_storage_lp（solve_m3 零储能同框架，对照纯度达标）。
    """
    T = len(d["D"])
    cap_h = np.minimum(d["W"], np.asarray(d["c_h"])[np.arange(T) % 24] * d["D"])
    Q = d["W"] - cap_h
    G = d["D"] - cap_h
    return G, np.zeros(T), Q


def metrics(G, S, Q, price, sellp, carbon, W, nu_denom=None):
    """四指标（主时段 0-2399 口径 + 全时段成本/碳）。"""
    cost = float((price * G - sellp * S).sum() / 1e4)
    carb = float((carbon * G).sum())
    net = G - S
    return {"cost_wan": cost, "carbon_t": carb,
            "peak_main_MW": float(net[:2400].max()),
            "std_main_MW": float(net[:2400].std()),
            "ramp_main_MW": float(np.abs(np.diff(net[:2400])).max()),
            "nu_pct": 100.0 * (1 - Q.sum() / max(nu_denom, 1e-9))}


def slope_binding(d, rows, r: str, s33) -> dict:
    """斜坡活跃率：|ΔPc|/R_c 与 |ΔPd|/R_d 的 binding 小时数（≥99% 记活跃）。"""
    rc, rd = s33.GEN_SLOPE.get(r, (np.nan, np.nan))
    pc = np.array([x["Pc"] for x in rows])
    pd_ = np.array([x["Pd"] for x in rows])
    dpc = np.abs(np.diff(pc))
    dpd = np.abs(np.diff(pd_))
    T = len(pc)
    return {"ramp_c_limit": rc, "ramp_d_limit": rd,
            "n_hours": T - 1,
            "binding_c_hours": int((dpc >= 0.99 * rc).sum()),
            "binding_d_hours": int((dpd >= 0.99 * rd).sum()),
            "max_dpc": float(dpc.max()), "max_dpd": float(dpd.max()),
            "binding_c_pct": round(float((dpc >= 0.99 * rc).mean() * 100), 3),
            "binding_d_pct": round(float((dpd >= 0.99 * rd).mean() * 100), 3)}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    s33 = c["s33"]
    consume = c["consume"]
    report = {"regions": {}, "summary": {}}
    week_improve = {"m3_vs_base": [], "m3_vs_nostore": []}

    for r in REGIONS:
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = (json.loads((OUTPUT / "q3" / "q3_dr_reverse.json").read_text(
            encoding="utf-8"))["templates"][r]["charge_hours"],
            json.loads((OUTPUT / "q3" / "q3_dr_reverse.json").read_text(
                encoding="utf-8"))["templates"][r]["discharge_hours"])
        price = d["price"]; sellp = d["sellp"]; carbon = d["carbon"]
        W = d["W"]

        Gb, Sb, Qb = baseline_series(c, r)
        mb = metrics(Gb, Sb, Qb, price, sellp, carbon, W,
                     nu_denom=W.sum())
        Gn, Sn, Qn = no_storage_series(c, d)
        mn = metrics(Gn, Sn, Qn, price, sellp, carbon, W,
                     nu_denom=W.sum())
        # 无储能 LP 对照（同框架零储能：solve_m3(d, [], []) → Pc=Pd=0、
        # SOC 恒定=init、终态恒满足）——对照纯度达标（只差储能，1-1 修正）
        m_ns = s33.solve_m3(d, [], [], region=r)
        assert m_ns["cost_wan"] is not None, f"{r} 无储能 LP 不可行"
        Gns = np.array([x["G"] for x in m_ns["rows"]])
        Sns = np.array([x["S"] for x in m_ns["rows"]])
        Qns = np.array([x["Q"] for x in m_ns["rows"]])
        mns = metrics(Gns, Sns, Qns, price, sellp, carbon, W,
                      nu_denom=W.sum())
        m3 = s33.solve_m3(d, ch, dh, region=r)
        G3 = np.array([x["G"] for x in m3["rows"]])
        S3 = np.array([x["S"] for x in m3["rows"]])
        Q3v = np.array([x["Q"] for x in m3["rows"]])
        mm = metrics(G3, S3, Q3v, price, sellp, carbon, W,
                     nu_denom=W.sum())

        # 三档分解（主对照=LP 版；模板版作参考）
        storage_value = mm["cost_wan"] - mns["cost_wan"]
        gen_subopt = mns["cost_wan"] - mb["cost_wan"]
        lp_freedom = mns["cost_wan"] - mn["cost_wan"]
        # 斜坡活跃率
        sl = slope_binding(d, m3["rows"], r, s33)
        # 基准终态对称性
        rt = c["rt"]
        sub = rt[rt.Region == r].sort_values("Hour").reset_index(drop=True)
        soc_2406 = float(sub.loc[sub.Hour == 2406, "SOC_MWh"].iloc[0]) \
            if (sub.Hour == 2406).any() else None
        init = float(d["init_soc"])
        closure = sub[(sub.Hour >= 2400) & (sub.Hour <= 2405)]
        closure_charge = int((closure["ChargePower_MW"] > 0.01).sum())
        closure_discharge = int((closure["DischargePower_MW"] > 0.01).sum())

        # 分窗（14 周 × 168h + 尾窗）
        n_weeks = 14
        wk = {"weeks": []}
        for w_i in range(n_weeks):
            s0, s1 = w_i * 168, min((w_i + 1) * 168, 2400)
            if s1 <= s0:
                continue
            cb = float((price[s0:s1] * Gb[s0:s1]).sum() / 1e4)
            cm = float((price[s0:s1] * G3[s0:s1]).sum() / 1e4)
            cn = float((price[s0:s1] * Gn[s0:s1]).sum() / 1e4)
            wk["weeks"].append({"week": w_i + 1, "hours": [s0, s1],
                                "base_cost_wan": round(cb, 1),
                                "m3_cost_wan": round(cm, 1),
                                "improve_vs_base_pct": round(
                                    (cb - cm) / max(abs(cb), 1e-9) * 100, 3),
                                "improve_vs_nostore_pct": round(
                                    (cn - cm) / max(abs(cn), 1e-9) * 100, 3)})
            if abs(cb) > 1e-9:
                week_improve["m3_vs_base"].append((cb - cm) / abs(cb) * 100)
            if abs(cn) > 1e-9:
                week_improve["m3_vs_nostore"].append((cn - cm) / abs(cn) * 100)

        report["regions"][r] = {
            "baseline": mb, "no_storage_template": mn,
            "no_storage_lp": mns, "m3_final": mm,
            "decomposition": {
                "storage_value_wan": round(storage_value, 1),
                "gen_subopt_wan": round(gen_subopt, 1),
                "lp_freedom_value_wan": round(lp_freedom, 1),
                "storage_value_pct": round(
                    storage_value / max(abs(mns["cost_wan"]), 1e-9) * 100, 3),
                "gen_subopt_pct": round(
                    gen_subopt / max(abs(mb["cost_wan"]), 1e-9) * 100, 3),
                "note": ("主对照=无储能 LP（同框架零储能，对照纯度达标）；"
                         "旧模板版无储能（无 LP 自由度+无外送）混入"
                         "lp_freedom_value——D 区约 1.44 亿（1-1 修正，总账 +52）")},
            "slope_binding": sl,
            "terminal_symmetry": {
                "generator_soc_2406": soc_2406,
                "init_soc": init,
                "strict_terminal": bool(
                    soc_2406 is not None and abs(soc_2406 - init) < 1e-6),
                "closure_charge_hours": closure_charge,
                "closure_discharge_hours": closure_discharge,
                "closure_note": ("Closure 段基准充放行为（2400-2405）——"
                                 "优化侧结算段禁充的对称性检查")},
            "weekly": wk["weeks"]}

    # 全局分窗分布
    report["summary"] = {
        "week_improve_m3_vs_base": {
            "n": len(week_improve["m3_vs_base"]),
            "mean_pct": round(float(np.mean(week_improve["m3_vs_base"])), 3),
            "min_pct": round(float(np.min(week_improve["m3_vs_base"])), 3),
            "max_pct": round(float(np.max(week_improve["m3_vs_base"])), 3),
            "p25_pct": round(float(np.percentile(week_improve["m3_vs_base"], 25)), 3),
            "p75_pct": round(float(np.percentile(week_improve["m3_vs_base"], 75)), 3)},
        "week_improve_m3_vs_nostore": {
            "n": len(week_improve["m3_vs_nostore"]),
            "mean_pct": round(float(np.mean(week_improve["m3_vs_nostore"])), 3),
            "min_pct": round(float(np.min(week_improve["m3_vs_nostore"])), 3),
            "max_pct": round(float(np.max(week_improve["m3_vs_nostore"])), 3)},
        "caliber": ("Q3 反思 P1 四件套 v2：三档双锚主对照=无储能 LP（solve_m3 零储能"
                    "同框架，1-1 纯度修正）；模板无储能作参考（lp_freedom_value="
                    "LP 自由度+外送收入红利，D 区 1.44 亿）；斜坡活跃率=binding 小时"
                    "（|ΔP|≥99%·R）；终态对称性=生成器 SOC(2406) vs Init；"
                    "分窗=14 周改进率分布（替代 bootstrap）")}

    with open(OUT_Q3 / "q3_rigor.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)

    # 控制台摘要（三档分解 + 斜坡 + 终态）
    for r in REGIONS:
        rr = report["regions"][r]
        dec = rr["decomposition"]
        sl = rr["slope_binding"]
        ts = rr["terminal_symmetry"]
        print(f"{r}: 储能价值 {dec['storage_value_wan']} 万 "
              f"({dec['storage_value_pct']}%) | LP自由度 {dec['lp_freedom_value_wan']} 万 | "
              f"生成器次优 {dec['gen_subopt_wan']} 万 ({dec['gen_subopt_pct']}%) | "
              f"斜坡 binding C{sl['binding_c_pct']}%/D{sl['binding_d_pct']}% | "
              f"SOC2406={ts['generator_soc_2406']} vs Init={ts['init_soc']} | "
              f"Closure 充{ts['closure_charge_hours']}h/放{ts['closure_discharge_hours']}h")


if __name__ == "__main__":
    main()
