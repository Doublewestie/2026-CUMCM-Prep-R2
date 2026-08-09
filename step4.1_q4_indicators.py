"""step4.1_q4_indicators — Q4 指标章程：六指标评估器 + 基线 + 独立维度预检 + QOS α 扫描 + 峰谷比预扫.

定位（Q4 方案定稿 step4.1）:
  Q4 六指标（前三问口径全部继承，零新发明）:
    Cost     = Σ_r Σ_t (price·G − sellp·S)（万元，全 2407h）
    Carbon   = Σ_r Σ_t (carb·G)（吨，全 2407h）
    Lat      = GPU-hours 加权时延（Q2 T1 纯加权形式，任务级）
    QOS      = α·完成率 + (1−α)·时延裕度（α 默认 0.5，敏感性扫描定案）
    NU       = Σ(W−Q)/ΣW × 100（I5 恒等式，消纳模板口径）
    P_peak   = max_r max_{t<2400} (G−S)（区域峰值净购电，主时段 0-2399，X13 教训）
  下层储能 = M3_final（solve_m3：时段状态机+结算段禁充+终态严格+生成器斜坡）
  D(r,t)   = (NonAI_IT_Load(r,t) + AI_IT_Load(r,t)) × PUE(r)   ← Q4 与 Q3 的本质区别：
           Q3 D 固定（Baseline），Q4 D 随上层任务调度变化（说明2/说明3 口径）

本文件三件事（指标章程）:
  ① evaluate_q4_six: 六指标全链评估器（step4.0 进化评价直接复用）
  ② baseline + dim_check: Q4 基线（Q2 TOPSIS 折中 × M3_final）+ 独立维度预检
  ③ alpha_scan + price_ratio_prescan: QOS α 敏感性 + 峰谷比边界预扫（定 step4.3 档位）

产物（output/q4/）:
  q4_indicators.json   六指标口径声明 + 基线六指标 + 维度预检 + α/峰谷比扫描
figures/step4/fig_q4_dimcheck.png  相关矩阵热图 + α 稳健区间
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_config import (CLEAN, DATA_RAW, FILES, FIGURES, MAX_LATENCY,
                          OUTPUT, REGIONS, TASK_TYPES, HOURS_TOTAL)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q4 = OUTPUT / "q4"
FIG_Q4 = FIGURES / "step4"
SIX_METRICS = ["cost_wan", "carbon_t", "latency_ms", "one_minus_qos",
               "one_minus_nu", "peak_net_MW"]

_CTX = {}


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.split(".")[0].replace(".", "_"), Path(__file__).resolve().parent / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ctx() -> dict:
    """一次性加载全链上下文（评估器/调度器/求解器 + 数据 + 模板）。"""
    global _CTX
    if _CTX:
        return _CTX
    s10 = _load_mod("step1.0_baseline_schedule.py")
    s20 = _load_mod("step2.0_construct.py")
    s32 = _load_mod("step3.2_q3_indicators.py")
    s33 = _load_mod("step3.3+_q3_model_evolve.py")
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = s10.load_params()
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    tpl = json.loads((OUTPUT / "q3" / "q3_dr_reverse.json").read_text(
        encoding="utf-8"))["templates"]
    _CTX.update(s10=s10, s20=s20, s32=s32, s33=s33, wt=wt, rt=rt,
                params=params, consume=consume, tpl=tpl)
    return _CTX


# ---------- ① 六指标全链评估器 ----------

def schedule_occupancy(c, sched: pd.DataFrame) -> np.ndarray:
    """调度 → 逐时 AI_IT_Load (2407, 6)（附件1 唯一口径，s10.build_occupancy）。"""
    _, ai_mw = c["s10"].build_occupancy(
        c["wt"], sched.set_index("TaskID")["Region"],
        sched.set_index("TaskID")["StartHour"])
    return ai_mw


def build_lower_data(c, ai_mw: np.ndarray, r: str) -> dict:
    """区域下层 LP 数据：D 用调度版 AI 负荷（Q4 与 Q3 的本质区别），其余继承 Q3。"""
    d = c["s33"]._load_region_data(r)
    sub = c["rt"][c["rt"].Region == r].sort_values("Hour").reset_index(drop=True)
    pue = float(pd.read_excel(DATA_RAW / FILES["gpu"],
                              sheet_name="GPU中心基础情况")
                .set_index("Region").loc[r, "PUE"])
    d["D"] = (sub["NonAI_IT_Load_MW"].to_numpy() + ai_mw[:, REGIONS.index(r)]) * pue
    d["c_h"] = np.asarray(c["consume"][r], dtype=float)
    return d


def qos_metrics(c, sched: pd.DataFrame) -> dict:
    """QOS = α·完成率 + (1−α)·裕度；裕度 = mean(min(1, (m_i−ω_i)/m_i))。"""
    lat = pd.read_excel(DATA_RAW / FILES["latency"], sheet_name="network_latency")
    lm = lat.pivot(index="FromRegion", columns="ToRegion",
                   values="NetworkLatency_ms")
    m = c["wt"].merge(sched, on="TaskID")
    omega = np.array([lm.loc[s, d] for s, d in zip(m["SourceRegion"],
                                                   m["Region"])], dtype=float)
    mx = np.array([MAX_LATENCY[t] for t in m["TaskType"]])
    margin = float(np.mean(np.minimum(1.0, (mx - omega) / mx)))
    done = ((m["StartHour"] + m["dur_h"]) <= 2406.0001).mean()
    return {"completion_rate": float(done), "margin_mean": margin}


def evaluate_q4_six(c, sched: pd.DataFrame, alpha: float = 0.5,
                    solver: str = "m3") -> dict:
    """六指标全链评估（上层调度 → 下层 M3_final LP → 六目标，最小化方向）。

    solver="m3" → solve_m3（M3_final 主口径，含斜坡/禁充/终态严格）
    solver="m1" → solve_region_timed 时段约束（无斜坡，进化内快速代理）
    返回最小化向量 obj = [cost, carbon, lat, 1−qos, 1−nu/100, peak] + 明细。
    viol_h>0 → 上层调度不可行（超容），obj 全 1e12（Deb 约束处理，Q2 教训）。
    """
    c = load_ctx() if c is None else c
    ai_mw = schedule_occupancy(c, sched)
    parallel = c["s10"].build_occupancy(
        c["wt"], sched.set_index("TaskID")["Region"],
        sched.set_index("TaskID")["StartHour"])[0]
    cap_arr = np.array([c["params"]["cap"][r] for r in REGIONS])
    viol_h = int((parallel > cap_arr).sum())
    if viol_h > 0:
        return {"obj": np.full(6, 1e12), "viol_h": viol_h}

    cost = carbon = qsum = wsum = 0.0
    peak_net = 0.0
    per_region = {}
    for r in REGIONS:
        d = build_lower_data(c, ai_mw, r)
        ch, dh = c["tpl"][r]["charge_hours"], c["tpl"][r]["discharge_hours"]
        if solver == "m3":
            m = c["s33"].solve_m3(d, ch, dh, region=r)
        else:
            m = c["s33"].solve_region_timed(d, charge_hours=ch,
                                            discharge_hours=dh)
        if m["cost_wan"] is None:
            return {"obj": np.full(6, 1e12), "viol_h": viol_h}
        G = np.array([x["G"] for x in m["rows"]])
        S = np.array([x["S"] for x in m["rows"]])
        Q = np.array([x["Q"] for x in m["rows"]])
        net = G - S
        cost += m["cost_wan"]
        carbon += m["carbon_t"]
        qsum += float(Q.sum())
        wsum += float(d["W"].sum())
        peak_net = max(peak_net, float(net[:2400].max()))
        per_region[r] = {"cost_wan": round(m["cost_wan"], 3),
                         "carbon_t": round(m["carbon_t"], 1),
                         "nu_pct": round(100.0 * m["nu"], 2),
                         "peak_net_MW": round(float(net[:2400].max()), 1)}
    lat = c["s20"].compute_latency(c["wt"], sched)
    qos = qos_metrics(c, sched)
    one_qos = 1.0 - (alpha * qos["completion_rate"]
                     + (1 - alpha) * qos["margin_mean"])
    nu_pct = 100.0 * (1.0 - qsum / max(wsum, 1e-9))
    obj = np.array([cost, carbon, lat, one_qos, 1.0 - nu_pct / 100.0,
                    peak_net])
    return {"obj": obj, "cost_wan": cost, "carbon_t": carbon,
            "latency_ms": lat, "qos": 1.0 - one_qos,
            "nu_pct": nu_pct, "peak_net_MW": peak_net,
            "viol_h": 0, "per_region": per_region,
            "qos_alpha": alpha, "solver": solver,
            "qos_metrics": qos}


# ---------- ② 基线 + 独立维度预检 ----------

def q2_compromise_policy(c) -> list:
    """Q4 基线上层策略 = Q2 TOPSIS 折中策略（nsga2_front.csv 熵权 TOPSIS 重算）。"""
    front = pd.read_csv(OUTPUT / "q2" / "nsga2_front.csv")
    obj = front[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]].to_numpy()
    obj[:, 3] = 1 - obj[:, 3] / 100
    X = (obj - obj.min(axis=0)) / (obj.max(axis=0) - obj.min(axis=0) + 1e-9)
    p = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))
    w = (1 - e) / (1 - e).sum()
    sd = np.sqrt(((X - X.min(axis=0)) ** 2 * w).sum(axis=1))
    nd = np.sqrt(((X - X.max(axis=0)) ** 2 * w).sum(axis=1))
    i = int(np.argmax(nd / (sd + nd)))
    return json.loads(front.loc[i, "policy"])


def baseline_q4(c, alpha: float = 0.5, solver: str = "m3") -> dict:
    """Q4 基线 = Q2 折中策略上层 × M3_final 下层（组合重现锚点）。"""
    pol = q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol))
    ev = evaluate_q4_six(c, sched, alpha=alpha, solver=solver)
    ev["policy"] = pol
    return ev


def _policy_pool(c, n_rand: int = 25) -> list[list]:
    """维度预检策略池：构造解 + Q2 折中 + 随机变体。"""
    rng = np.random.RandomState(0)
    lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ub = np.array([15.0, 6.65, 24.0, 24.0, 0.08])
    pool = [list(np.zeros(5)), q2_compromise_policy(c)]
    for _ in range(n_rand):
        pool.append(list(rng.rand(5) * (ub - lb) + lb))
    return pool


def dim_check(c, n_rand: int = 25) -> dict:
    """独立维度预检：可行策略采样 → 六指标相关矩阵 + 箱内 spread + 特征值有效维度.

    随机策略大多 viol>0（容量不可行）——重采样直到收集 n_rand 个可行解
    （可行区约束：mig_gpu≤8 / headroom≤0.03，Q2 可行区实证内），
    保证相关矩阵样本量有效性（样本 <8 时输出降级声明）。
    """
    rng = np.random.RandomState(0)
    lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ub = np.array([8.0, 6.65, 24.0, 24.0, 0.03])
    mat = []
    attempts = 0
    while len(mat) < n_rand + 2 and attempts < 400:
        attempts += 1
        pol = list(rng.rand(5) * (ub - lb) + lb)
        sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                               c["params"], tuple(pol))
        ev = evaluate_q4_six(c, sched)
        if ev["viol_h"] == 0:
            mat.append(ev["obj"])
    if len(mat) < 8:
        return {"n_policies": int(len(mat)), "corr": None,
                "eigen": None, "n_dim_90pct": None,
                "verdict": "可行样本不足（<8），维度预检降级——先跑 step4.0 后用前沿样本复核",
                "attempts": attempts}
    F = np.array(mat)
    corr = np.corrcoef(F.T)
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(F.T)))[::-1]
    cum = np.cumsum(eig / max(eig.sum(), 1e-12))
    n_dim = int(np.sum(cum < 0.90)) + 1
    spread = {}
    for i in range(6):
        for j in range(i + 1, 6):
            if abs(corr[i, j]) > 0.8:
                x = F[:, i]
                bins = np.quantile(x, [0, 0.33, 0.66, 1.0])
                ys = []
                for k in range(3):
                    sel = (x >= bins[k]) & (x < bins[k + 1]) if k < 2 \
                        else (x >= bins[k]) & (x <= bins[k + 1])
                    ys += [F[sel, j]]
                ratio = float(np.mean([y.std() for y in ys if len(y) > 1])
                              / max(F[:, j].std(), 1e-12)) \
                    if len(mat) > 3 else None
                spread[f"{SIX_METRICS[i]}|{SIX_METRICS[j]}"] = {
                    "corr": round(float(corr[i, j]), 3),
                    "bin_y_std_ratio": None if ratio is None
                    else round(ratio, 3)}
    return {"n_policies": int(len(F)), "corr": corr.tolist(),
            "eigen": eig.tolist(), "cum_var": cum.tolist(),
            "n_dim_90pct": int(n_dim),
            "pair_spread": spread, "attempts": attempts,
            "verdict": (f"六目标实际独立维度约 {n_dim}（累计方差 90%，"
                        f"{len(F)} 个可行策略采样）；"
                        "成本-碳共线/退化详情见 corr 与 pair_spread，"
                        "按退化呈现叙事入论文")}


# ---------- ③ QOS α 敏感性 + 峰谷比预扫 ----------

def alpha_scan(c, alphas=(0.0, 0.3, 0.5, 0.7, 1.0)) -> dict:
    """QOS α 敏感性：折中解漂移 vs α（α 只影响 1−QOS 目标分量，报告其变化）。"""
    pol = q2_compromise_policy(c)
    sched = c["s20"].schedule_constructive(c["wt"], c["rt"], c["s10"],
                                           c["params"], tuple(pol))
    rows = []
    for a in alphas:
        ev = evaluate_q4_six(c, sched, alpha=a)
        rows.append({"alpha": a, "one_minus_qos": round(float(ev["obj"][3]), 6),
                     "qos": round(ev["qos"], 6),
                     "completion": ev["qos_metrics"]["completion_rate"]
                     if "qos_metrics" in ev else None})
    ones = [r["one_minus_qos"] for r in rows]
    drift = float(max(ones) - min(ones))
    return {"scan": rows, "drift": drift,
            "verdict": ("QOS 定义域 [0,1]；α 只移动 1−QOS 分量（其余五目标结构解耦，"
                        "evaluate 中 cost/carbon/lat/nu/peak 与 α 无关）；"
                        f"1−QOS 跨度 {drift:.4f} 为定义使然的线性变化；"
                        "QOS 权重的真正稳健性由 TOPSIS 折中权重敏感性检验"
                        "（step2.5 C2 同款，±20% 扰动折中解漂移）")}


def _price_reconstruct(rt: pd.DataFrame, k: float) -> np.ndarray:
    """峰谷比重构：新价格 = 时段均值 + k×(原价格 − 时段均值)（均值不变，峰谷差放大 k 倍）。"""
    p = rt["ElectricityPrice_CNY_per_MWh"].to_numpy()
    mu = rt.groupby("PricePeriod")["ElectricityPrice_CNY_per_MWh"].transform(
        "mean").to_numpy()
    return mu + k * (p - mu)


def price_ratio_prescan(c, ks=(1.2, 1.5, 2.0, 2.5)) -> dict:
    """峰谷比边界预扫：价格统计量失真检查 + 基线策略成本响应 → step4.3 档位建议。"""
    rt = c["rt"]
    rows = []
    for k in ks:
        p = _price_reconstruct(rt, k)
        peak_ratio = p.max() / p.min()
        rows.append({"k": k, "price_min": round(float(p.min()), 1),
                     "price_max": round(float(p.max()), 1),
                     "peak_ratio": round(float(peak_ratio), 2),
                     "mean": round(float(p.mean()), 1),
                     "mean_drift_pct": round(float(
                         abs(p.mean() / rt["ElectricityPrice_CNY_per_MWh"].mean()
                             - 1) * 100), 4)})
    cur = float(rt["ElectricityPrice_CNY_per_MWh"].max()
                / rt["ElectricityPrice_CNY_per_MWh"].min())
    sensible = [r for r in rows if r["peak_ratio"] <= 10.0]
    return {"current_peak_ratio": round(cur, 2), "scan": rows,
            "suggested": [r["k"] for r in sensible],
            "verdict": ("峰谷比档位建议取失真边界内的 k（峰谷比≤10 视为合理）；"
                        "均值漂移必须≈0（重构保均值）；"
                        "正式档位由本预扫 + step4.3 场景成本响应共同定案")}


def plot_dimcheck(dim: dict, alpha: dict) -> None:
    corr = np.array(dim["corr"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    im = axes[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(6))
    axes[0].set_yticks(range(6))
    axes[0].set_xticklabels([m[:8] for m in SIX_METRICS], rotation=45,
                            fontsize=8)
    axes[0].set_yticklabels([m[:8] for m in SIX_METRICS], fontsize=8)
    axes[0].set_title(f"六目标相关矩阵（独立维度≈{dim['n_dim_90pct']}）")
    fig.colorbar(im, ax=axes[0])
    ascan = alpha["scan"]
    axes[1].plot([a["alpha"] for a in ascan], [a["one_minus_qos"] for a in ascan],
                 "o-", color="#2980b9")
    axes[1].set_xlabel("α（QOS 权重）")
    axes[1].set_ylabel("1 − QOS")
    axes[1].set_title(f"QOS α 敏感性（1−QOS 跨度 {alpha['drift']:.4f}）")
    fig.tight_layout()
    fig.savefig(FIG_Q4 / "fig_q4_dimcheck.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_Q4.mkdir(parents=True, exist_ok=True)
    FIG_Q4.mkdir(parents=True, exist_ok=True)
    c = load_ctx()

    base = baseline_q4(c)
    dim = dim_check(c)
    alpha = alpha_scan(c)
    pr = price_ratio_prescan(c)

    report = {
        "six_metrics_caliber": {
            "cost_wan": "Σ_r Σ_t (price·G − sellp·S)，全 2407h",
            "carbon_t": "Σ_r Σ_t (carb·G)，全 2407h",
            "latency_ms": "GPU-hours 加权时延（Q2 T1）",
            "qos": "α·完成率 + (1−α)·时延裕度，α 默认 0.5（本步敏感性定案）",
            "nu_pct": "Σ(W−Q)/ΣW × 100（I5，消纳模板口径）",
            "peak_net_MW": "max_r max_{t<2400}(G−S)（主时段 0-2399，X13）",
            "lower_model": "M3_final（solve_m3：时段状态机+结算段禁充+终态严格+生成器斜坡）",
            "D_caliber": "D(r,t) = (NonAI + 调度 AI_IT_Load) × PUE —— Q4 任务调度驱动（说明2/3）",
        },
        "baseline": {k: base[k] for k in
                     ("cost_wan", "carbon_t", "latency_ms", "qos", "nu_pct",
                      "peak_net_MW", "viol_h")},
        "baseline_policy": base["policy"],
        "baseline_per_region": base["per_region"],
        "dim_check": dim, "alpha_scan": alpha,
        "price_ratio_prescan": pr,
        "anchor_notes": ("基线上层=Q2 TOPSIS 折中策略（组合重现）；"
                         "下层对账见 test_q4（Baseline 负荷下 solve_m3 == "
                         "q3_indicators m3_final，rel<1e-6）"),
    }
    with open(OUT_Q4 / "q4_indicators.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    plot_dimcheck(dim, alpha)

    print(json.dumps({k: report[k] for k in (
        "baseline", "dim_check", "alpha_scan", "price_ratio_prescan")},
        ensure_ascii=False, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
