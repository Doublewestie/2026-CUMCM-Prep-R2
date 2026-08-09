"""step3.1_scenario_mpc — Q3 铺路：场景生成 + K-means 缩减 + 单窗口滚动验证.

范围（M4 前铺路）: 场景生成框架 + 缩减质量报告 + 单窗口滚动 LP 验证
（终态约束的窗口效应——M4 MPC 设计依据）。不求解 MPC 全量。

方法（PLAN_details §8.5-8.6）:
  场景: W 模板 × (1+ε)，ε~AR(1)（φ=0.8，σ∈{10%,20%,30%}）→ 每区 64 场景
  缩减: K-means（sklearn）→ 8-16 个代表场景 + 缩减质量（代表场景均值 vs 全场景均值）
  窗口验证: 24h 滚动 LP（step3.0 求解器）vs 全知 LP——终态约束窗口效应量化

产物（output/q3/）: scenario_report.json + scenario_fig.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"


def gen_scenarios(rt: pd.DataFrame, r: str, n_scen: int = 64,
                  sigma: float = 0.2, phi: float = 0.8) -> np.ndarray:
    """AR(1) 扰动场景：W 模板 × (1+ε_t)，ε_t=φ·ε_{t-1}+σ·z_t。

    L5（T1 实证投产）: W 模板 = round(800 + 300·sin(2π(h−4)/24), 2) 的确定性
    阶梯模板（数据列=公式，max 残差 0.0043MW）——扰动乘在确定性模板上。
    """
    sub = rt[rt.Region == r].sort_values("Hour")
    W = sub["AvailableRenewable_MW"].to_numpy()
    h = np.arange(len(W))
    formula = np.round(800 + 300 * np.sin(2 * np.pi * (h - 4) / 24), 2)
    max_dev = float(np.max(np.abs(W - formula)))
    assert max_dev < 0.05, f"{r} W 模板与公式偏差 {max_dev}（T1 实证应 <0.005）"
    T = len(W)
    rng = np.random.RandomState(hash(r) % 2**32)
    S = np.zeros((n_scen, T))
    for s in range(n_scen):
        z = rng.randn(T)
        eps = np.zeros(T)
        for t in range(1, T):
            eps[t] = phi * eps[t - 1] + sigma * z[t]
        S[s] = W * np.clip(1 + eps, 0.1, 2.0)
    return S


def reduce_scenarios(S: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray, float]:
    """K-means 缩减：返回代表场景、权重、质量（代表均值 vs 全均值偏差）。"""
    km = KMeans(n_clusters=k, random_state=0, n_init=5).fit(S)
    reps = km.cluster_centers_
    w = np.bincount(km.labels_, minlength=k) / len(km.labels_)
    full_mean = S.mean(axis=0)
    rep_mean = (reps * w[:, None]).sum(axis=0)
    quality = float(np.abs(rep_mean - full_mean).mean() / full_mean.mean())
    return reps, w, quality


def rolling_window_lp(r: str, w: int = 24) -> dict:
    """单窗口滚动 LP 验证（终态约束窗口效应）。"""
    import importlib.util
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "s30", root / "step3.0_lp_baseline.py")
    s30 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s30)
    spec2 = importlib.util.spec_from_file_location(
        "s10", root / "step1.0_baseline_schedule.py")
    s10 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s10)
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    consume = s10.fit_consume_ratio(rt)["consume_ratio"]
    d = s30.load_region_data(r)
    d["c_h"] = np.array(consume[r], dtype=float)
    # 全知
    full = s30.solve_region(d)
    # 滚动：每 24h 窗口独立求解（SOC 初值=窗口前真实 SOC 近似用 init）
    costs = []
    for t0 in range(0, 2407, w):
        tw = min(w, 2407 - t0)
        d_w = dict(d)
        d_w["D"] = d["D"][t0:t0 + tw]
        d_w["W"] = d["W"][t0:t0 + tw]
        d_w["price"] = d["price"][t0:t0 + tw]
        d_w["sellp"] = d["sellp"][t0:t0 + tw]
        d_w["carbon"] = d["carbon"][t0:t0 + tw]
        res = s30.solve_region(d_w, tw)
        if res["cost_wan"] is not None:
            costs.append(res["cost_wan"] * 1e4)
    roll_cost = float(np.sum(costs))
    return {"full_cost_wan": full["cost_wan"],
            "rolling_cost_wan": roll_cost / 1e4,
            "window_effect_pct": float((roll_cost / 1e4 - full["cost_wan"])
                                       / abs(full["cost_wan"]) * 100),
            "note": "窗口效应 = 滚动（终态约束每窗重置）vs 全知的成本差"}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    report = {"scenarios": {}, "window_effect": {}}
    for r in ["RegionD", "RegionE"]:
        S = gen_scenarios(rt, r)
        reps, w, qual = reduce_scenarios(S, 12)
        report["scenarios"][r] = {
            "n_gen": S.shape[0], "k": 12, "sigma": 0.2, "phi": 0.8,
            "quality_rel_err": round(qual, 4),
            "rep_weights": [round(x, 3) for x in w[:5]]}
        we = rolling_window_lp(r)
        report["window_effect"][r] = we
        print(f"[{r}] 缩减质量 {qual:.4f} | 窗口效应 {we['window_effect_pct']:.1f}%",
              flush=True)
    # 场景图（E 区前 8 场景 + 模板）
    S = gen_scenarios(rt, "RegionE")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s in range(8):
        ax.plot(S[s, :168], lw=0.7, alpha=0.6)
    tpl = rt[(rt.Region == "RegionE") & (rt.Hour < 168)]["AvailableRenewable_MW"]
    ax.plot(tpl, "k-", lw=2, label="模板")
    ax.set_xlabel("Hour"); ax.set_ylabel("MW")
    ax.set_title("Q3 MPC 场景生成（AR(1) 扰动，E 区前 168h）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_scenarios.png", bbox_inches="tight")
    plt.close(fig)
    report["caliber"] = ("AR(1) 扰动模板（σ=0.2, φ=0.8）；K-means k=12 缩减；"
                         "窗口效应=24h 滚动 LP vs 全知 LP；铺路产物，MPC 全量留 M4")
    with open(OUT_Q3 / "scenario_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
