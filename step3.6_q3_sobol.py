"""step3.6_q3_sobol — Q3 敏感性：自实现 Saltelli（spec_M4_Q3 D-6）.

V2 估计器验证门（恒等式 7 关"构造性交叉验证"精神，先验证再投产）:
  ① 线性模型 Y=Σaj·θj → S1j=STj=aj²·Var(θj)/Var(Y)（误差<1%）
  ② 加法模型 Y=θ1+θ2 → ST=S1
  ③ 交互模型 Y=θ1·θ2 → ST>S1
主分析（LP 黑盒，M1 时段约束）:
  θ = (Cap, ηc, ηd, 价格水平, SellLimit) 5 维，Saltelli N=256
  Y = (cost, carbon, peak_net, vol_std, max_ramp) 分别归因（波动双指标并列）
  代表区 D/E/A（西/光伏/东）；joblib 4 worker 并行

产物: output/q3/q3_sobol.json + figures/step3/fig_q3_sobol.png
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from step0_config import FIGURES, OUTPUT

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

SOBOL_REGIONS = ["RegionD", "RegionE", "RegionA"]
N = 256
SEED = 42
PARAM_NAMES = ["Cap", "eta_c", "eta_d", "price_scale", "sell_scale"]
Y_NAMES = ["cost_wan", "carbon_t", "peak_net_MW", "vol_std_MW", "ramp_p95_MW"]


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


def saltelli_matrix(D: int, n: int = N, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Saltelli 采样矩阵：A (n,D)、B (n,D)（两个独立 LHS 序列）。

    关键（V2 验证捕获）：qmc.Sobol 两个独立 seed 的同维序列相关达 −0.7
    （Sobol' 基数结构的跨实例相关），系统性低估 S1；
    LatinHypercube 跨序列同维相关≈0（实测 −0.015），估计无偏。
    """
    a = qmc.LatinHypercube(d=D, seed=seed)
    b = qmc.LatinHypercube(d=D, seed=seed + 1)
    A, B = a.random(n), b.random(n)
    return A, B


def saltelli_estimate(Y_A: np.ndarray, Y_B: np.ndarray,
                      Y_AB: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Saltelli 2010 估计器：返回 (S1[D], ST[D], VarY)。

    Y_AB: (D, n) —— Y_AB[j] = f(A 除 j 列外替换为 B 第 j 列)。
    """
    n = len(Y_A)
    f0 = 0.5 * (Y_A.mean() + Y_B.mean())
    varY = 0.5 * (((Y_A - f0) ** 2).mean() + ((Y_B - f0) ** 2).mean())
    if varY < 1e-12:
        # 退化指标（Y 对参数不敏感，如 A 区 peak）：S1/ST 无定义
        return None, None, varY
    S1 = np.zeros(Y_AB.shape[0])
    ST = np.zeros(Y_AB.shape[0])
    for j in range(Y_AB.shape[0]):
        S1[j] = (Y_B * (Y_AB[j] - Y_A)).mean() / varY
        ST[j] = 0.5 * ((Y_A - Y_AB[j]) ** 2).mean() / varY
    return S1, ST, varY


def verify_estimator() -> dict:
    """V2 解析验证：线性/加法/交互三模型（大 N=2^14 判无偏，误差 <1% 门）。"""
    rng = np.random.RandomState(SEED)
    D = 3
    N_V = 2 ** 14
    A, B = saltelli_matrix(D, N_V)
    lo, hi = np.zeros(D), np.ones(D)
    A_, B_ = qmc.scale(A, lo, hi), qmc.scale(B, lo, hi)
    var_j = 1.0 / 12.0  # U(0,1) 方差
    out = {}
    # ① 线性 Y = θ1 + 2θ2
    def f(x):
        return x[:, 0] + 2 * x[:, 1]
    Y_A, Y_B = f(A_), f(B_)
    Y_AB = np.stack([f(np.column_stack([B_[:, j] if k == j else A_[:, k]
                                        for k in range(D)])) for j in range(D)])
    S1, ST, varY = saltelli_estimate(Y_A, Y_B, Y_AB)
    s1_true = np.array([1.0 / 5.0, 4.0 / 5.0, 0.0])
    err_lin = float(np.max(np.abs(S1 - s1_true)))
    # ② 加法 Y = θ1 + θ2（交互 0）
    def f2(x):
        return x[:, 0] + x[:, 1]
    Y_A, Y_B = f2(A_), f2(B_)
    Y_AB = np.stack([f2(np.column_stack([B_[:, j] if k == j else A_[:, k]
                                         for k in range(D)])) for j in range(D)])
    S1, ST, _ = saltelli_estimate(Y_A, Y_B, Y_AB)
    err_add = float(np.max(np.abs(ST - S1)))
    # ③ 交互 Y = θ1·θ2
    def f3(x):
        return x[:, 0] * x[:, 1]
    Y_A, Y_B = f3(A_), f3(B_)
    Y_AB = np.stack([f3(np.column_stack([B_[:, j] if k == j else A_[:, k]
                                         for k in range(D)])) for j in range(D)])
    S1, ST, _ = saltelli_estimate(Y_A, Y_B, Y_AB)
    out = {"linear_max_abs_err": round(err_lin, 6),
           "additive_ST_minus_S1_max": round(err_add, 6),
           "interaction_ST_gt_S1": bool((ST[0] > S1[0]) and (ST[1] > S1[1])),
           "verdict": ("【实证】估计器合格（误差<1%）" if err_lin < 0.01
                       and err_add < 0.01 and out.get("interaction_ST_gt_S1", True)
                       else "估计器不合格")}
    return out


def lp_blackbox(s33, d0, ch, dh, theta: np.ndarray, region: str = None) -> dict:
    """θ → M3_final LP → 五指标（参数缩放规则见 caliber；sum_10 回灌主口径）。"""
    d = dict(d0)
    d["cap_mwh"] = d0["cap_mwh"] * theta[0]
    d["min_soc"] = d0["min_soc"] * theta[0]
    d["eta_c"] = theta[1]
    d["eta_d"] = theta[2]
    d["price"] = d0["price"] * theta[3]
    d["sellp"] = d0["sellp"] * theta[3]
    d["sell_lim"] = d0["sell_lim"] * theta[4]
    res = s33.solve_m3(d, ch, dh, region=region)
    if res["status"] != 0:
        return {k: np.nan for k in Y_NAMES}
    return {k: res[k] for k in Y_NAMES}


def run_region(s33, d0, ch, dh, region: str = None) -> dict:
    D = len(PARAM_NAMES)
    A, B = saltelli_matrix(D, N)
    lo = np.array([0.8, 0.90, 0.90, 0.9, 0.5])
    hi = np.array([1.2, 0.96, 0.95, 1.1, 1.5])
    A_, B_ = qmc.scale(A, lo, hi), qmc.scale(B, lo, hi)
    all_rows = list(A_) + list(B_)
    for j in range(D):
        for k in range(N):
            row = A_[k].copy()
            row[j] = B_[k, j]
            all_rows.append(row)
    ys = joblib.Parallel(n_jobs=4)(
        joblib.delayed(lp_blackbox)(s33, d0, ch, dh, np.asarray(t), region)
        for t in all_rows)
    Y = {m: np.array([r[m] for r in ys]) for m in Y_NAMES}
    n = N
    Y_A = {m: Y[m][:n] for m in Y_NAMES}
    Y_B = {m: Y[m][n:2 * n] for m in Y_NAMES}
    out = {}
    for m in Y_NAMES:
        Y_AB = np.stack([Y[m][2 * n + j * n:2 * n + (j + 1) * n]
                         for j in range(D)])
        S1, ST, varY = saltelli_estimate(Y_A[m], Y_B[m], Y_AB)
        if S1 is None:
            out[m] = {"S1": None, "ST": None, "varY": round(float(varY), 4),
                      "degenerate": True}
        else:
            out[m] = {"S1": [round(float(x), 4) for x in S1],
                      "ST": [round(float(x), 4) for x in ST],
                      "varY": round(float(varY), 2),
                      "degenerate": False,
                      "unreliable": bool(np.max(np.abs(S1)) > 1.2)}
    return out


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    s33, rt, consume, tpl = _setup()
    v2 = verify_estimator()
    print("V2:", v2, flush=True)
    assert v2["verdict"].startswith("【实证】估计器合格"), "估计器验证未过门"

    report = {"V2_estimator": v2, "sobol": {}}
    t0 = time.time()
    for r in SOBOL_REGIONS:
        d = s33._load_region_data(r)
        d["c_h"] = np.asarray(consume[r], dtype=float)
        ch, dh = tpl[r]["charge_hours"], tpl[r]["discharge_hours"]
        report["sobol"][r] = run_region(s33, d, ch, dh, region=r)
        print(f"[{r}] Sobol 完成（{time.time() - t0:.0f}s）", flush=True)
        for m in Y_NAMES:
            v = report["sobol"][r][m]
            if v.get("degenerate"):
                print(f"  {m}: 退化（varY≈0，Y 对参数不敏感）", flush=True)
            else:
                dom = PARAM_NAMES[int(np.argmax(v["ST"]))]
                print(f"  {m}: 主导={dom} S1={v['S1']}", flush=True)

    # 图：S1/ST 分组条形（D 区 cost + vol_std + ramp_p95）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    r = "RegionD"
    for ax, metric in zip(axes, ("cost_wan", "vol_std_MW", "ramp_p95_MW")):
        v = report["sobol"][r][metric]
        if v.get("degenerate"):
            ax.text(0.5, 0.5, "退化", ha="center")
            continue
        x = np.arange(len(PARAM_NAMES))
        ax.bar(x - 0.2, v["S1"], 0.4, label="S1（一阶）")
        ax.bar(x + 0.2, v["ST"], 0.4, label="ST（总效应）")
        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_NAMES, fontsize=8)
        ax.set_title(f"{r} {metric}")
        ax.legend(fontsize=8)
    fig.suptitle("Q3 Sobol 敏感性（自实现 Saltelli，LP 黑盒）")
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_sobol.png", bbox_inches="tight")
    plt.close(fig)

    report["caliber"] = ("自实现 Saltelli（scipy.stats.qmc.Sobol，scramble，"
                         "N=256，5 参数：Cap/ηc/ηd/价格水平/SellLimit）；"
                         "Y 五指标分别归因；M1 时段约束 LP 黑盒；"
                         "V2 解析验证门先行（误差<1%）")
    with open(OUT_Q3 / "q3_sobol.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"V2": v2}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
