"""step3.10_q3_sobol_interaction — #18 Sobol 交互热图：S1>1 的"强交互"证据化.

背景: E|carbon S1_Cap=2.06、A|ramp S1_Cap=1.78/S1_ηc=1.19、E|ramp S1_sell=0.93>
ST_sell=0.80（违反 S1≤ST）——仅标"unreliable"欠机理解释（Q3 反思 #18）。

方法:
  ① S1 vs ST 对比表（现有 q3_sobol.json 数据）——ST≈1 且 S1 异常 = 强交互/
    非单调证据
  ② 2D 交互切片: 对嫌疑参数对 (Cap,ηc)×(A|ramp)、(Cap,SellLimit)×(E|carbon)，
    固定其余参数中值，7×7 网格扫描 M3_final LP → 响应面
  ③ 交互强度: Δ_int(a,b) = f(a,b) − f(a0,b) − f(a,b0) + f(a0,b0)（非加性）；
    非单调检测: 响应面沿任一轴的非单调段占比
  ④ 可视化: 响应面热图 + 交互切片（fig_q3_sobol_interaction.png）

产物: output/q3/q3_sobol_interaction.json + figures/step3/fig_q3_sobol_interaction.png
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step0_config import FIGURES, OUTPUT

OUT_Q3 = OUTPUT / "q3"
FIG_Q3 = FIGURES / "step3"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

PARAMS = ["Cap", "eta_c", "eta_d", "price_scale", "sell_scale"]


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
    return {**c, "s33": s33}


def lp_black(c, d0, ch, dh, theta: np.ndarray, region: str) -> dict:
    """θ → M3_final LP（step3.6 同款参数缩放规则）。"""
    d = dict(d0)
    d["cap_mwh"] = d0["cap_mwh"] * theta[0]
    d["min_soc"] = d0["min_soc"] * theta[0]
    d["eta_c"] = theta[1]
    d["eta_d"] = theta[2]
    d["price"] = d0["price"] * theta[3]
    d["sellp"] = d0["sellp"] * theta[3]
    d["sell_lim"] = d0["sell_lim"] * theta[4]
    res = c["s33"].solve_m3(d, ch, dh, region=region)
    if res["cost_wan"] is None:
        return None
    return res


def interaction_slice(c, region: str, metric: str, pair: tuple[int, int],
                      lo: np.ndarray, hi: np.ndarray, grid: int = 7) -> dict:
    """2D 交互切片：pair 参数网格扫描（其余参数中值），响应面 + 非加性。"""
    d0 = c["s33"]._load_region_data(region)
    d0["c_h"] = np.asarray(c["consume"][region], dtype=float)
    ch, dh = c["tpl"][region]["charge_hours"], c["tpl"][region]["discharge_hours"]
    mid = 0.5 * (lo + hi)
    a_axis = np.linspace(lo[pair[0]], hi[pair[0]], grid)
    b_axis = np.linspace(lo[pair[1]], hi[pair[1]], grid)
    Z = np.full((grid, grid), np.nan)
    for i, a in enumerate(a_axis):
        for j, b in enumerate(b_axis):
            th = mid.copy()
            th[pair[0]] = a
            th[pair[1]] = b
            r = lp_black(c, d0, ch, dh, th, region)
            if r is not None:
                Z[i, j] = r[metric]
    # 非加性交互项 Δ_int 在中点邻域
    i0 = grid // 2
    f00 = Z[i0, i0]
    d_int = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            if np.isnan(Z[i, j]) or np.isnan(f00):
                continue
            fa0 = Z[i, i0]
            f0b = Z[i0, j]
            if np.isnan(fa0) or np.isnan(f0b):
                continue
            d_int[i, j] = Z[i, j] - fa0 - f0b + f00
    # 非单调检测：沿轴中值截面的符号变化
    def nonmonotonic(axis_vals, slice_vals):
        s = np.sign(np.diff(slice_vals))
        return float(np.mean(s != 0) ) if len(s) > 0 else 0.0, \
            bool((np.diff(np.sign(np.diff(slice_vals))) != 0).any())
    row_mid = Z[i0, :]
    col_mid = Z[:, i0]
    nm_a = nonmonotonic(a_axis, row_mid)
    nm_b = nonmonotonic(b_axis, col_mid)
    return {"region": region, "metric": metric,
            "pair": [PARAMS[pair[0]], PARAMS[pair[1]]],
            "Z": Z.tolist(), "a_axis": a_axis.tolist(),
            "b_axis": b_axis.tolist(),
            "interaction_max": round(float(np.nanmax(np.abs(d_int))), 3),
            "interaction_rel": round(float(
                np.nanmax(np.abs(d_int)) / max(abs(f00), 1e-9)), 3),
            "nonmonotonic_a": nm_a[1], "nonmonotonic_b": nm_b[1],
            "f_mid": round(float(f00), 2)}


def main() -> None:
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)
    c = load_ctx()
    lo = np.array([0.8, 0.90, 0.90, 0.9, 0.5])
    hi = np.array([1.2, 0.96, 0.95, 1.1, 1.5])

    slices = [
        interaction_slice(c, "RegionA", "ramp_p95_MW", (0, 1), lo, hi),
        interaction_slice(c, "RegionE", "carbon_t", (0, 4), lo, hi),
        interaction_slice(c, "RegionE", "ramp_p95_MW", (0, 4), lo, hi),
    ]
    # S1 vs ST 对比（现有数据）
    sob = json.loads((OUT_Q3 / "q3_sobol.json").read_text(encoding="utf-8"))
    s1st = {}
    for r, m in (("RegionE", "carbon_t"), ("RegionA", "ramp_p95_MW"),
                 ("RegionE", "ramp_p95_MW")):
        v = sob["sobol"][r][m]
        s1st[f"{r}|{m}"] = {"S1": v["S1"], "ST": v["ST"],
                            "unreliable": v["unreliable"]}
    report = {"S1_vs_ST": s1st, "slices": slices,
              "caliber": ("#18 Sobol 交互热图：S1>1（E carbon 2.06/A ramp 1.78）"
                          "的机理证据化——ST≈1 且 S1 异常=强交互/非单调；"
                          "2D 切片网格 7×7 固定其余参数中值；交互强度=非加性"
                          "Δ_int 相对中值响应的比值；非单调=中值截面符号变化")}
    with open(OUT_Q3 / "q3_sobol_interaction.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)

    # 图：三个切片的响应面
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sl in zip(axes, slices):
        Z = np.array(sl["Z"])
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=[sl["b_axis"][0], sl["b_axis"][-1],
                               sl["a_axis"][0], sl["a_axis"][-1]],
                       cmap="viridis")
        ax.set_xlabel(sl["pair"][1])
        ax.set_ylabel(sl["pair"][0])
        ax.set_title(f"{sl['region']} {sl['metric']}\n"
                     f"交互 Δ={sl['interaction_rel']} 非单调A={sl['nonmonotonic_a']}"
                     f"/B={sl['nonmonotonic_b']}", fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Q3 Sobol 交互切片（S1>1 的机理：强交互/非单调）")
    fig.tight_layout()
    fig.savefig(FIG_Q3 / "fig_q3_sobol_interaction.png", bbox_inches="tight")
    plt.close(fig)
    for sl in slices:
        print(f"{sl['region']}|{sl['metric']} pair={sl['pair']} "
              f"交互rel={sl['interaction_rel']} 非单调A={sl['nonmonotonic_a']}"
              f"/B={sl['nonmonotonic_b']}")


if __name__ == "__main__":
    main()
