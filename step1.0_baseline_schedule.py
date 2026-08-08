"""step1.0_baseline_schedule — 基础调度基线（全篇对照锚点）.

双基线二合一输出（口径决策 D1）:
  local    附件基线复现：st=ArrivalHour 原样执行，超容如实计入 Viol（缺陷标本）
  greedy   基础调度基线：实时固定 + 弹性任务按优先级贪心延后至容量允许（可行化）

统一评估器四指标（Q1 口径，附件1，R1 消纳模板升级）:
  cost_wan  购电成本 − 售电收入（万元）
  carbon_t  碳排放（tCO2）
  nu_pct    新能源利用率 = (直接消纳 + 外送) / 可用（与 D4 口径一致）
  viol_h    超容小时数 Σ 1[O_parallel(r,t) > C_r]
  消纳口径  基线评估: U = c_r(h)·D 日内模板（R1 逆向生成规则，命中率 100%）
            Closure 段(2400+): U = min(W, D)；优化场景: U = min(W, D)
辅助报告: curtail_MWh（弃电） / fail_tasks（贪心未就位任务）

锚点自检（数据核查纪律）:
  local 的 AI_IT_Load 应与 region_time_data.Baseline_AI_IT_Load_MW 吻合；
  NU≈32.9% / 弃电≈775.5 万 MWh / E 30h·F 67h 超容（D3/D4）。

产物（output/baseline/）:
  local_schedule.csv / greedy_schedule.csv   任务级调度（TaskID, Region, StartHour）
  local_hourly.csv / greedy_hourly.csv       区域小时级功率明细
  baseline_metrics.json                      双基线四指标 + 锚点核对
"""
import json

import numpy as np
import pandas as pd

from step0_config import (CLEAN, DATA_RAW, FILES, GPU_POWER_MW, HOURS_TOTAL,
                          OUTPUT, PRIORITY, REGIONS, SETTLE_HOUR)

OUT_BASE = OUTPUT / "baseline"


def load_params() -> dict:
    """软编码：GPU 容量/PUE/SellLimit 一律从数据表读取，不写死。"""
    gpu = pd.read_excel(DATA_RAW / FILES["gpu"], sheet_name="GPU中心基础情况")
    sto = pd.read_excel(DATA_RAW / FILES["storage"], sheet_name="storage_information")
    return {
        "cap": gpu.set_index("Region")["Available_GPU"].to_dict(),
        "pue": gpu.set_index("Region")["PUE"].to_dict(),
        "sell_limit": sto.set_index("Region")["SellLimit_MW"].to_dict(),
    }


def build_occupancy(wt: pd.DataFrame, region_map: pd.Series,
                    start_map: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """按调度方案展开占用（向量化）.

    返回 (parallel, ai_mw) 均为 (2407, 6)：
      parallel[r,t]  并行 GPU 数（容量核算口径，overlap>0 即计入）
      ai_mw[r,t]     AI_IT_Load = Σ GPU×overlap小时×功率（附件1 唯一口径）
    """
    wtw = wt.set_index("TaskID")
    g = wtw["GPU_Demand"].to_numpy(dtype=float)
    dur = wtw["dur_h"].to_numpy(dtype=float)
    power = wtw["TaskType"].map(GPU_POWER_MW).to_numpy(dtype=float)
    rmap = region_map.reindex(wtw.index)
    smap = start_map.reindex(wtw.index)
    ridx = np.array([REGIONS.index(r) for r in rmap], dtype=int)
    st = smap.to_numpy(dtype=float)
    end = st + dur
    h0 = np.floor(st).astype(int)
    h1 = np.clip(np.ceil(end).astype(int), h0, HOURS_TOTAL)
    lens = np.clip(h1 - h0, 0, None)
    tid = np.repeat(np.arange(len(wt), dtype=int), lens)
    h_idx = np.concatenate([np.arange(h0[i], h1[i]) for i in range(len(wt))]) \
        if lens.sum() else np.array([], dtype=int)
    ov = np.clip(np.minimum(end[tid], h_idx + 1) - np.maximum(st[tid], h_idx),
                 0.0, None)
    parallel = np.zeros((HOURS_TOTAL, len(REGIONS)))
    ai_mw = np.zeros((HOURS_TOTAL, len(REGIONS)))
    np.add.at(parallel, (h_idx, ridx[tid]), g[tid])
    np.add.at(ai_mw, (h_idx, ridx[tid]), g[tid] * ov * power[tid])
    return parallel, ai_mw


def build_local_schedule(wt: pd.DataFrame) -> pd.DataFrame:
    """附件基线：st=ArrivalHour、区域=SourceRegion（超容如实）。"""
    return pd.DataFrame({
        "TaskID": wt["TaskID"],
        "Region": wt["SourceRegion"],
        "StartHour": wt["ArrivalHour"],
    })


def build_greedy_schedule(wt: pd.DataFrame, cap: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基础调度基线：实时固定 + 弹性任务贪心延后至容量允许（不迁移区域）.

    返回 (schedule, fail)：fail 记录无法按时就位的任务（Viol 兜底）。
    """
    cap_arr = np.array([cap[r] for r in REGIONS], dtype=float)
    occ = np.zeros((HOURS_TOTAL, len(REGIONS)), dtype=float)
    w = wt.sort_values(["Priority", "ArrivalHour"], ascending=[False, True])
    rows, fail = [], []

    for rec in w.itertuples(index=False):
        r = REGIONS.index(rec.SourceRegion)
        g = float(rec.GPU_Demand)
        dur = float(rec.dur_h)
        h1_lim = int(np.ceil(rec.ArrivalHour + dur))
        if rec.TaskType == "RealTimeInference":
            st = int(rec.ArrivalHour)
            rows.append((rec.TaskID, rec.SourceRegion, st))
            h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
            if h1 > st:
                occ[st:h1, r] += g
            continue
        last = min(int(rec.LatestFinishHour - dur), HOURS_TOTAL - 1)
        placed = False
        for st in range(int(rec.ArrivalHour), last + 1):
            h1 = min(int(np.ceil(st + dur)), HOURS_TOTAL)
            if h1 <= st:
                break
            if occ[st:h1, r].max() + g <= cap_arr[r]:
                occ[st:h1, r] += g
                rows.append((rec.TaskID, rec.SourceRegion, st))
                placed = True
                break
        if not placed:
            fail.append(rec.TaskID)
            rows.append((rec.TaskID, rec.SourceRegion, int(rec.ArrivalHour)))

    sched = pd.DataFrame(rows, columns=["TaskID", "Region", "StartHour"])
    fail_df = pd.DataFrame({"TaskID": fail})
    return sched, fail_df


def fit_consume_ratio(rt: pd.DataFrame) -> dict:
    """从题目基线数据逆向消纳模板 c_r(h)（R1 升级：常数 c_r → 日内模板）.

    实证（R1/sum_3）: U/D 对每个小时跨天完全恒定（max std=0.000000），
    生成规则 U = c_r(h)·D(r,t)，c_r(h) 为确定性日内模板，与 W 无关：
      A/B/C: 夜间平台(0-6h) + 白天单峰(12-15h)，归一化形状完全相同（基准等差 0.12/0.15/0.18）
      D/F:   完美单正弦（R²=1.0000，相位 φ≈π/3 全域一致）
      E:     平台 0.55 + 白天对称钟形（峰 0.73@12h）
    尾段 2352-2399 与训练模板完全一致（max dev=0.000000）；Closure 段（2400-2406）
    U = min(W, D)（hit=1.0 实证）→ 评估器分口径处理。
    返回 consume_ratio[r] = np.ndarray(24)（模板小时表）。
    """
    main = rt[rt["DataPeriod"] == "Main_0_2399"].copy()
    main["h"] = main["Hour"] % 24
    main["ud"] = main["UsedRenewable_MW"] / main["Total_Load_MW"]
    fit, stats_rows = {}, []
    for r in REGIONS:
        m = main[main["Region"] == r]
        g = m.groupby("h")["ud"].agg(["mean", "std"])
        c_h = g["mean"]
        max_std = float(g["std"].max())
        pred = (c_h.loc[m["h"].values].to_numpy()
                * m["Total_Load_MW"].to_numpy())
        hit = np.isclose(m["UsedRenewable_MW"].to_numpy(), pred,
                         atol=0.6).mean()
        rel_err = float(np.abs(pred - m["UsedRenewable_MW"].to_numpy())
                        .mean() / m["Total_Load_MW"].to_numpy().mean())
        sine = _fit_sine(c_h.index.to_numpy(), c_h.to_numpy())
        fit[r] = c_h.to_numpy()
        stats_rows.append({
            "Region": r,
            "template_hit_rate": float(hit),
            "max_std_within_hour": max_std,
            "mean_rel_err": rel_err,
            "sine1": sine["sine1"],
            "sine2": sine["sine2"],
            "c_h": np.round(c_h.to_numpy(), 6).tolist(),
        })
    clos = rt[rt["DataPeriod"] == "Closure_2400_2406"]
    clos_hit = np.isclose(clos["UsedRenewable_MW"],
                          np.minimum(clos["AvailableRenewable_MW"],
                                     clos["Total_Load_MW"]), atol=0.6).mean()
    return {"consume_ratio": fit, "fit_stats": stats_rows,
            "closure_minWD_hit": float(clos_hit),
            "form": "U = c_r(h)*D（日内模板，R1 实证 max std=0）; "
                    "Closure 段 U=min(W,D)",
            "note": "常数 c_r 形式（U=min(W,cD)）命中率 D/F 仅 8-24% 系形式错误；"
                    "模板形式构造性命中 100%（R1 修正）"}


def _fit_sine(h: np.ndarray, y: np.ndarray) -> dict:
    """c(h) 正弦参数化（论文素材）：线性最小二乘（24h + 12h 谐波）.

    模型: c(h) = a + b1·sin(2πh/24) + b2·cos(2πh/24) [+ c1·sin(2πh/12) + c2·cos(2πh/12)]
    返回振幅-相位形式 (A, φ) 与 R²。
    """
    out = {}
    for nm, k in (("sine1", 1), ("sine2", 2)):
        cols = [np.ones_like(h, dtype=float)]
        for kk in range(1, k + 1):
            cols += [np.sin(2 * np.pi * kk * h / 24),
                     np.cos(2 * np.pi * kk * h / 24)]
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        a, b1, b2 = beta[0], beta[1], beta[2]
        amp = float(np.hypot(b1, b2))
        phi = float(np.arctan2(b2, b1))
        rec = {"a": float(a), "amp": amp, "phi_rad": phi,
               "r2": float(r2)}
        if nm == "sine2":
            c1, c2 = beta[3], beta[4]
            rec["amp_12h"] = float(np.hypot(c1, c2))
            rec["phi_12h"] = float(np.arctan2(c2, c1))
        out[nm] = rec
    return out


def evaluate_schedule(sched: pd.DataFrame, wt: pd.DataFrame, rt: pd.DataFrame,
                      params: dict, consume_ratio: dict | None = None,
                      include_baseline_charge: bool = False
                      ) -> tuple[dict, pd.DataFrame]:
    """统一评估器：四指标 + 弃电 + 区域小时级明细（step1.2/1.2+ 复用）.

    口径（Q1 无储能，R1 升级 + B9 充电项选项）：
      consume_ratio 给定（模板格式 {r: np.ndarray(24)}）→
        U = c_r(h)·D（主时段 0-2399 日内模板）；U = min(W, D)（Closure 2400+）
      consume_ratio 给定（标量格式 {r: float}，旧版兼容）→ U=min(W, c_r·D)
      consume_ratio None → U=min(W, D)（新能源全覆盖口径，优化场景用）
      S=min(SellLimit, W−U)；G=D−U（自洽功率平衡）；Curtail=W−U−S
      include_baseline_charge=True（B9，对照题目基线口径）→ 基线充电 RC 模板
        计入利用率分子与弃电扣减：NU=(U+S+RC)/W；Curtail=W−U−S−RC
        （实证：评估器弃电 vs 题目 775.5 万 偏差 9.18 万 = 充电 20.8 万 +
         外送口径差 11.6 万——B9 消除充电项部分）
    """
    region_map = sched.set_index("TaskID")["Region"]
    start_map = sched.set_index("TaskID")["StartHour"]
    parallel, ai_mw = build_occupancy(wt, region_map, start_map)

    rtv = rt.pivot_table(index="Hour", columns="Region",
                         values=["NonAI_IT_Load_MW", "AvailableRenewable_MW",
                                 "ElectricityPrice_CNY_per_MWh",
                                 "SellPrice_CNY_per_MWh",
                                 "CarbonIntensity_tCO2_per_MWh",
                                 "RenewableCharge_MW"],
                         aggfunc="first").reindex(index=range(HOURS_TOTAL))
    nonai = rtv["NonAI_IT_Load_MW"][REGIONS].fillna(0.0).to_numpy()
    w = rtv["AvailableRenewable_MW"][REGIONS].fillna(0.0).to_numpy()
    price = rtv["ElectricityPrice_CNY_per_MWh"][REGIONS].fillna(0.0).to_numpy()
    sellp = rtv["SellPrice_CNY_per_MWh"][REGIONS].fillna(0.0).to_numpy()
    carb = rtv["CarbonIntensity_tCO2_per_MWh"][REGIONS].fillna(0.0).to_numpy()

    pue_arr = np.array([params["pue"][r] for r in REGIONS])
    sell_arr = np.array([params["sell_limit"][r] for r in REGIONS])
    cap_arr = np.array([params["cap"][r] for r in REGIONS])

    total = (nonai + ai_mw) * pue_arr
    if consume_ratio is not None:
        first = next(iter(consume_ratio.values()))
        if isinstance(first, (np.ndarray, list, tuple)):
            c_mat = np.stack([np.asarray(consume_ratio.get(r, np.ones(24)),
                                         dtype=float) for r in REGIONS],
                             axis=1)
            cap_d = c_mat[np.arange(HOURS_TOTAL) % 24]
            w_used = np.minimum(w, cap_d * total)
            closure = np.arange(HOURS_TOTAL) >= 2400
            w_used[closure] = np.minimum(w, total)[closure]
        else:
            c_arr = np.array([consume_ratio.get(r, 1.0) for r in REGIONS])
            w_used = np.minimum(w, c_arr * total)
    else:
        w_used = np.minimum(w, total)
    s = np.minimum(sell_arr, np.maximum(0.0, w - w_used))
    g = np.maximum(0.0, total - w_used)
    curtail = w - w_used - s
    if include_baseline_charge:
        # B9：题目基线充电模板（RenewableCharge_MW）——对照口径
        rc = rtv["RenewableCharge_MW"][REGIONS].fillna(0.0).to_numpy()
        curtail = np.maximum(0.0, curtail - rc)
        nu_num = w_used + s + rc
    else:
        nu_num = w_used + s

    metrics = {
        "cost_wan": float((g * price - s * sellp).sum() / 1e4),
        "carbon_t": float((g * carb).sum()),
        "nu_pct": float(nu_num.sum() / w.sum() * 100.0),
        "curtail_MWh": float(curtail.sum()),
        "viol_h": int((parallel > cap_arr).sum()),
        "viol_by_region": {r: int((parallel[:, i] > cap_arr[i]).sum())
                           for i, r in enumerate(REGIONS)},
        "total_load_MWh": float(total.sum()),
        "ai_load_MWh": float(ai_mw.sum()),
    }
    hourly = pd.DataFrame({
        "Hour": np.tile(np.arange(HOURS_TOTAL), len(REGIONS)),
        "Region": np.repeat(REGIONS, HOURS_TOTAL),
        "AI_IT_Load_MW": ai_mw.ravel(),
        "Total_Load_MW": total.ravel(),
        "Renewable_Used_MW": w_used.ravel(),
        "GridSell_MW": s.ravel(),
        "GridPurchase_MW": g.ravel(),
        "Curtailment_MW": curtail.ravel(),
        "OverCapacity": (parallel > cap_arr).ravel(),
    })
    return metrics, hourly


def anchor_check(local_hourly: pd.DataFrame, rt: pd.DataFrame) -> dict:
    """锚点自检：local 调度的 AI_IT_Load vs 题目基线列.

    总量锚定（严格）：6 位有效数字一致（GPU-hour×功率守恒）；
    逐时参考（宽松）：题目基线列存在生成器内部口径差异（数据列间
    不自洽，恒等式残差 ~110MW），仅报告不作裁决。
    """
    m = local_hourly.merge(
        rt[["Hour", "Region", "Baseline_AI_IT_Load_MW"]], on=["Hour", "Region"])
    diff = m["AI_IT_Load_MW"] - m["Baseline_AI_IT_Load_MW"]
    return {
        "ai_load_sum_ours": float(m["AI_IT_Load_MW"].sum()),
        "ai_load_sum_baseline": float(m["Baseline_AI_IT_Load_MW"].sum()),
        "ai_load_sum_rel_err": float(diff.sum() / m["Baseline_AI_IT_Load_MW"].sum()),
        "ai_load_hourly_rmse": float(np.sqrt((diff ** 2).mean())),
        "ai_load_hourly_mean_abs": float(diff.abs().mean()),
        "note": "总量锚定严格（sum_rel_err≈1e-7）；逐时差异源于题目基线列内部口径，不影响评估器（附件1 精确 Overlap 口径）",
    }


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    wt = pd.read_csv(CLEAN / "workload_clean.csv")
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    params = load_params()

    local_sched = build_local_schedule(wt)
    greedy_sched, greedy_fail = build_greedy_schedule(wt, params["cap"])

    consume_fit = fit_consume_ratio(rt)
    consume_ratio = consume_fit["consume_ratio"]
    local_metrics, local_hourly = evaluate_schedule(
        local_sched, wt, rt, params, consume_ratio)
    greedy_metrics, greedy_hourly = evaluate_schedule(
        greedy_sched, wt, rt, params, consume_ratio)
    greedy_metrics["fail_tasks"] = int(len(greedy_fail))

    anchors = anchor_check(local_hourly, rt)
    consume_report = {
        **consume_fit,
        "consume_ratio": {r: v.tolist() for r, v in consume_fit["consume_ratio"].items()},
    }
    report = {"local": local_metrics, "greedy": greedy_metrics,
              "anchor": anchors, "consume_fit": consume_report}

    local_sched.to_csv(OUT_BASE / "local_schedule.csv", index=False)
    greedy_sched.to_csv(OUT_BASE / "greedy_schedule.csv", index=False)
    greedy_fail.to_csv(OUT_BASE / "greedy_fail_tasks.csv", index=False)
    local_hourly.to_csv(OUT_BASE / "local_hourly.csv", index=False)
    greedy_hourly.to_csv(OUT_BASE / "greedy_hourly.csv", index=False)
    with open(OUT_BASE / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
