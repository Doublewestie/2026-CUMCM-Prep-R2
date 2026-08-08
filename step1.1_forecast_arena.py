"""step1.1_forecast_arena — 四家族预测竞技榜（42 序列 × 全量评估）.

协议（防泄露三段，PLAN_details §5.3）:
  类选  0-2351 内部 5 折 TimeSeriesSplit（不 shuffle）→ 报告 mean±std
  校准  2352-2375 仅输出融合分位数供 step1.2 κ_ε 校准（本文件不触碰）
  测试  2376-2399 冻结，仅最终一次性评估（不回头调参）
分层指标（§5.3）:
  任务侧 覆盖率(90% 区间)/区间宽度/pinball 裁决；MAPE/RMSE 仅报告
  能源侧 MAPE/RMSE 裁决；覆盖率参考
验证门（CONSTITUTION）:
  任务侧 覆盖率校准达标(<80%→≥90%) 或 pinball 降幅≥5%
  能源侧 MAPE 相对降幅≥5%
融合: Ridge stacking → 最优单模型 → 统计基线（负增益回退链）
深度模型: 小网络（hidden 32 / 2 层 / epochs≤40 / 早停），GPU 自动

产物（output/forecast/）:
  arena_table.csv      对象×模型 竞技榜（指标 mean±std + 验证门裁决）
  arena_log.json       预算/降级/seed/环境/数据 hash（可复现性）
  fuse_quantiles_task.csv  任务侧 18 序列融合分位数 q10/q50/q90（0-2406）
  fuse_point_energy.csv    能源侧 24 序列融合点预测（0-2406）
"""
import hashlib
import json
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from step0_config import CLEAN, FIGURES, OUTPUT, REGIONS, TASK_TYPES, HOURS_TOTAL

SEED = 42
BUDGET_H = 3.0
SEG_TRAIN, SEG_CAL, SEG_TEST = 2352, 2376, 2400
OUT_F = OUTPUT / "forecast"
FIG_S1 = FIGURES / "step1"

MODEL_PARAMS = {
    "lgbm": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15,
             "min_child_samples": 30, "verbosity": -1},
    "xgboost": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4,
                "min_child_weight": 5, "verbosity": 0},
    "rf_quantile": {"n_estimators": 120, "min_samples_leaf": 20,
                    "max_features": 0.5},
    "gbm_quantile": {"n_estimators": 150, "learning_rate": 0.05,
                     "max_depth": 3, "min_samples_leaf": 30},
    "deep": {"hidden": 32, "layers": 2, "epochs": 40, "lr": 1e-3,
             "window": 24, "batch": 128, "early_patience": 5},
    "tabpfn": {"n_estimators": 8, "chunk": 512, "n_train": 1536},
}
QUANTILES = (0.10, 0.50, 0.90)
KAPPA_QUANTILES = (0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99)
ALL_QUANTILES = tuple(sorted(set(QUANTILES) | set(KAPPA_QUANTILES)))


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(p) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def make_series_dict() -> dict[str, dict]:
    """42 序列：任务侧 18（区域×类型 GPU 需求）+ 能源侧 24（新能源/电价/碳/NonAI×6）。"""
    s = {}
    gd = pd.read_csv(CLEAN / "series_gpu_demand.csv")
    for c in gd.columns[1:]:
        s[c] = {"y": gd[c].to_numpy(float), "layer": "task"}
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    for key, col in [("renewable", "AvailableRenewable_MW"),
                     ("price", "ElectricityPrice_CNY_per_MWh"),
                     ("carbon", "CarbonIntensity_tCO2_per_MWh"),
                     ("nonai", "NonAI_IT_Load_MW")]:
        for r in REGIONS:
            sub = rt[rt.Region == r].sort_values("Hour")
            s[f"energy|{r}|{key}"] = {
                "y": sub[col].to_numpy(float).ravel(), "layer": "energy",
                "region": r}
    return s


@lru_cache(maxsize=1)
def _price_lookup():
    """能源侧外生特征查找表：逐时 PricePeriod + 区域段内电价水平（H1 外生已知）。"""
    rt = pd.read_csv(CLEAN / "region_time_clean.csv")
    pp = rt.pivot_table(index="Hour", columns="Region", values="PricePeriod",
                        aggfunc="first")
    seg = rt.groupby(["Region", "PricePeriod"])[
        "ElectricityPrice_CNY_per_MWh"].mean().reset_index()
    level = {r: {row["PricePeriod"]: row["ElectricityPrice_CNY_per_MWh"]
                 for row in seg[seg.Region == r].to_dict("records")}
             for r in REGIONS}
    return pp, level


def build_features(y: np.ndarray, layer: str = "task", region: str | None = None,
                   n: int = HOURS_TOTAL) -> pd.DataFrame:
    """统一特征集 v1（§5.1b）：循环编码 + 滞后 shift≥1 + 滚动统计（防泄露）.

    能源侧追加 PricePeriod 独热 3 列 + 段内电价水平 1 列（外生已知）。
    """
    hour = np.arange(n)
    df = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": hour % 168 // 24,
        "is_weekend": ((hour % 168 // 24) >= 5).astype(int),
    })
    ys = pd.Series(y)
    df["lag1"] = ys.shift(1)
    df["lag2"] = ys.shift(2)
    df["lag24"] = ys.shift(24)
    df["lag168"] = ys.shift(168)
    df["roll24_mean"] = ys.shift(1).rolling(24, min_periods=6).mean()
    df["roll24_std"] = ys.shift(1).rolling(24, min_periods=6).std()
    df["roll168_mean"] = ys.shift(1).rolling(168, min_periods=24).mean()
    df["roll168_std"] = ys.shift(1).rolling(168, min_periods=24).std()
    if layer == "energy" and region is not None:
        pp, level = _price_lookup()
        per = pp[region].reindex(range(n)).fillna("Flat")
        df["pp_valley"] = (per == "Valley").astype(int)
        df["pp_flat"] = (per == "Flat").astype(int)
        df["pp_peak"] = (per == "Peak").astype(int)
        df["seg_price_level"] = per.map(level.get(region, {})).astype(float)
    return df.fillna(0.0)


class StatisticalBaseline:
    """统计基线（及格线 + 兜底，永远在场）.

    能源侧: 日模板（同小时历史均值/分位数）——模板化数据的最强简单基线
    任务侧: 历史分布（全训练段经验分位数 + 均值点预测）——白噪声下渐近最优
    """

    name = "stat_hist"
    family = "统计基线"

    def __init__(self, layer: str):
        self.layer = layer

    def fit(self, y_tr: np.ndarray, y_va=None):
        if self.layer == "energy":
            hod = pd.Series(y_tr).groupby(np.arange(len(y_tr)) % 24)
            self.tpl_mean = hod.mean()
            self.tpl_q = {a: hod.quantile(a) for a in ALL_QUANTILES}
        else:
            self.hist_mean = float(np.mean(y_tr))
            self.hist_q = {a: float(np.quantile(y_tr, a))
                           for a in ALL_QUANTILES}
        return self

    def predict_point(self, X, t0):
        n = len(X) if X is not None else HOURS_TOTAL
        t = np.arange(t0, t0 + n)
        if self.layer == "energy":
            return self.tpl_mean.reindex(t % 24).to_numpy()
        return np.full(n, self.hist_mean)

    def predict_quantile(self, X, t0):
        n = len(X) if X is not None else HOURS_TOTAL
        t = np.arange(t0, t0 + n)
        if self.layer == "energy":
            return {a: self.tpl_q[a].reindex(t % 24).to_numpy()
                    for a in ALL_QUANTILES}
        return {a: np.full(n, self.hist_q[a]) for a in ALL_QUANTILES}


class QuantileEnsemble:
    """分位数模型组：3 个单 α 模型打包（q10/q50/q90），点预测=q50."""

    family = "树类"

    def __init__(self, maker_fn, name: str):
        self.models = [maker_fn(a) for a in QUANTILES]
        self.name = name

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_point(self, X, t0):
        return self.models[1].predict(X)

    def predict_quantile(self, X, t0):
        return {a: m.predict(X) for a, m in zip(QUANTILES, self.models)}


def make_tree_models(layer: str) -> list:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor

    class LGBM_Point(LGBMRegressor):
        name = "lgbm_point"
        family = "树类"

        def predict_point(self, X, t0):
            return self.predict(X)

        def predict_quantile(self, X, t0):
            return None

    class XGB_Point(XGBRegressor):
        name = "xgboost_point"
        family = "树类"

        def predict_point(self, X, t0):
            return self.predict(X)

        def predict_quantile(self, X, t0):
            return None

    class QRF(RandomForestRegressor):
        name = "qrf_quantile"
        family = "树类"

        def fit(self, X, y):
            super().fit(X, y)
            self._y_train = np.asarray(y, dtype=float)
            self._order = np.argsort(self._y_train)
            self._leaf_inds = []
            for t in self.estimators_:
                leaves = t.apply(X)
                d = {}
                for i, l in enumerate(leaves):
                    d.setdefault(l, []).append(i)
                self._leaf_inds.append({l: np.array(v) for l, v in d.items()})
            return self

        def predict_point(self, X, t0):
            return self.predict(X)

        def predict_quantile(self, X, t0):
            """Meinshausen QRF：叶内样本权重条件分位数（非树均值分位数）.

            w_i(x) = Σ_tree 1[x_i ∈ leaf_tree(x)] / n_trees，防均值化陷阱。
            """
            app = np.stack([t.apply(X) for t in self.estimators_])
            order = self._order
            ys = self._y_train[order]
            out = np.zeros((len(QUANTILES), len(X)))
            for j in range(len(X)):
                w = np.zeros(len(self._y_train))
                for ti in range(app.shape[0]):
                    idx = self._leaf_inds[ti].get(app[ti, j])
                    if idx is not None:
                        w[idx] += 1.0
                cw = np.cumsum(w[order])
                total = cw[-1]
                for qi, a in enumerate(QUANTILES):
                    if total <= 0:
                        out[qi, j] = np.median(ys)
                    else:
                        pos = np.searchsorted(cw, a * total)
                        out[qi, j] = ys[min(pos, len(ys) - 1)]
            return {a: out[i] for i, a in enumerate(QUANTILES)}

    models = []
    models.append(LGBM_Point(**MODEL_PARAMS["lgbm"]))
    models.append(XGB_Point(**MODEL_PARAMS["xgboost"]))
    models.append(QRF(**MODEL_PARAMS["rf_quantile"]))
    models.append(QuantileEnsemble(
        lambda a: LGBMRegressor(objective="quantile", alpha=a,
                                **MODEL_PARAMS["lgbm"]), "lgbm_quantile"))
    from sklearn.ensemble import GradientBoostingRegressor
    models.append(QuantileEnsemble(
        lambda a: GradientBoostingRegressor(loss="quantile", alpha=a,
                                            **MODEL_PARAMS["gbm_quantile"]),
        "gbm_quantile"))
    return models


def make_deep_models(layer: str) -> list:
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self, window, hidden):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x.transpose(1, 2))
            return self.fc(out[:, -1, :])

    class _TCNNet(nn.Module):
        def __init__(self, window, hidden):
            super().__init__()
            self.c1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
            self.c2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2,
                                dilation=2)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            return self.fc(h[:, :, -1])

    class DeepModel:
        family = "深度"

        def __init__(self, kind):
            self.kind = kind
            self.name = f"deep_{kind}"
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._hist = None
            self._actual = None
            self._mu = 0.0
            self._sd = 1.0
            self._state = None

        def _to_windows(self, y):
            w = MODEL_PARAMS["deep"]["window"]
            xw = np.stack([y[i - w:i] for i in range(w, len(y))])
            return xw, y[w:]

        def _build_net(self):
            p = MODEL_PARAMS["deep"]
            return (_TCNNet(p["window"], p["hidden"]) if self.kind == "tcn"
                    else _LSTMNet(p["window"], p["hidden"])).to(self.dev)

        def fit(self, y_tr, y_va=None):
            p = MODEL_PARAMS["deep"]
            xw, yw = self._to_windows(y_tr)
            self._mu = float(yw.mean())
            self._sd = float(yw.std()) + 1e-8
            xw = (xw - self._mu) / self._sd
            net = self._build_net()
            opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
            lossf = torch.nn.MSELoss()
            ds = torch.utils.data.TensorDataset(
                torch.tensor(xw, dtype=torch.float32),
                torch.tensor((yw - self._mu) / self._sd, dtype=torch.float32))
            dl = torch.utils.data.DataLoader(ds, batch_size=p["batch"],
                                             shuffle=True, generator=torch.Generator().manual_seed(SEED))
            best, patience = float("inf"), 0
            net.train()
            for ep in range(p["epochs"]):
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = lossf(net(xb.unsqueeze(1).to(self.dev)).squeeze(-1),
                                 yb.to(self.dev))
                    loss.backward()
                    opt.step()
                val_loss = float("inf")
                if y_va is not None and len(y_va) >= p["window"] + 2:
                    xva, yva = self._to_windows(y_va)
                    xva = (xva - self._mu) / self._sd
                    net.eval()
                    with torch.no_grad():
                        pred = net(torch.tensor(xva, dtype=torch.float32)
                                   .unsqueeze(1).to(self.dev)).cpu().numpy().ravel()
                    val_loss = float(np.mean((pred - (yva - self._mu) / self._sd) ** 2))
                    net.train()
                if val_loss < best:
                    best = val_loss
                    patience = 0
                    self._state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= p["early_patience"]:
                        break
            if self._state is None:
                self._state = {k: v.clone() for k, v in net.state_dict().items()}
            return self

        def _predict_roll(self, n):
            p = MODEL_PARAMS["deep"]
            net = self._build_net()
            net.load_state_dict(self._state)
            net.eval()
            w = np.asarray(self._hist[-p["window"]:], dtype=float)
            out = []
            for i in range(n):
                xt = torch.tensor((w - self._mu) / self._sd,
                                  dtype=torch.float32).view(1, 1, -1)
                with torch.no_grad():
                    pred = net(xt.to(self.dev)).item() * self._sd + self._mu
                out.append(pred)
                nxt = (self._actual[i] if (self._actual is not None
                                           and i < len(self._actual))
                       else pred)
                w = np.r_[w[1:], nxt]
            return np.asarray(out)

        def predict_point(self, X, t0):
            return self._predict_roll(len(X))

        def predict_quantile(self, X, t0):
            p = self.predict_point(X, t0)
            z = norm.ppf(QUANTILES)
            return {a: p + z[i] * self._sd for i, a in enumerate(QUANTILES)}

    return [DeepModel(k) for k in ("lstm", "tcn")]


def make_tabpfn_models(layer: str) -> list:
    from tabpfn import TabPFNRegressor

    class TabPFNModel(TabPFNRegressor):
        family = "基础模型"
        name = "tabpfn"

        def fit(self, X, y):
            p = MODEL_PARAMS["tabpfn"]
            if len(y) > p["n_train"]:
                X = X.iloc[-p["n_train"]:]
                y = y[-p["n_train"]:]
            super().fit(X, y)
            self._sd = float(np.std(y)) + 1e-8
            return self

        def _predict_chunked(self, X, output_type, **kw):
            p = MODEL_PARAMS["tabpfn"]
            n = len(X)
            if n <= p["chunk"]:
                return self.predict(X, output_type=output_type, **kw)
            parts = [self.predict(X.iloc[i:i + p["chunk"]],
                                  output_type=output_type, **kw)
                     for i in range(0, n, p["chunk"])]
            if isinstance(parts[0], np.ndarray):
                return np.concatenate(parts)
            return np.concatenate([np.asarray(pr) for pr in parts], axis=0)

        def predict_point(self, X, t0):
            return np.asarray(self._predict_chunked(X, "mean"))

        def predict_quantile(self, X, t0):
            try:
                q = np.asarray(self._predict_chunked(
                    X, "quantiles", quantiles=list(QUANTILES)))
                return {a: q[i] for i, a in enumerate(QUANTILES)}
            except Exception:
                p = self.predict_point(X, t0)
                z = norm.ppf(QUANTILES)
                return {a: p + z[i] * self._sd for i, a in enumerate(QUANTILES)}

    return [TabPFNModel(n_estimators=MODEL_PARAMS["tabpfn"]["n_estimators"])]


def assert_model_interface(pool: list) -> None:
    """模型接口协议检查：防裸类缺方法/形状错位复发。"""
    for m in pool:
        for meth in ("fit", "predict_point", "predict_quantile"):
            assert hasattr(m, meth), f"{getattr(m, 'name', m.__class__.__name__)} 缺 {meth}"
        assert isinstance(m.name, str) and m.name
        assert m.family in ("统计基线", "树类", "深度", "基础模型")


def build_model_pool(layer: str) -> list:
    pool = [StatisticalBaseline(layer)]
    pool += make_tree_models(layer)
    pool += make_deep_models(layer)
    pool += make_tabpfn_models(layer)
    assert_model_interface(pool)
    return pool


def evaluate_fold(layer: str, y_true: np.ndarray, point: np.ndarray,
                  q: dict) -> dict:
    mape = float("nan")
    if layer == "energy":
        mape = float(np.mean(np.abs(y_true - point) / (np.abs(y_true) + 1e-9)) * 100)
    else:
        pos = y_true > 0
        if pos.any():
            mape = float(np.mean(np.abs(y_true[pos] - point[pos]) / y_true[pos]) * 100)
    rmse = float(np.sqrt(np.mean((y_true - point) ** 2)))
    q10, q50, q90 = q[0.10], q[0.50], q[0.90]
    cov = float(np.mean((y_true >= q10) & (y_true <= q90)))
    width = float(np.mean(q90 - q10))
    pinball = float(np.mean(
        np.where(y_true >= q10, 0.1 * (y_true - q10), 0.9 * (q10 - y_true))
        + np.where(y_true >= q50, 0.5 * (y_true - q50), 0.5 * (q50 - y_true))
        + np.where(y_true >= q90, 0.9 * (y_true - q90), 0.1 * (q90 - y_true))))
    return {"mape": mape, "rmse": rmse, "cov": cov, "width": width,
            "pinball": pinball}


def _fresh_model(m):
    """TabPFN 7.1.1 状态泄漏修复：同一实例多次 fit 后 predict 逐次退化（实证
    折1 8s→折5 31s→第6次 fit 后卡死），每次 fit 前重建实例。"""
    if m.family == "基础模型":
        return make_tabpfn_models("energy")[0]
    return m


def run_cv_series(name: str, layer: str, y: np.ndarray, X: pd.DataFrame,
                  pool: list) -> list[dict]:
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for m in pool:
        scores = []
        try:
            for tr, va in tscv.split(np.arange(SEG_TRAIN)):
                t0 = int(tr[-1]) + 1
                m = _fresh_model(m)
                y_tr, y_va = y[tr], y[va]
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                if isinstance(m, StatisticalBaseline):
                    m.fit(y_tr)
                    point = m.predict_point(X_va, t0)
                    q = m.predict_quantile(X_va, t0)
                else:
                    if m.family == "深度":
                        m._hist = y_tr.copy()
                        m._actual = y_va.copy()
                        m.fit(y_tr, y_va)
                    else:
                        m.fit(X_tr, y_tr)
                    point = m.predict_point(X_va, t0)
                    q = m.predict_quantile(X_va, t0)
                    if q is None:
                        sd = float(np.std(y_tr)) + 1e-8
                        z = norm.ppf(QUANTILES)
                        q = {a: point + z[i] * sd for i, a in enumerate(QUANTILES)}
                assert not np.isnan(point).any(), f"{m.name} point 含 NaN"
                for k, v in q.items():
                    assert not np.isnan(v).any(), f"{m.name} q{k} 含 NaN"
                scores.append(evaluate_fold(layer, y_va, point, q))
            sdf = pd.DataFrame(scores)
            rows.append({
                "series": name, "layer": layer, "family": m.family,
                "model": m.name,
                "mape_mean": sdf["mape"].mean(), "mape_std": sdf["mape"].std(),
                "rmse_mean": sdf["rmse"].mean(), "rmse_std": sdf["rmse"].std(),
                "cov_mean": sdf["cov"].mean(), "cov_std": sdf["cov"].std(),
                "width_mean": sdf["width"].mean(),
                "pinball_mean": sdf["pinball"].mean(),
                "n_folds": len(scores),
            })
        except Exception as e:
            rows.append({"series": name, "layer": layer, "family": m.family,
                         "model": m.name, "error": repr(e), "n_folds": 0})
    return rows


def apply_gate(row: dict, base: dict) -> str:
    """验证门 v2：v1 双通道 + 交叉指标守卫（防"宽度换覆盖/精度换覆盖率"）。

    v1 缺陷（T1 修复）: RegionC|RT TabPFN 经 cov 通道通过但 pinball 恶化 −37%
    （宽度换覆盖率）——单通道通过无交叉守卫。
    v2 规则（任务侧）:
      cov 通道: cov 显著增益 + 宽度 ≤1.15× + pinball 不显著恶化（≥ −2pp）
      pinball 通道: pinball 降幅 ≥5% + cov 不崩溃（gain ≥ −基线std）
    """
    if row.get("n_folds", 0) == 0:
        return "拒绝:训练失败"
    if base is None:
        return "通过:统计基线(门)"
    if row["layer"] == "task":
        cov_gain = row["cov_mean"] - base["cov_mean"]
        cov_sig = (cov_gain >= base["cov_std"] and row["cov_mean"] >= 0.90
                   and cov_gain > 0)
        width_ok = row["width_mean"] <= 1.15 * base["width_mean"]
        pin_imp = ((base["pinball_mean"] - row["pinball_mean"])
                   / max(base["pinball_mean"], 1e-9))
        pin_ok = pin_imp >= -0.02          # v2 守卫：pinball 不允许显著恶化
        cov_ok = cov_gain >= -base["cov_std"]  # v2 对称守卫（pinball 通道）
        if (cov_sig and width_ok and pin_ok) or (pin_imp >= 0.05 and cov_ok):
            return (f"通过:cov显著增益={cov_gain:+.3f} "
                    f"pinball降幅={pin_imp*100:.1f}%")
        return (f"拒绝:cov增益={cov_gain:+.3f}(<基线std={base['cov_std']:.3f}) "
                f"或宽度失控({row['width_mean']:.0f}>"
                f"{1.15*base['width_mean']:.0f}) "
                f"或pinball恶化({pin_imp*100:.1f}%<-2%) "
                f"pinball降幅={pin_imp*100:.1f}%")
    imp = ((base["mape_mean"] - row["mape_mean"])
           / max(base["mape_mean"], 1e-9))
    if imp >= 0.05:
        return f"通过:MAPE降幅={imp*100:.1f}%"
    return f"拒绝:MAPE降幅={imp*100:.1f}%(<5%)"


def regate_arena(path: Path | str = OUTPUT / "forecast" / "arena_table.csv"
                 ) -> pd.DataFrame:
    """基于现有竞技榜指标重算 gate（v2），免重训（论文数字追溯）。

    仅更新 gate 列 + 追加 gate_v2 说明；统计基线行标记不变。
    """
    t = pd.read_csv(path)
    base_map = {}
    for _, r in t[t.family == "统计基线"].iterrows():
        base_map[r["series"]] = r.to_dict()
    gates = []
    for _, r in t.iterrows():
        if r["family"] == "统计基线":
            gates.append("通过:统计基线(门)")
        else:
            gates.append(apply_gate(r.to_dict(), base_map.get(r["series"])))
    t["gate"] = gates
    t.to_csv(path, index=False)
    npass = sum(1 for g in gates
                if g.startswith("通过") and "统计基线" not in g)
    print(f"regate v2: 非基线通过 {npass}/{len(gates) - len(base_map)}")
    return t


def _build_meta(oofdf: pd.DataFrame, wdf: pd.DataFrame, yv: np.ndarray,
                idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """不确定性元特征：q50 + 区间宽度 + 滚动 24h 误差（shift≥1）+ 时间上下文.

    防泄露：滚动误差特征只用预测点之前的 OOF 误差。
    """
    names = list(oofdf.columns)
    err = np.abs(oofdf[names].to_numpy() - yv[:, None])
    Z, cols = [], []
    for j, n in enumerate(names):
        Z.append(oofdf[n].to_numpy())
        cols.append(f"{n}_q50")
        w = wdf[n].to_numpy() if n in wdf.columns else np.full(len(idx), np.nan)
        if not np.isnan(w).all():
            Z.append(w)
            cols.append(f"{n}_width")
        roll = pd.Series(err[:, j]).shift(1).rolling(24, min_periods=6).mean()
        Z.append(roll.to_numpy())
        cols.append(f"{n}_rollerr")
    hod = idx % 24
    Z.append(np.sin(2 * np.pi * hod / 24))
    Z.append(np.cos(2 * np.pi * hod / 24))
    cols += ["hour_sin", "hour_cos"]
    return np.nan_to_num(np.column_stack(Z), nan=0.0), cols


def fuse_energy(y: np.ndarray, X: pd.DataFrame, pool: list,
                passed: list) -> tuple[np.ndarray, dict]:
    """能源侧融合 v2：不确定性元特征 Ridge stacking → 增益>0.1% 才用，否则回退最优单.

    元特征 = 各模型 q50 + 区间宽度 + 滚动 24h OOF 误差 + 时间上下文（2023C 倒数第二层思想）。
    入选模型用 0-2351 全训练段 fit（深度早停用训练段内部末 100 点，不触校准段）。
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    if not passed:
        base = pool[0]
        base.fit(y[:SEG_TRAIN])
        full_base = base.predict_point(X, 0)
        return full_base, {"fuse": "统计基线(无入选)"}
    tscv = TimeSeriesSplit(n_splits=5)
    oof = {m.name: np.full(SEG_TRAIN, np.nan) for m in passed}
    oof_w = {m.name: np.full(SEG_TRAIN, np.nan) for m in passed}
    for tr, va in tscv.split(np.arange(SEG_TRAIN)):
        t0 = int(tr[-1]) + 1
        for m0 in passed:
            m = _fresh_model(m0)
            if m.family == "深度":
                m._hist = y[tr].copy()
                m._actual = y[va].copy()
                m.fit(y[tr], y[va])
            else:
                m.fit(X.iloc[tr], y[tr])
            oof[m.name][va] = m.predict_point(X.iloc[va], t0)
            q = m.predict_quantile(X.iloc[va], t0)
            if q is not None and 0.10 in q and 0.90 in q:
                oof_w[m.name][va] = q[0.90] - q[0.10]
    oofdf = pd.DataFrame(oof).dropna()
    wdf = pd.DataFrame(oof_w).reindex(oofdf.index)
    yv = y[oofdf.index]
    names = list(oofdf.columns)

    Z, cols = _build_meta(oofdf, wdf, yv, oofdf.index.to_numpy())
    meta = Ridge(alpha=1.0).fit(Z, yv)
    meta_pred = meta.predict(Z)
    best_single = oofdf.mean(axis=1).to_numpy()
    mape_meta = float(np.mean(np.abs(yv - meta_pred) / (np.abs(yv) + 1e-9)) * 100)
    mape_best = float(np.mean(np.abs(yv - best_single) / (np.abs(yv) + 1e-9)) * 100)

    full, full_w = {}, {}
    for m0 in passed:
        m = _fresh_model(m0)
        if m.family == "深度":
            m._hist = y[:MODEL_PARAMS["deep"]["window"]].copy()
            m._actual = y[MODEL_PARAMS["deep"]["window"]:].copy()
            m.fit(y[:SEG_TRAIN], y[SEG_TRAIN - 100:SEG_TRAIN])
        else:
            m.fit(X.iloc[:SEG_TRAIN], y[:SEG_TRAIN])
        full[m.name] = m.predict_point(X, 0)
        q = m.predict_quantile(X, 0)
        if q is not None and 0.10 in q and 0.90 in q:
            full_w[m.name] = q[0.90] - q[0.10]
    fdf = pd.DataFrame(full)
    fwdf = pd.DataFrame(full_w)
    idx_full = np.arange(len(y))
    Zf, _ = _build_meta(fdf, fwdf, y, idx_full)
    full_pred = meta.predict(Zf)

    if (mape_best - mape_meta) / mape_best > 0.001:
        return full_pred, {"fuse": "stacking", "meta_mape": mape_meta,
                           "best_single_mape": mape_best, "names": names,
                           "meta_cols": cols}
    full_best = np.mean([full[n] for n in names], axis=0)
    return full_best, {"fuse": "最优单模型(等权OOF均值)", "meta_mape": mape_meta,
                       "best_single_mape": mape_best, "names": names,
                       "meta_cols": cols}


def fuse_task(y: np.ndarray, pool: list, passed: list) -> tuple[dict, dict]:
    """任务侧融合：入选模型分位数等权平均；空则统计基线分位数（预期主路径）。"""
    base = pool[0]
    base.fit(y[:SEG_TRAIN])
    if not passed:
        return (base.predict_quantile(None, 0),
                {"fuse": "统计基线(无入选)"})
    qsum = {a: np.zeros(HOURS_TOTAL) for a in QUANTILES}
    X_task = build_features(y, "task")
    for m0 in passed:
        m = _fresh_model(m0)
        if m.family == "深度":
            m._hist = y[:MODEL_PARAMS["deep"]["window"]].copy()
            m._actual = y[MODEL_PARAMS["deep"]["window"]:].copy()
            m.fit(y[:SEG_TRAIN], y[SEG_TRAIN - 100:SEG_TRAIN])
            q = m.predict_quantile(None, 0)
        else:
            m.fit(X_task.iloc[:SEG_TRAIN], y[:SEG_TRAIN])
            q = m.predict_quantile(X_task, 0)
        for a in QUANTILES:
            qsum[a] += q[a]
    qmean = {a: qsum[a] / len(passed) for a in QUANTILES}
    return qmean, {"fuse": "入选模型等权分位数", "names": [m.name for m in passed]}


def main() -> None:
    seed_everything()
    OUT_F.mkdir(parents=True, exist_ok=True)
    FIG_S1.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    series = make_series_dict()
    log = {"seed": SEED, "budget_h": BUDGET_H,
           "data_hash": {f.name: sha256_file(CLEAN / f.name)
                         for f in sorted(CLEAN.glob("*.csv"))[:6]},
           "torch_cuda": torch.cuda.is_available(),
           "python": sys.version.split()[0],
           "start": time.strftime("%Y-%m-%d %H:%M:%S")}

    all_rows = []
    resume = (OUT_F / "arena_table.csv").exists()
    if resume:
        table0 = pd.read_csv(OUT_F / "arena_table.csv")
        log["resume"] = True
        print("[续跑] 检测到 arena_table.csv，跳过 run_cv 阶段")
    for i, (name, info) in enumerate(series.items()):
        layer = info["layer"]
        y = info["y"]
        X = build_features(y, layer, info.get("region"))
        if resume:
            continue
        if i == 0:
            t0 = time.time()
            run_cv_series(name, layer, y, X, build_model_pool(layer))
            dt = time.time() - t0
            est = dt * len(series) / 3600
            log["calibration"] = {"first_series_s": round(dt, 1),
                                  "projected_total_h": round(est, 2)}
            print(f"[标定] 单序列 {dt:.1f}s → 预计全量 {est:.2f}h"
                  f"（预算 {BUDGET_H}h）")
            if est > BUDGET_H:
                log["budget_warning"] = True
                print("[警告] 预计超预算，仅记录警告并继续（全量评估优先）")
        pool = build_model_pool(layer)
        all_rows += run_cv_series(name, layer, y, X, pool)
        if (i + 1) % 10 == 0:
            print(f"进度 {i+1}/{len(series)} 累计 {time.time()-t_start:.0f}s")

    if resume:
        table = table0
        log["resume"] = True
    else:
        table = pd.DataFrame(all_rows)
    base_rows = table[table.family == "统计基线"].copy()
    gates = []
    for _, row in table.iterrows():
        if row["family"] == "统计基线":
            gates.append("通过:统计基线(门)")
        else:
            bm = base_rows[base_rows.series == row["series"]]
            gates.append(apply_gate(row, bm.iloc[0].to_dict() if len(bm) else None))
    table["gate"] = gates
    table.to_csv(OUT_F / "arena_table.csv", index=False, encoding="utf-8-sig")

    passed_map = {}
    for _, row in table[table.family != "统计基线"].iterrows():
        if row["gate"].startswith("通过") and row["n_folds"] > 0:
            passed_map.setdefault(row["series"], []).append(row["model"])
    log["passed_counts"] = {
        "task": sum(1 for k, v in passed_map.items() if k in series and
                    series[k]["layer"] == "task"),
        "energy": sum(1 for k, v in passed_map.items() if k in series and
                      series[k]["layer"] == "energy"),
        "passed_models": {k: v for k, v in passed_map.items()}}

    q10t, q50t, q90t, act_t = {}, {}, {}, {}
    qkappa = {a: {} for a in ALL_QUANTILES}
    pred_e, act_e = {}, {}
    for name, info in series.items():
        layer = info["layer"]
        y = info["y"]
        X = build_features(y, layer, info.get("region"))
        pool = build_model_pool(layer)
        passed = [m for m in pool if m.name in passed_map.get(name, [])]
        if layer == "energy":
            fp, fmeta = fuse_energy(y, X, pool, passed)
            pred_e[name] = fp
            act_e[name] = y
            log.setdefault("fuse_energy", {})[name] = fmeta
        else:
            q, fmeta = fuse_task(y, pool, passed)
            q10t[name], q50t[name], q90t[name], act_t[name] = (
                q[0.10], q[0.50], q[0.90], y)
            for a in ALL_QUANTILES:
                qkappa[a][name] = q.get(
                    a, q[0.50] + norm.ppf(a) * np.std(y[:SEG_TRAIN]))
            log.setdefault("fuse_task", {})[name] = fmeta

    hour_idx = pd.RangeIndex(HOURS_TOTAL, name="Hour")
    fuse_t = pd.DataFrame({"Hour": hour_idx})
    for name in act_t:
        for a in ALL_QUANTILES:
            fuse_t[f"{name}_q{int(a*100)}"] = qkappa[a][name]
        fuse_t[f"{name}_actual"] = act_t[name]
    fuse_t.to_csv(OUT_F / "fuse_quantiles_task.csv", index=False,
                  encoding="utf-8-sig")
    fuse_e = pd.DataFrame({"Hour": hour_idx})
    for name in act_e:
        fuse_e[f"{name}_pred"] = pred_e[name]
        fuse_e[f"{name}_actual"] = act_e[name]
    fuse_e.to_csv(OUT_F / "fuse_point_energy.csv", index=False,
                  encoding="utf-8-sig")

    log["elapsed_s"] = round(time.time() - t_start, 1)
    log["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_F / "arena_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"[完成] 全量 {len(series)} 序列，耗时 {log['elapsed_s']}s")
    g = table[table.family != "统计基线"]
    for lay in ("task", "energy"):
        sub = g[g.layer == lay]
        if len(sub):
            ok = sub[sub.gate.str.startswith("通过")]
            print(f"通过率 {lay}: {len(ok)}/{len(sub)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "regate":
        regate_arena()
    else:
        main()
