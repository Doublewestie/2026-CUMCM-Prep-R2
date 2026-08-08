"""C4: 自研算法正确性测试（NSGA-II 非支配/ALNS 可行性/前沿互不支配）."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("s24", ROOT / "step2.4_method_arena.py")
s24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s24)


def test_nondom_transitivity():
    """非支配排序传递性：A≻B 且 B≻C → A 的层 ≤ B 的层 ≤ C 的层。"""
    rng = np.random.RandomState(0)
    F = rng.rand(30, 4)
    fronts = s24.fast_nondom(F)
    rank = {i: k for k, fr in enumerate(fronts) for i in fr}
    for i in range(len(F)):
        for j in range(len(F)):
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                assert rank[i] <= rank[j], "支配关系与层序矛盾"


def test_nondom_small_case():
    """已知小案例：3 点 → [[最优], [次优]] 层结构。"""
    F = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [0.5, 0.5]])
    fronts = s24.fast_nondom(F)
    assert fronts[0] == [3]
    assert sorted(fronts[1]) in ([0, 1], [1, 0])


def test_crowding_boundary_inf():
    F = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 3.0]])
    d = s24.crowding(F, [0, 1, 2, 3])
    assert d[0] == float("inf") and d[3] == float("inf")


def test_alns_feasible_and_no_worse():
    """ALNS：输出调度 viol=0 且成本不差于构造解（无退化）。"""
    import pandas as pd
    c = s24._load_ctx()
    wt, rt, params = c["wt"], c["rt"], c["params"]
    s10, s20 = c["s10"], c["s20"]
    construct = s20.schedule_constructive(wt, rt, s10, params, (0, 0, 0, 0, 0))
    base = s20.evaluate_4obj(wt, rt, construct, s10, params, c["consume"])
    alnsF = s24.run_alns(0, wt, rt, s10, s20, params, n_iter=5)
    assert alnsF[0, 0] <= 1e11          # 可行
    assert alnsF[0, 0] <= base["cost_wan"] * 1.001   # 不差于构造解


def test_nsga2_front_mutually_nondominated():
    """NSGA-II 前沿解互不支配（正确性核心）。"""
    import pandas as pd
    fr = pd.read_csv(ROOT / "output" / "q2" / "nsga2_front.csv")
    obj = fr[["cost_wan", "carbon_t", "latency_ms", "nu_pct"]].to_numpy()
    obj[:, 3] = 1 - obj[:, 3] / 100
    n = len(obj)
    for i in range(min(n, 40)):
        for j in range(min(n, 40)):
            if i == j:
                continue
            assert not (np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j])), \
                f"前沿解 {i} 支配 {j}"
