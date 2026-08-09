"""test_identity — Phase 1 恒等式集守卫（7 关协议产物断言）.

数据事实层零错误纪律：I1-I7/I10/I13 六区【实证】状态为硬约束，
任何回归（如 loader 口径变更）必须显式重跑 step0.6_identity_proof.py。
L1 分层实验：S1 冻结段必须优于 S0（恒等式分层的投产价值守卫）。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ID = ROOT / "output" / "clean" / "identity_proof.json"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
PROVEN = ["I1_it_equals_nonai_plus_ai", "I2_power_balance",
          "I3_carbon", "I4_netgrid", "I5_utilization",
          "I6_soc_recurrence", "I7_charge_decomp",
          "I10_consume_template", "I13_total_eq_it_x_pue"]


@pytest.fixture(scope="module")
def proof():
    if not ID.exists():
        pytest.skip("identity_proof.json 未生成（先跑 step0.6_identity_proof.py）")
    return json.loads(ID.read_text(encoding="utf-8"))


@pytest.mark.parametrize("ident", PROVEN)
@pytest.mark.parametrize("r", REGIONS)
def test_identity_proven(proof, ident, r):
    assert proof["identities"][r][ident]["grade"] == "实证", \
        f"{r} {ident} 未达实证：{proof['identities'][r][ident]['grade']}"


def test_identity_resid_machine_precision(proof):
    """I1 残差须达浮点精度极限（<1e-6 相对），证明表内恒等式非近似。"""
    for r in REGIONS:
        c = proof["identities"][r]["I1_it_equals_nonai_plus_ai"]["checks"]
        assert c["c1_resid"]["rel_mean"] < 1e-6
        assert c["c2_segments"]["all_ok"]


def test_i11_caliber_finding_recorded(proof):
    """I11 构造性口径差异（ceil vs 分数 Overlap）须有记录（生成口径发现）。"""
    for r in REGIONS:
        v = proof["identities"][r]["I11_constructive_ai"]
        assert v["grade"] == "存疑"
        assert "M4" in proof["identities"][r]["I1_it_equals_nonai_plus_ai"] \
            ["checks"]["c4_constructive"]["note"]


def test_nonai_layered_frozen_gain():
    """L1：S1（IT_Load 模板 - AI 实际）冻结段必须全面优于 S0（直接模板）。"""
    p = ROOT / "output" / "robust" / "nonai_layered.json"
    if not p.exists():
        pytest.skip("nonai_layered.json 未生成（先跑 step0.7_nonai_layered.py）")
    rep = json.loads(p.read_text(encoding="utf-8"))
    for r in REGIONS:
        fz = rep[r]["frozen_mape"]
        assert fz["s1_layered_actual"] < fz["s0_direct"], \
            f"{r} S1 未优于 S0：{fz['s1_layered_actual']} vs {fz['s0_direct']}"


def test_nonai_layered_identity_mean():
    """L1 理论恒等：S2（IT_Load 模板 - AI 均值模板）≡ S0（模板线性性）。"""
    p = ROOT / "output" / "robust" / "nonai_layered.json"
    if not p.exists():
        pytest.skip("nonai_layered.json 未生成")
    rep = json.loads(p.read_text(encoding="utf-8"))
    for r in REGIONS:
        fz = rep[r]["frozen_mape"]
        assert abs(fz["s2_layered_mean"] - fz["s0_direct"]) < 0.02, \
            f"{r} S2 与 S0 应数学恒等：{fz['s2_layered_mean']} vs {fz['s0_direct']}"
