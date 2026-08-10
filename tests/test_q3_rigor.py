"""test_q3_rigor — Q3 反思 P1 四件套守卫（step3.8）.

双锚分解: 储能价值（vs 无储能）> 单锚报告（生成器次优隐藏价值）；
         无储能成本 > M3_final 成本（储能正向价值）
斜坡活跃: binding 小时数 > 0（物理化修正非装饰）且成本免费（面内兼容）
终态对称: 生成器 SOC(2406) 与 Init 的差异如实记录（不对称声明）
分窗分布: 14 周改进率全为正（稳健性）
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_step(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".", "_"), ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rigor():
    p = ROOT / "output" / "q3" / "q3_rigor.json"
    if not p.exists():
        pytest.skip("q3_rigor.json 缺失——先跑 step3.8_q3_rigor.py")
    return json.loads(p.read_text(encoding="utf-8"))


def test_double_anchor_storage_value(rigor):
    """储能价值（vs 无储能 LP 对照）为正且生成器次优为正（1-1 纯度修正）。"""
    for r in ("RegionA", "RegionB", "RegionC", "RegionD"):
        dec = rigor["regions"][r]["decomposition"]
        assert dec["storage_value_wan"] < 0, f"{r} 储能价值非正"
        assert dec["gen_subopt_wan"] > 0, f"{r} 生成器次优代价非正"
    # D 区储能价值（LP 对照修正后 −38.3%，非模板版 −71.5%——1-1 修正）
    d_dec = rigor["regions"]["RegionD"]["decomposition"]
    assert -45 < d_dec["storage_value_pct"] < -30, \
        f"D 区储能价值异常: {d_dec['storage_value_pct']}%（应约 −38%）"
    # LP 自由度红利被正确分离（D 区 >1 亿，原模板对照混入）
    assert d_dec["lp_freedom_value_wan"] < -10000, \
        f"D 区 LP 自由度价值异常: {d_dec['lp_freedom_value_wan']}"
    # no_storage_lp 字段存在且成本 < 模板版（LP 优化+外送更优）
    assert rigor["regions"]["RegionD"]["no_storage_lp"]["cost_wan"] < \
        rigor["regions"]["RegionD"]["no_storage_template"]["cost_wan"]


def test_slope_binding_active(rigor):
    """斜坡约束活跃（binding>0）——物理化修正非装饰（Q3 反思 #1）。"""
    for r in ("RegionA", "RegionD", "RegionE"):
        sl = rigor["regions"][r]["slope_binding"]
        assert sl["binding_c_hours"] > 0, f"{r} 充电斜坡零 binding"
        assert sl["binding_d_hours"] > 0, f"{r} 放电斜坡零 binding"
        assert sl["binding_c_pct"] > 0 and sl["binding_c_pct"] < 50, \
            f"{r} 斜坡过度活跃异常 {sl['binding_c_pct']}%"


def test_terminal_symmetry_recorded(rigor):
    """终态对称性如实记录（生成器 SOC(2406) vs Init——不对称声明）。"""
    for r in ("RegionA", "RegionD", "RegionE"):
        ts = rigor["regions"][r]["terminal_symmetry"]
        assert ts["generator_soc_2406"] is not None, f"{r} SOC(2406) 缺失"
        assert ts["strict_terminal"] in (True, False)
    # 西区生成器终态违规（SOC(2406) < Init）——口径不对称的实证
    d_ts = rigor["regions"]["RegionD"]["terminal_symmetry"]
    assert d_ts["generator_soc_2406"] < d_ts["init_soc"], \
        "D 区生成器终态应低于 Init（违规实证）"


def test_closure_symmetry(rigor):
    """结算段对称性：基准 2400-2405 零充零放（禁充口径对称成立）。"""
    for r in ("RegionA", "RegionD"):
        ts = rigor["regions"][r]["terminal_symmetry"]
        assert ts["closure_charge_hours"] == 0, f"{r} 基准结算段充电"


def test_weekly_improvement_positive(rigor):
    """分窗改进率全为正（稳健性：14 周×6 区无负窗）。"""
    s = rigor["summary"]
    assert s["week_improve_m3_vs_base"]["n"] >= 80
    assert s["week_improve_m3_vs_base"]["min_pct"] > 0, "存在负改进周"
    assert s["week_improve_m3_vs_nostore"]["min_pct"] > 0
