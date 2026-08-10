"""step5.2_paper_figures — 论文机理图（M6 素材，figures/paper/）.

图清单:
  fig_m1_t1_theorem.png    T1 购电下界定理因果链（R1 模板→弃电下界→G 下界→限额 infeasible）
  fig_m2_four_layer.png    四层研究框架（公理层→分层→边界→协同→风险）
  fig_m3_t2_evidence.png   T2 储能结构锁定三证据链（判别 A/B + 容量钳制）
  fig_m4_model_spectrum.png 储能模型谱价值区间（M0 假象/M3_final 保守/M0x 上界）
  fig_eta_profile_v2.png   η 连续谱 + 分层策略映射（强化版）
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step0_config import FIGURES

FIG_PAPER = FIGURES / "paper"
FIG_PAPER.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def box(ax, x, y, w, h, text, fc="#eaf2f8", ec="#2c6fbb", fs=9, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center",
                fontsize=7.5, color=color)


def fig_t1():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.2, 4.8, 2.4, 0.9, "R1 消纳模板\nU = c_r(h)·D（命中率 100%）", "#fef9e7", "#b7950b", 8.5, True)
    box(ax, 3.4, 4.8, 2.4, 0.9, "弃电下界\nQ ≥ W−cap_h−Pc−S", "#fdeaea", "#c0392b", 8.5, True)
    box(ax, 6.6, 4.8, 2.6, 0.9, "功率平衡 + Pd≥0\nG ≥ D − cap_h", "#eafaf1", "#1e8449", 8.5, True)
    box(ax, 0.2, 2.6, 3.0, 1.2, "碳排下界\nΣcarb·G ≥ Σcarb·(D−cap_h)", "#eaf2f8", "#2c6fbb", 8.5)
    box(ax, 4.0, 2.6, 3.0, 1.2, "峰值下界\nmax(G−S) ≥ max(D−cap_h−S)", "#eaf2f8", "#2c6fbb", 8.5)
    box(ax, 7.8, 2.6, 2.9, 1.2, "成本最优解已压 G 至下界\n（price>0 最小化）", "#f4ecf7", "#7d3c98", 8.5)
    box(ax, 0.8, 0.3, 4.2, 1.1, "区域级：碳/峰值限额 2% 即 infeasible\n（六区一致，q4_shadow）", "#fdeaea", "#c0392b", 8.5, True)
    box(ax, 5.8, 0.3, 4.6, 1.1, "全网级：削峰约束 5% 即 infeasible\n（六区联合 LP，q4_system_peak）", "#fdeaea", "#c0392b", 8.5, True)
    arrow(ax, 2.6, 4.8, 3.4, 4.8)
    arrow(ax, 5.8, 4.8, 6.6, 4.8)
    arrow(ax, 7.9, 4.8, 7.9, 3.8)
    arrow(ax, 1.7, 4.8, 1.7, 3.8)
    arrow(ax, 5.5, 4.8, 5.5, 3.8)
    arrow(ax, 9.25, 2.6, 9.25, 1.4, "推论")
    arrow(ax, 2.9, 2.6, 2.9, 1.4, "推论")
    ax.set_title("T1 购电下界定理：消纳模板锁定储能层内减碳/降峰空间=0",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_PAPER / "fig_m1_t1_theorem.png", bbox_inches="tight")
    plt.close(fig)


def fig_four_layer():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    layers = [
        (0.2, "数据公理层\n恒等式 I1-I13（rel 1e-9）·模板 100%·W 公式 0.0043", "#fef9e7", "#b7950b"),
        (2.4, "可预测性分层\nη 连续谱 0.0~1.0 → E1：预测无价值", "#fdeaea", "#c0392b"),
        (4.6, "结构边界（T1）\nG ≥ D−cap_h → 储能层内减碳/降峰=0", "#eafaf1", "#1e8449"),
        (6.8, "结构协同（T2）\n储能模板锁定任务层可优化面 0.14-0.55%", "#eaf2f8", "#2c6fbb"),
        (9.0, "风险对冲\n波动 64 场景·尾部瓶颈·MPC 结构性保守", "#f4ecf7", "#7d3c98"),
    ]
    for x, txt, fc, ec in layers:
        box(ax, x, 1.8, 2.0, 1.6, txt, fc, ec, 8)
    for i in range(4):
        arrow(ax, 2.2 + i * 2.2, 2.6, 2.4 + i * 2.2, 2.6)
    ax.text(5.5, 4.2, "研究路线：数据是什么 → 该不该预测 → 能优化什么 → 协同空间 → 风险多高",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(5.5, 0.6, "每一步由数据裁决，每个结论经'先怀疑自己→验证→修正'链条",
            ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(FIG_PAPER / "fig_m2_four_layer.png", bbox_inches="tight")
    plt.close(fig)


def fig_t2():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(5.5, 5.5, "T2 储能结构锁定：三证据链（判别 A/B + 容量钳制）",
            ha="center", fontsize=12, fontweight="bold")
    box(ax, 0.2, 3.6, 3.2, 1.3, "判别 A：极端策略跨评估器\nQ2 极端策略在 Q4 框架重评\n成本跨度 0.55%（Q2 的 1/8）", "#eaf2f8", "#2c6fbb", 8.5)
    box(ax, 3.9, 3.6, 3.4, 1.3, "判别 B：oracle 贪心迁移\n解析增益 4,574 万 → 真实评估\n单调反噬 +3.5%（方向性错误）", "#fdeaea", "#c0392b", 8.5)
    box(ax, 7.8, 3.6, 3.0, 1.3, "容量钳制：极端重排\n全迁 D 超容 2.6 倍\n错峰 1,553 小时违例", "#fef9e7", "#b7950b", 8.5)
    box(ax, 0.2, 1.3, 5.0, 1.2, "正式前沿跨度 0.14%（40×30×3 收敛）\nvs 任务层原始价值 4.57%（Q2）", "#eafaf1", "#1e8449", 8.5, True)
    box(ax, 5.8, 1.3, 5.0, 1.2, "反证排除：宽时段下层（充电 h0-8+20-23）\n任务层跨度仍 0.27%——时段宽度非根因", "#f4ecf7", "#7d3c98", 8.5)
    arrow(ax, 1.8, 3.6, 1.8, 2.5)
    arrow(ax, 5.6, 3.6, 5.6, 2.5)
    arrow(ax, 9.3, 3.6, 9.3, 2.5)
    box(ax, 2.0, 0.15, 7.0, 0.8, "结论【实证】：任务层电力侧可优化面被储能时段模板结构性压缩至 0.14-0.55%\n联合帕累托主轴 = 时延/QOS；与 Q3 权衡面坍缩同构传承", "#e8f8f5", "#148f77", 9, True)
    fig.tight_layout()
    fig.savefig(FIG_PAPER / "fig_m3_t2_evidence.png", bbox_inches="tight")
    plt.close(fig)


def fig_spectrum():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    # 模型谱价值区间（D 区成本，万元）
    models = ["M0\n无互斥 LP\n(传送带假象)", "M3_final\n时段状态机\n(保守口径)", "M0x\n互斥 MILP\n(物理上界)"]
    costs = [-9637, 7561, 222]
    colors = ["#e74c3c", "#3498db", "#27ae60"]
    bars = ax.bar(models, costs, color=colors, alpha=0.85, width=0.5)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 300,
                f"{c:,.0f} 万", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(8593, color="#95a5a6", linestyle="--", linewidth=1.2)
    ax.text(2.42, 8593 + 400, "生成器基准 8,593 万", fontsize=8.5, color="#7f8c8d")
    ax.set_ylabel("D 区总成本（万元）")
    ax.set_title("储能模型谱价值区间：M0 假象 / M3_final 保守 / M0x 物理上界\n"
                 "（时段红利 4,000-7,300 万/区 = 时段状态机的保守代价）",
                 fontsize=11, fontweight="bold")
    ax.text(0.4, 3000, "M0 含传送带假象\n不得引用（方法学警示）", fontsize=8, color="#c0392b")
    ax.text(1.0, 10000, "价值区间 [M3_final, M0x]", fontsize=9.5, color="#1e8449", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_PAPER / "fig_m4_model_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def fig_eta():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    # η 连续谱（任务/能源/价格分层）
    cats = ["任务×18\n（RT/BI/AT×6 区）", "carbon\n周模板", "NonAI\n负荷", "price\n价格", "renewable\n新能源"]
    eta = [0.0, 0.877, 0.889, 0.99, 1.0]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    ax.bar(cats, eta, color=colors, alpha=0.85, width=0.55)
    for i, e in enumerate(eta):
        ax.text(i, e + 0.02, f"{e:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="#95a5a6", linestyle="--", linewidth=1)
    ax.text(4.45, 0.52, "可预测性阈值", fontsize=8, color="#7f8f8d")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("可预测性 η = 1 − Var(残差)/Var(Y)")
    ax.set_title("可预测性分层：白噪声（分布刻画）→ 模板（机理开采）→ 随机（鲁棒对冲）",
                 fontsize=11.5, fontweight="bold")
    ax.text(0.02, 0.95, "任务白噪声\n→ E1：预测无价值\n→ 分布刻画+规则", fontsize=8.5,
            color="#c0392b", va="top")
    ax.text(2.2, 0.95, "能源模板\n→ 机理启发式直接开采\n→ 消纳模板/W 公式", fontsize=8.5,
            color="#1e8449", va="top")
    fig.tight_layout()
    fig.savefig(FIG_PAPER / "fig_eta_profile_v2.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_t1()
    fig_four_layer()
    fig_t2()
    fig_spectrum()
    fig_eta()
    print("paper figures done:", sorted(p.name for p in FIG_PAPER.glob("*.png")))


if __name__ == "__main__":
    main()
