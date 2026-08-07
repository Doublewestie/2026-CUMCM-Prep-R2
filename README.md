# 2026-CUMCM-Prep-R2

2026 华数杯 C 题（面向算电协同的多目标调度优化）备赛代码仓库。

## 当前进度
- **M1 ✅ 数据地基 + EDA 证据链**（step0 三件套跑通，产物见 output/clean/ 与 figures/eda/）
- M2 进行中：step1 预测竞技榜 + 基础调度基线

## 核心结果速览（关键数字，供论文引用）
| 指标 | 值 |
|---|---|
| 任务到达白噪声 | 18 序列 mean lag1=0.011 / lag24=-0.006 |
| 新能源模板化 | 六区曲线 100% 重合，严格 24h 日周期（500-1100MW） |
| 本地执行超容 | E 30h / F 67h（最大超 62%） |
| 基线弃电 | 775.5 万 MWh，新能源利用率 32.9% |
| 东区弃电率 | 88-91%（A/B/C，无外送 + 相位错位） |
| 相位错位 | NonAI 负荷凌晨峰/傍晚谷差 67.4% |
| 时延白名单 | RT≤20 / BI≤80 / AT≤150ms 三层 |
| 数据质量 | 零 NaN、零矛盾任务（50,000 任务） |

## 文档指针
- 作战概要 → `PLAN.md`
- 完整建模方案（逐句解析/公式推导/实验/消融/附录）→ `PLAN_details.md`
- 工作日志/总结/设计文档 → `docs/`（logs/sums/specs）
- 硬约束速查 → `Reference/docs/CONSTITUTION.md`
- 公式↔代码映射 → `Reference/docs/INDEX.md`
- 新会话入口 → `docs/migration_prompt.md`

## 环境
mathorcup conda 环境（Python 3.13.12；torch 2.11+cu130 GPU；tabpfn 7.1.1；pulp）
