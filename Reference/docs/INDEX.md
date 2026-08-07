# INDEX — 论文公式 ↔ 代码 ↔ 文档全映射

> 随项目推进逐条填充。代码平铺 Code 根目录。

## 阶段映射
| 阶段 | 代码文件 | 文档 | 状态 |
|------|----------|------|------|
| 初始化 | — | PLAN.md / PLAN_details.md / docs/ / Reference/ | ✅ |
| M1 | step0_config.py | docs/sums/sum_1_数据侦察与证据链.md | ✅ |
| M1 | step0_loader.py → output/clean/* | docs/migration_prompt.md Step4 口径 | ✅ |
| M1 | step0_eda.py → figures/eda/figD1-D9 | docs/sums/sum_1（D1-D9 实证表） | ✅ |
| M2 | step1.0_baseline_schedule.py / step1.1_forecast_arena.py / step1.2_robust_schedule.py（待建） | （待写） | ⏳ |

## 公式映射（论文编号 ↔ 代码位置）
| 论文公式 | 代码文件:行号 | 说明 |
|----------|--------------|------|
| 白名单可达集合（时延≤MaxLatency） | step0_loader.py:build_whitelist | 预计算三层可达集 |
| 任务小时占用展开（GPU-hour 重叠折算） | step0_loader.py:expand_occupancy | 195,047 行占用表 |
| 18 序列聚合（区域×类型 GPU 需求） | step0_loader.py:build_series | series_gpu_demand.csv |
| 质量检查（NaN/矛盾任务/时段完整性） | step0_loader.py:check_quality | quality_report.json |
| （后续公式 1.1-4.4 随 step1-4 落地时填充） | | |

## 产物清单（output/clean/）
workload_clean.csv / region_time_clean.csv / whitelist.csv / occupancy_local.csv
series_gpu_demand.csv / series_arrivals.csv / occupancy_parallel.csv
storage_params.csv / quality_report.json / figures/eda/eda_summary.json

## Phase 文档
| Phase | Reference/docs/PhaseN | 内容 |
|-------|------------------------|------|
| （待建） | | |
