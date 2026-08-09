# INDEX — 论文公式 ↔ 代码 ↔ 文档全映射

> 随项目推进逐条填充。代码平铺 Code 根目录。

## 阶段映射
| 阶段 | 代码文件 | 文档 | 状态 |
|------|----------|------|------|
| 初始化 | — | PLAN.md / PLAN_details.md / docs/ / Reference/ | ✅ |
| M1 | step0_config.py | docs/sums/sum_1_数据侦察与证据链.md | ✅ |
| M1 | step0_loader.py → output/clean/* | docs/migration_prompt.md Step4 口径 | ✅ |
| M1 | step0_eda.py → figures/eda/figD1-D9 | docs/sums/sum_1（D1-D9 实证表） | ✅ |
| M2 | step1.0_baseline_schedule.py | docs/sums/sum_2 + specs/spec_M2 | ✅ |
| M2 | step1.1_forecast_arena.py | docs/sums/sum_2（竞技榜/类选） | ✅ |
| M2 | step1.2_robust_schedule.py | docs/sums/sum_2（κ_ε/E1） | ✅ |
| M2 | step1.2+/step1.2++ | docs/sums/sum_2（机理/实验群） | ✅ |
| M2 | step1.3/1.4/1.5/1.6（补强收尾） | docs/sums/sum_2（严谨性/构造数据） | ✅ |
| R0 | step1.2_robust_schedule.py（校准段 2352-2375 修复） | docs/sums/sum_3 | ✅ |
| R1 | step1.0_baseline_schedule.py（c_r(h) 模板升级） | docs/sums/sum_3 | ✅ |
| R2 | step1.7_frozen_structure.py（冻结段结构研究） | docs/sums/sum_3 | ✅ |
| R3 | step0.5_soc_rebuild.py（SOC 递推重建） | docs/sums/sum_3 | ✅ |
| M3 | step2.0_construct.py（Q2 构造层） | docs/sums/sum_4 | ✅ |
| M3 | step2.1_nsga2.py（自研 NSGA-II） | docs/sums/sum_4 | ✅ |
| M3 | step2.2_rule_extraction.py（CART 规则） | docs/sums/sum_4 | ✅ |
| M3 | step2.3_delay_scan.py（时延裁决+proof） | docs/sums/sum_4 | ✅ |
| B1 | step1.8_deploy_gate.py（类选 v3 部署口径） | docs/sums/sum_5 | ✅ |
| B5 | step1.9_e1_v2.py（E1 干净对照） | docs/sums/sum_5 | ✅ |
| B6 | step2.4_method_arena.py / step2.4+_verdict.py（五方法裁决） | docs/sums/sum_5 | ✅ |
| C4 | step3.0_lp_baseline.py（Q3 LP 框架+标定） | docs/sums/sum_5 | ✅ |
| B7 | step1.8_deploy_gate.py v4（类选集合级+假阳性清零） | docs/sums/sum_6 | ✅ |
| B9 | step1.0_baseline_schedule.py（充电口径选项） | docs/sums/sum_6 | ✅ |
| B10 | step1.2+_rolling_kappa.py（滚动 κ 闭环） | docs/sums/sum_6 | ✅ |
| C5-8 | step2.5_materials.py（甘特图/边际/剪枝/Sobol 探索） | docs/sums/sum_6 | ✅ |
| D | step3.0 --all / step3.1_scenario_mpc.py（Q3 探路扩大） | docs/sums/sum_6 | ✅ |
| P1/A1/A2 | step1.2+_deploy_retrain.py（说明5 重训+冻结段覆盖率） | docs/sums/sum_7 | ✅ |
| P2 | step1.2++_t3_reverse.py（T3 反证链） | docs/sums/sum_7 | ✅ |
| A3-5/C2/C5 | step2.5+_review_fix.py（跨段/分类型/区域NU/TOPSIS/边际） | docs/sums/sum_7 | ✅ |
| B3 | step1.9_e1_v2.py（E1 真 κ 版+语义修正） | docs/sums/sum_7 | ✅ |
| C4 | tests/test_algorithm_correctness.py（算法正确性） | docs/sums/sum_7 | ✅ |
| 审查 | docs/reviews/review_A0/A1-A6/B8（错误清算/审查/正确率清单） | — | ✅ |
| 总结 | docs/sums/sum_8_高层总结与论文叙事.md | 论文叙事总纲 | ✅ |
| S9 | step0.6_identity_proof.py（恒等式 7 关证明 I1-I13） | docs/sums/sum_9 + CONSTITUTION 恒等式集 | ✅ |
| S9 | step0.6_spectrum.py（D1-D7 漂移/相关/周期/变点全谱） | docs/sums/sum_9 | ✅ |
| S9 | step0.6_thresholds.py（T1-T9 临界点：W 公式/SellLimit/SOC 边界） | docs/sums/sum_9 + CONSTITUTION | ✅ |
| S9 | step0.6_mechanism.py（M1-M4：三因子分解/分数 Overlap 破案） | docs/sums/sum_9 | ✅ |
| L1 | step0.7_nonai_layered.py（NonAI 恒等式分层） | output/robust/nonai_layered.json | ✅ |
| L4 | step0.7_rolling_energy.py（滚动重估适用域裁决） | output/robust/rolling_energy_price.json | ✅ |
| L10 | step1.1/1.5（点模型 cov 剔除 + seg_price_level train-only + QRF 修复） | tests/test_identity.py 守卫 | ✅ |
| Q3-M4 | step3.2_q3_indicators.py（四指标评估器+基准对照） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.3_q3_dr_reverse.py（DR 储能状态机逆向 E-DR1） | docs/sums/sum_10 + CONSTITUTION DR 条目 | ✅ |
| Q3-M4 | step3.3+_q3_model_evolve.py（M0/M1/M2 演进+三对照+E2a/E2b） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.3++_q3_pareto.py（E1 共线/E3 双方法/E6 坍缩） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.4_q3_ablation.py（E4-E9 判别实验群） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.4+_q3_storage_rules.py（CART 策略规则+模拟损失） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.5_q3_mpc.py（确定性等价 MPC+open-loop 对照+窗口效应） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.6_q3_sobol.py（自实现 Saltelli+LHS+V2 验证门） | docs/sums/sum_10 | ✅ |
| Q3-M4 | step3.7_q3_cross_experiments.py（X0-X14 交叉实验：斜坡/规则逐时/结算段/主时段指标） | docs/sums/sum_10 §10 | ✅ |
| Q3-M4 | step3.7+_q3_mutex.py（充放互斥 MILP M0x——传送带假象） | docs/sums/sum_10 §10.1 | ✅ |
| Q3-M4 | step3.7++_q3_extras.py（X15 σ 区分度 + X16 Sobol 对称） | docs/sums/sum_10 §10.2 | ✅ |
| Q3-M4 | solve_m3/main_period_indicators（M3_final 主口径回灌） | docs/sums/sum_10 §11 | ✅ |
| Q3-M4 | tests/test_q3.py（22 守卫，175 全绿） | docs/sums/sum_10 | ✅ |
| Q4 | step4.0_q4_bilevel.py（正式预算 40×30×3 + 收敛曲线 + 种子方差） | docs/sums/sum_11 | ✅ |
| Q4 | step4.1_q4_indicators.py（六指标章程/D 口径/维度预检/α 扫描/峰谷比预扫） | docs/sums/sum_11 | ✅ |
| Q4 | step4.2_q4_shadow.py（数值差分影子价格 + 对偶退化诊断 dual_verify） | docs/sums/sum_11 §五 | ✅ |
| Q4 | step4.3_q4_scenarios.py（压力矩阵 + 波动峰值双口径 + 碳杠杆三族 + lever_floor） | docs/sums/sum_11 §二/§四 | ✅ |
| Q4 | step4.4_q4_ablation.py（组件消融/joint vs alternate/单调性） | docs/sums/sum_11 §六 | ✅ |
| Q4 | step4.5_q4_rules.py（CART 迁移规则 + 规则化代价） | docs/sums/sum_11 §六 | ✅ |
| Q4 | step4.6_q4_collapse.py（前沿坍缩归因：判别 A 极端策略 + 判别 B oracle） | docs/sums/sum_11 §三 | ✅ |
| Q4 | tests/test_q4_v2.py（+10 定稿守卫，195 全绿） | docs/sums/sum_11 §六 | ✅ |

## 公式映射（论文编号 ↔ 代码位置）
| 论文公式 | 代码文件:行号 | 说明 |
|----------|--------------|------|
| 恒等式集 I1-I13（7 关证明，CONSTITUTION 入册） | step0.6_identity_proof.py:run_identity | identity_proof.json |
| I1 IT=NonAI+AI | step0.6_identity_proof.py:main（I1_it_equals_nonai_plus_ai） | rel 6e-10 |
| I2 功率平衡 G+W+Pd=Load+Pc+S+Q | step0.6_identity_proof.py:main（I2_power_balance） | W=Available 全进 |
| I10 消纳=c_r(h)×Total_Load | step0.6_identity_proof.py:main（I10） | 主时段命中 100% |
| W 精确公式 round(800+300sin,2) | step0.6_thresholds.py:t1_w_template + step3.1_scenario_mpc.py:gen_scenarios | 残差 0.0043MW |
| 白名单可达集合（时延≤MaxLatency） | step0_loader.py:build_whitelist | 预计算三层可达集 |
| 任务小时占用展开（GPU-hour 重叠折算） | step0_loader.py:expand_occupancy | 195,047 行占用表 |
| 18 序列聚合（区域×类型 GPU 需求） | step0_loader.py:build_series | series_gpu_demand.csv |
| 质量检查（NaN/矛盾任务/时段完整性） | step0_loader.py:check_quality | quality_report.json |
| 消纳模板 c_r(h)（U=c_r(h)·D，R1 升级） | step1.0_baseline_schedule.py:fit_consume_ratio | baseline_metrics.json |
| 四指标评估器（成本/碳/利用率/超容，模板口径） | step1.0_baseline_schedule.py:evaluate_schedule | local/greedy_hourly.csv |
| η 可预测性（η=1−Var(残差)/Var(Y)） | step1.3_rigor_analysis.py:a1_eta_profile | rigor_pack.json |
| 分位数预测 q_α（pinball 最小化） | step1.1_forecast_arena.py:QuantileEnsemble | fuse_quantiles_task.csv |
| 验证门 v2（cov+宽度+pinball 守卫） | step1.1_forecast_arena.py:apply_gate | arena_table.csv |
| 融合（不确定性元特征 Ridge stacking） | step1.1_forecast_arena.py:fuse_energy | fuse_point_energy.csv |
| κ_ε 预留（κ=1−q_{1−ε}/C_r，校准段 2352-2375） | step1.2_robust_schedule.py:compute_kappa | kappa_fit.json |
| E1 三方对照（gap<5pp 判据） | step1.2_robust_schedule.py:run_e1 | e1_three_way.json |
| 套利分解（迁移/错峰/oracle 上界） | step1.3_rigor_analysis.py:a3_oracle_upper_bound | rigor_pack.json |
| 冻结段结构（判别/条件覆盖率/滚动重估/功效曲线） | step1.7_frozen_structure.py:main | frozen_structure.json |
| SOC 递推重建（分区效率 + E 区偏移修正） | step0.5_soc_rebuild.py:rebuild_region | soc_rebuilt.csv |
| 冻结段最终评估（三段协议闭环） | step1.5_frozen_test.py:main | frozen_test.json |
| 构造数据指纹（KS/对称性/零膨胀） | step1.6_generator_fingerprint.py:f1_fingerprints | generator_fingerprint.json |
| Q2 构造层（白名单+排序启发式，策略参数化） | step2.0_construct.py:schedule_constructive | construct_schedule.csv |
| Q2 四目标评估（Cost₂/CE₂/Lat/NU，模板口径） | step2.0_construct.py:evaluate_4obj | construct_metrics.json |
| 时延 GPU-hours 加权（ω=L(s,r)） | step2.0_construct.py:compute_latency | — |
| Q2 NSGA-II（策略阈值进化，Deb 2002） | step2.1_nsga2.py:run_seed | nsga2_front.csv |
| Q2 规则提取（CART 路径→运营规则） | step2.2_rule_extraction.py:extract_rules | rules.json |
| 时延形式裁决（T1 纯加权，T2/T3 零违约） | step2.3_delay_scan.py:delay_forms | delay_scan.json |
| baseline_proof（四方案四目标对照） | step2.3_delay_scan.py:s2_baseline_proof | baseline_proof.json |
| 类选 v3（部署口径复检，MAPE 降幅≥5%） | step1.8_deploy_gate.py:main | deploy_arena.csv |
| E1 v2 干净对照（套利/预留分解） | step1.9_e1_v2.py:main | e1_v2.json |
| 五方法判别（NSGA-II/III/MOEA-D/ALNS/拉格朗日） | step2.4_method_arena.py:main | method_front_*.csv |
| 五方法裁决（2D HV/成本端点共识） | step2.4+_verdict.py:main | method_verdict.json |
| Q3 LP（附件1 平衡+弃电下界，HiGHS） | step3.0_lp_baseline.py:solve_region | lp_calibration.json |
| Q3 LP 六区域全量（西区负成本/东区充电瓶颈） | step3.0_lp_baseline.py:solve_all_regions | lp_all_regions.json |
| Q3 MPC 场景（AR(1)+K-means+窗口效应） | step3.1_scenario_mpc.py:main | scenario_report.json |
| 滚动 κ（nowcast 336h，B1/B10） | step1.2+_rolling_kappa.py:rolling_q | kappa_fit_rolling.json |
| 说明5 重训+任务侧冻结段覆盖率（P1/A1/A2） | step1.2+_deploy_retrain.py:main | deploy_retrain.json |
| T3 反证链（白名单放宽，P2） | step1.2++_t3_reverse.py:main | t3_reverse.json |
| E1 v2 真 κ 版（κ 过度预留发现，B3） | step1.9_e1_v2.py:main | e1_v2.json |
| 跨段结清/分类型/区域NU/TOPSIS/时延边际（A3-5/C2/C5） | step2.5+_review_fix.py:main | review_fix.json |
| 素材四件套（甘特图/边际/剪枝/Sobol 探索，C5-8） | step2.5_materials.py:main | materials.json |
| 充电口径选项（B2/B9） | step1.0_baseline_schedule.py:evaluate_schedule | — |
| 类选 v4（集合级 Jaccard 主指标） | step1.8_deploy_gate.py:main | deploy_gate.json |
| Q4 六指标评估器（六目标最小化，α 固定 0.5） | step4.1_q4_indicators.py:evaluate_q4_six | q4_indicators.json |
| Q4 下层储能 LP（M3_final + 碳/峰值/卖电限额参数化） | step4.0_q4_bilevel.py:solve_lower_constrained | — |
| Q4 影子价格（数值差分 + HiGHS marginals 退化诊断） | step4.2_q4_shadow.py:shadow_prices/dual_verify | q4_shadow.json |
| Q4 碳杠杆上限（mig 旋钮扫描 0.239%） | step4.3_q4_scenarios.py:lever_floor_scan | q4_pressure.json |
| Q4 坍缩归因（oracle 解析边际贪心迁移） | step4.6_q4_collapse.py:oracle_migration | q4_collapse.json |

## 产物清单（output/clean/）
workload_clean.csv / region_time_clean.csv / whitelist.csv / occupancy_local.csv
series_gpu_demand.csv / series_arrivals.csv / occupancy_parallel.csv
storage_params.csv / quality_report.json / figures/eda/eda_summary.json

## Phase 文档
| Phase | Reference/docs/PhaseN | 内容 |
|-------|------------------------|------|
| （不适用） | | 竞赛题无开源代码对照（paper_code_summary/reference_analysis/guidance 为论文复现项目模板）——决策：跳过 PhaseN 三文件，方法论学习记入 Reference/sums/（sum_9 已建） |

## 方法学记录（Reference/sums/）
| 文件 | 内容 | 状态 |
|------|------|------|
| sum_9_结构侦察方法学.md | S9 结构侦察：10 条系统性教训 + 侦察工具箱 + 适用边界 | ✅ |
| sum_10_高层方法论语病与更新.md | math-methods v2 修订记录：高层缺陷 10 条 + 抽象检查项 8 条 + 落地清单 | ✅ |
