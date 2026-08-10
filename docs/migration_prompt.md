# Migration Prompt — C 题项目入口

## Step 1: 加载方法论 skill（项目级 .opencode/skills/）

- `math-methods` — 全局方法哲学（逻辑链/三关闸门/结论分级/国特思维）
- `math-verification` — 模型验证思维（验证工具箱/强度分级/证伪轮——异常数字先过判别实验）
- `math-materials` — 研究素材组织思维（素材七要素/口径声明/数字总表/路线叙事——论文手交接协议）
- `math-review` — 数模复查协议（质量门：五大类拷问/问题分级 P0-P1-P2/收敛判定——
  对已完成一问做系统性自检，产出"第 N 问复查报告.md"）
- `math-auto` — 自主迭代执行协议（集大成五层能力模型：战略领导/指挥编排/
  设计探索/基层执行/支撑记录提交 + 研究循环控制（质量门收敛）——用户不在场时
  从起点跑到终点）
  回灌+记录+提交——用户不在场时从起点跑到终点）
- `math-name` — 代码命名规范；`agent-memory` — 文档体系

## Step 1: 加载 project-reference skill

加载 `agent-memory`，了解 `Code/docs/` 下的 `logs/`、`sums/`、`specs/` 目录结构。

---

## Step 2: 阅读全部关键文档

### 必读（项目状态）
- `Code/PLAN.md` — 作战概要（一页）
- **`Code/PLAN_details.md` — 完整数学模型详细实施方案（逐句解析/证据链 D1-D9/四问数理推导/预测竞技榜/实验与消融设计/附录参数表）**
- `Code/README.md` — 核心结果速览（lag1=0.011, 弃电 775 万 MWh, 超容 E30/F67h 等）
- `Code/docs/specs/` 下所有 .md（随架构决策生成）

### 必读（决策历史）
- `Code/docs/sums/sum_1` 至 `sum_8`（数据侦察→论文叙事总纲）
- **`Code/docs/sums/sum_9_结构完备性全谱侦察.md` — 恒等式集 I1-I13 七关证明/W 精确公式/价格变点/分数 Overlap 破案/NonAI 分层 L1/L10 评估链修复**
- **`Code/docs/sums/sum_10_Q3全量执行.md` — DR 状态机逆向/M3_final 主口径/互斥物理上界 M0x/传送带假象**
- **`Code/docs/sums/sum_11_Q4定稿.md` — Q4 正式预算/前沿坍缩归因（储能结构锁定）/碳杠杆上限 0.239%/对偶退化发现/195 全绿**
- **`Code/docs/sums/sum_12_结论证伪与口径修正.md` — 双锚净化（D −38.3%）/波动截断修正/碳构造上界 1.4%/MPC 归因修正/#5 全网峰值/#18 Sobol 交互/215 全绿**
- `Code/docs/reviews/review_A0_错误结论清算.md` — 错误 vs 结论三关清算
- `Code/docs/reviews/review_A1-A6_审查定稿.md` — 审查结论
- `Code/docs/reviews/review_B8_正确率清单.md` — 全指标透明化
- `Code/docs/reviews/review_A11_错误修复总账.md` — 动态错误-修复总账（#1-51，含 Q4 轮 #44-51）
- `Code/Reference/docs/CONSTITUTION.md` — 硬约束速查（口径/恒等式集/约束/预测纪律/工程纪律）
- `Code/Reference/docs/INDEX.md` — 公式↔代码↔文档映射

### 速读（了解近期动态）
- `Code/docs/logs/latest_14.log` — 结论证伪轮（双锚净化/波动截断/碳上界/MPC 归因修正）
- `Code/docs/logs/latest_12.log` — Q4 定稿全记录（正式预算/坍缩归因/碳杠杆/对偶退化）
- `Code/docs/logs/latest_11.log` — Q3 发现回灌定稿（M3_final 主口径/斜坡假象修正/M0x gap 验证）
- `Code/docs/logs/latest_10.log` — Q3 M4 全量执行（DR 状态机/传送带假象/交叉实验）

### 必读（代码现状）
- 运行 `python step0_loader.py` 生成 clean/ 产物（或核验 output/clean/ 已存在）
- 运行 `python step0_eda.py` 生成 figures/eda/figD1-D9
- 运行 `python step0.5_soc_rebuild.py` 生成 SOC 递推重建（Q3 地基）
- 运行 `python step1.7_frozen_structure.py` 生成冻结段结构研究（R2）
- 运行 `python step0.6_identity_proof.py` 生成恒等式集证明（S9；tests 守卫依赖）
- 运行 `python step0.7_nonai_layered.py` 生成 NonAI 分层实验（L1）
- 运行 `python step3.3_q3_dr_reverse.py` 生成 DR 储能状态机逆向（Q3 主口径基础）
- 运行 `python step3.8_q3_rigor.py` 生成 Q3 严谨性四件套（双锚分解/斜坡活跃率/终态对称性/分窗分布）
- 运行 `python step4.0_q4_bilevel.py` 生成 Q4 双层正式前沿（默认 40×30×3；
  `--smoke` 12/4/1 供 CI；`--pop/--gen/--seed` 灵活预算；产物含收敛曲线+种子方差）
- 运行 `python step4.1_q4_indicators.py` 生成 Q4 六指标章程
- 运行 `python step4.2_q4_shadow.py` 生成影子价格（数值差分+对偶退化诊断）
- 运行 `python step4.3_q4_scenarios.py` 生成压力矩阵（波动峰值双口径/碳杠杆三族）
- 运行 `python step4.4_q4_ablation.py` / `step4.5_q4_rules.py` 消融与规则层
- 运行 `python step4.6_q4_collapse.py` 生成前沿坍缩归因判别（储能结构锁定）
- 测试：`python -m pytest tests/ -q`（当前 215 passed）

---

## Step 3: 恢复当前任务上下文

### 已完成（按 Phase）
- **Phase 0（题前侦察）**: 三题对比选定 C 题；数据侦察 D1-D9 全实证（lag1=0.011 白噪声 / 新能源模板 / 基线超容 E30·F67h / 弃电 775 万 MWh / 相位错位 / 白名单三层 / 东区弃电率 88-91%）；方法方案四问定稿
- **Phase 1（M1 地基）**: step0 三件套（config/loader/eda）跑通并验证；output/clean/ 全量产物（质量报告零 NaN、白名单、占用 195,047 行、18 序列）；figures/eda/figD1-D9 九图；sum_1 落盘；PLAN_details v2.0 完整建模方案
- **Phase 2（M2 预测系统+调度闭环 ✅）**: step1.0 基线锚点（消纳系数逆向，锚点全中）→ step1.1 竞技榜（42×8 全量，任务侧 143/144 拒、TabPFN 统治结构类）→ step1.2 预留闭环（κ_ε=0.05）+ E1 胜负手（gap 0.073pp，结构套利主导）→ step1.2+ 机理（空间 4.6%>>时间 0.001%）→ step1.2++ 实验群 → step1.3/1.4 严谨性补强（η 档案/oracle 上界/敏感性全稳）→ step1.5/1.6 Q1 收尾（冻结段/构造数据指纹）；sum_2 落盘；50+ 测试全绿
- **Phase 2+（R0-R3 建模缺口闭环 ✅）**: 队友批判驱动重构——①消纳形式升级 c_r→c_r(h) 日内模板（U=c(h)·D，命中率 100%，正弦参数化）②校准段修复 2352-2375（原误用冻结段 2376-2399，漂移假象解除，伪/真差 0.2pp）③冻结段结构研究（需求抬升 z=2.86/100 百分位，滚动重估修复 0.917→0.944，超容率 4.86%≈ε）④SOC 效率分区制（A/B/C 0.93/0.92；D/E/F 0.94/0.93）+ E 区偏移修正（官方背书）；sum_3 落盘；70 测试全绿
- **Phase 3（M3 Q2 三段式 ✅ + 准确层修复 ✅）**: step2.0 构造层（成本 −4.75%）→ step2.1 自研 NSGA-II（五方法裁决：成本端点五方法收敛=构造解）→ step2.2 CART 规则 → step2.3 时延裁决（T1）+ baseline_proof（Q2 vs local −4.57%）→ 准确层 B1-B6（类选 v3/v4 集合级 Jaccard 0.80、E1 干净对照、五方法裁决）→ 地基 F1-F3（充电口径/滚动 κ 0.965/目标共线实锤）→ 16 项问题闭环（任务侧冻结段覆盖率 88.2%、跨段 89 结清等）→ **E1 语义修正：实际任务全知下预测与预留均无价值（Q2 语境）**；sum_4-7 落盘；95 测试全绿
- **Q3 探路 ✅（M4 主线暂缓）**: step3.0 LP（HiGHS 0.1s/区域、六区域全量：西区利用率 75-85%+负成本=弃电充电外送套利、东区充电功率瓶颈）+ step3.1 MPC 场景框架（AR(1)+K-means+窗口效应 D 0.06%/E 2.17%）
- **S9 结构完备性全谱侦察 ✅**: step0.6 四件套（恒等式 7 关证明 I1-I13/漂移相关周期变点全谱/临界点扫描/任务机制分解）+ step0.7 两实验（NonAI 恒等式分层 L1：E/F 冻结段 22.9→4.6、滚动重估适用域 L4）+ L10 评估链修复（点模型 cov 假象剔除/seg_price_level 训练段口径/QRF warning）；恒等式集入 CONSTITUTION；153 测试全绿
- **Q3 M4 全量 ✅（sum_10）**: DR 储能状态机逆向（时段模板六区同构/充放互斥/SOC 全深度循环）→ 模型演进 M0/M1/M2/M3_final → **M3_final 主口径**（时段状态机+结算段禁充+终态严格+生成器量级斜坡，主时段指标）→ **互斥物理上界 M0x**（传送带假象实证：无互斥 LP 同时充放 91-97%）→ 交叉实验矩阵（斜坡物理性/规则逐时/σ 区分度/Sobol 对称）→ MPC/Sobol；175 测试全绿
- **Q4 定稿 ✅（sum_11，本轮）**: 正式预算 40×30×3（收敛曲线+种子方差 0.014%）→ α 游戏化修复（固定 0.5）→ 前沿坍缩归因（**储能结构锁定**【实证】：跨度 0.14-0.55% vs Q2 4.57%，oracle 反噬 −3.4% 相位错配）→ 碳杠杆三族全败 + **任务层碳杠杆上限 0.239%**（结构性）→ 影子价格对偶验证门（**储能 LP 退化**【实证】：marginals 不可投产，数值差分为准，π_sell 唯一可行载体）→ 波动峰值双口径 → 195 测试全绿

### 待完成（按优先级）
1. **M6**: step5 消融 A-J + baseline_proof + 论文写作（叙事总纲见 sum_8；Q3 主口径见 sum_10 §11 + step3.8 双锚；Q4 口径见 sum_11 §八）
2. **Q3 反思 P2 项**（sum_11 §九）：全网峰值/波动 MPC 跨题闭环/循环寿命/CVaR/弃电边际价值曲线（按论文需要取舍）

### 当前聚焦
**Q4 已定稿**（正式预算 + 坍缩归因 + 碳杠杆结论 + 对偶退化，195 测试全绿）；下一步 M6（论文素材包）或 Q3 反思 P1 项。
文档维护中（sum_11/latest_12/总账 44-51/README/PLAN 已同步）。

---

## Step 4: 硬约束速查（直接遵守）

1. 口径公式（附件1 唯一口径，见 CONSTITUTION.md）：
   - AI_IT_Load(r,t) = Σ_i GPU_Demand×Overlap×GPU_Power(TaskType)，功率 0.08/0.10/0.16 MW/GPU
   - IT_Load = NonAI + AI；Total_Load = IT×PUE；功率平衡含弃电松弛项
   - SOC(t) = SOC(t−1) + ηc·Charge − Discharge/ηd；SOC(2406) ≥ InitialSOC
   - 效率分区制：A/B/C ηc=0.93/ηd=0.92；D/E/F ηc=0.94/ηd=0.93（R3 修正）
   - SOC 以递推公式重建（step0.5 产物；RegionE 数据表偏移 −1.0 系录入误差，官方确认）
   - 消纳口径：基线评估 U = c_r(h)·D 日内模板（R1 实证命中率 100%）；Closure 段 U=min(W,D)；优化场景可用全覆盖口径
2. 时延白名单：RT≤20ms / BI≤80ms / AT≤150ms（预计算可达集合，禁止出白名单调度）
3. 实时推理到达即开工（slack 1-7h 实证无弹性）；任务不可抢占/拆分/中途迁移
4. 时域：0-2399 决策；2400-2405 仅结清；2406 不安排任务
5. GPU 容量约束用 Available_GPU（A630 B585 C540 D1472 E1012 F966）
6. 预测切分：0-2351 训练 / 2352-2375 校准 / 2376-2399 冻结测试（R0 修复：校准段不得使用冻结段；冻结段仅验证）；2376-2399 调度输入=实际到达任务
7. 类选验证门：复杂模型须优于统计基线（MAPE 降幅≥5% 或覆盖率校准达标）才入选；统计基线永远在场
8. 环境：mathorcup（D:\Anaconda\envs\mathorcup\python.exe）；pandas 3.0.2
9. 命名：step{N}.{M}{suffix}_{desc}.py 平铺根目录；产物 output/、图 figures/
10. 论文每个数字可追溯到唯一文件（数据核查纪律）
11. Q4 口径（sum_11）：六指标六目标最小化（QOS=α·完成率+(1−α)·裕度，α 固定 0.5——
    评估器参数不可优化，α=1 目标游戏化总账 #45）；D=(NonAI+调度 AI)×PUE；
    正式预算 40×30×3（--smoke 12/4/1 供 CI）；储能结构锁定【实证】
    （任务层电力侧可优化面 0.14-0.55%，oracle 相位错配反噬 −3.4%）；
    碳限额 τ≥1% 结构性不可达（杠杆上限 0.239%）；影子价格数值差分为准
    （储能 LP 退化，HiGHS 对偶不可投产）；π_sell 唯一可行载体

## 论文素材包（docs/materials/，供论文手直接取用）

- 00_素材总览（索引/模板/核心发现速览/口径红线）
- 01 数据公理层 / 02 可预测性分层 / 03 任务层调度 / 04 储能层结构（T1 下界定理）/
  05 双层协同（T2 结构锁定）/ 06 风险与鲁棒 / 07 方法学与证伪
- 08 口径声明大全（引用数字前必查）/ 09 数字总表（数字×产物×分级）
- 10 路线叙事_完整故事线 + 10b 要点式（研究深度素材）
- 11 图素材清单（figures/paper/ 机理图 5 张 + 全部数据图）
- 全题改进率汇总：output/q5/baseline_proof_all.json

## Step 5: 回退策略
1. 先读 PLAN_details.md 对应章节（完整模型/公式/流程图）
2. 再看 docs/sums/ 学习记录 + Reference/docs/ 映射
3. 仍不确定 → 直接提问
