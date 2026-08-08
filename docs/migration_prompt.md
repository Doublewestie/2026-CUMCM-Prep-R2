# Migration Prompt — C 题项目入口

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
- `Code/docs/sums/sum_1_数据侦察与证据链.md` — D1-D9 全实证 + 东区弃电率新发现
- `Code/docs/sums/sum_2_M2_预测系统与调度闭环.md` — 竞技榜/κ_ε/E1/套利结构/融合
- **`Code/docs/sums/sum_3_建模缺口闭环.md` — R0-R3 修复（消纳模板/SOC/校准段/冻结段结构）**
- `Code/docs/sums/sum_4_M3_Q2三段式调度.md` — 构造→NSGA-II→规则
- **`Code/docs/sums/sum_5_准确层修复与Q3铺路.md` — B1-B6 修复闭环/五方法裁决/Q3 LP 铺路**
- `Code/docs/reviews/review_A0_错误结论清算.md` — 错误 vs 结论三关清算
- `Code/docs/reviews/review_A1-A6_审查定稿.md` — 审查结论
- `Code/Reference/docs/CONSTITUTION.md` — 硬约束速查（口径/约束/预测纪律/工程纪律）
- `Code/Reference/docs/INDEX.md` — 公式↔代码↔文档映射

### 速读（了解近期动态）
- `Code/docs/logs/latest_4.log` — R0-R3 建模缺口闭环
- `Code/docs/logs/latest_3.log` — agent-memory 衔接修正 + PLAN_details v2.1

### 必读（代码现状）
- 运行 `python step0_loader.py` 生成 clean/ 产物（或核验 output/clean/ 已存在）
- 运行 `python step0_eda.py` 生成 figures/eda/figD1-D9
- 运行 `python step0.5_soc_rebuild.py` 生成 SOC 递推重建（Q3 地基）
- 运行 `python step1.7_frozen_structure.py` 生成冻结段结构研究（R2）
- 测试：`python -m pytest tests/ -q`（当前 70 passed）

---

## Step 3: 恢复当前任务上下文

### 已完成（按 Phase）
- **Phase 0（题前侦察）**: 三题对比选定 C 题；数据侦察 D1-D9 全实证（lag1=0.011 白噪声 / 新能源模板 / 基线超容 E30·F67h / 弃电 775 万 MWh / 相位错位 / 白名单三层 / 东区弃电率 88-91%）；方法方案四问定稿
- **Phase 1（M1 地基）**: step0 三件套（config/loader/eda）跑通并验证；output/clean/ 全量产物（质量报告零 NaN、白名单、占用 195,047 行、18 序列）；figures/eda/figD1-D9 九图；sum_1 落盘；PLAN_details v2.0 完整建模方案
- **Phase 2（M2 预测系统+调度闭环 ✅）**: step1.0 基线锚点（消纳系数逆向，锚点全中）→ step1.1 竞技榜（42×8 全量，任务侧 143/144 拒、TabPFN 统治结构类）→ step1.2 预留闭环（κ_ε=0.05）+ E1 胜负手（gap 0.073pp，结构套利主导）→ step1.2+ 机理（空间 4.6%>>时间 0.001%）→ step1.2++ 实验群 → step1.3/1.4 严谨性补强（η 档案/oracle 上界/敏感性全稳）→ step1.5/1.6 Q1 收尾（冻结段/构造数据指纹）；sum_2 落盘；50+ 测试全绿
- **Phase 2+（R0-R3 建模缺口闭环 ✅）**: 队友批判驱动重构——①消纳形式升级 c_r→c_r(h) 日内模板（U=c(h)·D，命中率 100%，正弦参数化）②校准段修复 2352-2375（原误用冻结段 2376-2399，漂移假象解除，伪/真差 0.2pp）③冻结段结构研究（需求抬升 z=2.86/100 百分位，滚动重估修复 0.917→0.944，超容率 4.86%≈ε）④SOC 效率分区制（A/B/C 0.93/0.92；D/E/F 0.94/0.93）+ E 区偏移修正（官方背书）；sum_3 落盘；70 测试全绿

### 待完成（按优先级）
1. **M3**: step2 三段式调度 + 规则提取（时延目标 T1-T3 两阶段探索；Q2 定调=迁移为主，E1a 实证）
2. **M4**: step3 LP + MPC + Sobol
3. **M5**: step4 双层 + 压力矩阵
4. **M6**: step5 消融 A-J + baseline_proof + 论文素材包

### 当前聚焦
**M3：Q2 碳感知多目标调度（迁移为主）** — 构造层（白名单+价差启发式，E1a 代码即雏形）→ NSGA-II → CART 规则提取

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

## Step 5: 回退策略
1. 先读 PLAN_details.md 对应章节（完整模型/公式/流程图）
2. 再看 docs/sums/ 学习记录 + Reference/docs/ 映射
3. 仍不确定 → 直接提问
