# Migration Prompt — C 题项目入口

## Step 1: 加载 project-reference skill

加载 `agent-memory`，了解 `Code/docs/` 下的 `logs/`、`sums/`、`specs/` 目录结构。

---

## Step 2: 阅读全部关键文档

### 必读（项目状态）
- `Code/PLAN.md` — 作战概要（一页）
- **`Code/PLAN_details.md` — 完整数学模型详细实施方案（568 行：逐句解析/证据链 D1-D9/四问数理推导/预测竞技榜/实验与消融设计/附录参数表）**
- `Code/README.md` — 核心结果速览（lag1=0.011, 弃电 775 万 MWh, 超容 E30/F67h 等）
- `Code/docs/specs/` 下所有 .md（随架构决策生成）

### 必读（决策历史）
- `Code/docs/sums/sum_1_数据侦察与证据链.md` — D1-D9 全实证 + 东区弃电率新发现
- `Code/Reference/docs/CONSTITUTION.md` — 硬约束速查（口径/约束/预测纪律/工程纪律）
- `Code/Reference/docs/INDEX.md` — 公式↔代码↔文档映射

### 速读（了解近期动态）
- `Code/docs/logs/latest_2.log` — 文档重构与代码迁移（PLAN_details v2.0、src→根目录）
- `Code/docs/logs/latest_1.log` — M1 地基完成（step0 加载器+EDA 证据链）

### 必读（代码现状）
- 运行 `python step0_loader.py` 生成 clean/ 产物（或核验 output/clean/ 已存在）
- 运行 `python step0_eda.py` 生成 figures/eda/figD1-D9
- 测试：`python -m pytest tests/ -q`（tests/ 待建）

---

## Step 3: 恢复当前任务上下文

### 已完成（按 Phase）
- **Phase 0（题前侦察）**: 三题对比选定 C 题；数据侦察 D1-D9 全实证（lag1=0.011 白噪声 / 新能源模板 / 基线超容 E30·F67h / 弃电 775 万 MWh / 相位错位 / 白名单三层 / 东区弃电率 88-91%）；方法方案四问定稿
- **Phase 1（M1 地基）**: step0 三件套（config/loader/eda）跑通并验证；output/clean/ 全量产物（质量报告零 NaN、白名单、占用 195,047 行、18 序列）；figures/eda/figD1-D9 九图；sum_1 落盘；PLAN_details v2.0 完整建模方案

### 待完成（按优先级）
1. **M2**: step1.0_baseline_schedule.py（基础调度基线，全篇对照锚点）→ step1.1_forecast_arena.py（四家族预测竞技榜）→ step1.2_robust_schedule.py（预留闭环+最后24h甘特图）
2. **M3**: step2 三段式调度 + 规则提取（时延目标 T1-T3 两阶段探索）
3. **M4**: step3 LP + MPC + Sobol
4. **M5**: step4 双层 + 压力矩阵
5. **M6**: step5 消融 A-J + baseline_proof + 论文素材包

### 当前聚焦
**M2：预测系统（四家族竞技榜，任务侧预期全拒=类选结论）** + 基础调度基线

---

## Step 4: 硬约束速查（直接遵守）

1. 口径公式（附件1 唯一口径，见 CONSTITUTION.md）：
   - AI_IT_Load(r,t) = Σ_i GPU_Demand×Overlap×GPU_Power(TaskType)，功率 0.08/0.10/0.16 MW/GPU
   - IT_Load = NonAI + AI；Total_Load = IT×PUE；功率平衡含弃电松弛项
   - SOC(t) = SOC(t−1) + ηc·Charge − Discharge/ηd；SOC(2406) ≥ InitialSOC
2. 时延白名单：RT≤20ms / BI≤80ms / AT≤150ms（预计算可达集合，禁止出白名单调度）
3. 实时推理到达即开工（slack 1-7h 实证无弹性）；任务不可抢占/拆分/中途迁移
4. 时域：0-2399 决策；2400-2405 仅结清；2406 不安排任务
5. GPU 容量约束用 Available_GPU（A630 B585 C540 D1472 E1012 F966）
6. 预测切分：0-2351 训练 / 2352-2375 调参 / 2376-2399 测试；2376-2399 调度输入=实际到达任务
7. 类选验证门：复杂模型须优于统计基线（MAPE 降幅≥5% 或覆盖率校准达标）才入选；统计基线永远在场
8. 环境：mathorcup（D:\Anaconda\envs\mathorcup\python.exe）；pandas 3.0.2
9. 命名：step{N}.{M}{suffix}_{desc}.py 平铺根目录；产物 output/、图 figures/
10. 论文每个数字可追溯到唯一文件（数据核查纪律）

## Step 5: 回退策略
1. 先读 PLAN_details.md 对应章节（完整模型/公式/流程图）
2. 再看 docs/sums/ 学习记录 + Reference/docs/ 映射
3. 仍不确定 → 直接提问
