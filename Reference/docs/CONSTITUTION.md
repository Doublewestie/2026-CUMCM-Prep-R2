# CONSTITUTION — 项目硬约束速查

> 全局参数、口径、纪律。每约束一行。可调参数标注 [可调]。

## 口径（不可变）
- AI_IT_Load(r,t) = Σ_i GPU_Demand(i) × Overlap(i,r,t) × GPU_Power(TaskType_i)  [power_mapping 唯一口径]
- IT_Load = NonAI_IT_Load + AI_IT_Load；Total_Load = IT_Load × PUE(r)
- 功率平衡: GridPurchase + AvailableRenewable + DischargePower = Total_Load + ChargePower + GridSell + Curtailment
- CarbonEmission = GridPurchase × CarbonIntensity；电费 = GridPurchase×Price − GridSell×SellPrice
- SOC 递推: SOC(t) = SOC(t−1) + ηc·ChargePower(t) − DischargePower(t)/ηd
- 新能源利用率 = (直接消纳 + 新能源充电 + 外送) / 可用新能源

## 约束（不可变）
- GPU 容量: 调度占用 ≤ Available_GPU（A630 B585 C540 D1472 E1012 F966）
- 时延白名单: 实时≤20ms / 批量≤80ms / 训练≤150ms（预计算可达集合）
- 实时推理: 到达即开工；任务不可抢占/不可拆分/不可中途迁移
- 时域: 0-2399 决策；2400-2405 仅结清末端；2406 不安排任务
- 储能: MinSOC ≤ SOC ≤ Cap；充放功率上限；SOC(2406) ≥ InitialSOC
- SOC 口径: 效率分区制（A/B/C 0.93/0.92；D/E/F 0.94/0.93）；SOC 以递推公式重建
  （RegionE 数据表全程偏移 −1.0 MWh 系录入误差，官方确认；Q3 用 step0.5 产物）
- 购售电: MaxGridImport / MaxGridExport / SellLimit 硬约束

## 预测纪律
- 切分: 0-2351 训练 / 2352-2375 调参 / 2376-2399 测试；2376-2399 调度输入=实际到达任务
- 验证门: 复杂模型须优于统计基线（MAPE 降幅≥5% 或覆盖率校准达标）才入选

## 工程纪律
- 环境: mathorcup（D:\Anaconda\envs\mathorcup\python.exe）；pandas 3.0.2
- 命名: step{N}.{M}{suffix}_{desc}.py；产物落 output/，图落 figures/
- 论文每个数字可追溯到唯一文件
