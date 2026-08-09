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
- 类选 v4: 部署口径复检（全段拟合+冻结段）为最终裁决；主指标=集合级 Jaccard（0.80）；
  第一名一致率废弃（CV 噪声 11/24 + 24h 排名物理不可辨）；任务侧部署验证=无崩溃
- κ 双版本: 静态（kappa_fit.json，历史追溯）/ 滚动 nowcast 336h（kappa_fit_rolling.json，
  冻结段覆盖 0.944，论文优先引用）；预留只压缩弹性任务容量（RT 刚性不受预留）
- E1 语义: 实际任务全知下预测与预留均无价值（Q2 语境，说明 2 推论）；
  κ 过度预留实证（真 κ 版 gap 4.81pp = 需求水平被当预留）
- 跨段任务定义: 执行跨入收尾段（StartHour+dur>2399），实测 89 个（RT 24+弹性 65）

## 工程纪律
- 环境: mathorcup（D:\Anaconda\envs\mathorcup\python.exe）；pandas 3.0.2
- 命名: step{N}.{M}{suffix}_{desc}.py；产物落 output/，图落 figures/
- 论文每个数字可追溯到唯一文件
- 评估器口径: 基线/对照用模板消纳（U=c_r(h)·D）；B9 充电口径选项（include_baseline_charge）
  用于与题目基线绝对数字对照；相对比较同口径稳健
