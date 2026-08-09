# CONSTITUTION — 项目硬约束速查

> 全局参数、口径、纪律。每约束一行。可调参数标注 [可调]。

## 口径（不可变）
- AI_IT_Load(r,t) = Σ_i GPU_Demand(i) × Overlap(i,r,t) × GPU_Power(TaskType_i)  [power_mapping 唯一口径]
- IT_Load = NonAI_IT_Load + AI_IT_Load；Total_Load = IT_Load × PUE(r)
- 功率平衡: GridPurchase + AvailableRenewable + DischargePower = Total_Load + ChargePower + GridSell + Curtailment
- CarbonEmission = GridPurchase × CarbonIntensity；电费 = GridPurchase×Price − GridSell×SellPrice
- SOC 递推: SOC(t) = SOC(t−1) + ηc·ChargePower(t) − DischargePower(t)/ηd
- 新能源利用率 = (直接消纳 + 新能源充电 + 外送) / 可用新能源

## 恒等式集（step0.6 七关证明，output/clean/identity_proof.json，实证入册）
- I1 IT_Load = NonAI + Baseline_AI（六区逐点 rel 6e-10）；I13 Total_Load = IT_Load×PUE
- I2 功率平衡（W=Available 全进口径实证；UsedRenewable 变体不成立）
- I3 碳排 = 购电×碳强度；I4 NetGrid = 购−售；I7 充电 = 购电充电+新能源充电
- I5 利用率 ≡ 1−Q/W；I6 SOC 递推（分区效率）；I10 消纳 = c_r(h)×Total_Load（主时段）
- I11 生成口径：Baseline_AI = Σ g×P×overlap_frac（分数重叠；ceil 整点总量等价/逐时差 15%）
- W 精确公式: W = round(800+300·sin(2π(h−4)/24), 2)（六区同，max 残差 0.0043MW）
- 模板族（D6 周期谱）: renewable 日模板 / carbon 周模板 / GridSell·GridCharge·SOC 强周期
- 价格 t=1245 变点: 六区同步等比缩放 0.955（三时段同幅）
- 冻结段任务收尾结构: AI 漂移谱区域异构（A−21/B+18/C−28/D−20/E+55/F+52），AT 数量主导

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
- 临界点（step0.6_thresholds）: GridSell 顶 SellLimit（D 86%/E 66%/F 63% 饱和）；
  SOC 顶 Cap（A-C 62%/D-F 50-54%）；W 永不瓶颈；谷价充电阈值（价格>60 分位充电恒 0）
- NonAI 分层口径（L1）: NonAÎ = IT_Load 模板 − AI 实际（I1 投产；E/F 冻结段 22.9→4.6）；
  均值口径与直接模板数学等价（S2≡S0）——增益=AI 实际信息（说明2 合法）
- 点模型不产出 cov/width/pinball（L10：原 ±1.28σ 合成 = cov≈1.0 度量假象）；
  确定性序列（renewable/carbon 模板）覆盖率为退化指标
- **DR 储能状态机（Q3 逆向，sum_10）**: 时段模板六区同构——充电域 h0-4(5)+h22-23、
  放电域 h17-21、充放互斥、SOC 每日全深度循环（D 90→900 深度 810 模板化）、
  充放功率从不顶格（T4/T5 复核）；DR 为确定性日模板（D 一致率 1.000，
  A/B/C/F 0.94-0.96、E 0.89）；D 区 DR=Medium⟺Peak 严格，A 区 Medium 76%
  （区域异构，时段模板为通用规则）；I14 候选：充电时段规则
- **Q3 模型口径（M1，sum_10）**: 储能价值主口径=时段状态机约束 LP（M1）vs 基准；
  无约束 LP（M0）收益 ~90% 为模型自由度红利（D 区 1.72 亿 vs 真实价值 0.10 亿），
  不得作为储能价值引用；M1 充电 100% 弃电（Pc_grid=0，弃电优先，I7 归因）；
  E2a 结构性爬坡（D 258 MW 最优面内不可降）；成本-碳共线 0.91-1.00（E1）；
  成本-爬坡权衡面坍缩（跨度 ≤0.14%，成本与波动可兼得）
- **Q3 量化锚点（sum_10）**: 西区外送价值 D 1.43 亿/E 1.54 亿（E7）；
  SellLimit 100% 饱和（西区）、SOC Cap 67%（东区）（E5）；Cap×1.3 扩容价值
  A 906 万/D 1,142 万 vs E/F ~10 万（E8）；充电上限非瓶颈（E9，M1 框架）；
  规则化代价 1.5-5.8%（CART 可解释策略损失，step3.4+）；
  窗口效应 A 0.0/D 3.5/E 7.8/F 22.1%（F 储能循环周期 >24h，step3.5）；
  Sobol 归因：cost 主导 sell_scale（西区）/price_scale（东区）、碳排与波动主导 Cap
  （step3.6，V2 估计器验证误差 0.76%，LHS 采样——Sobol 同维跨序列相关 −0.7 陷阱）
- **充放互斥（step3.7+ 传送带假象）**: LP 储能建模**必须含充放互斥约束**
  （Pc·Pd=0；MILP 二进制 z_t）——无互斥时 LP 解 91-97% 小时同时充放
  （"充电吃弃电+放电供负荷"传送带，吞吐超物理上限 6 倍），制造 0.56-0.99 亿/区
  不物理收益（M0x−M0）；**M0x（互斥物理上界）比 M1（时段状态机）便宜
  4,000-7,300 万/区（D 区 222 vs 7,561 万）**——时段状态机严重次优；
  M1 定位=生成器同构保守口径（价值下限），真实储能价值区间 [M1, M0x]
- **Q3 指标口径（sum_10 修正）**: 波动/峰值指标用主时段 0-2399（Closure
  边界 2399→2400 跳变污染 max ramp：东区全时段 385 vs 主时段 195）；
  结算段（2400-2405）禁充与终态严格（SOC(2406)=Init）均免费（X7/X8）→
  干净口径采纳；斜坡 120 免费（D 区爬坡 258→189，成本面平坦性最终实证）
  vs E/F 有成本（3.6-10.5%）；规则化代价逐时版 0.78-2.93%（聚合高估一半）；
  MPC σ 区分度实证（场景期望成本随 σ +0.8%——波动有真实成本，确定性等价
  无区分度是方法局限）
- **M3_final 主口径（sum_10 §11 定稿）**: 论文主模型 = M1（时段状态机）
  + 结算段禁充 + 终态严格 SOC(2406)=Init + **生成器量级斜坡**（各区实测
  |ΔPc|max/|ΔPd|max：A 66/45、B 59/40、C 56/38、D 121/182、E 155/167、
  F 145/163——全免费；120 统一值对 E/F 过紧造成"斜坡有成本"假象，已修正）；
  主时段指标（ramp_main/peak_main/std_main）；M0x（互斥物理上界，gap 验证
  逐位稳定 222 万）为区间上界；M0（无互斥）含传送带假象不得引用；
  step3.0 solve_region 已 deprecated（口径对齐 solve_region_timed）

## 工程纪律
- 环境: mathorcup（D:\Anaconda\envs\mathorcup\python.exe）；pandas 3.0.2
- 命名: step{N}.{M}{suffix}_{desc}.py；产物落 output/，图落 figures/
- 论文每个数字可追溯到唯一文件
- 评估器口径: 基线/对照用模板消纳（U=c_r(h)·D）；B9 充电口径选项（include_baseline_charge）
  用于与题目基线绝对数字对照；相对比较同口径稳健
