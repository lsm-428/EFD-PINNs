# 电润湿像素设计指南（EWPINN 专用）

**目标**
- 将电润湿像素的关键工程参数系统化映射到模型的 62 维输入与 24 维输出指标，提供可直接套用的配置模板与试验方案，支撑专业像素设计与验证。

**输入 62 维特征映射（建议分组）**
- 时空与电压
  - 归一化坐标 `X_norm,Y_norm,Z_norm`，时间编码 `T_norm,T_phase`，施加电压 `V_norm`
- 几何结构
  - 像素腔体长度/宽度/高度，边界到像素中心距离，曲率、角隅特征、拓扑标签
- 材料与界面
  - 介电层材质与厚度、油水界面张力、疏水涂层参数、表面粗糙度、接触线特征编码
- 电场与介电
  - 电势与场强初值、介电常数、击穿阈值、频率参数占位
- 流体动力学
  - 粘度、密度、润湿性系数、界面张力、接触角初值及范围、界面曲率先验
- 时间动态
  - 历史状态摘要、步长编码、周期性与频率特征
- 电润湿特异参数
  - Young–Lippmann 参数族、接触线钉扎强度、滞后系数、介电层泄漏参数

**输出 24 维指标映射（建议分组）**
- 物理场与电场
  - 压力场、速度场 `u,v,w`、涡量、电势 `phi`、电场强度 `E_x,E_z`
- 界面与接触线
  - 高度场 `h`、曲率 `kappa`、斜率、局部/平衡接触角、接触线半径、形状与速度
- 工程性能
  - 响应时间常数、开关速度、稳定性指标、能量效率

**典型像素结构参数建议**
- 像素腔体：长 `50–200 µm`，宽 `50–200 µm`，高 `5–50 µm`
- 介电层厚度 `d`：`0.5–5 µm`（匹配工作电压与击穿裕度）
- 介电常数 `ε_r`：`2–4`（常见聚合物）
- 表面张力 `γ`：`20–50 mN/m`（体系相关）
- 疏水涂层：接触角初值 `θ0`：`100–120°`
- 电压范围：直流 `0–30 V`，交流频扫 `10 Hz–100 kHz`

**频率响应试验设计**
- 频点对数分布覆盖低/中/高频段；记录 `φ, E_x, h, θ` 的幅值与相位响应；对比 Young–Lippmann 预测与实际接触角变化。

**配置模板**
- 直流阶跃响应：`docs/config_samples/dc_step_config.json`
- 交流频扫：`docs/config_samples/ac_sweep_config.json`
- 接触线滞后验证：`docs/config_samples/contact_line_hysteresis_config.json`
- 温升工况：`docs/config_samples/thermal_rise_config.json`

**残差权重建议与训练时长估算**
- 直流阶跃：`young_lippmann=0.8`，`contact_line_dynamics=0.4`，`interface_stability=0.5`；总轮次 `1–2e4`
- 交流频扫：`frequency_response=0.7`，`dielectric_charge=0.4`，`energy_efficiency=0.3`；总轮次 `2–3e4`
- 滞后验证：`contact_line_dynamics=0.8`，`interface_stability=0.6`；总轮次 `2e4`
- 温升工况：`thermodynamic=0.7`，`energy_efficiency=0.4`；总轮次 `2–3e4`

**使用说明**
- 将所选模板复制到 `config/` 并按项目数据源更新数值范围与阶段轮次；配合 `--dynamic_weight` 以在中后期自动平衡数据与物理一致性。

