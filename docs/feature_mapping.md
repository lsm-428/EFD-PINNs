# 输入/输出特征映射详表（含像素3D结构参与）

**目的**
- 将器件结构、材料、工况与实验量系统化映射到 62 维输入；将模型输出 24 维指标定义为可验收的工程量，并提供归一化公式、单位、典型范围与校验规则。
 - 接入 `generate_pyvista_3d.py` 的像素 3D 结构，保证几何/材料/驱动的物理一致性与可解释性。

**输入 62 维布局（索引、符号、单位、归一化）**
- 时空与电压（0–5）
  - `0:X_norm` = `(X - X0)/Lx`（无量纲，X0 为参考点，Lx 为像素长度）
  - `1:Y_norm` = `(Y - Y0)/Ly`
  - `2:Z_norm` = `(Z - Z0)/Lz`
  - `3:T_norm` = `t/Tmax`（0–1）
  - `4:T_phase` = `2πft`（rad，或相位编码）
  - `5:V_norm` = `V/Vmax`（无量纲）
- 几何结构（6–17）
  - `6:pixel_length (µm)` → 归一化 `L/L_ref`
  - `7:pixel_width (µm)` → `W/W_ref`
  - `8:pixel_height (µm)` → `H/H_ref`
  - `9:corner_radius (µm)` → `Rc/Rc_ref`
  - `10:boundary_distance_x (µm)` → `d_x/d_ref`
  - `11:boundary_distance_y (µm)` → `d_y/d_ref`
  - `12:curvature_hint (1/µm)` → 直接数值或归一化 `κ/κ_ref`
  - `13:topology_id` → one-hot 或整数索引归一化到 [0,1]
  - `14–17`：可用作腔体开口形状、壁厚、引线位置等结构编码，归一化到 [0,1]
- 材料与界面（18–27）
  - `18:dielectric_thickness_d (µm)` → `d/d_ref`
  - `19:epsilon_r` → `ε_r/ε_ref`（建议 ε_ref=3）
  - `20:leakage_coeff (A·m⁻²·V⁻¹)` → 对数缩放后归一化
  - `21:surface_tension_gamma (mN/m)` → `γ/γ_ref`
  - `22:roughness_ra (nm)` → `Ra/Ra_ref`
  - `23:theta0 (deg)` → `θ0/180`
  - `24:pinning_strength` → [0,1]
  - `25–27`：界面改性参数（涂层厚度、能垒、亲疏水标量），归一化到 [0,1]
- 电场与介质响应（双相）（28–35）
  - `28:E_z (归一化)`、`29:E_magnitude (归一化)`、`30:field_gradient (归一化)`
  - `31:V_effective = V/Vmax`
  - `32:charge_relaxation_norm ≈ (ε_r ε0 / σ) / τ_ref`（极性液体电荷松弛时间归一化，建议 `τ_ref=1ms`）
  - `33:ink_permittivity_norm = ε_r(ink)/ε_ref`（油墨相对介电常数的弱影响占位）
  - `34–35`：电极与波形编码（保留位）
- 流体动力学（36–45）
  - `36:viscosity (mPa·s)` → `μ/μ_ref`
  - `37:density (kg/m³)` → `ρ/ρ_ref`
  - `38:wetting_coeff` → [0,1]
  - `39:interface_tension (mN/m)` → `σ/σ_ref`
  - `40:contact_angle_min (deg)` → `/180`
  - `41:contact_angle_max (deg)` → `/180`
  - `42–45`：界面摩擦、滑移长度、Marangoni 数等编码，归一化到 [0,1]
- 时间动态（46–51）
  - `46:dt (s)` → `Δt/Δt_ref`
  - `47:history_loss_ema` → 历史损失指数平均（0–1）
  - `48:history_mae_ema` → 历史 MAE 指标（0–1）
  - `49:periodicity_hint` → [0,1]
  - `50:prev_theta (deg)` → `/180`
  - `51:prev_h (µm)` → `/H_ref`
- 电润湿特异（双相）（52–61）
  - `52:young_lippmann_coeff` ≈ `ε0ε_r/(γd)`（归一化到 [0,1]）
  - `53:contact_line_hysteresis` → [0,1]
  - `54:pinning_threshold` → [0,1]
  - `55:dielectric_leakage_rate` → 对数缩放归一化
  - `56:stability_margin` → [0,1]
  - `57:optical_contrast_target` → [0,1]
  - `58:energy_budget` → 归一化能耗指标
  - `59:polar_conductivity_norm` → 极性液体导电归一化（对数尺度）
  - `60:ink_volume_fraction` → 油墨体积分数（体积守恒软约束的输入刻画）
  - `61`：保留扩展位

典型范围建议（像素）：`L,W=50–200 µm`、`H=5–50 µm`、`d=0.5–5 µm`、`ε_r=2–4`、`γ=20–50 mN/m`、`θ0=100–120°`、`f=10 Hz–100 kHz`。

**归一化与单位规则**
- 长度类使用相对参考值归一化，电压与电场按最大工作值归一化；频率使用对数缩放以覆盖多数量级。
- 角度统一归一化到 `/180`；耗能指标按目标范围归一化到 [0,1]。
- 建议在数据管线中执行单位校验与范围裁剪，超界值记录到训练元数据。

**输出 24 维指标（定义、单位、验收）**
- 物理场与电场（0–7）
  - `0:p (Pa)`、`1:u (m/s)`、`2:v (m/s)`、`3:w (m/s)`、`4:vorticity_z (1/s)`
  - `5:phi (V)`、`6:E_x (V/m)`、`7:E_z (V/m)`
- 界面与接触线（8–19）
  - `8:h (µm)`、`9:kappa (1/µm)`、`10:slope (rad)`
  - `11:theta_local (deg)`、`12:theta_eq (deg)`
  - `13:r_cl (µm)`、`14:cl_shape_param`、`15:cl_curvature`、`16:v_cl (µm/s)`、`17:adv_angle (deg)`、`18:rec_angle (deg)`、`19:contact_line_energy (J/m)`
- 工程性能（20–23）
  - `20:tau_response (ms)`、`21:switch_speed (ms)`
  - `22:stability_index (0–1)`、`23:energy_efficiency (0–1)`

验收建议（示例）：
- Young–Lippmann 一致性：`|cosθ - cosθ0 + (ε0ε_rV²)/(2γd)|` 的均值 ≤ `1e-2`
- 频率响应偏差：目标频段的幅/相误差均值 ≤ `5%`
- 界面稳定性指数 ≥ `0.8`，能量效率 ≥ `0.7`

**数据记录与训练元数据**
- 输入单位检查：记录校验通过/失败计数、裁剪比例、异常值处理策略到 `training_history.json` 的元数据段
- 场景标识：记录 `scenario_id`（dc_step、ac_sweep、hysteresis、thermal）、关键权重策略（固定/自适应/阶段式）
- 约束诊断：每阶段与最终保存 `constraint_diagnostics_*.json`；图表生成于 `consistency_data/`

**示例数据记录（单条）**
- 结构与材料
  - `pixel_length=120 µm`、`pixel_width=120 µm`、`pixel_height=20 µm`、`corner_radius=5 µm`
  - `dielectric_thickness_d=1.0 µm`、`epsilon_r=3.0`、`surface_tension_gamma=35 mN/m`、`theta0=110°`
- 工况与电场
  - `V=20 V`、`frequency_hz=1000 Hz`、`phi_init=0 V`、`E_x=0`、`E_z=0`
- 映射
  - 归一化到上述输入索引；角度除以 180，频率以对数缩放归一化到 [0,1]
  - 3D参与：
    - 厚度分数：`polar_thickness_norm=17/(17+3)`、`ink_thickness_norm=3/(17+3)`
    - 极性液体导电：`polar_conductivity_norm = log10(σ)/log10区间` 映射到 [0,1]
    - 电荷松弛：`charge_relaxation_norm = (ε_r ε0 / σ)/τ_ref`（建议 `τ_ref=1ms`）
    - 油墨介电：`ink_permittivity_norm = ε_r(ink)/ε_ref`（弱影响占位）
    - 体积分数：`ink_volume_fraction = ink_thickness / (ink_thickness + polar_thickness)`

**实现参考**
- 输入层实现与最终索引以 `ewp_pinn_input_layer.py` 为准；本映射为工程规范，建议在数据接口中进行字段→索引绑定与单位校验。
 - 3D→输入：`ewp_parameter_mapper.py:create_feature_dict_from_3d`、`create_stage3_batch_from_3d`；训练脚本通过 `--use_3d_mapping` 直接生成 `dataset.npz`。
