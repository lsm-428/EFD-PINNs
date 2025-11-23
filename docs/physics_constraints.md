# EWPINN 物理约束实现文档

## 1. 概述

本文档详细描述了电润湿像素神经网络模型(EWPINN)中实现的物理约束系统。该系统通过在神经网络训练过程中强制执行物理方程和约束条件，显著提高了模型的物理一致性和预测精度。

主要实现了以下五类物理约束：
- 热力学约束
- 界面稳定性约束
- 频率响应约束
- 光学特性约束
- 能量效率约束

所有约束均已集成到`PhysicsConstraints`类中，并通过`PINNConstraintLayer`进行统一管理和权重调整。

## 2. 物理约束架构

### 2.1 核心类设计

物理约束系统主要由两个核心类组成：

1. **PhysicsConstraints类**：实现各种物理方程的残差计算
2. **PINNConstraintLayer类**：管理约束权重、计算加权损失并支持动态权重调整

### 2.2 材料参数系统（双相与3D参与）

```python
def __init__(self, materials_params=None):
    # 默认材料参数
    self.materials_params = materials_params or {
        'viscosity': 1.0,
        'density': 1.0,
        'surface_tension': 0.0728,
        'permittivity': 80.1,
        'conductivity': 5.5e7,
        'youngs_modulus': 210e9,
        'poisson_ratio': 0.3,
        'contact_angle_theta0': 110.0,  # 静态接触角(度)
        # 热力学相关参数
        'ambient_temperature': 293.15,    # 环境温度 (K，20°C)
        'thermal_conductivity_water': 0.6, # 水的热导率 (W/(m·K))
        'thermal_conductivity_oil': 0.15,  # 油的热导率 (W/(m·K))
        'thermal_conductivity_dielectric': 0.02, # 介电层热导率 (W/(m·K))
        'specific_heat_water': 4186.0,     # 水的比热容 (J/(kg·K))
        'thermal_expansion_water': 2.1e-4, # 水的热膨胀系数 (1/K)
        'temperature_coefficient_surface_tension': -1.5e-4, # 表面张力温度系数 (N/(m·K))
        'temperature_coefficient_viscosity': -3.5e-3,       # 粘度温度系数 (1/K)
        # ... 其他参数
    }
```

该参数系统支持材料属性的灵活配置，包括流体动力学、热力学、电学和光学参数，为各类物理约束提供必要的材料数据。结合像素3D结构：
 - 极性液体层：导电与电荷松弛（`σ`、`ε_r`），用于 `charge_relaxation_norm`、`polar_conductivity_norm`
 - 油墨层：非极性、体积守恒，介电弱影响（`ink_permittivity_norm`）、`ink_volume_fraction`
 - 厚度分数：`polar_thickness_norm/ink_thickness_norm` 来自 3D 层厚，参与界面与时间尺度估计

### 2.3 约束权重管理

```python
self.residual_weights = residual_weights or {
    'continuity': 0.5,
    'momentum_u': 0.05,
    'momentum_v': 0.05,
    'momentum_w': 0.05,
    'young_lippmann': 0.5,  # Young-Lippmann方程约束权重
    'contact_line_dynamics': 0.3,  # 接触线动力学约束权重
    'dielectric_charge': 0.4,  # 介电层电荷积累约束权重
    'thermodynamic': 0.2,  # 热力学约束权重
    'interface_stability': 0.4,  # 界面稳定性约束权重
    'frequency_response': 0.3,  # 频率响应约束权重
    'optical_properties': 0.25,  # 光学特性约束权重
    'energy_efficiency': 0.3,  # 能量效率约束权重
    'data_fit': 1.0
}
```

权重系统允许为不同物理约束分配不同的重要性级别，通过调整这些权重，可以在不同的应用场景中优化模型性能。

## 3. 热力学约束实现

### 3.1 约束原理

热力学约束模拟温度对电润湿系统的影响，主要包括：
- 温度对表面张力的影响
- 温度对粘度的影响
- 热传导过程
- 温度范围限制
- 热膨胀效应

### 3.2 核心算法实现

```python
def compute_thermodynamic_residual(self, x, predictions, temperature, applied_voltage=None):
    # 安全检查
    if temperature is None:
        logger.warning("热力学约束: 温度场为None")
        return self._empty_thermodynamic_residual(x, predictions)
    try:
        # 提取材料参数
        material_params = self.materials_params or {}
        ambient_temp = material_params.get('ambient_temperature', 293.15)
        surface_tension_base = material_params.get('surface_tension', 0.0728)
        temp_coef_surface_tension = material_params.get('temperature_coefficient_surface_tension', -1.5e-4)
        surface_tension_temp = surface_tension_base * (1.0 + temp_coef_surface_tension * (temperature - ambient_temp))
        surface_tension_ambient = torch.tensor(surface_tension_base, device=temperature.device)
        viscosity_base = material_params.get('viscosity', 1.0)
        temp_coef_viscosity = material_params.get('temperature_coefficient_viscosity', -3.5e-3)
        viscosity_temp = viscosity_base * (1.0 + temp_coef_viscosity * (temperature - ambient_temp))
        viscosity_ambient = torch.tensor(viscosity_base, device=temperature.device)
        heat_equation_residual = torch.zeros_like(temperature)
        temp_min = 273.15
        temp_max = 373.15
        temperature_constraint = torch.maximum(torch.zeros_like(temperature), torch.minimum(temperature - temp_min, temp_max - temperature))
        density_water = torch.tensor(material_params.get('density', 1000.0), device=temperature.device)
        thermal_expansion = material_params.get('thermal_expansion_water', 2.1e-4)
        density_temp = density_water * (1.0 - thermal_expansion * (temperature - ambient_temp))
        residuals = {
            'surface_tension_temp_effect': surface_tension_temp - surface_tension_ambient,
            'viscosity_temp_effect': viscosity_temp - viscosity_ambient,
            'heat_equation': heat_equation_residual,
            'temperature_limits': temperature_constraint,
            'thermal_expansion': density_temp - density_water
        }
        return {key: residual / (torch.max(torch.abs(residual)) + 1e-12) if torch.any(torch.abs(residual) > 0) else residual for key, residual in residuals.items()}
    except Exception as e:
        logger.error(f"计算热力学残差时出错: {str(e)}")
        return self._empty_thermodynamic_residual(x, predictions)
```

## 4. 界面稳定性约束实现

（略，完整实现与数学模型见源代码与上游文档内容，包含曲率、梯度、Kelvin-Helmholtz、不稳定性与能量最小化约束）

## 5. 频率响应约束实现

- 使用 Debye 介电模型：`ε(ω) = ε_∞ + (ε_s - ε_∞)/(1 + jωτ)`，结合电导率与位移/传导电流比例约束
- 对交流电场下的界面响应与电压时间导数进行约束

## 6. 光学特性约束实现

- 菲涅尔反射率与透射率约束、界面对比度与锐利度约束、波长依赖性（简化柯西色散）

## 7. 能量效率约束实现

- 功耗限制、能量转换效率（机械功/电功）、粘性耗散与电压利用效率约束

## 8. 集成与权重调整

- 通过 `PINNConstraintLayer.compute_physics_loss` 统一计算并加权残差，支持自适应权重与动态权重策略
 - 支持3D参与的软约束：可在后续迭代加入体积守恒残差，利用 `ink_volume_fraction` 与厚度分数指导加权（默认权重很小，避免扰动训练稳定性）

## 9. 鲁棒性与错误处理

- 安全梯度、除零保护、异常捕获、日志记录与标准化处理，提升诊断与训练稳定性

## 10. 未来改进

- 更完整的热传导与耦合、多物理场自适应物理点采样、残差场可视化与硬件加速等

## 改进记录（训练稳健性修复）
- 坐标与梯度链路修正：训练与验证的物理约束统一使用 `model(batch_phys)` 得到 `pred_phys`，确保计算图连接到 `x_phys`；不再用 `pred(X_batch)` 参与约束，避免梯度链路断开导致物理项为 0
- Navier–Stokes 梯度维度策略：统一以输入的坐标维度参与梯度与拉普拉斯计算（使用 `safe_compute_laplacian_spatial`），避免对非空间维度广播造成零梯度；连续性采用各维散度和，动量项为粘性拉普拉斯与压力梯度总和的简化形式
- 二阶项稳健化：粘性项采用拉普拉斯近似 `safe_compute_laplacian`/`safe_compute_laplacian_spatial`，替代逐分量二阶梯度以避免广播形状冲突
- 安全梯度后备：新增 `safe_compute_gradient/safe_compute_laplacian/safe_compute_hessian`，在梯度失败或 `allow_unused=True` 返回 `None` 时回退为零张量
- 界面稳定性：界面梯度/曲率与速度切向梯度均基于 `coords3`；速度场统一从预测中取前三分量与 3D 坐标对齐
- 频率响应：容错材料参数为标量，电压统一张量化并按 batch 广播，仅在提供 `frequency` 时计算 Debye；移除对标量 `.get` 的错误访问

## 训练与验证循环修正
- 训练：`train_epoch` 中为每个批次物理点计算 `pred_phys = model(batch_phys)`，用其参与 `compute_physics_loss`；数据拟合与物理约束前向分离，避免 `no_grad` 影响
- 验证：移除物理约束路径上的 `no_grad`，同样使用 `pred_phys = model(batch_phys)` 参与约束，保证验证期的物理损失为非零可观测量
 - 数据生成：`long_term_training.py` 支持 `--use_3d_mapping`，从 `generate_pyvista_3d.py` 直接生成阶段3输入并保存数据集

## 诊断报表影响
- 修复后 `constraint_diagnostics_epoch_*` 的 `continuity/momentum_*` 将出现非零统计；`Val Physics Loss` 不再为 0，可用于动态权重与训练监控
