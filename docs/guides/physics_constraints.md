# 物理约束详解

## 约束类型概览

EFD3D项目实现了9类核心物理约束，确保模型预测的物理合理性：

### 1. Navier–Stokes 约束
**作用**：确保流体运动满足动量守恒
```python
# 核心实现位置：ewp_pinn_physics.py
class NavierStokesConstraint:
    def compute_constraint(self, velocity, pressure, density, viscosity):
        """计算Navier-Stokes方程残差"""
        # 连续性方程：∇·u = 0
        continuity = torch.autograd.grad(velocity, ...)
        
        # 动量方程：ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f
        momentum = self.calculate_momentum_residual(...)
        
        return continuity + momentum
```

**应用场景**：电润湿像素中的流体动力学模拟

### 2. Young–Lippmann 约束
**作用**：描述接触角与电压的关系
```python
class YoungLippmannConstraint:
    def compute_contact_angle(self, voltage, surface_tension, dielectric_constant):
        """Young-Lippmann方程：cosθ = cosθ₀ + (ε₀εᵣV²)/(2γd)"""
        cos_theta_0 = torch.cos(self.initial_contact_angle)
        electric_term = (self.epsilon_0 * dielectric_constant * voltage**2) / \
                       (2 * surface_tension * self.dielectric_thickness)
        return cos_theta_0 + electric_term
```

**物理意义**：控制电润湿效应的核心方程

### 3. 质量守恒约束
**作用**：确保流体质量在运动过程中守恒
```python
class MassConservationConstraint:
    def compute_mass_balance(self, density, velocity, volume):
        """计算质量守恒残差"""
        # 质量流量：ρ·u·A
        mass_flow = density * velocity * self.cross_sectional_area
        
        # 质量变化率：∂(ρV)/∂t
        mass_change = torch.autograd.grad(density * volume, ...)
        
        return mass_flow - mass_change
```

### 4. 能量守恒约束
**作用**：确保能量在系统中的守恒
```python
class EnergyConservationConstraint:
    def compute_energy_balance(self, temperature, heat_flux, specific_heat):
        """计算能量守恒残差"""
        # 热传导：-k∇T
        conduction = -self.thermal_conductivity * torch.autograd.grad(temperature, ...)
        
        # 能量变化率：ρc_p∂T/∂t
        energy_change = self.density * specific_heat * torch.autograd.grad(temperature, ...)
        
        return conduction - energy_change + heat_flux
```

### 5. 电荷守恒约束
**作用**：确保电荷在电润湿系统中的守恒
```python
class ChargeConservationConstraint:
    def compute_charge_balance(self, electric_field, charge_density, permittivity):
        """计算电荷守恒残差"""
        # 高斯定律：∇·(εE) = ρ
        gauss_law = torch.autograd.grad(permittivity * electric_field, ...) - charge_density
        
        # 电流连续性：∇·J + ∂ρ/∂t = 0
        current_continuity = self.calculate_current_continuity(...)
        
        return gauss_law + current_continuity
```

## 约束权重策略

### 动态权重调整
```python
class DynamicConstraintWeight:
    def adjust_weights(self, constraint_residuals, training_stage):
        """根据训练阶段动态调整约束权重"""
        if training_stage == "initial":
            # 初期：数据拟合为主
            weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  # 数据拟合权重较高
        elif training_stage == "refinement":
            # 精炼阶段：物理约束权重增加
            weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]  # 物理约束权重均衡
        else:
            # 最终阶段：物理约束主导
            weights = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]
        
        return weights
```

### 自适应权重
```python
class AdaptiveConstraintWeight:
    def compute_adaptive_weights(self, constraint_errors):
        """根据约束误差自适应调整权重"""
        # 误差较大的约束获得更高权重
        normalized_errors = constraint_errors / constraint_errors.sum()
        weights = 1.0 / (normalized_errors + 1e-8)  # 防止除零
        return weights / weights.sum()  # 归一化
```

## 约束验证方法

### 1. 物理一致性验证
```python
def validate_physics_consistency(model, validation_data):
    """验证模型预测的物理一致性"""
    predictions = model(validation_data)
    
    # 检查质量守恒
    mass_residual = mass_constraint.compute(predictions)
    mass_violation = torch.abs(mass_residual).mean()
    
    # 检查能量守恒
    energy_residual = energy_constraint.compute(predictions)
    energy_violation = torch.abs(energy_residual).mean()
    
    return {
        'mass_conservation_violation': mass_violation.item(),
        'energy_conservation_violation': energy_violation.item(),
        'overall_consistency': (mass_violation + energy_violation).item()
    }
```

### 2. 边界条件验证
```python
def validate_boundary_conditions(model, boundary_data):
    """验证边界条件的满足程度"""
    boundary_predictions = model(boundary_data)
    
    # Dirichlet边界条件
    dirichlet_error = torch.abs(boundary_predictions - boundary_data.targets).mean()
    
    # Neumann边界条件（梯度边界）
    gradient_predictions = torch.autograd.grad(boundary_predictions, ...)
    neumann_error = torch.abs(gradient_predictions - boundary_data.gradients).mean()
    
    return {
        'dirichlet_error': dirichlet_error.item(),
        'neumann_error': neumann_error.item()
    }
```

## 约束诊断报告

### 生成约束诊断
```python
class ConstraintDiagnosticReport:
    def generate_report(self, model, test_data):
        """生成详细的约束诊断报告"""
        report = {}
        
        # 各约束残差统计
        for constraint_name, constraint in self.constraints.items():
            residuals = constraint.compute(model(test_data))
            report[constraint_name] = {
                'mean_residual': residuals.mean().item(),
                'max_residual': residuals.max().item(),
                'std_residual': residuals.std().item(),
                'violation_ratio': (residuals > self.tolerance).float().mean().item()
            }
        
        # 约束相关性分析
        constraint_correlations = self.analyze_constraint_correlations(report)
        report['correlation_analysis'] = constraint_correlations
        
        return report
```

## 最佳实践

### 1. 约束权重调优
- **初期训练**：数据拟合权重为主，物理约束权重较低
- **中期训练**：逐步增加物理约束权重
- **后期训练**：物理约束主导，确保物理一致性

### 2. 约束组合策略
- **强约束**：必须满足的物理定律（如质量守恒）
- **弱约束**：指导性的物理关系（如经验公式）
- **软约束**：正则化项形式的约束

### 3. 约束验证频率
- **每100步**：快速约束检查
- **每1000步**：详细约束诊断
- **训练结束**：完整约束验证报告

## 故障排除

### 常见问题
1. **约束残差过大**：降低学习率或调整约束权重
2. **训练不稳定**：检查梯度爆炸，使用梯度裁剪
3. **物理不一致**：验证边界条件和初始条件

### 调试工具
```python
# 约束调试工具
def debug_constraints(model, sample_data):
    """调试约束计算的工具函数"""
    with torch.autograd.detect_anomaly():
        predictions = model(sample_data)
        for name, constraint in model.constraints.items():
            residual = constraint(predictions)
            print(f"{name}: mean_residual={residual.mean().item()}")
```

这个详细的物理约束文档为开发者提供了完整的约束实现、验证和调试指南。