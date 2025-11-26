# 物理约束 API

## PhysicsConstraints 类

**文件**: `ewp_pinn_model.py` 和 `ewp_pinn_physics.py`

### 类定义
```python
class PhysicsConstraints:
    """物理约束系统 - 确保模型遵循物理规律"""
```

### 构造函数
```python
def __init__(self, materials_params=None):
    """初始化材料参数和边界条件"""
```

**参数说明**:
- `materials_params` (dict, 可选): 材料参数字典

### 主要方法

#### compute_navier_stokes_residual
```python
def compute_navier_stokes_residual(self, x, predictions):
    """计算Navier-Stokes方程残差"""
```

**参数**:
- `x` (torch.Tensor): 空间坐标张量
- `predictions` (dict): 模型预测结果字典

**返回值**:
- `torch.Tensor`: Navier-Stokes方程残差

#### compute_electrostatic_constraints
```python
def compute_electrostatic_constraints(self, x, predictions):
    """计算静电学约束"""
```

**参数**:
- `x` (torch.Tensor): 空间坐标张量
- `predictions` (dict): 模型预测结果字典

**返回值**:
- `torch.Tensor`: 静电学约束残差

#### compute_interface_constraints
```python
def compute_interface_constraints(self, x, predictions):
    """计算界面约束"""
```

**参数**:
- `x` (torch.Tensor): 空间坐标张量
- `predictions` (dict): 模型预测结果字典

**返回值**:
- `torch.Tensor`: 界面约束残差

#### compute_thermal_constraints
```python
def compute_thermal_constraints(self, x, predictions):
    """计算热力学约束"""
```

**参数**:
- `x` (torch.Tensor): 空间坐标张量
- `predictions` (dict): 模型预测结果字典

**返回值**:
- `torch.Tensor`: 热力学约束残差

#### compute_total_residual
```python
def compute_total_residual(self, x, predictions):
    """计算总物理残差"""
```

**参数**:
- `x` (torch.Tensor): 空间坐标张量
- `predictions` (dict): 模型预测结果字典

**返回值**:
- `torch.Tensor`: 总物理残差

#### set_constraint_weights
```python
def set_constraint_weights(self, weights_dict):
    """设置约束权重"""
```

**参数**:
- `weights_dict` (dict): 权重字典

#### validate_physical_bounds
```python
def validate_physical_bounds(self, predictions):
    """验证物理边界条件"""
```

**参数**:
- `predictions` (dict): 模型预测结果字典

**返回值**:
- `dict`: 边界验证结果

## 物理方程实现

### Navier-Stokes 方程
```python
# 连续性方程
continuity_residual = du_dx + dv_dy + dw_dz

# 动量方程
x_momentum = u*du_dx + v*du_dy + w*du_dz + dp_dx - nu*(d2u_dx2 + d2u_dy2 + d2u_dz2)
y_momentum = u*dv_dx + v*dv_dy + w*dv_dz + dp_dy - nu*(d2v_dx2 + d2v_dy2 + d2v_dz2)
z_momentum = u*dw_dx + v*dw_dy + w*dw_dz + dp_dz - nu*(d2w_dx2 + d2w_dy2 + d2w_dz2)
```

### 静电学方程
```python
# 泊松方程
electrostatic_residual = d2phi_dx2 + d2phi_dy2 + d2phi_dz2 + rho/epsilon

# 电场强度
E_x = -dphi_dx
E_y = -dphi_dy
E_z = -dphi_dz
```

### 界面方程
```python
# 界面形状方程
interface_residual = kappa * (d2h_dx2 + d2h_dy2) - (p_liquid - p_gas) / sigma

# 接触线动力学
contact_line_residual = u_slip - mu * (theta - theta_eq)
```

### 热力学方程
```python
# 能量方程
energy_residual = u*dT_dx + v*dT_dy + w*dT_dz - alpha*(d2T_dx2 + d2T_dy2 + d2T_dz2)
```

## 使用示例

### 基础用法
```python
import torch
from ewp_pinn_physics import PhysicsConstraints

# 创建物理约束实例
physics = PhysicsConstraints()

# 设置约束权重
weights = {
    'navier_stokes': 1.0,
    'electrostatic': 0.8,
    'interface': 0.5,
    'thermal': 0.3
}
physics.set_constraint_weights(weights)

# 模拟输入数据
x = torch.randn(100, 3)  # 100个空间点，3维坐标
predictions = {
    'velocity': torch.randn(100, 3),
    'pressure': torch.randn(100, 1),
    'electric_potential': torch.randn(100, 1),
    'temperature': torch.randn(100, 1),
    'interface_height': torch.randn(100, 1)
}

# 计算物理残差
residuals = physics.compute_total_residual(x, predictions)
print(f"总物理残差: {residuals.mean().item()}")

# 验证物理边界
bounds_check = physics.validate_physical_bounds(predictions)
print(f"边界验证: {bounds_check}")
```

### 高级用法
```python
# 计算特定方程的残差
ns_residual = physics.compute_navier_stokes_residual(x, predictions)
electro_residual = physics.compute_electrostatic_constraints(x, predictions)
interface_residual = physics.compute_interface_constraints(x, predictions)

print(f"NS残差: {ns_residual.mean().item()}")
print(f"静电残差: {electro_residual.mean().item()}")
print(f"界面残差: {interface_residual.mean().item()}")

# 自定义材料参数
materials = {
    'density': 1000.0,      # kg/m³
    'viscosity': 0.001,     # Pa·s
    'surface_tension': 0.072, # N/m
    'dielectric_constant': 80.0,
    'thermal_conductivity': 0.6  # W/(m·K)
}

custom_physics = PhysicsConstraints(materials_params=materials)
```

### 训练集成示例
```python
# 在训练循环中使用物理约束
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 模型预测
        predictions = model(batch_x)
        
        # 计算数据损失
        data_loss = criterion(predictions, batch_y)
        
        # 计算物理损失
        physics_loss = physics.compute_total_residual(batch_x, predictions)
        
        # 总损失
        total_loss = data_loss + physics_weight * physics_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
```

## 配置参数

### 默认材料参数
```python
DEFAULT_MATERIALS = {
    'water_density': 997.0,           # kg/m³
    'water_viscosity': 0.00089,       # Pa·s
    'air_density': 1.225,             # kg/m³
    'air_viscosity': 1.81e-5,         # Pa·s
    'surface_tension': 0.072,         # N/m
    'dielectric_constant_water': 80.0,
    'dielectric_constant_air': 1.0,
    'thermal_conductivity_water': 0.6, # W/(m·K)
    'specific_heat_water': 4186.0     # J/(kg·K)
}
```

### 约束权重配置
```python
DEFAULT_WEIGHTS = {
    'continuity': 1.0,
    'momentum_x': 1.0,
    'momentum_y': 1.0,
    'momentum_z': 1.0,
    'electrostatic': 0.8,
    'interface_shape': 0.5,
    'contact_line': 0.3,
    'energy': 0.2
}
```

## 最佳实践

1. **权重调整**: 根据问题重要性调整不同物理约束的权重
2. **边界验证**: 训练前后验证预测结果是否满足物理边界
3. **残差监控**: 监控各物理方程的残差变化趋势
4. **材料参数**: 使用真实材料参数提高模拟准确性

## 故障排除

- **残差过大**: 检查输入数据范围和单位一致性
- **梯度爆炸**: 降低物理权重或使用梯度裁剪
- **收敛困难**: 逐步增加物理约束权重
- **边界违反**: 加强边界条件约束