# 物理约束 API

**最后更新**: 2026-02-04
**版本**: v4.5

## PhysicsConstraints

**类路径**: `src.physics.constraints.PhysicsConstraints`

物理方程的"唯一真理源"，实现了 Navier-Stokes、VOF、连续性方程及电润湿驱动力。

### v4.5 核心特性
- **界面锐化损失**: `L_sharp = λ·φ(1-φ)` 防止界面扩散
- **体积守恒**: 使用蒙特卡洛采样验证体积守恒

### v4.5 损失权重

| 损失项 | v4.5 权重 | 说明 |
|--------|-----------|------|
| `interface` | 500.0 | 界面数据拟合 |
| `ic` | 300.0 | 初始条件 |
| `bc` | 80.0 | 边界条件 |
| `continuity` | 0.5 | 连续性方程 |
| `vof` | 0.5 | VOF 输运方程 |
| `ns` | 0.1 | Navier-Stokes 方程 |
| `surface_tension` | 0.01 | 表面张力 |
| `sharpening` | 1.0 | 界面锐化 |

### 主要方法

#### `compute_core_residuals`
统一计算所有核心物理残差。

```python
def compute_core_residuals(
    self, 
    x: torch.Tensor, 
    predictions: torch.Tensor, 
    model: nn.Module = None
) -> Dict[str, torch.Tensor]:
    """
    参数:
    - x: (N, 6) 输入坐标 (x,y,z, V_from, V_to, t_since)
    - predictions: (N, 5) 模型输出 (u,v,w,p,phi)
    
    返回字典包含:
    - 'continuity': ∇·u
    - 'momentum_u/v/w': N-S 残差 (含 F_ew)
    - 'vof': ∂φ/∂t + u·∇φ
    - 'surface_tension': 表面张力项
    - 'sharpening': 界面锐化损失 (v4.5 新增)
    - 'volume_conservation': 体积守恒残差 (v4.5)
    """
```

### v4.5 物理模型

#### 1. 界面锐化损失 (v4.5)
防止 VOF 界面扩散，保持油墨与极性液体的锐利分界：

$$
L_{sharp} = \lambda_{sharp} \cdot \phi(1-\phi)
$$

- **作用**: 惩罚中间值 $\phi \approx 0.5$ 的区域
- **权重**: `sharpening = 0.1`
- **效果**: 保持清晰的相界面

#### 2. 电润湿驱动力 (Electrowetting Force)
显式引入电润湿力作为体积力项：

$$
\mathbf{F}_{ew} = \frac{\varepsilon_0 \varepsilon_r V^2}{2d} \cdot \phi \cdot 4\phi(1-\phi) \cdot e^{-z/2h} \cdot \mathbf{n}
$$

详细参数说明请参见[物理理论与器件规格指南](../guides/physics_and_device_guide.md#physics-parameters)。

- **作用区域**: 仅在油墨相 ($\phi \approx 1$) 和界面处 ($\phi(1-\phi)$) 有效。
- **方向 $\mathbf{n}$**: 指向油墨外侧，驱动收缩。

#### 3. Navier-Stokes 方程

$$
\rho(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{F}_{st} + \mathbf{F}_{ew}
$$

- **$\mathbf{F}_{st}$**: CSF 表面张力模型。
- **$\rho, \mu$**: 基于 $\phi$ 的混合密度和粘度。

#### 4. 体积守恒

$$
V(t) = \int_{\Omega} \phi(\mathbf{x}, t) d\Omega = V_0
$$

- **验证方法**: 蒙特卡洛采样
- **目标**: 体积误差 < 1%

#### 5. 连续性与 VOF
- **连续性**: $\nabla \cdot \mathbf{u} = 0$
- **VOF**: $\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0$

---

## PhysicsLoss

**类路径**: `src.models.pinn_two_phase.PhysicsLoss`

训练损失计算的适配层，负责：
1. 调用 `PhysicsConstraints` 获取原始残差。
2. 对残差进行数值保护 (NaN/Inf 清理)。
3. 应用 log1p 缩放和动态加权。

### v4.5 损失权重

| 损失项 | v4.5 权重 | 说明 |
|--------|-----------|------|
| `interface` | 500.0 | 界面数据拟合 |
| `ic` | 300.0 | 初始条件 |
| `bc` | 80.0 | 边界条件 |
| `continuity` | 0.5 | 连续性方程 |
| `vof` | 0.5 | VOF 输运方程 |
| `ns` | 0.1 | Navier-Stokes 方程 |
| `surface_tension` | 0.01 | 表面张力 |
| `sharpening` | 1.0 | 界面锐化 |

### 主要方法

#### `compute_total_loss`
```python
def compute_total_loss(
    self, 
    model: nn.Module, 
    x_phys: torch.Tensor, 
    weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict]:
    """
    计算总物理损失。
    
    Args:
        model: PINN 模型
        x_phys: (N, 6) 训练点坐标
        weights: 损失权重字典
    
    Returns:
        - total_loss: 加权求和后的标量损失
        - details: 各分项损失详情
    """
```

---

## v4.5 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83.4%** | 基于校准物理参数 |
| 体积误差 | **<1%** | VOF 输运方程体积守恒 |
| 界面锐化 | **启用** | φ(1-φ) 正则化 |
