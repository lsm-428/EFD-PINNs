# EFD3D 损失函数设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**对应代码**: `src/models/pinn_two_phase.py`, `src/physics/constraints.py`

---

## 1. 设计概述

EFD3D 的损失函数分为三大类：

| 类别 | 损失项 | 目的 |
|------|--------|------|
| **数据损失** | interface_loss, ic_loss, bc_loss | 学习数据映射 |
| **物理损失** | continuity, vof, ns, surface_tension | 满足物理方程 |
| **约束损失** | volume_conservation, sharpening | 界面质量、体积守恒 |

---

## 2. 损失函数详解

### 2.1 数据拟合损失

#### 2.1.1 界面数据损失 (interface_loss)

```python
def _compute_data_loss(self, data, physics_cfg, stage1_factor):
    """界面数据拟合损失"""
    idx = torch.randperm(len(data["interface_points"]))[:self.batch_size]
    interface_pts = data["interface_points"][idx]
    interface_tgt = data["interface_targets"][idx]
    
    phi_pred = self.model(interface_pts)[:, 4]
    interface_loss = F.mse_loss(phi_pred, interface_tgt)
    
    base_weight = physics_cfg.get("interface_weight", 500.0)
    return interface_loss * base_weight * stage1_factor, interface_loss
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| interface_weight | 500.0 | 高权重确保界面形状正确 |

#### 2.1.2 初始条件损失 (ic_loss)

```python
# t=0 时油墨在底部 (z < h_ink)
phi_ic = 0.5 * (1 - np.tanh((z - self.h_ink) / interface_width))

ic_loss = F.mse_loss(pred_ic[:, 4:5], data["ic_values"][idx_ic][:, 4:5])
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ic_weight | 100.0 | 初始条件约束强度 |

#### 2.1.3 边界条件损失 (bc_loss)

```python
# 壁面无滑移条件 u=v=w=0
bc_loss = F.mse_loss(pred_bc[:, :3], data["bc_values"][idx_bc][:, :3])
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| bc_weight | 50.0 | 边界约束强度 |

---

### 2.2 物理方程损失

#### 2.2.1 连续性方程 (continuity)

$$\nabla \cdot \mathbf{u} = 0$$

```python
def compute_continuity_residual(self, u, v, w, coords):
    """∇·u = 0"""
    grad_u = self.compute_gradient(u, coords)
    grad_v = self.compute_gradient(v, coords)
    grad_w = self.compute_gradient(w, coords)
    
    continuity = grad_u[:, 0] + grad_v[:, 1] + grad_w[:, 2]
    return continuity
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| continuity_weight | 0.1 | 质量守恒约束 |

#### 2.2.2 VOF 输运方程 (vof)

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0$$

```python
def _compute_vof_residual(self, x_phys, predictions, model):
    """∂φ/∂t + u·∇φ = 0"""
    # 时间导数
    phi_t = g_phi[:, 5]
    
    # 对流项
    vof_advection = phi_t + u * phi_x + v * phi_y + w * phi_z
    
    return vof_advection
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| vof_weight | 0.1 | 界面输运约束 |

#### 2.2.3 Navier-Stokes 方程 (ns)

$$\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{F}_{st} + \mathbf{F}_{ew}$$

```python
def compute_navier_stokes_residual(self, x_phys, predictions, model):
    """ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew"""
    # 混合密度/粘度
    rho = phi * rho_oil + (1 - phi) * rho_polar
    mu = phi * mu_oil + (1 - phi) * mu_polar
    
    # 对流项
    u_conv = u * u_x + v * u_y + w * u_z
    
    # 粘性项
    u_laplacian = u_xx + u_yy + u_zz
    
    # NS 残差
    ns_u = rho * (u_t + u_conv) + p_x - mu * u_laplacian
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ns_weight | 0.01 | 动量守恒约束 |

#### 2.2.4 表面张力 (surface_tension)

CSF 模型: $\mathbf{F}_{st} = \sigma \kappa \nabla \phi$

```python
def compute_surface_tension_residual(self, phi, coords):
    """CSF 表面张力模型"""
    # 曲率 κ = -∇·(∇φ/|∇φ|)
    grad_phi_mag = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-10)
    kappa = -laplacian_phi / grad_phi_mag
    
    return kappa
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| surface_tension_weight | 0.001 | 表面张力约束 |

---

### 2.3 约束损失

#### 2.3.1 体积守恒 (volume_conservation)

$$\frac{1}{V_0} \int_V \phi \, dV = 1$$

详见 [`interface_sharpening_volume_conservation.md`](./interface_sharpening_volume_conservation.md)

#### 2.3.2 界面锐化 (sharpening)

$$\mathcal{L}_{sharp} = \phi (1 - \phi)$$

详见 [`interface_sharpening_volume_conservation.md`](./interface_sharpening_volume_conservation.md)

---

## 3. 三阶段损失权重

```python
def get_physics_weights(self, epoch):
    if epoch < self.stage1_epochs:
        # Stage 1: 纯数据学习
        return {"continuity": 0.0, "vof": 0.0, "ns": 0.0, "surface_tension": 0.0}
    
    elif epoch < self.stage2_epochs:
        # Stage 2: 渐进引入物理 (10% 权重)
        progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
        smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))
        
        return {
            "continuity": 0.1 * smooth_factor * 0.1,
            "vof": 0.1 * smooth_factor * 0.1,
            "ns": 0.01 * smooth_factor * 0.05,
            "surface_tension": 0.0
        }
    
    else:
        # Stage 3: 完整物理约束
        return {
            "continuity": 0.1,
            "vof": 0.1,
            "ns": 0.01,
            "surface_tension": 0.001
        }
```

---

## 4. 损失函数全景图

```
总损失 = Σ w_i × L_i

├── 数据损失 (Data Loss)
│   ├── interface_loss (权重 500.0) ⭐ 核心
│   ├── ic_loss (权重 100.0)
│   ├── bc_loss (权重 50.0)
│   ├── early_time_loss (权重 300.0)
│   └── zero_voltage_loss (权重 500.0)
│
├── 物理损失 (Physics Loss)
│   ├── continuity_loss (权重 0.1) ─── ∇·u = 0
│   ├── vof_loss (权重 0.1) ───────── ∂φ/∂t + u·∇φ = 0
│   ├── ns_loss (权重 0.01) ───────── ρ(∂u/∂t + u·∇u) = ...
│   └── surface_tension (权重 0.001) ─ CSF 模型
│
└── 约束损失 (Constraint Loss)
    ├── volume_conservation (权重 10.0) ⭐ 重要
    └── sharpening (权重 0.1) ─────── 界面锐化
```

---

## 5. 损失项优先级

| 优先级 | 损失项 | 权重 | 目的 |
|--------|--------|------|------|
| **P0** | interface_loss | 500.0 | 界面形状正确 |
| **P0** | volume_conservation | 10.0 | 体积守恒 |
| **P1** | ic_loss | 100.0 | 初始条件 |
| **P1** | bc_loss | 50.0 | 边界条件 |
| **P2** | continuity | 0.1 | 质量守恒 |
| **P2** | vof | 0.1 | 界面输运 |
| **P3** | ns | 0.01 | 动量守恒 |
| **P3** | surface_tension | 0.001 | 表面张力 |

---

## 6. 数值稳定性

### 6.1 NaN/Inf 清理

```python
def _sanitize_tensor(self, tensor, name=""):
    """清理 NaN/Inf 值"""
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logger.warning(f"张量 {name} 包含 {nan_count} 个 NaN, {inf_count} 个 Inf")
        tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
    return tensor
```

### 6.2 Log1p 缩放

```python
# 对大损失值使用 log1p 稳定训练
ns_loss = torch.mean(ns_u_norm**2 + ns_v_norm**2 + ns_w_norm**2)
return torch.log1p(ns_loss)
```

---

## 7. 相关文档

| 文档 | 说明 |
|------|------|
| [interface_sharpening_volume_conservation.md](./interface_sharpening_volume_conservation.md) | 界面锐化与体积守恒 |
| [sampling_strategy.md](./sampling_strategy.md) | 采样策略 |
| [physics_refactor_checklist.md](./physics_refactor_checklist.md) | 物理残差重构 |

---

**最后更新**: 2026-02-04
