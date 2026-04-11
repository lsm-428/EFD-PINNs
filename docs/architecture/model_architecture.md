# 模型架构详解

**最后更新**: 2026-02-04

## 整体架构概览

```
Stage 1: 物理基准校准 (Calibrated)
    EnhancedApertureModel: V → θ → η (解析映射)

Stage 2: 6D 两相流 PINN (Core)
    Input: (x, y, z, V_from, V_to, t_since)
    Output: (u, v, w, p, φ)
    Physics: N-S + 电润湿力 + VOF + 连续性
```

---

## TwoPhasePINN 网络结构

**文件**: `src/models/pinn_two_phase.py`

### 双网络架构

```python
class TwoPhasePINN(nn.Module):
    # Phi 网络：预测体积分数
    # Input: 6D (x,y,z,V_from,V_to,t_since) → Output: 1 (phi)
    self.phi_net = [6 → 128 → 128 → 64 → 32 → 1]

    # 速度网络：预测速度和压力
    # Input: 7D (6D + phi) → Output: 4 (u,v,w,p)
    self.vel_net = [7 → 64 → 64 → 32 → 4]
```

### 前向传播

```python
def forward(self, x):
    # x: (batch, 6) → [x, y, z, V_from, V_to, t_since]
    
    # 1. 归一化
    x_norm = x / Lx, y_norm = y / Ly, z_norm = z / Lz
    V_from_norm = V_from / 30, V_to_norm = V_to / 30
    t_norm = t_since / t_max
    
    # 详细参数说明请参见[物理理论与器件规格指南](../guides/physics_and_device_guide.md#physics-parameters)。
    
    # 2. Phi 预测
    phi_input = [x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm]
    phi = sigmoid(phi_net(phi_input))  # 约束在 [0, 1]
    
    # 3. 速度/压力预测（phi 作为输入）
    vel_input = [phi_input, phi]
    u, v, w, p = vel_net(vel_input)
    
    return (u, v, w, p, phi)
```

---

## PhysicsLoss 物理损失

### 含电润湿力的 N-S 方程

```python
class PhysicsLoss:
    def navier_stokes_residual(self, grads, V):
        # 混合流体属性
        rho = phi * rho_oil + (1-phi) * rho_polar
        mu = phi * mu_oil + (1-phi) * mu_polar
        
        # 对流项
        u_conv = u*u_x + v*u_y + w*u_z
        
        # 粘性项
        u_visc = mu * (u_xx + u_yy + u_zz) + ...
        
        # 表面张力 (CSF)
        kappa = compute_curvature(grads)
        F_st = sigma * kappa * grad_phi
        
        # 电润湿力 (CSF) - 关键！
        V_eff = max(0, V - V_threshold)
        sigma_ew = epsilon_0 * epsilon_r * V_eff**2 / (2*d)
        F_ew = -sigma_ew * grad_phi
        
        # N-S 残差
        ns = rho*(u_t + u_conv) + p_x - u_visc - F_st - F_ew
        return mean(ns**2)
```

### 电润湿力物理意义

```
F_ew = -σ_ew × ∇φ

其中：
σ_ew = ε₀εᵣ(V-V_T)²/(2d)  [N/m]

物理解释：
- 电场在介电层产生 Maxwell 应力
- 等效于降低界面表面张力
- 驱动三相接触线移动
- 使油墨从中心向边缘移动
```

---

## 损失函数结构

### compute_losses 结构

```python
def compute_losses(data, epoch):
    losses = {}

    # 1. 数据损失 (Stage 1)
    losses["interface"] = interface_loss * 500.0
    losses["ic"] = ic_loss * 300.0
    losses["bc"] = bc_loss * 80.0

    # 2. 物理方程 (Stage 2-3)
    losses["continuity"] = continuity_loss * 0.5
    losses["vof"] = vof_loss * 0.5
    losses["ns"] = ns_loss * 0.1
    losses["surface_tension"] = st_loss * 0.01

    # 3. 界面锐化 (Stage 3)
    losses["sharpening"] = sharpening_loss * 1.0

    return losses
```

### 三阶段权重调度

| 阶段 | Epoch 范围 | continuity | vof | ns | surface_tension | sharpening |
|------|-----------|------------|-----|-----|-----------------|-------------|
| Stage 1 | 0 - 1,500 | 0 | 0 | 0 | 0 | 0 |
| Stage 2 | 1,500 - 5,500 | 0 → 0.5 | 0 → 0.5 | 0 → 0.01 | 0 | 0 |
| Stage 3 | 5,500 - 60,000 | 0.5 | 0.5 | 0.1 | 0.01 | 1.0 |

---

## 训练器配置

```python
class Trainer:
    # 阶段划分
    stage1_epochs = 1500    # 纯数据学习
    stage2_epochs = 4000  # 连续性 + VOF
    stage3_epochs = 60000  # 完整物理
    
    # 优化器
    optimizer = Adam(lr=0.0003)
    scheduler = CosineAnnealingLR(T_max=epochs)
    
    # 学习率预热
    warmup_epochs = 500
    warmup_start_lr = lr * 0.01
    
    # 梯度裁剪
    gradient_clip = 1.0
    
    # 批次大小
    batch_size = 4096
```

---

## 物理参数

```python
PHYSICS = {
    # 几何参数
    "Lx": 174e-6,        # 像素宽度 (m)
    "Ly": 174e-6,        # 像素高度 (m)
    "Lz": 20e-6,         # 总高度 (m)
    "h_ink": 3e-6,       # 油墨层厚度 (m)
    
    # 流体参数
    "rho_oil": 800.0,     # 油墨密度 (kg/m³)
    "rho_polar": 1000.0, # 极性液体密度 (kg/m³)
    "mu_oil": 0.003,     # 油墨粘度 (Pa·s)
    "mu_polar": 0.001,   # 极性液体粘度 (Pa·s)
    "gamma": 0.015,       # 表面张力 (N/m)
    
    # 电润湿参数
    "theta0": 120.0,      # 初始接触角 (°)
    "theta_wall": 71.0,   # 像素墙接触角 (°)
    "epsilon_r": 12.0,    # 有效介电常数
    "d_dielectric": 4e-7, # 介电层厚度 (m)
    "V_threshold": 3.0,   # 阈值电压 (V)
    
    # 动态参数
    "tau": 0.005,        # 响应时间常数 (s)
    "tau_recovery": 0.0075, # 恢复时间常数 (s)
    "zeta": 0.8,         # 阻尼比
    "t_max": 0.05,       # 最大时间 (s)
    
    # 训练约束
    "eta_max": 0.85,     # 最大开口率
}
```

---

## 数据采样配置

```python
DATA_CONFIG = {
    "n_interface": 100000,  # 界面数据点
    "n_initial": 10000,     # 初始条件点
    "n_boundary": 10000,   # 边界条件点
    "n_domain": 20000,      # 域内配点
    "voltages": [0, 0.5, 1.0, ..., 30.0],  # 0.5V 步进 (61点)
    "times": 50,           # 时间采样密度
}
```
