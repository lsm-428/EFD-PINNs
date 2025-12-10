# 模型架构详解

**最后更新**: 2025-12-08

## 整体架构概览

EFD-PINNs 采用两阶段架构：

```
Stage 1: 接触角预测 (解析公式)
    V → θ(t)

Stage 2: 两相流 PINN
    (x, y, z, t, V) → (u, v, w, p, φ)
```

---

## TwoPhasePINN 架构

**文件**: `src/models/pinn_two_phase.py`

### 网络结构

```python
class TwoPhasePINN(nn.Module):
    def __init__(self, config):
        # Phi 网络：预测体积分数
        self.phi_net = self._build_network(5, 1, [64, 64, 64, 32])
        
        # 速度网络：预测速度和压力
        self.vel_net = self._build_network(6, 4, [64, 64, 32])
```

### 前向传播

```python
def forward(self, x):
    # 输入: (batch, 5) - (x, y, z, t, V)
    
    # 归一化
    x_norm = x_coord / Lx
    y_norm = y_coord / Ly
    z_norm = z_coord / Lz
    t_norm = t / t_max
    V_norm = V / 30.0
    
    # Phi 预测
    phi_input = [x_norm, y_norm, z_norm, t_norm, V_norm]
    phi = sigmoid(phi_net(phi_input))
    
    # 速度预测
    vel_input = [x_norm, y_norm, z_norm, t_norm, V_norm, phi]
    u, v, w, p = vel_net(vel_input)
    
    # 输出: (batch, 5) - (u, v, w, p, phi)
    return [u, v, w, p, phi]
```

### 网络配置

| 网络 | 输入维度 | 输出维度 | 隐藏层 |
|------|----------|----------|--------|
| phi_net | 5 | 1 | [64, 64, 64, 32] |
| vel_net | 6 | 4 | [64, 64, 32] |

---

## HybridPredictor 架构

**文件**: `src/predictors/hybrid_predictor.py`

### 核心方法

```python
class HybridPredictor:
    def young_lippmann(self, V):
        """Young-Lippmann 方程"""
        cos_theta0 = cos(radians(theta0))
        ew_term = (epsilon_0 * epsilon_r * V**2) / (2 * gamma * d)
        cos_theta = clip(cos_theta0 + ew_term, -1, 1)
        return degrees(arccos(cos_theta))
    
    def dynamic_response(self, t, theta_start, theta_eq):
        """二阶欠阻尼响应"""
        exp_term = exp(-zeta * omega_0 * t)
        damping = zeta / sqrt(1 - zeta**2)
        return theta_eq + (theta_start - theta_eq) * exp_term * (
            cos(omega_d * t) + damping * sin(omega_d * t)
        )
    
    def predict(self, voltage, time):
        """混合预测"""
        theta_eq = self.young_lippmann(voltage)
        theta_start = self.young_lippmann(0)
        return self.dynamic_response(time, theta_start, theta_eq)
```

---

## 物理损失

### PhysicsLoss 类

```python
class PhysicsLoss:
    def continuity_residual(self, grads):
        """连续性方程: ∇·u = 0"""
        return mean((u_x + v_y + w_z)**2)
    
    def vof_residual(self, grads):
        """VOF 方程: ∂φ/∂t + u·∇φ = 0"""
        return mean((phi_t + u*phi_x + v*phi_y + w*phi_z)**2)
    
    def navier_stokes_residual(self, grads):
        """Navier-Stokes 方程"""
        # ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
        ...
```

---

## 训练策略

### 渐进式训练

```
阶段 1 (0-5000 epochs):
    - 纯数据学习
    - 物理约束权重: 0

阶段 2 (5000-15000 epochs):
    - 引入连续性和 VOF 约束
    - 平滑过渡

阶段 3 (15000+ epochs):
    - 完整物理约束
    - 包括 N-S 方程
```

### 损失函数

```python
total_loss = (
    contact_angle_loss * 500.0 +
    ic_loss * 100.0 +
    bc_loss * 50.0 +
    early_time_loss * 500.0 +
    continuity_loss * 0.5 +
    vof_loss * 0.5 +
    ns_loss * 0.1
)
```

---

## 性能优化

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 学习率调度

```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6
)
```

### 批量大小

```python
batch_size = 4096  # 大批量提高训练效率
```
