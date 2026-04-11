# Stage 2: TwoPhasePINN 双 MLP 架构设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**对应代码**: `src/models/pinn_two_phase.py`

---

## 1. 设计概述

### 1.1 为什么是"双 MLP"而不是单 MLP?

Stage 2 的模型叫 `TwoPhasePINN`，**它就是 PINN**。所谓"双 MLP"是指网络内部有两个独立分支：

| 分支 | 输入 | 输出 | 物理意义 |
|------|------|------|----------|
| **φ 网络** | 6D (x, y, z, V_from, V_to, t_since) | 1D (φ ∈ [0,1]) | VOF 场，界面位置 |
| **速度网络** | 7D (x, y, z, V_from, V_to, t_since, φ) | 4D (u, v, w, p) | 速度场 + 压力场 |

**为什么分开**：
1. **物理解耦**: φ 是界面场，需要 sigmoid 限制在 [0,1]；速度场无此约束
2. **梯度隔离**: 两个场物理意义不同，合在一起会影响梯度传播
3. **便于约束**: 物理损失计算需要分别处理 φ 场和速度场

---

## 2. 架构详解

### 2.1 整体架构图

```
输入: (x, y, z, V_from, V_to, t_since) ── 6D ──┐
                                                  │
    ┌─────────────────────────────────────────────┼──────────────────────────────┐
    │                                             │                              │
    ▼                                             ▼                              │
┌───────────────┐                         ┌─────────────────────┐               │
│   φ 网络       │                         │    速度网络         │               │
│               │                         │                     │               │
│ 输入: 6       │                         │ 输入: 7             │               │
│ hidden:       │                         │ hidden:             │               │
│ [64, 64,      │                         │ [64, 64,            │               │
│  64, 32]      │                         │  32]                │               │
│ output: 1     │                         │ output: 4           │               │
│ activation:   │                         │ activation:         │               │
│ Tanh + Sigmoid│                         │ Tanh                │               │
└───────┬───────┘                         └──────────┬──────────┘               │
        │                                            │                           │
        ▼                                            ▼                           │
    φ ∈ [0,1]                              (u, v, w, p)                         │
        │                                            │                           │
        └────────────────────────┬───────────────────┘                           │
                                 ▼                                               │
                      输出: (u, v, w, p, φ) ── 5D ──┐
                                                   │
                                                   ▼
                                        物理约束计算 (PhysicsConstraints)
                                        ├── continuity: ∇·u = 0
                                        ├── vof: ∂φ/∂t + u·∇φ = 0
                                        ├── ns: ρ(∂u/∂t + u·∇u) = ...
                                        └── volume_conservation
```

### 2.2 网络配置

```python
DEFAULT_CONFIG = {
    "model": {
        "hidden_phi": [128, 128, 64, 32],    # φ 网络: 6 → 128 → 128 → 64 → 32 → 1
        "hidden_vel": [64, 64, 32],        # 速度网络: 7 → 64 → 64 → 32 → 4
    }
}
```

| 网络 | 输入维度 | 隐藏层 | 输出维度 | 总参数量 |
|------|----------|--------|----------|----------|
| **φ 网络** | 6 | [128, 128, 64, 32] | 1 | 6×128 + 128×128 + 128×64 + 64×32 + 32×1 = **27,424** |
| **速度网络** | 7 | [64, 64, 32] | 4 | 7×64 + 64×64 + 64×32 + 32×4 = **8,320** |
| **总计** | - | - | - | **~17K 参数** |

### 2.3 激活函数

```python
def _build_network(self, input_dim, output_dim, hidden_layers):
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.Tanh())  # 隐藏层: Tanh
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
```

| 位置 | 激活函数 | 原因 |
|------|----------|------|
| **隐藏层** | Tanh | 平滑梯度，有界输出 |
| **φ 输出层** | Sigmoid | φ ∈ [0, 1] |
| **速度输出层** | Linear | 无约束 |

---

## 3. 输入格式

### 3.1 6D Triad 输入

```
输入: (x, y, z, V_from, V_to, t_since)

参数说明:
- x, y, z:   空间坐标 (m)，归一化到 [0, 1]
- V_from:    跳变前电压 (V)，归一化到 [0, 1]
- V_to:      跳变后电压 (V)，归一化到 [0, 1]
- t_since:   跳变后经过的时间 (s)，归一化到 [0, 1]
```

### 3.2 归一化

```python
def forward(self, x):
    """前向传播"""
    # 空间坐标
    x_norm = x[:, 0:1] / self.Lx   # 174e-6
    y_norm = x[:, 1:2] / self.Ly   # 174e-6
    z_norm = x[:, 2:3] / self.Lz   # 20e-6
    
    # 时间
    t_norm = x[:, 5:6] / self.t_max  # 0.05s
    
    # 电压
    V_from_norm = x[:, 3:4] / 30.0
    V_to_norm = x[:, 4:5] / 30.0
    
    # 组合归一化输入
    x_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1)
```

---

## 4. 输出格式

### 4.1 5D 输出

```
输出: (u, v, w, p, φ)

参数说明:
- u, v, w:  速度场 (m/s)
- p:        压力场 (Pa)
- φ:        VOF 场 [0, 1]
```

### 4.2 速度网络输入

```python
# 速度网络需要 φ 作为额外输入
vel_input = torch.cat([
    x_norm, y_norm, z_norm,
    V_from_norm, V_to_norm, t_norm,
    phi  # φ 网络的输出，作为条件
], dim=-1)
```

**设计理由**: 速度场依赖于界面位置（油墨 vs 极性液体），φ 提供了这种条件信息。

---

## 5. 三阶段训练策略

### 5.1 阶段定义

```python
DEFAULT_CONFIG = {
    "training": {
        "stage1_epochs": 1500,    # 纯数据学习
        "stage2_epochs": 4000,    # 渐进物理约束
        "stage3_epochs": 50000,   # 完整物理约束
    }
}
```

### 5.2 阶段1 (0 - 1,500): 纯数据学习

| 项目 | 说明 |
|------|------|
| **目标** | 学习基本界面形状 |
| **物理损失** | **全部为 0** |
| **数据损失** | interface_loss, ic_loss, bc_loss |

```python
def get_physics_weights(self, epoch):
    if epoch < self.stage1_epochs:
        return {"continuity": 0.0, "vof": 0.0, "ns": 0.0, "surface_tension": 0.0}
```

### 5.3 阶段2 (1,500 - 5,500): 渐进物理约束

| 项目 | 说明 |
|------|------|
| **目标** | 缓慢引入物理约束 |
| **物理权重** | 10% (平滑过渡) |
| **过渡函数** | Sigmoid 曲线 |

```python
progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))

continuity = 0.1 * smooth_factor * 0.1  # → 0.01
vof = 0.1 * smooth_factor * 0.1         # → 0.01
```

### 5.4 阶段3 (5,500 - 60,000): 完整物理约束

| 项目 | 说明 |
|------|------|
| **目标** | 完全启用物理约束 |
| **物理权重** | 100% |

```python
return {
    "continuity": 0.1,
    "vof": 0.1,
    "ns": 0.01,
    "surface_tension": 0.001,
}
```

---

## 6. 损失函数

### 6.1 损失全景图

```
总损失 = Σ w_i × L_i

├── 数据损失
│   ├── interface_loss (权重 500.0) ⭐ 核心
│   ├── ic_loss (权重 300.0)
│   └── bc_loss (权重 80.0)
│
├── 物理损失
│   ├── continuity_loss (权重 0.1) ─── ∇·u = 0
│   ├── vof_loss (权重 0.1) ───────── ∂φ/∂t + u·∇φ = 0
│   ├── ns_loss (权重 0.01) ───────── Navier-Stokes
│   └── surface_tension (权重 0.001)
│
└── 约束损失
    ├── volume_conservation (权重 10.0) ⭐ 重要
    └── sharpening (权重 1.0)
```

### 6.2 物理损失详解

| 损失项 | 方程 | 代码位置 |
|--------|------|----------|
| **continuity** | ∇·u = 0 | `PhysicsConstraints.compute_continuity_residual` |
| **vof** | ∂φ/∂t + u·∇φ = 0 | `PhysicsConstraints._compute_vof_residual` |
| **ns** | ρ(∂u/∂t + u·∇u) = ... | `PhysicsConstraints.compute_navier_stokes_residual` |
| **surface_tension** | CSF 模型 | `PhysicsConstraints.compute_surface_tension_residual` |

---

## 7. 与 Stage 1/3 的关系

### 7.1 架构对比

| Stage | 模型 | 输入 | 核心创新 |
|-------|------|------|----------|
| **Stage 1** | EnhancedApertureModel | (V, t) | 解析模型，接触角预测 |
| **Stage 2** | TwoPhasePINN | (x, y, z, V, t) | 双 MLP，PINN 场预测 |
| **Stage 3** | LSTM-PINN | (x, y, z, t, seq) | LSTM 编码电压历史 |

### 7.2 数据流

```
Stage 1 输出                    Stage 2 输入
┌─────────────────────┐       ┌─────────────────────────────┐
│ θ (接触角)          │  ──▶  │ 6D Triad                     │
│ η (开口率)          │       │ (x, y, z, V_from, V_to,     │
└─────────────────────┘       │  t_since)                    │
                              │                             │
                              ▼                             ▼
                    TwoPhasePINN                    物理损失计算
                    ┌─────────────┐                ┌─────────────────┐
                    │ φ 网络      │ ──▶ φ 场 ──▶  │ VOF 方程        │
                    │ 速度网络    │ ──▶ 速度 ──▶ │ NS 方程         │
                    └─────────────┘                └─────────────────┘
```

---

## 8. 代码实现

### 8.1 网络构建

```python
class TwoPhasePINN(nn.Module):
    def __init__(self, config):
        # φ 网络: 输入 6 维，输出 1 维
        self.phi_net = self._build_network(6, 1, hidden_phi)
        
        # 速度网络: 输入 7 维 (6 + φ)，输出 4 维
        self.vel_net = self._build_network(7, 4, hidden_vel)
```

### 8.2 前向传播

```python
def forward(self, x):
    """前向传播"""
    # 提取输入
    x_coord, y_coord, z_coord = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    V_from, V_to, t_since = x[:, 3:4], x[:, 4:5], x[:, 5:6]
    
    # 归一化
    x_norm = x_coord / self.Lx
    y_norm = y_coord / self.Ly
    z_norm = z_coord / self.Lz
    t_norm = t_since / self.t_max
    V_from_norm = V_from / 30.0
    V_to_norm = V_to / 30.0
    
    # φ 预测
    phi_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1)
    phi_raw = self.phi_net(phi_input)
    phi = torch.sigmoid(phi_raw)  # 限制在 [0, 1]
    
    # 速度预测
    vel_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm, phi], dim=-1)
    vel_out = self.vel_net(vel_input)
    u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]
    
    return torch.cat([u, v, w, p, phi], dim=-1)
```

---

## 9. 验证与测试

### 9.1 相关测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_pinn_complete.py` | 完整训练流程测试 |
| `tests/test_continuity.py` | 连续性方程验证 |
| `tests/test_physics_sanity.py` | 物理 sanity check |

### 9.2 验证指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 30V 开口率 | 86.8% | 与 Stage 1 对齐 |
| 体积误差 | <1% | VOF 体积守恒 |
| 损失收敛 | 稳定下降 | 无 NaN/Inf |

---

## 10. 相关文档

| 文档 | 说明 |
|------|------|
| [pinn_input_redesign.md](./pinn_input_redesign.md) | 6D Triad 输入设计 |
| [loss_function_design.md](./loss_function_design.md) | 损失函数详解 |
| [interface_sharpening_volume_conservation.md](./interface_sharpening_volume_conservation.md) | 界面锐化与体积守恒 |
| [sampling_strategy.md](./sampling_strategy.md) | 采样策略 |
| [lstm_pinn_stage3_design.md](../../experimental/lstm_pinn/docs/lstm_pinn_stage3_design.md) | Stage 3 LSTM 架构 |

---

**最后更新**: 2026-02-04
