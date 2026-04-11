# 界面锐化与体积守恒设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**对应代码**: `src/models/pinn_two_phase.py`, `src/physics/constraints.py`

---

## 1. 设计背景

### 1.1 问题描述

在 PINN 训练中，两个关键问题：

| 问题 | 现象 | 原因 |
|------|------|------|
| **界面模糊** | φ 在 [0,1] 之间平滑过渡，界面不清晰 | sigmoid 输出过于平滑 |
| **体积不守恒** | 油墨体积随时间变化 | VOF 方程数值扩散 |

### 1.2 解决方案

| 问题 | 解决方案 | 公式 |
|------|----------|------|
| **界面模糊** | 界面锐化损失 | $\mathcal{L}_{sharp} = \phi(1-\phi)$ |
| **体积不守恒** | 体积守恒损失 | $\frac{1}{V_0} \int_V \phi \, dV = 1$ |

---

## 2. 界面锐化 (Sharpening)

### 2.1 设计原理

**核心思想**: 惩罚处于中间状态的点，迫使 φ 趋向 0 或 1。

$$\mathcal{L}_{sharp} = \frac{1}{N} \sum_{i=1}^{N} \phi_i (1 - \phi_i)$$

**几何意义**:
- 当 φ = 0 或 φ = 1 时，损失 = 0（理想状态）
- 当 φ = 0.5 时，损失 = 0.25（最大惩罚）

```
φ ∈ [0,1]        损失函数 φ(1-φ)
    ↑                  ↑
1.0 ┤  φ=1          0.25 ┤           ╭
    │                0.20 ┤          ╱
0.5 ┤  φ=0.5        0.15 ┤         ╱
    │                0.10 ┤        ╱
0.0 ┤  φ=0          0.05 ┤       ╱
    └────────────→ 0.00 ┼───────╯───────────→
      0.0  0.5  1.0        0.0   0.5   1.0
```

### 2.2 代码实现

```python
def compute_sharpening_loss(self, phi_pred):
    """
    计算界面锐化损失
    
    φ(1-φ) 在 φ=0 和 φ=1 时为 0，
    在 φ=0.5 时达到最大值 0.25
    """
    sharpening_val = torch.mean(phi_pred * (1.0 - phi_pred))
    return sharpening_val
```

### 2.3 配置参数

```python
DEFAULT_CONFIG = {
    "physics": {
        "sharpening": 0.1,  # 界面锐化损失权重
    }
}
```

### 2.4 训练效果

| 阶段 | sharpening 权重 | 界面效果 |
|------|-----------------|----------|
| Stage 1 | 0.0 | 不启用 |
| Stage 2 | 0.05 | 缓慢锐化 |
| Stage 3 | 0.1 | 完全锐化 |

---

## 3. 体积守恒 (Volume Conservation)

### 3.1 设计原理

**核心思想**: 强制油墨体积在整个模拟过程中保持不变。

$$V_{ink} = \int_V \phi \, dV = V_0 = L_x \times L_y \times h_{ink}$$

**相对误差**:

$$\mathcal{L}_{vol} = \left| \frac{V_{ink} - V_0}{V_0} \right|$$

### 3.2 蒙特卡洛积分

```python
def compute_volume_conservation_residual(self, x_phys, predictions):
    """
    计算体积守恒残差
    
    使用蒙特卡洛积分估计油墨体积
    """
    # 提取 φ 场
    phi = predictions[:, 4]
    phi_clamped = torch.clamp(phi, 0.0, 1.0)
    
    # 计算平均 φ 值
    phi_mean = torch.mean(phi_clamped)
    
    # 初始油墨体积分数
    ink_fraction_target = 0.15  # h_ink / Lz = 3μm / 20μm
    
    # 体积守恒残差（相对误差）
    volume_residual = (phi_mean - ink_fraction_target) / ink_fraction_target
    
    return volume_residual
```

### 3.3 体积守恒 vs VOF 方程

| 约束 | 类型 | 作用 |
|------|------|------|
| **VOF 方程** | 微分约束 | ∂φ/∂t + u·∇φ = 0 |
| **体积守恒** | 积分约束 | ∫φ dV = const |

**两者的关系**:
- VOF 方程保证局部的质量输运
- 体积守恒约束保证全局的质量守恒
- 两者结合可以实现精确的体积守恒

### 3.4 配置参数

```python
DEFAULT_CONFIG = {
    "physics": {
        "volume_base_weight": 10.0,  # 体积守恒基础权重
    }
}
```

### 3.5 训练效果

| 阶段 | volume 权重 | 体积误差 |
|------|-------------|----------|
| Stage 1 | 0.0 | 无约束 |
| Stage 2 | 5.0 | ~5% |
| Stage 3 | 10.0 | <1% |

---

## 4. 人工压缩项 (Artificial Compression)

### 4.1 设计原理

在 VOF 方程中添加人工压缩项，防止界面数值扩散：

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi + \nabla \cdot (\mathbf{u}_c \phi (1 - \phi)) = 0$$

其中 $\mathbf{u}_c = C_\alpha |\mathbf{u}| \mathbf{n}_{interface}$

### 4.2 代码实现

```python
def _compute_vof_residual(self, x_phys, predictions, model):
    """VOF 方程 + 人工压缩项"""
    # 1. 标准 VOF 残差
    vof_advection = phi_t + u * phi_x + v * phi_y + w * phi_z
    
    # 2. 界面法向量 n = ∇φ / |∇φ|
    grad_phi_mag = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-10)
    n_x, n_y, n_z = phi_x / grad_phi_mag, phi_y / grad_phi_mag, phi_z / grad_phi_mag
    
    # 3. 压缩速度 |u_c| = C_α × |u|
    vel_mag = torch.sqrt(u**2 + v**2 + w**2 + 1e-10)
    c_alpha = 1.0  # 压缩系数
    
    # 4. 压缩通量 φ(1-φ) 在 φ=0,1 时为 0
    factor = c_alpha * vel_mag * phi * (1 - phi)
    flux_c = factor * n
    
    # 5. 压缩项散度
    compression_term = div(flux_c)
    
    # 6. 组合残差
    vof_residual = vof_advection + compression_term
    
    return vof_residual
```

### 4.3 压缩项作用

```
界面区域 (φ ∈ [0,1])        远离界面 (φ ≈ 0 或 1)
         ↑                        ↑
         │    ╔═══════════════╗    │
 φ=1 ────┼────╢   压缩通量    ║────┼─── φ=1
         │    ╠═══════════════╣    │
 φ=0.5 ──┼────╢   最大压缩    ║────┼─── φ=0.5
         │    ╠═══════════════╣    │
 φ=0 ────┼────╚═══════════════╝────┼─── φ=0
         └────────────────────────→ 速度方向
```

---

## 5. 完整物理损失组合

### 5.1 损失函数公式

$$\mathcal{L}_{total} = w_{data} \mathcal{L}_{data} + w_{cont} \mathcal{L}_{cont} + w_{vof} \mathcal{L}_{vof} + w_{sharp} \mathcal{L}_{sharp} + w_{vol} \mathcal{L}_{vol}$$

### 5.2 v4.5 默认配置

```json
{
  "physics": {
    "interface_weight": 500.0,
    "continuity_weight": 0.1,
    "vof_weight": 0.1,
    "ns_weight": 0.01,
    "surface_tension_weight": 0.001,
    "sharpening": 0.1,
    "volume_base_weight": 10.0
  }
}
```

### 5.3 损失权重演化

```
权重值
    ↑
10.0 ┤                                              ── volume (10.0)
     │                                        ╱
 5.0 ┤                                   ╱────╯
     │                              ╱────╯
 1.0 ┤                         ╱────╯          ── interface (500)
     │                   ╱────╯               ╱
 0.1 ┤             ╱────╯                   ╱   ── continuity (0.1)
     │       ╱────╯                        ╱
 0.01┤──╮╱                               ╱    ── vof (0.1)
     │ ╱                                ╱
 0.001┤╱                                ╱     ── ns (0.01)
     └──────────────────────────────────────────────→ epoch
         Stage 1      Stage 2         Stage 3
       (0-5,000)   (5,000-15,000)  (15,000-30,000)
```

---

## 6. 验证指标

### 6.1 界面质量

| 指标 | 目标 | 说明 |
|------|------|------|
| 界面宽度 | < 5μm | φ 从 0.1 到 0.9 的距离 |
| 界面锐度 | > 10 | 界面处梯度 |

### 6.2 体积守恒

| 指标 | 目标 | 说明 |
|------|------|------|
| 相对误差 | < 1% | |V - V₀| / V₀ |
| 体积漂移 | < 0.1%/ms | 长时间模拟 |

---

## 7. 相关文档

| 文档 | 说明 |
|------|------|
| [loss_function_design.md](./loss_function_design.md) | 完整损失函数设计 |
| [sampling_strategy.md](./sampling_strategy.md) | 采样策略 |
| [physics_refactor_checklist.md](./physics_refactor_checklist.md) | 物理残差重构 |

---

## 8. 参考文献

| 论文 | 贡献 |
|------|------|
| Hirt & Nichols (1981) | VOF 方法原始论文 |
| Rusinking et al. | 界面锐化技术 |
| Sussman et al. | 人工压缩项 |

---

**最后更新**: 2026-02-04
