# Stage 2 动态电压响应（升压与降压）- 设计文档

**最后更新**: 2026-02-04
**版本**: v4.5

---

## 1. 物理背景与挑战

在电润湿显示 (EWD) 过程中，升压与降压过程具有完全不同的物理机制：

| 阶段 | 核心驱动力 | 接触角 θ 行为 | 开口率 η 行为 | 建模挑战 |
| :--- | :--- | :--- | :--- | :--- |
| **升压 (Step Up)** | 电润湿力 (主动) | 动态变化 (120° → 67°) | 随 θ 动态收缩 | 界面拓扑快速变化 |
| **降压 (Step Down)** | 表面张力 (被动) | 瞬间恢复 (θ ≈ 120°) | 粘滞阻力限制恢复速度 | θ 瞬间恢复导致驱动力"突变"，传统 PINN 难以捕捉恢复惯性 |

### 核心问题

降压时接触角 θ 瞬间恢复到 120°，但油墨铺展（η 恢复）需要时间（基于实验校准 τ_recovery ≈ 7.5ms）。若使用简单的 `(x, y, z, t, V)` 输入，模型无法在 $V=0$ 时区分"初始静止态"与"降压恢复态"。

---

## 2. 最终选定方案：6D Triad 统一建模

经过方案对比，我们选择了 **6D Triad 输入空间方案**，结合 **物理软约束** 来实现升降压过程的统一建模。

### 2.1 6D 输入定义

`Input = (x, y, z, V_from, V_to, t_since)`

- **V_from**: 记录初始状态，区分是从 30V 降压到 0V 还是持续维持 0V。
- **t_since**: 提供相对时间基准，解耦绝对时间，支持任意时刻的电压跳变预测。

### 2.2 物理驱动机制

#### 升压驱动 (Active Drive)

- 使用 `EnhancedApertureModel` 提供的动态接触角 $\theta(t_{since})$ 作为底面 $z=0$ 的边界条件。
- 电润湿力通过界面形状改变直接驱动相场 $\phi$ 演化。

#### 降压驱动 (Passive Recovery + Soft Constraint)

- **边界条件**: 底面接触角固定为 $\theta = 120^\circ$。
- **解析引导损失 (Analytic Guidance Loss)**: 将实验观测到的指数恢复特性作为软约束加入 Loss：

$$\eta_{target}(t_{since}) = \eta_{steady}(V_{from}) \cdot \exp(-t_{since} / \tau_{recovery})$$

其中 $\tau_{recovery} = 7.5ms$。

---

## 3. 物理参数（与 Stage 1 同步）

从 `config/device_calibrated_physics.json` 读取：

```json
{
  "dynamics_params": {
    "tau": 0.005,
    "tau_recovery": 0.0075,
    "zeta": 0.8
  },
  "contact_angle": {
    "theta_0": 120.0,
    "theta_min": 67.5
  }
}
```

**物理含义**：
- τ = 5ms：升压时接触角变化的时间常数（电润湿驱动）
- τ_recovery = 7.5ms：降压时开口率恢复的时间常数（表面张力驱动）
- 恢复比响应慢 50%，符合"油推水走"比"水推油走"慢的物理直觉

---

## 4. 动态响应特性

### 升压响应 (0V → 20V)

| 时间 | 接触角 θ | 开口率 η | 说明 |
|------|-----------|----------|------|
| t=0ms | θ=120.0° | η=0% | 初始状态 (0V) |
| t=1ms | θ≈119.0° | η≈20% | 快速响应开始 |
| t=2ms | θ≈117.5° | η≈40% | |
| t=5ms | θ≈116.0° | η≈58% | 响应时间 t₉₀ |
| t=10ms | θ≈115.5° | η≈64% | 接近平衡 |
| t=20ms | θ≈115.2° | η≈66.7% | 稳态 (20V平衡) |

**关键特征**：
- 响应时间 t₉₀ ≈ 5ms（达到 90% 稳态值的时间）
- 响应曲线呈单调抛物线形（无振荡）
- 油墨开口率变化滞后于电压（粘滞阻尼效应）

### 降压响应 (20V → 0V)

| 时间 | 接触角 θ | 开口率 η | 说明 |
|------|-----------|----------|------|
| t=0ms | θ=120.0° | η≈66.7% | 初始状态 (20V) |
| t=1ms | θ≈120.0° | η≈50% | 接触角瞬间恢复，开口率刚开始下降 |
| t=5ms | θ≈120.0° | η≈25% | 粘滞阻力限制恢复速度 |
| t=10ms | θ≈120.0° | η≈10% | |
| t=20ms | θ≈120.0° | η≈2% | |
| t=50ms | θ≈120.0° | η≈0% | 完全恢复 (0V稳态) |

**关键特征**：
- 降压恢复时间常数 τ_recovery = 7.5ms
- 接触角瞬间恢复，但开口率按指数曲线恢复
- 恢复过程单调，无振荡

---

## 5. 训练策略实现

### 5.1 数据采样分布

- **升压采样**: 集中在 $V_{from} < V_{to}$ 的 Triad 区域。
- **降压采样**: 集中在 $V_{from} > V_{to}$ 的 Triad 区域，且 $t_{since} \in [0, 40ms]$。
- **稳态采样**: $V_{from} = V_{to}$，作为边界收敛点。
- **0.5V 高密度采样**: 确保 20V 附近非线性区域的连续性。
- **时间指数分布**: 前 5ms 加密采样，捕捉快速界面变化。

### 5.2 损失函数构成

1. **PDE Loss**: 完整的 Navier-Stokes (含表面张力) + VOF 输运方程。
2. **Boundary Loss**: $z=0$ 处的接触角约束。
3. **Data Loss**: 来自 `EnhancedApertureModel` 的 $\eta$ 目标映射。
4. **Dynamic Loss (降压专用)**: 上述的解析引导损失，确保降压过程的响应曲线符合实验特性。

### 5.3 损失函数伪代码实现

```python
def compute_total_loss(model, triad_data, epoch):
    """
    triad_data: (x, y, z, V_from, V_to, t_since)
    """
    losses = {}
    
    # 1. 基础物理损失 (NS + VOF + Continuity)
    # 在 Triad 空间中，dt 对应 d(t_since)
    losses["physics"] = compute_pde_residuals(model, triad_data)
    
    # 2. 边界条件损失 (z=0)
    # 根据 (V_from, V_to, t_since) 确定目标接触角 theta_target
    losses["contact_angle"] = compute_bc_loss(model, triad_data)
    
    # 3. 动态响应引导损失 (仅在降压区间 V_to < V_from 激活)
    if triad_data.is_step_down:
        # 强制符合 η(t_since) = η_start * exp(-t_since / 7.5ms)
        losses["eta_recovery"] = compute_recovery_guidance(
            model, 
            triad_data,
            tau_recovery=0.0075
        )
    
    # 4. 表面张力损失 (CSF 模型)
    # 强化降压过程中的界面回流动力
    losses["surface_tension"] = compute_surface_tension_loss(model, triad_data)
    
    return losses
```

---

## 6. 验证结果 (基于 v4.5 60,000 Epoch 训练)

### 典型响应指标

| 场景 | 跳变过程 | 响应时间 (90%) | 稳态精度 (MAE) |
| :--- | :--- | :--- | :--- |
| **快速升压** | 0V -> 30V | ~12.5 ms | < 0.015 |
| **快速降压** | 30V -> 0V | ~25.0 ms | < 0.02 |
| **中间态维持** | 10V -> 20V | ~8.0 ms | < 0.01 |

### 物理合理性

- ✅ **单调性**: 升压过程 $\partial \eta / \partial t_{since} > 0$；降压过程相反
- ✅ **0V 稳态**: 当 `V_to = 0` 且 `t_since > 40ms` 时，$\phi$ 场回归 0.5 (油墨平铺状态)
- ✅ **界面拓扑**: 升压时油墨中心变薄并向边缘推移；降压时从四周向中心平滑回流

---

## 7. 方案优势

- **统一性**: 无需维护两个独立的 PINN 模型，通过 Triad 输入自然过渡升压与降压。
- **物理鲁棒性**: 结合了纯物理 PDE 和半经验恢复模型，既保证了界面演化的物理合理性，又满足了宏观响应时间的准确性。
- **易扩展**: 该框架可直接扩展至多阶电压阶梯跳变（如 0V → 15V → 30V → 10V）。

---

## 8. 关键实现细节

### 8.1 降压后接触角恒定约束 (Triad 空间)

```python
def theta_constant_loss(model, triad_data, theta0=120.0):
    """降压后底面接触角恒定为 θ₀"""
    # 过滤出降压区间的数据点
    step_down_mask = triad_data.V_to < triad_data.V_from
    if not step_down_mask.any():
        return 0.0
    
    # 获取底面 z=0 的梯度
    grad_phi = compute_phi_gradient(model, triad_data.z0_points)
    
    # 接触角：cos(θ) = -∂φ/∂z / |∇φ|
    dphi_dz = grad_phi[:, 2]
    grad_mag = torch.norm(grad_phi, dim=1) + 1e-10
    
    cos_theta_pred = -dphi_dz / grad_mag
    cos_theta_target = torch.cos(torch.tensor(theta0 * np.pi / 180))
    
    # 只在界面附近 (phi 约为 0.5) 施加约束
    interface_mask = (phi > 0.1) & (phi < 0.9)
    return F.mse_loss(cos_theta_pred[interface_mask], cos_theta_target)
```

### 8.2 开口率恢复约束 (Triad 空间)

```python
def eta_recovery_constraint_loss(model, triad_data, tau_recovery=0.0075):
    """
    开口率恢复的软约束 (V_from -> V_to, t_since)
    """
    # 获取起始电压 V_from 的稳态开口率
    eta_start = model.aperture_model.get_steady_eta(triad_data.V_from)
    
    # 目标开口率按指数衰减
    eta_target = eta_start * torch.exp(-triad_data.t_since / tau_recovery)
    
    # 模型在当前 triad 状态下的预测开口率
    eta_pred = model.predict_aperture_ratio(triad_data)
    
    return F.mse_loss(eta_pred, eta_target)
```

### 8.3 降压恢复模型

```python
# 恢复模型: η(t) = η_start * exp(-t_since / τ_recovery)
tau_recovery = 0.0075  # 7.5ms (校准值)
```

---

## 9. 验证指标

### 升压过程

- 接触角误差 < 2°
- 开口率与 Stage 1 解析结果一致
- t90 响应时间 ≈ 11ms

### 降压过程

- 接触角瞬间恢复到 120° ± 2°
- 开口率恢复曲线符合 exp(-t/τ_recovery)
- 15ms 后开口率 < 20%
- 25ms 后开口率 < 5%

---

## 10. 完成状态

- [x] 6D Triad 架构实现 (`src/models/pinn_two_phase.py`)
- [x] 动态跳变数据采样 (`src/physics/data_generator.py`)
- [x] 降压引导损失 (Analytic Guidance Loss / `eta_recovery`)
- [x] 60,000 Epochs 三阶段渐进式训练验证
- [x] 方波响应自动化验证脚本 (`evaluate.py`)

---

## 🧪 使用验证工具

```bash
# 生成方波响应曲线 (0-30-0V)
python evaluate.py outputs/train/pinn_xxx --mode dynamic
```

**预期效果**:
- 升压: 油墨中心变薄并向边缘推移
- 降压: 油墨从四周向中心平滑回流，无断裂或孤岛现象

---

**关联文档**:
- 架构设计: [pinn_input_redesign.md](./pinn_input_redesign.md)
- 验证框架: [validation_framework.md](./validation_framework.md)
