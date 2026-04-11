# Stage 3: LSTM-PINN 混合架构设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**对应代码**: `src/models/lstm_pinn/hybrid_model.py`

---

## 1. 设计概述

Stage 3 (LSTM-PINN) 是 EFD3D 架构的第三阶段，在 Stage 2 TwoPhasePINN 的基础上添加 LSTM 编码器来处理**任意长度的电压跳变序列**。

### 核心创新

| 创新点 | 说明 |
|--------|------|
| **电压序列编码** | LSTM 编码 `(V_from, V_to, t_since)` 三元组序列 |
| **历史累积效应** | 学习升压/降压的路径依赖性 |
| **预训练复用** | 复用 TwoPhasePINN 的 φ 网络和速度网络 |

---

## 2. 架构设计

### 2.1 整体架构

```
输入层
├── spatial_coords: (batch, 3) ─── (x, y, z) 空间坐标
├── t: (batch, 1) ──────────────── t_since 当前相对时间
└── voltage_seq: (batch, seq_len, 3) ── 电压跳变序列
    ├── [:, :, 0] = V_from ─────── 跳变前电压
    ├── [:, :, 1] = V_to ───────── 跳变后电压
    └── [:, :, 2] = t_since ────── 该步持续时间

┌─────────────────────────────────────────────────────────────┐
│  LSTM Encoder                                               │
│  ├── input_dim: 3 (三元组)                                  │
│  ├── hidden_dim: 128                                        │
│  ├── num_layers: 2                                          │
│  └── dropout: 0.0                                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              hidden_state: (batch, 128)
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐                 ┌─────────────────────┐
│ V_eff Head    │                 │ V_prev_eff Head     │
│ (Linear→32→1) │                 │ (Linear→32→1)       │
│ Sigmoid 输出  │                 │ Sigmoid 输出        │
│ [0, 1]        │                 │ [0, 1]              │
└───────────────┘                 └─────────────────────┘
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
              (V_from_eff, V_to_eff): (batch, 2)
                        │
    ┌───────────────────┴───────────────────┐
    │                                       │
    ▼                                       ▼
┌───────────────┐                 ┌─────────────────────┐
│ φ 网络        │                 │ 速度网络            │
│ 输入: 6       │                 │ 输入: 7             │
│ 输出: 1       │                 │ 输出: 4             │
│ hidden:       │                 │ hidden:             │
│ [64,64,64,32] │                 │ [64,64,32]          │
└───────────────┘                 └─────────────────────┘
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
              输出: (u, v, w, p, φ) ── (batch, 5)
```

### 2.2 电压三元组格式

```
电压跳变序列示例: 0→20→30→20→0

Step 1: (0, 20, 0.010)   # 0→20V，持续10ms
Step 2: (20, 30, 0.005)  # 20→30V，持续5ms
Step 3: (30, 20, 0.008)  # 30→20V，持续8ms
Step 4: (20, 0, 0.003)   # 20→0V，持续3ms（当前时刻）

电压序列输入: (batch, 4, 3)
├── batch: 批量大小
├── seq_len: 4（序列长度）
└── 3: (V_from, V_to, t_since)
```

### 2.3 等效电压语义

```python
# LSTM 输出两个等效电压，语义上对应：
V_eff = V_to      # 当前等效电压（当前跳变后的电压）
V_prev_eff = V_from  # 历史等效电压（跳变前的电压）

# 这样设计是为了让 TwoPhasePINN 的 φ 网络可以直接理解
# φ_net 输入格式: (x, y, z, V_from, V_to, t_since)
```

---

## 3. 训练策略

### 3.1 两阶段训练

```python
# 阶段1: 冻结 PINN，只训练 LSTM
stage1_epochs = 2000
freeze_pinn = True

# 阶段2: 端到端微调
stage2_epochs = 5000
freeze_pinn = False  # 解冻 PINN
```

### 3.2 训练流程

| 阶段 | 冻结策略 | 目的 |
|------|----------|------|
| **Stage 1** | φ_net, vel_net 冻结 | 让 LSTM 学习编码电压历史 |
| **Stage 2** | 全部可训练 | 端到端微调，恢复物理精度 |

### 3.3 损失函数

```python
# Stage 1 损失（仅数据拟合）
loss = interface_loss * 500.0

# Stage 2 损失（数据 + 物理）
loss = (
    interface_loss * 500.0 +
    continuity_loss * 0.1 +
    vof_loss * 0.1 +
    volume_conservation_loss * 10.0
)
```

---

## 4. 与 Stage 2 的关系

### 4.1 复用设计

```python
class LSTMHybridPINN(nn.Module):
    def __init__(self, pretrained_pinn=None, ...):
        # 复用预训练的 TwoPhasePINN 网络
        if pretrained_pinn is not None:
            self.phi_net = pretrained_pinn.phi_net
            self.vel_net = pretrained_pinn.vel_net
```

### 4.2 输入格式对比

| 模型 | 输入格式 | 说明 |
|------|----------|------|
| **TwoPhasePINN** | `(x, y, z, V_from, V_to, t_since)` | 单步三元组 |
| **LSTM-PINN** | `(x, y, z, t, voltage_seq)` | 序列输入 |

### 4.3 物理意义对比

| 模型 | 电压处理 | 路径依赖 |
|------|----------|----------|
| **TwoPhasePINN** | 瞬时值 `(V, V_prev)` | ❌ 不支持 |
| **LSTM-PINN** | 完整历史序列 | ✅ 支持 |

---

## 5. 物理场景示例

### 5.1 升压路径依赖

```
场景: 0→30V 的两种不同路径

路径 A: 0 → 30V (直接升压)
  序列: [(0, 30, 0.050)]

路径 B: 0 → 20V → 30V (阶梯升压)
  序列: [(0, 20, 0.025), (20, 30, 0.025)]

预期: 两种路径可能产生不同的响应（由于液体的惯性）
```

### 5.2 降压恢复

```
场景: 30V → 0V 的恢复过程

序列: [(30, 0, 0.050)]
恢复过程:
- t=0ms: 开口率 ~87%
- t=5ms: 开口率 ~60%
- t=10ms: 开口率 ~40%
- t=50ms: 开口率 ~0%（完全恢复）
```

### 5.3 多次升降压

```
场景: 0→30→0→30→0 循环

序列: [(0, 30, 0.050), (30, 0, 0.050), (0, 30, 0.050), (30, 0, 0.050)]

预期: LSTM 应该学习到循环的累积效应
```

---

## 6. 配置参数

### 6.1 LSTM 配置

```json
{
  "lstm": {
    "input_dim": 3,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.0
  },
  "phi_decoder": {
    "spatial_dim": 3,
    "hidden_layers": [128, 64, 32],
    "activation": "tanh"
  },
  "velocity_decoder": {
    "enabled": false
  }
}
```

### 6.2 训练配置

```json
{
  "training": {
    "stage1_epochs": 2000,
    "stage2_epochs": 5000,
    "learning_rate": 1e-3
  }
}
```

---

## 7. 代码实现

### 7.1 前向传播

```python
def forward(self, spatial_coords, t, voltage_seq):
    # 1. LSTM 编码电压历史
    hidden, _ = self.lstm_encoder(voltage_seq)
    
    # 2. 输出等效电压
    V_eff = self.v_eff_head(hidden)
    V_prev_eff = self.v_prev_eff_head(hidden)
    
    # 3. φ 网络预测
    phi_input = torch.cat([
        x_norm, y_norm, z_norm,
        V_prev_eff, V_eff, t_norm
    ], dim=-1)
    phi = torch.sigmoid(self.phi_net(phi_input))
    
    # 4. 速度网络预测
    vel_input = torch.cat([
        x_norm, y_norm, z_norm,
        V_prev_eff, V_eff, t_norm, phi
    ], dim=-1)
    vel_out = self.vel_net(vel_input)
    
    return torch.cat([vel_out, phi], dim=-1)
```

---

## 8. 验证与测试

### 8.1 相关测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_lstm_pinn_properties.py` | LSTM-PINN 属性测试 |
| `tests/test_hybrid_predictor.py` | 混合预测器测试 |

### 8.2 验证命令

```bash
# 运行 LSTM-PINN 测试
python -m pytest tests/test_lstm_pinn_properties.py -v

# 运行混合预测器测试
python -m pytest tests/test_hybrid_predictor.py -v
```

---

## 9. 局限性与未来改进

### 9.1 当前局限

| 局限 | 说明 |
|------|------|
| 序列长度固定 | 需要预先指定最大序列长度 |
| 计算成本 | LSTM 增加了额外的计算开销 |

### 9.2 未来改进

| 改进方向 | 预期收益 |
|----------|----------|
| Transformer 替代 LSTM | 更好的长序列处理能力 |
| 注意力机制 | 可解释性提升 |

---

## 10. 相关文档

| 文档 | 说明 |
|------|------|
| [pinn_input_redesign.md](./pinn_input_redesign.md) | 6D Triad 输入格式 |
| [dynamic_response.md](./dynamic_response.md) | 动态电压响应 |
| [sampling_strategy.md](./sampling_strategy.md) | 采样策略 |
| [loss_function_design.md](./loss_function_design.md) | 损失函数设计 |

---

**最后更新**: 2026-02-04
