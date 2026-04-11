# LSTM 电压序列编码设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**对应代码**: `src/models/lstm_pinn/encoder.py`, `src/models/lstm_pinn/hybrid_model.py`

---

## 1. 设计概述

### 1.1 问题背景

Stage 2 的 TwoPhasePINN 使用单步三元组 `(V_from, V_to, t_since)`，无法捕获：

| 问题 | 现象 | 原因 |
|------|------|------|
| **路径依赖** | 0→30 vs 0→20→30 可能响应不同 | 无法感知历史 |
| **多次升降压** | 循环电压下的累积效应 | 单步无法建模时序 |
| **恢复滞后** | 降压后开口率恢复过程 | 无历史状态记忆 |

### 1.2 解决方案

使用 LSTM 编码完整的电压跳变序列：

```
Stage 2: 单步输入
输入: (x, y, z, V_from, V_to, t_since) ── 6D

Stage 3: 序列输入
输入: (x, y, z, t, voltage_seq) ── 空间 + 时间 + 序列
      voltage_seq: (V_from_1, V_to_1, dt_1), (V_from_2, V_to_2, dt_2), ...
```

---

## 2. 电压三元组格式

### 2.1 三元组定义

```
电压跳变序列由多个三元组组成:

三元组: (V_from, V_to, t_since)

参数说明:
- V_from:  跳变前电压 (V)，归一化到 [0, 1]
- V_to:    跳变后电压 (V)，归一化到 [0, 1]
- t_since: 该步持续时间 (s)，归一化到 [0, 1]
```

### 2.2 归一化

```python
# 电压归一化 (最大电压 30V)
V_from_norm = V_from / 30.0
V_to_norm = V_to / 30.0

# 时间归一化 (最大时间 50ms)
t_since_norm = t_since / 0.05
```

### 2.3 示例序列

```
场景: 0→20→30→20→0 的电压跳变序列

Step 1: (0, 20, 0.010)   # 0→20V，持续10ms
       归一化: (0.000, 0.667, 0.200)

Step 2: (20, 30, 0.005)  # 20→30V，持续5ms
       归一化: (0.667, 1.000, 0.100)

Step 3: (30, 20, 0.008)  # 30→20V，持续8ms
       归一化: (1.000, 0.667, 0.160)

Step 4: (20, 0, 0.003)   # 20→0V，持续3ms
       归一化: (0.667, 0.000, 0.060)

序列输入: (batch, seq_len=4, 3)
```

---

## 3. 物理场景编码

### 3.1 升压过程

```
电压: 0 → 30V

序列: [(0, 30, 0.050)]  # 单步升压
或:   [(0, 15, 0.025), (15, 30, 0.025)]  # 阶梯升压

特点:
- V_from < V_to
- 接触角单调下降
- 开口率按抛物线响应
```

### 3.2 降压过程

```
电压: 30V → 0

序列: [(30, 0, 0.050)]

特点:
- V_from > V_to
- 接触角瞬间恢复
- 开口率按指数曲线恢复
```

### 3.3 多次升降压

```
电压: 0→30→0→30→0

序列: [(0, 30, 0.050),
       (30, 0, 0.050),
       (0, 30, 0.050),
       (30, 0, 0.050)]

LSTM 应该学习:
- 每次升压的累积效应
- 每次降压的恢复过程
- 循环后的稳定状态
```

### 3.4 阶梯升压

```
电压: 0→10→20→30V

序列: [(0, 10, 0.017),
       (10, 20, 0.017),
       (20, 30, 0.017)]

与直接升压 (0, 30, 0.050) 的区别:
- 路径不同，响应可能不同
- 液体惯性的累积效应
```

---

## 4. LSTM 编码器设计

### 4.1 架构

```python
class VoltageEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 3,      # (V_from, V_to, t_since)
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        """
        LSTM 电压编码器
        """
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 双向 LSTM 需要投影层
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
```

### 4.2 前向传播

```python
def forward(self, sequence, initial_state=None):
    """
    Args:
        sequence: (batch, seq_len, 3) - 电压序列
        initial_state: 可选的初始隐状态 (h_0, c_0)
    
    Returns:
        hidden: (batch, hidden_dim) - 最终隐状态
        all_hidden: (batch, seq_len, hidden_dim) - 所有时刻隐状态
    """
    # LSTM 前向传播
    output, (h_n, c_n) = self.lstm(sequence, initial_state)
    
    # 取最后一层隐状态
    hidden = h_n[-1]
    
    # 层归一化
    hidden = self.layer_norm(hidden)
    
    return hidden, output
```

### 4.3 配置参数

```python
DEFAULT_LSTM_CONFIG = {
    "input_dim": 3,           # (V_from, V_to, t_since)
    "hidden_dim": 128,        # 隐状态维度
    "num_layers": 2,          # LSTM 层数
    "dropout": 0.1,           # Dropout
    "bidirectional": False    # 是否双向
}
```

---

## 5. 等效电压映射

### 5.1 设计动机

TwoPhasePINN 的 φ 网络期望输入 `(x, y, z, V_from, V_to, t_since)`，需要将 LSTM 的隐状态映射为等效电压。

### 5.2 映射头

```python
class LSTMHybridPINN(nn.Module):
    def __init__(self, pretrained_pinn, config):
        # LSTM 编码器
        self.lstm_encoder = VoltageEncoder(
            input_dim=3,
            hidden_dim=128,
            num_layers=2
        )
        
        # 等效电压映射头
        self.v_eff_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        self.v_prev_eff_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
```

### 5.3 语义对应

```python
# LSTM 输出
V_eff = self.v_eff_head(hidden)        # 等效 V_to
V_prev_eff = self.v_prev_eff_head(hidden)  # 等效 V_from

# 传递给 TwoPhasePINN
phi_input = torch.cat([
    x_norm, y_norm, z_norm,
    V_prev_eff, V_eff, t_norm  # (V_from, V_to, t_since)
], dim=-1)
```

---

## 6. 与 Stage 2 的关系

### 6.1 复用设计

```
Stage 2: TwoPhasePINN
输入: (x, y, z, V_from, V_to, t_since) ── 6D
      │
      ▼
┌───────────────────────────────┐
│                               │
│  φ 网络 (6→64→64→64→32→1)     │
│  速度网络 (7→64→64→32→4)      │
│                               │
└───────────────────────────────┘

Stage 3: LSTM-PINN
输入: (x, y, z, t, voltage_seq) ── 序列
      │
      ▼
┌───────────────────────────────┐
│  LSTM Encoder                 │
│  (seq_len=4, hidden_dim=128)  │
└───────────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│  等效电压映射                  │
│  V_eff, V_prev_eff            │
└───────────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│  TwoPhasePINN (复用)          │
│  φ 网络 + 速度网络             │
└───────────────────────────────┘
```

### 6.2 权重复用

```python
class LSTMHybridPINN(nn.Module):
    def __init__(self, pretrained_pinn, freeze_pinn=True):
        # 复用预训练的 TwoPhasePINN
        self.phi_net = pretrained_pinn.phi_net
        self.vel_net = pretrained_pinn.vel_net
        
        # 可选：冻结 PINN 权重
        if freeze_pinn:
            for param in self.phi_net.parameters():
                param.requires_grad = False
            for param in self.vel_net.parameters():
                param.requires_grad = False
```

---

## 7. 训练策略

### 7.1 两阶段训练

```python
TRAINING_CONFIG = {
    "stage1_epochs": 2000,    # 冻结 PINN，只训练 LSTM
    "stage2_epochs": 5000,    # 端到端微调
}
```

| 阶段 | 冻结策略 | 学习率 |
|------|----------|--------|
| **Stage 1** | φ_net, vel_net 冻结 | LSTM: 1e-3 |
| **Stage 2** | 全部可训练 | 整体: 1e-4 |

### 7.2 损失函数

```python
# Stage 1: 仅数据损失
loss = interface_loss * 500.0

# Stage 2: 数据 + 物理损失
loss = (
    interface_loss * 500.0 +
    continuity_loss * 0.1 +
    vof_loss * 0.1 +
    volume_conservation_loss * 10.0
)
```

---

## 8. 局限性

### 8.1 当前局限

| 局限 | 说明 |
|------|------|
| **序列长度固定** | 需要预先指定最大序列长度 |
| **计算成本** | LSTM 增加额外计算开销 |
| **初始化状态** | 初始隐状态设为零 |

### 8.2 未来改进

| 改进方向 | 预期收益 |
|----------|----------|
| **Transformer** | 更好的长序列处理 |
| **注意力机制** | 可解释性提升 |
| **可学习初始化** | 更好的初始状态 |

---

## 9. 验证场景

### 9.1 测试用例

| 场景 | 预期结果 |
|------|----------|
| 单步升压 0→30V | 与 TwoPhasePINN 结果一致 |
| 阶梯升压 0→20→30V | 正确学习路径依赖 |
| 多次升降压 | 累积效应正确 |
| 降压恢复 | 恢复过程合理 |

### 9.2 验证命令

```bash
# 运行 LSTM-PINN 测试
python -m pytest tests/test_lstm_pinn_properties.py -v

# 验证等效电压输出
python -c "
from src.models.lstm_pinn import LSTMHybridPINN
import torch

# 创建测试序列
seq = torch.randn(4, 10, 3)  # (batch, seq_len, 3)
model = LSTMHybridPINN()
hidden = model.lstm_encoder(seq)
print(f'Hidden shape: {hidden.shape}')
"
```

---

## 10. 相关文档

| 文档 | 说明 |
|------|------|
| [lstm_pinn_stage3_design.md](./lstm_pinn_stage3_design.md) | Stage 3 完整架构 |
| [pinn_input_redesign.md](./pinn_input_redesign.md) | 6D Triad 输入 |
| [sampling_strategy.md](./sampling_strategy.md) | 序列数据生成 |

---

## 11. 设计演进：多步序列的正确理解 (2026-03-13)

### 11.1 问题发现

**用户的方波例子揭示了多步序列的正确物理意义：**

```
方波序列: 0→20V (5ms) → 30V (5ms)

Step 1: 0→20V, 5ms
  - 不是真实的20V开口率(66%)
  - 油墨刚刚响应，可能只有30%的开口率
  - 这是"等效20V持续5ms"的响应

Step 2: 20→30V, 5ms  
  - 是在30%的基础上继续响应
  - 相当于30V等效电压持续了5ms
  - 开口率继续增加，但可能比"30V持续10ms"的开口率小
```

### 11.2 核心洞察

**多步序列的关键特性：**

1. **每一步都在上一步状态的基础上继续响应**
   - 不是独立的单步预测
   - 状态是累积的

2. **存在累积效应**
   - 路径依赖：0→20→30 vs 直接 0→30
   - 时间累积：总响应时间影响最终状态

3. **当前训练数据的致命缺陷**
   ```python
   # 错误：只看最后一步
   last_V_from, last_V_to, last_t_since = transitions[-1]
   eta = get_opening_rate(last_V_to, last_t_since)
   ```

### 11.3 正确的多步目标计算

**应该是逐步累积的状态：**

```python
# 正确逻辑
state = initial_state  # 初始油墨状态
for V_from, V_to, t_since in voltage_sequence:
    state = PINN(state, V_from, V_to, t_since)  # 在上一状态基础上响应
final_phi = state.phi  # 最终状态
```

### 11.4 架构设计重新思考

**LSTM 的正确职责：**

```
错误设计:
voltage_seq → LSTM → (V_eff, V_prev) → PINN → φ
                         ↓
                   输出"等效单步电压"
                   破坏了多步累积意义

正确设计:
voltage_seq → LSTM → 累积状态编码 → φ
                   ↑
              学习多步累积效应
```

### 11.5 设计方向

| 方向 | 描述 | 实现方式 |
|------|------|----------|
| **A: 状态累积** | LSTM 输出每步的状态增量 | `φ_final = φ_0 + Σ LSTM_step` |
| **B: 修正因子** | LSTM 输出对单步预测的修正 | `φ = PINN单步 × LSTM修正` |
| **C: 直接预测** | LSTM 直接输出最终 φ | 需要正确的训练数据 |

### 11.6 待解决问题

1. **训练数据如何正确生成多步目标？**
   - 需要用 PINN 逐步模拟多步序列
   - 或者使用实验数据

2. **LSTM 输出应该是什么？**
   - 直接 φ？
   - 状态修正因子？
   - 隐状态传递给下游网络？

3. **如何验证多步累积效应？**
   - 设计对比实验：不同路径的最终 φ

---

**最后更新**: 2026-03-13
