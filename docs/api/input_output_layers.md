# 输入输出层 API

**最后更新**: 2026-02-04
**版本**: v4.5

## 概述

EFD-PINNs 的输入输出层定义了模型的数据接口，支持 6D Triad (x, y, z, V_from, V_to, t_since) 动态输入空间。

## TwoPhasePINN 输入输出

### 输入格式 (6D Triad)

```python
# 输入: (batch, 6)
# (x, y, z, V_from, V_to, t_since)
inputs = torch.tensor([
    [x, y, z, V_from, V_to, t_since],
    ...
])
```

| 维度 | 含义 | 范围 | 说明 |
|------|------|------|------|
| 0 | x 坐标 | [0, 174μm] | 归一化到 [0, 1] |
| 1 | y 坐标 | [0, 174μm] | 归一化到 [0, 1] |
| 2 | z 坐标 | [0, 20μm] | 归一化到 [0, 1] |
| 3 | V_from | 跳变前电压 | [0, 30V] | 归一化到 [0, 1] |
| 4 | V_to | 跳变后/当前电压 | [0, 30V] | 归一化到 [0, 1] |
| 5 | t_since | 跳变后经过的时间 | [0, 50ms] | 归一化到 [0, 1] |

**电压三元组语义**

- `V_from = V_to`：恒定电压工况（稳态或准稳态）
- `V_from < V_to`：升压过程（电润湿驱动，tau=5ms）
- `V_from > V_to`：降压过程（表面张力恢复，tau_recovery=7.5ms）

### 输出格式

```python
# 输出: (batch, 5)
# (u, v, w, p, phi)
outputs = model(inputs)
```

| 维度 | 含义 | 说明 |
|------|------|------|
| 0 | u | x 方向速度 (m/s) |
| 1 | v | y 方向速度 (m/s) |
| 2 | w | z 方向速度 (m/s) |
| 3 | p | 压力 (Pa) |
| 4 | phi | 体积分数 [0, 1] (0.5 为气液界面) |

## 归一化策略

模型内部自动处理输入归一化：

```python
# 归一化策略 (关键！)
x_norm = x / Lx           # 空间归一化 [0,1]
y_norm = y / Ly
z_norm = z / Lz
t_norm = t_since / t_max  # 时间归一化 [0,1]
V_from_norm = V_from / 30.0  # 电压归一化 [0,1]
V_to_norm = V_to / 30.0

# 详细参数说明请参见[物理理论与器件规格指南](../guides/physics_and_device_guide.md#physics-parameters)。
```

- **物理尺度**: $L \sim 10^{-4}m, t \sim 10^{-2}s$
- **归一化后**: 所有输入均在 $[0, 1]$ 区间内，有利于神经网络训练稳定性。
