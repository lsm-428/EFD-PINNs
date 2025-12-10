# 输入输出层 API

**最后更新**: 2025-12-08

## 概述

EFD-PINNs 的输入输出层定义了模型的数据接口。

## TwoPhasePINN 输入输出

### 输入格式

```python
# 输入: (batch, 5)
# (x, y, z, t, V)
inputs = torch.tensor([
    [x, y, z, t, V],  # 空间坐标 + 时间 + 电压
    ...
])
```

| 维度 | 含义 | 范围 |
|------|------|------|
| x | x 坐标 | [0, 174μm] |
| y | y 坐标 | [0, 174μm] |
| z | z 坐标 | [0, 20μm] |
| t | 时间 | [0, 20ms] |
| V | 电压 | [0, 30V] |

### 输出格式

```python
# 输出: (batch, 5)
# (u, v, w, p, phi)
outputs = model(inputs)
```

| 维度 | 含义 | 说明 |
|------|------|------|
| u | x 方向速度 | m/s |
| v | y 方向速度 | m/s |
| w | z 方向速度 | m/s |
| p | 压力 | Pa |
| phi | 体积分数 | [0, 1] |

## HybridPredictor 输入输出

### 输入

```python
predictor.predict(
    voltage=30,      # 电压 (V)
    time=0.005,      # 时间 (s)
    V_initial=0.0,   # 初始电压 (V)
    t_step=0.0       # 阶跃时间 (s)
)
```

### 输出

```python
theta = predictor.predict(...)  # 接触角 (度)
```

## PINNAperturePredictor 输入输出

### 输入

```python
predictor.predict(
    voltage=30,      # 电压 (V)
    time=0.015,      # 时间 (s)
    n_points=100     # 采样点数
)
```

### 输出

```python
eta = predictor.predict(...)  # 开口率 [0, 1]
phi_field = predictor.predict_phi_field(...)  # φ 场 (n_points, n_points)
```

## 归一化

### 输入归一化

```python
x_norm = x / Lx  # [0, 1]
y_norm = y / Ly  # [0, 1]
z_norm = z / Lz  # [0, 1]
t_norm = t / t_max  # [0, 1]
V_norm = V / 30.0  # [0, 1]
```

### 输出范围

| 输出 | 范围 | 说明 |
|------|------|------|
| phi | [0, 1] | sigmoid 激活 |
| u, v, w | 无限制 | 速度场 |
| p | 无限制 | 压力场 |
