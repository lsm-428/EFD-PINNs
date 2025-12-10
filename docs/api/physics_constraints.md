# 物理约束 API

**最后更新**: 2025-12-08

## 概述

EFD-PINNs 实现了电润湿显示的核心物理约束。

## 核心物理方程

### 1. Young-Lippmann 方程

描述接触角与电压的关系：

```
cos(θ) = cos(θ₀) + ε₀εᵣV²/(2γd)
```

**参数**:
- θ₀: 初始接触角 (120°)
- εᵣ: 相对介电常数 (4.0)
- γ: 表面张力 (0.072 N/m)
- d: 介电层厚度 (0.4 μm)

### 2. 二阶欠阻尼响应

描述接触角的动态响应：

```
θ(t) = θ_eq + (θ₀-θ_eq)·e^(-ζω₀t)·[cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
```

**参数**:
- τ: 时间常数 (5 ms)
- ζ: 阻尼比 (0.8)
- ω₀ = 1/τ: 自然频率
- ω_d = ω₀√(1-ζ²): 阻尼频率

### 3. 两相流方程

#### 连续性方程
```
∇·u = 0
```

#### VOF 方程
```
∂φ/∂t + u·∇φ = 0
```

#### Navier-Stokes 方程
```
ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st
```

## PhysicsLoss 类

**文件**: `src/models/pinn_two_phase.py`

### 主要方法

#### continuity_residual
```python
def continuity_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    """连续性方程残差：∇·u = 0"""
```

#### vof_residual
```python
def vof_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    """VOF 方程残差：∂φ/∂t + u·∇φ = 0"""
```

#### navier_stokes_residual
```python
def navier_stokes_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Navier-Stokes 方程残差"""
```

## 使用示例

```python
from src.predictors import HybridPredictor

predictor = HybridPredictor()

# Young-Lippmann 计算
for V in [0, 10, 20, 30]:
    theta = predictor.young_lippmann(V)
    print(f"V={V}V: θ={theta:.1f}°")
```

## 物理参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 初始接触角 | θ₀ | 120° | Teflon AF 1600X |
| 相对介电常数 | εᵣ | 4.0 | SU-8 |
| 表面张力 | γ | 0.072 N/m | 水-空气界面 |
| 介电层厚度 | d | 0.4 μm | SU-8 层 |
| 时间常数 | τ | 5 ms | 响应速度 |
| 阻尼比 | ζ | 0.8 | 欠阻尼 |
