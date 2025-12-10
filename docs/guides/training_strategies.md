# 训练策略详解

**最后更新**: 2025-12-08

## 训练架构概览

EFD-PINNs 采用两阶段训练策略：

```
Stage 1: 接触角预测 (解析公式)
    V → θ(t)
    
Stage 2: 两相流 PINN
    θ(t) → φ(x,y,z,t) → η(t)
```

## Stage 1: 接触角预测

### 核心原理

使用解析公式，无需训练：

1. **Young-Lippmann 方程** (稳态)
```
cos(θ) = cos(θ₀) + ε₀εᵣV²/(2γd)
```

2. **二阶欠阻尼响应** (动态)
```
θ(t) = θ_eq + (θ₀-θ_eq)·e^(-ζω₀t)·[cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
```

### 使用方法

```python
from src.predictors import HybridPredictor

predictor = HybridPredictor()
theta = predictor.predict(voltage=30, time=0.005)
```

## Stage 2: 两相流 PINN

### 渐进式训练策略

训练分为三个阶段：

#### 阶段 1: 纯数据学习 (0 - 5000 epochs)
- 物理约束权重: 0
- 专注于学习数据分布
- 学习率: 5e-4

#### 阶段 2: 引入物理约束 (5000 - 15000 epochs)
- 逐步引入连续性和 VOF 约束
- 使用平滑过渡（sigmoid 曲线）
- 学习率: 自适应调整

#### 阶段 3: 完整物理约束 (15000+ epochs)
- 完整的物理约束
- 包括 Navier-Stokes 和表面张力
- 微调阶段

### 训练命令

```bash
# 快速测试
python train_two_phase.py --epochs 1000

# 完整训练
python train_two_phase.py --epochs 30000
```

## 损失函数设计

### 数据损失

1. **接触角边界条件损失**
   - 底面 z=0 处的 φ 梯度由 θ(t) 决定
   - 权重: 500.0

2. **初始条件损失**
   - t=0 时油墨均匀铺在底部
   - 权重: 100.0

3. **壁面边界条件损失**
   - 无滑移条件
   - 权重: 50.0

### 物理损失

1. **连续性方程**: ∇·u = 0
2. **VOF 方程**: ∂φ/∂t + u·∇φ = 0
3. **Navier-Stokes**: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u

## 关键改进

### 1. Stage 1 → Stage 2 集成

Stage 2 使用 Stage 1 的接触角预测作为边界条件：

```python
# 在 DataGenerator 中
theta = self.contact_angle_predictor.predict(V, t)
```

### 2. 低电压约束

V < 10V 时强制 φ ≈ 0.5（油墨平铺）：

```python
# 早期时间约束
target_phi_early = 0.5 * torch.ones_like(phi_early)
losses["early_time"] = F.mse_loss(phi_early, target_phi_early) * 500.0
```

### 3. 开口率阈值

使用 φ < 0.3 而非 φ < 0.5 定义透明区域。

## 最佳实践

1. **渐进式训练**: 先数据后物理
2. **权重平衡**: 数据损失和物理损失平衡
3. **学习率调度**: 使用 ReduceLROnPlateau
4. **梯度裁剪**: 防止梯度爆炸

## 训练监控

```python
# 训练历史保存在输出目录
outputs_pinn_YYYYMMDD_HHMMSS/
├── best_model.pth
├── config.json
└── training_history.json
```
