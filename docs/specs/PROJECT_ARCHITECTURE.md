# EFD-PINNs 项目架构

**最后更新**: 2025-12-08

## 🎯 项目目标

使用物理信息神经网络 (PINN) 预测电润湿显示器件的动态响应。

## 📊 核心流程

```
Stage 1: 接触角预测
    V → θ(t)
    使用 Young-Lippmann + 二阶欠阻尼响应

Stage 2: 两相流 PINN
    θ(t) → φ(x,y,z,t) → η(t)
    使用 PINN 求解两相流方程
```

---

## 🏗️ 架构层次

### 第1层：预测器

```
src/predictors/
├── hybrid_predictor.py    # Stage 1 混合预测器
└── pinn_aperture.py       # Stage 2 PINN 开口率预测器
```

### 第2层：模型

```
src/models/
├── pinn_two_phase.py      # 两相流 PINN 模型
├── aperture_model.py      # 开口率模型
└── optimized_ewpinn.py    # 优化 PINN 模型
```

### 第3层：物理约束

```
src/physics/
├── constraints.py         # 物理约束
└── data_generator.py      # 数据生成器
```

### 第4层：训练

```
src/training/
├── components.py          # 训练组件
├── optimizer.py           # 优化器
└── tracker.py             # 训练追踪
```

---

## 🔄 数据流

### Stage 1: 接触角预测

```
输入: V (电压), t (时间)
    ↓
Young-Lippmann 方程
    cos(θ) = cos(θ₀) + ε₀εᵣV²/(2γd)
    ↓
二阶欠阻尼响应
    θ(t) = θ_eq + (θ₀-θ_eq)·e^(-ζω₀t)·[...]
    ↓
输出: θ(t) (接触角)
```

### Stage 2: 两相流 PINN

```
输入: (x, y, z, t, V)
    ↓
TwoPhasePINN 模型
    ├── phi_net: 预测 φ 场
    └── vel_net: 预测速度场
    ↓
物理约束
    ├── 连续性: ∇·u = 0
    ├── VOF: ∂φ/∂t + u·∇φ = 0
    └── N-S: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
    ↓
输出: (u, v, w, p, φ)
    ↓
积分计算开口率
    η = φ < 0.3 的面积比例
```

---

## 📁 关键文件

| 文件 | 说明 |
|------|------|
| `train_contact_angle.py` | Stage 1 训练入口 |
| `train_two_phase.py` | Stage 2 训练入口 |
| `src/predictors/hybrid_predictor.py` | 混合预测器 |
| `src/predictors/pinn_aperture.py` | PINN 开口率预测器 |
| `src/models/pinn_two_phase.py` | 两相流 PINN 模型 |
| `config/stage6_wall_effect.json` | Stage 1 最佳配置 |

---

## 🔗 模块依赖

```
train_two_phase.py
    └── src/models/pinn_two_phase.py
        ├── TwoPhasePINN (模型)
        ├── PhysicsLoss (物理损失)
        ├── DataGenerator (数据生成)
        │   └── src/predictors/hybrid_predictor.py
        └── Trainer (训练器)

src/predictors/pinn_aperture.py
    └── src/models/pinn_two_phase.py
```

---

## 🚀 快速开始

```bash
# Stage 1: 接触角预测
python train_contact_angle.py --quick-run

# Stage 2: 两相流 PINN
python train_two_phase.py --epochs 30000

# 可视化
python visualize_pinn_results.py

# 验证
python validate_pinn_physics.py
```

---

## 📊 性能指标

### Stage 1

| 指标 | 目标 | 实现 |
|------|------|------|
| 稳态精度 | <3° | 0.7° ✅ |
| 角度变化 | 33° | 30.1° ✅ |
| 超调 | <15% | 3.9% ✅ |
| 响应时间 | <30ms | 13ms ✅ |

### Stage 2

| 指标 | 状态 |
|------|------|
| V=0V 开口率 | 0% ✅ |
| V=30V 开口率 | ~61% ✅ |
| 单调递增 | ✅ |
