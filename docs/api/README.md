# EFD-PINNs API 参考文档

**最后更新**: 2025-12-08

## 概述

EFD-PINNs 是一个基于物理信息神经网络的电润湿显示动力学预测框架。

## 核心模块

### 1. 预测器 (`src/predictors/`)
- `HybridPredictor`: Stage 1 混合预测器（接触角预测）
- `PINNAperturePredictor`: Stage 2 PINN 开口率预测器

### 2. 模型 (`src/models/`)
- `TwoPhasePINN`: 两相流 PINN 模型
- `EnhancedApertureModel`: 开口率模型

### 3. 物理约束 (`src/physics/`)
- 物理约束计算
- 数据生成器

## 快速开始

```python
# Stage 1: 接触角预测
from src.predictors import HybridPredictor

predictor = HybridPredictor()
theta = predictor.predict(voltage=30, time=0.005)

# Stage 2: 开口率预测
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=30, time=0.015)
```

## 文档结构

- [核心模型](core_models.md) - 模型架构
- [物理约束](physics_constraints.md) - 物理方程实现
- [训练系统](training_system.md) - 训练流程

## 版本信息

- **当前版本**: 5.0.0
- **状态**: Stage 1 + Stage 2 完成
