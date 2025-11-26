# EFD3D API 参考文档

## 概述

EFD3D (Enhanced Physics-Informed Neural Networks for 3D Fluid Dynamics) 是一个基于物理约束的深度学习框架，专门用于三维流体动力学模拟。本API文档详细描述了框架的核心组件、接口和使用方法。

## 核心模块

### 1. 模型架构 (Core Models)
- **EfficientEWPINN**: 高效增强型物理信息神经网络主模型
- **OptimizedEWPINN**: 优化版训练模型，支持配置文件加载

### 2. 输入输出层 (Input/Output Layers)
- **EWPINNInputLayer**: 62维物理特征输入处理层
- **EWPINNOutputLayer**: 24维物理输出特征处理层

### 3. 物理约束 (Physics Constraints)
- **PhysicsConstraints**: 物理约束计算核心类

### 4. 训练系统 (Training System)
- **MultiStageTrainer**: 多阶段训练器
- **EnhancedDataAugmenter**: 增强数据增强器

## 快速开始

### 安装依赖
```bash
pip install torch numpy matplotlib scipy
```

### 基础使用示例
```python
from ewp_pinn_optimized_architecture import EfficientEWPINN
from ewp_pinn_input_layer import EWPINNInputLayer
from ewp_pinn_output_layer import EWPINNOutputLayer
from ewp_pinn_physics import PhysicsConstraints

# 创建模型实例
input_layer = EWPINNInputLayer()
output_layer = EWPINNOutputLayer()
physics_constraints = PhysicsConstraints()
model = EfficientEWPINN(input_dim=62, output_dim=24)

# 设置实现阶段
input_layer.set_implementation_stage(3)
output_layer.set_implementation_stage(3)

# 前向传播
input_data = torch.randn(32, 62)
output = model(input_data)
```

## 文档结构

- [核心模型](core_models.md) - 主要神经网络架构
- [输入输出层](input_output_layers.md) - 数据预处理和后处理
- [物理约束](physics_constraints.md) - 物理方程实现
- [训练系统](training_system.md) - 训练流程和优化
- [示例与最佳实践](examples_and_best_practices.md) - 使用指南

## 版本信息

- **当前版本**: 1.0.0
- **框架类型**: 物理信息神经网络 (PINN)
- **主要应用**: 三维流体动力学模拟
- **支持功能**: 多物理场耦合、自适应训练、物理约束优化

## 技术支持

如有问题或建议，请参考项目文档或联系开发团队。