# EFD-PINNs API 参考文档

**最后更新**: 2026-03-13
**版本**: v4.5

## 概述

EFD-PINNs 是一个基于物理信息神经网络 (PINN) 的电润湿显示 (EWD) 动力学预测框架，采用先进的 6D Triad 输入架构和三阶段渐进式训练策略。

### 🆕 当前配置

**v4.5 标准配置**:
- ✅ 网络架构: `hidden_phi: [128, 128, 64, 32]`
- ✅ 训练配置: `epochs: 60000`, `stage1_epochs: 1500`, `stage2_epochs: 4000`
- ✅ 损失权重: `continuity: 0.5`, `vof: 0.5`, `sharpening: 1.0`
- ✅ 推荐配置文件: `config/v4.5-standard.json`

---

## 核心模块索引

### 1. 模型 (`src/models/`)

| 模块 | 说明 | 文档 |
|------|------|------|
| **TwoPhasePINN** | 核心 6D PINN 模型，预测三维两相流场 (u, v, w, p, φ) | [core_models.md](core_models.md#twophasepinn) |
| **EnhancedApertureModel** | Stage 1 物理基准模型，提供校准后的接触角与开口率目标 | [core_models.md](core_models.md#enhancedaperturemodel) |

### 2. 物理系统 (`src/physics/`)

| 模块 | 说明 | 文档 |
|------|------|------|
| **PhysicsConstraints** | 物理方程的"唯一真理源"，实现 N-S、VOF、连续性方程及电润湿力 | [physics_constraints.md](physics_constraints.md) |
| **PhysicsLoss** | 训练损失计算的适配层，负责残差聚合与权重管理 | [training_system.md](training_system.md#physicsloss) |

### 3. 预测器 (`src/predictors/`)

| 模块 | 说明 | 文档 |
|------|------|------|
| **PINNAperturePredictor** | 面向用户的预测接口，封装模型加载与后处理 | [core_models.md](core_models.md#pinnaperturepredictor) |

### 4. 训练系统 (`src/models/pinn_two_phase.py`)

| 模块 | 说明 | 文档 |
|------|------|------|
| **Trainer** | 管理训练循环、数据采样与模型保存 | [training_system.md](training_system.md#trainer) |
| **DataGenerator** | 生成 6D Triad 训练数据 | [training_system.md](training_system.md#datagenerator) |

---

## 快速调用示例

### 1. 使用训练好的 PINN 进行预测

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

# 自动加载最新模型
predictor = PINNAperturePredictor()

# 预测从 0V 跳变到 20V 后 15ms 的开口率
eta = predictor.predict(voltage=20, time=0.015)
print(f"Predicted Aperture: {eta:.2%}")

# 预测完整动态响应
times = np.linspace(0, 0.03, 100)
etas = [predictor.predict(voltage=20, time=t) for t in times]
```

### 2. 获取 3D 物理场

```python
# 获取完整的 3D 物理场
fields = predictor.predict_full_field(
    voltage=20,
    time=0.015
)

# fields 包含:
# - 'u', 'v', 'w': 速度场 (m/s)
# - 'p': 压力场 (Pa)
# - 'phi': 相场 (0=极性液体, 1=油墨)
```

### 3. 使用命令行工具查询知识库

```bash
cd /home/scnu/Gitee/EFD3D/
source .venv/bin/activate
```

---

## 📚 完整文档导航

| 文档 | 内容 |
|------|------|
| [core_models.md](core_models.md) | 核心模型详解 (TwoPhasePINN, EnhancedApertureModel) |
| [physics_constraints.md](physics_constraints.md) | 物理约束与方程 (N-S, VOF, 连续性) |
| [training_system.md](training_system.md) | 训练系统与采样策略 |
| [input_output_layers.md](input_output_layers.md) | 6D Triad 输入输出定义 |
| [examples_and_best_practices.md](examples_and_best_practices.md) | 示例与最佳实践 |

---

## 📊 版本信息

**当前版本**: v4.5 (2026-03-13)

**关键特性**:
- ✅ 6D Triad 输入: `(x, y, z, V_from, V_to, t_since)`
- ✅ 界面锐化损失: `L_sharp = λ·φ(1-φ)`
- ✅ 三阶段训练: 60,000 epochs 渐进式训练
- ✅ 网络架构: `phi_net: [128,128,64,32], vel_net: [64,64,32]`

**配置文件**: [config/v4.5-standard.json](../../config/v4.5-standard.json)

**训练状态**: ✅ 可训练 (Ready)

---

## 🔗 相关资源

- **项目核心文档**: [../../CLAUDE.md](../../CLAUDE.md)
- **训练报告**: [../research/TRAINING_REPORTS.md](../research/TRAINING_REPORTS.md)
- **知识库导航**: [../../CLAUDE.md](../../CLAUDE.md)
- **CHANGELOG**: [../CHANGELOG.md](../CHANGELOG.md)

