# EFD-PINNs 项目迁移指南

本文档说明项目重构后的文件路径变化和代码迁移方法。

**最后更新**: 2025-12-08

---

## ✅ 迁移状态

项目已完成重构，所有核心代码已迁移到 `src/` 目录。

---

## 目录结构

### 源代码位置

| 模块 | 路径 |
|------|------|
| 模型定义 | `src/models/` |
| 预测器 | `src/predictors/` |
| 物理约束 | `src/physics/` |
| 训练相关 | `src/training/` |
| 工具函数 | `src/utils/` |
| 可视化 | `src/visualization/` |
| 求解器 | `src/solvers/` |

### 关键文件

| 文件 | 说明 |
|------|------|
| `src/predictors/hybrid_predictor.py` | Stage 1 混合预测器 |
| `src/predictors/pinn_aperture.py` | Stage 2 PINN 开口率预测器 |
| `src/models/pinn_two_phase.py` | 两相流 PINN 模型 |
| `src/models/aperture_model.py` | 开口率模型 |

---

## 导入语句

### 推荐方式

```python
# Stage 1: 接触角预测
from src.predictors import HybridPredictor

predictor = HybridPredictor()
theta = predictor.predict(voltage=30, time=0.005)

# Stage 2: 开口率预测
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=30, time=0.015)

# 两相流 PINN 模型
from src.models.pinn_two_phase import TwoPhasePINN, Trainer
```

---

## 配置文件

配置文件位于 `config/` 目录：

| 文件 | 说明 |
|------|------|
| `config/stage6_wall_effect.json` | Stage 1 最佳配置 |
| `config/stage1_config.json` | 阶段1初始训练 |

---

## 训练入口

| 脚本 | 说明 |
|------|------|
| `train_contact_angle.py` | Stage 1 训练 |
| `train_two_phase.py` | Stage 2 训练 |

```bash
# Stage 1 训练
python train_contact_angle.py --quick-run

# Stage 2 训练
python train_two_phase.py --epochs 30000
```

---

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v
```
