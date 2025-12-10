# 训练系统 API

**最后更新**: 2025-12-08

## 概述

EFD-PINNs 采用两阶段训练策略：
- Stage 1: 接触角预测（解析公式，无需训练）
- Stage 2: 两相流 PINN 训练

## 训练入口

### Stage 1: 接触角训练

```bash
# 快速测试
python train_contact_angle.py --quick-run

# 标准训练
python train_contact_angle.py --config config/stage6_wall_effect.json --epochs 3000
```

### Stage 2: 两相流 PINN 训练

```bash
# 快速测试
python train_two_phase.py --epochs 1000

# 完整训练
python train_two_phase.py --epochs 30000
```

## Trainer 类

**文件**: `src/models/pinn_two_phase.py`

### 类定义
```python
class Trainer:
    """两相流 PINN 训练器"""
```

### 构造函数
```python
def __init__(self, config: Dict[str, Any] = None):
```

### 主要方法

#### train
```python
def train(self):
    """执行训练"""
```

#### get_physics_weights
```python
def get_physics_weights(self, epoch: int) -> Dict[str, float]:
    """根据训练阶段返回物理损失权重"""
```

## 渐进式训练策略

训练分为三个阶段：

### 阶段 1: 纯数据学习 (0 - 5000 epochs)
- 物理约束权重: 0
- 专注于学习数据分布

### 阶段 2: 引入物理约束 (5000 - 15000 epochs)
- 逐步引入连续性和 VOF 约束
- 使用平滑过渡

### 阶段 3: 完整物理约束 (15000+ epochs)
- 完整的物理约束
- 包括 Navier-Stokes 和表面张力

## 损失函数

### 数据损失
- 接触角边界条件损失
- 初始条件损失
- 壁面边界条件损失

### 物理损失
- 连续性方程残差
- VOF 方程残差
- Navier-Stokes 残差
- 表面张力残差

## 配置示例

```python
DEFAULT_CONFIG = {
    "model": {
        "hidden_phi": [64, 64, 64, 32],
        "hidden_vel": [64, 64, 32],
    },
    "training": {
        "epochs": 30000,
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "stage1_epochs": 5000,
        "stage2_epochs": 15000,
    },
    "physics": {
        "interface_weight": 500.0,
        "ic_weight": 100.0,
        "bc_weight": 50.0,
        "continuity_weight": 0.5,
        "vof_weight": 0.5,
    },
}
```

## 使用示例

```python
from src.models.pinn_two_phase import Trainer, DEFAULT_CONFIG

# 创建训练器
trainer = Trainer(DEFAULT_CONFIG)

# 执行训练
trainer.train()
```
