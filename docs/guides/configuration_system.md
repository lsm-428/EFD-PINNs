# 配置系统详解

**最后更新**: 2025-12-08

## 配置文件位置

配置文件位于 `config/` 目录：

```
config/
├── stage6_wall_effect.json    # Stage 1 最佳配置
├── stage1_config.json         # 阶段1初始训练
├── stage2_optimized.json      # 阶段2优化
└── stage7_fast_dynamics.json  # 快速动态
```

## 配置结构

### Stage 1 配置示例

```json
{
  "metadata": {
    "stage": "stage6",
    "description": "像素墙效应配置"
  },
  "model": {
    "input_dim": 62,
    "output_dim": 24,
    "hidden_dims": [256, 128, 64],
    "activation": "gelu"
  },
  "training": {
    "epochs": 3000,
    "batch_size": 64,
    "learning_rate": 0.001
  },
  "materials": {
    "theta0": 120.0,
    "theta_wall": 70.0,
    "epsilon_r": 4.0,
    "gamma": 0.072,
    "dielectric_thickness": 4e-7
  },
  "data": {
    "num_samples": 15000,
    "dynamics_params": {
      "tau": 0.005,
      "zeta": 0.8
    }
  }
}
```

### Stage 2 配置

Stage 2 使用 `src/models/pinn_two_phase.py` 中的 `DEFAULT_CONFIG`:

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
    "data": {
        "n_interface": 100000,
        "n_initial": 10000,
        "n_boundary": 10000,
        "n_domain": 20000,
        "voltages": [0, 5, 10, 15, 20, 25, 30],
        "times": 30,
    },
}
```

## 物理参数

### 材料参数

| 参数 | 键名 | 默认值 | 说明 |
|------|------|--------|------|
| 初始接触角 | `theta0` | 120° | Teflon AF 1600X |
| 相对介电常数 | `epsilon_r` | 4.0 | SU-8 |
| 表面张力 | `gamma` | 0.072 N/m | 水-空气界面 |
| 介电层厚度 | `dielectric_thickness` | 0.4 μm | SU-8 层 |

### 动力学参数

| 参数 | 键名 | 默认值 | 说明 |
|------|------|--------|------|
| 时间常数 | `tau` | 5 ms | 响应速度 |
| 阻尼比 | `zeta` | 0.8 | 欠阻尼 |

### 几何参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 像素尺寸 | 174×174 μm | 电极尺寸 |
| 围堰高度 | 20 μm | 油墨层高度 |
| 油墨层厚度 | 3 μm | 初始油墨厚度 |

## 使用配置

### Stage 1

```python
from src.predictors import HybridPredictor

predictor = HybridPredictor(
    config_path='config/stage6_wall_effect.json'
)
```

### Stage 2

```bash
python train_two_phase.py --epochs 30000
```

## 配置验证

```python
import json

with open('config/stage6_wall_effect.json', 'r') as f:
    config = json.load(f)

# 检查必需字段
required = ['model', 'training', 'materials']
for field in required:
    assert field in config, f"缺少必需字段: {field}"
```
