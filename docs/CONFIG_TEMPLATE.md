# EFD-PINNs 配置文件模板

**最后更新**: 2025-12-08

---

## 📋 配置文件概述

EFD-PINNs 使用 JSON 配置文件控制训练参数。配置文件位于 `config/` 目录。

---

## 🏗️ 两阶段训练配置

### Stage 1: 接触角预测

Stage 1 使用解析公式，无需复杂配置。主要参数在 `train_contact_angle.py` 中设置：

```python
# 材料参数
epsilon_r = 4.0      # 相对介电常数
gamma = 0.072        # 表面张力 (N/m)
theta0 = 120.0       # 初始接触角 (°)
d = 1e-6             # 介电层厚度 (m)

# 动力学参数
tau = 0.003          # 时间常数 (s)
zeta = 0.7           # 阻尼比
```

### Stage 2: 两相流 PINN

Stage 2 使用 PINN 模型，配置文件示例：

```json
{
  "model": {
    "hidden_dims": [128, 256, 256, 128],
    "activation": "tanh"
  },
  "training": {
    "epochs": 5000,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": "cosine"
  },
  "physics": {
    "lambda_pde": 0.1,
    "lambda_bc": 1.0,
    "lambda_data": 1.0
  }
}
```

---

## 🔑 配置键说明

### 模型配置 (model)

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dims` | list[int] | [128, 256, 256, 128] | 隐藏层维度 |
| `activation` | str | "tanh" | 激活函数 (tanh/gelu/silu) |
| `use_fourier` | bool | true | 是否使用傅里叶特征 |
| `fourier_scale` | float | 1.0 | 傅里叶特征缩放 |

### 训练配置 (training)

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | int | 5000 | 训练轮次 |
| `batch_size` | int | 256 | 批次大小 |
| `learning_rate` | float | 1e-3 | 学习率 |
| `weight_decay` | float | 1e-5 | 权重衰减 |
| `scheduler` | str | "cosine" | 学习率调度器 |
| `warmup_epochs` | int | 100 | 预热轮次 |

### 物理损失权重 (physics)

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lambda_pde` | float | 0.1 | PDE 残差权重 |
| `lambda_bc` | float | 1.0 | 边界条件权重 |
| `lambda_data` | float | 1.0 | 数据拟合权重 |
| `lambda_interface` | float | 0.5 | 界面约束权重 |

### 材料参数 (materials)

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epsilon_r` | float | 4.0 | 相对介电常数 |
| `gamma` | float | 0.072 | 表面张力 (N/m) |
| `theta0` | float | 120.0 | 初始接触角 (°) |
| `rho_oil` | float | 850.0 | 油相密度 (kg/m³) |
| `rho_water` | float | 1000.0 | 水相密度 (kg/m³) |
| `mu_oil` | float | 0.01 | 油相粘度 (Pa·s) |
| `mu_water` | float | 0.001 | 水相粘度 (Pa·s) |

---

## 📁 现有配置文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `config/stage1_config.json` | Stage 1 基础配置 | 接触角预测参数 |
| `config/stage2_optimized.json` | Stage 2 优化配置 | 两相流 PINN |
| `config/stage2_dynamic_response.json` | 动态响应配置 | 时间序列预测 |
| `config/optimized_small.json` | 小规模测试 | 快速验证 |
| `config/optimized_medium.json` | 中等规模 | 平衡精度与速度 |
| `config/optimized_large.json` | 大规模训练 | 最高精度 |

---

## 🔧 使用示例

### Stage 1: 接触角预测

```bash
# 训练 Stage 1 (使用解析公式，无需配置文件)
python train_contact_angle.py

# 或指定输出目录
python train_contact_angle.py --output-dir outputs_contact_angle
```

### Stage 2: 两相流 PINN

```bash
# 使用默认配置训练
python train_two_phase.py

# 使用指定配置文件
python train_two_phase.py --config config/stage2_optimized.json

# 指定输出目录
python train_two_phase.py --output-dir outputs_pinn_custom
```

### 代码中加载配置

```python
import json
from src.models.pinn_two_phase import TwoPhaseFlowPINN

# 加载配置
with open('config/stage2_optimized.json', 'r') as f:
    config = json.load(f)

# 创建模型
model = TwoPhaseFlowPINN(
    hidden_dims=config['model']['hidden_dims'],
    activation=config['model']['activation']
)
```

---

## ⚠️ 注意事项

1. **Stage 1 无需配置文件**: Stage 1 使用解析公式，参数直接在代码中设置

2. **Stage 2 配置文件**: 主要用于 PINN 模型的超参数调优

3. **物理参数一致性**: 确保 Stage 1 和 Stage 2 使用相同的材料参数

4. **输出目录**: 训练结果自动保存到带时间戳的目录

---

**更新**: 2025-12-08 | **状态**: ✅ 已更新
