# 配置系统与权重指南

**最后更新**: 2026-03-06

---

## 1. 核心原则：单一真理源 (SSOT)

本项目采用 **Single Source of Truth (SSOT)** 设计原则，确保物理参数、训练超参的一致性。

- **唯一真理源**：所有的物理常量（如表面张力、粘度、接触角等）必须且只能从 `config/device_calibrated_physics.json` 加载。
- **禁止硬编码**：代码中严禁出现 `h_ink = 3e-6` 这样的魔法数字。所有参数必须通过 `src/config` 模块获取。
- **分层管理**：
  - `src/config/physics_config.py`: 定义参数的数据结构和类型安全接口。
  - `config/*.json`: 存储具体的参数值（实验配置）。
  - `src/models/*.py`: 仅包含逻辑，消费配置。

详细参数说明请参见[物理理论与器件规格指南](physics_and_device_guide.md#physics-parameters) 和 [配置系统指南](configuration_guide.md)。

---

## 2. 配置文件位置

配置文件位于 `config/` 目录：

```
config/
├── v4.5-standard.json            # ⭐ 推荐训练配置（已验证收敛）
├── device_calibrated_physics.json    # 核心物理参数
└── v4.5_lbfgs_tuned.json            # L-BFGS 微调配置（备选）
```

### 2.1 主配置文件说明

| 配置文件 | 场景 | 说明 |
|----------|------|------|
| `v4.5-standard.json` | ⭐ 推荐 | 标准配置，已验证收敛，60,000 epochs |
| `device_calibrated_physics.json` | 设备校准 | 核心物理参数配置 |
| `v4.5_lbfgs_tuned.json` | LBFGS优化 | 备选配置，高权重设置 |

---

## 3. 物理参数配置

### 3.1 材料参数（推荐值，已校准）

| 参数 | 键名 | 默认值 | 说明 |
|------|------|--------|------|
| 初始接触角 | `theta0` | 120° | 疏水层初始接触角 |
| 像素墙接触角 | `theta_wall` | 71° | 围堰壁接触角 |
| 有效介电常数 | `epsilon_r` | 12.0 | 有效介电层（含渗透修正） |
| 表面张力 | `gamma` | 0.015 N/m | 有效界面张力 |
| 介电层厚度 | `dielectric_thickness` | 0.4 μm | SU-8 层 |

### 3.2 动力学参数

| 参数 | 键名 | 默认值 | 说明 |
|------|------|--------|------|
| 时间常数 | `tau` | 5 ms | 电润湿响应速度 |
| 恢复时间常数 | `tau_recovery` | 7.5 ms | 降压表面张力恢复 |
| 阻尼比 | `zeta` | 0.8 | 欠阻尼 |

### 3.3 几何参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 像素尺寸 | 174×174 μm | 电极尺寸 |
| 围堰高度 | 20 μm | 油墨层高度 |
| 油墨层厚度 | 3 μm | 初始油墨厚度 |

---

## 4. 训练超参数配置

### 4.1 核心配置结构

```json
{
  "model": {
    "hidden_phi": [128, 128, 64, 32],
    "hidden_vel": [64, 64, 32]
  },
  "training": {
    "epochs": 60000,
    "batch_size": 4096,
    "learning_rate": 0.0003,
    "stage1_epochs": 1500,
    "stage2_epochs": 4000,
    "stage3_epochs": 50000
  },
  "physics": {
    "interface_weight": 500.0,
    "ic_weight": 300.0,
    "bc_weight": 80.0,
    "continuity_weight": 0.5,
    "vof_weight": 0.5,
    "ns_weight": 0.1,
    "surface_tension_weight": 0.01,
    "sharpening_weight": 1.0,
    "explicit_volume_weight": 100.0
  }
}
```

### 4.2 三阶段训练 (60,000 Epochs)

| 阶段 | Epoch 范围 | 说明 |
|------|-----------|------|
| Phase 1 | 0 - 1,500 | 几何引导与相场初始化 |
| Phase 2 | 1,500 - 5,500 | 物理约束加载 (Continuity + VOF) |
| Phase 3 | 5,500 - 60,000 | 完整物理约束 (NS + Surface Tension) |

---

## 5. 物理权重配置详解

物理约束权重用于在PINN训练中平衡不同物理方程的重要性。正确的权重配置对训练稳定性和模型精度至关重要。

### 5.1 核心权重说明

#### VOF权重 (`vof`)

| 用途 | 推荐值 | 说明 |
|------|--------|------|
| 标准PINN训练 | 1.0 | 默认值 |
| LBFGS微调 | 10.0 | 需要更高权重保证界面追踪 |
| LSTM时序模型 | 1.0 | 使用归一化权重 |

**说明**: VOF权重控制流体体积法输运方程的约束强度。

#### 连续性权重 (`continuity`)

| 用途 | 推荐值 | 说明 |
|------|--------|------|
| 标准配置 | 0.3-0.5 | 默认值 |
| 高精度需求 | 1.0-2.0 | 增强质量守恒 |
| 快速原型 | 0.1 | 减少训练时间 |

**说明**: 连续性权重控制质量守恒方程的约束强度。

#### 动量权重 (`momentum_u`, `momentum_v`, `momentum_w`)

| 用途 | 推荐值 | 说明 |
|------|--------|------|
| 标准配置 | 0.3 | 各方向一致 |
| 各向异性流场 | 单独调整 | 根据流动特性 |

#### 界面权重

| 权重键 | 推荐值 | 说明 |
|--------|--------|------|
| `interface_stability` | 3.0 | 界面稳定性 |
| `interface_curvature` | 1.0 | 曲率约束 |
| `surface_tension` | 2.0 | 表面张力 |

#### 体积守恒 (`volume_conservation`)

| 用途 | 推荐值 | 说明 |
|------|--------|------|
| 标准配置 | 10.0 | 默认值 |
| 严格体积守恒 | 20.0-50.0 | 高精度需求 |

### 5.2 配置差异说明

#### 为什么不同配置文件有不同的权重？

不同配置文件针对不同的训练场景和模型架构，因此权重配置不同：

| 配置文件 | 场景 | vof_weight | continuity_weight | 说明 |
|----------|------|------------|------------------|------|
| `v4.5-standard.json` | ⭐ 推荐 | 0.5 | 0.5 | 标准配置，已验证收敛 |
| `device_calibrated_physics.json` | 设备校准 | 1.0 | 0.3 | 物理参数配置 |
| `v4.5_lbfgs_tuned.json` | LBFGS优化 | 10.0 | 2000.0 | 备选配置 |

#### LBFGS配置的高权重

LBFGS优化器需要更高的物理约束权重，因为：
- LBFGS是二阶优化器，收敛更快
- 高权重确保在快速收敛过程中物理约束不被违反
- 有助于避免局部最优解

### 5.3 快速参考表

| 场景 | 配置文件 | vof | continuity | volume |
|------|----------|-----|------------|--------|
| ⭐ 推荐训练 | v4.5-standard.json | 0.5 | 0.5 | 100.0 |
| 设备校准 | device_calibrated_physics.json | 1.0 | 0.3 | 10.0 |
| LBFGS优化 | v4.5_lbfgs_tuned.json | 10.0 | 2000.0 | - |

---

## 6. 如何在代码中使用配置

### 6.1 获取物理参数 (推荐)

在任何 Python 脚本中：

```python
from src.config.physics_config import PHYSICS, get_physics_config

# 方式 1: 直接使用全局单例（自动从默认配置加载）
theta0 = PHYSICS["theta0"]
viscosity = PHYSICS["mu_polar"]

# 方式 2: 获取类型安全的配置对象
config = get_physics_config()  # 加载默认配置
print(f"油墨厚度: {config.h_ink} m")

# 方式 3: 加载特定实验配置
config_v2 = get_physics_config("config/v4.5-standard.json")
```

### 6.2 获取文件路径

```python
from src.config.paths import CONFIG_PATH, OUTPUT_DIR

print(f"当前使用的配置文件: {CONFIG_PATH}")
print(f"输出目录: {OUTPUT_DIR}")
```

---

## 7. 如何开始新实验

1. **复制配置**：
   ```bash
   cp config/device_calibrated_physics.json config/experiment_new_material.json
   ```

2. **修改参数**：
   编辑 `config/experiment_new_material.json`，例如修改 `theta0` 或 `viscosity`。

3. **运行训练**：
   ```bash
   python train_two_phase.py --config config/experiment_new_material.json
   ```

   脚本会自动加载指定文件，并更新全局 `PHYSICS` 常量，确保模型使用的是新参数。

---

## 8. 使用示例

### Stage 2: 两相流 PINN

```bash
# 使用推荐配置训练
python train_two_phase.py --config config/v4.5-standard.json
```

### 代码中加载配置

```python
import json
from src.models.pinn_two_phase import Trainer

# 加载配置
with open('config/v4.5-standard.json', 'r') as f:
    config = json.load(f)

# 创建模型
trainer = Trainer(config)
```

---

## 9. 配置验证

```python
import json

with open('config/v4.5-standard.json', 'r') as f:
    config = json.load(f)

# 检查必需字段
required = ['model', 'training', 'physics']
for field in required:
    assert field in config, f"缺少必需字段: {field}"
```

---

## 10. 故障排除

### 10.1 配置相关问题

**Q: 我修改了代码里的 DEFAULT_CONFIG，为什么没生效？**
A: `pinn_two_phase.py` 中的 `DEFAULT_CONFIG_TEMPLATE` 仅用于参考结构，不再参与实际逻辑。实际参数完全由 JSON 文件决定。请修改 `config/*.json` 文件。

**Q: 为什么训练脚本启动时会打印 "全局 PHYSICS 已更新"？**
A: 这是为了确认配置已成功注入。如果不显示此日志，说明可能意外使用了硬编码回退（这种情况在新代码中已被杜绝）。

**Q: 推荐使用哪个配置文件？**
A: 建议使用 `config/device_calibrated_physics.json` 作为基础，它包含了经过设备校准的最佳物理参数。

### 10.2 权重配置问题

#### 训练不收敛

**症状**: 损失不下降或振荡

**可能原因**: 权重配置不当

**解决方案**:
1. 检查 `vof_weight` 是否太低（尝试 1.0-10.0）
2. 检查 `continuity_weight` 是否太低（尝试 0.3-1.0）
3. 启用自适应权重调整

#### 界面模糊

**症状**: VOF界面不清晰

**可能原因**: VOF权重太低

**解决方案**:
1. 增加 `vof_weight` 到 5.0-10.0
2. 增加 `interface_stability` 到 5.0
3. 检查 `surface_tension` 是否合理

#### 体积不守恒

**症状**: 流体体积随时间变化

**可能原因**: 体积守恒权重太低

**解决方案**:
1. 增加 `volume_conservation` 到 20.0-50.0
2. 检查初始条件是否正确设置
3. 验证数据生成器的体积计算

---

## 11. 相关文档

- [README.md](../../README.md) - 项目主文档
- [CLAUDE.md](../../CLAUDE.md) - 开发者指南
- [中央参考文档](../../README.md) - 物理参数和训练配置权威来源

---