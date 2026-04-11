# Level Set PINN 用户使用指南 (v5.5)

本文档提供了 Level Set 3D PINN 模型的完整使用指南，涵盖从模型训练、实时监控到结果评估与可视化的全流程。

---

## 目录

1.  [🚀 训练模型](#1-训练模型)
    *   [启动训练](#11-启动训练)
    *   [配置文件说明](#12-配置文件说明)
    *   [理解日志输出](#13-理解日志输出)
    *   [恢复训练](#14-恢复训练)
2.  [📈 实时监控与曲线](#2-实时监控与曲线)
    *   [自动生成的图表](#21-自动生成的图表)
    *   [图表详细解读](#22-图表详细解读)
3.  [📊 模型评估与可视化](#3-模型评估与可视化)
    *   [评估脚本用法](#31-评估脚本用法)
    *   [可视化仪表板解读](#32-可视化仪表板解读)
    *   [3D 等值面可视化](#33-3d-等值面可视化)
4.  [💻 Python API 调用](#4-python-api-调用)
5.  [❓ 常见问题排查](#5-常见问题排查)

---

## 1. 训练模型

### 1.1 启动训练

使用 `train_levelset_3d.py` 脚本启动训练。建议使用 v5.5 版本的全量配置：

```bash
# 推荐：使用 v5.5 完整配置进行训练
python train_levelset_3d.py --config config/v5.5_full.json

# 指定输出目录（可选）
python train_levelset_3d.py --config config/v5.5_full.json --output-dir outputs_my_experiment
```

### 1.2 配置文件说明

配置文件（如 `config/v5.5_full.json`）是 JSON 格式，核心参数包括：

*   **`loss_weights`**: 定义各项物理约束的权重。
    *   `data`: 数据拟合损失（最重要，确保数值正确）。
    *   `levelset_transport`: 界面演化方程约束。
    *   `sign_constraint`: 强制 ψ 符号约定（ψ >= 0，界面高度）。
    *   `contact_angle`: 接触角边界条件。
*   **`training.training_stages`**: 定义多阶段训练策略。
    *   **Stage 1 (Data)**: 仅学习数据分布，快速拟合。
    *   **Stage 2 (Transport)**: 引入 Level Set 输运方程。
    *   **Stage 3 (Multiphysics)**: 引入接触角和静电场约束。

### 1.3 理解日志输出

训练过程中，控制台会输出如下日志：

```text
Epoch [500/50000] Loss: 1.23e-02 (Data: 1.0e-02, LevelSet: 2.0e-03, ... Ink>0: 98.5%, Polar≈0: 99.1%)
```

*   **Loss**: 总损失值（加权和）。
*   **Data / LevelSet / ...**: 各分项损失的原始值（未加权）。
*   **Ink<0**: 模型预测油墨区域为负值的正确率（目标 > 95%）。
*   **Polar>0**: 模型预测极性液体区域为正值的正确率（目标 > 95%）。

### 1.4 恢复训练

如果训练中断，可以通过 `--resume` 参数从检查点恢复：

```bash
python train_levelset_3d.py --resume outputs_levelset_XXXXXX/checkpoint_epoch_20000.pt
```

---

## 2. 实时监控与曲线

训练脚本会自动在输出目录生成并更新以下图表（每 500 epochs）：

### 2.1 自动生成的图表

| 文件名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`training_dashboard.png`** | 综合面板 | 包含 Loss 曲线、符号准确率、分量占比、近期趋势的 2x2 视图。**首选监控图表**。 |
| **`training_curve.png`** | 趋势图 | 展示 Total Loss 和所有分项 Loss 的下降趋势（对数坐标）。 |
| **`symbol_accuracy.png`** | 诊断图 | 展示 `Ink<0` 和 `Polar>0` 的准确率曲线。用于判断模型是否学到了正确的物理相分布。 |
| **`loss_components_fraction.png`** | 堆叠图 | 展示各 Loss 分量在总 Loss 中的占比。用于分析当前阶段模型主要在优化什么。 |

### 2.2 图表详细解读

#### 1. 为什么 Loss 会突然跳变？
这是 **多阶段训练 (Multi-Stage Training)** 的正常现象。
*   当从 Stage 1 切换到 Stage 2 时，引入了新的 PDE 约束（如输运方程），Loss 会突然升高，然后随着训练继续下降。
*   **注意**：只要跳变后 Loss 呈下降趋势，即为正常。

#### 2. 符号准确率 (Symbol Accuracy) 的重要性
*   **理想状态**：两条曲线（Ink 和 Polar）都应接近 100%。
*   **危险信号**：如果 `Ink<0` 长期低于 80%，说明模型未能区分油墨和极性液体，物理预测（如体积、开口率）将完全失效。此时应增加 `sign_constraint` 权重。

---

## 3. 模型评估与可视化

训练完成后，使用 `evaluate.py` 对模型进行物理性能评估。

### 3.1 评估脚本用法

```bash
# 基础评估（计算 0V-30V 开口率）
python evaluate.py outputs_levelset_XXXXXX

# 生成 4 面板可视化仪表板（推荐）
python evaluate.py outputs_levelset_XXXXXX --plot-dashboard

# 生成 3D 等值面可视化（较慢）
python evaluate.py outputs_levelset_XXXXXX --plot-3d

# 指定使用 CPU 或 GPU
python evaluate.py outputs_levelset_XXXXXX --device cpu
```

### 3.2 可视化仪表板解读

生成的 `evaluation_dashboard.png` 包含四个面板：

1.  **Top View (左上)**:
    *   俯视 $\psi$ 场热图 (z = 1.5μm)。
    *   **红色**: 极性液体 (透明)。 **蓝色/紫色**: 油墨 (不透明)。
    *   **黑线**: $\psi=0$ 界面。
    *   **箭头**: 流体速度矢量。
2.  **Side View (右上)**:
    *   侧视截面图 (y = Ly/2)。
    *   可以观察油墨的堆积高度和接触角形状。
3.  **Dynamic Response (左下)**:
    *   展示 0V $\to$ 30V $\to$ 0V 过程中开口率随时间的变化。
    *   用于验证响应时间（上升沿/下降沿）。
4.  **Aperture vs Voltage (右下)**:
    *   稳态开口率随电压变化的曲线 (0V - 30V)。
    *   **蓝色实线**: PINN 预测值。
    *   **黑色虚线**: 理论参考值（如有）。

### 3.3 3D 等值面可视化

生成的 `3d_isosurface.png` 展示了 $\psi=0$ 的三维曲面。
*   这直接反映了油墨-极性液体的界面形状。
*   正常情况下，30V 时应看到油墨被挤压到像素四周的角落。

---

## 4. Python API 调用

如果您需要在自定义代码中调用模型（例如集成到其他系统），请参考以下示例：

```python
import torch
import numpy as np
from pinn_levelset_3d import LevelSet3DPINN

# 1. 加载模型
ckpt_path = 'outputs_levelset_XXXXXX/best_model.pt'
checkpoint = torch.load(ckpt_path, map_location='cuda')
config = checkpoint['config']

model = LevelSet3DPINN(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# 2. 构造输入张量 (Batch_Size, 6)
# 输入列: [x, y, z, V_from, V_to, t_since]
n_points = 1000
x = torch.linspace(0, 174e-6, n_points)
y = torch.linspace(0, 174e-6, n_points)
# ... 构造网格 ...

# 示例输入：中心点，30V，稳态
input_tensor = torch.tensor([
    [87e-6, 87e-6, 10e-6, 0.0, 30.0, 0.05]
], dtype=torch.float32).cuda()

# 3. 预测
with torch.no_grad():
    output = model(input_tensor)
    
    # 解析输出 (参考 pinn_levelset_3d.py)
    # [u1, v1, w1, u2, v2, w2, psi, p1, p2]
    psi = output[:, 6]      # Level Set 值
    u_ink = output[:, 0:3]  # 油墨速度
    
    print(f"Psi value: {psi.item():.6f}")
    if psi < epsilon:
        print("状态: 开口 (界面在 z=0)")
    else:
        print("状态: 油墨覆盖")
```

---

## 5. 常见问题排查

### Q1: 训练初期 Loss 不下降
*   **原因**: 学习率过大或过小，或者数据生成有问题。
*   **解决**: 检查 `training_curve.png`。如果 Data Loss 很高且不下降，尝试降低学习率。检查 `generate_levelset_data.py` 是否生成了有效的 `.pt` 文件。

### Q2: 符号准确率 (Symbol Accuracy) 很低 (< 50%)
*   **原因**: 模型学反了符号（ψ 全为负），或者完全坍塌为常数。
*   **解决**:
    1.  增加 `loss_weights['sign_constraint']` 的权重（例如从 50 增加到 100）。
    2.  增加 `loss_weights['psi_spatial']` 权重以防止模式坍塌。

### Q3: 评估时 CUDA Out of Memory
*   **原因**: `evaluate.py` 默认使用较高分辨率 (96x96x96) 进行计算。
*   **解决**: 使用 `--device cpu` 强制使用 CPU，或在代码中降低 `self.spatial_res`。

### Q4: 3D 等值面生成失败
*   **原因**: `skimage.measure.marching_cubes` 未找到 ψ=0 的等值面。通常是因为预测出的 ψ 全大于 epsilon（无开口）。
*   **解决**: 检查 `symbol_accuracy.png`。如果符号准确率低，模型无法生成有效的界面。需要重新训练。

---
*文档最后更新：2026-01-27*
