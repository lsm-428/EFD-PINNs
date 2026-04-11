# 示例和最佳实践

**最后更新**: 2026-02-04
**版本**: v4.5

## 基础示例

### Stage 1: 接触角与开口率基准 (EnhancedApertureModel)

`EnhancedApertureModel` 提供了包含电容反馈、体积守恒和 RC 充电动态的增强预测。

```python
from src.models.aperture_model import EnhancedApertureModel

# 初始化模型 (自动加载配置文件)
model = EnhancedApertureModel()

# 1. 增强预测 (包含有效电压、充电进度等详细信息)
# 模拟 30V 电压在 10ms 时的状态
result = model.predict_enhanced(voltage=30, time=0.010)

print(f"输入电压: {result['voltage']} V")
print(f"有效电压: {result['effective_voltage']:.2f} V (充电进度: {result['charging_progress']:.1f}%)")
print(f"接触角: {result['theta']:.1f}°")
print(f"开口率: {result['aperture_percent']:.1f}%")
print(f"透明半径: {result['r_open_um']:.1f} μm")

# 2. 动态阶跃响应 (计算 t_10, t_90, 超调量)
t, eta = model.aperture_step_response(V_start=0, V_end=30, duration=0.05)
metrics = model.get_aperture_metrics(t, eta)
print(f"上升时间 (t_90): {metrics['t_90_ms']:.2f} ms")
```

### Stage 2: 6D PINN 开口率预测 (PINNAperturePredictor)

`PINNAperturePredictor` 封装了训练好的 PINN 模型，提供简化的推理接口。

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

# 初始化预测器 (自动加载 outputs/train/ 下最新的 best_model.pth)
predictor = PINNAperturePredictor()

if predictor.is_available:
    # 1. 预测标量开口率 (30V 阶跃, t=15ms)
    eta = predictor.predict(voltage=30, time=0.015)
    print(f"PINN 预测开口率: {eta:.3f}")

    # 2. 预测完整 3D 物理场
    # 返回字典包含: u, v, w (速度), p (压力), phi (相场), X, Y, Z (网格)
    fields = predictor.predict_full_field(voltage=30, time=0.015, n_points=(50, 50, 20))
    print(f"中心点相场值: {fields['phi'][25, 25, 10]:.4f}")
    
    # 3. 获取接触线轮廓
    contour_info = predictor.compute_aperture_contour(voltage=30, time=0.015)
    print(f"接触线等效半径: {contour_info['contour_r']*1e6:.2f} μm")
else:
    print("未找到训练好的 PINN 模型")
```

### Stage 3: LSTM-PINN 时序预测 (LSTMPINNModel)

`LSTMPINNModel` 用于处理变长电压序列，捕捉迟滞效应。

```python
import torch
from src.models.lstm_pinn.model import LSTMPINNModel

# 假设已加载模型
# model = LSTMPINNModel(config)
# model.load_state_dict(...)

# 1. 准备输入数据
# 空间坐标 (batch=100, 3)
spatial_coords = torch.rand(100, 3) 

# 电压序列 (batch=100, seq_len=10, 1)
# 模拟 0V -> 10V -> 20V -> 30V 的阶梯升压
voltage_sequence = torch.zeros(100, 10, 1)
for i in range(10):
    voltage_sequence[:, i, :] = i * 3.0

# 时间序列 (batch=100, seq_len=10, 1)
time_sequence = torch.zeros(100, 10, 1)
for i in range(10):
    time_sequence[:, i, :] = i * 0.002  # 2ms 间隔

# 2. 前向传播
output = model(spatial_coords, voltage_sequence, time_sequence)
phi_pred = output["phi"]  # (100, 1)

print(f"序列末端预测 φ 均值: {phi_pred.mean().item():.4f}")
```

## 训练示例

### 启动 TwoPhasePINN 训练

```bash
# 标准训练 (默认配置，30,000 epochs)
python train_two_phase.py

# 指定配置文件
python train_two_phase.py --config config/v4.5-standard.json

# 自定义训练轮次
python train_two_phase.py --epochs 50000

# 快速冒烟测试 (1000 epochs)
python train_two_phase.py --epochs 1000

# 从 Checkpoint 恢复
python train_two_phase.py --resume_from outputs/train/pinn_latest/latest_model.pth
```

### v4.5 训练配置

```json
{
  "training": {
    "epochs": 60000,
    "stage1_epochs": 1500,
    "stage2_epochs": 4000,
    "stage3_epochs": 50000,
    "batch_size": 4096,
    "learning_rate": 0.0003
  },
  "model": {
    "hidden_phi": [128, 128, 64, 32],
    "hidden_vel": [64, 64, 32]
  },
  "physics": {
    "interface_weight": 500.0,
    "ic_weight": 300.0,
    "bc_weight": 80.0,
    "continuity_weight": 0.5,
    "vof_weight": 0.5,
    "ns_weight": 0.1,
    "surface_tension_weight": 0.01,
    "sharpening_weight": 1.0
  }
}
```

## 最佳实践

### 1. 模型选择指南

| 场景 | 推荐模型 | 优势 |
|------|----------|------|
| **快速基准/参数校准** | `EnhancedApertureModel` | 毫秒级计算，物理可解释性强，包含 RC 动态 |
| **三维流场细节/微观分析** | `TwoPhasePINN` | 全 3D N-S 方程求解，捕捉接触线钉扎和局部流场 |
| **复杂波形/迟滞研究** | `LSTMPINNModel` | 支持任意电压波形输入，记忆历史状态 |

### 2. 6D Triad 数据格式

TwoPhasePINN 使用 6D 输入 `(x, y, z, V_from, V_to, t_since)` 来统一描述稳态和瞬态：

- **稳态**: `V_from = V_to`, `t_since` 任意 (通常取大值)
- **升压 (On)**: `V_from < V_to` (如 0 -> 30), `t_since` 为经过时间
- **降压 (Off)**: `V_from > V_to` (如 30 -> 0), `t_since` 为经过时间

### 3. 训练策略 (Stage 1 -> 2 -> 3)

- **Stage 1 (Data Only)**: 仅使用 `EnhancedApertureModel` 生成的数据进行监督学习。重点是让神经网络学会基本的几何形状和接触角映射。
- **Stage 2 (Continuity + VOF)**: 引入连续性方程 ($\nabla \cdot u = 0$) 和 VOF 方程。此时物理权重较小，防止模型崩溃。
- **Stage 3 (Full Physics)**: 引入 Navier-Stokes 方程和表面张力约束。这是最难训练的阶段，需要精细调节 `ns_weight`。

### 4. 常见问题排查

- **开口率卡在 0% 或 100%**: 检查 `interface_weight` 是否过低。PINN 容易陷入由边界条件主导的局部极小值（全 0 或全 1）。建议 Stage 1 训练更久一些。
- **体积不守恒**: 检查 `vof_weight` 和 `continuity_weight`。如果误差 > 5%，说明流场解可能有问题。
- **预测速度为 0**: 检查是否处于稳态 (`V_from == V_to`)。稳态下流速理论上为 0。
- **界面模糊**: 检查 `sharpening` 权重，建议设置为 0.1。

### 5. 验证训练结果

```bash
# 运行评估脚本
python evaluate.py

# 检查输出
# - 30V 开口率应接近 83.4%
# - 体积误差应 < 1%
# - 动态响应时间应符合预期
```
