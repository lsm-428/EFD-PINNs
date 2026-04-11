# 故障排除指南

**最后更新**: 2026-02-04

## 常见问题

### 1. 模块导入失败

**症状**:
```
ModuleNotFoundError: No module named 'src'
```

**解决方案**:
```bash
# 确保在正确的环境
source .venv/bin/activate

# 确保在项目根目录
cd /path/to/EFD3D
```

### 2. PINN 模型不可用

**症状**:
```
RuntimeError: PINN 模型不可用
```

**解决方案**:
```python
# 检查模型是否存在
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
print(f"模型可用: {predictor.is_available}")
```

如果不可用：
- 确认已经完成 Stage 1 参数校准（`config/device_calibrated_physics.json`）
- 运行 Stage 2 训练生成最新 PINN 模型：

```bash
python train_two_phase.py --config config/device_calibrated_physics.json
```

### 3. CUDA 内存不足

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少批次大小 (`batch_size` 64 -> 32)
- 减少 `num_physics_points` (例如 5000 -> 2000)
- 使用 CPU 训练 (不推荐用于完整 60,000 epochs)

```bash
python train_two_phase.py --cpu
```

### 4. 训练损失爆炸

**症状**:
- 损失突然变为 NaN 或 Inf
- 损失剧烈波动

**解决方案**:
- 降低学习率 (修改配置中的 `learning_rate`)
- 增加梯度裁剪 (clip_grad_norm)
- 检查 `device_calibrated_physics.json` 中的物理参数是否异常

### 5. 开口率不正确 (0V 不为 0 或 30V 太低)

**症状**:
- V=0V 时开口率 > 0.1%
- V=30V 时开口率 < 80%

**解决方案**:
- 检查 Stage 1 的 `EnhancedApertureModel` 校准是否正确
- 增加训练中的边界条件权重 (`theta_weight`)
- 运行评估脚本查看动态响应：

```bash
python evaluate.py
```

## 调试工具

### 验证物理与动态响应

```bash
# 使用评估脚本
python evaluate.py
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## 日志和监控

### 训练日志

训练日志保存在输出目录：
```
outputs/train/pinn_YYYYMMDD_HHMMSS/
├── best_model.pth
├── config.json
└── training_history.json
```

### 检查训练历史

```python
import json

with open('outputs/train/pinn_xxx/training_history.json', 'r') as f:
    history = json.load(f)

print(f"最终损失: {history['loss'][-1]}")
```

## 性能优化

### GPU 加速

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
```

### 批量预测

```python
# 批量预测比循环更快
voltages = [0, 10, 20, 30]
times = [0.005, 0.010, 0.015]

for V in voltages:
    for t in times:
        eta = predictor.predict(voltage=V, time=t)
```

## 获取帮助

1. 查看文档: `docs/`
2. 运行测试: `python -m pytest tests/ -v`
3. 检查示例: `docs/api/examples_and_best_practices.md`
