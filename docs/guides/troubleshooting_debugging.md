# 故障排除指南

**最后更新**: 2025-12-08

## 常见问题

### 1. 模块导入失败

**症状**:
```
ModuleNotFoundError: No module named 'src'
```

**解决方案**:
```bash
# 确保在正确的环境
conda activate efd

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

# 如果不可用，需要先训练
python train_two_phase.py --epochs 30000
```

### 3. CUDA 内存不足

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少批次大小
- 使用 CPU 训练
- 清理 GPU 内存

```python
import torch
torch.cuda.empty_cache()
```

### 4. 训练损失爆炸

**症状**:
- 损失突然变为 NaN 或 Inf
- 损失剧烈波动

**解决方案**:
- 降低学习率
- 增加梯度裁剪
- 检查数据归一化

### 5. 开口率不正确

**症状**:
- V=0V 时开口率不为 0
- V=30V 时开口率太低

**解决方案**:
- 检查 φ 场定义
- 验证边界条件
- 增加低电压约束权重

## 调试工具

### 验证物理

```bash
python validate_pinn_physics.py
```

### 可视化结果

```bash
python visualize_pinn_results.py
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## 日志和监控

### 训练日志

训练日志保存在输出目录：
```
outputs_pinn_YYYYMMDD_HHMMSS/
├── best_model.pth
├── config.json
└── training_history.json
```

### 检查训练历史

```python
import json

with open('outputs_pinn_xxx/training_history.json', 'r') as f:
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
