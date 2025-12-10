# 示例和最佳实践

**最后更新**: 2025-12-08

## 基础示例

### Stage 1: 接触角预测

```python
from src.predictors import HybridPredictor

# 初始化预测器
predictor = HybridPredictor()

# 单点预测
theta = predictor.predict(voltage=30, time=0.005)
print(f"接触角: {theta:.1f}°")

# 阶跃响应
t, theta = predictor.step_response(V_start=0, V_end=30, duration=0.02)

# 方波响应
t, V, theta = predictor.square_wave_response(V_low=0, V_high=30)

# 响应指标
metrics = predictor.get_response_metrics(t, theta)
print(f"响应时间: {metrics['t_90_ms']:.2f} ms")
```

### Stage 2: 开口率预测

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

# 初始化预测器
predictor = PINNAperturePredictor()

# 预测开口率
eta = predictor.predict(voltage=30, time=0.015)
print(f"开口率: {eta:.3f}")

# 预测 φ 场
phi_field = predictor.predict_phi_field(voltage=30, time=0.015)

# 预测完整 3D 场
fields = predictor.predict_full_field(voltage=30, time=0.015)
```

## 训练示例

### Stage 1 训练

```bash
# 快速测试
python train_contact_angle.py --quick-run

# 标准训练
python train_contact_angle.py --config config/stage6_wall_effect.json --epochs 3000
```

### Stage 2 训练

```bash
# 快速测试
python train_two_phase.py --epochs 1000

# 完整训练
python train_two_phase.py --epochs 30000
```

## 可视化示例

```python
import matplotlib.pyplot as plt
from src.predictors import HybridPredictor

predictor = HybridPredictor()

# 绘制阶跃响应
t, theta = predictor.step_response(V_start=0, V_end=30)

plt.figure(figsize=(10, 6))
plt.plot(t * 1000, theta)
plt.xlabel('Time (ms)')
plt.ylabel('Contact Angle (°)')
plt.title('Step Response: 0V → 30V')
plt.grid(True)
plt.savefig('step_response.png')
```

## 最佳实践

### 1. 模型选择

- Stage 1: 使用 `HybridPredictor`（解析公式，快速准确）
- Stage 2: 使用 `PINNAperturePredictor`（需要训练好的模型）

### 2. 参数范围

| 参数 | 推荐范围 |
|------|----------|
| 电压 | 0-30V |
| 时间 | 0-20ms |
| 接触角 | 60°-130° |

### 3. 训练建议

- 使用渐进式训练策略
- 监控物理残差
- 验证开口率单调性

### 4. 性能优化

- 使用 GPU 加速训练
- 批量预测提高效率
- 缓存常用结果

## 故障排除

### 模块导入失败

```bash
conda activate efd
```

### PINN 模型不可用

```python
predictor = PINNAperturePredictor()
print(f"模型可用: {predictor.is_available}")
```

### 预测结果异常

```python
# 验证 Young-Lippmann
for V in [0, 10, 20, 30]:
    theta = predictor.young_lippmann(V)
    print(f"V={V}V: θ={theta:.1f}°")
```
