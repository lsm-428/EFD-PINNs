# EFD-PINNs 快速开始指南

**最后更新**: 2025-12-10

## 🚀 快速上手

### 1. 环境准备

```bash
# 激活conda环境
conda activate efd

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. Stage 1: 开口率预测（已校准）

```python
from src.models.aperture_model import EnhancedApertureModel

# 初始化模型（使用校准后的配置）
model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')

# 预测开口率
for V in [0, 10, 20, 30]:
    theta = model.get_contact_angle(V)
    eta = model.contact_angle_to_aperture_ratio(theta)
    print(f"V={V}V: θ={theta:.1f}°, η={eta*100:.1f}%")

# 输出:
# V=0V: θ=120.0°, η=0.0%
# V=10V: θ=119.2°, η=10.3%
# V=20V: θ=115.2°, η=66.7%  ← 实验值 67%
# V=30V: θ=108.2°, η=84.4%
```

### 3. Stage 1: 接触角动态响应

```python
from src.predictors import HybridPredictor

# 初始化预测器
predictor = HybridPredictor(config_path='config/stage6_wall_effect.json')

# 单点预测
theta = predictor.predict(voltage=20, time=0.01)
print(f"20V, 10ms 时接触角: {theta:.1f}°")

# 阶跃响应
t, theta = predictor.step_response(V_start=0, V_end=20, duration=0.02)
```

### 4. Stage 2: PINN φ 场预测

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

# 初始化预测器（自动加载最新模型）
predictor = PINNAperturePredictor()

# 预测开口率
eta = predictor.predict(voltage=20, time=0.02)
print(f"PINN 开口率: {eta:.3f}")  # ~0.736
```

### 5. 训练模型

```bash
# Stage 1: 接触角训练
python train_contact_angle.py --quick-run

# Stage 2: 两相流 PINN 训练
python train_two_phase.py --epochs 10000

# 物理验证
python validate_pinn_physics.py
```

### 6. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v
```

## 📋 完整工作流程

### 步骤1：验证 Stage 1 校准

```bash
# 验证 20V 开口率
python -c "
from src.models.aperture_model import EnhancedApertureModel
model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')
theta = model.get_contact_angle(20)
eta = model.contact_angle_to_aperture_ratio(theta)
print(f'20V: θ={theta:.1f}°, η={eta*100:.1f}% (实验值: 67%)')
"
```

### 步骤2：Stage 2 训练

```bash
# 快速测试
python train_two_phase.py --epochs 1000

# 完整训练
python train_two_phase.py --epochs 10000
```

### 步骤3：物理验证

```bash
# 验证 PINN 物理合理性
python validate_pinn_physics.py
```

### 步骤4：可视化结果

```bash
# 可视化 PINN 结果
python visualize_pinn_results.py
```

## 🔧 常用命令速查

### 训练相关
```bash
# Stage 1 多阶段训练
python train_contact_angle.py --multi-stage --epochs 10000

# Stage 2 完整训练
python train_two_phase.py --epochs 10000
```

### 测试与验证
```bash
# 运行所有测试
python -m pytest tests/ -v

# 物理验证
python validate_pinn_physics.py
```

## 📊 预期结果

### Stage 1 开口率（已校准）

| 电压 | 接触角 | 开口率 | 实验值 |
|------|--------|--------|--------|
| 0V | 120.0° | 0% | 0% |
| 20V | 115.2° | 66.7% | **67%** ✓ |
| 30V | 108.2° | 84.4% | - |

### Stage 2 PINN（t=20ms）

| 电压 | Stage 1 η | PINN η | 误差 |
|------|-----------|--------|------|
| 0V | 0% | 0% | 0% |
| 20V | 66.7% | 73.6% | +6.9% |
| 30V | 84.4% | 84.6% | +0.2% |

## 🚨 故障排除

### 常见问题

**问题1：模块导入失败**
```bash
# 确保在正确的环境
conda activate efd
```

**问题2：开口率预测不准确**
```python
# 确保使用校准后的配置
from src.models.aperture_model import EnhancedApertureModel
model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')

# 检查参数
print(f"k = {model.aperture_k}")  # 应为 0.8
print(f"theta_scale = {model.aperture_theta_scale}")  # 应为 6.0
```

**问题3：PINN 模型不可用**
```python
# 检查模型是否存在
from src.predictors.pinn_aperture import PINNAperturePredictor
predictor = PINNAperturePredictor()
print(f"模型可用: {predictor.is_available}")
```

**问题4：CUDA内存不足**
```bash
# 降低批次大小
python train_two_phase.py --epochs 5000
```

## 🎉 下一步

完成基础训练后，您可以：

1. **查看详细使用指南**: [../../USAGE_GUIDE.md](../../USAGE_GUIDE.md)
2. **了解项目架构**: [../specs/MODULE_OVERVIEW.md](../specs/MODULE_OVERVIEW.md)
3. **查看训练策略**: [training_strategies.md](training_strategies.md)
4. **查看器件规格**: [../specs/DEVICE_SPECS.md](../specs/DEVICE_SPECS.md)

---

**需要帮助？** 查看[故障排除指南](troubleshooting_debugging.md)
