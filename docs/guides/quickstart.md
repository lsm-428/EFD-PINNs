# EFD3D 快速开始指南

**最后更新**: 2026-02-04
**版本**: v4.5

## 🚀 快速上手

本指南提供EFD3D的完整工作流程，从环境验证到模型训练和评估。

### 1. 环境准备

确保已按照[安装指南](installation.md)完成环境设置，然后激活环境：

```bash
# 激活环境
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.config import PHYSICS; print('EFD3D 导入成功')"
```

### 2. Stage 1: 物理基准校准验证

使用 `EnhancedApertureModel` 验证解析模型的校准状态：

```python
from src.models.aperture_model import EnhancedApertureModel

# 创建模型（使用默认配置）
model = EnhancedApertureModel()

# 预测开口率
for V in [0, 10, 20, 30]:
    result = model.predict(V)
    print(f"V={V}V: θ={result['theta']:.1f}°, η={result['aperture_percent']:.1f}%")
```

### 3. Stage 2: 6D PINN 训练

启动完整的 60,000 轮渐进式训练：

```bash
# 使用默认配置训练
python train_two_phase.py

# 或指定 epochs
python train_two_phase.py --epochs 60000

# 快速测试 (1000 轮)
python train_two_phase.py --epochs 1000
```

### 4. 动态响应验证 (6D Triad)

验证升压/降压过程中的电压跳变响应：

```bash
# 运行评估脚本
python evaluate.py

# 指定模型目录
python evaluate.py outputs/train/pinn_latest

# 对比最近两个模型
python evaluate.py --compare
```

### 5. 结果可视化

生成 6D 输入空间的物理场切片与响应曲线：

```bash
python evaluate.py
```

---

## 📋 完整工作流程

### 步骤 1：验证 Stage 1 校准

```python
from src.models.aperture_model import EnhancedApertureModel
from src.config import get_physics_config

config = get_physics_config()
model = EnhancedApertureModel(config)

# 验证 30V 开口率
result = model.predict(30)
print(f"30V: θ={result['theta']:.1f}°, η={result['aperture_percent']:.1f}%")
```

### 步骤 2：Stage 2 训练

```bash
# 快速测试
python train_two_phase.py --epochs 1000

# 完整训练 (60,000 epochs)
python train_two_phase.py --epochs 60000
```

### 步骤 3：物理验证

```bash
python evaluate.py
```

### 步骤 4：可视化结果

```bash
python evaluate.py  # 自动生成可视化
```

---

## 📊 预期结果

### Stage 1 开口率（已校准）

| 电压 | 接触角 | 开口率 |
|------|--------|--------|
| 0V | 120.0° | 0% |
| 20V | ~67.5° | ~60% |
| 30V | ~67.5° | **83.4%** |

### Stage 2 PINN 目标

| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83-87%** | 像素开口率 |
| 体积误差 | **<1%** | VOF 体积守恒 |
| 训练轮次 | **60,000** | 三阶段渐进式 |
| 输入表示 | **6D Triad** | (x, y, z, V_from, V_to, t_since) |

---

## 🎉 下一步

完成基础训练后，您可以：

1. **查看项目架构**: [README.md](../../README.md)
2. **了解详细训练策略**: [training_guide.md](training_guide.md)
3. **查看器件规格**: [../guides/physics_and_device_guide.md](../guides/physics_and_device_guide.md)
4. **解决常见问题**: [troubleshooting.md](troubleshooting.md)

---

**需要帮助？** 查看[故障排除指南](troubleshooting.md)
