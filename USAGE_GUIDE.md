# EFD-PINNs 使用指南

**最后更新**: 2025-12-10  
**适用版本**: v6.0 (Stage 1 已校准 + Stage 2 已验证)

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate efd

# 检查依赖
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

# 方波响应
t, V, theta = predictor.square_wave_response(V_low=0, V_high=20)

# 获取响应指标
metrics = predictor.get_response_metrics(t, theta)
print(f"响应时间: {metrics['t_90_ms']:.2f} ms")
print(f"超调: {metrics['overshoot_percent']:.1f}%")
```

### 4. Stage 2: PINN φ 场预测

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

# 初始化预测器（自动加载最新模型）
predictor = PINNAperturePredictor()

# 预测开口率
eta = predictor.predict(voltage=20, time=0.02)
print(f"PINN 开口率: {eta:.3f}")  # ~0.736

# 预测 φ 场
phi_field = predictor.predict_phi_field(voltage=20, time=0.02)

# 获取完整 3D 场
fields = predictor.predict_full_field(voltage=20, time=0.02)
```

---

## 📊 训练模型

### Stage 1: 接触角训练

```bash
# 快速测试
python train_contact_angle.py --quick-run

# 标准训练
python train_contact_angle.py --config config/stage6_wall_effect.json --epochs 3000

# 多阶段训练
python train_contact_angle.py --multi-stage --epochs 10000
```

### Stage 2: 两相流 PINN 训练

```bash
# 快速测试
python train_two_phase.py --epochs 1000

# 完整训练
python train_two_phase.py --epochs 10000

# 物理验证
python validate_pinn_physics.py
```

---

## 📈 已校准的物理参数

### 材料参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 初始接触角 | θ₀ | 120° | |
| SU-8 介电常数 | εᵣ | 3.0 | 介电层 |
| Teflon 介电常数 | εₕ | 1.9 | 疏水层 |
| SU-8 厚度 | d | 400nm | |
| Teflon 厚度 | dₕ | 400nm | |
| 极性液体表面张力 | γ | 0.050 N/m | 乙二醇混合液 |
| 阈值电压 | V_T | 3V | |

### 动力学参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 时间常数 | τ | 5 ms | 响应速度 |
| 阻尼比 | ζ | 0.8 | 欠阻尼 |

### 开口率映射参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 0.8 | 陡度参数 |
| theta_scale | 6.0 | 角度缩放因子 |
| alpha | 0.05 | 电容反馈强度 |
| aperture_max | 0.85 | 最大开口率 |

### 几何参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 像素尺寸 | 174×174 μm | 内沿尺寸 |
| 油墨厚度 | 3 μm | |
| 围堰高度 | 3.5 μm（实际）/ 20 μm（模型） | |

---

## 📊 预期结果

### 稳态预测（已校准）

| 电压 | 接触角 | 开口率 | 实验值 |
|------|--------|--------|--------|
| 0V | 120.0° | 0% | 0% |
| 6V | ~119° | ~1% | 开始响应 |
| 10V | 119.2° | 10.3% | - |
| 20V | 115.2° | 66.7% | **67%** ✓ |
| 30V | 108.2° | 84.4% | - |

### PINN 预测 (t=20ms)

| 电压 | Stage 1 η | PINN η | 误差 |
|------|-----------|--------|------|
| 0V | 0% | 0% | 0% |
| 10V | 10.3% | 9.2% | -1.1% |
| 20V | 66.7% | 73.6% | +6.9% |
| 30V | 84.4% | 84.6% | +0.2% |

### 动态响应指标

| 指标 | 目标 | 实现 |
|------|------|------|
| 响应时间 (t90) | <30ms | ~14ms |
| 超调 | <15% | ~4% |
| 角度变化 | ~33° | ~30° |

---

## 📁 关键文件

### 核心代码

| 文件 | 说明 |
|------|------|
| `src/models/aperture_model.py` | 开口率模型（已校准） |
| `src/predictors/hybrid_predictor.py` | 混合预测器 |
| `src/models/pinn_two_phase.py` | 两相流 PINN 模型 |
| `src/predictors/pinn_aperture.py` | PINN 开口率预测器 |

### 训练入口

| 文件 | 说明 |
|------|------|
| `train_contact_angle.py` | Stage 1 训练 |
| `train_two_phase.py` | Stage 2 训练 |

### 工具脚本

| 文件 | 说明 |
|------|------|
| `validate_pinn_physics.py` | 物理验证脚本 |
| `visualize_pinn_results.py` | 可视化脚本 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `config/stage6_wall_effect.json` | 校准后的配置（推荐） |

---

## 🔍 故障排除

### 常见问题

1. **模块导入失败**
   ```bash
   # 确保在正确的环境
   conda activate efd
   ```

2. **PINN 模型不可用**
   ```python
   # 检查模型是否存在
   from src.predictors.pinn_aperture import PINNAperturePredictor
   predictor = PINNAperturePredictor()
   print(f"模型可用: {predictor.is_available}")
   ```

3. **开口率预测不准确**
   ```python
   # 确保使用校准后的配置
   from src.models.aperture_model import EnhancedApertureModel
   model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')
   
   # 验证 20V 开口率
   theta = model.get_contact_angle(20)
   eta = model.contact_angle_to_aperture_ratio(theta)
   print(f"20V 开口率: {eta*100:.1f}% (应为 ~67%)")
   ```

4. **参数硬编码问题**
   ```python
   # 检查参数是否从配置文件读取
   print(f"k = {model.aperture_k}")  # 应为 0.8
   print(f"theta_scale = {model.aperture_theta_scale}")  # 应为 6.0
   print(f"alpha = {model.aperture_alpha}")  # 应为 0.05
   ```

---

## 📊 物理机制说明

### 电润湿机制

1. **电润湿作用在极性液体上**（不是油墨）
2. **极性液体铺展**，将油墨从像素中心挤向边缘/角落
3. **油墨亲疏水层**（底部 Teflon），不亲围堰壁（相对亲水）
4. **油墨贴底收缩**，形成液滴，不会主动爬墙
5. **翻墙条件**：20V 以上油墨被挤压到极限可能翻墙

### φ 场定义（标准 VOF）

- **φ=1**: 纯油墨
- **φ=0**: 纯极性液体（透明）
- **0<φ<1**: 界面过渡区
- **开口率**: η = 底面 φ<0.5 的面积比例

---

**更新**: 2025-12-10 | **状态**: ✅ Stage 1 已校准 | ✅ Stage 2 已验证
