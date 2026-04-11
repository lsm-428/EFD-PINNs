# EFD-PINNs 使用指南

**最后更新**: 2026-02-04
**适用版本**: v4.5

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活环境
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.config import PHYSICS; print('EFD3D 导入成功')"
```

### 2. Stage 1: 开口率预测（已校准）

```python
from src.models.aperture_model import EnhancedApertureModel

# 创建模型（使用默认配置）
model = EnhancedApertureModel()

# 预测开口率
for V in [0, 10, 20, 30]:
    result = model.predict(V)
    print(f"V={V}V: θ={result['theta']:.1f}°, η={result['aperture_percent']:.1f}%")
```

### 3. Stage 1: 接触角动态响应

```python
from src.predictors import HybridPredictor

# 初始化预测器
predictor = HybridPredictor()

# 单点预测
theta = predictor.predict(voltage=20, time=0.01)
print(f"20V, 10ms 时接触角: {theta:.1f}°")

# 阶跃响应
t, theta = predictor.step_response(V_start=0, V_end=20, duration=0.02)

# 方波响应
t, V, theta, eta = predictor.square_wave_response(V_low=0, V_high=20)

# 获取响应指标
metrics = predictor.get_response_metrics(t, theta)
print(f"响应时间: {metrics['t_90_ms']:.2f} ms")
print(f"超调: {metrics['overshoot_percent']:.1f}%")
```

### 4. Stage 2: PINN φ 场预测与验证

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

# 初始化预测器（自动加载最新模型）
predictor = PINNAperturePredictor()

# 预测开口率
eta = predictor.predict(voltage=20, time=0.02)
print(f"PINN 开口率: {eta:.3f}")

# 预测 φ 场
phi_field = predictor.predict_phi_field(voltage=20, time=0.02)
```

#### 自动化验证脚本

```bash
# 综合评估（自动选择最新 outputs/train/pinn_*）
python evaluate.py

# 对比最近两个模型
python evaluate.py --compare
```

---

## 📊 训练模型

### Stage 1: 接触角预测（解析模型，无需训练）

Stage 1 使用解析物理模型（HybridPredictor），**无需训练**，直接使用即可。

```python
from src.predictors.hybrid_predictor import HybridPredictor

predictor = HybridPredictor()
theta = predictor.predict(voltage=20, time=0.02)
print(f"20V 接触角: {theta:.1f}°")
```

### Stage 2: 两相流 PINN 训练

```bash
# 使用默认配置训练
python train_two_phase.py

# 自定义 epochs
python train_two_phase.py --epochs 60000

# 快速冒烟测试
python train_two_phase.py --epochs 1000

# 输出目录：outputs/train/pinn_<timestamp>/
# 训练日志：outputs/train/pinn_<timestamp>/training.log
```

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
| `train_two_phase.py` | Stage 2 PINN 训练 |
| `evaluate.py` | 模型评估与可视化 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `config/v4.5-standard.json` | ⭐ 推荐配置（已验证收敛） |
| `config/device_calibrated_physics.json` | 核心物理配置 |
| `config/v4.5_lbfgs_tuned.json` | L-BFGS 微调配置（备选） |

---

## 🔍 故障排除

### 常见问题

1. **模块导入失败**
   ```bash
   source .venv/bin/activate
   ```

2. **PINN 模型不可用**
   ```python
   from src.predictors.pinn_aperture import PINNAperturePredictor
   predictor = PINNAperturePredictor()
   print(f"模型可用: {predictor.is_available}")
   ```

3. **开口率预测不准确**
   ```python
   from src.models.aperture_model import EnhancedApertureModel
   from src.config import get_physics_config

   config = get_physics_config()
   model = EnhancedApertureModel(config)

   result = model.predict(20)
   print(f"20V 开口率: {result['aperture_percent']:.1f}%")
   ```

4. **参数硬编码问题**
   ```python
   from src.config import PHYSICS
   # 详细参数说明请参见[物理理论与器件规格指南](physics_and_device_guide.md#physics-parameters)
   print(f"theta0: {PHYSICS['theta0']}")
   print(f"epsilon_r: {PHYSICS['epsilon_r']}")
   print(f"gamma: {PHYSICS['gamma']}")
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

**更新**: 2026-02-04 | **状态**: ✅ Stage 1 已校准 | ✅ Stage 2 可训练
