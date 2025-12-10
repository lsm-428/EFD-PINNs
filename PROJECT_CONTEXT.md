# EFD-PINNs 项目完整 Context

**最后更新**: 2025-12-10  
**当前状态**: ✅ Stage 1 已校准 | ✅ Stage 2 已验证

---

## 🎯 项目目标

使用物理信息神经网络 (PINN) 预测电润湿显示器件的动态响应。

**电润湿显示工作原理**：
- 无电压（关态）：油墨平铺在像素底部 → 显色状态
- 施加电压（开态）：极性液体润湿疏水层 → 油墨被动收缩 → 形成开口率 → 透明

**关键理解**：电润湿作用在极性液体上，油墨是被动的

---

## 📁 项目结构

```
EFD3D/
├── src/                            # 源代码目录
│   ├── models/                     # 模型定义
│   │   ├── pinn_two_phase.py      # 两相流 PINN 模型
│   │   ├── aperture_model.py      # 开口率模型（已校准）
│   │   └── optimized_ewpinn.py    # 优化 PINN 模型
│   ├── physics/                    # 物理约束
│   ├── training/                   # 训练相关
│   ├── predictors/                 # 预测器
│   │   ├── hybrid_predictor.py    # 混合预测器
│   │   └── pinn_aperture.py       # PINN 开口率预测器
│   ├── solvers/                    # 求解器
│   ├── visualization/              # 可视化
│   └── utils/                      # 工具函数
├── config/                         # 配置文件
│   └── stage6_wall_effect.json    # 校准后的配置
├── tests/                          # 测试文件
├── docs/                           # 文档
├── outputs_pinn_*/                 # 训练输出
├── train_contact_angle.py          # Stage 1 训练入口
├── train_two_phase.py              # Stage 2 训练入口
├── visualize_pinn_results.py       # 可视化脚本
└── validate_pinn_physics.py        # 物理验证脚本
```

---

## 🔧 核心物理

### Young-Lippmann 方程（含阈值电压）
```
cos(θ) = cos(θ₀) + ε₀εᵣ(V-V_T)²/(2γd)
```

### 二阶欠阻尼响应
```
θ(t) = θ_eq + (θ₀-θ_eq)·e^(-ζω₀t)·[cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
```

### 已校准参数

| 参数 | 值 | 说明 |
|------|-----|------|
| θ₀ | 120° | 初始接触角 |
| εᵣ (SU-8) | **3.0** | 介电层介电常数 |
| εₕ (Teflon) | **1.9** | 疏水层介电常数 |
| γ | **0.050 N/m** | 极性液体表面张力 |
| d | 400nm | SU-8 厚度 |
| dₕ | 400nm | Teflon 厚度 |
| V_T | **3V** | 阈值电压 |
| τ | 5 ms | 时间常数 |
| ζ | 0.8 | 阻尼比 |

### 开口率映射参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 0.8 | 陡度参数 |
| theta_scale | 6.0 | 角度缩放因子 |
| alpha | 0.05 | 电容反馈强度 |
| aperture_max | 0.85 | 最大开口率 |

---

## 🔬 物理机制

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

## 🚀 快速开始

```python
# Stage 1: 开口率预测（已校准）
from src.models.aperture_model import EnhancedApertureModel

model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')
theta = model.get_contact_angle(20)
eta = model.contact_angle_to_aperture_ratio(theta)
print(f"20V: θ={theta:.1f}°, η={eta*100:.1f}%")  # 115.2°, 66.7%

# Stage 1: 接触角动态响应
from src.predictors import HybridPredictor

predictor = HybridPredictor(config_path='config/stage6_wall_effect.json')
theta = predictor.predict(voltage=20, time=0.01)
print(f"接触角: {theta:.1f}°")

# Stage 2: PINN φ 场预测
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=20, time=0.02)
print(f"PINN 开口率: {eta:.3f}")  # ~0.736
```

---

## 📊 项目成果

### Stage 1: 接触角 + 开口率预测（已校准）

| 指标 | 目标 | 实现 | 状态 |
|------|------|------|------|
| 20V 开口率 | 67% | 66.7% | ✅ 误差 0.3% |
| 稳态精度 (30V) | <3° | 0.7° | ✅ |
| 角度变化 | 33° | 30.1° | ✅ |
| 超调 | <15% | 3.9% | ✅ |
| 响应时间 | <30ms | 13ms | ✅ |

### Stage 2: 两相流 PINN（已验证）

| 电压 | Stage 1 η | PINN η | 误差 | 状态 |
|------|-----------|--------|------|------|
| 0V | 0% | 0% | 0% | ✅ |
| 10V | 10.3% | 9.2% | -1.1% | ✅ |
| 20V | 66.7% | 73.6% | +6.9% | ✅ |
| 30V | 84.4% | 84.6% | +0.2% | ✅ |

**物理验证**: 16/16 测试点通过

---

## 📖 相关文档

- [README.md](README.md) - 项目概述
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - 当前状态
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - 项目路线图
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - 使用指南
- [TRAINING_HISTORY.md](TRAINING_HISTORY.md) - 训练历史
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - 更新日志
- [docs/specs/DEVICE_SPECS.md](docs/specs/DEVICE_SPECS.md) - 器件规格
