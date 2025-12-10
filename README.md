# EFD-PINNs: 电润湿显示动力学预测

**Physics-Informed Neural Networks for Electrowetting Display Dynamics**

[![Status](https://img.shields.io/badge/status-Stage1_Stage2_Complete-green)](CURRENT_STATUS.md)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🎉 项目成果

### Stage 1: 接触角预测 ✅ 已校准

| 指标 | 目标 | 实现 | 状态 |
|------|------|------|------|
| 20V 开口率 | 67% | 66.7% | ✅ 误差 0.3% |
| 稳态精度 (30V) | <3° | 0.7° | ✅ |
| 角度变化 | 33° | 30.1° | ✅ |
| 超调 | <15% | 3.9% | ✅ |
| 响应时间 | <30ms | 13ms | ✅ |

### Stage 2: 两相流 PINN ✅ 已验证

| 电压 | Stage 1 η | PINN η | 状态 |
|------|-----------|--------|------|
| 0V | 0% | 0% | ✅ |
| 10V | 10.3% | 9.2% | ✅ |
| 20V | 66.7% | 73.6% | ✅ |
| 30V | 84.4% | 84.6% | ✅ |

---

## 🔬 电润湿显示工作原理

```
无电压（关态）：油墨平铺在像素底部 → 显色状态
施加电压（开态）：极性液体润湿疏水层 → 油墨被动收缩 → 形成开口率 → 透明
```

**关键理解**：电润湿作用在极性液体上，油墨是被动的

**像素结构**：ITO电极 → SU-8介电层(400nm) → Teflon疏水层(400nm) → 油墨+极性液体 → 顶层ITO

**关键概念**：开口率 = 透明区域面积 / 像素面积，决定像素亮度

---

## 🚀 快速开始

### 1. 环境准备

```bash
conda activate efd
```

### 2. Stage 1: 开口率预测

```python
from src.models.aperture_model import EnhancedApertureModel

model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')

# 预测开口率
theta = model.get_contact_angle(20)  # 20V
eta = model.contact_angle_to_aperture_ratio(theta)
print(f"20V 开口率: {eta*100:.1f}%")  # 66.7%
```

### 3. Stage 2: PINN φ 场预测

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=20, time=0.02)
print(f"PINN 开口率: {eta:.3f}")  # ~0.736
```

---

## 🔬 核心物理

### Young-Lippmann 方程 (稳态)
```
cos(θ) = cos(θ₀) + ε₀εᵣ(V-V_T)²/(2γd)
```

### 二阶欠阻尼响应 (动态)
```
θ(t) = θ_eq + (θ₀-θ_eq)·e^(-ζω₀t)·[cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
```

### 已校准参数

| 参数 | 值 | 说明 |
|------|-----|------|
| θ₀ | 120° | 初始接触角 |
| εᵣ (SU-8) | 3.0 | 介电层介电常数 |
| εᵣ (Teflon) | 1.9 | 疏水层介电常数 |
| γ | 0.050 N/m | 极性液体表面张力 |
| V_T | 3V | 阈值电压 |
| τ | 5 ms | 时间常数 |
| ζ | 0.8 | 阻尼比 |

---

## 📁 项目结构

```
EFD3D/
├── src/                            # 源代码目录
│   ├── models/                     # 模型定义
│   │   ├── pinn_two_phase.py      # 两相流 PINN
│   │   └── aperture_model.py      # 开口率模型（已校准）
│   ├── predictors/                 # 预测器
│   │   ├── hybrid_predictor.py    # 混合预测器
│   │   └── pinn_aperture.py       # PINN 开口率预测器
│   ├── physics/                    # 物理约束
│   ├── training/                   # 训练相关
│   └── utils/                      # 工具函数
│
├── config/                         # 配置文件
│   └── stage6_wall_effect.json    # 校准后的配置
│
├── tests/                          # 测试文件
├── docs/                           # 文档目录
├── outputs_pinn_*/                 # 训练输出
│
├── train_contact_angle.py          # Stage 1 训练入口
├── train_two_phase.py              # Stage 2 训练入口
├── validate_pinn_physics.py        # 物理验证脚本
└── visualize_pinn_results.py       # 可视化脚本
```

---

## 📊 预期结果

### 稳态预测 (Young-Lippmann + 开口率映射)

| 电压 | 接触角 | 开口率 | 状态 |
|------|--------|--------|------|
| 0V | 120.0° | 0% | 关态(显色) |
| 6V | ~119° | ~1% | 开始响应 |
| 10V | 119.2° | 10.3% | |
| 20V | 115.2° | 66.7% | **实验验证** |
| 30V | 108.2° | 84.4% | 开态(透明) |

---

## 🎯 项目路线图

```
电压变化 → 接触角变化 → 油墨被排开 → 开口率 → 像素亮度
   因          因           果          果        果
   
Stage 1       ←─────── Stage 2 ───────→
(✅ 已校准)        (✅ 已验证)
```

---

## 📖 文档

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - 详细使用指南
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - 项目路线图
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - 项目完整 Context
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - 当前状态
- [TRAINING_HISTORY.md](TRAINING_HISTORY.md) - 训练历史记录
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - 更新日志

---

## ✅ 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 物理验证
python validate_pinn_physics.py
```

---

**更新**: 2025-12-10 | **状态**: ✅ Stage 1 已校准 | ✅ Stage 2 已验证
