# 📚 EFD-PINNs 文档中心

**最后更新**: 2025-12-10

---

## 🎯 快速导航

### 入门
- **[../README.md](../README.md)** - 项目入口
- **[guides/quickstart.md](guides/quickstart.md)** - 快速开始
- **[../CURRENT_STATUS.md](../CURRENT_STATUS.md)** - 当前状态
- **[../USAGE_GUIDE.md](../USAGE_GUIDE.md)** - 使用指南

### 技术文档
- **[../PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)** - 完整技术背景
- **[specs/PROJECT_ARCHITECTURE.md](specs/PROJECT_ARCHITECTURE.md)** - 项目架构
- **[specs/MODULE_OVERVIEW.md](specs/MODULE_OVERVIEW.md)** - 模块概览
- **[architecture/model_architecture.md](architecture/model_architecture.md)** - 模型架构

### API参考
- **[api/README.md](api/README.md)** - API概览
- **[api/core_models.md](api/core_models.md)** - 核心模型
- **[api/physics_constraints.md](api/physics_constraints.md)** - 物理约束

### 配置与贡献
- **[CONFIG_TEMPLATE.md](CONFIG_TEMPLATE.md)** - 配置模板
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - 贡献指南
- **[CHANGELOG.md](CHANGELOG.md)** - 更新日志

---

## 📁 文档结构

```
docs/
├── README.md                    # 本文档
├── CHANGELOG.md                 # 更新日志
├── CONTRIBUTING.md              # 贡献指南
├── CONFIG_TEMPLATE.md           # 配置模板
│
├── guides/                      # 📘 使用指南
│   ├── quickstart.md           # 快速开始
│   ├── installation_and_configuration.md  # 安装配置
│   ├── training_strategies.md  # 训练策略
│   ├── advanced_training_strategies.md  # 高级训练
│   ├── physics_constraints.md  # 物理约束
│   ├── configuration_system.md # 配置系统
│   └── troubleshooting_debugging.md  # 故障排除
│
├── specs/                       # 📋 技术规格
│   ├── DEVICE_SPECS.md         # 器件规格
│   ├── PROJECT_ARCHITECTURE.md # 项目架构
│   ├── MODULE_OVERVIEW.md      # 模块概览
│   └── MODULE_DEPENDENCIES.md  # 模块依赖
│
├── api/                         # 💻 API文档
│   ├── README.md               # API概览
│   ├── core_models.md          # 核心模型
│   ├── physics_constraints.md  # 物理约束
│   ├── training_system.md      # 训练系统
│   ├── input_output_layers.md  # 输入输出层
│   └── examples_and_best_practices.md  # 示例与最佳实践
│
├── architecture/                # 🏗️ 架构文档
│   └── model_architecture.md   # 模型架构详解
│
└── reports/                     # 📊 训练报告
    └── TRAINING_REPORTS.md     # 训练报告汇总
```

---

## 🔍 按需求查找

| 需求 | 文档 |
|------|------|
| 快速开始 | [guides/quickstart.md](guides/quickstart.md) |
| 当前进展 | [../CURRENT_STATUS.md](../CURRENT_STATUS.md) |
| 使用指南 | [../USAGE_GUIDE.md](../USAGE_GUIDE.md) |
| 配置模板 | [CONFIG_TEMPLATE.md](CONFIG_TEMPLATE.md) |
| 模块概览 | [specs/MODULE_OVERVIEW.md](specs/MODULE_OVERVIEW.md) |
| 器件参数 | [specs/DEVICE_SPECS.md](specs/DEVICE_SPECS.md) |
| 训练策略 | [guides/training_strategies.md](guides/training_strategies.md) |
| 故障排除 | [guides/troubleshooting_debugging.md](guides/troubleshooting_debugging.md) |
| API文档 | [api/README.md](api/README.md) |
| 更新日志 | [CHANGELOG.md](CHANGELOG.md) |

---

## 🏗️ 项目架构概览

EFD-PINNs 采用两阶段预测架构：

### Stage 1: 接触角 + 开口率预测 ✅ 已校准
- 使用解析公式 (Young-Lippmann + 二阶欠阻尼响应)
- 开口率映射（已校准：20V→67%）
- 入口: `train_contact_angle.py`
- 核心: `src/models/aperture_model.py`, `src/predictors/hybrid_predictor.py`
- 配置: `config/stage6_wall_effect.json`

### Stage 2: 两相流 PINN ✅ 已验证
- 从 φ 场预测开口率
- 入口: `train_two_phase.py`
- 核心: `src/models/pinn_two_phase.py`, `src/predictors/pinn_aperture.py`
- 验证: `validate_pinn_physics.py`

---

## 📊 已校准的物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| SU-8 厚度 | 400nm | 介电层 |
| SU-8 介电常数 | ε=3.0 | |
| Teflon 厚度 | 400nm | 疏水层 |
| Teflon 介电常数 | ε=1.9 | |
| 极性液体表面张力 | γ=0.050 N/m | 乙二醇混合液 |
| 阈值电压 | V_T=3V | |
| 初始接触角 | θ₀=120° | |

### 开口率映射参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 0.8 | 陡度参数 |
| theta_scale | 6.0 | 角度缩放因子 |
| alpha | 0.05 | 电容反馈强度 |
| aperture_max | 0.85 | 最大开口率 |

---

## 📂 源码结构

```
src/
├── models/              # 神经网络模型
│   ├── aperture_model.py      # 开口率模型（已校准）
│   └── pinn_two_phase.py      # 两相流 PINN
├── predictors/          # 预测器
│   ├── hybrid_predictor.py    # Stage 1 混合预测器
│   └── pinn_aperture.py       # Stage 2 PINN 预测器
├── physics/             # 物理约束
├── training/            # 训练系统
├── utils/               # 工具函数
└── visualization/       # 可视化
```

---

## 🎯 项目成果

### Stage 1 (已校准)
- 20V 开口率: 66.7% (实验值 67%，误差 0.3%)
- 稳态精度: 0.7°
- 响应时间: 13ms

### Stage 2 (已验证)
- φ 范围检查: 16/16 通过
- V=30V 开口率: 84.6% (Stage 1: 84.4%)
- 开口率随时间增加（物理正确）

---

**返回**: [项目主页](../README.md)
