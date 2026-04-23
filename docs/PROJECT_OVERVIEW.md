# EFD3D 项目深度概览

**最后更新**: 2026年4月21日
**版本**: v4.5

## 🎯 项目概述

EFD3D (Electrowetting Fluid Dynamics 3D) 是一个工业级的物理信息神经网络 (PINN) 仿真框架，专为解决微流控和电子纸显示技术中的三维两相流问题而设计。

### 核心创新点

1. **6D Triad 输入表示**: `(x, y, z, V_from, V_to, t_since)` - 支持任意电压序列的连续模拟
2. **两阶段混合架构**: Stage 1 解析模型 + Stage 2 PINN 完整流场求解
3. **物理约束完整性**: Navier-Stokes + VOF + 电润湿力场 + 界面张力
4. **工业级精度**: 30V 开口率 83.4%，体积守恒误差 <1%

## 🏗️ 系统架构

### 核心模块

```
EFD3D/
├── src/                           # 核心源代码 (21个模块)
│   ├── models/                   # 神经网络模型
│   │   ├── pinn_two_phase.py     # 主PINN模型 (TwoPhasePINN + Trainer)
│   │   └── aperture_model.py     # Stage 1 解析模型
│   ├── physics/                  # 物理约束引擎
│   │   └── constraints.py        # Navier-Stokes, VOF, 连续性方程
│   ├── training/                 # 训练基础设施
│   │   ├── scheduler.py          # 动态损失权重调度
│   │   ├── stabilizer.py         # NaN恢复, 梯度裁剪
│   │   └── components.py         # 训练工具
│   ├── config/                   # 配置管理
│   │   ├── __init__.py           # PHYSICS参数导出
│   │   └── physics_config.py     # 类型安全配置
│   └── utils/                    # 工具函数
│       ├── model_utils.py        # 模型加载和预测提取
│       └── logging_config.py     # 统一日志配置
├── config/                       # 训练配置
│   ├── v4.5-standard.json        # 推荐配置 (已验证收敛)
│   └── device_calibrated_physics.json # 物理校准
├── scripts/                      # 用户工具
│   ├── dashboard.py              # Streamlit交互式面板
│   └── run_ablation.sh           # 消融研究脚本
├── tests/                        # 测试套件 (15个模块)
│   ├── test_pinn_complete.py     # 端到端PINN训练管道
│   ├── test_physics_sanity.py    # 物理验证检查
│   └── test_vof_transport.py     # VOF方程验证
└── outputs/                      # 训练输出
    └── train/                    # 模型检查点和日志
```

### 关键依赖关系

- **入口点**: `train_two_phase.py`, `evaluate.py`, `scripts/dashboard.py`
- **核心模型**: `src/models/pinn_two_phase.py` (TwoPhasePINN类)
- **物理引擎**: `src/physics/constraints.py` (PhysicsConstraints类)
- **配置**: `src/config/__init__.py` (PHYSICS参数), `src/config/physics_config.py`
- **训练控制**: `src/training/scheduler.py`, `src/training/stabilizer.py`
- **预测器**: `src/predictors/hybrid_predictor.py`, `src/predictors/pinn_aperture.py`
- **工具**: `src/utils/model_utils.py` (模型加载), `src/utils/logging_config.py` (日志设置)

## ⚙️ 物理模型

### 控制方程

1. **连续性方程**: ∇·u = 0
2. **VOF输运**: ∂φ/∂t + u·∇φ = 0
3. **Navier-Stokes**: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st

### 关键物理参数

```python
# 从 src/config/physics_config.py
PHYSICS = {
    # 几何参数
    "Lx": 174e-6,           # 像素宽度 (m)
    "Ly": 174e-6,           # 像素高度 (m)
    "Lz": 20e-6,            # 围堰/流体层高度 (m)
    "h_ink": 3e-6,          # 油墨层厚度 (m)
    "h_polar": 17e-6,       # 极性液体层厚度 (m)
    "wall_height": 3.5e-6,  # 围堰高度 (m)

    # 流体属性
    "rho_oil": 800.0,       # 油墨密度 (kg/m³)
    "mu_oil": 0.003,        # 油墨粘度 (Pa·s)
    "rho_polar": 1000.0,    # 极性液体密度 (kg/m³)
    "mu_polar": 0.001,      # 极性液体粘度 (Pa·s)
    "sigma": 0.045,         # 界面张力 (N/m)
    "gamma": 0.015,         # 表面张力 (N/m)

    # 电学参数
    "epsilon_0": 8.854e-12, # 真空介电常数 (F/m)
    "epsilon_r": 12.0,      # 介电层相对介电常数
    "epsilon_h": 1.9,       # 疏水层相对介电常数
    "d_dielectric": 4e-7,   # 介电层厚度 (m)
    "d_hydrophobic": 4e-7,  # 疏水层厚度 (m)

    # 接触角参数
    "theta0": 120.0,        # 本征接触角 (度)
    "theta_wall": 71.0,     # 围堰壁接触角 (度)
    "theta_min": 60.0,      # 最小接触角 (度)

    # 动力学参数
    "tau": 0.005,           # 电润湿响应时间常数 (s)
    "tau_recovery": 0.0075, # 表面张力恢复时间常数 (s)
    "zeta": 0.8,            # 阻尼比
    "t_max": 0.05,          # 最大仿真时间 (s)

    # 电压参数
    "V_threshold": 3.0,     # 阈值电压 (V)
    "V_max": 30.0,          # 最大工作电压 (V)

    # 开口率参数
    "eta_max": 0.85,        # 最大开口率
    "ink_initial_fraction": 0.15,  # 初始油墨体积分数
}
```

## 📊 训练策略

### 三阶段渐进式训练

1. **Stage 1 (几何阶段)**: 1,500轮 - 学习几何约束和边界条件
2. **Stage 2 (运动学阶段)**: 4,000轮 - 学习VOF输运和界面动力学
3. **Stage 3 (完整物理阶段)**: 54,500轮 - 学习Navier-Stokes方程

### 动态损失权重调度

```python
# config/v4.5-standard.json
"physics": {
    "interface_weight": 500.0,      # 界面约束权重
    "ic_weight": 300.0,             # 初始条件权重
    "bc_weight": 80.0,              # 边界条件权重
    "continuity_weight": 0.5,       # 连续性方程权重
    "vof_weight": 0.5,              # VOF输运方程权重
    "ns_weight": 0.1,               # Navier-Stokes方程权重
    "surface_tension_weight": 0.01, # 表面张力权重
    "sharpening_weight": 1.0,       # 界面锐化权重
    "explicit_volume_weight": 100.0 # 显式体积守恒权重
}
```

### 训练配置参数

```python
# config/v4.5-standard.json
"training": {
    "epochs": 60000,                # 总训练轮次
    "batch_size": 4096,             # 批量大小
    "learning_rate": 0.0003,        # 学习率
    "min_lr": 1e-06,               # 最小学习率
    "gradient_clip": 1.0,           # 梯度裁剪
    "stage1_epochs": 1500,          # 阶段1轮次
    "stage2_epochs": 4000,          # 阶段2轮次
    "stage3_epochs": 60000,         # 阶段3轮次
    "early_stop_patience": 60000,   # 早停耐心
    "warmup_epochs": 500,           # 预热轮次
    "volume_n_vol": 20000,          # 体积采样点数
    "volume_base_weight": 2000.0,   # 体积基础权重
}
```

## 🧪 测试体系

### 测试组织

```
tests/
├── test_pinn_complete.py              # 端到端PINN训练管道
├── test_physics_sanity.py             # 物理验证检查
├── test_vof_transport.py              # VOF方程验证
├── test_hybrid_predictor.py           # Stage 1+2集成
├── test_enhanced_aperture_properties.py # Stage 1模型验证
├── test_dynamic_weights.py            # 损失权重调度
├── test_flow_solver_properties.py     # CFD求解器验证
├── test_two_phase_data_generator.py   # 数据生成测试
├── test_vof_3d.py                     # 3D VOF实现
├── test_curvature_computation.py      # 曲率计算
├── test_vof_sensitivity.py            # VOF敏感性分析
├── test_model_dimensions.py           # 模型架构验证
├── test_scripts_framework.py          # 脚本功能
├── test_3d_visualization_properties.py # 3D可视化
└── test_code_changes.py               # 代码修改跟踪
```

### 常见测试模式

- **物理验证**: 验证控制方程残差 < 1e-3
- **守恒定律**: 质量/体积守恒误差 < 1%
- **边界条件**: 无滑移壁面，界面连续性
- **阶段集成**: Stage 1 → Stage 2兼容性

## 📈 性能指标

### 当前里程碑

| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83.4%** | 像素开口率优化 |
| 体积误差 | **<1%** | VOF输运方程体积守恒 |
| 训练轮次 | **60,000** | 三阶段渐进式训练 |
| 输入表示 | **6D Triad** | (x, y, z, V_from, V_to, t_since) |
| 代码规模 | **57文件** | 41 src + 1 scripts + 15 tests |
| 代码行数 | **13,937+** | Python代码 |

### 器件规格

- **像素尺寸**: 174μm × 174μm
- **围堰高度**: 20μm（模型）/ 3.5μm（实际）
- **油墨层**: 3μm
- **工作电压范围**: 0-30V
- **典型工作电压**: 20V
- **介电层**: SU-8 (400nm, ε=3.0)
- **疏水层**: Teflon (400nm, ε=1.9)
- **表面张力**: 0.015 N/m
- **油墨密度**: 1000 kg/m³
- **初始接触角**: 120°

## 🚀 快速开始

### 训练

```bash
# 使用推荐配置训练
uv run train_two_phase.py --config config/v4.5-standard.json

# 快速测试 (1,000轮)
uv run train_two_phase.py --epochs 1000

# 从检查点恢复训练
uv run train_two_phase.py --config config/v4.5-standard.json \
    --resume_from outputs/train/pinn_YYYYMMDD_HHMMSS/best_model.pth
```

### 评估与可视化

```bash
# 运行评估并生成可视化
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/

# 启动交互式面板
uv run scripts/dashboard.py

# 运行消融研究
uv run scripts/run_ablation.sh

# 启动TensorBoard进行训练监控
tensorboard --logdir outputs/train/pinn_YYYYMMDD_HHMMSS/runs/
```

### 测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行特定测试模块
uv run pytest tests/test_pinn_complete.py -v

# 运行测试并生成详细输出
uv run pytest tests/ -v --tb=short

# 生成覆盖率报告
uv run pytest tests/ --cov=src --cov-report=html
```

## 📚 文档结构

### 用户指南

- **[快速开始指南](docs/guides/quickstart.md)** - 新用户完整学习路径
- **[训练指南](docs/guides/training_guide.md)** - 详细训练策略
- **[配置指南](docs/guides/configuration_guide.md)** - 配置系统详解
- **[可视化指南](docs/guides/visualization_guide.md)** - 结果可视化方法
- **[数据采样策略](docs/guides/data_sampling_evolution.md)** - 数据生成演进与优化

### 开发者文档

- **[开发者指南](CLAUDE.md)** - 配置、训练、API使用和调试
- **[项目架构](docs/architecture/system_design.md)** - 系统架构设计
- **[模型架构](docs/architecture/model_architecture.md)** - 神经网络架构

### 研究文档

- **[研究人员指南](docs/research/DEEP_UNDERSTANDING_GUIDE.md)** - 物理理论和技术理解
- **[训练报告](docs/research/training_reports.md)** - 训练实验记录
- **[物理理论与器件规格](docs/guides/physics_and_device_guide.md)** - 完整物理参数

## 🔧 开发工具

### 环境管理

- **包管理器**: `uv` (见 `pyproject.toml`)
- **Python版本**: 3.12-3.13
- **GPU加速**: CUDA 11.8 (PyTorch 2.7.1)

### 代码质量

- **代码检查**: `ruff` (配置在 `pyproject.toml`)
- **格式化**: `black` (行长度88)
- **测试**: `pytest` + `hypothesis`

### 输出结构

训练输出组织如下：

```
outputs/train/pinn_YYYYMMDD_HHMMSS/
├── best_model.pth               # 最佳模型权重
├── best_model_epoch_XXXXX.pth   # 特定轮次检查点
├── training.log                 # 训练进度日志
├── interface_3d_steady.png      # 3D界面可视化
├── training_curve.png           # 损失曲线
└── config.json                  # 训练配置快照
```

## 🔍 调试策略

### 常见问题

- **训练不稳定**: 检查 `TrainingStabilizer` 日志进行NaN恢复
- **物理违规**: 使用 `test_physics_sanity.py` 进行方程验证
- **性能问题**: 使用面板性能指标进行配置
- **界面问题**: 使用 `test_vof_transport.py` 进行验证

### 关键代码导航

- **Navier-Stokes**: `src/physics/constraints.py` 120-180行
- **VOF输运**: `src/physics/constraints.py` 200-250行
- **电润湿力**: `src/physics/constraints.py` 300-350行

## 📊 版本历史

| 版本 | 发布日期 | 核心改进 |
|------|---------|---------|
| **v4.5** | 2026-01-29 | 界面锐化损失，30V开口率83.4%，体积误差<1% |
| v4.4 | 2026-01-13 | Stage 1 Tutor约束 |
| v4.3 | 2026-01-08 | 基础PINN架构 |

**推荐配置**: `config/v4.5-standard.json` (已验证收敛)

---

*本概览基于EFD3D v4.5版本，最后更新于2026年4月21日*
