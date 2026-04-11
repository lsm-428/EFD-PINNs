# 📚 EFD-PINNs 文档中心

**最后更新**: 2026-02-04

---

## 🎯 快速导航

### 入门
- **[../README.md](../README.md)** - 项目入口
- **[../CLAUDE.md](../CLAUDE.md)** - 项目核心文档（包含知识库系统）⭐
- **[guides/installation.md](guides/installation.md)** - 安装与环境配置
- **[guides/usage.md](guides/usage.md)** - 使用指南
- **[guides/visualization_guide.md](guides/visualization_guide.md)** - 可视化指南

### 技术文档
- **[architecture/system_design.md](architecture/system_design.md)** - 系统架构设计
- **[guides/physics_and_device_guide.md](guides/physics_and_device_guide.md)** - 物理理论与器件规格

### 训练与验证
- **[research/TRAINING_REPORTS.md](research/TRAINING_REPORTS.md)** - 训练报告与结果
- **[guides/training_guide.md](guides/training_guide.md)** - 训练策略
- **[guides/troubleshooting.md](guides/troubleshooting.md)** - 故障排除
- **[architecture/problem_analysis/volume_conservation_complete_analysis.md](architecture/problem_analysis/volume_conservation_complete_analysis.md)** - 体积守恒问题深度分析

### 配置与贡献
- **[guides/configuration_guide.md](guides/configuration_guide.md)** - 配置系统详解
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - 贡献指南
- **[CHANGELOG.md](CHANGELOG.md)** - 更新日志

---

## 📁 文档结构

```
docs/
├── README.md                    # 本文档
├── CHANGELOG.md                 # 更新日志
├── CONTRIBUTING.md              # 贡献指南
│
├── guides/                      # 📘 使用指南
│   ├── quickstart.md           # 快速开始
│   ├── installation.md           # 安装配置
│   ├── data-generation-core.md   # 数据生成核心
│   ├── training_guide.md        # 训练策略与高级优化
│   ├── physics_and_device_guide.md  # 物理理论与器件规格
│   ├── configuration_guide.md      # 配置系统与权重
│   ├── troubleshooting.md        # 故障排除
│   ├── scripts_guide.md          # 脚本工具使用指南 ⭐
│   └── visualization_guide.md    # 可视化指南
│
├── api/                         # 💻 API 文档
│   ├── README.md               # API 概览
│   ├── core_models.md          # 核心模型
│   ├── physics_constraints.md  # 物理约束
│   ├── training_system.md      # 训练系统
│   ├── input_output_layers.md  # 输入输出层
│   └── examples_and_best_practices.md  # 示例与最佳实践
│
├── architecture/                # 🏗️ 架构文档
│   ├── system_design.md       # 系统设计与架构
│   ├── model_architecture.md   # 模型架构详解
│   ├── design_evolution/       # 设计演进记录
│   └── problem_analysis/       # 问题分析
```

---

## 🔍 按需求查找

| 需求 | 文档 |
|------|------|
| 项目总览 | [../README.md](../README.md) |
| 核心文档 | [../CLAUDE.md](../CLAUDE.md) ⭐ |
| 安装配置 | [guides/installation.md](guides/installation.md) |
| 快速使用 | [guides/usage.md](guides/usage.md) |
| 系统架构 | [architecture/system_design.md](architecture/system_design.md) |
| 物理理论 | [guides/physics_and_device_guide.md](guides/physics_and_device_guide.md) |
| 训练策略 | [guides/training_guide.md](guides/training_guide.md) |
| 故障排除 | [guides/troubleshooting.md](guides/troubleshooting.md) |
| 可视化 | [guides/visualization_guide.md](guides/visualization_guide.md) |
| **新用户** | **[guides/quickstart.md](guides/quickstart.md)** |
| **开发者** | **[../CLAUDE.md](../CLAUDE.md)** |
| **研究人员** | **[research/DEEP_UNDERSTANDING_GUIDE.md](research/DEEP_UNDERSTANDING_GUIDE.md)** |

## 🆕 2026-02-04 重要更新

### 项目里程碑 (v4.5 / Stage 2)

| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83.4%** | 像素开口率优化至工业级水平 |
| 体积误差 | **<1%** | VOF 输运方程体积守恒修正成效 |
| 代码规模 | **41 文件** | 21 src + 5 scripts + 15 tests |
| 输入表示 | **6D Triad** | (x, y, z, V_from, V_to, t_since) |

### 技术参数详情 {#physics-parameters}

> 📖 **完整技术参数**: 材料参数、动态参数、训练配置等详见 **[物理理论与器件规格指南](guides/physics_and_device_guide.md#physics-parameters)**

#### 材料参数（校准值）

| 参数 | 符号 | 值 | 单位 | 描述 |
|-----------|--------|-------|------|-------------|
| 初始接触角 | `theta0` | 120.0 | 度 | Teflon表面的初始接触角 |
| 像素壁接触角 | `theta_wall` | 71.0 | 度 | 像素壁处的接触角 |
| 有效介电常数 | `epsilon_r` | 12.0 | - | 有效介电常数（SU-8 + 渗透效应） |
| 表面张力 | `gamma` | 0.015 | N/m | 有效界面张力 |
| 介电层厚度 | `d_dielectric` | 400e-9 | m | SU-8层厚度 |
| 疏水层厚度 | `d_hydrophobic` | 400e-9 | m | Teflon层厚度 |
| 阈值电压 | `V_threshold` | 3.0 | V | 电润湿阈值电压 |
| 真空介电常数 | `epsilon_0` | 8.854e-12 | F/m | 真空介电常数 |

#### 动态参数

| 参数 | 符号 | 值 | 单位 | 描述 |
|-----------|--------|-------|------|-------------|
| 时间常数 | `tau` | 5e-3 | s | 电润湿响应时间常数 |
| 恢复时间常数 | `tau_recovery` | 7.5e-3 | s | 电压下降期间的恢复时间常数 |
| 阻尼比 | `zeta` | 0.8 | - | 欠阻尼系统阻尼比 |

#### 几何参数

| 参数 | 符号 | 值 | 单位 | 描述 |
|-----------|--------|-------|------|-------------|
| 像素宽度 | `Lx` | 174e-6 | m | 像素宽度（内沿尺寸） |
| 像素长度 | `Ly` | 174e-6 | m | 像素长度 |
| 围堰高度 | `Lz` | 20e-6 | m | 围堰高度（模型值） |
| 油墨层厚度 | `h_ink` | 3e-6 | m | 油墨层厚度 |
| 实际围堰高度 | `h_wall_real` | 3.5e-6 | m | 实际围堰高度 |

### 训练配置 {#training-configuration}

#### 训练超参数

| 参数 | 默认值 | 描述 |
|-----------|---------------|-------------|
| 总训练轮次 | 60000 | 总训练轮次 |
| 批次大小 | 4096 | 训练批次大小 |
| 学习率 | 0.0003 | 基础学习率 |
| 第一阶段轮次 | 1500 | 几何阶段训练轮次 (0-1500) |
| 第二阶段轮次 | 4000 | 运动学阶段训练轮次 (1500-5500) |
| 第三阶段轮次 | 50000 | 完整物理阶段训练轮次 (5500-60000) |

#### 损失权重

| 损失项 | 默认权重 | 描述 |
|----------------|----------------|-------------|
| 界面 | 500.0 | 界面数据拟合 |
| 初始条件 | 300.0 | 初始条件约束 |
| 边界条件 | 80.0 | 边界条件约束 |
| 连续性 | 0.5 | 质量守恒（∇·u = 0） |
| VOF | 0.5 | 流体体积法输运方程 |
| Navier-Stokes | 0.1 | 动量守恒 |
| 表面张力 | 0.01 | 表面张力力 |
| 锐化 | 1.0 | 界面锐化损失 |
| 显式体积守恒 | 100.0 | 显式体积守恒约束 |

### 器件规格 {#device-specifications}

#### 像素尺寸

- **像素尺寸**: 174μm × 174μm
- **围堰高度**: 20μm（模型）/ 3.5μm（实际）
- **油墨层**: 3μm
- **工作电压范围**: 0-30V
- **典型工作电压**: 20V

#### 层结构

```
z = 0 μm
├─ 底层 ITO 玻璃 - 刚性界面
│
├─ 介电层 SU-8 (400nm, ε=3.0) - 电场隔离
│
├─ 疏水层 Teflon (400nm, ε=1.9) - 控制润湿性
│
├─ 围堰 SU-8 (3.5μm 实际 / 20μm 模型)
│  └─ 内部填充 (174×174μm):
│      ├─ 油墨层 (3μm) - 底部，疏水性
│      └─ 极性液体层 (17μm) - 顶部
│
└─ 顶层 ITO - 透明电极
```

### API 参考

#### 关键类和函数

##### `TwoPhasePINN` (`src/models/pinn_two_phase.py`)
- **输入**: 6D Triad `[x,y,z,V_from,V_to,t_since]`
- **输出**: 5D物理场 `[u,v,w,p,φ]`
- **网络**: 双分支MLP（速度网络 + φ网络）

##### `Trainer` (`src/models/pinn_two_phase.py`)
- **功能**: 管理三阶段训练
- **特性**: 动态权重调度，训练稳定性管理

##### `PhysicsConstraints` (`src/physics/constraints.py`)
- **实现**: 控制方程残差
  - 连续性: `∇·u = 0`
  - Navier-Stokes: `ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew`
  - VOF: `∂φ/∂t + ∇·(φu) = 0`

##### `EnhancedApertureModel` (`src/models/aperture_model.py`)
- **功能**: 基于Young-Lippmann方程的解析模型
- **输入**: `[V_from, V_to, t_since]`
- **输出**: `[θ, η]`（接触角，开口率）

### 训练进展

- **当前版本**: v4.5
- **核心改进**: 界面锐化 + 体积守恒强化
- **详细信息**: [research/TRAINING_REPORTS.md](research/TRAINING_REPORTS.md)

---

## 📊 脚本工具

项目包含以下脚本：

### 主要脚本

| 脚本 | 说明 |
|------|------|
| `scripts/dashboard.py` | Streamlit 交互式仪表板，用于 PINN 模型分析和可视化 |
| `scripts/visualizer_3d/visualizer_3d.py` | 3D 可视化模块，包含体积渲染和界面重建功能 |

### 使用示例

```bash
# 启动交互式仪表板
python scripts/dashboard.py

# 运行 3D 可视化
python scripts/visualizer_3d/visualizer_3d.py --config config/v4.5-standard.json
```

有关脚本使用的详细信息，请参阅 **[scripts_guide.md](guides/scripts_guide.md)**。

---

## 📂 源码结构（关键部分）

```
src/
├── models/                    # 神经网络模型
│   ├── aperture_model.py      # Stage 1 开口率模型（已校准）
│   └── pinn_two_phase.py      # Stage 2 两相流 PINN（6D Triad 输入）
├── predictors/                # 预测器
│   └── pinn_aperture.py       # Stage 2 PINN 预测器 (支持 6D 输入)
├── physics/                   # 物理约束
├── training/                  # 训练系统
├── utils/                     # 工具函数
└── solvers/                   # CFD 求解器
```

---

**返回**: [项目主页](../README.md)
