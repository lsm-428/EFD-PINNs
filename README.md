# EFD3D: 基于物理信息神经网络 (PINN) 的电润湿流体动力学 3D 仿真框架

![Version](https://img.shields.io/badge/version-v4.5-blue.svg)
![Python](https://img.shields.io/badge/python-3.12--3.13-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.7.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Latest Config](https://img.shields.io/badge/latest_config-v4.5--standard-success.svg)

**EFD3D (Electrowetting Fluid Dynamics 3D)** 是一个工业级的 **物理信息神经网络 (Physics-Informed Neural Network, PINN)** 仿真框架，专为解决微流控和电子纸显示技术 (Electronic Paper Display, EPD) 中极端复杂的 **三维两相流 (3D Two-Phase Flow)** 问题而设计。

本框架采用 **VOF (Volume of Fluid)** 方法追踪油墨-极性液体界面，结合完整的 **Navier-Stokes** 方程和电润湿物理力场建模，实现了高精度的两相流仿真。通过创新的 **6D Triad 输入表示**，单一模型可连续模拟任意电压序列驱动下的流体响应，实现"一次训练，任意推理"。

---

## 🎯 核心特性

- **6D Triad 输入**: `(x, y, z, V_from, V_to, t_since)` - 支持任意电压跳变序列
- **两阶段架构**:
  - Stage 1: 解析模型预测接触角/开口率（已校准）
  - Stage 2: PINN求解完整流场（Navier-Stokes + VOF）
- **无网格仿真**: 连续空间采样，避免网格生成
- **工业级精度**: 体积守恒误差 <1%
- **混合CFD-PINN求解器**: 传统CFD求解器用于验证，PINN用于加速
- **端到端训练**: 无需Stage 1依赖的替代训练方法
- **动态损失权重调度**: 自适应平衡各项物理约束
- **训练稳定性增强**: NaN恢复机制和梯度裁剪

---

## 📊 版本演进

| 版本 | 发布日期 | 核心改进 |
|------|---------|---------|
| **v4.5** | 2026-01-29 | 界面锐化损失，30V 开口率 83.4%，体积误差 <1% |
| v4.4 | 2026-01-13 | Stage 1 Tutor 约束 |
| v4.3 | 2026-01-08 | 基础 PINN 架构 |

**最新配置**: `config/v4.5-standard.json`

### 当前里程碑

| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83.4%** | 像素开口率优化 |
| 体积误差 | **<1%** | VOF 输运方程体积守恒 |
| 代码规模 | **~41 文件** | 21 src + 5 scripts + 15 tests |
| 项目规模 | **13,937+** | Python代码文件总数 |

### 器件规格 {#device-specifications}

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

> 📖 **完整技术参数**: 材料参数、动态参数、训练配置等详见 **[物理理论与器件规格指南](docs/guides/physics_and_device_guide.md)** 和 **[配置系统指南](docs/guides/configuration_guide.md)**


---

## 🎯 快速入门

### 新用户
- **[快速开始指南](docs/guides/quickstart.md)** - 专为新用户设计的完整学习路径

### 开发者  
- **[开发者指南](CLAUDE.md)** - 配置、训练、API 使用和调试的完整指南

### 研究人员
- **[研究人员指南](docs/research/DEEP_UNDERSTANDING_GUIDE.md)** - 物理理论、深度技术理解和实验验证

### 🚀 快速命令

#### 训练
```bash
# 使用推荐配置训练
uv run train_two_phase.py --config config/v4.5-standard.json

# 从检查点恢复训练
uv run train_two_phase.py --config config/v4.5-standard.json --resume_from outputs/train/pinn_YYYYMMDD_HHMMSS/best_model.pth
```

#### 评估与可视化
```bash
# 运行评估并生成可视化
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/

# 启动交互式仪表板
uv run scripts/dashboard.py

# 运行消融研究
uv run scripts/run_ablation.sh
```

#### 测试
```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行特定测试模块
uv run pytest tests/test_pinn_complete.py -v

# 生成覆盖率报告
uv run pytest tests/ --cov=src --cov-report=html
```

---

## 📚 文档导航

### 入门指南
- **快速开始**: [`docs/guides/quickstart.md`](docs/guides/quickstart.md) ⭐
- **安装配置**: [`docs/guides/installation.md`](docs/guides/installation.md)
- **使用指南**: [`docs/guides/usage.md`](docs/guides/usage.md)
- **训练策略**: [`docs/guides/training_guide.md`](docs/guides/training_guide.md)
- **可视化指南**: [`docs/guides/visualization_guide.md`](docs/guides/visualization_guide.md)

### 技术文档
- **物理理论与器件规格**: [`docs/guides/physics_and_device_guide.md`](docs/guides/physics_and_device_guide.md) ⭐
- **配置系统**: [`docs/guides/configuration_guide.md`](docs/guides/configuration_guide.md) ⭐
- **系统架构**: [`docs/architecture/system_design.md`](docs/architecture/system_design.md)
- **API参考**: [`docs/api/README.md`](docs/api/README.md)
- **脚本工具使用**: [`docs/guides/scripts_guide.md`](docs/guides/scripts_guide.md)

### 开发者资源
- **开发者指南**: [`CLAUDE.md`](CLAUDE.md) ⭐
- **研究人员深度指南**: [`docs/research/DEEP_UNDERSTANDING_GUIDE.md`](docs/research/DEEP_UNDERSTANDING_GUIDE.md)
- **系统深入解析**: [`docs/research/DEEP_DIVE_GUIDE.md`](docs/research/DEEP_DIVE_GUIDE.md)
- **贡献指南**: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)
- **故障排除**: [`docs/guides/troubleshooting.md`](docs/guides/troubleshooting.md)
- **更新日志**: [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
- **文档中心**: [`docs/README.md`](docs/README.md) ⭐
- **训练报告**: [`docs/research/TRAINING_REPORTS.md`](docs/research/TRAINING_REPORTS.md)

### 架构设计
- **设计演进**: [`docs/architecture/design_evolution/README.md`](docs/architecture/design_evolution/README.md)
- **模型架构**: [`docs/architecture/model_architecture.md`](docs/architecture/model_architecture.md)

---

## 📊 交互式仪表板

EFD3D 提供了功能完整的 Streamlit 交互式仪表板，集成了所有可视化和分析功能：

### 仪表板功能
**`scripts/dashboard.py`** - 综合分析和可视化平台，包含以下8个功能模块：

1. **📊 2D场分析** - 截面场分析，支持任意电压序列模拟
2. **🧊 3D体渲染视图** - 三维体积渲染和界面重建
3. **📈 瞬态响应** - 时域响应分析和动态可视化
4. **🩺 物理诊断** - 物理约束验证和残差分析
5. **📊 训练输出分析** - 训练曲线、模型性能和收敛分析
6. **⏱️ 基准测试** - 性能指标和性能对比
7. **🔄 对比分析** - 多模型/多配置对比
8. **📐 Stage 1** - 解析模型验证和接触角分析

### 使用示例
```bash
# 启动交互式仪表板
uv run scripts/dashboard.py

# 运行消融研究
uv run scripts/run_ablation.sh
```

### 输出目录结构
训练和分析生成的输出文件保存在：

```
outputs/train/
└── pinn_YYYYMMDD_HHMMSS/          # 单次训练运行目录
    ├── best_model.pth             # 最佳模型权重
    ├── best_model_epoch_XXXXX.pth # 特定轮次检查点
    ├── training.log               # 训练进度日志
    ├── interface_3d_steady.png    # 3D界面可视化
    ├── training_curve.png         # 损失曲线
    └── config.json                # 训练配置快照
```

详细使用说明请参考：[脚本使用指南](docs/guides/scripts_guide.md)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

详见: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)

---

## 📖 引用本文

如果您在研究中使用了 EFD3D，请引用以下基础工作：

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2018.10.045}
}
```

EFD3D 论文发表后，我们将更新此引用信息。如需提前获取论文预印本，请关注本仓库或联系作者。

---

## 📄 许可证

MIT License - 详见 [`LICENSE`](LICENSE) 文件

---

## 🛠️ 开发环境

- **包管理器**: `uv` (见 `pyproject.toml`)
- **Python版本**: 3.12-3.13
- **GPU加速**: CUDA 11.8 (PyTorch 2.7.1)
- **代码质量**: `ruff` (linting), `black` (formatting, 行长度88)
- **测试框架**: `pytest` + `hypothesis`

### 依赖管理
详细依赖列表和版本要求请参考: [`docs/guides/dependencies.md`](docs/guides/dependencies.md)

---

*Copyright © 2026 SCNU EFD Team. All Rights Reserved.*
*Powered by PyTorch & Physics-Informed Neural Networks.*

---

*最后更新: 2026-04-12 | Version: v4.5*
