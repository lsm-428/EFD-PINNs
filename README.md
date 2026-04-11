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

### 器件规格 {#device-specifications}

- **像素尺寸**: 174μm × 174μm
- **围堰高度**: 20μm（模型）/ 3.5μm（实际）
- **油墨层**: 3μm
- **工作电压范围**: 0-30V
- **典型工作电压**: 20V

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

---

## 📚 文档导航

### 用户指南
- **快速开始**: [`docs/guides/quickstart.md`](docs/guides/quickstart.md)
- **安装配置**: [`docs/guides/installation.md`](docs/guides/installation.md)
- **训练策略**: [`docs/guides/training_guide.md`](docs/guides/training_guide.md)
- **故障排除**: [`docs/guides/troubleshooting.md`](docs/guides/troubleshooting.md)
- **脚本工具使用**: [`docs/guides/scripts_guide.md`](docs/guides/scripts_guide.md)

### 技术文档
- **物理理论与器件规格**: [`docs/guides/physics_and_device_guide.md`](docs/guides/physics_and_device_guide.md)
- **API参考**: [`docs/api/README.md`](docs/api/README.md)
- **配置系统**: [`docs/guides/configuration_guide.md`](docs/guides/configuration_guide.md)
- **文档中心**: [`docs/README.md`](docs/README.md) ⭐

### 开发者资源
- **开发者指南**: [`CLAUDE.md`](CLAUDE.md)
- **贡献指南**: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)
- **更新日志**: [`docs/CHANGELOG.md`](docs/CHANGELOG.md)

### 配置相关
- **配置与权重指南**: [`docs/guides/configuration_guide.md`](docs/guides/configuration_guide.md) ⭐

---

## 📊 脚本使用

当前项目包含以下主要脚本：

### 主要脚本
- **`scripts/dashboard.py`** - Streamlit交互式仪表板，用于PINN模型分析和可视化
- **`scripts/visualizer_3d/`** - 3D可视化模块，包含体积渲染和界面重建功能

### 使用示例
```bash
# 启动交互式仪表板
python scripts/dashboard.py

# 运行3D可视化（具体用法请参考visualizer_3d模块文档）
python scripts/visualizer_3d/visualizer_3d.py [options]
```

### 输出目录
脚本生成的输出文件通常保存在以下位置：
- `outputs/train/` - 训练输出，包含模型检查点、训练日志和可视化图像
- 单次训练运行会创建类似 `pinn_YYYYMMDD_HHMMSS/` 的子目录，包含：
  - 模型检查点 (`.pth` 文件)
  - 训练日志 (`training.log`)
  - 可视化图像 (例如 `interface_3d_steady.png`, `training_curve.png`)

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

*Copyright © 2026 SCNU EFD Team. All Rights Reserved.*
*Powered by PyTorch & Physics-Informed Neural Networks.*

---

*最后更新: 2026-02-04 | Version: v4.5*
