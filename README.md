# EFD-PINNs: 电润湿显示动力学预测

**Physics-Informed Neural Networks for Electrowetting Display Dynamics**

[![Status](https://img.shields.io/badge/status-training-yellow)](CURRENT_STATUS.md)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🎯 项目简介

使用物理信息神经网络(PINNs)预测电润湿显示器件中油墨的动态行为，实现毫秒级快速仿真，替代传统CFD方法。

**核心优势**:
- ⚡ **快速**: 训练后毫秒级推理 (vs CFD的小时级)
- 🎯 **准确**: 嵌入物理约束，保证合理性
- 🔧 **灵活**: 可学习不同材料和几何参数

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate efd

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. 查看当前状态

```bash
# 查看最新进展
cat CURRENT_STATUS.md

# 查看训练进度
grep -E "Epoch.*train=" training_stage2_10k.log | tail -10
```

### 3. 开始训练

```bash
# 使用10000 epochs配置训练
python efd_pinns_train.py --config config_stage2_10k.json --mode train --epochs 10000

# 或使用优化配置
python efd_pinns_train.py --config config_stage2_optimized.json --mode train
```

---

## 📁 项目结构

```
EFD3D/
├── README.md                    # 本文件
├── CURRENT_STATUS.md            # 当前状态 (频繁更新)
├── PROJECT_CONTEXT.md           # 完整技术文档
├── PROJECT_ROADMAP.md           # 项目路线图
│
├── efd_pinns_train.py          # 主训练脚本
├── config_stage2_10k.json      # 当前训练配置 (10000 epochs)
├── config_stage2_optimized.json # 优化配置
│
├── ewp_pinn_*.py               # 模型组件
├── analyze_*.py                # 分析工具
│
├── docs/                       # 详细文档
└── outputs_*/                  # 训练输出
```

---

## 📊 当前进展

**最新训练** (2025-12-01):
- 🔄 阶段2 v2 长期训练进行中 (10000 epochs)
- 当前进度: ~1295/10000 (13%)
- 训练损失: ~1.17 (稳定)
- 动力学参数: tau=5ms, zeta=0.85

**目标指标**:
| 指标 | 目标 | 当前最佳 |
|------|------|----------|
| 响应时间 | 1-10 ms | 3.64 ms ✅ |
| 超调 | <10% | 38.9% ❌ |
| 稳定时间 | <20 ms | 4.24 ms ✅ |

详见: [CURRENT_STATUS.md](CURRENT_STATUS.md)

---

## 🔬 技术特点

### 真实器件参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 像素尺寸 | 184×184 μm | 真实器件 |
| 总厚度 | 20.855 μm | 7层结构 |
| 介电层 | SU-8, 0.4μm, ε_r=4.0 | 光刻胶 |
| 疏水层 | Teflon AF, 0.4μm | 超疏水 |
| 工作电压 | 0-30V | 电润湿驱动 |

### 动力学参数 (v2优化)

| 参数 | 值 | 说明 |
|------|-----|------|
| tau | 5 ms | 时间常数 |
| zeta | 0.85 | 阻尼比 (接近临界阻尼) |

### 模型架构

- **输入**: 62维物理特征 (时空坐标+电学+几何+材料)
- **输出**: 24维物理量 (接触角+速度场+压力+界面)
- **网络**: [256, 256, 128, 64] + BatchNorm + Residual
- **激活**: GELU

### 物理约束

- Young-Lippmann方程 (静态平衡)
- 接触线动力学 (界面演化)
- 界面稳定性约束
- 体积守恒 (质量守恒)

---

## 📖 文档导航

### 核心文档
- **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - 当前状态和最新进展
- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** - 完整技术背景
- **[PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)** - 项目路线图

### 配置文件
- `config_stage2_10k.json` - 当前训练配置 (10000 epochs)
- `config_stage2_optimized.json` - 优化配置

### 工具脚本
- `analyze_dynamic_response.py` - 动态响应分析
- `analyze_young_lippmann.py` - 静态分析
- `verify_parameters.py` - 参数验证

---

## 🛠️ 常用命令

### 训练相关
```bash
# 查看当前训练进度
grep -E "Epoch.*train=" training_stage2_10k.log | tail -10

# 开始新训练
python efd_pinns_train.py --config config_stage2_10k.json --mode train --epochs 10000
```

### 分析相关
```bash
# 分析动态响应 (训练完成后)
python analyze_dynamic_response.py --model outputs_*/final_model.pth --output outputs_*/

# 验证参数
python verify_parameters.py
```

### 监控相关
```bash
# 查看训练日志
tail -f training_stage2_10k.log

# 检查GPU使用
nvidia-smi
```

---

## 📈 训练历史

| 训练 | 配置 | Epochs | 响应时间 | 超调 | 状态 |
|------|------|--------|----------|------|------|
| #1 | stage2_optimized | 41 | 3.64ms | 38.9% | ✅ 参考 |
| #11 | stage2_optimized | 200 | 0.20ms | 38.8% | ⚠️ 太快 |
| #12 | stage2_10k | 10000 | - | - | 🔄 进行中 |

---

## 🔧 故障排除

### GPU内存不足
```bash
python efd_pinns_train.py --device cpu --batch_size 16
```

### 训练不收敛
```bash
python efd_pinns_train.py --lr 1e-4
```

---

## 📚 参考资料

- Raissi et al. (2019) "Physics-informed neural networks"
- Mugele & Baret (2005) "Electrowetting: from basics to applications"

---

**快速链接**:
[当前状态](CURRENT_STATUS.md) | [完整文档](PROJECT_CONTEXT.md) | [路线图](PROJECT_ROADMAP.md)

**更新**: 2025-12-01
