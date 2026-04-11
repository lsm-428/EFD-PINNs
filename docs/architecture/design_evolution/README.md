# EFD3D 设计文档概览

> 本目录包含 11 个设计文档，涵盖了 Stage 1/2/3 的核心设计、实现、验证全流程。

---

## 快速导航

| 文档 | 用途 | 状态 |
|------|------|------|
| [pinn_input_redesign.md](./pinn_input_redesign.md) | PINN 6D Triad 架构设计 | ✅ |
| [stage2_architecture.md](./stage2_architecture.md) | Stage 2 双 MLP 架构详解 | ✅ |
| [dynamic_response.md](./dynamic_response.md) | 动态电压响应设计 | ✅ |
| [sampling_strategy.md](./sampling_strategy.md) | 空间/时间/电压采样策略 | ✅ |
| [lstm_pinn_stage3_design.md](../../experimental/lstm_pinn/docs/lstm_pinn_stage3_design.md) | Stage 3 LSTM-PINN 架构 | ✅ |
| [lstm_voltage_encoding.md](../../experimental/lstm_pinn/docs/lstm_voltage_encoding.md) | LSTM 电压序列编码 | ✅ NEW |
| [loss_function_design.md](./loss_function_design.md) | 损失函数详解 | ✅ |
| [interface_sharpening_volume_conservation.md](./interface_sharpening_volume_conservation.md) | 界面锐化与体积守恒 | ✅ |
| [physics_refactor_checklist.md](./physics_refactor_checklist.md) | 物理残差统一重构 | ✅ |
| [stage1_improvements.md](./stage1_improvements.md) | Stage 1 模型改进 | ✅ |
| [validation_framework.md](./validation_framework.md) | 验证工具与流程 | ✅ |

---

## 内容结构说明

### 输入与编码
- **`pinn_input_redesign.md`**: 6D Triad 输入设计
- **`lstm_voltage_encoding.md`**: LSTM 电压序列编码

### Stage 2: 双 MLP PINN
- **`stage2_architecture.md`**: 双 MLP 架构详解
- **`sampling_strategy.md`**: 采样策略
- **`loss_function_design.md`**: 损失函数详解
- **`interface_sharpening_volume_conservation.md`**: 界面锐化 + 体积守恒

### Stage 3: LSTM 混合
- **`lstm_pinn_stage3_design.md`**: LSTM 编码电压序列

### 物理与验证
- **`dynamic_response.md`**: 升压/降压物理机制
- **`physics_refactor_checklist.md`**: 物理残差重构
- **`validation_framework.md`**: 验证框架

### 历史记录
- **`stage1_improvements.md`**: Stage 1 改进

---

## 核心设计决策

### 1. 三阶段架构

| Stage | 模型 | 输入 | 目的 |
|-------|------|------|------|
| **Stage 1** | EnhancedApertureModel | (V_from, V_to, t_since) | 接触角/开口率预测 |
| **Stage 2** | TwoPhasePINN | (x, y, z, V_from, V_to, t_since) | 两相流场预测 |
| **Stage 3** | LSTM-PINN | (x, y, z, t, voltage_seq) | 电压序列建模 |

### 2. 电压序列编码
- 三元组格式: (V_from, V_to, t_since)
- 支持任意电压跳变序列
- LSTM 学习历史累积效应

### 3. 损失函数组合
- 数据损失: interface_loss (500.0) ⭐
- 物理损失: continuity (0.1) + vof (0.1) + ns (0.01)
- 约束损失: volume (10.0) + sharpening (0.1)

---

## 文档更新历史

| 日期 | 文档 | 说明 |
|------|------|------|
| 2025-12-22 | pinn_input_redesign.md | 6D Triad 架构设计 |
| 2026-02-04 | stage2_architecture.md | Stage 2 双 MLP 架构 |
| 2026-02-04 | dynamic_response.md | 动态电压响应 |
| 2026-02-04 | sampling_strategy.md | 采样策略 |
| 2026-02-04 | lstm_pinn_stage3_design.md | Stage 3 LSTM 架构 |
| 2026-02-04 | lstm_voltage_encoding.md | LSTM 电压序列编码 |
| 2026-02-04 | loss_function_design.md | 损失函数详解 |
| 2026-02-04 | interface_sharpening_volume_conservation.md | 界面锐化 + 体积守恒 |
| 2025-12-31 | physics_refactor_checklist.md | 物理残差重构 |
| 2026-01-08 | stage1_improvements.md | Stage 1 改进 |
| 2025-12-22 | validation_framework.md | 验证框架 |

---

**提示**: 本目录文档为设计决策的完整记录，保留原始设计思路和演进过程，便于追溯和参考。
