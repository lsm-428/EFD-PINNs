# LSTM-PINN 实验模块

**状态**: 🔬 实验性 / 需要重新设计
**移动时间**: 2026-03-13
**原因**: 发现根本性设计问题

---

## ⚠️ 当前问题

### 核心问题

**LSTM 的设计目标错误**：

```
错误设计:
voltage_seq → LSTM → (V_eff, V_prev_eff) → PINN phi_net → φ
                         ↓
                   试图输出"等效单步电压"
                   破坏了多步累积的物理意义
```

### 训练数据问题

**多步序列目标计算错误**：

```python
# 当前实现 (错误)
last_V_from, last_V_to, last_t_since = transitions[-1]
eta = get_opening_rate(last_V_to, last_t_since)  # 只看最后一步！

# 正确实现 (待设计)
# 应该逐步累积状态：
state = initial_state
for V_from, V_to, t_since in voltage_sequence:
    state = accumulate(state, V_from, V_to, t_since)
```

### 实测数据

| 场景 | 预期 φ | 实际 φ (LSTM) | 问题 |
|------|--------|---------------|------|
| 0→30V, 10ms | ~1.0 | 0.82 | LSTM V_prev_eff 输出错误 |
| 0→20V→30V | 介于两者之间 | 取决于最后一步 | 忽略累积效应 |

---

## 📁 目录结构

```
experimental/lstm_pinn/
├── README.md                    # 本文档
├── train_lstm_hybrid.py         # 训练脚本
├── config/
│   └── lstm_dynamic_response.json  # 配置文件
├── src/
│   ├── models/lstm_pinn/        # LSTM 模型源码
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_generator.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   ├── hybrid_model.py      # 主模型
│   │   ├── model.py
│   │   ├── physics_loss.py
│   │   ├── physics.py
│   │   ├── response_time.py
│   │   ├── trainer.py
│   │   └── visualization.py
│   ├── predictors/
│   │   └── lstm_hybrid_predictor.py  # 预测器
│   └── dashboard/               # Dashboard 模块（依赖 LSTM）
│       ├── __init__.py
│       ├── datastore.py
│       ├── inference.py
│       ├── model_manager.py
│       └── training_output_analyzer.py
├── scripts/
│   └── test_lstm_predictor.py   # 测试脚本
├── tests/
│   ├── test_lstm_pinn_properties.py
│   ├── test_training_output_analyzer.py
│   ├── test_config_loading.py
│   ├── test_dashboard_engine.py
│   ├── test_training_output_analyzer_integration.py
│   └── TEST_REPORT.md
├── docs/
│   ├── lstm_pinn_stage3_design.md    # 设计文档
│   └── lstm_voltage_encoding.md      # 电压编码文档
├── plans/
│   ├── lstm-progressive-unfreezing.md    # 渐进解冻计划
│   └── lstm-voltage-encoding-fix.md      # 电压编码修复计划
└── outputs/
    └── train/                   # 训练输出
        ├── lstm_hybrid_20260205_174333/
        └── lstm_hybrid_20260312_220955/
```

---

## 🔮 待解决问题

### 1. 多步累积物理模型

**问题**：如何正确计算多步序列的目标 φ？

**可能的方案**：

| 方案 | 描述 | 可行性 |
|------|------|--------|
| **PINN 逐步模拟** | 用 PINN 模拟每一步的状态 | 需要实现状态传递 |
| **解析公式** | 推导多步累积公式 | 需要物理建模 |
| **实验数据** | 使用实际测量数据 | 需要实验支持 |

### 2. LSTM 架构重新设计

**问题**：LSTM 应该输出什么？

| 方案 | 输出 | 优点 | 缺点 |
|------|------|------|------|
| **A: 直接 φ** | φ_final | 简单直接 | 需要正确训练数据 |
| **B: 状态增量** | Δφ | 物理意义明确 | 需要修改训练逻辑 |
| **C: 修正因子** | k × PINN_单步 | 复用 PINN | 需要定义修正因子语义 |

### 3. 数据生成重构

**需要的改进**：

1. **逐步状态传递**：每一步从上一状态开始
2. **路径记录**：记录完整的状态演变
3. **边界条件**：正确处理降压后的恢复

---

## 📝 设计讨论记录

### 2026-03-13 用户洞察

> "举个方波的例子，0-20v 5ms 相当于 20V 的电压等效持续了 5ms，开口率肯定不是真实的 20V 开口率比如 66%，油墨刚刚响应有一个小 20V 等效电压的开口率涨幅，此时可能只有 30%；20-30v 5ms 是在 30% 的基础上相当于 30V 等效电压持续了 5ms..."

**关键洞察**：
- 每一步都在上一步状态基础上继续响应
- 存在累积效应
- 当前训练数据忽略了这一点

---

## 🚀 下一步

1. **研究多步累积物理**：确定正确的目标计算方法
2. **重新设计 LSTM 输出**：明确 LSTM 应该学习什么
3. **重构训练数据生成**：实现逐步状态累积
4. **设计验证实验**：对比不同路径的响应

---

## 📚 参考

- 前一级 PINN: `src/models/pinn_two_phase.py`
- 设计演进: `docs/architecture/design_evolution/lstm_voltage_encoding.md`