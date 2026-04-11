# EFD3D 采样策略设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**对应代码**: `src/models/pinn_two_phase.py::DataGenerator`

---

## 1. 设计概述

采样策略是 PINN 训练的关键组成部分，直接影响模型收敛速度和最终精度。本文档描述 EFD3D 项目中的采样策略设计，包括：

- **空间采样**：界面加密、油墨层加密
- **时间采样**：Beta 分布，聚焦早期动态
- **电压场景采样**：稳态、升压、降压
- **边界/初始条件采样**

---

## 2. 空间采样策略

### 2.1 按开口率采样 (`_sample_point_by_eta`)

核心方法：根据当前开口率 `η` 在界面附近采样空间点。

```python
def _sample_point_by_eta(self, eta: float) -> tuple:
    """
    根据开口率采样空间点
    
    策略：
    - eta < 0.5 (中心开口): 在界面附近采样，中心透明
    - eta >= 0.5 (角落开口): 在四角附近采样
    - 40% 概率选择"界面模式"，60% 概率均匀采样
    """
```

**采样逻辑**：

| 开口率 η | 概率 | 采样区域 |
|---------|------|----------|
| η < 0.5 且随机 < 0.4 | 40% | 界面附近 (r_open + Gaussian noise) |
| η >= 0.5 且随机 < 0.4 | 40% | 四角附近 (角落 + Gaussian noise) |
| 其他 | 60% | 均匀随机 |

### 2.2 Z 方向加密采样

```python
# z 方向：在油墨层附近加密
if np.random.rand() < 0.5:
    z = np.random.rand() * self.h_ink * 3  # 油墨层加密
else:
    z = np.random.rand() * self.Lz          # 均匀采样
```

**原理**：油墨层 (z < 3μm) 是界面形成的关键区域，需要更多采样点。

### 2.3 界面采样分布

```
采样点密度分布:

z
↑
Lz ─────────────────────────────  极性液体区 (φ ≈ 0)
    ·
  ·     ·
h_ink ────────────────────────   油墨-极性液体界面 (φ ∈ [0,1])
    ·     ·
  ·     ·
 0 ─────────────────────────────  油墨基底 (φ ≈ 1)

        → r (径向)
```

---

## 3. 时间采样策略

### 3.1 Beta 分布采样

```python
def sample_continuous_times(n_samples):
    """
    连续时间采样 (Beta 分布)
    
    使用 Beta(0.5, 1.0) 分布，自然地在早期 (t=0) 附近采样更多点，
    同时平滑覆盖整个时间域。
    
    概率密度函数：p(t) ∝ t^(-0.5)
    """
    t_samples = np.random.beta(0.5, 1.0, n_samples) * t_max
    return t_samples
```

**分布特性**：

```
Beta(0.5, 1.0) 概率密度:

概率密度
    ↑
    │    ╱
    │   ╱
    │  ╱
    │ ╱
    │╱
    └──────────────────────→ t
    0          t_max/2     t_max

特点：
- t=0 处概率密度最高
- 向右指数递减
- 适合模拟暂态过程（变化集中在早期）
```

### 3.2 为什么不用均匀分布？

| 分布 | 问题 |
|------|------|
| 均匀分布 | 忽略早期动态的重要性，训练后期界面位置变化小 |
| 分段均匀 | 造成密度突变，人为引入边界效应 |

**Beta 分布优势**：
- 早期更多采样点 → 捕捉暂态响应
- 平滑覆盖全时间域 → 无人工边界
- 与物理直觉一致（动态响应集中在电压变化后）

---

## 4. 电压场景采样

### 4.1 三类场景

训练数据按电压变化场景分为三类：

| 场景 | 比例 | 三元组格式 | 物理含义 |
|------|------|------------|----------|
| **稳态** | 40% | `(V, V, t)` | 电压恒定，系统已达稳态 |
| **升压** | 30% | `(0, V, t)` | 电压从 0 升至 V |
| **降压** | 30% | `(V, 0, t)` | 电压从 V 降至 0 |

### 4.2 场景分布代码

```python
# 1.1 稳态数据 (40%) - V_from = V_to
n_steady = int(n_interface * 0.4)
V_steady = np.random.uniform(0, 30.0, n_steady)

# 1.2 升压响应 (30%) - 0 -> V
n_up = int(n_interface * 0.3)
V_up = np.random.uniform(1.0, 30.0, n_up)

# 1.3 降压响应 (30%) - V -> 0
n_down = n_interface - n_steady - n_up
V_down = np.random.uniform(1.0, 30.0, n_down)
```

### 4.3 场景采样逻辑

```
电压场景采样分布:

V (Voltage)
↑
30 ─────────────────────────────────
    │     ╭─── 稳态 (40%)
    │    ╱
 20 │   ╱ 升压 (30%)
    │  ╱
 10 │ ╱
    │╱ 降压 (30%)
  0 ─────────────────────────────────
    └──────────────────────────────→ t
    0        15ms        30ms       50ms
```

---

## 5. 初始条件采样

### 5.1 初始状态定义

t=0 时，油墨均匀铺在底部 (z < 3μm)：

```python
# 初始条件：油墨在底部，极性液体在上方
phi = 0.5 * (1 - np.tanh((z - self.h_ink) / interface_width))
```

**物理图像**：

```
t=0 初始状态:

z
↑
Lz ────────────────────  φ = 0 (极性液体)
    ════════════════    界面 (φ ∈ [0,1])
 0 ────────────────────  φ = 1 (油墨层, h_ink=3μm)

     x → y →
```

### 5.2 初始条件采样代码

```python
# 初始条件：t=0 时油墨均匀铺在底部 3μm
n_ic = data_cfg.get("n_initial", 10000)

for V in V_ic:
    # 空间均匀采样
    x = np.random.rand() * self.Lx
    y = np.random.rand() * self.Ly
    z = np.random.rand() * self.Lz
    
    # tanh 界面
    phi = 0.5 * (1 - np.tanh((z - self.h_ink) / interface_width))
    
    # 三元组格式: (x, y, z, V_from, V_to, t_since=0)
    ic_points.append([x, y, z, V, V, 0.0])
```

---

## 6. 边界条件采样

### 6.1 边界类型

| 边界 | 位置 | 约束类型 |
|------|------|----------|
| x=0 | 左壁面 | 无滑移/滑移 |
| x=Lx | 右壁面 | 无滑移/滑移 |
| y=0 | 前壁面 | 无滑移/滑移 |
| y=Ly | 后壁面 | 无滑移/滑移 |

### 6.2 边界采样代码

```python
# 混合场景采样：40% 稳态, 30% 升压, 30% 降压
n_bc = data_cfg.get("n_boundary", 10000)

for V_from, V_to, t in scenarios:
    # 随机选择一个壁面
    boundary_type = np.random.randint(0, 4)  # 0:x0, 1:xL, 2:y0, 3:yL
    
    if boundary_type == 0:   # x=0
        x, y = 0, np.random.rand() * self.Ly
    elif boundary_type == 1: # x=Lx
        x, y = self.Lx, np.random.rand() * self.Ly
    elif boundary_type == 2: # y=0
        x, y = np.random.rand() * self.Lx, 0
    else:                    # y=Ly
        x, y = np.random.rand() * self.Lx, self.Ly
    
    z = np.random.rand() * self.Lz
```

---

## 7. 采样配置参数

### 7.1 默认配置

```python
DEFAULT_DATA_CONFIG = {
    "n_interface": 100000,   # 界面数据点数量
    "n_initial": 10000,      # 初始条件点数量
    "n_boundary": 10000,     # 边界条件点数量
}
```

### 7.2 配置来源优先级

```
1. 用户配置文件 (config/*.json) → 最高优先级
2. DEFAULT_DATA_CONFIG  → 兜底默认值
```

### 7.3 采样点统计

| 数据类型 | 默认数量 | 占比 |
|----------|----------|------|
| 界面数据 | 100,000 | 83% |
| 初始条件 | 10,000 | 8% |
| 边界条件 | 10,000 | 8% |
| **总计** | **120,000** | **100%** |

---

## 8. 端到端训练的采样策略

### 8.1 EndToEndDataGenerator

端到端训练使用不同的数据生成器，位于 `experimental/end_to_end/train_end_to_end.py`：

```python
class EndToEndDataGenerator:
    """
    端到端训练数据生成器
    
    与标准 DataGenerator 的区别：
    - 不依赖 Stage 1 解析模型
    - 直接使用 TwoPhasePINN 的预测作为软约束
    - 采样策略更简单
    """
```

### 8.2 SequenceDataGenerator (LSTM-PINN)

LSTM-PINN 使用 `SequenceDataGenerator`：

```python
class SequenceDataGenerator:
    """
    LSTM-PINN 序列数据生成器
    
    特点：
    - 生成电压序列输入 (seq_len, 3) 三元组
    - 每步包含 (V_from, V_to, t_since)
    - 支持任意长度的电压跳变序列
    """
```

---

## 9. 采样策略演进

### 9.1 历史版本

| 版本 | 变更 | 原因 |
|------|------|------|
| v4.3 | 均匀时间采样 | 初始实现 |
| v4.4 | 引入 Beta(0.5, 1.0) 时间采样 | 早期动态采样不足 |
| v4.5 | 界面加密 + 油墨层加密 | 界面精度提升 |

### 9.2 未来改进方向

| 改进方向 | 预期收益 |
|----------|----------|
| 自适应采样 | 根据损失梯度动态调整采样密度 |
| 重要性采样 | 高损失区域增加采样 |
| 多尺度采样 | 同时捕捉宏观和微观特征 |

---

## 10. 验证与测试

### 10.1 相关测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_two_phase_data_generator.py` | DataGenerator 功能测试 |
| `tests/test_continuity.py` | 连续性方程验证 |
| `tests/test_vof_sensitivity.py` | 界面采样重要性测试 |

### 10.2 验证命令

```bash
# 运行 DataGenerator 测试
python -m pytest tests/test_two_phase_data_generator.py -v

# 验证采样分布
python -c "
from src.models.pinn_two_phase import DataGenerator, DEFAULT_CONFIG
import torch

config = DEFAULT_CONFIG
data_gen = DataGenerator(config, torch.device('cpu'))

# 生成数据并统计
data = data_gen.generate_all_data()
print(f'界面点: {data[\"interface_points\"].shape}')
print(f'IC 点: {data[\"ic_points\"].shape}')
print(f'BC 点: {data[\"bc_points\"].shape}')
"
```

---

## 11. 相关文档

| 文档 | 说明 |
|------|------|
| [pinn_input_redesign.md](./pinn_input_redesign.md) | 6D Triad 输入格式 |
| [dynamic_response.md](./dynamic_response.md) | 动态电压响应 |
| [loss_function_design.md](./loss_function_design.md) | 损失函数设计 |
| [lstm_pinn_stage3_design.md](../../experimental/lstm_pinn/docs/lstm_pinn_stage3_design.md) | LSTM-PINN 架构 |

---

**最后更新**: 2026-02-04
