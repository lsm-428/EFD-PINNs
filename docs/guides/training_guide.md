# 训练策略与高级优化指南

**最后更新**: 2026-03-06
**版本**: v4.5

## 训练架构概览

EFD-PINNs 采用两阶段耦合训练架构：

```
Stage 1: 接触角 + 开口率解析映射 (Calibrated Analytical)
    V(t) → θ(t) → η(t)
    - 作用: 提供物理基准和监督信号

Stage 2: 两相流 6D PINN (Physics-Informed Neural Network)
    (x, y, z, V_from, V_to, t_since) → φ, u, v, w, p
    - 作用: 捕捉完整的三维多相流动力学过程
    - 关键: N-S 方程含电润湿驱动力
```

## 核心原则：单一真理源 (SSOT)

详细参数说明请参见[物理理论与器件规格指南](physics_and_device_guide.md#physics-parameters) 和 [配置系统指南](configuration_guide.md)。

## 三阶段渐进式训练详解

### 训练时间线（当前配置）

```
Epoch 0 -------- 1500 -------- 5500 -------- 60000
        | Stage 1 |  Stage 2  |    Stage 3     |
        |纯数据学习| 连续性+VOF | 完整物理+NS全加载 |
```

### 阶段 1 (0-1500 epochs): 纯数据学习 (Data-Driven Phase)

**目标：** 让模型先学会 φ 场的基本分布

| 损失项 | 权重 | 作用 |
|--------|------|------|
| interface | 500.0 | 界面数据拟合 |
| ic | 100.0 | 初始条件 |
| bc | 50.0 | 壁面边界条件 |

**物理方程权重：** 全部为 0

- **目的**: 让网络先学习φ场的基本分布
  - 知道z<h_interface是油墨，z>h_interface是极性液
  - 知道电压增加，开口率增大
  - 建立粗略的流场直觉

### 阶段 2 (1500-5500 epochs): 引入物理一致性 (Physics-Consistency Phase)

| 损失项 | 权重变化 | 物理意义 |
|--------|----------|----------|
| continuity | 0 → 0.5 | ∇·u = 0 |
| vof | 0 → 0.5 | ∂φ/∂t + u·∇φ = 0 |
| ns (预热) | 0 → 0.01 | Navier-Stokes 预热 |

- **目的**: 逐步引入物理约束，但NS权重很小
  - 连续性: ∇·u = 0（质量守恒）
  - VOF输运: ∂φ/∂t + u·∇φ = 0（界面演化）
  - NS预热: 让网络适应N-S约束形式

### 阶段 3 (5500-60000 epochs): 完整物理约束 (Full Physics Phase)

**N-S 方程（含电润湿力）：**
```
ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew

F_st = σκ∇φ           (表面张力)
F_ew = -σ_ew∇φ        (电润湿力)
σ_ew = ε₀εᵣ(V-V_T)²/(2d)
```

| 损失项 | 最终权重 |
|--------|----------|
| continuity | 0.5 |
| vof | 0.5 |
| ns | 0.1 |
| surface_tension | 0.01 |
| sharpening | 0.1 |

- **目的**: 全物理约束，精细调整
  - 完整N-S（含电润湿力、表面张力）
  - 所有物理约束协同工作

## 权重调度逻辑

### tanh平滑过渡

为什么要用tanh而不是线性过渡？

```python
def weight_schedule(epoch, transition_start, transition_width):
    # tanh: 平滑的S形曲线
    return 0.5 * (1 + np.tanh((epoch - transition_start) / transition_width))

# 线性过渡 vs tanh过渡
# 线性：0 → 0.1 → 0.2 → ... → 1.0 (突变多)
# tanh：0 → 0.02 → 0.1 → 0.5 → 0.9 → 1.0 (平滑)

# 物理直觉：
# - 网络需要时间适应新的约束
# - 突变导致损失震荡
# - 平滑过渡让网络逐步调整
```

实际效果：
```
Epoch 4900: NS weight = 0.00 (NS loss = 123.45)
Epoch 5000: NS weight = 0.00 (NS loss = 120.11)  ← 开始Stage 2
Epoch 5100: NS weight = 0.00 (NS loss = 115.23)
Epoch 5500: NS weight = 0.00 (NS loss = 98.76)
Epoch 6000: NS weight = 0.01 (NS loss = 45.32)  ← 缓慢增加
Epoch 7000: NS weight = 0.05 (NS loss = 23.11)
Epoch 12000: NS weight = 0.10 (NS loss = 12.45)  ← Stage 3，完整权重
```

### 动态权重调度

使用 `DynamicPhysicsWeightScheduler` 来避免梯度病理问题：
- **策略**: `stage` (固定阶段权重) / `adaptive` (基于梯度范数)
- **调用**: `scheduler.update(data_loss, physics_loss)` 每 `update_interval` 步

```python
class DynamicPhysicsWeightScheduler:
    def __call__(self, epoch, base_weight, loss_history):
        # 第1层：配置文件的基础权重
        w_base = base_weight  # 来自config文件

        # 第2层：Epoch依赖的缩放
        w_epoch = w_base * self.tanh_scale(epoch)  # 随epoch变化

        # 第3层：Loss依赖的适应
        if loss_history[-1] > loss_history[-10] * 1.5:
            # 损失突然增大，降低权重
            w_adapted = w_epoch * 0.5
        else:
            w_adapted = w_epoch

        return w_adapted
```

**效果**:
- 防止某个损失项主导训练
- 自动平衡不同物理约束
- 40%训练稳定性提升

## 训练稳定性特性

`TrainingStabilizer` 提供额外的鲁棒性：
- **学习率预热**: 前100轮从0线性增加到基础学习率
- **NaN/Inf恢复**: 自动恢复到最后有效状态并减半学习率
- **梯度裁剪**: 使用可配置阈值防止梯度爆炸
- **混合精度**: 启用 AMP 加速训练

### 梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 学习率调度 (LR Scheduler)
采用 `ReduceLROnPlateau`，并设置最小学习率以防陷入死循环：
- **Factor**: 0.5
- **Patience**: 1000 epochs
- **Min LR**: 1e-6

## 关键改进

### 1. 高密度电压采样 (0.5V 步进)
为了解决早期版本在中间电压段（如 15V）预测不准确的问题，引入了精细化的采样策略：
- **采样范围**: 0V - 30V
- **步进**: 0.5V (共 61 个采样点)
- **优势**: 确保 PINN 能够平滑地捕捉从低电压（无开口）到高电压（完全收缩）的连续相变过程。

### 2. Triad 输入格式 (6-Parameter Input)
全面采用三元组输入格式 `(x, y, z, V_from, V_to, t_since)`：
- **语义对齐**: 显式区分稳态 (`V_from = V_to`) 和动态跳变 (`V_from != V_to`)。
- **时间归一化**: `t_since` 统一相对于电压跳变时刻，极大降低了网络学习时间相关性的难度。

### 3. 界面加密采样逻辑改进
在 `_sample_point_by_eta` 方法中，根据开口率 $\eta$ 动态调整采样重心：
- **低开口率**: 集中在像素中心。
- **高开口率**: 向四角和边缘偏移。
- **优势**: 显著提升了油墨边缘（三相接触线附近）的梯度捕捉精度。

## 端到端训练方法

无需Stage 1依赖的替代训练策略：
- **监督模式**: 使用Stage 1预测作为强监督目标
- **无监督模式**: 纯物理约束配合渐进式课程
- **渐进式课程**:
  - Stage 1 (0-15%): 仅初始/边界条件
  - Stage 2 (15-35%): 添加接触角边界条件
  - Stage 3 (35-60%): 添加连续性 + VOF约束
  - Stage 4 (60-100%): 添加电润湿力约束

## 验证与监控

### 损失归一化

```python
# 归一化残差
div_u_norm = div_u * self.L_char / self.U_char
```

### 自动化验证
- **自动化验证**: 训练完成后自动调用 `evaluate.py` 生成评估报告。

## 训练监控

### 关键指标

1. **总损失**: 应稳定下降至 1e-4 以下。
2. **物理残差**: 各方程残差应减小，且 VOF 残差通常是主导项。
3. **开口率**: V=0V 时应为 0，V=30V 时应约 84% (已校准)。

### 验证脚本

```bash
# 使用评估脚本
python evaluate.py
```

## 核心训练命令

```bash
# 使用推荐配置训练 (60,000 epochs)
python train_two_phase.py --config config/v4.5-standard.json

# 快速冒烟测试 (1000 epochs)
python train_two_phase.py --epochs 1000

# 使用 LBFGS 微调配置（备选）
python train_two_phase.py --config config/v4.5_lbfgs_tuned.json
```

## 故障排除

### 损失爆炸

- 降低学习率
- 增加梯度裁剪
- 检查数据归一化

### 开口率不正确

- 检查 φ 场定义
- 验证边界条件
- 增加低电压约束权重

### 收敛缓慢

- 增加批次大小
- 调整物理权重
- 使用预训练模型

---

**相关文档**:
- [快速开始指南](quickstart.md)
- [安装配置指南](installation.md)
- [故障排除指南](troubleshooting.md)