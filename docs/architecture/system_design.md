# EFD-PINNs 系统设计文档

**最后更新**: 2026-02-04

---

## 1. 系统目标

构建一个基于物理信息神经网络 (PINN) 的三维仿真框架，用于精确预测和优化电润湿显示 (EWD) 器件的动态开口率、响应时间及功耗。

---

## 2. 核心架构 (Tiered Architecture)

### 2.1 整体数据流 (Coupled Pipeline)

```mermaid
graph TD
    V[电压信号 V] --> S1[Stage 1: 物理参数校准模型]
    S1 --> Theta[动态接触角 θ(t)]
    Theta --> S2[Stage 2: 两相流 6D PINN]
    S2 --> Phi[三维相场分布 φ(x,y,z,t)]
    Phi --> Eta[开口率 η(t)]
    
    subgraph "Physics Engine (含电润湿力)"
        S2 -- 物理损失 -- NS[Navier-Stokes + 电润湿力]
        S2 -- 界面损失 -- VOF[VOF 界面追踪]
        S2 -- 质量损失 -- Cont[连续性方程]
    end
```

### 2.2 模块层次

#### 层级 1: 模型层 (`src/models/`)
- **`pinn_two_phase.py`**: 核心 6D 输入 PINN 模型。
  - **输入**: `(x, y, z, V_from, V_to, t_since)`
  - **输出**: `(u, v, w, p, phi)`
  - **特性**: 包含电润湿驱动力 `F_ew`，支持三阶段渐进式训练。
- **`aperture_model.py`**: Stage 1 物理基准模型 (`EnhancedApertureModel`)。
  - **功能**: 提供经过校准的 Young-Lippmann 接触角目标。
  - **精度**: 稳态误差 < 0.7°，响应时间误差 < 3ms。
- **`pinn_aperture.py`**: Stage 2 开口率预测器。
  - **功能**: 从 φ 场积分计算宏观开口率 η。

#### 层级 2: 物理与数据层 (`src/physics/`)
- **`constraints.py`**: 物理方程约束集。
  - **Navier-Stokes**: 含混合密度/粘度与 CSF 表面张力模型。
  - **VOF**: 相场输运方程。
  - **Continuity**: 不可压缩流体连续性。
说明：
- 两相流 PINN 的训练数据生成器位于 `src/models/pinn_two_phase.py:DataGenerator`（Triplet/Triad 输入），并不在 `src/physics/` 下。

#### 层级 3: 训练与优化层 (`src/training/`)
- **`components.py`**: 动态权重调度与 Checkpoint 管理等通用组件。

#### 层级 4: 工具与自动化层 (`root` & `scripts/`)
- **`evaluate.py`**: 核心评估工具。生成动态响应曲线、3D 界面重构、响应时间统计。
- **`scripts/analysis/training/analyze_log.py`**: 训练日志解析与可视化（输出 loss_breakdown.csv 等）。
- **`scripts/analysis/diagnose_oscillation.py`**: 动态响应波动/突变诊断（针对 0→30→0V）。
- **`scripts/analysis/volume/check_conservation.py`**: 体积守恒与升压→降压边界连续性检查。

---

## 3. 关键技术细节

### 3.1 6D Triad 输入空间
从原始 `(x, y, z, t)` 演进为 `(x, y, z, V_from, V_to, t_since)`，以解决降压恢复态的时间歧义性。

- **V_from**: 跳变前电压
- **V_to**: 跳变后电压（当前电压）
- **t_since**: 跳变后经过的时间

### 3.2 电润湿驱动力 (Electrowetting Force)
在 N-S 方程中显式引入电润湿力，驱动油墨移动：
\[
F_{ew} = -\sigma_{ew} \nabla \phi, \quad \sigma_{ew} = \frac{\varepsilon_0 \varepsilon_r (V - V_T)^2}{2d}
\]

详细参数说明请参见[物理理论与器件规格指南](../guides/physics_and_device_guide.md#physics-parameters)。

物理意义：电场在介电层产生 Maxwell 应力，等效于降低界面表面张力。

### 3.3 渐进式训练策略
训练轮数与阶段划分以配置为准（不要在文档里硬编码数值）：
- `training.epochs`: 总轮数
- `training.stage1_epochs`: Stage 1 结束轮
- `training.stage2_epochs`: Stage 2 结束轮（之后进入 Stage 3）

数据生成与采样的权威说明：
- [data-generation-core.md](../guides/data-generation-core.md)

---

## 4. 模块依赖关系

```
train_two_phase.py
    └── src/models/pinn_two_phase.py
        ├── TwoPhasePINN (模型)
        ├── PhysicsLoss (物理损失)
        │   └── src/physics/constraints.py (方程定义)
        ├── DataGenerator (数据生成)
        │   └── src/models/aperture_model.py (物理基准)
        └── Trainer (训练器)

evaluate.py
    ├── TwoPhasePINN (加载模型)
    └── EnhancedApertureModel (对比基准)
```

---

## 5. 性能指标参考

| 指标 | 目标 | 当前状态 |
|------|------|----------|
| 稳态精度 | <3° | 0.7° ✅ |
| 响应时间 | <30ms | 13-22ms ✅ |
| V=0V 开口率 | 0% | 0% ✅ |
| 物理守恒误差 | <0.5% | <0.5% ✅ |
