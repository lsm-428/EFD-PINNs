# 两相流 PINN：数据核心逻辑说明（Triplet 数据）

**最后更新**: 2026-02-04

本文总结两相流模块（`src/models/pinn_two_phase.py`）的数据核心逻辑：数据输入语义、数据集构成、采样策略、φ 标签生成方式，以及容易导致训练“跑偏/离谱”的关键一致性问题。

适用范围：
- 训练入口：`train_two_phase.py` → `src/models/pinn_two_phase.py:main()`
- 数据生成入口：`DataGenerator.generate_all_data()`
- 数据格式：Triplet（`x, y, z, V_from, V_to, t_since`）

---

## 1. Triplet 数据语义

单条样本的输入格式：
- `x, y, z`：空间坐标（单位 m）
- `V_from`：电压跳变前的电压
- `V_to`：电压跳变后的电压（当前电压）
- `t_since`：跳变后经过的时间（单位 s）

Triplet 的语义重点在于：**同一 `V_to` 在不同 `V_from` 下的物理状态可能不同**（例如降压过程与稳态过程），因此 `V_from` 与 `t_since` 不可省略。

---

## 2. 数据集构成（generate_all_data）

数据由 `DataGenerator.generate_all_data()` 一次性生成并返回一个 dict，训练循环中会重复使用这批数据（不是每个 epoch 重新生成）。

主要字段：
- `interface_points` / `interface_targets`：核心监督数据（φ）
- `ic_points` / `ic_values`：初始条件（t=0）
- `bc_points` / `bc_values`：边界条件（壁面）
- `domain_points`：域内配点（用于物理残差约束）
- `contact_points` / `contact_theta`：接触角边界点（用于接触角边界约束）

代码位置：
- `DataGenerator.generate_all_data()`：`src/models/pinn_two_phase.py`

---

## 3. 界面数据（interface_points/interface_targets）的生成逻辑

界面数据是训练的“主监督”，总数由 `data.n_interface` 控制。

### 3.1 时间采样策略：early / mid / late 三段

使用 τ（电润湿响应时间常数）构造时间采样：
- early：`0.001 ~ 2τ`（早期密集）
- mid：`2τ ~ 4τ`（中期密集，通常也是最容易出问题的区间）
- late：`4τ ~ t_max`（稀疏，靠近稳态）

这部分直接决定：
- 训练更关注哪些时间区间
- 数据是否能覆盖 10–20ms 这类中期动态段

### 3.2 界面数据的两部分：稳态 60% + 方波响应 40%

#### A) 稳态数据（约 60%）

稳态样本满足：
- `V_from = V_to = V`
- `t_since` 取自 `time_samples`（early/mid/late 拼接）

生成流程：
1) 计算开口率 `eta = get_opening_rate(V, t_since)`（用于“采点分布”）
2) 根据 eta 进行空间采样：`(x,y,z) = _sample_point_by_eta(eta)`
3) 生成 φ 标签：`phi = target_phi_3d(x,y,z,t_since,V,V_prev=V)`
4) 组装 Triplet：`[x, y, z, V, V, t_since]`

#### B) 方波响应数据（约 40%）

方波响应覆盖多电压：
- `V_highs = linspace(1, 30, 30)`（每 1V 一档）
- 对每个 V_high 同时生成升压与降压（各 50%）
- 时间采样偏向早期：70% 采 early，30% 采 late

升压样本：
- `V_from=0, V_to=V_high`
- `t_since` 直接采样
- 采点 eta：`get_opening_rate(V_to, t_since)`
- φ 标签：`target_phi_3d(..., V_prev=V_from)`

降压样本：
- `V_from=V_high, V_to=0`
- `t_since` 直接采样
- 采点 eta：`eta_at_fall * exp(-t_since/tau_recovery)`，其中 `eta_at_fall = get_opening_rate(V_from, 0.020)`
- φ 标签：`target_phi_3d(..., V_prev=V_from, t_step=0)`

---

## 4. 空间采样策略（_sample_point_by_eta）

`_sample_point_by_eta(eta)` 用于让训练点更集中在界面附近，从而提高界面学习效率：

- η < 0.50 且 η > 0.01：
  - 约 40% 概率在开口边界 `r_open` 附近采样（“界面加密”）
- η ≥ 0.50：
  - 约 40% 概率在角落附近采样（“角落/边缘加密”）
- 其他情况：
  - 均匀采样

z 方向：
- 50% 概率在油墨层附近加密（`z ~ h_ink * 3`）
- 50% 均匀采样整个高度（`z ~ Lz`）

详细参数说明请参见[物理理论与器件规格指南](physics_and_device_guide.md#physics-parameters)。

---

## 5. φ 标签生成（target_phi_3d）的关键点

`target_phi_3d(x,y,z,t,V,V_prev,t_step)` 是整个数据体系中最关键的一环：它把 Triplet 语义映射到“几何上合理”的 φ 分布。

### 5.1 优先使用 Stage1 的 triad η（如果可用）

如果 Stage1 Aperture 模型可用（`HAS_APERTURE`）：
- 会尝试构建 `EnhancedApertureModel`
- 直接调用 `theta_eta_from_triad(V_prev, V, t_since)` 得到 `eta`

这一点非常关键：它意味着 φ 的标签几何优先对齐 Stage1 的校准曲线，而不是纯解析公式。

### 5.2 triad 不可用时回退逻辑

若 triad 失败：
- 若 `V < V_prev`（降压）：
  - `eta = eta_max * exp(-t_since / tau_recovery)`
  - 并使用中心开口模式生成 φ
- 否则（稳态/升压）：
  - `eta = get_opening_rate(V, t)`

### 5.3 φ 的几何模式：中心开口为主

当前实现核心是“中心开口模式”：
- 中心透明（φ≈0）
- 边缘油墨（φ≈1）
- 界面用 tanh 过渡带平滑（`interface_width`）
- 对 eta 做上限裁剪（`max_eta=0.85`）以避免体积守恒被破坏

---

## 6. 核心一致性风险：采样 η vs 标签 η 可能不一致

在界面数据生成中，存在一个容易被忽略但影响巨大的点：

- **采样分布**（决定点更靠近哪里）主要通过 `get_opening_rate()` 计算 eta
- **标签几何**（决定 φ 的开口半径/边缘堆高）在 `target_phi_3d()` 内部优先通过 `theta_eta_from_triad()` 计算 eta

如果两套 eta 计算在某些区域差异大，会导致：
- 点分布“像 A”，标签几何“像 B”
- 训练出现强烈冲突：同一类点在空间上被采到，但标签与采样假设不一致
- 典型现象：界面模糊、中心区域不透明、体积守恒/开口率趋势异常、loss 震荡明显

排查建议：
- 临时把两处 eta 统一来源（要么都用 triad，要么都用解析），先验证训练是否稳定，再逐步引入混合策略。

---

## 7. 与配置的对应关系

以 `config/device_calibrated_physics.json` 为例，关键字段影响数据生成与训练规模：

- `data.n_interface / n_initial / n_boundary / n_domain`：直接控制各类数据点数量
- `data.voltages`：稳态与域内配点覆盖的电压集合
- `data.times`：`generate_all_data()` 中时间采样密度（并影响域内配点按时间分桶）
- `training.batch_size`：训练时每次从各集合抽取的随机子集大小

注意：
- `generate_all_data()` 当前会把所有数据张量直接放到 `device` 上（显存压力与数据规模强相关）
- `device_calibrated_physics.json` 的数据规模适中，需要关注显存占用与速度

---

## 8. 如何验证“数据是否按预期工作”

建议从训练日志与少量 sanity check 入手：

- 训练日志中应能看到各类点数量：
  - “界面数据点 / 初始条件点 / 壁面边界条件点 / 域内配点 / 接触角边界条件点”
- 对关键 Triplet（如 `0→30V, t_since=20ms`）抽样若干点，检查 φ 的空间分布是否满足：
  - 中心更透明、边缘更油墨
  - z 方向底部更油墨，上部更透明

---

## 9. 版本对比说明（关于“覆盖回 GitHub 前的版本”）

你之前要求“重新覆盖回来”，因此当前工作区仅保留 GitHub 版本的实现。

如果你希望做逐行对比（GitHub 版本 vs 覆盖前版本）：
- 需要你提供覆盖前的 `src/models/pinn_two_phase.py` 备份（例如 IDE Local History、手动备份文件、或你当时保存的 patch）
- 仅凭当前仓库无法还原“覆盖前”实现细节

