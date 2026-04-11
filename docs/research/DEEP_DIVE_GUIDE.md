# EWD-PINNs 项目深度理解指南

**目标读者**: 需要深入理解项目物理原理、架构设计和实现细节的开发者
**更新日期**: 2026-02-04
**项目状态**: Stage 2 v4.5-sharpening 训练完成

---

## 目录

1. [物理原理：电润湿显示机制](#1-物理原理电润湿显示机制)
2. [两阶段架构设计](#2-两阶段架构设计)
3. [6D三元组输入的深刻含义](#36d三元组输入的深刻含义)
4. [VOF vs Level Set方法对比](#4-vof-vs-level-set方法对比)
5. [训练策略的物理直觉](#5-训练策略的物理直觉)
6. [代码实现的关键细节](#6-代码实现的关键细节)
7. [常见陷阱与调试](#7-常见陷阱与调试)

---

## 1. 物理原理：电润湿显示机制

### 1.1 核心物理现象

**电润湿效应**不是直接驱动油墨，而是通过极性液体间接推动油墨：

```
施加电压 → Maxwell应力在介电层 → 极性液被吸引向下 →
极性液铺展在疏水层 → 油墨被"被动"挤到边缘 → 中心开口变大
```

**关键误解纠正**：
- ❌ 错误理解：电压直接驱动油墨移动
- ✅ 正确理解：电压驱动极性液体，油墨是被体积守恒约束"挤"走的

### 1.2 器件结构的物理层次

```
┌─────────────────────────────────────┐
│  上电极（ITO）                       │  ← 接地
├─────────────────────────────────────┤
│  极性液体层（17μm）                  │  ← 导电，与上电极形成电路
│  - 被电场驱动                        │
│  - 下沉到疏水层表面                  │
├─────────────────────────────────────┤  ← 界面（高度随电压变化）
│  油墨层（3μm） ← 贴底！              │  ← 不导电，被极性液挤压
│  - 体积守恒                          │
│  - 位置由疏水层表面疏水性决定        │
├─────────────────────────────────────┤
│  疏水层（Teflon AF，400nm）          │  ← 提供疏水表面
├─────────────────────────────────────┤
│  介电层（SU-8，400nm）               │  ← 电容储能层
├─────────────────────────────────────┤
│  下电极（ITO）                       │  ← 施加电压
└─────────────────────────────────────┘
```

**物理常量的工程意义**：

详细参数说明请参见[物理理论与器件规格指南](../guides/physics_and_device_guide.md#physics-parameters)。

**为什么标定值与物理值不同？**
- 极性液会渗透进介电层微孔 → 有效εᵣ增大
- 油墨/水混合界面 → 有效表面张力降低
- 这些是"工程有效参数"，不是材料固有属性

### 1.3 Young-Lippmann方程的物理图像

**静态方程**（稳态接触角）：
```
cos(θ) = cos(θ₀) + (ε₀εᵣ/2γd) × (V-V_T)²
```

**物理图像**：
- 无电压时，油墨表面张力 γ 与疏水层表面能平衡 → θ₀ = 120°
- 施加电压，介电层电容储能 → 等效表面张力降低 → θ 减小
- 电场越强，等效张力越低 → θ 越小（极性液更润湿）

**动态响应**（二阶欠阻尼系统）：
```
θ(t) = θ_eq + (θ₀ - θ_eq) × e^(-ζω₀t) × [cos(ω_d·t) + (ζ/√(1-ζ²))sin(ω_d·t)]
```

**物理类比**：弹簧-质量-阻尼系统
- 表面张力 → 弹簧（恢复力）
- 流体惯性和粘性 → 质量和阻尼
- 电润湿力 → 外力驱动

**典型动态响应**（以 0V→20V 跳变为例）：

接触角和开口率随时间单调变化，呈现类似指数/抛物线的响应曲线：

```
时间 | 接触角 θ | 开口率 η | 说明
------|-----------|----------|----------
t=0ms | θ=120.0° | η=0% | 初始状态 (0V)
t=1ms | θ≈119.0° | η≈20% | 快速响应开始
t=2ms | θ≈117.5° | η≈40% |
t=5ms | θ≈116.0° | η≈58% | 响应时间 t₉₀
t=10ms | θ≈115.5° | η≈64% | 接近平衡
t=20ms | θ≈115.2° | η≈66.7% | 稳态 (20V平衡)
t=50ms | θ≈115.2° | η≈66.7% | 完全稳定

**关键特征**：
- 响应时间 t₉₀ ≈ 5ms（达到 90% 稳态值的时间）
- 响应曲线呈单调抛物线形（无振荡）
- 油墨开口率变化滞后于电压（粘滞阻尼效应）
- 超调 <5%（几乎无振荡）

### 1.4 体积守恒的几何约束

**为什么油墨高度会增加？**

```
初始状态（0V）:
  V_ink = A_total × h_ink = [Lx × Ly × h_ink](../guides/physics_and_device_guide.md#physics-parameters)

施加20V后:
  中心开口率 η = 66.7% ([详细参数见物理理论与器件规格指南](../guides/physics_and_device_guide.md#physics-parameters))
  油墨被挤到边缘，占据面积 A_ink = A_total × (1-0.67) = A_total × 0.33
  体积守恒: V_ink = A_ink × h_interface
  因此: h_interface = h_ink / 0.33 ≈ 3μm / 0.33 ≈ 9μm
```

**关键洞察**：
- 油墨不是"向上堆"，而是"横向收缩到边缘"
- 体积守恒 → 横截面积减小 → 高度必须增加
- 边缘油墨堆高，中心形成透明区域

**几何推导**：
```python
# 简化公式（假设均匀高度）
h_interface = h_initial / (1 - η)

# 精确公式（考虑环形几何，VOF实现）
r_open = sqrt(η × Lx × Ly / π)       # 中心开口半径
A_ink = Lx × Ly - π × r_open²         # 环形油墨面积
h_edge = V_ink / A_ink                # 边缘油墨高度
```

**两者区别**：
- 简化公式：假设油墨均匀分布在边缘
- 精确公式：考虑环形几何，更符合实际
- 在中等开口率（η=60%）时，两者结果接近

---

## 2. 两阶段架构设计

### 2.1 为什么要分两阶段？

**问题**：直接用PINN学习 电压 → 开口率 映射困难
- PINN需要学习复杂的三维流场演化
- 没有先验知识，收敛慢且不稳定
- 物理约束（N-S方程）容易爆炸

**解决方案**：分解问题，分而治之

```
Stage 1（解析解）：
  输入：电压波形
  输出：接触角 θ(t)、开口率 η(t)
  方法：Young-Lippmann方程 + 二阶欠阻尼动力学
  优势：精确、快速、可微、已校准
  局限：只提供z=0表面的边界条件

Stage 2（PINN 3D）：
  输入：(x,y,z, V_from, V_to, t_since)
  输出：三维流场 (u,v,w,p,φ)
  方法：物理信息神经网络
  优势：完整3D流场、可预测任意位置
  数据：使用Stage1结果作为边界条件和训练目标
```

### 2.2 Stage 1的物理本质

**本质**：2D边界条件模型

```python
# Stage 1提供的只是z=0平面的信息
def stage1_model(V_from, V_to, t_since):
    theta = get_contact_angle_dynamic(V_to, t_since)  # 接触角
    eta = contact_angle_to_aperture(theta)            # 开口率（只在z=0）
    return theta, eta
```

**局限性**：
- ✅ 知道：z=0平面，中心开口率η=67%
- ✅ 知道：界面平均高度h≈9μm（通过体积守恒推算）
- ❌ 不知道：z=5μm处是什么流体？
- ❌ 不知道：界面具体弯曲形状？
- ❌ 不知道：内部速度场分布？

### 2.3 Stage 2的完整3D求解

**本质**：从2D边界推断3D内部

```python
# Stage 2学习的是完整3D体积内的流场
def stage2_pinn(x, y, z, V_from, V_to, t_since):
    # 学习整个3D空间的物理场
    phi = predict_volume_fraction(x, y, z)   # 任意z位置的φ值
    u, v, w, p = predict_velocity_pressure()  # 完整速度压力场
    return phi, u, v, w, p
```

**训练数据的Z维度推断**：

Stage 1只提供z=0边界，如何生成3D训练数据？

```python
# 从2D边界推断3D内部
h_interface = h_initial / (1 - eta)  # 用体积守恒推断界面高度

# 生成3D训练数据
for (x, y, z) in sampling_points:
    if z < h_interface:
        psi_target = -0.5  # 油墨区域（推断）
    else:
        psi_target = +0.5  # 极性液区域（推断）
```

**关键洞察**：
- PINN不是简单地记住训练数据的Z分布
- PINN学习的是：**在体积守恒约束下，界面如何在3D空间中演化**
- 训练数据提供"边界条件+体积守恒"，PINN学习"内部流场规律"

### 2.4 两阶段耦合的数据流

```
┌─────────────────────────────────────────────────────┐
│  训练阶段（离线）                                    │
└─────────────────────────────────────────────────────┘

输入：电压波形 (V_from, V_to, t)
  ↓
Stage 1: EnhancedApertureModel
  - Young-Lippmann → θ_eq(V)
  - 二阶欠阻尼 → θ(t)
  - 几何映射 → η(t)
  ↓
生成训练数据：
  - 界面数据：z ≈ h_interface 处 φ=0.5
  - 边界条件：z=0处 θ, η
  - 体积守恒：∫φ dV = V_ink
  ↓
Stage 2: TwoPhasePINN
  - 输入：(x,y,z, V_from, V_to, t)
  - 输出：(u,v,w,p, φ)
  - 损失：数据拟合 + N-S + VOF + 连续性
  ↓
训练完成：保存模型权重

┌─────────────────────────────────────────────────────┐
│  推理阶段（在线）                                    │
└─────────────────────────────────────────────────────┘

输入：任意电压波形
  ↓
Stage 2 PINN推理：
  - 预测完整3D流场 φ(x,y,z,t)
  - 计算任意截面的流体分布
  ↓
后处理：
  - 在z=0平面统计开口率 η(t)
  - 可视化界面演化
  - 输出速度场
```

---

## 3. 6D三元组输入的深刻含义

### 3.1 从绝对时间到相对时间

**原始设计的问题**：
```python
# ❌ 使用绝对时间
input = (x, y, z, V, t)  # t从0开始单调递增

问题：
1. 无法处理任意电压跳变
   - t=10ms时，可能是0→10V，也可能是20→10V
   - PINN无法区分这两种情况
2. 降压过程建模困难
   - 20V→0V的恢复过程与0V→20V不同
   - 但绝对时间t无法表达这种差异
3. 无法支持复杂驱动波形
   - 实际显示驱动是连续的电压序列
   - 0V→20V→10V→30V→0V...
```

**6D三元组的设计**：
```python
# ✅ 使用电压三元组
input = (x, y, z, V_from, V_to, t_since)

物理意义：
- V_from：之前稳态的电压（历史）
- V_to：当前目标电压（现在）
- t_since：跳变后经过的时间（演化）

优势：
1. 明确表达电压转换
   - V_from=0, V_to=20: 升压过程
   - V_from=20, V_to=0: 降压过程
2. 时间重新归零
   - 每次跳变后t_since从0开始
   - PINN学习"跳变后的演化"而非"绝对时间"
3. 支持任意波形
   - 可以组合多个跳变段
   - 0→20 (t<20ms) + 20→10 (t<20ms) + ...
```

### 3.2 三种基本状态

```python
# 状态1：稳态（Steady State）
V_from = V_to
t_since → ∞（或>50ms，已达到新平衡）
PINN学习：恒定电压下的稳态流场分布

# 状态2：升压过程（Step-up）
V_from < V_to
t_since ∈ [0, 50ms]
PINN学习：极性液被吸向下，油墨被挤压到边缘

# 状态3：降压过程（Step-down）
V_from > V_to
t_since ∈ [0, 50ms]
PINN学习：表面张力恢复，油墨向中心回扩
```

### 3.3 训练数据的电压覆盖

当前主线的电压覆盖与跳变组合由两部分共同决定：
- `config.data.voltages`（稳态与域内配点覆盖）
- `DataGenerator` 中的方波响应采样（升压/降压工况覆盖）

 为了避免文档与实现漂移，本节不再写死“0.5V 步进”等具体策略；以单一真源为准：
- [data-generation-core.md](../guides/data-generation-core.md)

**空间分布的对称性利用**：
```python
# 轴对称性：像素是方形，x和y等价
# 只需要学习一个象限，可以镜像到其他象限

# 但由于4个角落可能有细微差异（壁面效应），
# 实际实现中没有强制对称性，让PINN自己学习
```

### 3.4 6D输入的神经网络实现

```python
class TwoPhasePINN(nn.Module):
    def __init__(self):
        # φ网络：输入6D，输出1D
        self.phi_net = nn.Sequential(
            nn.Linear(6, 64),   # [x,y,z,V_from,V_to,t] → 64
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)    # → φ ∈ [0,1]
        )

        # 速度网络：输入7D（6D+φ），输出4D（u,v,w,p）
        self.vel_net = nn.Sequential(
            nn.Linear(7, 64),   # [x,y,z,V_from,V_to,t,φ] → 64
            ...
            nn.Linear(32, 4)    # → (u,v,w,p)
        )
```

**关键设计**：
- φ网络和速度网络分离
- 速度网络以φ为输入（物理耦合）
- 总参数量：~17K-34K（取决于隐藏层大小）

---

## 4. VOF vs Level Set方法对比

### 4.1 VOF（Volume of Fluid）方法

**核心思想**：用体积分数表示两相分布

```python
# φ场的物理定义
φ = 1:    纯油墨（不透明）
φ = 0:    纯极性液体（透明）
0<φ<0.5:  界面过渡区

# 输运方程
∂φ/∂t + u·∇φ = 0  # 与Level Set形式相同

# 界面定义
interface: φ = 0.5  # 模糊界面
```

**优势**：
1. **自然体积守恒**：φ∈[0,1]，积分自动守恒
2. **单一速度场**：u在整个域上定义一致
3. **训练稳定**：30K-40K epochs收敛
4. **实现简单**：不需要额外约束（如Eikonal）

**劣势**：
1. **界面模糊**：φ=0.5不是精确几何位置
2. **几何量不准**：
   - 法向量：n = ∇φ/|∇φ|（在界面处不精确）
   - 曲率：κ = ∇²φ（对φ=0.5模糊界面不够准确）
3. **体积守恒误差**：<1% (v4.5已解决)

**VOF的体积守恒问题**：
```python
# 理论上：∂φ/∂t + u·∇φ = 0 应保证体积守恒
# 实际上：数值误差导致体积漂移

# 体积守恒损失
loss_volume = (torch.sum(phi[phi>0.5]) - V_ink_target)²

# 但即便加了体积损失，仍有15-20%误差
# 原因：VOF输运方程的数值误差
```

### 4.2 Level Set方法

**核心思想**：用符号距离函数表示界面

```python
# ψ场的物理定义
ψ > 0:    油墨相
ψ < 0:    极性液相
ψ = 0:    界面（精确位置）

# 输运方程
∂ψ/∂t + u·∇ψ = 0  # 与VOF形式相同

# Eikonal约束（保持符号距离函数性质）
|∇ψ| = 1  # 确保ψ真的是距离函数
```

**优势**：
1. **精确界面**：ψ=0是明确的几何位置
2. **几何量精确**：
   - 法向量：n = ∇ψ/|∇ψ|（精确，因为|∇ψ|=1）
   - 曲率：κ = ∇²ψ（精确，因为ψ是SDF）
3. **研究新颖性**：Level Set + PINN在EWD中是创新

**劣势**：
1. **体积守恒困难**：
   - ψ无界，不自然守恒
   - 需要体积约束：∫_{ψ<0} dV = V_ink
   - 但Heaviside函数 H(-ψ) 不可微，数值困难
2. **Eikonal约束困难**：
   - |∇ψ|=1 是强约束，难以精确满足
   - 放松约束会导致ψ不再是SDF
3. **训练不稳定**：
   - NS方程容易爆炸
   - 需要禁用NS方程才能训练
4. **二阶导数问题**：
   - 曲率计算需要∇²ψ
   - 数值不稳定

### 4.3 当前训练状态对比 (2026-01-29)

| 指标 | VOF | Level Set |
|------|-----|-----------|
| 训练完成度 | ✅ 已完成 (60,000 epochs, v4.5-sharpening) | ⚠️ 17.8K/40K (实验，主线已切换回) |
| NS方程 | ✅ 正常工作 | ⚠️ 不稳定（未在主线使用） |
| 体积守恒 | ✅ <1% 误差 (v4.5已解决) | ❌ 不适用 |
| 界面清晰度 | ⚠️ 已改善 (新增锐化损失) | ✅ 理论精确（ψ=0） |
| 生产就绪 | ✅ 是 (主线) | ❌ 实验性 |

**推荐策略**：
- **生产应用**：使用 VOF 方法（稳定、已验证）
- **研究方向**：优先改进 VOF 体积守恒（方案 C），暂不使用 Level Set

**v4.5 锐化改进 (2026-01-29)**：
- 新增界面锐化损失 `L_sharp = λ·φ(1-φ)`
- 权重: 1.0
- φS 损失: 1.12 → 0.18 (↓84%)

---

## 5. 训练策略的物理直觉

### 5.1 渐进式训练的动机

**问题**：直接训练全物理PINN，损失立即爆炸

```
Epoch 0: Loss = 757 (主要是数据损失)
Epoch 10: Loss = 1.2e15 (NS方程爆炸！)
Epoch 20: NaN (训练崩溃)
```

**原因**：
- 网络随机初始化，预测毫无物理意义
- 直接施加N-S强约束，梯度巨大
- 物理损失相互冲突，无法收敛

**解决方案**：渐进式训练（课程学习）

### 5.2 三阶段训练策略

```python
# Stage 1: 纯数据学习 (Epoch 0-5000)
def get_stage1_loss(epoch, predictions, targets):
    # 只拟合数据，不施加物理约束
    loss_data = interface_loss(predictions, targets)
    loss_ic = initial_condition_loss(predictions)
    loss_bc = boundary_condition_loss(predictions)

    # 物理权重全部为0
    return loss_data + loss_ic + loss_bc

# 目的：让网络先学习φ场的基本分布
# - 知道z<h_interface是油墨，z>h_interface是极性液
# - 知道电压增加，开口率增大
# - 建立粗略的流场直觉


# Stage 2: 引入连续性+VOF (Epoch 5000-17000)
def get_stage2_loss(epoch, predictions, targets):
    alpha = tanh((epoch - 5000) / 6000)  # 0→1平滑过渡

    loss_data = ...
    loss_continuity = alpha * 0.5 * continuity_loss(predictions)
    loss_vof = alpha * 0.5 * vof_transport_loss(predictions)
    loss_ns = alpha * 0.01 * ns_loss(predictions)  # 预热，权重很小

    return loss_data + loss_continuity + loss_vof + loss_ns

# 目的：逐步引入物理约束，但NS权重很小
# - 连续性：∇·u = 0（质量守恒）
# - VOF输运：∂φ/∂t + u·∇φ = 0（界面演化）
# - NS预热：让网络适应N-S约束形式


# Stage 3: 完整物理约束 (Epoch 5500-60000)
def get_stage3_loss(epoch, predictions, targets):
    loss_data = ...
    loss_continuity = 0.5 * continuity_loss(predictions)
    loss_vof = 0.5 * vof_transport_loss(predictions)
    loss_ns = 0.1 * ns_loss(predictions)  # 完整NS权重
    loss_surface_tension = 0.01 * surface_tension_loss(predictions)

    return loss_data + loss_continuity + loss_vof + loss_ns + loss_surface_tension

# 目的：全物理约束，精细调整
# - 完整N-S（含电润湿力、表面张力）
# - 所有物理约束协同工作
```

### 5.3 权重调度的tanh函数

```python
# 为什么要用tanh而不是线性过渡？
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

**实际效果**：
```
Epoch 4900: NS weight = 0.00 (NS loss = 123.45)
Epoch 5000: NS weight = 0.00 (NS loss = 120.11)  ← 开始Stage 2
Epoch 5100: NS weight = 0.00 (NS loss = 115.23)
Epoch 5500: NS weight = 0.00 (NS loss = 98.76)
Epoch 6000: NS weight = 0.01 (NS loss = 45.32)  ← 缓慢增加
Epoch 7000: NS weight = 0.05 (NS loss = 23.11)
Epoch 12000: NS weight = 0.10 (NS loss = 12.45)  ← Stage 3，完整权重
```

### 5.4 动态权重调整（三层机制）

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

**效果**：
- 防止某个损失项主导训练
- 自动平衡不同物理约束
- 40%训练稳定性提升

---

## 6. 代码实现的关键细节

### 6.1 自动微分与物理残差

```python
def compute_ns_residual(x, predictions, model):
    """
    计算Navier-Stokes方程残差
    """
    # 提取预测变量
    u, v, w, p, phi = predictions[:, 0:5]

    # 关键：启用自动微分
    x.requires_grad_(True)

    # 重新计算预测（以启用梯度跟踪）
    predictions = model(x)
    u, v, w, p, phi = predictions[:, 0:5]

    # 一阶导数
    u_t = torch.autograd.grad(u, x[:, 5:6], ...)[0]  # ∂u/∂t
    u_x = torch.autograd.grad(u, x[:, 0:1], ...)[0]  # ∂u/∂x
    ...

    # 二阶导数（拉普拉斯）
    u_xx = torch.autograd.grad(u_x, x[:, 0:1], ...)[0]  # ∂²u/∂x²
    ...

    # N-S方程
    # ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew
    residual_u = rho * (u_t + u*u_x + v*u_y + w*u_z) \
                 + p_x \
                 - mu * (u_xx + u_yy + u_zz) \
                 - F_st_x \
                 - F_ew_x

    return residual_u, residual_v, residual_w, residual_continuity
```

**关键点**：
1. `x.requires_grad_(True)`：启用自动微分
2. 分离导数计算：先求一阶，再求二阶
3. 批量计算：避免循环，用张量操作

### 6.2 电润湿力的实现

```python
def compute_electrowetting_force(x, phi, physics_params):
    """
    计算电润湿驱动力
    F_ew = -σ_ew × ∇φ

    其中 σ_ew = ε₀εᵣ(V-V_T)²/(2d)
    """
    # 从输入中提取电压
    V_to = x[:, 4]  # 当前电压

    # Maxwell应力
    epsilon_0 = physics_params['epsilon_0']
    epsilon_r = physics_params['epsilon_r']
    d = physics_params['dielectric_thickness']
    V_threshold = physics_params['V_threshold']

    # 归一化（防止爆炸）
    sigma_surface_tension = physics_params['surface_tension']
    sigma_ew = epsilon_0 * epsilon_r * (V_to - V_threshold)**2 / (2 * d)
    sigma_ew_normalized = sigma_ew / sigma_surface_tension  # 关键：归一化！

    # φ的梯度
    phi_grad = torch.autograd.grad(phi, x, create_graph=True)[0]
    grad_x, grad_y, grad_z = phi_grad[:, 0:1], phi_grad[:, 1:2], phi_grad[:, 2:3]

    # 电润湿力 = -σ_ew × ∇φ
    F_ew_x = -sigma_ew_normalized * grad_x
    F_ew_y = -sigma_ew_normalized * grad_y
    F_ew_z = -sigma_ew_normalized * grad_z

    return F_ew_x, F_ew_y, F_ew_z
```

**为什么需要归一化？**
```python
# 未归一化（v4.3之前）：
sigma_ew = 8.854e-12 * 12.0 * (20)**2 / (2 * 4e-7)
         = 0.531 N/m

sigma_surface_tension = 0.045 N/m

F_ew = -0.531 / 0.045 * ∇φ = -11.8 * ∇φ

# 问题：F_ew远大于其他力，主导NS方程
# → 导致NS loss爆炸到1e15
# → 训练崩溃

# 归一化后（v4.3）：
F_ew = -(0.531 / 0.045) / 0.045 * ∇φ = -1.0 * ∇φ（相对大小）

# 效果：NS loss稳定在0.1量级
```

### 6.3 数据生成中的Z采样策略

当前实现的空间采样不是“按界面高度做高斯采样”，而是以开口率 η 为引导进行加密：
- η 小：倾向在开口边界附近加密
- η 大：倾向在角落/边缘加密
- z 方向：一部分点在油墨层附近加密（对界面/底部形态更敏感）

详细逻辑与代码口径以单一真源为准：
- [data-generation-core.md](../guides/data-generation-core.md)

**为什么界面附近高密度采样？**
- 界面是物理最复杂的区域
- 曲率、表面张力、接触角都发生在界面
- 高密度采样帮助网络学习界面细节

### 6.4 体积守恒损失的实现

```python
def compute_volume_conservation_loss(phi, x, V_ink_target):
    """
    计算体积守恒损失
    """
    # 体积分数积分（数值）
    # ∫ φ dV ≈ Σ φ_i × ΔV_i
    phi_sum = torch.sum(phi)  # 假设均匀网格，ΔV=1
    V_ink_predicted = phi_sum / len(phi) * Lx * Ly * Lz

    # 体积守恒误差
    volume_error = (V_ink_predicted - V_ink_target)**2

    return volume_error
```

**问题**：即使加了体积损失，仍有15-20%误差
- VOF输运方程的数值误差
- 网络预测的φ场不完全满足∇·u=0
- 梯度下降优化，无法精确满足等式约束

**改进方案**：
- 方案A：增加VOF权重（效果有限）
- 方案B：显式体积损失（效果差，18.92%误差）
- 方案C：连续性损失改进（测试中）

---

## 7. 常见陷阱与调试

### 7.1 NS方程爆炸

**症状**：
```
Epoch 100: NS loss = 123.45
Epoch 110: NS loss = 1.2e6
Epoch 120: NS loss = 1.5e15
Epoch 130: NaN
```

**原因**：
1. 电润湿力未归一化（v4.3之前）
2. NS权重增加太快
3. 学习率太大

**解决方案**：
```python
# 1. 电润湿力归一化
sigma_ew_normalized = sigma_ew / sigma_surface_tension

# 2. 渐进式增加NS权重
ns_weight = base_weight * tanh((epoch - start) / width)

# 3. 降低学习率
lr = 5e-4 * (0.95 ** (epoch // 1000))
```

### 7.2 损失震荡不收敛

**症状**：
```
Epoch 1000: Loss = 45.32
Epoch 1100: Loss = 52.18
Epoch 1200: Loss = 38.91
Epoch 1300: Loss = 61.25
```

**原因**：
1. 不同损失项权重不平衡
2. 某个损失项主导训练
3. 数据分布不均匀

**解决方案**：
```python
# 1. 动态权重调整
if loss_ns > 10 * loss_data:
    ns_weight *= 0.5  # 降低NS权重

# 2. Loss归一化
loss_data_normalized = loss_data / n_data_points
loss_ns_normalized = loss_ns / n_collocation_points

# 3. 检查数据分布
print(f"Voltage range: {V.min():.1f} - {V.max():.1f}")
print(f"Time range: {t.min():.3f} - {t.max():.3f}s")
print(f"Z distribution: {z.mean():.2e} ± {z.std():.2e}")
```

### 7.3 开口率预测错误

**症状**：
```
20V: 预测开口率 = 10.2% (目标67%)
30V: 预测开口率 = 1.9% (目标84%)
```

**原因**：
1. 训练数据物理关系错误
2. 符号约定错误
3. 体积守恒公式错误

**检查清单**：
```python
# 1. 检查训练数据
for V in [0, 10, 20, 30]:
    h = get_interface_height(V, t=0.02)
    eta = contact_angle_to_aperture(get_contact_angle(V))
    print(f"V={V}V: η={eta:.1%}, h={h*1e6:.1f}μm")
    # 应该：V↑ → η↑ → h↑

# 2. 检查符号约定
psi_at_z0 = compute_level_set(x=0.5e-3, y=0.5e-3, z=0, V=20, t=0.02)
print(f"ψ at z=0 (20V): {psi_at_z0:.3f}")
# Level Set: ψ>0 (油墨), ψ<0 (极性液)
# VOF: φ=1 (油墨), φ=0 (极性液)

# 3. 检查体积守恒
h_check = h_initial / (1 - eta)
print(f"Volume check: h={h*1e6:.1f}μm, h_check={h_check*1e6:.1f}μm")
```

### 7.4 梯度消失/爆炸

**症状**：
```
Epoch 500: grad_norm = 1.2e-8 (消失)
Epoch 500: grad_norm = 5.6e7 (爆炸)
```

**调试**：
```python
# 在训练循环中监控梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100 or grad_norm < 1e-6:
            print(f"Warning: {name} grad_norm = {grad_norm:.2e}")

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 检查激活值
for name, module in model.named_modules():
    if isinstance(module, nn.Tanh):
        output = module(x)
        if output.abs().max() < 0.1:
            print(f"Warning: {name} output saturated near 0")
```

---

## 8. 总结：项目的核心洞察

### 8.1 物理与机器学习的结合

**PINN不是黑盒**：
- 物理约束提供归纳偏置（inductive bias）
- 数据约束提供边界条件和目标
- 两者结合才能收敛到合理的解

**数据的质量比数量重要**：
- 61个电压点 × 50个时间点 = 3050个样本
- 但这些样本必须满足物理关系
- 错误的物理关系会导致错误的模型

### 8.2 工程折衷

**VOF vs Level Set**：
- VOF：稳定但模糊（生产可用）
- Level Set：精确但脆弱（研究价值）

**体积守恒**：
- 理论：VOF输运方程自动守恒
- 实际：15-20%误差（可接受）
- 改进：需额外的体积损失项

**训练时间 vs 精度**：
- 30K epochs：2-3小时，误差~7%
- 40K epochs：3-4小时，误差~5%
- 边际收益递减

### 8.3 未来方向

1. **体积守恒改进**：方案C（连续性损失改进）
2. **Level Set探索**：解决NS方程爆炸问题
3. **多像素耦合**：考虑像素间的相互作用
4. **参数迁移**：不同器件几何的参数化建模
5. **不确定性量化**：PINN预测的置信区间

---

**最后更新**: 2026-02-04
**作者**: EFD-PINNs Team
**反馈**: 请提issue到项目仓库
