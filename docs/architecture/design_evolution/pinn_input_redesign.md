# PINN 6D Triad 输入空间设计文档

**最后更新**: 2026-02-04
**状态**: ✅ 已实现 (Implemented)
**版本**: v2.0 (Triad Architecture)

---

## 1. 设计概述

本设计已在 **v6.2.0** 版本中完全实现。新的 PINN 模型采用 **6D Triad** 输入空间 `(x, y, z, V_from, V_to, t_since)`，解决了旧版输入 `(t, V, V_prev)` 的时间歧义性和动态响应建模难题。

### 核心变更
- **输入维度**: 5D `(x,y,z,t,V)` → 6D `(x,y,z, V_from, V_to, t_since)`
- **物理机制**: 明确区分 **升压 (Electrowetting)** 与 **降压 (Surface Tension Recovery)**
- **数据结构**: 引入 `Triad` 数据类，统一管理空间、电压状态和时间


当前 PINN 模型输入为 `(x, y, z, t, V, V_prev)`，存在以下问题：

1. **绝对时间 `t` 的歧义性**
   - 模型不知道电压变化发生在什么时刻
   - 无法区分 "20V 刚施加 2ms" 和 "20V 已经稳定 20ms"

2. **升压/降压物理机制不同**
   - 升压：电润湿力驱动（极性液体推动油墨）
   - 降压：表面张力驱动（油墨自发铺展回中心）
   - 当前设计无法让模型有效区分这两种模式

3. **可视化异常现象**
   - 升压时 20V/30V 曲线在 t≈10ms 处有非物理的峰值
   - 降压后开口率不降反升

---

## 2. 新输入设计

### 2.1 输入格式

```
输入: (x, y, z, t_since, V_from, V_to)

参数说明:
- x, y, z:   空间坐标 (m)
- t_since:   电压变化后经过的相对时间 (s)
- V_from:    变化前的电压 (V)
- V_to:      变化后的电压 (V)
```

### 2.2 输入归一化

```python
x_norm = x / Lx           # [0, 1]
y_norm = y / Ly           # [0, 1]
z_norm = z / Lz           # [0, 1]
t_norm = t_since / t_max  # [0, 1], t_max = 0.05s
V_from_norm = V_from / 30 # [0, 1]
V_to_norm = V_to / 30     # [0, 1]
```

---

## 3. 物理模型

### 3.1 升压过程 (V_from < V_to)

**驱动机制**: 电润湿力
- 电场作用于极性液体，使其润湿疏水层
- 极性液体铺展，将油墨从中心推向边缘/角落

**接触角动力学** (二阶欠阻尼系统):
```
θ(t_since) = θ_eq + (θ_0 - θ_eq) × exp(-ζ×ω₀×t_since) × [cos(ω_d×t_since) + (ζ/√(1-ζ²))×sin(ω_d×t_since)]

参数:
- θ_0 = 120°     初始接触角
- θ_eq = f(V_to) 稳态接触角 (Young-Lippmann)
- τ = 5ms        时间常数
- ζ = 0.8        阻尼比
- ω₀ = 1/τ       自然频率
- ω_d = ω₀×√(1-ζ²) 阻尼频率
```

**开口率**: η = f(θ)，由接触角映射得到

### 3.2 降压过程 (V_from > V_to)

**驱动机制**: 表面张力
- 电场消失，接触角瞬间恢复到 θ_0
- 油墨在表面张力作用下从边缘向中心铺展

**接触角**: 瞬间恢复
```
θ(t_since) = θ_0 = 120°  (对于 V_to = 0)
```

**开口率动力学** (一阶指数衰减):
```
η(t_since) = η_initial × exp(-t_since / τ_recovery)

参数:
- η_initial = 降压前的稳态开口率 = f(V_from)
- τ_recovery = 7.5ms  表面张力恢复时间常数
```

### 3.3 恒定电压 (V_from = V_to)

**稳态条件**:
```
θ = θ_eq(V)  稳态接触角
η = f(θ)     稳态开口率
```

---

## 4. 训练数据生成

### 4.1 采样策略 (v6.2.1)

```python
def generate_training_data():
    # 采用 0.5V 步进采样策略 (61 个电压状态点)
    voltages = np.linspace(0, 30, 61)
    
    # 1. 升压采样 (0 -> V)
    # 覆盖 0.5V 间隔的 61 个目标电压
    
    # 2. 降压采样 (V -> 0)
    # 覆盖从 0.5V 到 30V 的 60 个初始电压状态
    
    # 3. 稳态采样 (V -> V)
    # 确保模型在 t_since 较大时收敛到静态物理场
```

### 4.2 训练阶段

采用 **60,000 轮三阶段渐进式训练**:
1. **Phase 1 (0-1,500)**: 几何引导与相场初始化
2. **Phase 2 (1,500-5,500)**: 物理约束加载 (NS + VOF)
3. **Phase 3 (5,500-60,000)**: 动态响应精调与电压跳变平滑

---

## 5. 验证与评估

### 5.1 Response Time 测量

```python
# 升压响应时间
def measure_rise_time(model, V_target):
    """测量开口率从 10% 到 90% 稳态值的时间"""
    η_steady = get_steady_aperture(V_target)
    η_10 = 0.1 * η_steady
    η_90 = 0.9 * η_steady
    
    for t_since in np.linspace(0, 0.030, 100):
        η = compute_aperture(model, t_since, V_from=0, V_to=V_target)
        if η >= η_90:
            return t_since
    return None

# 降压恢复时间
def measure_fall_time(model, V_initial):
    """测量开口率从 90% 降到 10% 初始值的时间"""
    η_initial = get_steady_aperture(V_initial)
    η_90 = 0.9 * η_initial
    η_10 = 0.1 * η_initial
    
    for t_since in np.linspace(0, 0.030, 100):
        η = compute_aperture(model, t_since, V_from=V_initial, V_to=0)
        if η <= η_10:
            return t_since
    return None
```

### 5.2 C-V 曲线 (稳态)

```python
def plot_cv_curve(model):
    """绘制稳态开口率 vs 电压曲线"""
    voltages = np.linspace(0, 30, 31)
    apertures = []
    
    for V in voltages:
        # 稳态: V_from = V_to = V, t_since 足够大
        η = compute_aperture(model, t_since=0.030, V_from=V, V_to=V)
        apertures.append(η)
    
    plt.plot(voltages, apertures)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Aperture Ratio η')
```

### 5.3 动态响应曲线

```python
def plot_dynamic_response(model, V_target):
    """绘制升压/降压动态响应"""
    times = np.linspace(0, 0.030, 100)
    
    # 升压响应
    η_rise = [compute_aperture(model, t, V_from=0, V_to=V_target) for t in times]
    
    # 降压响应
    η_fall = [compute_aperture(model, t, V_from=V_target, V_to=0) for t in times]
    
    plt.plot(times*1000, η_rise, label='Rise (0→V)')
    plt.plot(times*1000, η_fall, label='Fall (V→0)')
```

---

## 6. 局限性与扩展

### 6.1 当前设计的局限性

1. **假设每次跳变前系统处于稳态**
   - 不支持多步连续跳变（如 0→20→30→20→0）
   - 不支持任意波形输入

2. **只支持单次阶跃响应**
   - 升压: 0 → V_target
   - 降压: V_initial → 0

### 6.2 扩展方案：LSTM-PINN

如果需要支持多步序列或任意波形，可扩展为 LSTM 架构：

```
输入:
- 空间坐标: (x, y, z)
- 电压序列: [V(t0), V(t1), ..., V(tn)]

架构:
- LSTM 编码器: 处理电压时间序列，输出隐状态 h
- MLP 解码器: (x, y, z, h) → φ

优势:
- 自动学习历史依赖
- 支持任意波形输入
- 支持多步连续跳变

开发时间估算: 15-22 小时 (2-3天)
```

---

## 7. 核心实现 (v6.2.0+)

### 7.1 主要修改
- **模型架构**: `TwoPhasePINN` 及其前向传播已适配 6D Triad 输入。
- **数据生成**: `DataGenerator` 现在支持 0.5V 步进采样及 `t_since` 相对时间生成。
- **可视化**: `evaluate.py` 已更新，支持 6D 输入切片展示。
- **配置**: 使用 `device_calibrated_physics.json` 作为统一物理参数源。

---

## 8. 参考资料

- Young-Lippmann 方程: cos(θ) = cos(θ₀) + ε₀εᵣV²/(2γd)
- 电润湿动力学: 二阶欠阻尼系统，τ = 5ms, ζ = 0.8
- 表面张力恢复: 一阶指数衰减，τ_recovery = 7.5ms
