# 项目当前状态

**最后更新**: 2025-12-10  
**状态**: ✅ Stage 1 完成（已校准） | ✅ Stage 2 完成（已验证）

---

## 🔬 物理理解

### 电润湿机制（已校准）

**关键理解**：电润湿作用在极性液体上，油墨是被动的

1. **电润湿作用**：施加电压后，极性液体润湿疏水层（Teflon）
2. **油墨被动收缩**：极性液体铺展，将油墨从像素中心挤向边缘/角落
3. **油墨亲疏水层**：油墨亲底部 Teflon，不亲围堰壁（相对亲水）
4. **贴底收缩**：油墨贴底形成液滴，不会主动爬墙
5. **翻墙条件**：20V 以上油墨被挤压到极限可能翻墙

### φ 场物理定义（标准 VOF）
- **φ=1**: 纯油墨
- **φ=0**: 纯极性液体（透明，开口区域）
- **0<φ<1**: 界面过渡区
- **开口率**: η = 底面 φ<0.5 的面积比例

### Stage 1: 接触角动态响应
- **稳态关系**: V → θ_eq 由 Young-Lippmann 方程解析给出
- **动态过程**: θ(t) = θ_eq + (θ_0 - θ_eq) × 二阶欠阻尼响应

### Stage 2: 两相流 PINN
- **输入**: (x, y, z, t, V) 直接坐标
- **边界条件**: 接触角 θ(t) 来自 Stage 1
- **输出**: φ(x,y,z,t,V) 体积分数场 → 开口率 η

---

## 📊 已校准的物理参数

### 材料参数（实验校准）

| 参数 | 值 | 说明 |
|------|-----|------|
| SU-8 厚度 | 400nm | 介电层 |
| SU-8 介电常数 | ε=3.0 | |
| Teflon 厚度 | 400nm | 疏水层 |
| Teflon 介电常数 | ε=1.9 | |
| 极性液体表面张力 | γ=0.050 N/m | 乙二醇/丙三醇混合液 |
| 阈值电压 | V_T=3V | 开始响应的电压 |
| 初始接触角 | θ₀=120° | |
| 围堰壁接触角 | θ_wall=70-72° | 亲油，毛细束缚油墨 |

### 开口率映射参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 0.8 | 陡度参数 |
| theta_scale | 6.0 | 角度缩放因子 |
| alpha | 0.05 | 电容反馈强度 |
| aperture_max | 0.85 | 最大开口率 |

### 动力学参数

| 参数 | 值 | 说明 |
|------|-----|------|
| τ | 5ms | 响应时间常数 |
| ζ | 0.8 | 阻尼比（欠阻尼） |

### 几何参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 像素尺寸 | 174×174 μm | 内沿尺寸 |
| 油墨厚度 | 3 μm | |
| 围堰高度 | 3.5 μm（实际）/ 20 μm（模型） | 模型用 20μm 防翻墙 |

---

## 📊 Stage 1 验证结果

### 开口率 vs 电压（稳态）

| 电压 | 接触角 | 开口率 | 实验值 | 误差 |
|------|--------|--------|--------|------|
| 0V | 120.0° | 0.0% | 0% | 0% |
| 6V | ~119° | ~1% | 开始响应 | ✓ |
| 10V | 119.2° | 10.3% | - | - |
| 20V | 115.2° | 66.7% | 67% | **0.3%** ✓ |
| 30V | 108.2° | 84.4% | - | - |

---

## 📊 Stage 2 训练结果

### 最新训练 (outputs_pinn_20251210_005737)
- **训练轮数**: 10000 epochs（最佳在 9400）
- **最佳损失**: 7.29
- **模型文件**: `best_model.pth`

### φ 场验证（16/16 通过）

| 电压 | 时间 | 开口率 η | φ 范围 | 状态 |
|------|------|----------|--------|------|
| 0V | 0ms | 0.000 | [0.996, 1.000] | ✅ |
| 0V | 20ms | 0.000 | [1.000, 1.000] | ✅ |
| 10V | 5ms | 0.038 | [0.045, 1.000] | ✅ |
| 10V | 20ms | 0.092 | [0.013, 1.000] | ✅ |
| 20V | 5ms | 0.196 | [0.001, 1.000] | ✅ |
| 20V | 20ms | 0.736 | [0.001, 1.000] | ✅ |
| 30V | 5ms | 0.479 | [0.000, 1.000] | ✅ |
| 30V | 20ms | 0.846 | [0.000, 1.000] | ✅ |

### Stage 1 vs Stage 2 对比（t=20ms 稳态）

| 电压 | Stage 1 η | PINN η | 误差 |
|------|-----------|--------|------|
| 0V | 0.0% | 0.0% | 0% ✓ |
| 10V | 10.3% | 9.2% | -1.1% ✓ |
| 20V | 66.7% | 73.6% | +6.9% |
| 30V | 84.4% | 84.6% | +0.2% ✓ |

---

## 📁 关键文件

| 文件 | 说明 |
|------|------|
| `src/models/aperture_model.py` | Stage 1 开口率模型（已校准） |
| `src/predictors/hybrid_predictor.py` | Stage 1 接触角预测器 |
| `src/models/pinn_two_phase.py` | Stage 2 两相流 PINN 模型 |
| `config/stage6_wall_effect.json` | 校准后的配置文件 |
| `train_contact_angle.py` | Stage 1 训练入口 |
| `train_two_phase.py` | Stage 2 训练入口 |
| `validate_pinn_physics.py` | 物理验证脚本 |

---

## 🚀 快速开始

```python
# Stage 1: 接触角预测
from src.predictors import HybridPredictor
predictor = HybridPredictor(config_path='config/stage6_wall_effect.json')
theta = predictor.predict(voltage=20, time=0.02)
print(f"接触角: {theta:.1f}°")  # ~115.2°

# Stage 1: 开口率预测
from src.models.aperture_model import EnhancedApertureModel
model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')
eta = model.contact_angle_to_aperture_ratio(theta)
print(f"开口率: {eta*100:.1f}%")  # ~66.7%

# Stage 2: PINN φ 场预测
from src.predictors.pinn_aperture import PINNAperturePredictor
predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=20, time=0.02)
print(f"PINN 开口率: {eta:.3f}")  # ~0.736
```

---

## 🎯 完成情况

- [x] Stage 1: 接触角预测（解析公式）
- [x] Stage 1: 物理参数校准（20V→67% 实验验证）
- [x] Stage 1: 开口率映射参数从配置文件读取
- [x] Stage 2: 两相流 PINN 训练
- [x] Stage 2: 物理验证（16/16 通过）
- [x] Stage 1 → Stage 2 物理正确集成
- [x] V=0V 时开口率≈0（油墨平铺）
- [x] V=20V 时开口率≈67%（符合实验）
- [x] V=30V 时开口率≈85%（最大开口）
- [x] 开口率随电压单调增加
- [ ] 论文初稿

---

**文档**: [README.md](README.md) | [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)
