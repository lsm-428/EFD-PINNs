# EFD-PINNs 项目路线图

**最后更新**: 2025-12-10  
**状态**: ✅ Stage 1 完成（已校准） | ✅ Stage 2 完成（已验证）

---

## 🔬 电润湿显示工作原理

```
无电压（关态）：油墨平铺在像素底部 → 显色状态
施加电压（开态）：极性液体润湿疏水层 → 油墨被动收缩 → 形成开口率 → 透明
```

**关键理解**：电润湿作用在极性液体上，油墨是被动的

---

## 🎯 因果链分步求解

```
电压变化 → 接触角变化 → 油墨被排开(φ场) → 开口率 → 像素亮度
   因          因              果            果        果
   
Stage 1       ←─────────── Stage 2 ───────────→
(✅ 已校准)              (✅ 已验证)
HybridPredictor       TwoPhasePINN
   V → θ(t)      θ(t) 作为边界条件 → φ(x,y,z,t) → η(t)
```

**关键设计**: Stage 2 的 PINN 使用 Stage 1 的接触角预测作为边界条件输入，
确保了物理因果链的正确性。

---

## 📊 阶段详情

### Stage 1：接触角预测 ✅ 已校准

**目标**: 电压 V → 接触角 θ → 开口率 η

| 指标 | 目标 | 实现 | 状态 |
|------|------|------|------|
| 20V 开口率 | 67% | 66.7% | ✅ 误差 0.3% |
| 稳态精度 (30V) | <3° | 0.7° | ✅ |
| 角度变化 | 33° | 30.1° | ✅ |
| 超调 | <15% | 3.9% | ✅ |
| 响应时间 | <30ms | 13ms | ✅ |

**已校准参数**:
- SU-8 400nm (ε=3.0) + Teflon 400nm (ε=1.9)
- 极性液体 γ=0.050 N/m（乙二醇混合液）
- 阈值电压 V_T=3V
- 开口率映射：k=0.8, theta_scale=6.0, alpha=0.05

**核心模块**: 
- `src/models/aperture_model.py` - 开口率模型
- `src/predictors/hybrid_predictor.py` - 混合预测器

**配置文件**: `config/stage6_wall_effect.json`

```bash
# 训练命令
python train_contact_angle.py --config config/stage6_wall_effect.json --epochs 3000
```

### Stage 2：两相流 PINN (φ 场求解) ✅ 已验证

**目标**: 接触角 θ(t) → 体积分数场 φ(x,y,z,t) → 开口率 η(t)

| 指标 | Stage 1 | PINN | 误差 | 状态 |
|------|---------|------|------|------|
| V=0V 开口率 | 0% | 0% | 0% | ✅ |
| V=10V 开口率 | 10.3% | 9.2% | -1.1% | ✅ |
| V=20V 开口率 | 66.7% | 73.6% | +6.9% | ✅ |
| V=30V 开口率 | 84.4% | 84.6% | +0.2% | ✅ |
| φ 范围检查 | - | 16/16 | - | ✅ |

**最新训练**: `outputs_pinn_20251210_005737`
- 训练轮数: 10000 epochs
- 最佳损失: 7.29

**核心模块**:
- `src/models/pinn_two_phase.py` - 两相流 PINN 模型
- `src/predictors/pinn_aperture.py` - PINN 开口率预测器

**物理方程**:
- 连续性：∇·u = 0
- VOF：∂φ/∂t + u·∇φ = 0
- N-S：ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st

```bash
# 训练命令
python train_two_phase.py --epochs 10000

# 验证命令
python validate_pinn_physics.py
```

---

## 🏗️ 训练架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    EFD-PINNs 训练架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │    Stage 1      │         │    Stage 2      │               │
│  │  接触角预测      │ ──────▶ │  两相流 PINN    │               │
│  │  (已校准)       │  θ(t)   │  (已验证)       │               │
│  │                 │ 边界条件 │                 │               │
│  │  V → θ(t) → η  │         │  θ(t) → φ → η  │               │
│  └────────┬────────┘         └────────┬────────┘               │
│           │                           │                        │
│           ▼                           ▼                        │
│  config/stage6_wall_effect.json  train_two_phase.py            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 关键文件

| 文件 | 说明 | 阶段 |
|------|------|------|
| `config/stage6_wall_effect.json` | 校准后的配置文件 | Stage 1 |
| `src/models/aperture_model.py` | 开口率模型 | Stage 1 |
| `src/predictors/hybrid_predictor.py` | 混合预测器 | Stage 1 |
| `train_contact_angle.py` | 接触角训练入口 | Stage 1 |
| `src/models/pinn_two_phase.py` | 两相流 PINN 模型 | Stage 2 |
| `src/predictors/pinn_aperture.py` | PINN 开口率预测器 | Stage 2 |
| `train_two_phase.py` | 两相流训练入口 | Stage 2 |
| `validate_pinn_physics.py` | 物理验证脚本 | 通用 |
| `visualize_pinn_results.py` | 可视化脚本 | 通用 |

---

## 📋 任务清单

### 已完成 ✅
- [x] Stage 1：接触角预测 (HybridPredictor)
- [x] Stage 1：物理参数校准（20V→67% 实验验证）
- [x] Stage 1：开口率映射参数从配置文件读取
- [x] Stage 2：两相流 PINN 训练
- [x] Stage 2：物理验证（16/16 通过）
- [x] Stage 1 → Stage 2 集成（接触角作为边界条件）
- [x] 项目重构（src/ 目录结构）
- [x] 测试覆盖
- [x] 文档更新

### 待完成 📝
- [ ] 论文初稿
- [ ] 更多电压/时间组合验证
- [ ] 性能优化

---

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate efd

# 2. 运行测试
python -m pytest tests/ -v

# 3. Stage 1: 验证开口率校准
python -c "
from src.models.aperture_model import EnhancedApertureModel
model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')
theta = model.get_contact_angle(20)
eta = model.contact_angle_to_aperture_ratio(theta)
print(f'20V: θ={theta:.1f}°, η={eta*100:.1f}% (实验值: 67%)')
"

# 4. Stage 2: 两相流 PINN 训练
python train_two_phase.py --epochs 10000

# 5. 物理验证
python validate_pinn_physics.py

# 6. 可视化结果
python visualize_pinn_results.py
```

---

**文档**: [README.md](README.md) | [CURRENT_STATUS.md](CURRENT_STATUS.md) | [USAGE_GUIDE.md](USAGE_GUIDE.md)
