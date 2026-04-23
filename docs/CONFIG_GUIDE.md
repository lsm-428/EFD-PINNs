# EFD3D 配置文件使用指南

## 📋 配置文件版本

### v4.5-standard (基准版本)
- **文件**: `config/v4.5-standard.json`
- **对应训练**: `pinn_20260418_130834` (已完成训练)
- **特点**: 经过验证的稳定配置，能够可靠收敛
- **动量权重**: 0.002 (0.02 × 0.1)
- **适用场景**: 基准测试、稳定性验证、结果复现

### v4.6-momentum-optimized (优化版本)
- **文件**: `config/v4.6-momentum-optimized.json`
- **基准**: 基于 `pinn_20260420_085640` 训练结果优化
- **特点**: 优化动量方程学习，提高流场预测精度
- **动量权重**: 0.250 (0.5 × 0.5)
- **提升倍数**: 125×
- **问题**: 动量权重过大导致训练后期不稳定
- **适用场景**: 流场学习优化研究（需谨慎使用）

### v4.7-momentum-balanced (平衡版本) 🆕
- **文件**: `config/v4.7-momentum-balanced.json`
- **基准**: 基于 `pinn_20260420_085640` 问题分析优化
- **特点**: 平衡动量方程学习，保持可见性同时避免不稳定
- **动量权重**: 0.090 (0.3 × 0.3)
- **提升倍数**: 45×
- **适用场景**: 生产训练、稳定性优先的流场学习

## 🚀 使用方式

### 1. 基准复现 (v4.5)
```bash
# 复现pinn_20260418_130834的训练结果
uv run train_two_phase.py --config config/v4.5-standard.json

# 监控训练过程
tensorboard --logdir outputs/train/pinn_*/runs
```

### 2. 优化训练 (v4.6)
```bash
# 使用优化配置进行训练
uv run train_two_phase.py --config config/v4.6-momentum-optimized.json

# 监控训练过程
tensorboard --logdir outputs/train/pinn_*/runs
```

### 3. 平衡训练 (v4.7) 🆕
```bash
# 使用平衡配置进行训练（推荐）
uv run train_two_phase.py --config config/v4.7-momentum-balanced.json

# 监控训练过程
tensorboard --logdir outputs/train/pinn_*/runs
```

### 3. 配置对比分析
```bash
# 比较两个版本的差异
python scripts/compare_configs.py

# 验证权重配置
python scripts/verify_config_weights.py
```

## ⚖️ 权重优化详情

### 相对于pinn_20260418_130834的改进

| 参数 | 训练时 | v4.6 | 提升倍数 | 改进目标 |
|------|--------|------|---------|----------|
| momentum_u/v/w (残差) | 0.02 | 0.5 | 25× | 动量学习 |
| ns_weight (全局) | 0.1 | 0.5 | 5× | 流场优化 |
| **复合动量权重** | **0.002** | **0.250** | **125×** | **核心改进** |
| surface_tension (残差) | 0.3 | 1.0 | 3.3× | 界面精度 |
| surface_tension_weight | 0.01 | 0.1 | 10× | 表面张力 |
| **复合表面张力** | **0.003** | **0.100** | **33×** | **界面优化** |

### v4.7平衡版本权重
| 参数 | 训练时 | v4.7 | 提升倍数 | 改进目标 |
|------|--------|------|---------|----------|
| momentum_u/v/w (残差) | 0.02 | 0.3 | 15× | 动量学习平衡 |
| ns_weight (全局) | 0.1 | 0.3 | 3× | 流场平衡优化 |
| **复合动量权重** | **0.002** | **0.090** | **45×** | **平衡改进** |
| surface_tension (残差) | 0.3 | 1.0 | 3.3× | 界面精度 |
| surface_tension_weight | 0.01 | 0.1 | 10× | 表面张力 |
| **复合表面张力** | **0.003** | **0.100** | **33×** | **界面优化** |

## 🎯 训练效果对比

### pinn_20260418_130834 (v4.5基准)
- ✅ **总损失**: 34.28 (良好收敛)
- ✅ **界面损失(LV)**: 0.01 (优秀)
- ✅ **体积守恒**: 0.15 (高精度)
- ⚠️ **动量损失**: 不可见 (权重过低)
- ⚠️ **表面张力**: 0.5 (收敛一般)

### v4.6预期效果
- ✅ **总损失**: 30-32 (进一步提升)
- ✅ **界面损失**: 保持或改善
- ✅ **体积守恒**: 保持高精度
- ✅ **动量损失**: 清晰可见且收敛 (核心改进)
- ✅ **表面张力**: 0.2-0.3 (更快收敛)

### v4.7预期效果 🆕
- ✅ **总损失**: 28-30 (最佳平衡点)
- ✅ **界面损失**: 保持或改善
- ✅ **体积守恒**: 保持高精度
- ✅ **动量损失**: 清晰可见且**稳定收敛** (平衡改进)
- ✅ **表面张力**: 0.2-0.3 (稳定收敛)
- ✅ **训练稳定性**: 避免后期震荡 (关键改进)

## 📊 监控指标详解

### TensorBoard监控重点

**v4.5训练时的监控局限**：
- ❌ 缺少动量相关损失曲线
- ❌ 表面张力收敛过程不清晰
- ✅ 界面、体积等几何约束监控完整

**v4.6的监控增强**：
- ✅ **Loss/momentum_u/v/w** - 动量损失活跃可见
- ✅ **Loss/surface_tension** - 表面张力快速收敛
- ✅ **gradients/momentum** - 动量梯度分布
- ✅ **完整物理约束** - 所有损失项平衡发展

### 训练日志关键指标

**几何约束 (v4.5已优化)**：
- **LV** (界面损失): < 0.01 ✅
- **Vol** (体积守恒): < 0.2 ✅  
- **θ** (接触角): < 0.5 ✅
- **IF** (界面): < 2.5 ✅

**新增关注指标 (v4.6重点)**：
- **动量损失**: 预期在日志中清晰显示
- **表面张力**: 预期收敛到0.2-0.3
- **物理平衡性**: 各项损失比例更均衡

## 🔧 自定义配置指南

### 创建新配置版本

```bash
# 1. 基于v4.6创建自定义版本
cp config/v4.6-momentum-optimized.json config/v4.7-custom.json

# 2. 更新元数据信息
{
  "metadata": {
    "version": "v4.7-custom",
    "description": "Custom configuration for specific research",
    "based_on": "v4.6-momentum-optimized"
  }
}

# 3. 调整关键参数
{
  "physics": {
    "ns_weight": 0.3,  // 调整全局动量权重
    "residual_weights": {
      "momentum_u": 0.3,  // 调整残差权重
      "surface_tension": 0.8  // 调整表面张力
    }
  }
}
```

### 权重调整策略

#### 动量优化方向
```json
{
  "ns_weight": 0.3-0.8,  // 全局权重
  "residual_weights": {
    "momentum_u": 0.3-0.8,  // 残差权重
    "momentum_v": 0.3-0.8,
    "momentum_w": 0.3-0.8
  }
}
```
**复合权重范围**: 0.09-0.64 (45-320倍于原始)

#### 表面张力优化
```json
{
  "surface_tension_weight": 0.05-0.2,  // 全局
  "residual_weights": {
    "surface_tension": 0.5-1.5  // 残差
  }
}
```
**复合权重范围**: 0.025-0.3 (8-100倍于原始)

## 🔄 实验流程建议

### A/B测试对比
```bash
# 终端1: 基准测试
gpu=0 uv run train_two_phase.py --config config/v4.5-standard.json

# 终端2: 优化测试  
gpu=1 uv run train_two_phase.py --config config/v4.6-momentum-optimized.json

# 终端3: 监控对比
tensorboard --logdir outputs/train/pinn_*/runs
```

### 阶段性验证
1. **初期 (0-5000 epochs)**: 观察动量损失是否可见
2. **中期 (5000-20000 epochs)**: 检查物理平衡性
3. **后期 (20000+ epochs)**: 评估最终收敛质量

### 评估指标
- **动量学习**: TensorBoard中动量曲线活跃程度
- **物理平衡**: 各项损失比例是否合理
- **收敛质量**: 总体损失是否进一步降低
- **预测精度**: 流场和界面预测的物理合理性

## 📝 配置管理最佳实践

### 版本控制
- ✅ 每个实验使用独立配置文件
- ✅ 配置文件随代码一起版本控制
- ✅ 记录配置变更和实验结果

### 参数记录
```markdown
## 实验记录: v4.6-momentum-optimized
- **目标**: 优化动量学习
- **关键变更**: 动量权重提升125倍
- **预期效果**: 动量损失可见，流场精度提升
- **训练命令**: uv run train_two_phase.py --config config/v4.6-momentum-optimized.json
```

### 故障排除

**Q: 动量损失仍然不可见？**
A: 检查是否使用了正确的配置文件，验证权重加载

**Q: 训练不稳定或震荡？**
A: 逐步增加动量权重：0.5 → 0.3 → 0.2，找到稳定点

**Q: 其他损失项恶化？**
A: 动量权重过大，需要重新平衡，或增加总训练epochs

**Q: 收敛速度变慢？**
A: 适当增加学习率或调整动态权重参数

## 🚀 进阶使用

### 参数扫描实验
```bash
# 创建多个权重配置
for momentum_weight in 0.2 0.3 0.5 0.8; do
  sed "s/\"momentum_u\": 0.5/\"momentum_u\": $momentum_weight/g" config/v4.6-momentum-optimized.json > config/v4.6-momentum-$momentum_weight.json
done

# 批量训练
for config in config/v4.6-momentum-*.json; do
  uv run train_two_phase.py --config $config --epochs 20000
done
```

### 自动化评估
```bash
# 训练后自动评估
python evaluate.py outputs/train/pinn_*/ --metrics all

# 生成对比报告
python scripts/generate_comparison_report.py
```

---

**配置版本历史**:
- **v4.5-standard**: 2026-04-19, 基于pinn_20260418_130834训练
- **v4.6-momentum-optimized**: 2026-04-20, 动量权重优化版本 (125×)
- **v4.7-momentum-balanced**: 2026-04-20, 平衡动量权重版本 (45×)

*最后更新: 2026-04-20 | 版本: v4.7*