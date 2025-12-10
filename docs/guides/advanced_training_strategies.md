# 高级训练策略

**最后更新**: 2025-12-08

## 概述

本文档介绍 EFD-PINNs 的高级训练策略和优化技巧。

## 渐进式训练

### 三阶段策略

```
阶段 1 (0-5000 epochs): 纯数据学习
    - 物理约束权重: 0
    - 学习数据分布

阶段 2 (5000-15000 epochs): 引入物理约束
    - 逐步增加连续性和 VOF 约束
    - 平滑过渡

阶段 3 (15000+ epochs): 完整物理约束
    - 完整的 N-S 方程
    - 表面张力约束
```

### 权重调度

```python
def get_physics_weights(self, epoch: int) -> Dict[str, float]:
    if epoch < self.stage1_epochs:
        return {"continuity": 0.0, "vof": 0.0, "ns": 0.0}
    elif epoch < self.stage2_epochs:
        progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
        smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))
        return {
            "continuity": 0.1 * smooth_factor,
            "vof": 0.1 * smooth_factor,
            "ns": 0.0
        }
    else:
        return {
            "continuity": 0.5,
            "vof": 0.5,
            "ns": 0.1
        }
```

## 关键改进

### 1. Stage 1 → Stage 2 集成

Stage 2 使用 Stage 1 的接触角预测作为边界条件：

```python
class DataGenerator:
    def __init__(self, ...):
        self.contact_angle_predictor = HybridPredictor()
    
    def get_contact_angle(self, V: float, t: float) -> float:
        return self.contact_angle_predictor.predict(voltage=V, time=t)
```

### 2. 低电压约束

确保 V < 10V 时无开口：

```python
# 早期时间约束
t_early = torch.rand(n_early) * 0.002  # t < 2ms
target_phi_early = 0.5 * torch.ones_like(phi_early)
losses["early_time"] = F.mse_loss(phi_early, target_phi_early) * 500.0
```

### 3. 界面加密采样

在界面附近增加采样密度：

```python
# 界面权重
interface_weight = torch.exp(-20 * (phi - 0.5)**2)
```

## 数值稳定性

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 学习率调度

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6
)
```

### 损失归一化

```python
# 归一化残差
div_u_norm = div_u * self.L_char / self.U_char
```

## 训练监控

### 关键指标

1. **总损失**: 应稳定下降
2. **物理残差**: 各方程残差应减小
3. **开口率**: V=0V 时应为 0，V=30V 时应 > 50%

### 验证脚本

```bash
python validate_pinn_physics.py
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
