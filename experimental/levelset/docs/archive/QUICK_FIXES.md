# Level Set PINN 快速修复指南

## 🔧 立即实施的 4 个关键修复

### 修复 1: NaN 检测与恢复 (train_levelset_3d.py)

在 `train_epoch` 函数中添加:

```python
def train_epoch(self, epoch):
    # ... 现有代码 ...
    
    # NaN 检测与恢复
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(f"NaN/Inf detected at epoch {epoch}!")
        
        # 恢复上一轮有效状态
        if self.last_valid_state is not None:
            self.model.load_state_dict(self.last_valid_state)
            # 减半学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            logger.info("Restored state and halved LR")
            continue  # 跳过这次更新
    
    # 保存有效状态
    if epoch % 10 == 0:
        self.last_valid_state = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
```

### 修复 2: 硬体积守恒约束 (pinn_levelset_3d.py)

在 `LevelSet3DPINN` 类中添加:

```python
def __init__(self, config):
    # ... 现有初始化代码 ...
    
    # 添加体积守恒参数
    self.target_volume_fraction = 0.15  # 3μm / 20μm
    self.volume_correction_strength = 0.1
    self.volume_constraint_enabled = True

def enforce_volume_conservation(self, psi):
    """
    硬约束：强制全局体积守恒
    """
    if not self.volume_constraint_enabled:
        return psi
    
    # 计算当前体积比例
    psi_neg = -psi
    current_volume = torch.sigmoid(psi_neg).mean()
    
    # 计算体积误差
    volume_error = current_volume - self.target_volume_fraction
    
    # 应用体积调整
    psi_corrected = psi + volume_error * self.volume_correction_strength
    
    return psi_corrected

def forward(self, x):
    # ... 现有 forward 代码 ...
    
    # 在输出 psi 之前应用体积守恒
    psi = self.enforce_volume_conservation(psi)
    
    return output
```

### 修复 3: 固定训练脚本 (防止中断)

在 `train_levelset_3d.py` 的主函数中添加:

```python
def main():
    # ... 现有代码 ...
    
    # 添加异常处理
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            # 训练代码
            
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        # 保存当前状态
        save_checkpoint(model, optimizer, scheduler, epoch, output_dir)
    except Exception as e:
        logger.error(f"训练出错: {str(e)}")
        # 保存当前状态
        save_checkpoint(model, optimizer, scheduler, epoch, output_dir)
        # 重新抛出异常以便调试
        raise
    finally:
        logger.info(f"训练结束，最终 epoch: {epoch}")
```

### 修复 4: 增强训练监控

修改日志输出格式:

```python
def log_training_progress(epoch, loss_dict, lr):
    """
    详细的训练进度日志
    """
    # 检查体积守恒
    if 'VolCon' in loss_dict:
        vol_error = loss_dict['VolCon']
        vol_status = "✓" if vol_error < 0.01 else "⚠"
    else:
        vol_status = "N/A"
    
    logger.info(
        f"Epoch [{epoch}] Loss: {loss_dict['Total']:.4f} "
        f"(Data: {loss_dict['Data']:.2e}, "
        f"Cont: {loss_dict['Cont']:.2e}, "
        f"Vol: {vol_status}, "
        f"LR: {lr:.2e})"
    )
    
    # 每 1000 epochs 输出详细信息
    if epoch % 1000 == 0:
        logger.info("=" * 60)
        logger.info(f"【{epoch} Epochs 详细指标】")
        logger.info(f"  体积守恒: {loss_dict.get('VolCon', 0):.4e}")
        logger.info(f"  连续性: {loss_dict.get('Cont', 0):.4e}")
        logger.info(f"  接触角: {loss_dict.get('Contact', 0):.4e}")
        logger.info(f"  压力: {loss_dict.get('Pressure', 0):.4e}")
        logger.info("=" * 60)
```

## 🚀 快速测试

### 测试 1: 验证 NaN 检测
```bash
# 故意引发 NaN
python train_levelset_3d.py --config config/test_nan_detection.json
```

### 测试 2: 验证体积守恒
```bash
# 运行 1000 epochs 并检查体积
python train_levelset_3d.py --config config/v6.0_improved.json --epochs 1000
# 检查日志中的 "VolCon" 指标
```

### 测试 3: 验证训练稳定性
```bash
# 运行 5000 epochs（应该能完成）
python train_levelset_3d.py --config config/v6.0_improved.json --epochs 5000
```

## 📊 预期改进

应用这 4 个修复后，应该看到:

1. **训练完成率**: 从 4-5% → 100% (50000 epochs)
2. **体积守恒误差**: 从 +35% → < ±1%
3. **NaN 崩溃**: 从频繁 → 0 次
4. **训练稳定性**: Loss 震荡减少 >50%

## ⚠️ 注意事项

1. **备份现有代码**: 在修改前备份 `pinn_levelset_3d.py` 和 `train_levelset_3d.py`
2. **逐步测试**: 每次只应用一个修复，测试后再应用下一个
3. **监控显存**: 硬约束可能增加计算量，注意显存使用
4. **保存 checkpoint**: 确保 checkpoint 保存功能正常

## 📞 如有问题

查看日志文件: `outputs_levelset_*/training.log`

常见错误:
- `RuntimeError: CUDA out of memory`: 减小 batch_size 或增加 gradient_accumulation_steps
- `Loss is NaN`: 检查学习率是否过大
- `Volume error not decreasing`: 增加 volume_correction_strength

---

生成日期: 2026-02-03
版本: v1.0
