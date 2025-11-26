# 训练系统 API 参考

## 概述

EFD3D训练系统基于渐进式训练策略，通过统一的训练函数和辅助类实现复杂的物理约束神经网络训练。系统支持多阶段训练、自适应优化和高级损失稳定化功能。

## 核心训练函数

### progressive_training

主训练函数，实现多阶段渐进式训练。

#### 函数定义
```python
def progressive_training(config_path, resume_training=False, resume_checkpoint=None,
                       mixed_precision=False, model_init_seed=None,
                       use_efficient_architecture=True, model_compression_factor=1.0,
                       output_dir='./results'):
    """
    渐进式训练主函数 - 实现多阶段训练策略
    
    Args:
        config_path: 训练配置文件路径
        resume_training: 是否从检查点恢复训练
        resume_checkpoint: 恢复训练使用的检查点路径
        mixed_precision: 是否启用混合精度训练
        model_init_seed: 模型初始化随机种子
        use_efficient_architecture: 是否使用高效架构
        model_compression_factor: 模型压缩因子
        output_dir: 输出目录路径
        
    Returns:
        tuple: (训练完成的模型, 归一化器, 训练元数据)
    """
```

#### 参数说明
- `config_path` (str): JSON配置文件路径，包含训练参数和模型配置
- `resume_training` (bool): 是否从检查点恢复训练，默认为False
- `resume_checkpoint` (str): 恢复训练使用的检查点文件路径
- `mixed_precision` (bool): 是否启用混合精度训练，默认为False
- `model_init_seed` (int): 模型初始化随机种子，用于可重复性
- `use_efficient_architecture` (bool): 是否使用高效神经网络架构，默认为True
- `model_compression_factor` (float): 模型压缩因子，控制网络大小，默认为1.0
- `output_dir` (str): 训练结果输出目录，默认为'./results'

#### 返回值
- `model` (nn.Module): 训练完成的神经网络模型
- `normalizer` (object): 数据归一化器
- `metadata` (dict): 训练元数据，包含训练统计和配置信息

### unified_progressive_training

统一渐进式训练函数，整合多种训练模式。

#### 函数定义
```python
def unified_progressive_training(config_path, resume_training=False, resume_checkpoint=None,
                               mixed_precision=True, model_init_seed=None,
                               use_efficient_architecture=True, model_compression_factor=1.0,
                               output_dir='./results'):
    """
    统一渐进式训练函数 - 整合多种训练模式
    
    Args:
        config_path: 训练配置文件路径
        resume_training: 是否从检查点恢复训练
        resume_checkpoint: 恢复训练使用的检查点路径
        mixed_precision: 是否启用混合精度训练，默认为True
        model_init_seed: 模型初始化随机种子
        use_efficient_architecture: 是否使用高效架构
        model_compression_factor: 模型压缩因子
        output_dir: 输出目录路径
        
    Returns:
        tuple: (训练完成的模型, 归一化器, 训练元数据)
    """
```

## 辅助类

### LossStabilizer

高级损失稳定器，提供数值稳定性和自适应优化。

#### 类定义
```python
class LossStabilizer:
    """高级损失稳定器 - 提供数值稳定性和自适应优化"""
    
    def __init__(self, epsilon=1e-10, loss_type='mse', safe_clamp=True,
                 config_path=None, patience=5, reduction_factor=0.5):
        """
        初始化损失稳定器
        
        Args:
            epsilon: 数值稳定性参数，防止除零错误
            loss_type: 损失函数类型 ('mse', 'relative', 'huber', 'combined')
            safe_clamp: 是否启用安全裁剪，防止极端值
            config_path: 配置文件路径，可选的配置加载
            patience: 自适应稳定化的耐心参数
            reduction_factor: 损失减少因子
        """
```

#### 构造函数参数
- `epsilon` (float): 数值稳定性参数，默认为1e-10
- `loss_type` (str): 损失函数类型，支持'mse', 'relative', 'huber', 'combined'，默认为'mse'
- `safe_clamp` (bool): 是否启用安全裁剪，默认为True
- `config_path` (str, optional): 配置文件路径，用于加载额外配置
- `patience` (int): 自适应稳定化的耐心参数，默认为5
- `reduction_factor` (float): 损失减少因子，默认为0.5

#### 主要方法

##### `safe_mse_loss(pred, target, max_loss_value=1e6)`
安全的MSE损失计算，包含数值稳定性处理。

**参数:**
- `pred` (Tensor): 预测值张量
- `target` (Tensor): 目标值张量
- `max_loss_value` (float): 最大损失值限制，默认为1e6

**返回:**
- `Tensor`: 稳定后的MSE损失值

##### `relative_loss(pred, target, epsilon=None)`
相对损失计算，对数据量级不敏感。

**参数:**
- `pred` (Tensor): 预测值张量
- `target` (Tensor): 目标值张量
- `epsilon` (float, optional): 数值稳定性参数，默认为None（使用实例epsilon）

**返回:**
- `Tensor`: 相对损失值

##### `huber_loss(pred, target, delta=1.0)`
Huber损失，结合MSE和MAE的优点。

**参数:**
- `pred` (Tensor): 预测值张量
- `target` (Tensor): 目标值张量
- `delta` (float): Huber损失的分界点，默认为1.0

**返回:**
- `Tensor`: Huber损失值

##### `combined_loss(pred, target, mse_weight=0.5, relative_weight=0.5)`
组合损失函数，平衡MSE和相对误差。

**参数:**
- `pred` (Tensor): 预测值张量
- `target` (Tensor): 目标值张量
- `mse_weight` (float): MSE损失权重，默认为0.5
- `relative_weight` (float): 相对损失权重，默认为0.5

**返回:**
- `Tensor`: 组合损失值

## 使用示例

### 基础训练流程
```python
import torch
from ewp_pinn_optimized_train import progressive_training
from ewp_pinn_optimized_architecture import EfficientEWPINN

# 配置训练参数
config_path = 'config/training_config.json'
output_dir = './training_results'

# 执行渐进式训练
model, normalizer, metadata = progressive_training(
    config_path=config_path,
    mixed_precision=True,
    use_efficient_architecture=True,
    model_compression_factor=0.8,
    output_dir=output_dir
)

print(f"训练完成！模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"训练元数据: {metadata}")
```

### 高级训练配置
```python
# 使用统一训练函数进行高级训练
from run_enhanced_training import unified_progressive_training

# 高级训练配置
model, normalizer, metadata = unified_progressive_training(
    config_path='config/advanced_config.json',
    resume_training=True,
    resume_checkpoint='checkpoints/latest.pth',
    mixed_precision=True,
    model_init_seed=42,
    output_dir='./advanced_results'
)

print(f"高级训练完成！最佳验证损失: {metadata.get('best_val_loss', 'N/A')}")
```

### 损失稳定化示例
```python
import torch
from ewp_pinn_optimized_train import LossStabilizer

# 创建损失稳定器
stabilizer = LossStabilizer(
    epsilon=1e-8,
    loss_type='combined',
    safe_clamp=True,
    patience=10,
    reduction_factor=0.7
)

# 模拟训练数据
pred = torch.randn(32, 24)
target = torch.randn(32, 24)

# 计算稳定损失
stable_loss = stabilizer.combined_loss(pred, target, mse_weight=0.6, relative_weight=0.4)
print(f"稳定损失值: {stable_loss.item():.6f}")

# 使用Huber损失
huber_loss = stabilizer.huber_loss(pred, target, delta=0.5)
print(f"Huber损失值: {huber_loss.item():.6f}")
```

## 训练阶段配置

EFD3D支持多阶段训练策略，包括：

1. **预训练阶段**: 基础数据拟合，建立初步映射关系
2. **物理约束阶段**: 引入物理约束，确保物理一致性
3. **微调阶段**: 优化模型性能，提高预测精度

每个阶段可以配置不同的学习率、批次大小和物理约束权重。

## 最佳实践

1. **配置管理**: 使用JSON配置文件管理训练参数
2. **检查点保存**: 定期保存训练检查点，支持恢复训练
3. **混合精度**: 启用混合精度训练以提高训练效率
4. **模型压缩**: 根据硬件资源调整模型压缩因子
5. **损失监控**: 使用损失稳定器防止训练不稳定