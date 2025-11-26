# 示例和最佳实践

## 基础训练示例

### 单GPU训练
```python
import torch
import torch.nn as nn
from ewp_pinn_optimized_architecture import EfficientEWPINN
from ewp_pinn_model import PhysicsConstraints
from efd_pinns_train import MultiStageTrainer

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型
model = EfficientEWPINN(
    input_dim=62,
    output_dim=24,
    hidden_layers=[256, 128, 64],
    device=device
).to(device)

# 创建物理约束
physics_constraints = PhysicsConstraints(device=device)

# 训练配置
config = {
    'model': {
        'input_dim': 62,
        'output_dim': 24,
        'hidden_layers': [256, 128, 64]
    },
    'training': {
        'epochs': 10000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'stages': {
            'pretrain': {'epochs': 1000, 'physics_weight': 0.1},
            'physics': {'epochs': 2000, 'physics_weight': 0.5},
            'finetune': {'epochs': 1000, 'physics_weight': 1.0}
        }
    }
}

# 创建训练器
trainer = MultiStageTrainer(args=None, config=config, device=device)

# 开始训练
print("开始训练...")
trainer.run()
print("训练完成!")
```

### 推理示例
```python
import torch
from ewp_pinn_optimized_architecture import EfficientEWPINN
from ewp_pinn_input_layer import EWPINNInputLayer
from ewp_pinn_output_layer import EWPINNOutputLayer

# 加载训练好的模型
model = EfficientEWPINN(input_dim=62, output_dim=24, device='cuda')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 创建输入输出层
input_layer = EWPINNInputLayer(device='cuda')
output_layer = EWPINNOutputLayer(device='cuda')

# 准备输入数据
input_data = torch.randn(10, 62).to('cuda')  # 10个样本

# 推理
with torch.no_grad():
    predictions = model(input_data)
    
# 后处理输出
processed_output = output_layer.post_process(predictions)

print(f"输入形状: {input_data.shape}")
print(f"预测形状: {predictions.shape}")
print(f"处理后输出形状: {processed_output.shape}")

# 输出物理量解释
feature_names = output_layer.get_feature_names()
for i, name in enumerate(feature_names):
    print(f"{name}: 均值={processed_output[:, i].mean():.4f}, 标准差={processed_output[:, i].std():.4f}")
```

## 高级用法示例

### 自定义物理约束
```python
import torch
from ewp_pinn_model import PhysicsConstraints

class CustomPhysicsConstraints(PhysicsConstraints):
    """自定义物理约束类"""
    
    def __init__(self, device='cuda', custom_params=None):
        super().__init__(device=device)
        self.custom_params = custom_params or {}
        
    def compute_custom_constraint(self, x, predictions):
        """计算自定义约束"""
        # 提取预测的物理量
        u, v, w, p = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        
        # 自定义约束逻辑
        constraint = torch.sqrt(u**2 + v**2 + w**2) - self.custom_params.get('max_velocity', 10.0)
        
        return torch.relu(constraint)  # 只关心违反约束的部分
    
    def compute_total_residual(self, x, predictions):
        """重写总残差计算，包含自定义约束"""
        # 基础物理约束
        base_residual = super().compute_total_residual(x, predictions)
        
        # 自定义约束
        custom_residual = self.compute_custom_constraint(x, predictions)
        
        # 组合约束
        total_residual = base_residual + 0.1 * custom_residual  # 自定义约束权重较低
        
        return total_residual

# 使用自定义约束
custom_constraints = CustomPhysicsConstraints(
    device='cuda',
    custom_params={'max_velocity': 5.0}
)

# 在训练中使用
config = {
    'physics_constraints': custom_constraints,
    # ... 其他配置
}
```

### 多GPU训练
```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ewp_pinn_optimized_architecture import EfficientEWPINN

def setup_ddp():
    """设置分布式训练环境"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()

# 设置分布式训练
setup_ddp()

# 创建模型
model = EfficientEWPINN(input_dim=62, output_dim=24, device='cuda')

# 包装为DDP模型
model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

# 训练配置（分布式版本）
config = {
    'training': {
        'epochs': 20000,
        'batch_size': 64,
        'learning_rate': 0.001,
        'distributed': True,
        'stages': {
            'pretrain': {'epochs': 2000, 'physics_weight': 0.1},
            'physics': {'epochs': 5000, 'physics_weight': 0.5},
            'finetune': {'epochs': 3000, 'physics_weight': 1.0}
        }
    }
}

# 分布式训练循环
for epoch in range(config['training']['epochs']):
    # 训练步骤
    for batch in dataloader:
        # 前向传播
        predictions = model(batch['input'])
        
        # 计算损失
        loss = compute_loss(predictions, batch['target'])
        
        # 反向传播
        loss.backward()
        
        # 梯度同步（DDP自动处理）
        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()
    
    # 验证步骤
    if epoch % 100 == 0:
        validate_model(model, val_dataloader)

# 清理
torch.save(model.module.state_dict(), 'ddp_model.pth')  # 保存主模型
cleanup_ddp()
```

### 渐进式训练策略
```python
from ewp_pinn_optimized_train import progressive_training

# 渐进式训练配置
progressive_config = {
    'base_config': {
        'input_dim': 62,
        'output_dim': 24,
        'hidden_layers': [256, 128, 64],
        'learning_rate': 0.001
    },
    'progressive_stages': [
        {
            'name': 'stage1_basic',
            'epochs': 1000,
            'complexity': 'low',
            'data_subset': 0.3
        },
        {
            'name': 'stage2_medium',
            'epochs': 2000,
            'complexity': 'medium',
            'data_subset': 0.6
        },
        {
            'name': 'stage3_full',
            'epochs': 3000,
            'complexity': 'high',
            'data_subset': 1.0
        }
    ]
}

# 执行渐进式训练
model, training_history = progressive_training(progressive_config)

print("渐进式训练完成!")
print(f"最终模型参数数量: {sum(p.numel() for p in model.parameters())}")
```

## 最佳实践

### 模型选择指南

#### 小型数据集（<10,000样本）
```python
# 推荐配置
model_config = {
    'input_dim': 62,
    'output_dim': 24,
    'hidden_layers': [128, 64],  # 较浅的网络
    'use_residual': False,       # 关闭残差连接
    'use_attention': False       # 关闭注意力机制
}

training_config = {
    'epochs': 5000,
    'batch_size': 16,           # 较小的批次
    'learning_rate': 0.0005,     # 较低的学习率
    'physics_weight': 0.3        # 较低的物理权重
}
```

#### 中型数据集（10,000-100,000样本）
```python
# 推荐配置
model_config = {
    'input_dim': 62,
    'output_dim': 24,
    'hidden_layers': [256, 128, 64],  # 中等深度
    'use_residual': True,             # 启用残差连接
    'use_attention': True             # 启用注意力机制
}

training_config = {
    'epochs': 10000,
    'batch_size': 32,
    'learning_rate': 0.001,
    'physics_weight': 0.5
}
```

#### 大型数据集（>100,000样本）
```python
# 推荐配置
model_config = {
    'input_dim': 62,
    'output_dim': 24,
    'hidden_layers': [512, 256, 128, 64],  # 较深网络
    'use_residual': True,                  # 启用残差连接
    'use_attention': True,                 # 启用注意力机制
    'dropout_rate': 0.1                    # 添加dropout
}

training_config = {
    'epochs': 20000,
    'batch_size': 64,
    'learning_rate': 0.001,
    'physics_weight': 0.8,
    'distributed': True                    # 使用分布式训练
}
```

### 训练策略优化

#### 学习率调度
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 余弦退火调度器
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,        # 初始周期
    T_mult=2,        # 周期倍增因子
    eta_min=1e-6     # 最小学习率
)

# 在训练循环中使用
for epoch in range(num_epochs):
    # 训练步骤
    train_loss = train_epoch(model, train_loader)
    
    # 更新学习率
    scheduler.step()
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}: Loss = {train_loss:.4f}, LR = {current_lr:.6f}")
```

#### 早停机制
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

# 使用早停
early_stopping = EarlyStopping(patience=20, min_delta=0.001)

for epoch in range(num_epochs):
    # 训练和验证
    train_loss = train_epoch(model, train_loader)
    val_loss = validate_model(model, val_loader)
    
    # 检查早停
    if early_stopping(val_loss):
        print(f"早停在epoch {epoch}")
        break
```

### 数据预处理最佳实践

#### 特征标准化
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
    
    def fit(self, X, y):
        """拟合数据"""
        self.input_scaler.fit(X)
        self.output_scaler.fit(y)
        return self
    
    def transform(self, X, y=None):
        """转换数据"""
        X_scaled = self.input_scaler.transform(X)
        
        if y is not None:
            y_scaled = self.output_scaler.transform(y)
            return X_scaled, y_scaled
        return X_scaled
    
    def inverse_transform_output(self, y_scaled):
        """反转换输出"""
        return self.output_scaler.inverse_transform(y_scaled)

# 使用示例
preprocessor = DataPreprocessor()
X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)

# 模型训练使用标准化数据
model.fit(X_train_scaled, y_train_scaled)

# 预测时反标准化
y_pred_scaled = model.predict(X_test_scaled)
y_pred = preprocessor.inverse_transform_output(y_pred_scaled)
```

#### 数据增强策略
```python
from efd_pinns_train import EnhancedDataAugmenter

# 配置数据增强
augmenter_config = {
    'noise_level': 0.01,           # 噪声水平
    'scale_range': (0.95, 1.05),   # 缩放范围
    'rotation_range': (-2, 2),     # 旋转角度范围
    'flip_probability': 0.3        # 翻转概率
}

augmenter = EnhancedDataAugmenter(augmenter_config)

# 在数据加载器中使用
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augmenter=None):
        self.X = X
        self.y = y
        self.augmenter = augmenter
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augmenter and torch.rand(1) < 0.5:  # 50%概率增强
            x, y = self.augmenter.augment(x.unsqueeze(0), y.unsqueeze(0))
            x, y = x.squeeze(0), y.squeeze(0)
        
        return x, y
```

### 性能优化技巧

#### 内存优化
```python
# 使用梯度检查点（适用于大模型）
model = EfficientEWPINN(
    input_dim=62,
    output_dim=24,
    hidden_layers=[512, 256, 128, 64],
    use_gradient_checkpointing=True  # 启用梯度检查点
)

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        predictions = model(batch['input'])
        loss = compute_loss(predictions, batch['target'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 推理优化
```python
# 模型量化（减小模型大小，提高推理速度）
model = EfficientEWPINN(input_dim=62, output_dim=24, device='cpu')
model.load_state_dict(torch.load('best_model.pth'))

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化模型
torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')

# 加载量化模型进行推理
quantized_model = torch.jit.load('quantized_model.pt')
quantized_model.eval()

# 推理（速度更快，内存占用更少）
with torch.no_grad():
    predictions = quantized_model(input_data)
```

### 调试和监控

#### 训练监控
```python
import wandb  # 权重和偏置监控

def setup_monitoring():
    """设置训练监控"""
    wandb.init(project="efd3d-pinn")
    
    # 配置监控指标
    wandb.config.update({
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10000
    })

def log_training_metrics(epoch, train_loss, val_loss, learning_rate):
    """记录训练指标"""
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate
    })

# 在训练循环中使用
setup_monitoring()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate_model(model, val_loader)
    
    log_training_metrics(epoch, train_loss, val_loss, scheduler.get_last_lr()[0])
```

#### 梯度监控
```python
def monitor_gradients(model, epoch):
    """监控梯度统计信息"""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            
            # 记录每个参数的梯度
            wandb.log({f'grad_norm/{name}': grad_norm})
    
    # 记录总体统计
    if grad_norms:
        wandb.log({
            'grad_norm/mean': np.mean(grad_norms),
            'grad_norm/max': np.max(grad_norms),
            'grad_norm/min': np.min(grad_norms)
        })

# 在训练循环中调用
for epoch in range(num_epochs):
    # 训练步骤
    train_loss = train_epoch(model, train_loader)
    
    # 监控梯度
    monitor_gradients(model, epoch)
```

## 常见问题解决方案

### 训练不稳定
**症状**: 损失值剧烈波动或变为NaN

**解决方案**:
1. 降低学习率
2. 启用梯度裁剪
3. 检查数据预处理
4. 使用更稳定的激活函数（如GELU）

```python
# 稳定训练配置
stable_config = {
    'learning_rate': 0.0001,           # 降低学习率
    'gradient_clipping': True,         # 启用梯度裁剪
    'max_grad_norm': 1.0,             # 最大梯度范数
    'activation': 'GELU'              # 使用GELU激活函数
}
```

### 过拟合
**症状**: 训练损失持续下降，但验证损失开始上升

**解决方案**:
1. 增加数据增强
2. 使用早停机制
3. 添加正则化（L2权重衰减）
4. 减少模型复杂度

```python
# 防止过拟合配置
anti_overfit_config = {
    'data_augmentation': True,        # 启用数据增强
    'early_stopping': True,           # 启用早停
    'weight_decay': 1e-4,             # L2正则化
    'dropout_rate': 0.1               # 添加dropout
}
```

### 收敛缓慢
**症状**: 损失值下降非常缓慢

**解决方案**:
1. 增加学习率
2. 检查数据预处理
3. 使用更好的优化器（如AdamW）
4. 调整批次大小

```python
# 加速收敛配置
fast_converge_config = {
    'learning_rate': 0.001,           # 增加学习率
    'optimizer': 'adamw',             # 使用AdamW
    'batch_size': 64,                 # 增加批次大小
    'warmup_epochs': 100              # 学习率预热
}
```

### 内存不足
**症状**: CUDA内存错误

**解决方案**:
1. 减少批次大小
2. 使用梯度检查点
3. 启用混合精度训练
4. 使用CPU进行部分计算

```python
# 内存优化配置
memory_optimized_config = {
    'batch_size': 16,                 # 减小批次大小
    'gradient_checkpointing': True,   # 启用梯度检查点
    'mixed_precision': True,          # 混合精度训练
    'device': 'cuda'                  # 仅在必要时使用GPU
}
```

通过遵循这些最佳实践和示例，您可以更有效地使用EFD3D框架，并解决常见的训练和部署问题。