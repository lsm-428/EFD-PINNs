# 核心模型 API

## EfficientEWPINN 类

**文件**: `ewp_pinn_optimized_architecture.py`

### 类定义
```python
class EfficientEWPINN(nn.Module):
    """高效EWPINN模型 - 增强型神经网络架构"""
```

### 构造函数
```python
def __init__(self, input_dim=62, output_dim=24, hidden_layers=None, 
             dropout_rate=0.1, activation='ReLU', batch_norm=True, 
             use_residual=True, use_attention=True, config_path=None, 
             device='cpu', compression_factor=1.0, gradient_checkpointing=False):
```

**参数说明**:
- `input_dim` (int, 默认62): 输入特征维度
- `output_dim` (int, 默认24): 输出预测维度  
- `hidden_layers` (list, 可选): 隐藏层配置列表，如 `[256, 128, 64]`
- `dropout_rate` (float, 默认0.1): Dropout比率
- `activation` (str, 默认'ReLU'): 激活函数类型，支持 'ReLU', 'GELU', 'SiLU'
- `batch_norm` (bool, 默认True): 是否使用批量归一化
- `use_residual` (bool, 默认True): 是否使用残差连接
- `use_attention` (bool, 默认True): 是否使用注意力机制
- `config_path` (str, 可选): 配置文件路径
- `device` (str, 默认'cpu'): 运行设备
- `compression_factor` (float, 默认1.0): 网络压缩因子
- `gradient_checkpointing` (bool, 默认False): 梯度检查点

### 主要方法

#### forward
```python
def forward(self, x):
    """前向传播"""
```

**参数**:
- `x` (torch.Tensor): 输入张量，形状为 `(batch_size, input_dim)`

**返回值**:
- `torch.Tensor`: 预测结果，形状为 `(batch_size, output_dim)`

#### save_model
```python
def save_model(self, filepath):
    """保存模型和配置"""
```

**参数**:
- `filepath` (str): 保存路径

#### load_model
```python
def load_model(self, filepath):
    """加载模型和配置"""
```

**参数**:
- `filepath` (str): 模型文件路径

## OptimizedEWPINN 类

**文件**: `efd_pinns_train.py`

### 类定义
```python
class OptimizedEWPINN(nn.Module):
    """优化版EWPINN模型 - 简化架构"""
```

### 构造函数
```python
def __init__(self, input_dim, hidden_dims, output_dim, 
             activation='relu', config=None):
```

**参数说明**:
- `input_dim` (int): 输入维度
- `hidden_dims` (list): 隐藏层维度列表，如 `[128, 64, 32]`
- `output_dim` (int): 输出维度
- `activation` (str, 默认'relu'): 激活函数类型
- `config` (dict, 可选): 配置字典

### 主要方法

#### forward
```python
def forward(self, x):
    """前向传播"""
```

**参数**:
- `x` (torch.Tensor): 输入张量

**返回值**:
- `torch.Tensor`: 预测结果

## EfficientResidualLayer 类

**文件**: `ewp_pinn_optimized_architecture.py`

### 类定义
```python
class EfficientResidualLayer(nn.Module):
    """高效残差层 - 支持批量归一化和激活函数"""
```

### 构造函数
```python
def __init__(self, input_dim, output_dim, activation='ReLU', 
             batch_norm=True, dropout_rate=0.1):
```

## AttentionMechanism 类

**文件**: `ewp_pinn_optimized_architecture.py`

### 类定义
```python
class AttentionMechanism(nn.Module):
    """通道注意力机制 - 增强特征表示"""
```

### 构造函数
```python
def __init__(self, channels, reduction=16):
```

## 使用示例

### 基础用法
```python
import torch
from ewp_pinn_optimized_architecture import EfficientEWPINN

# 创建模型实例
model = EfficientEWPINN(
    input_dim=62,
    output_dim=24,
    hidden_layers=[256, 128, 64],
    activation='GELU',
    use_residual=True,
    use_attention=True,
    device='cuda'
)

# 前向传播
input_data = torch.randn(32, 62).cuda()
output = model(input_data)
print(f"输出形状: {output.shape}")  # torch.Size([32, 24])
```

### 高级配置
```python
# 使用配置文件
model = EfficientEWPINN(
    config_path='model_config.json',
    device='cuda',
    compression_factor=0.8,
    gradient_checkpointing=True
)

# 保存和加载模型
model.save_model('best_model.pth')
model.load_model('best_model.pth')
```

### 优化版本使用
```python
from efd_pinns_train import OptimizedEWPINN

# 创建简化模型
model = OptimizedEWPINN(
    input_dim=62,
    hidden_dims=[128, 64, 32],
    output_dim=24,
    activation='relu'
)
```

## 性能优化建议

1. **内存优化**: 启用 `gradient_checkpointing=True` 和 `compression_factor=0.8`
2. **训练加速**: 使用混合精度训练和CUDA设备
3. **模型选择**: 复杂问题用EfficientEWPINN，简单问题用OptimizedEWPINN
4. **架构调整**: 根据问题复杂度调整隐藏层配置

## 注意事项

- 确保输入维度与模型配置一致
- 使用合适的激活函数避免梯度消失
- 残差连接有助于深层网络训练
- 注意力机制可提升特征表示能力