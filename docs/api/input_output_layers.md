# 输入输出层 API

## EWPINNInputLayer 类

**文件**: `ewp_pinn_input_layer.py`

### 类定义
```python
class EWPINNInputLayer:
    """电润湿PINN输入层 - 负责特征提取和预处理"""
```

### 构造函数
```python
def __init__(self):
    """初始化输入层配置"""
```

### 主要方法

#### set_implementation_stage
```python
def set_implementation_stage(self, stage):
    """设置实现阶段"""
```

**参数**:
- `stage` (int): 实现阶段 (1-基础, 2-扩展, 3-完整)

**说明**:
- 阶段1: 基础物理特征 (32维)
- 阶段2: 扩展特征 (47维)  
- 阶段3: 完整特征 (62维)

#### create_input_vector
```python
def create_input_vector(self, physical_params):
    """从物理参数创建输入向量"""
```

**参数**:
- `physical_params` (dict): 物理参数字典

**返回值**:
- `torch.Tensor`: 归一化后的输入向量

#### validate_input
```python
def validate_input(self, input_vector):
    """验证输入向量有效性"""
```

**参数**:
- `input_vector` (torch.Tensor): 输入向量

**返回值**:
- `bool`: 验证结果

#### get_feature_names
```python
def get_feature_names(self):
    """获取当前阶段的特征名称列表"""
```

**返回值**:
- `list`: 特征名称列表

#### normalize_features
```python
def normalize_features(self, features):
    """特征归一化处理"""
```

## EWPINNOutputLayer 类

**文件**: `ewp_pinn_output_layer.py`

### 类定义
```python
class EWPINNOutputLayer:
    """电润湿PINN输出层 - 负责结果后处理和验证"""
```

### 构造函数
```python
def __init__(self):
    """初始化输出层配置"""
```

### 主要方法

#### set_implementation_stage
```python
def set_implementation_stage(self, stage):
    """设置实现阶段"""
```

**参数**:
- `stage` (int): 实现阶段 (1-基础, 2-扩展, 3-完整)

#### create_output_dict
```python
def create_output_dict(self, model_output):
    """将模型输出转换为字典格式"""
```

**参数**:
- `model_output` (torch.Tensor): 模型输出张量

**返回值**:
- `dict`: 物理量字典

#### validate_physical_constraints
```python
def validate_physical_constraints(self, output_dict):
    """验证物理约束条件"""
```

**参数**:
- `output_dict` (dict): 输出物理量字典

**返回值**:
- `dict`: 约束验证结果

#### denormalize_output
```python
def denormalize_output(self, normalized_output):
    """反归一化输出结果"""
```

**参数**:
- `normalized_output` (torch.Tensor): 归一化输出

**返回值**:
- `torch.Tensor`: 反归一化后的物理量

#### get_output_names
```python
def get_output_names(self):
    """获取输出物理量名称列表"""
```

**返回值**:
- `list`: 物理量名称列表

## 使用示例

### 基础用法
```python
import torch
from ewp_pinn_input_layer import EWPINNInputLayer
from ewp_pinn_output_layer import EWPINNOutputLayer

# 创建输入输出层实例
input_layer = EWPINNInputLayer()
output_layer = EWPINNOutputLayer()

# 设置实现阶段
input_layer.set_implementation_stage(3)  # 完整特征
output_layer.set_implementation_stage(3)  # 完整输出

# 创建输入向量
physical_params = {
    'velocity_x': 1.0,
    'velocity_y': 0.5,
    'pressure': 101325.0,
    'temperature': 300.0,
    # ... 更多物理参数
}

input_vector = input_layer.create_input_vector(physical_params)
print(f"输入向量形状: {input_vector.shape}")  # torch.Size([62])

# 验证输入
is_valid = input_layer.validate_input(input_vector)
print(f"输入验证: {is_valid}")

# 获取特征名称
feature_names = input_layer.get_feature_names()
print(f"特征数量: {len(feature_names)}")
```

### 输出处理示例
```python
# 模拟模型输出
model_output = torch.randn(24)  # 24维输出

# 转换为物理量字典
output_dict = output_layer.create_output_dict(model_output)
print(f"输出物理量: {list(output_dict.keys())}")

# 验证物理约束
constraint_results = output_layer.validate_physical_constraints(output_dict)
print(f"约束验证: {constraint_results}")

# 反归一化
physical_output = output_layer.denormalize_output(model_output)
print(f"反归一化形状: {physical_output.shape}")

# 获取输出名称
output_names = output_layer.get_output_names()
print(f"输出物理量数量: {len(output_names)}")
```

### 批量处理示例
```python
# 批量输入处理
batch_size = 32
batch_inputs = []

for i in range(batch_size):
    params = {
        'velocity_x': torch.randn(1).item(),
        'velocity_y': torch.randn(1).item(),
        # ... 其他参数
    }
    input_vec = input_layer.create_input_vector(params)
    batch_inputs.append(input_vec)

batch_tensor = torch.stack(batch_inputs)
print(f"批量输入形状: {batch_tensor.shape}")  # torch.Size([32, 62])

# 批量输出处理
batch_output = torch.randn(batch_size, 24)
output_dicts = []

for i in range(batch_size):
    output_dict = output_layer.create_output_dict(batch_output[i])
    output_dicts.append(output_dict)
```

## 特征说明

### 输入特征 (阶段3 - 62维)
- **基础流体特征**: 速度、压力、温度等 (16维)
- **电润湿特征**: 电场强度、介电常数等 (18维)
- **界面特征**: 接触角、表面张力等 (15维)
- **材料特征**: 密度、粘度等 (13维)

### 输出物理量 (24维)
- **流场变量**: u, v, w, p, T (5维)
- **电场变量**: Ex, Ey, Ez, φ (4维)
- **界面变量**: 界面形状、接触线位置等 (8维)
- **辅助变量**: 残差、约束条件等 (7维)

## 最佳实践

1. **阶段选择**: 根据问题复杂度选择合适的实现阶段
2. **特征工程**: 确保输入参数在合理范围内
3. **验证检查**: 始终验证输入输出数据的有效性
4. **批量处理**: 使用批量处理提高效率
5. **内存管理**: 及时释放不需要的张量

## 注意事项

- 输入参数需要符合物理约束范围
- 输出结果需要经过物理约束验证
- 不同阶段的特征维度不同，注意兼容性
- 归一化范围基于物理量的典型值