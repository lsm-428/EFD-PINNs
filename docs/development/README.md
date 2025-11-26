# EFD3D 开发指南

## 开发环境设置

### 1. 开发依赖安装
```bash
# 基础开发环境
pip install torch numpy matplotlib scikit-learn pytest

# 开发工具
pip install black flake8 mypy pylint pre-commit
pip install jupyter notebook ipython

# 文档生成
pip install sphinx sphinx-rtd-theme myst-parser
```

### 2. 代码风格配置

#### .flake8 配置文件
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
```

#### pre-commit 配置
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## 项目结构说明

### 核心模块
```
EFD3D/
├── efd_pinns_train.py          # 主训练脚本
├── ewp_pinn_model.py           # 基础模型定义
├── ewp_pinn_optimized_architecture.py  # 优化架构
├── ewp_pinn_physics.py         # 物理约束层
├── ewp_pinn_optimizer.py       # 优化器管理
├── ewp_pinn_regularization.py  # 正则化
└── scripts/                    # 辅助脚本
```

### 配置文件
```
configs/
├── model_config.json           # 模型配置
├── training_config.json        # 训练配置
├── physics_config.json         # 物理约束配置
└── scenario_templates/         # 场景模板
```

## 开发流程

### 1. 功能开发

#### 添加新的物理约束
1. 在 `ewp_pinn_physics.py` 中扩展 `PINNConstraintLayer`
2. 实现新的约束计算方法
3. 更新配置系统支持新约束
4. 添加相应的测试用例

#### 实现新的训练策略
1. 继承 `MultiStageTrainer` 类
2. 重写训练循环逻辑
3. 更新配置解析逻辑
4. 添加性能监控指标

### 2. 测试开发

#### 单元测试
```python
import pytest
import torch
from ewp_pinn_model import EWPINN

class TestEWPINN:
    def test_forward_shape(self):
        model = EWPINN(input_dim=62, output_dim=24)
        x = torch.randn(10, 62)
        output = model(x)
        assert output['main_predictions'].shape == (10, 24)
```

#### 集成测试
```python
class TestTrainingIntegration:
    def test_end_to_end_training(self):
        # 完整的训练流程测试
        pass
```

### 3. 性能优化

#### 内存优化
- 使用梯度检查点（Gradient Checkpointing）
- 实现数据流式加载
- 优化张量操作

#### 计算优化
- 启用混合精度训练
- 使用CUDA内核优化
- 实现并行计算

## 代码规范

### 命名规范
- 类名：`CamelCase`
- 函数名：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有成员：`_private_member`

### 文档规范
- 所有公共API必须有docstring
- 使用Google风格文档字符串
- 包含类型注解

### 示例代码
```python
class OptimizedEWPINN(nn.Module):
    """优化的电润湿PINN模型。
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_layers: 隐藏层配置
        activation: 激活函数类型
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_layers: List[int], activation: str = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # ... 实现细节
```

## 调试技巧

### 1. 训练调试
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 梯度检查
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm = {param.grad.norm().item()}")
```

### 2. 内存调试
```python
import torch

def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### 3. 性能分析
```python
import cProfile

def profile_training():
    with cProfile.Profile() as pr:
        # 训练代码
        pass
    pr.print_stats()
```

## 版本控制

### 分支策略
- `main`: 稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 热修复分支

### 提交规范
```
feat: 添加新的物理约束类型
fix: 修复梯度爆炸问题
docs: 更新API文档
style: 代码格式调整
test: 添加单元测试
refactor: 重构训练循环
```

## 部署指南

### 1. 模型打包
```python
# 创建模型包
def create_model_package(model, config, training_data):
    package = {
        'model_state': model.state_dict(),
        'config': config,
        'training_history': training_data,
        'metadata': {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat()
        }
    }
    return package
```

### 2. 环境配置
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "efd_pinns_train.py"]
```

## 贡献指南

### 1. 问题报告
- 使用模板报告bug
- 提供复现步骤
- 包含环境信息

### 2. 功能请求
- 描述使用场景
- 提供实现建议
- 讨论API设计

### 3. 代码审查
- 遵循代码规范
- 确保测试覆盖
- 验证性能影响

这个开发指南为EFD3D项目的开发者提供了完整的开发流程和最佳实践。