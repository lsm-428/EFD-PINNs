# 贡献指南 (Contributing Guide)

感谢您对 EFD-PINNs 项目的关注！本文档将指导您如何为项目做出贡献。

---

## 🚀 快速开始

### 1. Fork 和克隆

```bash
# Fork 项目到您的 GitHub 账户
# 然后克隆到本地
git clone https://github.com/YOUR_USERNAME/EFD3D.git
cd EFD3D
```

### 2. 设置开发环境

```bash
# 创建 conda 环境
conda create -n efd python=3.12
conda activate efd

# 安装依赖
pip install -r requirements.txt
```

### 3. 创建分支

```bash
git checkout -b feature/your-feature-name
```

---

## 📁 项目结构

```
EFD3D/
├── train_contact_angle.py         # Stage 1 训练入口
├── train_two_phase.py             # Stage 2 训练入口
├── validate_pinn_physics.py       # 物理验证脚本
├── visualize_pinn_results.py      # 结果可视化
│
├── src/                           # 源代码目录
│   ├── models/                   # 神经网络模型
│   │   └── pinn_two_phase.py    # 两相流 PINN 模型
│   ├── predictors/               # 预测器
│   │   ├── hybrid_predictor.py  # Stage 1 混合预测器
│   │   └── pinn_aperture.py     # Stage 2 开口率预测器
│   ├── physics/                  # 物理约束模块
│   ├── training/                 # 训练系统
│   ├── utils/                    # 工具函数
│   └── visualization/            # 可视化工具
│
├── config/                        # 配置文件
│   ├── stage1_config.json
│   ├── stage2_optimized.json
│   └── ...
│
├── tests/                         # 测试文件
│   ├── test_config_loading.py
│   └── test_model_dimensions.py
│
├── docs/                          # 文档
│   ├── api/                      # API 文档
│   ├── guides/                   # 使用指南
│   └── specs/                    # 技术规格
│
└── outputs_pinn_*/                # 训练输出目录
```

---

## 📝 代码规范

### Python 代码风格

- 遵循 PEP 8 规范
- 使用 4 空格缩进
- 最大行长度 100 字符
- 使用类型注解

```python
def calculate_contact_angle(
    voltage: float,
    theta0: float = 120.0,
    epsilon_r: float = 4.0
) -> float:
    """
    计算接触角。
    
    Args:
        voltage: 施加电压 (V)
        theta0: 初始接触角 (度)
        epsilon_r: 相对介电常数
        
    Returns:
        接触角 (度)
    """
    pass
```

### 导入规范

```python
# 标准库
import json
from pathlib import Path

# 第三方库
import torch
import numpy as np

# 项目模块
from src.models.pinn_two_phase import TwoPhaseFlowPINN
from src.predictors.hybrid_predictor import HybridPredictor
from src.predictors.pinn_aperture import PINNAperturePredictor
```

---

## ✅ 测试要求

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_config_loading.py -v

# 运行属性测试
python -m pytest tests/test_model_dimensions.py -v
```

### 编写测试

```python
import pytest
from hypothesis import given, strategies as st
from src.predictors.hybrid_predictor import HybridPredictor

class TestHybridPredictor:
    def test_basic_prediction(self):
        """测试基本预测功能"""
        predictor = HybridPredictor()
        result = predictor.predict(voltage=15.0, t=0.01)
        assert 60 <= result <= 130
    
    @given(st.floats(min_value=0, max_value=30))
    def test_voltage_range(self, voltage):
        """属性测试：电压范围"""
        predictor = HybridPredictor()
        angle = predictor.predict(voltage=voltage, t=0.01)
        assert 60 <= angle <= 130
```

---

## 📊 提交规范

### Commit 消息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 示例

```
feat(pinn): 添加自适应损失权重

- 实现 GradNorm 算法自动平衡损失项
- 添加权重历史记录功能
- 更新训练日志输出

Closes #123
```

---

## 🔄 Pull Request 流程

### 1. 确保测试通过

```bash
python -m pytest tests/ -v
```

### 2. 更新文档

如果添加了新功能，请更新相关文档：
- `README.md`
- `USAGE_GUIDE.md`
- `docs/CHANGELOG.md`

### 3. 创建 PR

- 标题清晰描述更改内容
- 描述中说明更改原因和影响
- 关联相关 Issue

### 4. 代码审查

- 等待维护者审查
- 根据反馈进行修改
- 合并后删除分支

---

## 🐛 报告 Bug

### Bug 报告模板

```markdown
## 问题描述
简要描述遇到的问题

## 复现步骤
1. 运行命令 `python train_two_phase.py --config config/xxx.json`
2. 等待训练开始
3. 出现错误

## 预期行为
描述您期望发生的情况

## 实际行为
描述实际发生的情况

## 错误信息
```
粘贴完整的错误堆栈
```

## 环境信息
- Python 版本: 3.12
- PyTorch 版本: 2.0
- 操作系统: Ubuntu 22.04
- GPU: NVIDIA RTX 3090
```

---

## 💡 功能建议

### 功能建议模板

```markdown
## 功能描述
描述您希望添加的功能

## 使用场景
说明这个功能的应用场景

## 建议实现方式
如果有想法，可以描述实现思路

## 相关资料
提供相关论文、文档链接
```

---

## 📚 资源

- [项目文档](README.md)
- [API 参考](api/README.md)
- [训练历史](../TRAINING_HISTORY.md)
- [配置模板](CONFIG_TEMPLATE.md)

---

## 📧 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**感谢您的贡献！** 🎉

**最后更新**: 2025-12-08
