# 实验管理模块 (experiment_management)

EFD3D 实验管理系统，提供完整的实验跟踪、对比分析和报告生成功能。

## 📁 文件夹结构

```
experiment_management/
├── __init__.py          # 模块入口文件
├── experiment_manager.py    # 实验管理器 - 核心功能
├── experiment_comparison.py # 实验对比器 - 多实验分析
├── experiment_reporter.py    # 报告生成器 - 可视化报告
└── README.md            # 本文件
```

## 🚀 核心功能

### 1. 实验管理器 (ExperimentManager)
- **实验配置版本化**: 自动保存每次实验的配置快照
- **训练过程记录**: 实时记录训练指标和检查点
- **实验信息查询**: 提供完整的实验元数据管理

### 2. 实验对比器 (ExperimentComparator)
- **多实验对比**: 支持多个实验的全面对比分析
- **配置差异分析**: 自动识别不同实验的配置差异
- **性能排名**: 基于验证损失等指标进行实验排名
- **可视化对比**: 生成训练曲线对比图

### 3. 实验报告生成器 (ExperimentReporter)
- **详细报告**: 生成HTML和文本格式的详细实验报告
- **训练分析**: 自动分析收敛性、稳定性等训练特征
- **可视化图表**: 生成专业的训练过程图表
- **实验建议**: 基于分析结果提供优化建议

## 📖 使用方法

### 基本导入
```python
from experiment_management import ExperimentManager, ExperimentComparator, ExperimentReporter

# 初始化实验管理器
manager = ExperimentManager('./experiments')

# 创建实验
config = {
    'model': {'input_dim': 62, 'output_dim': 24, 'hidden_layers': [64, 32, 16]},
    'training': {'epochs': 100, 'batch_size': 64, 'learning_rate': 0.001}
}

exp_id, exp_dir = manager.create_experiment(config, '测试实验')

# 记录训练指标
metrics = {
    'epoch': 1,
    'train_loss': 0.5,
    'val_loss': 0.3,
    'physics_loss': 0.2
}
manager.log_training_metrics(exp_id, metrics)
```

### 实验对比分析
```python
# 初始化对比器
comparator = ExperimentComparator()

# 比较多个实验
comparison = comparator.compare_experiments(['exp1', 'exp2', 'exp3'])

# 获取性能排名
ranking = comparison['performance_ranking']
print(f"最佳实验: {ranking[0]['experiment_id']}")
```

### 生成实验报告
```python
# 初始化报告生成器
reporter = ExperimentReporter()

# 生成HTML报告
html_report = reporter.generate_detailed_report(exp_id, 'html')

# 生成文本报告
text_report = reporter.generate_detailed_report(exp_id, 'txt')
```

## 🔧 与训练脚本集成

实验管理系统已经集成到 `efd_pinns_train.py` 训练脚本中：

```python
# 在训练脚本中的导入
from experiment_management import ExperimentManager

# 在 progressive_training 函数中使用
manager = ExperimentManager('./experiments')
exp_id, exp_dir = manager.create_experiment(config, 'PINNs训练实验')

# 训练过程中自动记录指标
manager.log_training_metrics(exp_id, metrics)
```

## 📊 输出文件结构

实验数据保存在 `experiments/` 目录下：

```
experiments/
├── experiments/           # 实验数据目录
│   ├── exp_20241126_143103/
│   │   ├── config.json       # 实验配置
│   │   ├── checkpoints/      # 模型检查点
│   │   └── reports/          # 训练报告
│   └── exp_20241126_143221/
├── comparison_figures/    # 对比图表
├── reports/               # 生成的报告
└── configs/              # 配置版本
```

## 🎯 优势特性

1. **模块化设计**: 各功能模块独立，便于维护和扩展
2. **路径灵活性**: 支持自定义实验目录路径
3. **错误处理**: 完善的错误处理和日志记录
4. **类型安全**: 使用类型注解提高代码可靠性
5. **可视化友好**: 支持多种图表格式和报告样式

## 🔄 更新日志

- **v1.0.0**: 初始版本，包含完整的实验管理、对比分析和报告生成功能
- **模块重构**: 从主目录移动到独立的 `experiment_management` 文件夹

## 📞 技术支持

如有问题或建议，请参考项目主目录的文档或联系开发团队。