"""
实验管理模块
提供完整的实验跟踪、对比分析和报告生成功能
"""

from .experiment_manager import ExperimentManager
from .experiment_comparison import ExperimentComparator
from .experiment_reporter import ExperimentReporter

__all__ = [
    'ExperimentManager',
    'ExperimentComparator', 
    'ExperimentReporter'
]

__version__ = '1.0.0'

# 模块描述
__description__ = """
EFD3D 实验管理系统
===================

功能特性：
- 实验配置版本化管理
- 训练过程实时监控和记录
- 多实验对比分析
- 详细实验报告生成
- 训练过程可视化

使用示例：
>>> from experiment_management import ExperimentManager, ExperimentComparator
>>> manager = ExperimentManager('./experiments')
>>> comparator = ExperimentComparator()
"""