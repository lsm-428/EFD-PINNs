#!/usr/bin/env python3
"""
简单的EWPINN配置文件用于测试训练跟踪
"""

import torch

# 基础配置
config = {
    # 基础配置
    'stage': 1,
    'input_dim': 4,  # 简化输入维度
    'output_dim': 4,  # 简化输出维度
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 数据配置
    'num_train_samples': 100,
    'num_val_samples': 20,
    'num_test_samples': 20,
    'batch_size': 16,
    'num_workers': 1,
    
    # 训练配置
    'early_stopping': True,
    'patience': 5,
    'save_best_only': True,
    'monitor_metric': 'val_loss',
    'min_delta': 1e-4,
    
    # 多阶段训练配置
    'multi_stage_config': {
        1: {
            'epochs': 5,  # 减少训练轮次用于测试
            'learning_rate': 1e-3,
            'optimizer': 'AdamW',
            'weight_decay': 1e-5,
            'scheduler': 'cosine',
        }
    },
    
    # 优化器配置
    '优化器': {
        '类型': 'AdamW',
        '学习率': 1e-3,
        '权重衰减': 1e-5,
        '梯度裁剪': 1.0
    },
    
    # 损失函数配置
    '损失函数': {
        '物理残差损失权重': 0.5,
        '数据损失权重': 1.0
    },
    
    # 正则化配置
    '正则化': {
        'dropout_rate': 0.1,
        '权重衰减': 1e-5,
        '梯度裁剪': 1.0
    },
    
    # 网络架构配置
    '网络架构': {
        '隐藏层数': 3,
        '每层神经元数': [64, 32, 16],
        '激活函数': 'ReLU'
    }
}