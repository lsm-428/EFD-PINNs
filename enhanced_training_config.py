#!/usr/bin/env python3
"""
EWPINN增强训练配置文件
"""

import torch

config = {
    # 基础配置
    'stage': 3,
    'input_dim': 62,
    'output_dim': 24,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 数据配置
    'num_train_samples': 500,  # 减少样本数量用于测试
    'num_val_samples': 100,
    'num_test_samples': 50,
    'batch_size': 16,
    'num_workers': 2,
    
    # 训练配置
    'early_stopping': True,
    'patience': 5,
    'save_best_only': True,
    'monitor_metric': 'val_loss',
    'min_delta': 1e-4,
    
    # 多阶段训练配置
    'multi_stage_config': {
        1: {
            'epochs': 3,  # 减少训练轮次用于测试
            'learning_rate': 1e-3,
            'optimizer': 'AdamW',
            'weight_decay': 1e-5,
            'scheduler': 'cosine',
            'warmup_epochs': 1,
            'description': '预训练阶段'
        },
        2: {
            'epochs': 3,
            'learning_rate': 5e-4,
            'optimizer': 'AdamW',
            'weight_decay': 1e-5,
            'scheduler': 'cosine',
            'description': '物理强化训练'
        },
        3: {
            'epochs': 3,
            'learning_rate': 1e-4,
            'optimizer': 'AdamW',
            'weight_decay': 1e-6,
            'scheduler': 'cosine',
            'description': '精细调优阶段'
        }
    },
    
    # 损失函数配置
    '损失函数': {
        '物理残差损失权重': 0.5,
        '边界条件损失权重': 1.0,
        '初始条件损失权重': 1.0,
        '数据损失权重': 1.0
    },
    
    # 优化器配置
    '优化器': {
        '类型': 'AdamW',
        '学习率': 1e-3,
        '权重衰减': 1e-5,
        '梯度裁剪': 1.0
    },
    
    # 正则化配置
    '正则化': {
        'dropout_rate': 0.1,
        '权重衰减': 1e-5,
        '梯度裁剪': 1.0
    },
    
    # 保存配置
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'save_frequency': 2,
    'save_optimizer': True,
    
    # 验证配置
    'validation_frequency': 1,
    'physics_validation_frequency': 2,
    
    # 梯度裁剪
    'gradient_clip_value': 1.0,
    'gradient_clip_norm': 1.0,
    
    # 随机种子
    'seed': 42,
    'deterministic': True
}