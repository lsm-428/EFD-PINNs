#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试脚本，用于验证efd_pinns_train.py的集成功能
"""

import os
import torch
import numpy as np
from efd_pinns_train import (
    MultiStageTrainer,
    PhysicsEnhancedLoss,
    EnhancedDataAugmenter,
    EWPINNOptimizerManager,
    PINNConstraintLayer,
    generate_training_data,
    create_model
)

def test_integration():
    """测试集成功能"""
    print("开始集成测试...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建测试配置
    config = {
        'model_type': 'EWPINN',
        'input_dim': 3,
        'output_dim': 1,
        'hidden_dims': [64, 64, 64],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'epochs': 5,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-5,
        'physics_weight': 0.1,
        'physics_weight_strategy': 'linear',
        'physics_weight_adaptive': True,
        'enable_noise_augmentation': True,
        'noise_level': 0.01,
        'enable_scaling': True,
        'scaling_range': [0.95, 1.05],
        'enable_shifting': True,
        'shifting_range': [-0.05, 0.05],
        'constraint_alpha': 1.0,
        'constraint_beta': 1.0,
        'gradient_clipping': True,
        'max_grad_norm': 1.0,
        'num_samples': 1000,
        'val_split': 0.1,
        'test_split': 0.1,
        'x_range': [-1, 1],
        'y_range': [-1, 1],
        'z_range': [-1, 1],
        'physics_verification_batch_size': 200,
        'output_dir': './test_output',
        # 四阶段训练配置
        'training_stages': [
            {
                'epochs': 2,
                'learning_rate': 0.001,
                'physics_weight': 0.01
            },
            {
                'epochs': 2,
                'learning_rate': 0.0005,
                'physics_weight': 0.1
            },
            {
                'epochs': 1,
                'learning_rate': 0.0001,
                'physics_weight': 0.5
            }
        ]
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 测试1: 数据生成
    print("\n测试1: 数据生成")
    try:
        data = generate_training_data(config, device)
        print(f"✓ 数据生成成功: train={len(data['train'][0])}, val={len(data['val'][0])}, test={len(data['test'][0])}")
    except Exception as e:
        print(f"✗ 数据生成失败: {str(e)}")
        return False
    
    # 测试2: 模型创建
    print("\n测试2: 模型创建")
    try:
        model = create_model(config, device)
        print(f"✓ 模型创建成功: {model.__class__.__name__}")
    except Exception as e:
        print(f"✗ 模型创建失败: {str(e)}")
        return False
    
    # 测试3: 损失函数
    print("\n测试3: 物理增强损失函数")
    try:
        loss_function = PhysicsEnhancedLoss(config)
        # 测试损失计算
        sample_inputs = data['train'][0][:10]
        sample_targets = data['train'][1][:10]
        sample_physics = data['physics'][:10]
        
        total_loss, physics_loss = loss_function.compute(
            model, sample_inputs, sample_targets, sample_physics, device
        )
        print(f"✓ 损失计算成功: total_loss={total_loss.item():.6f}, physics_loss={physics_loss.item():.6f}")
    except Exception as e:
        print(f"✗ 损失函数失败: {str(e)}")
        return False
    
    # 测试4: 数据增强器
    print("\n测试4: 数据增强器")
    try:
        augmenter = EnhancedDataAugmenter(config)
        augmented_inputs, augmented_targets = augmenter(sample_inputs, sample_targets)
        print(f"✓ 数据增强成功")
    except Exception as e:
        print(f"✗ 数据增强器失败: {str(e)}")
        return False
    
    # 测试5: 优化器管理器
    print("\n测试5: 优化器管理器")
    try:
        optimizer_manager = EWPINNOptimizerManager(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        improved = optimizer_manager.step(1.0)
        print(f"✓ 优化器管理器测试成功: improved={improved}")
    except Exception as e:
        print(f"✗ 优化器管理器失败: {str(e)}")
        return False
    
    # 测试6: 物理约束层
    print("\n测试6: 物理约束层")
    try:
        constraint_layer = PINNConstraintLayer(config).to(device)
        sample_physics.requires_grad_(True)
        outputs = model(sample_physics)
        constraint = constraint_layer(sample_physics, outputs)
        print(f"✓ 物理约束层测试成功: constraint_shape={constraint.shape}")
    except Exception as e:
        print(f"✗ 物理约束层失败: {str(e)}")
        return False
    
    # 测试7: 多阶段训练器 (只进行少量迭代以验证功能)
    print("\n测试7: 多阶段训练器")
    try:
        # 使用简单的MSE损失进行训练器测试
        mse_loss = torch.nn.MSELoss()
        trainer = MultiStageTrainer(config, model, mse_loss, optimizer, device)
        
        # 进行少量训练
        losses = trainer.train(
            data['train'], 
            data['val'], 
            data['physics']
        )
        print(f"✓ 多阶段训练器测试成功")
        print(f"  训练损失: {losses['train'][-1]:.6f}")
        print(f"  验证损失: {losses['val'][-1]:.6f}")
    except Exception as e:
        print(f"✗ 多阶段训练器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 所有集成测试通过！")
    return True

if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)
