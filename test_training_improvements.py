#!/usr/bin/env python3
"""
训练效果测试脚本

此脚本用于测试和比较不同训练配置的效果：
1. 基准测试（原始配置）
2. 改进的学习率调度策略测试
3. 物理损失权重动态调整机制测试
4. 综合改进测试

用法:
    python test_training_improvements.py --test_type all --output_dir ./test_results
"""

import os
import sys
import json
import torch
import numpy as np
import time
import argparse
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestTraining')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from ewp_pinn_model import EWPINN, EWPINNDataset
from ewp_pinn_optimizer import EWPINNOptimizerManager, WarmupCosineLR
from ewp_pinn_dynamic_weight import DynamicPhysicsWeightScheduler, PhysicsWeightIntegration
from ewp_pinn_physics import PINNConstraintLayer
from long_term_training import (
    load_or_create_config, initialize_model, create_optimizer_and_scheduler,
    create_dynamic_weight_scheduler, generate_training_data, train_epoch, validate_model
)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练效果测试脚本')
    parser.add_argument('--test_type', type=str, default='all',
                        choices=['baseline', 'lr_scheduler', 'dynamic_weight', 'combined', 'all'],
                        help='测试类型')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='测试训练轮次')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=500,
                        help='学习率预热轮次')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--physics_weight', type=float, default=2.0,
                        help='物理损失权重')
    parser.add_argument('--weight_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'stage_based', 'loss_ratio'],
                        help='动态权重调整策略')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='每个测试的运行次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    
    return parser.parse_args()

def set_global_seed(seed=42, deterministic=True):
    """设置全局随机种子以确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_environment(args):
    """设置测试环境"""
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'baseline'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'lr_scheduler'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'dynamic_weight'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'combined'), exist_ok=True)
    
    return device

def run_baseline_test(args, device, run_id):
    """运行基准测试"""
    logger.info(f"运行基准测试 (运行 {run_id}/{args.num_runs})")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'baseline', f'run_{run_id}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置
    config = load_or_create_config(args)
    config['训练流程']['总轮次'] = args.epochs
    config['训练流程']['阶段1']['lr'] = args.lr
    config['训练流程']['阶段1']['轮次'] = args.epochs
    config['数据']['批次大小'] = args.batch_size
    config['损失函数']['物理残差损失权重'] = args.physics_weight
    
    # 设置基准学习率调度器（简单的StepLR）
    config['学习率调度']['类型'] = 'StepLR'
    
    # 生成数据
    X_train, y_train, X_val, y_val, X_test, y_test, physics_points = generate_training_data(
        config, 2000, device, output_dir
    )
    
    # 初始化模型
    model = initialize_model(config, device)
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, args)
    
    # 训练历史
    loss_history = {
        'train_losses': [],
        'val_losses': [],
        'physics_losses': [],
        'lr_history': [],
        'epochs_completed': 0
    }
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, optimizer, scheduler, X_train, y_train, physics_points,
            config, device, epoch
        )
        
        # 记录训练损失
        loss_history['train_losses'].append(train_loss)
        loss_history['physics_losses'].append(train_physics_loss)
        loss_history['lr_history'].append(scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'])
        
        # 验证（每100轮）
        if epoch % 100 == 0:
            val_loss, val_data_loss, val_physics_loss = validate_model(
                model, X_val, y_val, physics_points, config, device
            )
            
            # 记录验证损失
            loss_history['val_losses'].append(val_loss)
            
            # 打印进度
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            logger.info(
                f"Baseline Run {run_id} - Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}"
            )
    
    # 最终验证
    final_val_loss, _, _ = validate_model(
        model, X_val, y_val, physics_points, config, device
    )
    loss_history['val_losses'].append(final_val_loss)
    
    # 保存结果
    loss_history['epochs_completed'] = args.epochs
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss_history': loss_history
    }, final_model_path)
    
    return loss_history

def run_lr_scheduler_test(args, device, run_id):
    """运行改进的学习率调度策略测试"""
    logger.info(f"运行学习率调度策略测试 (运行 {run_id}/{args.num_runs})")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'lr_scheduler', f'run_{run_id}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置
    config = load_or_create_config(args)
    config['训练流程']['总轮次'] = args.epochs
    config['训练流程']['阶段1']['lr'] = args.lr
    config['训练流程']['阶段1']['轮次'] = args.epochs
    config['数据']['批次大小'] = args.batch_size
    config['损失函数']['物理残差损失权重'] = args.physics_weight
    
    # 设置改进的学习率调度器（WarmupCosineLR）
    config['学习率调度']['类型'] = 'WarmupCosineLR'
    config['学习率调度']['预热轮次'] = args.warmup_epochs
    config['学习率调度']['最小学习率'] = args.min_lr
    
    # 生成数据
    X_train, y_train, X_val, y_val, X_test, y_test, physics_points = generate_training_data(
        config, 2000, device, output_dir
    )
    
    # 初始化模型
    model = initialize_model(config, device)
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, args)
    
    # 训练历史
    loss_history = {
        'train_losses': [],
        'val_losses': [],
        'physics_losses': [],
        'lr_history': [],
        'epochs_completed': 0
    }
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, optimizer, scheduler, X_train, y_train, physics_points,
            config, device, epoch
        )
        
        # 记录训练损失
        loss_history['train_losses'].append(train_loss)
        loss_history['physics_losses'].append(train_physics_loss)
        loss_history['lr_history'].append(scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'])
        
        # 验证（每100轮）
        if epoch % 100 == 0:
            val_loss, val_data_loss, val_physics_loss = validate_model(
                model, X_val, y_val, physics_points, config, device
            )
            
            # 记录验证损失
            loss_history['val_losses'].append(val_loss)
            
            # 打印进度
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            logger.info(
                f"LR Scheduler Run {run_id} - Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}"
            )
    
    # 最终验证
    final_val_loss, _, _ = validate_model(
        model, X_val, y_test, physics_points, config, device
    )
    loss_history['val_losses'].append(final_val_loss)
    
    # 保存结果
    loss_history['epochs_completed'] = args.epochs
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss_history': loss_history
    }, final_model_path)
    
    return loss_history

def run_dynamic_weight_test(args, device, run_id):
    """运行物理损失权重动态调整机制测试"""
    logger.info(f"运行动态权重调整测试 (运行 {run_id}/{args.num_runs})")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'dynamic_weight', f'run_{run_id}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置
    config = load_or_create_config(args)
    config['训练流程']['总轮次'] = args.epochs
    config['训练流程']['阶段1']['lr'] = args.lr
    config['训练流程']['阶段1']['轮次'] = args.epochs
    config['数据']['批次大小'] = args.batch_size
    config['损失函数']['物理残差损失权重'] = args.physics_weight
    
    # 设置基准学习率调度器（简单的StepLR）
    config['学习率调度']['类型'] = 'StepLR'
    
    # 生成数据
    X_train, y_train, X_val, y_val, X_test, y_test, physics_points = generate_training_data(
        config, 2000, device, output_dir
    )
    
    # 初始化模型
    model = initialize_model(config, device)
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, args)
    
    # 创建动态权重调度器
    dynamic_weight_integration = create_dynamic_weight_scheduler(args)
    
    # 训练历史
    loss_history = {
        'train_losses': [],
        'val_losses': [],
        'physics_losses': [],
        'lr_history': [],
        'dynamic_weights': [],
        'epochs_completed': 0
    }
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, optimizer, scheduler, X_train, y_train, physics_points,
            config, device, epoch, dynamic_weight_integration
        )
        
        # 记录训练损失
        loss_history['train_losses'].append(train_loss)
        loss_history['physics_losses'].append(train_physics_loss)
        loss_history['lr_history'].append(scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'])
        
        # 记录动态权重
        if dynamic_weight_integration:
            loss_history['dynamic_weights'].append(dynamic_weight_integration.get_current_weight())
        
        # 验证（每100轮）
        if epoch % 100 == 0:
            val_loss, val_data_loss, val_physics_loss = validate_model(
                model, X_val, y_val, physics_points, config, device, dynamic_weight_integration
            )
            
            # 记录验证损失
            loss_history['val_losses'].append(val_loss)
            
            # 打印进度
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            dynamic_weight = dynamic_weight_integration.get_current_weight() if dynamic_weight_integration else None
            logger.info(
                f"Dynamic Weight Run {run_id} - Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}" +
                (f" - Dynamic Weight: {dynamic_weight:.4f}" if dynamic_weight else "")
            )
    
    # 最终验证
    final_val_loss, _, _ = validate_model(
        model, X_val, y_val, physics_points, config, device, dynamic_weight_integration
    )
    loss_history['val_losses'].append(final_val_loss)
    
    # 保存结果
    loss_history['epochs_completed'] = args.epochs
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss_history': loss_history
    }, final_model_path)
    
    return loss_history

def run_combined_test(args, device, run_id):
    """运行综合改进测试"""
    logger.info(f"运行综合改进测试 (运行 {run_id}/{args.num_runs})")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'combined', f'run_{run_id}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置
    config = load_or_create_config(args)
    config['训练流程']['总轮次'] = args.epochs
    config['训练流程']['阶段1']['lr'] = args.lr
    config['训练流程']['阶段1']['轮次'] = args.epochs
    config['数据']['批次大小'] = args.batch_size
    config['损失函数']['物理残差损失权重'] = args.physics_weight
    
    # 设置改进的学习率调度器（WarmupCosineLR）
    config['学习率调度']['类型'] = 'WarmupCosineLR'
    config['学习率调度']['预热轮次'] = args.warmup_epochs
    config['学习率调度']['最小学习率'] = args.min_lr
    
    # 生成数据
    X_train, y_train, X_val, y_val, X_test, y_test, physics_points = generate_training_data(
        config, 2000, device, output_dir
    )
    
    # 初始化模型
    model = initialize_model(config, device)
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, args)
    
    # 创建动态权重调度器
    dynamic_weight_integration = create_dynamic_weight_scheduler(args)
    
    # 训练历史
    loss_history = {
        'train_losses': [],
        'val_losses': [],
        'physics_losses': [],
        'lr_history': [],
        'dynamic_weights': [],
        'epochs_completed': 0
    }
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, optimizer, scheduler, X_train, y_train, physics_points,
            config, device, epoch, dynamic_weight_integration
        )
        
        # 记录训练损失
        loss_history['train_losses'].append(train_loss)
        loss_history['physics_losses'].append(train_physics_loss)
        loss_history['lr_history'].append(scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'])
        
        # 记录动态权重
        if dynamic_weight_integration:
            loss_history['dynamic_weights'].append(dynamic_weight_integration.get_current_weight())
        
        # 验证（每100轮）
        if epoch % 100 == 0:
            val_loss, val_data_loss, val_physics_loss = validate_model(
                model, X_val, y_val, physics_points, config, device, dynamic_weight_integration
            )
            
            # 记录验证损失
            loss_history['val_losses'].append(val_loss)
            
            # 打印进度
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            dynamic_weight = dynamic_weight_integration.get_current_weight() if dynamic_weight_integration else None
            logger.info(
                f"Combined Run {run_id} - Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}" +
                (f" - Dynamic Weight: {dynamic_weight:.4f}" if dynamic_weight else "")
            )
    
    # 最终验证
    final_val_loss, _, _ = validate_model(
        model, X_val, y_val, physics_points, config, device, dynamic_weight_integration
    )
    loss_history['val_losses'].append(final_val_loss)
    
    # 保存结果
    loss_history['epochs_completed'] = args.epochs
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss_history': loss_history
    }, final_model_path)
    
    return loss_history

def generate_comparison_plots(results, args):
    """生成比较图表"""
    logger.info("生成比较图表")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失比较
    for test_type, histories in results.items():
        if histories:
            # 计算平均损失
            avg_train_losses = np.mean([h['train_losses'] for h in histories], axis=0)
            epochs = range(len(avg_train_losses))
            axes[0, 0].plot(epochs, avg_train_losses, label=test_type)
    
    axes[0, 0].set_title('训练损失比较')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证损失比较
    for test_type, histories in results.items():
        if histories:
            # 计算平均验证损失
            avg_val_losses = np.mean([h['val_losses'] for h in histories], axis=0)
            val_epochs = np.linspace(0, args.epochs, len(avg_val_losses))
            axes[0, 1].plot(val_epochs, avg_val_losses, label=test_type)
    
    axes[0, 1].set_title('验证损失比较')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 学习率比较
    for test_type, histories in results.items():
        if histories:
            # 计算平均学习率
            avg_lr = np.mean([h['lr_history'] for h in histories], axis=0)
            epochs = range(len(avg_lr))
            axes[1, 0].plot(epochs, avg_lr, label=test_type)
    
    axes[1, 0].set_title('学习率比较')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 最终验证损失比较
    test_types = []
    final_val_losses = []
    final_val_stds = []
    
    for test_type, histories in results.items():
        if histories:
            test_types.append(test_type)
            # 获取每个运行的最终验证损失
            run_final_losses = [h['val_losses'][-1] for h in histories]
            final_val_losses.append(np.mean(run_final_losses))
            final_val_stds.append(np.std(run_final_losses))
    
    x_pos = np.arange(len(test_types))
    axes[1, 1].bar(x_pos, final_val_losses, yerr=final_val_stds, capsize=5)
    axes[1, 1].set_title('最终验证损失比较')
    axes[1, 1].set_xlabel('测试类型')
    axes[1, 1].set_ylabel('最终验证损失')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(test_types)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison_plots.png'), dpi=300)
    plt.close()
    
    logger.info(f"比较图表已保存到: {os.path.join(args.output_dir, 'comparison_plots.png')}")

def generate_summary_table(results, args):
    """生成总结表格"""
    logger.info("生成总结表格")
    
    # 创建表格数据
    table_data = []
    
    for test_type, histories in results.items():
        if histories:
            # 计算统计数据
            run_final_losses = [h['val_losses'][-1] for h in histories]
            min_val_losses = [min(h['val_losses']) for h in histories]
            
            # 计算平均最终验证损失和标准差
            mean_final_val = np.mean(run_final_losses)
            std_final_val = np.std(run_final_losses)
            
            # 计算平均最小验证损失和标准差
            mean_min_val = np.mean(min_val_losses)
            std_min_val = np.std(min_val_losses)
            
            # 计算收敛轮次（达到最小验证损失的轮次）
            convergence_epochs = []
            for h in histories:
                min_idx = np.argmin(h['val_losses'])
                # 验证损失记录的间隔是100轮
                conv_epoch = min_idx * 100
                convergence_epochs.append(conv_epoch)
            
            mean_conv_epoch = np.mean(convergence_epochs)
            std_conv_epoch = np.std(convergence_epochs)
            
            # 添加到表格
            table_data.append({
                '测试类型': test_type,
                '平均最终验证损失': mean_final_val,
                '最终验证损失标准差': std_final_val,
                '平均最小验证损失': mean_min_val,
                '最小验证损失标准差': std_min_val,
                '平均收敛轮次': mean_conv_epoch,
                '收敛轮次标准差': std_conv_epoch
            })
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存表格
    table_path = os.path.join(args.output_dir, 'summary_table.csv')
    df.to_csv(table_path, index=False)
    
    # 打印表格
    logger.info("测试结果总结:")
    print(df.to_string(index=False))
    
    logger.info(f"总结表格已保存到: {table_path}")
    
    return df

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    set_global_seed(args.seed)
    
    # 设置环境
    device = setup_environment(args)
    
    # 确定要运行的测试
    tests_to_run = []
    if args.test_type == 'all':
        tests_to_run = ['baseline', 'lr_scheduler', 'dynamic_weight', 'combined']
    else:
        tests_to_run = [args.test_type]
    
    # 运行测试
    results = {}
    
    for test_type in tests_to_run:
        logger.info(f"开始测试: {test_type}")
        
        test_results = []
        
        for run_id in range(1, args.num_runs + 1):
            # 为每次运行设置不同的随机种子
            run_seed = args.seed + run_id
            set_global_seed(run_seed)
            
            if test_type == 'baseline':
                history = run_baseline_test(args, device, run_id)
            elif test_type == 'lr_scheduler':
                history = run_lr_scheduler_test(args, device, run_id)
            elif test_type == 'dynamic_weight':
                history = run_dynamic_weight_test(args, device, run_id)
            elif test_type == 'combined':
                history = run_combined_test(args, device, run_id)
            
            test_results.append(history)
        
        results[test_type] = test_results
    
    # 生成比较图表和总结表格
    if len(tests_to_run) > 1:
        generate_comparison_plots(results, args)
        generate_summary_table(results, args)
    
    logger.info("所有测试完成！")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)