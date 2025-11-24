#!/usr/bin/env python3
"""
长期训练优化脚本

基于long_run_longer实验的最佳配置，集成了以下改进：
1. 改进的学习率调度策略
2. 物理损失权重动态调整机制
3. 优化的训练循环，支持长时间训练
4. 增强的监控和检查点保存机制

用法:
    python long_term_training.py --output_dir ./long_term_run --epochs 100000
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
import copy
import random
import shutil
import glob
import math

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("long_term_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LongTermTraining')

# 抑制物理模块日志输出
physics_logger = logging.getLogger('EWPINN_Physics')
physics_logger.setLevel(logging.INFO)
physics_logger.propagate = True

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from ewp_pinn_model import EWPINN, EWPINNDataset, EWPINNInputLayer, EWPINNOutputLayer, extract_predictions
from ewp_pinn_config import ConfigManager
from ewp_pinn_performance_monitor import ModelPerformanceMonitor
from ewp_pinn_adaptive_hyperoptimizer import AdaptiveHyperparameterOptimizer
from ewp_pinn_optimizer import EWPINNOptimizerManager, WarmupCosineLR
from ewp_pinn_dynamic_weight import DynamicPhysicsWeightScheduler, PhysicsWeightIntegration
from ewp_pinn_physics import PINNConstraintLayer
from scripts.generate_constraint_report import compute_constraint_stats
from scripts.visualize_constraint_report import plot_residual_stats, plot_weight_series
from ewp_data_interface import validate_units
from ewp_pinn_optimized_train import progressive_training, save_model

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='长期EWPINN训练脚本')
    parser.add_argument('--output_dir', type=str, default='./long_term_run',
                        help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=1000,
                        help='学习率预热轮次')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--physics_weight', type=float, default=2.0,
                        help='物理损失权重')
    parser.add_argument('--dynamic_weight', action='store_true',
                        help='启用动态物理权重调整')
    parser.add_argument('--weight_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'stage_based', 'loss_ratio'],
                        help='动态权重调整策略')
    parser.add_argument('--checkpoint_interval', type=int, default=5000,
                        help='检查点保存间隔')
    parser.add_argument('--validation_interval', type=int, default=1000,
                        help='验证间隔')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--use_3d_mapping', action='store_true',
                        help='使用generate_pyvista_3d.py的3D结构参与输入映射')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='数据样本数（生成数据时使用）')
    parser.add_argument('--gpu_safe', action='store_true',
                        help='在GPU上按小批生成数据，避免OOM')
    parser.add_argument('--model_fp16', action='store_true',
                        help='将模型参数转换为FP16以降低显存占用')
    
    return parser.parse_args()

def set_global_seed(seed=42, deterministic=True):
    """设置全局随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"全局随机种子设置为: {seed}")

def setup_environment(args):
    """设置训练环境"""
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    if device.type == 'cuda':
        try:
            if args.gpu_safe:
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.3)
        except Exception:
            pass
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    return device

def load_or_create_config(args):
    """加载或创建配置"""
    if args.config and os.path.exists(args.config):
        logger.info(f"从文件加载配置: {args.config}")
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
    else:
        logger.info("创建默认配置")
        config = create_default_config(args)
    
    # 更新配置以匹配命令行参数
    config['训练流程']['总轮次'] = args.epochs
    config['训练流程']['阶段1']['lr'] = args.lr
    config['训练流程']['阶段1']['轮次'] = args.epochs // 4
    config['训练流程']['阶段2']['lr'] = args.lr * 0.5
    config['训练流程']['阶段2']['轮次'] = args.epochs // 4
    config['训练流程']['阶段3']['lr'] = args.lr * 0.2
    config['训练流程']['阶段3']['轮次'] = args.epochs // 4
    config['训练流程']['阶段4']['lr'] = args.lr * 0.1
    config['训练流程']['阶段4']['轮次'] = args.epochs // 4
    
    config['数据']['批次大小'] = args.batch_size
    config['损失函数']['物理残差损失权重'] = args.physics_weight
    
    return config

def create_default_config(args):
    """创建默认配置"""
    config = {
        '模型': {
            '输入维度': 62,
            '紧凑': True,
            '通道宽度高': 64,
            '通道宽度低': 32,
            '注意力头数': 1,
            '输出维度': 24,
            '激活函数': 'tanh',
            'dropout': 0.2
        },
        '训练流程': {
            '总轮次': args.epochs,
            '阶段1': {
                'lr': args.lr,
                '轮次': args.epochs // 4
            },
            '阶段2': {
                'lr': args.lr * 0.5,
                '轮次': args.epochs // 4
            },
            '阶段3': {
                'lr': args.lr * 0.2,
                '轮次': args.epochs // 4
            },
            '阶段4': {
                'lr': args.lr * 0.1,
                '轮次': args.epochs // 4
            }
        },
        '数据': {
            '批次大小': args.batch_size,
            '训练集比例': 0.7,
            '验证集比例': 0.2,
            '测试集比例': 0.1
        },
        '损失函数': {
            '物理残差损失权重': args.physics_weight
        },
        '物理约束': {
            '启用': True,
            '残差权重': {
                '质量守恒': 1.0,
                '动量守恒': 1.0,
                '能量守恒': 0.5
            },
            '自适应权重': True
        },
        '优化器': {
            '类型': 'AdamW',
            '权重衰减': 1e-5
        },
        '学习率调度': {
            '类型': 'WarmupCosineLR',
            '预热轮次': args.warmup_epochs,
            '最小学习率': args.min_lr
        }
    }
    
    return config

def generate_training_data(config, num_samples, device, output_dir):
    """生成训练数据"""
    logger.info(f"生成训练数据，样本数: {num_samples}")
    input_dim = config['模型']['输入维度']
    output_dim = config['模型']['输出维度']
    # 合成数据（可替换为真实数据加载）
    X_data = torch.randn(num_samples, input_dim)
    y_data = torch.randn(num_samples, output_dim)
    
    # 划分训练集、验证集和测试集
    n = len(X_data)
    train_size = int(n * config['数据']['训练集比例'])
    val_size = int(n * config['数据']['验证集比例'])
    
    indices = torch.randperm(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    X_train = X_data[train_indices]
    y_train = y_data[train_indices]
    X_val = X_data[val_indices]
    y_val = y_data[val_indices]
    X_test = X_data[test_indices]
    y_test = y_data[test_indices]
    
    # 生成物理点
    physics_points = X_data[torch.randperm(n)[:min(1000, n)]]
    
    # 保存数据集
    dataset_path = os.path.join(output_dir, 'dataset.npz')
    np.savez(
        dataset_path,
        X_train=X_train.cpu().numpy(),
        y_train=y_train.cpu().numpy(),
        X_val=X_val.cpu().numpy(),
        y_val=y_val.cpu().numpy(),
        X_test=X_test.cpu().numpy(),
        y_test=y_test.cpu().numpy(),
        physics_points=physics_points.cpu().numpy()
    )
    
    logger.info(f"数据集已保存到: {dataset_path}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, physics_points

def initialize_model(config, device, args):
    """初始化模型"""
    logger.info("初始化模型")
    model = EWPINN(
        input_dim=config['模型']['输入维度'],
        output_dim=config['模型']['输出维度'],
        device=device,
        config=config
    )
    if device.type == 'cuda' and args.model_fp16:
        model = model.half()
    model = model.to(device)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model

def create_optimizer_and_scheduler(model, config, args):
    """创建优化器和学习率调度器"""
    # 创建优化器
    optimizer_type = config['优化器']['类型']
    weight_decay = config['优化器'].get('权重衰减', config['优化器'].get('权重_decay', 1e-5))
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=weight_decay,
            momentum=0.9
        )
    
    # 创建学习率调度器
    scheduler_type = config['学习率调度']['类型']
    
    if scheduler_type == 'WarmupCosineLR':
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            min_lr=args.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.epochs // 10,
            gamma=0.9
        )
    
    return optimizer, scheduler

def create_dynamic_weight_scheduler(args):
    """创建动态物理权重调度器"""
    if not args.dynamic_weight:
        return None
    
    dynamic_weight_config = {
        'strategy': args.weight_strategy,
        'initial_weight': args.physics_weight,
        'min_weight': 0.01,
        'max_weight': 5.0,
        'adjustment_factor': 0.1,
        'smoothing_factor': 0.9,
        'stage_thresholds': [0.25, 0.5, 0.75],
        'stage_weights': [2.0, 1.5, 1.0, 0.5],
        'loss_ratio_threshold': 2.0,
        'verbose': True
    }
    
    scheduler = DynamicPhysicsWeightScheduler(dynamic_weight_config)
    integration = PhysicsWeightIntegration(scheduler)
    
    return integration

def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, args):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_history': loss_history,
        'args': args
    }
    
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最新检查点
    latest_path = os.path.join(args.output_dir, 'checkpoints', 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    logger.info(f"检查点已保存: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    
    logger.info(f"检查点已加载: {checkpoint_path}, 从轮次 {epoch} 恢复")
    
    return epoch, loss_history

def train_epoch(model, optimizer, scheduler, X_train, y_train, physics_points, 
               config, device, epoch, args, dynamic_weight_integration=None):
    """训练一个epoch"""
    model.train()
    
    batch_size = config['数据']['批次大小']
    num_batches = (len(X_train) + batch_size - 1) // batch_size
    
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    
    # 创建物理约束层
    pinn_layer = PINNConstraintLayer(
        residual_weights=config['物理约束']['残差权重']
    ).to(device)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_train))
        
        X_batch = X_train[start_idx:end_idx].to(device)
        y_batch = y_train[start_idx:end_idx].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播（数据拟合）
        # 混合精度前向（仅数据拟合部分）
        use_amp = (device.type == 'cuda' and getattr(args, 'model_fp16', False))
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                outputs = model(X_batch)
        else:
            outputs = model(X_batch)
        # 统一提取预测张量（数据拟合用）
        try:
            pred = extract_predictions(outputs)
        except Exception:
            if isinstance(outputs, dict) and len(outputs) > 0:
                first_key = list(outputs.keys())[0]
                pred = outputs[first_key]
            else:
                pred = outputs
        
        data_loss = torch.nn.functional.mse_loss(pred, y_batch)
        phys_size = pred.shape[0]
        if physics_points.shape[0] >= phys_size:
            idx = torch.randperm(physics_points.shape[0])[:phys_size]
            batch_phys = physics_points[idx]
        else:
            reps = phys_size // physics_points.shape[0] + 1
            batch_phys = physics_points.repeat(reps, 1)[:phys_size]
        batch_phys = batch_phys.to(device)
        batch_phys.requires_grad_(True)
        # 使用与物理点对应的预测参与物理约束计算，确保计算图连接到x_phys
        # 物理约束部分使用float32以保证稳定的梯度链路
        phys_outputs = model(batch_phys)
        try:
            pred_phys = extract_predictions(phys_outputs)
        except Exception:
            if isinstance(phys_outputs, dict) and len(phys_outputs) > 0:
                first_key = list(phys_outputs.keys())[0]
                pred_phys = phys_outputs[first_key]
            else:
                pred_phys = phys_outputs
        physics_loss, _ = pinn_layer.compute_physics_loss(
            batch_phys, pred_phys,
            data_loss=data_loss,
            epoch=epoch
        )
        
        # 应用动态权重调整（如果启用）
        if dynamic_weight_integration:
            physics_loss = dynamic_weight_integration.apply_dynamic_weight(
                physics_loss, data_loss, epoch=epoch
            )
        
        # 计算总损失
        total_batch_loss = data_loss + config['损失函数']['物理残差损失权重'] * physics_loss
        
        # 反向传播
        if device.type == 'cuda' and getattr(args, 'model_fp16', False):
            from torch.cuda.amp import GradScaler
            if not hasattr(train_epoch, '_scaler'):
                train_epoch._scaler = GradScaler()
            scaler = train_epoch._scaler
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            optimizer.step()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 在AMP路径中已执行step
        
        # 记录损失
        total_loss += total_batch_loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
    
    # 更新学习率
    if scheduler:
        scheduler.step()
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_data_loss = total_data_loss / num_batches
    avg_physics_loss = total_physics_loss / num_batches
    
    return avg_loss, avg_data_loss, avg_physics_loss

def validate_model(model, X_val, y_val, physics_points, config, device, args, dynamic_weight_integration=None):
    """验证模型"""
    model.eval()
    
    batch_size = config['数据']['批次大小']
    num_batches = (len(X_val) + batch_size - 1) // batch_size
    
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    
    # 创建物理约束层
    pinn_layer = PINNConstraintLayer(
        residual_weights=config['物理约束']['残差权重']
    ).to(device)
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_val))
            
            X_batch = X_val[start_idx:end_idx].to(device)
            y_batch = y_val[start_idx:end_idx].to(device)
            
            # 前向传播（数据拟合）
            use_amp = (device.type == 'cuda' and getattr(args, 'model_fp16', False))
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast(dtype=torch.float16):
                    outputs = model(X_batch)
            else:
                outputs = model(X_batch)
            # 统一提取预测张量（数据拟合用）
            try:
                pred = extract_predictions(outputs)
            except Exception:
                if isinstance(outputs, dict) and len(outputs) > 0:
                    first_key = list(outputs.keys())[0]
                    pred = outputs[first_key]
                else:
                    pred = outputs
            
            data_loss = torch.nn.functional.mse_loss(pred, y_batch)
            phys_size = pred.shape[0]
            if physics_points.shape[0] >= phys_size:
                idx = torch.randperm(physics_points.shape[0])[:phys_size]
                batch_phys = physics_points[idx]
            else:
                reps = phys_size // physics_points.shape[0] + 1
                batch_phys = physics_points.repeat(reps, 1)[:phys_size]
            batch_phys = batch_phys.to(device)
            batch_phys.requires_grad_(True)
            # 使用与物理点对应的预测参与物理约束计算，确保计算图连接到x_phys
            phys_outputs = model(batch_phys)
            try:
                pred_phys = extract_predictions(phys_outputs)
            except Exception:
                if isinstance(phys_outputs, dict) and len(phys_outputs) > 0:
                    first_key = list(phys_outputs.keys())[0]
                    pred_phys = phys_outputs[first_key]
                else:
                    pred_phys = phys_outputs
            physics_loss, _ = pinn_layer.compute_physics_loss(
                batch_phys, pred_phys,
                data_loss=data_loss
            )
            
            # 应用动态权重调整（如果启用）
            if dynamic_weight_integration:
                physics_loss = dynamic_weight_integration.apply_dynamic_weight(
                    physics_loss, data_loss
                )
            
            # 计算总损失
            total_batch_loss = data_loss + config['损失函数']['物理残差损失权重'] * physics_loss
            
            # 记录损失
            total_loss += total_batch_loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_data_loss = total_data_loss / num_batches
    avg_physics_loss = total_physics_loss / num_batches
    
    return avg_loss, avg_data_loss, avg_physics_loss

def run_training(model, optimizer, scheduler, config, data, device, args, dynamic_weight_integration=None):
    """运行训练"""
    X_train, y_train, X_val, y_val, X_test, y_test, physics_points = data
    
    # 初始化训练历史
    loss_history = {
        'train_losses': [],
        'val_losses': [],
        'physics_losses': [],
        'lr_history': [],
        'epochs_completed': 0
    }
    
    # 从检查点恢复（如果指定）
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume
        if os.path.exists(checkpoint_path):
            start_epoch, loss_history = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        else:
            logger.warning(f"检查点文件不存在: {checkpoint_path}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, optimizer, scheduler, X_train, y_train, physics_points,
            config, device, epoch, args, dynamic_weight_integration
        )
        
        # 记录训练损失
        loss_history['train_losses'].append(train_loss)
        loss_history['physics_losses'].append(train_physics_loss)
        loss_history['lr_history'].append(scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'])
        
        # 验证（按指定间隔）
        if epoch % args.validation_interval == 0:
            val_loss, val_data_loss, val_physics_loss = validate_model(
                model, X_val, y_val, physics_points, config, device, args, dynamic_weight_integration
            )
            
            # 记录验证损失
            loss_history['val_losses'].append(val_loss)
            
            # 打印进度
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            dynamic_weight = dynamic_weight_integration.get_current_weight() if dynamic_weight_integration else None
            
            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.6f} (Data: {train_data_loss:.6f}, Physics: {train_physics_loss:.6f}) - "
                f"Val Loss: {val_loss:.6f} (Data: {val_data_loss:.6f}, Physics: {val_physics_loss:.6f}) - "
                f"LR: {current_lr:.2e}" +
                (f" - Dynamic Weight: {dynamic_weight:.4f}" if dynamic_weight else "")
            )
            try:
                rep = compute_constraint_stats(model, X_val, y_val, physics_points, device)
                reports_dir = os.path.join(args.output_dir, 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                with open(os.path.join(reports_dir, f'constraint_diagnostics_epoch_{epoch}.json'), 'w', encoding='utf-8') as f:
                    json.dump(rep, f, indent=2, ensure_ascii=False)
                try:
                    plots_dir = os.path.join(args.output_dir, 'visualizations')
                    os.makedirs(plots_dir, exist_ok=True)
                    plot_residual_stats(rep, plots_dir)
                    plot_weight_series(rep, plots_dir)
                except Exception:
                    pass
            except Exception:
                pass
        
        # 保存检查点（按指定间隔）
        if epoch % args.checkpoint_interval == 0 and epoch > 0:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history, args)
    
    # 最终验证
    final_val_loss, final_val_data_loss, final_val_physics_loss = validate_model(
        model, X_val, y_val, physics_points, config, device, args, dynamic_weight_integration
    )
    loss_history['val_losses'].append(final_val_loss)
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'args': args,
        'loss_history': loss_history
    }, final_model_path)
    
    # 更新完成的轮次
    loss_history['epochs_completed'] = args.epochs
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.json')
    reports_dir = os.path.join(args.output_dir, 'reports')
    visuals_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    history_path = os.path.join(reports_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    try:
        rep = compute_constraint_stats(model, X_val, y_val, physics_points, device)
        with open(os.path.join(reports_dir, 'constraint_diagnostics_final.json'), 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        try:
            plot_residual_stats(rep, visuals_dir)
            plot_weight_series(rep, visuals_dir)
        except Exception:
            pass
    except Exception:
        pass

    # 统一写出验证结果（与README/docs一致）
    try:
        validation_results = {
            'test_loss': float(final_val_loss),
            'physics_loss': float(final_val_physics_loss),
            'normalized_loss': float(final_val_loss),
            'test_samples': int(len(X_test)),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(reports_dir, 'validation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    
    logger.info(f"训练完成！最终模型已保存到: {final_model_path}")
    
    return model, loss_history

def main():
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir
    config_path = args.config if args.config else os.path.join(os.getcwd(), 'config', 'long_run_config.json')
    model, normalizer, _ = progressive_training(
        config_path=config_path,
        resume_training=bool(args.resume),
        resume_checkpoint=args.resume if args.resume else None,
        mixed_precision=True,
        model_init_seed=None,
        use_efficient_architecture=True,
        model_compression_factor=1.0,
        output_dir=args.output_dir
    )
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_model(model, normalizer, final_model_path, config=config_path, metadata={'source':'long_term_wrapper'})
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
