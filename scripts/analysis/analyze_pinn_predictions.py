#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN 预测分析脚本
================

分析 PINN 模型的预测结果，诊断学习问题。

作者: EFD-PINNs Team
日期: 2024-12
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.pinn_two_phase import (
    TwoPhasePINN as TwoPhaseFlowPINN, 
    DataGenerator as TwoPhaseDataGenerator,
    DEFAULT_CONFIG, PHYSICS as PHYSICS_CONSTANTS
)

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: dict):
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoPhaseFlowPINN(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, checkpoint.get("epoch", 0)


def analyze_phi_profile(model, device, config, output_dir):
    """分析 phi 沿径向的分布"""
    domain = config.get("domain", {})
    Lx = domain.get("Lx", PHYSICS_CONSTANTS["Lx"])
    Ly = domain.get("Ly", PHYSICS_CONSTANTS["Ly"])
    cx, cy = Lx / 2, Ly / 2
    
    # 沿 x 方向的剖面（y=cy, z=0）
    n_points = 200
    x = np.linspace(0, Lx, n_points)
    y = np.full(n_points, cy)
    z = np.zeros(n_points)
    
    voltages = [0, 10, 20, 30]
    times = [0.0, 0.005, 0.01, 0.015, 0.02]
    
    fig, axes = plt.subplots(len(voltages), len(times), figsize=(16, 12))
    
    # 数据生成器用于计算解析解
    data_gen = TwoPhaseDataGenerator(config, device)
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(times):
            ax = axes[i, j]
            
            # PINN 预测
            t_arr = np.full(n_points, t)
            V_arr = np.full(n_points, V)
            inputs = np.stack([x, y, z, t_arr, V_arr], axis=1).astype(np.float32)
            inputs_tensor = torch.tensor(inputs, device=device)
            
            with torch.no_grad():
                out = model(inputs_tensor).cpu().numpy()
            phi_pinn = out[:, 4]
            
            # 解析解 (使用 target_phi_3d)
            phi_analytical = np.array([
                data_gen.target_phi_3d(xi, cy, 0, t, V) for xi in x
            ])
            
            # 计算开口半径 (从 aperture_model 获取)
            r_open = 0  # 简化处理
            
            # 绘图
            ax.plot((x - cx) * 1e6, phi_pinn, 'b-', linewidth=2, label='PINN')
            ax.plot((x - cx) * 1e6, phi_analytical, 'r--', linewidth=2, label='Analytical')
            
            if r_open > 0:
                ax.axvline(r_open * 1e6, color='g', linestyle=':', label=f'r_open={r_open*1e6:.1f}μm')
                ax.axvline(-r_open * 1e6, color='g', linestyle=':')
            
            ax.set_xlim([-Lx/2 * 1e6, Lx/2 * 1e6])
            ax.set_ylim([-0.1, 1.1])
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title(f't={t*1000:.0f}ms')
            if j == 0:
                ax.set_ylabel(f'V={V}V\nφ')
            if i == len(voltages) - 1:
                ax.set_xlabel('x - cx (μm)')
            if i == 0 and j == len(times) - 1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('φ Profile along x-axis (y=cy, z=0)', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "phi_profile_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"剖面分析图已保存: {fig_path}")


def analyze_phi_statistics(model, device, config, output_dir):
    """分析 phi 的统计特性"""
    domain = config.get("domain", {})
    Lx = domain.get("Lx", PHYSICS_CONSTANTS["Lx"])
    Ly = domain.get("Ly", PHYSICS_CONSTANTS["Ly"])
    
    # 生成网格
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.zeros_like(x_flat)
    
    voltages = [0, 10, 20, 30]
    times = [0.0, 0.01, 0.02]
    
    stats = []
    
    for V in voltages:
        for t in times:
            t_arr = np.full(len(x_flat), t)
            V_arr = np.full(len(x_flat), V)
            inputs = np.stack([x_flat, y_flat, z_flat, t_arr, V_arr], axis=1).astype(np.float32)
            inputs_tensor = torch.tensor(inputs, device=device)
            
            with torch.no_grad():
                out = model(inputs_tensor).cpu().numpy()
            phi = out[:, 4]
            
            stats.append({
                'V': V, 't': t,
                'mean': phi.mean(),
                'std': phi.std(),
                'min': phi.min(),
                'max': phi.max(),
                'median': np.median(phi),
                'q25': np.percentile(phi, 25),
                'q75': np.percentile(phi, 75),
            })
    
    # 打印统计
    print("\n" + "=" * 80)
    print("φ 统计分析")
    print("=" * 80)
    print(f"{'V':>4} {'t':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'median':>8}")
    print("-" * 80)
    for s in stats:
        print(f"{s['V']:>4} {s['t']*1000:>6.0f}ms {s['mean']:>8.3f} {s['std']:>8.3f} "
              f"{s['min']:>8.3f} {s['max']:>8.3f} {s['median']:>8.3f}")
    print("=" * 80)
    
    # 绘制统计图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 按电压分组
    for V in voltages:
        subset = [s for s in stats if s['V'] == V]
        t_vals = [s['t'] * 1000 for s in subset]
        mean_vals = [s['mean'] for s in subset]
        std_vals = [s['std'] for s in subset]
        
        axes[0, 0].plot(t_vals, mean_vals, 'o-', label=f'V={V}V')
        axes[0, 1].plot(t_vals, std_vals, 'o-', label=f'V={V}V')
    
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Mean φ')
    axes[0, 0].set_title('Mean φ vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Std φ')
    axes[0, 1].set_title('Std φ vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 按时间分组
    for t in times:
        subset = [s for s in stats if s['t'] == t]
        V_vals = [s['V'] for s in subset]
        mean_vals = [s['mean'] for s in subset]
        std_vals = [s['std'] for s in subset]
        
        axes[1, 0].plot(V_vals, mean_vals, 'o-', label=f't={t*1000:.0f}ms')
        axes[1, 1].plot(V_vals, std_vals, 'o-', label=f't={t*1000:.0f}ms')
    
    axes[1, 0].set_xlabel('Voltage (V)')
    axes[1, 0].set_ylabel('Mean φ')
    axes[1, 0].set_title('Mean φ vs Voltage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Voltage (V)')
    axes[1, 1].set_ylabel('Std φ')
    axes[1, 1].set_title('Std φ vs Voltage')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "phi_statistics.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"统计分析图已保存: {fig_path}")


def analyze_training_history(history_path: str, output_dir: str):
    """分析训练历史"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epoch']
    
    # 总损失
    ax = axes[0, 0]
    ax.semilogy(epochs, history['total_loss'], 'b-', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    
    # 各项损失
    ax = axes[0, 1]
    ax.semilogy(epochs, history['ic_loss'], label='IC', alpha=0.7)
    ax.semilogy(epochs, history['bc_loss'], label='BC', alpha=0.7)
    ax.semilogy(epochs, history['interface_loss'], label='Interface', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 损失比例
    ax = axes[1, 0]
    total = np.array(history['total_loss'])
    ic = np.array(history['ic_loss'])
    bc = np.array(history['bc_loss'])
    interface = np.array(history['interface_loss'])
    
    ax.fill_between(epochs, 0, ic/total, label='IC', alpha=0.7)
    ax.fill_between(epochs, ic/total, (ic+bc)/total, label='BC', alpha=0.7)
    ax.fill_between(epochs, (ic+bc)/total, 1, label='Interface', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Fraction')
    ax.set_title('Loss Composition')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 学习率
    ax = axes[1, 1]
    ax.semilogy(epochs, history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"训练分析图已保存: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="PINN 预测分析")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点")
    parser.add_argument("--config", type=str, default="config_two_phase_flow.json", help="配置文件")
    parser.add_argument("--output", type=str, default="analysis_results", help="输出目录")
    parser.add_argument("--history", type=str, default=None, help="训练历史文件")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # 加载模型
    model, device, epoch = load_model(args.checkpoint, config)
    logger.info(f"加载模型 epoch={epoch}")
    
    # 分析
    analyze_phi_profile(model, device, config, args.output)
    analyze_phi_statistics(model, device, config, args.output)
    
    # 训练历史分析
    if args.history and os.path.exists(args.history):
        analyze_training_history(args.history, args.output)


if __name__ == "__main__":
    main()
