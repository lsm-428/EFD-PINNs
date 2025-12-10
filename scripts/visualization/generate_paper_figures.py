#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图表生成脚本 - 三模式物理模型版
====================================

生成用于论文的高质量图表：
1. 训练曲线（损失收敛）
2. φ 分布对比图（PINN vs 目标）
3. 三模式演化图
4. 速度场可视化
5. 开口率动态响应
6. 侧视图截面
7. 物理示意图
8. 网络架构图

作者: EFD-PINNs Team
日期: 2024-12
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置论文风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.pinn_two_phase import TwoPhasePINN, DataGenerator, PHYSICS, DEFAULT_CONFIG

# 配置
OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用指定的训练输出（自动查找最新目录）
import glob
_output_dirs = sorted(glob.glob("outputs_pinn_*"))
MODEL_DIR = _output_dirs[-1] if _output_dirs else "outputs_pinn_20251210_022147"
MODEL_PATH = f"{MODEL_DIR}/best_model.pth"


def load_model():
    """加载训练好的模型"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', DEFAULT_CONFIG)
    
    model = TwoPhasePINN(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 加载模型: {MODEL_PATH}")
    print(f"   最佳损失: {checkpoint.get('best_loss', 'N/A')}")
    print(f"   训练轮数: {checkpoint.get('epoch', 'N/A')}")
    
    return model, checkpoint, config


def load_history():
    """加载训练历史"""
    # 优先从 final_model.pth 加载
    final_path = f"{MODEL_DIR}/final_model.pth"
    if os.path.exists(final_path):
        checkpoint = torch.load(final_path, map_location=DEVICE, weights_only=False)
        history = checkpoint.get('history', {})
        if history:
            return history
    
    # 尝试从 best_model.pth 加载
    best_path = f"{MODEL_DIR}/best_model.pth"
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
        history = checkpoint.get('history', {})
        if history:
            return history
    
    # 尝试从 training_curve.png 同目录的其他输出读取
    return {}


def predict_phi(model, V, t, z=None, n=100):
    """预测 φ 场"""
    Lx, Ly = PHYSICS['Lx'], PHYSICS['Ly']
    h_ink = PHYSICS['h_ink']
    
    if z is None:
        z = h_ink / 2  # 默认在油墨层中间
    
    x = np.linspace(0, Lx, n)
    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    
    inputs = np.stack([
        X.flatten(), Y.flatten(),
        np.full(n*n, z), np.full(n*n, t), np.full(n*n, V)
    ], axis=1).astype(np.float32)
    
    with torch.no_grad():
        outputs = model(torch.tensor(inputs, device=DEVICE)).cpu().numpy()
    
    phi = outputs[:, 4].reshape(n, n)
    return X, Y, phi


def figure1_training_curves():
    """图1: 训练曲线"""
    print("生成图1: 训练曲线...")
    
    history = load_history()
    
    # 如果没有历史数据，尝试复制已有的训练曲线图
    if not history or not history.get('epoch'):
        existing_curve = f"{MODEL_DIR}/training_curve.png"
        if os.path.exists(existing_curve):
            import shutil
            shutil.copy(existing_curve, f"{OUTPUT_DIR}/fig1_training_curves.png")
            print(f"  ✅ 复制已有训练曲线: {existing_curve}")
            return
        else:
            print("  ⚠️ 未找到训练历史，跳过")
            return
    
    epochs = history.get('epoch', [])
    total_loss = history.get('loss', [])
    interface_loss = history.get('interface', [])
    physics_loss = history.get('physics', [])
    lr = history.get('lr', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 获取阶段分界点
    stage1 = 5000
    stage2 = 15000
    
    # (a) 总损失
    ax = axes[0, 0]
    ax.semilogy(epochs, total_loss, 'b-', linewidth=1.5, label='Total Loss')
    ax.axvline(x=stage1, color='r', linestyle='--', alpha=0.5, label='Stage 2')
    ax.axvline(x=stage2, color='g', linestyle='--', alpha=0.5, label='Stage 3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Total Loss Convergence')
    ax.legend(loc='upper right')
    
    # (b) 界面损失
    ax = axes[0, 1]
    ax.semilogy(epochs, interface_loss, 'orange', linewidth=1.5)
    ax.axvline(x=stage1, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=stage2, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Interface Loss')
    
    # (c) 物理损失
    ax = axes[1, 0]
    ax.semilogy(epochs, physics_loss, 'green', linewidth=1.5)
    ax.axvline(x=stage1, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=stage2, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(c) Physics Loss')
    
    # (d) 学习率
    ax = axes[1, 1]
    ax.plot(epochs, lr, 'purple', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(d) Learning Rate Schedule')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_training_curves.png")
    plt.savefig(f"{OUTPUT_DIR}/fig1_training_curves.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig1_training_curves.png")


def figure2_phi_distribution():
    """图2: φ 分布图 - PINN 预测 vs 目标"""
    print("生成图2: φ 分布对比图...")
    
    model, _, config = load_model()
    data_gen = DataGenerator(config, torch.device('cpu'))
    
    h_ink = PHYSICS['h_ink']
    Lx, Ly = PHYSICS['Lx'], PHYSICS['Ly']
    
    voltages = [0, 15, 30]
    times = [0.005, 0.010, 0.020]
    
    fig, axes = plt.subplots(len(voltages), len(times)*2, figsize=(16, 10))
    
    n = 80
    x = np.linspace(0, Lx, n)
    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    z = h_ink / 2
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(times):
            # PINN 预测
            _, _, phi_pinn = predict_phi(model, V, t, z=z, n=n)
            
            # 目标值
            phi_target = np.zeros((n, n))
            for ii in range(n):
                for jj in range(n):
                    phi_target[ii, jj] = data_gen.target_phi_3d(x[jj], y[ii], z, t, V)
            
            # PINN 图
            ax = axes[i, j*2]
            im = ax.contourf(X*1e6, Y*1e6, phi_pinn, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X*1e6, Y*1e6, phi_pinn, levels=[0.5], colors='black', linewidths=1.5)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'PINN (t={t*1000:.0f}ms)')
            if j == 0:
                ax.set_ylabel(f'V={V}V\ny (μm)')
            if i == len(voltages)-1:
                ax.set_xlabel('x (μm)')
            
            # 目标图
            ax = axes[i, j*2+1]
            im = ax.contourf(X*1e6, Y*1e6, phi_target, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X*1e6, Y*1e6, phi_target, levels=[0.5], colors='black', linewidths=1.5)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'Target (t={t*1000:.0f}ms)')
            if i == len(voltages)-1:
                ax.set_xlabel('x (μm)')
    
    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('φ (1=ink, 0=transparent)')
    
    plt.savefig(f"{OUTPUT_DIR}/fig2_phi_distribution.png")
    plt.savefig(f"{OUTPUT_DIR}/fig2_phi_distribution.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig2_phi_distribution.png")


def figure3_three_mode_evolution():
    """图3: 三模式演化图"""
    print("生成图3: 三模式演化图...")
    
    model, _, _ = load_model()
    h_ink = PHYSICS['h_ink']
    Lx, Ly = PHYSICS['Lx'], PHYSICS['Ly']
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # 模式1：中心开口 (V=15V, η<50%)
    V_mode1 = 15
    times_mode1 = [0.002, 0.005, 0.008, 0.012, 0.020]
    
    for j, t in enumerate(times_mode1):
        ax = axes[0, j]
        X, Y, phi = predict_phi(model, V_mode1, t, z=h_ink/2, n=100)
        
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
        ax.set_aspect('equal')
        
        eta = np.mean(phi < 0.5)
        ax.set_title(f't={t*1000:.0f}ms, η={eta:.2f}', fontsize=10)
        
        if j == 0:
            ax.set_ylabel(f'Mode 1: Center Opening\nV={V_mode1}V\ny (μm)')
    
    # 模式2：四角液滴 (V=30V, 早期)
    V_mode2 = 30
    times_mode2 = [0.002, 0.004, 0.006, 0.008, 0.010]
    
    for j, t in enumerate(times_mode2):
        ax = axes[1, j]
        X, Y, phi = predict_phi(model, V_mode2, t, z=h_ink/2, n=100)
        
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
        ax.set_aspect('equal')
        
        eta = np.mean(phi < 0.5)
        ax.set_title(f't={t*1000:.0f}ms, η={eta:.2f}', fontsize=10)
        
        if j == 0:
            ax.set_ylabel(f'Mode 2: Four Corners\nV={V_mode2}V (early)\ny (μm)')
    
    # 模式3：单角液滴 (V=30V, 稳态)
    V_mode3 = 30
    times_mode3 = [0.012, 0.014, 0.016, 0.018, 0.020]
    
    for j, t in enumerate(times_mode3):
        ax = axes[2, j]
        X, Y, phi = predict_phi(model, V_mode3, t, z=h_ink/2, n=100)
        
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
        ax.set_aspect('equal')
        
        eta = np.mean(phi < 0.5)
        ax.set_title(f't={t*1000:.0f}ms, η={eta:.2f}', fontsize=10)
        ax.set_xlabel('x (μm)')
        
        if j == 0:
            ax.set_ylabel(f'Mode 3: Single Corner\nV={V_mode3}V (steady)\ny (μm)')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('φ (1=ink, 0=transparent)')
    
    plt.suptitle('Three-Mode Physical Model Evolution', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/fig3_three_mode_evolution.png")
    plt.savefig(f"{OUTPUT_DIR}/fig3_three_mode_evolution.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig3_three_mode_evolution.png")


def figure4_side_view():
    """图4: 侧视图截面"""
    print("生成图4: 侧视图截面...")
    
    model, _, _ = load_model()
    
    Lx, Lz = PHYSICS['Lx'], PHYSICS['Lz']
    Ly = PHYSICS['Ly']
    h_ink = PHYSICS['h_ink']
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    voltages = [0, 10, 20, 30]
    times = [0, 0.005, 0.010, 0.015, 0.020]
    
    nx, nz = 100, 50
    x = np.linspace(0, Lx, nx)
    z = np.linspace(0, Lz, nz)
    X, Z = np.meshgrid(x, z)
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(times):
            ax = axes[i, j]
            
            inputs = np.stack([
                X.flatten(), np.full(nx*nz, Ly/2),
                Z.flatten(), np.full(nx*nz, t), np.full(nx*nz, V)
            ], axis=1).astype(np.float32)
            
            with torch.no_grad():
                outputs = model(torch.tensor(inputs, device=DEVICE)).cpu().numpy()
            phi = outputs[:, 4].reshape(nz, nx)
            
            im = ax.contourf(X*1e6, Z*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X*1e6, Z*1e6, phi, levels=[0.5], colors='black', linewidths=2)
            
            # 标记油墨层高度
            ax.axhline(y=h_ink*1e6, color='green', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_title(f'V={V}V, t={t*1000:.0f}ms', fontsize=10)
            
            if j == 0:
                ax.set_ylabel('z (μm)')
            if i == len(voltages) - 1:
                ax.set_xlabel('x (μm)')
    
    plt.suptitle('Side View (y=Ly/2): φ Field Cross-Section\n(green dashed: initial ink layer height)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_side_view.png")
    plt.savefig(f"{OUTPUT_DIR}/fig4_side_view.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig4_side_view.png")


def figure5_aperture_dynamics():
    """图5: 开口率动态响应"""
    print("生成图5: 开口率动态响应...")
    
    model, _, config = load_model()
    data_gen = DataGenerator(config, torch.device('cpu'))
    
    h_ink = PHYSICS['h_ink']
    Lx, Ly = PHYSICS['Lx'], PHYSICS['Ly']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    voltages = [10, 15, 20, 25, 30]
    times = np.linspace(0, 0.02, 40)
    colors = plt.cm.viridis(np.linspace(0, 1, len(voltages)))
    
    # (a) 开口率 vs 时间
    ax = axes[0]
    for V, color in zip(voltages, colors):
        ar_pinn = []
        ar_target = []
        
        for t in times:
            # PINN 预测
            _, _, phi = predict_phi(model, V, t, z=h_ink/2, n=50)
            ar_pinn.append(np.mean(phi < 0.5) * 100)
            
            # 目标值
            ar_target.append(data_gen.get_opening_rate(V, t) * 100)
        
        ax.plot(times*1000, ar_pinn, '-', color=color, linewidth=2, label=f'{V}V (PINN)')
        ax.plot(times*1000, ar_target, '--', color=color, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Aperture Ratio (%)')
    ax.set_title('(a) Aperture Ratio vs Time\n(solid: PINN, dashed: target)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 80])
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='η=50% threshold')
    
    # (b) 稳态开口率 vs 电压
    ax = axes[1]
    V_range = np.linspace(0, 30, 31)
    
    for t_val in [0.010, 0.015, 0.020]:
        ar_pinn = []
        ar_target = []
        
        for V in V_range:
            _, _, phi = predict_phi(model, V, t_val, z=h_ink/2, n=50)
            ar_pinn.append(np.mean(phi < 0.5) * 100)
            ar_target.append(data_gen.get_opening_rate(V, t_val) * 100)
        
        ax.plot(V_range, ar_pinn, 'o-', linewidth=2, markersize=3, label=f't={t_val*1000:.0f}ms (PINN)')
        ax.plot(V_range, ar_target, '--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Aperture Ratio (%)')
    ax.set_title('(b) Aperture Ratio vs Voltage')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 80])
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_aperture_dynamics.png")
    plt.savefig(f"{OUTPUT_DIR}/fig5_aperture_dynamics.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig5_aperture_dynamics.png")


def figure6_velocity_field():
    """图6: 速度场可视化"""
    print("生成图6: 速度场可视化...")
    
    model, _, _ = load_model()
    
    h_ink = PHYSICS['h_ink']
    Lx, Ly = PHYSICS['Lx'], PHYSICS['Ly']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    voltages = [10, 20, 30]
    times = [0.005, 0.015]
    
    nx, ny = 25, 25
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    for i, t in enumerate(times):
        for j, V in enumerate(voltages):
            ax = axes[i, j]
            
            inputs = np.stack([
                X.flatten(), Y.flatten(),
                np.full(nx*ny, h_ink/2), np.full(nx*ny, t), np.full(nx*ny, V)
            ], axis=1).astype(np.float32)
            
            with torch.no_grad():
                outputs = model(torch.tensor(inputs, device=DEVICE)).cpu().numpy()
            
            u = outputs[:, 0].reshape(ny, nx)
            v = outputs[:, 1].reshape(ny, nx)
            phi = outputs[:, 4].reshape(ny, nx)
            
            speed = np.sqrt(u**2 + v**2)
            
            # 背景：φ 分布
            im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', alpha=0.5, vmin=0, vmax=1)
            ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
            
            # 速度矢量
            scale = 1e4
            q = ax.quiver(X*1e6, Y*1e6, u*scale, v*scale, speed, cmap='viridis', alpha=0.8)
            
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'V={V}V')
            if j == 0:
                ax.set_ylabel(f't={t*1000:.0f}ms\ny (μm)')
            if i == 1:
                ax.set_xlabel('x (μm)')
    
    plt.suptitle('Velocity Field at z=h_ink/2', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_velocity_field.png")
    plt.savefig(f"{OUTPUT_DIR}/fig6_velocity_field.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig6_velocity_field.png")


def figure7_error_analysis():
    """图7: 误差分析"""
    print("生成图7: 误差分析...")
    
    model, _, config = load_model()
    data_gen = DataGenerator(config, torch.device('cpu'))
    
    h_ink = PHYSICS['h_ink']
    Lx, Ly = PHYSICS['Lx'], PHYSICS['Ly']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    n = 50
    x = np.linspace(0, Lx, n)
    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    z = h_ink / 2
    
    # (a) MAE vs 电压
    ax = axes[0]
    voltages = np.arange(0, 31, 5)
    times_test = [0.005, 0.010, 0.015, 0.020]
    
    for t in times_test:
        mae_list = []
        for V in voltages:
            _, _, phi_pinn = predict_phi(model, V, t, z=z, n=n)
            
            phi_target = np.zeros((n, n))
            for ii in range(n):
                for jj in range(n):
                    phi_target[ii, jj] = data_gen.target_phi_3d(x[jj], y[ii], z, t, V)
            
            mae = np.mean(np.abs(phi_pinn - phi_target))
            mae_list.append(mae)
        
        ax.plot(voltages, mae_list, 'o-', linewidth=2, markersize=6, label=f't={t*1000:.0f}ms')
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('MAE')
    ax.set_title('(a) MAE vs Voltage')
    ax.legend()
    
    # (b) MAE vs 时间
    ax = axes[1]
    times_range = np.linspace(0.001, 0.02, 20)
    voltages_test = [10, 20, 30]
    
    for V in voltages_test:
        mae_list = []
        for t in times_range:
            _, _, phi_pinn = predict_phi(model, V, t, z=z, n=n)
            
            phi_target = np.zeros((n, n))
            for ii in range(n):
                for jj in range(n):
                    phi_target[ii, jj] = data_gen.target_phi_3d(x[jj], y[ii], z, t, V)
            
            mae = np.mean(np.abs(phi_pinn - phi_target))
            mae_list.append(mae)
        
        ax.plot(times_range*1000, mae_list, 'o-', linewidth=2, markersize=4, label=f'V={V}V')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('MAE')
    ax.set_title('(b) MAE vs Time')
    ax.legend()
    
    # (c) 误差分布热图
    ax = axes[2]
    V, t = 30, 0.015
    
    _, _, phi_pinn = predict_phi(model, V, t, z=z, n=n)
    
    phi_target = np.zeros((n, n))
    for ii in range(n):
        for jj in range(n):
            phi_target[ii, jj] = data_gen.target_phi_3d(x[jj], y[ii], z, t, V)
    
    error = np.abs(phi_pinn - phi_target)
    
    im = ax.contourf(X*1e6, Y*1e6, error, levels=20, cmap='hot')
    ax.contour(X*1e6, Y*1e6, phi_target, levels=[0.5], colors='white', linewidths=2, linestyles='--')
    ax.set_aspect('equal')
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title(f'(c) Error Distribution (V={V}V, t={t*1000:.0f}ms)')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('|Error|')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig7_error_analysis.png")
    plt.savefig(f"{OUTPUT_DIR}/fig7_error_analysis.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig7_error_analysis.png")


def figure8_physics_schematic():
    """图8: 物理示意图"""
    print("生成图8: 物理示意图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pixel_size = 174  # μm
    h_ink = 3  # μm
    Lz = 20  # μm
    
    # (a) 模式1：中心开口
    ax = axes[0]
    
    # 像素边界
    rect = plt.Rectangle((0, 0), pixel_size, pixel_size, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # 油墨区域（外环）
    ax.fill([0, pixel_size, pixel_size, 0], [0, 0, pixel_size, pixel_size], color='brown', alpha=0.3)
    
    # 透明区域（圆形开口）
    cx, cy = pixel_size/2, pixel_size/2
    r_open = 50
    circle = plt.Circle((cx, cy), r_open, color='lightblue', alpha=0.7)
    ax.add_patch(circle)
    
    # 界面
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + r_open*np.cos(theta), cy + r_open*np.sin(theta), 'r-', linewidth=2)
    
    ax.set_xlim([-10, pixel_size+10])
    ax.set_ylim([-10, pixel_size+10])
    ax.set_aspect('equal')
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title('(a) Mode 1: Center Opening\n(η < 50%)')
    
    # (b) 模式2：四角液滴
    ax = axes[1]
    
    rect = plt.Rectangle((0, 0), pixel_size, pixel_size, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # 透明背景
    ax.fill([0, pixel_size, pixel_size, 0], [0, 0, pixel_size, pixel_size], color='lightblue', alpha=0.3)
    
    # 四个角落的油墨液滴
    corners = [(0, 0), (pixel_size, 0), (0, pixel_size), (pixel_size, pixel_size)]
    r_droplet = 40
    for cx, cy in corners:
        circle = plt.Circle((cx, cy), r_droplet, color='brown', alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_xlim([-10, pixel_size+10])
    ax.set_ylim([-10, pixel_size+10])
    ax.set_aspect('equal')
    ax.set_xlabel('x (μm)')
    ax.set_title('(b) Mode 2: Four Corner Droplets\n(η > 50%, transient)')
    
    # (c) 模式3：单角液滴
    ax = axes[2]
    
    rect = plt.Rectangle((0, 0), pixel_size, pixel_size, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # 透明背景
    ax.fill([0, pixel_size, pixel_size, 0], [0, 0, pixel_size, pixel_size], color='lightblue', alpha=0.3)
    
    # 单个角落的大液滴
    r_single = 70
    circle = plt.Circle((0, 0), r_single, color='brown', alpha=0.5)
    ax.add_patch(circle)
    
    # 箭头表示汇聚方向
    ax.annotate('', xy=(20, 20), xytext=(pixel_size-30, pixel_size-30),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(pixel_size/2, pixel_size/2+20, 'Capillary\nmerging', fontsize=9, ha='center', color='green')
    
    ax.set_xlim([-10, pixel_size+10])
    ax.set_ylim([-10, pixel_size+10])
    ax.set_aspect('equal')
    ax.set_xlabel('x (μm)')
    ax.set_title('(c) Mode 3: Single Corner Droplet\n(η > 50%, steady state)')
    
    plt.suptitle('Three-Mode Physical Model Schematic (Top View at z=h_ink/2)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig8_physics_schematic.png")
    plt.savefig(f"{OUTPUT_DIR}/fig8_physics_schematic.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig8_physics_schematic.png")


def figure9_network_architecture():
    """图9: 网络架构图"""
    print("生成图9: 网络架构图...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 标题
    ax.text(0.5, 0.95, 'Two-Phase Flow PINN Architecture', fontsize=16, fontweight='bold', 
            ha='center', transform=ax.transAxes)
    
    # 输入层
    ax.text(0.08, 0.78, 'Input Layer', fontsize=12, fontweight='bold')
    ax.text(0.08, 0.72, '(x, y, z, t, V)', fontsize=11)
    ax.text(0.08, 0.66, '↓ Normalize', fontsize=10, color='blue')
    ax.text(0.08, 0.60, '(x/Lx, y/Ly, z/Lz, t/t_max, V/30)', fontsize=10)
    
    # Phi 网络
    ax.add_patch(plt.Rectangle((0.28, 0.50), 0.20, 0.35, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(0.38, 0.82, 'φ Network', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.38, 0.76, 'Input: 5D', fontsize=10, ha='center')
    ax.text(0.38, 0.70, 'Hidden: [128,128,64,64,32]', fontsize=9, ha='center')
    ax.text(0.38, 0.64, 'Activation: Tanh', fontsize=10, ha='center')
    ax.text(0.38, 0.58, 'Output: φ ∈ [0,1]', fontsize=10, ha='center')
    ax.text(0.38, 0.52, '(Sigmoid)', fontsize=9, ha='center', color='gray')
    
    # 速度网络
    ax.add_patch(plt.Rectangle((0.55, 0.50), 0.20, 0.35, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(0.65, 0.82, 'Velocity Network', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.65, 0.76, 'Input: 6D (+ φ)', fontsize=10, ha='center')
    ax.text(0.65, 0.70, 'Hidden: [64,64,64,32]', fontsize=9, ha='center')
    ax.text(0.65, 0.64, 'Activation: Tanh', fontsize=10, ha='center')
    ax.text(0.65, 0.58, 'Output: (u,v,w,p)', fontsize=10, ha='center')
    
    # 输出层
    ax.text(0.50, 0.42, 'Output: (u, v, w, p, φ)', fontsize=12, fontweight='bold', ha='center')
    
    # 物理损失
    ax.add_patch(plt.Rectangle((0.15, 0.08), 0.70, 0.28, fill=True, facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(0.50, 0.32, 'Physics-Informed Losses', fontsize=12, fontweight='bold', ha='center')
    
    losses = [
        '• Interface Data: MSE(φ_pred, φ_target)',
        '• Continuity: ∇·u = 0',
        '• VOF Transport: ∂φ/∂t + u·∇φ = 0',
        '• Navier-Stokes: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u',
        '• Volume Conservation: ∫φdV = const'
    ]
    for i, loss in enumerate(losses):
        ax.text(0.18, 0.26 - i*0.04, loss, fontsize=9, color='darkgreen')
    
    # 箭头
    ax.annotate('', xy=(0.28, 0.67), xytext=(0.20, 0.67),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.55, 0.67), xytext=(0.48, 0.67),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.50, 0.45), xytext=(0.50, 0.50),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.50, 0.36), xytext=(0.50, 0.40),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # φ 传递到速度网络
    ax.annotate('', xy=(0.55, 0.58), xytext=(0.48, 0.58),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='--'))
    ax.text(0.515, 0.56, 'φ', fontsize=10, color='blue')
    
    plt.savefig(f"{OUTPUT_DIR}/fig9_network_architecture.png")
    plt.savefig(f"{OUTPUT_DIR}/fig9_network_architecture.pdf")
    plt.close()
    print(f"  ✅ 保存到 {OUTPUT_DIR}/fig9_network_architecture.png")


def main():
    """生成所有论文图表"""
    print("=" * 60)
    print("生成论文图表")
    print(f"模型目录: {MODEL_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请确认训练输出目录正确")
        return
    
    # 生成所有图表
    figure1_training_curves()
    figure2_phi_distribution()
    figure3_three_mode_evolution()
    figure4_side_view()
    figure5_aperture_dynamics()
    figure6_velocity_field()
    figure7_error_analysis()
    figure8_physics_schematic()
    figure9_network_architecture()
    
    print("=" * 60)
    print(f"✅ 所有图表已保存到 {OUTPUT_DIR}/")
    print("=" * 60)
    
    # 列出生成的文件
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') or f.endswith('.pdf')])
    print(f"\n生成的文件 ({len(files)} 个):")
    for f in files:
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024
        print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
