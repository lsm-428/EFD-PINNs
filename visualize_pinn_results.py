#!/usr/bin/env python3
"""
PINN 结果完整可视化
==================

生成：
1. φ 场分布图（底面俯视图）
2. φ 场截面图（侧视图）
3. 开口率 vs 时间曲线
4. 开口率 vs 电压曲线
5. 接触角 vs 开口率对比
6. 3D 界面可视化

作者: EFD-PINNs Team
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS, DEFAULT_CONFIG
from src.predictors.hybrid_predictor import HybridPredictor


def load_model(checkpoint_path: str):
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", DEFAULT_CONFIG)
    model = TwoPhasePINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def predict_phi_field(model, device, V, t, z=None, n=100):
    """
    预测 φ 场
    
    Args:
        model: PINN 模型
        device: 计算设备
        V: 电压 (V)
        t: 时间 (s)
        z: z 坐标 (m)，默认为油墨层中间 h_ink/2
        n: 采样点数
    
    Returns:
        X, Y, phi: 网格坐标和 φ 值
    """
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    h_ink = PHYSICS["h_ink"]
    
    # 默认在油墨层中间采样（更能反映油墨分布）
    if z is None:
        z = h_ink / 2  # 1.5 μm
    
    x = np.linspace(0, Lx, n)
    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    
    inputs = np.stack([
        X.flatten(), Y.flatten(),
        np.full(n*n, z), np.full(n*n, t), np.full(n*n, V)
    ], axis=1).astype(np.float32)
    
    with torch.no_grad():
        outputs = model(torch.tensor(inputs, device=device))
        phi = outputs[:, 4].cpu().numpy().reshape(n, n)
    
    return X, Y, phi


def compute_aperture(model, device, V, t, n=100, method="threshold"):
    """
    计算开口率 - 在油墨层中间采样
    
    物理定义（标准 VOF）：
    - φ=1: 油墨
    - φ=0: 透明/极性液体
    
    开口率 = 透明区域面积 / 像素面积
    透明区域定义：φ < 0.5（油墨被推开的区域）
    
    Args:
        model: PINN 模型
        device: 计算设备
        V: 电压 (V)
        t: 时间 (s)
        n: 采样点数
        method: "threshold" (阈值法，默认) 或 "contour" (接触线法)
    
    Returns:
        开口率 η ∈ [0, 1]
    
    Note:
        阈值法更稳定可靠。contour 方法在高开口率时会出错，
        因为 φ=0.5 等值线围绕的是角落的油墨区域而非中心透明区。
    """
    h_ink = PHYSICS["h_ink"]
    # 在油墨层中间采样
    _, _, phi = predict_phi_field(model, device, V, t, z=h_ink/2, n=n)
    
    # 透明阈值：φ < 0.5 表示透明区域（油墨被推开）
    TRANSPARENT_THRESHOLD = 0.5
    
    # 阈值法：直接统计 φ < 0.5 的像素比例
    # 这是最稳定可靠的方法
    return float(np.mean(phi < TRANSPARENT_THRESHOLD))


def plot_phi_bottom_view(model, device, output_dir):
    """
    绘制油墨层 φ 场俯视图
    
    在油墨层中间 (z=h_ink/2) 采样，更能反映油墨的实际分布：
    - φ=1: 油墨（红色）
    - φ=0: 透明/极性液体（蓝色）
    
    三种模式：
    1. 中心开口模式 (η<50%): 中心透明，油墨环绕
    2. 四角液滴模式 (η>50%, 早期): 油墨在四个角落
    3. 单角液滴模式 (η>50%, 稳态): 油墨汇聚到一个角落
    """
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    voltages = [0, 10, 20, 30]
    times = [0, 0.005, 0.010, 0.015, 0.020]
    h_ink = PHYSICS["h_ink"]
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(times):
            ax = axes[i, j]
            # 在油墨层中间采样 (z=h_ink/2 ≈ 1.5μm)
            X, Y, phi = predict_phi_field(model, device, V, t, z=h_ink/2, n=100)
            
            im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
            ax.set_aspect('equal')
            
            # 计算开口率：φ<0.5 的区域比例
            eta = np.mean(phi < 0.5)
            ax.set_title(f'V={V}V, t={t*1000:.0f}ms\nη={eta:.2f}', fontsize=10)
            
            if j == 0:
                ax.set_ylabel(f'y (μm)')
            if i == len(voltages) - 1:
                ax.set_xlabel('x (μm)')
    
    plt.suptitle(f'Ink Layer View (z={h_ink*1e6:.1f}μm): φ Field Distribution\n(φ=1: ink, φ=0: transparent)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phi_bottom_view.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/phi_bottom_view.png")


def plot_phi_side_view(model, device, output_dir):
    """绘制侧面 φ 场截面图"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    voltages = [0, 10, 20, 30]
    times = [0, 0.005, 0.010, 0.015, 0.020]
    
    Lx, Lz = PHYSICS["Lx"], PHYSICS["Lz"]
    Ly = PHYSICS["Ly"]
    
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
                outputs = model(torch.tensor(inputs, device=device))
                phi = outputs[:, 4].cpu().numpy().reshape(nz, nx)
            
            im = ax.contourf(X*1e6, Z*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X*1e6, Z*1e6, phi, levels=[0.5], colors='black', linewidths=2)
            
            # 标记油墨层高度
            ax.axhline(y=PHYSICS["h_ink"]*1e6, color='green', linestyle='--', alpha=0.5)
            
            ax.set_title(f'V={V}V, t={t*1000:.0f}ms', fontsize=10)
            
            if j == 0:
                ax.set_ylabel('z (μm)')
            if i == len(voltages) - 1:
                ax.set_xlabel('x (μm)')
    
    plt.suptitle('Side View (y=Ly/2): φ Field Cross-Section', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phi_side_view.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/phi_side_view.png")


def plot_aperture_dynamics(model, device, output_dir):
    """绘制开口率动态响应"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 开口率 vs 时间
    ax1 = axes[0]
    times = np.linspace(0, 0.02, 50)
    voltages = [0, 10, 20, 30]
    
    for V in voltages:
        apertures = [compute_aperture(model, device, V, t, n=50) for t in times]
        ax1.plot(times*1000, apertures, label=f'V={V}V', linewidth=2)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Aperture Ratio η')
    ax1.set_title('Aperture Ratio vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 1.0)  # 修正：开口率范围 0-100%
    
    # 开口率 vs 电压（稳态）
    ax2 = axes[1]
    voltages_fine = np.linspace(0, 30, 31)
    
    for t in [0.005, 0.010, 0.015, 0.020]:
        apertures = [compute_aperture(model, device, V, t, n=50) for V in voltages_fine]
        ax2.plot(voltages_fine, apertures, label=f't={t*1000:.0f}ms', linewidth=2)
    
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Aperture Ratio η')
    ax2.set_title('Aperture Ratio vs Voltage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1.0)  # 修正：开口率范围 0-100%
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aperture_dynamics.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/aperture_dynamics.png")


def plot_contact_angle_comparison(model, device, output_dir):
    """绘制接触角与开口率对比"""
    predictor = HybridPredictor()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    times = np.linspace(0, 0.02, 50)
    voltages = [0, 10, 20, 30]
    
    # 接触角 vs 时间
    ax1 = axes[0]
    for V in voltages:
        thetas = [predictor.predict(voltage=V, time=t) for t in times]
        ax1.plot(times*1000, thetas, label=f'V={V}V', linewidth=2)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Contact Angle θ (°)')
    ax1.set_title('Stage 1: Contact Angle Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='θ=90°')
    
    # 开口率 vs 时间
    ax2 = axes[1]
    for V in voltages:
        apertures = [compute_aperture(model, device, V, t, n=50) for t in times]
        ax2.plot(times*1000, apertures, label=f'V={V}V', linewidth=2)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Aperture Ratio η')
    ax2.set_title('Stage 2: PINN Aperture Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)  # 修正：开口率范围 0-100%
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/contact_angle_vs_aperture.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/contact_angle_vs_aperture.png")


def plot_training_curve(output_dir):
    """绘制训练曲线"""
    import json
    
    config_path = f"{output_dir}/config.json"
    if not os.path.exists(config_path):
        print("⚠️ 未找到配置文件，跳过训练曲线")
        return
    
    # 训练曲线已经在训练时生成
    print(f"✅ 训练曲线已存在: {output_dir}/training_curve.png")


def plot_contact_line_analysis(model, device, output_dir):
    """
    绘制接触线分析图
    
    展示：
    1. 接触线轮廓（在油墨层中间采样）
    2. 开口率随时间变化
    3. 开口率 vs 电压
    4. 接触角 vs 开口率关系
    """
    try:
        from skimage import measure
    except ImportError:
        print("⚠️ 需要 scikit-image 来绘制接触线分析")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    h_ink = PHYSICS["h_ink"]
    
    # 第一行：不同电压下的 φ 场和接触线轮廓 (t=15ms)
    voltages = [0, 10, 20, 30]
    t = 0.015
    
    for i, V in enumerate(voltages):
        ax = axes[0, i]
        # 在油墨层中间采样
        X, Y, phi = predict_phi_field(model, device, V, t, z=h_ink/2, n=100)
        
        # 绘制 φ 场
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # 找到并绘制接触线 (φ=0.5)
        contours = measure.find_contours(phi, level=0.5)
        
        dx = Lx / 99 * 1e6  # 转换为 μm
        dy = Ly / 99 * 1e6
        
        for contour in contours:
            x_coords = contour[:, 1] * dx
            y_coords = contour[:, 0] * dy
            ax.plot(x_coords, y_coords, 'k-', linewidth=2)
        
        # 使用阈值法计算开口率
        aperture = np.mean(phi < 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'V={V}V, t={t*1000:.0f}ms\nη={aperture:.3f}', fontsize=10)
        ax.set_xlabel('x (μm)')
        if i == 0:
            ax.set_ylabel('y (μm)')
    
    # 第二行第一列：开口率随时间变化
    ax2 = axes[1, 0]
    times = np.linspace(0, 0.02, 30)
    
    for V in voltages:
        apertures = [compute_aperture(model, device, V, t, n=50) for t in times]
        ax2.plot(times*1000, apertures, label=f'V={V}V', linewidth=2)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Aperture Ratio η')
    ax2.set_title('Aperture vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.0)
    
    # 第二行第二列：Stage 1 vs Stage 2 对比
    ax3 = axes[1, 1]
    predictor = HybridPredictor()
    
    for V in [10, 20, 30]:
        # Stage 1 开口率（从接触角计算）
        stage1_apertures = []
        for t in times:
            theta = predictor.predict(voltage=V, time=t)
            # 简化映射：θ < 90° 时开口
            if theta < 90:
                eta = (90 - theta) / 90 * 0.85  # 最大 85%
            else:
                eta = 0
            stage1_apertures.append(eta)
        
        # Stage 2 开口率（PINN）
        stage2_apertures = [compute_aperture(model, device, V, t, n=50) for t in times]
        
        ax3.plot(times*1000, stage1_apertures, '--', label=f'Stage1 V={V}V', alpha=0.7)
        ax3.plot(times*1000, stage2_apertures, '-', label=f'Stage2 V={V}V', linewidth=2)
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Aperture Ratio η')
    ax3.set_title('Stage 1 vs Stage 2')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    # 第二行第三列：开口率 vs 电压（稳态）
    ax4 = axes[1, 2]
    voltages_fine = np.linspace(0, 30, 16)
    
    for t in [0.010, 0.015, 0.020]:
        apertures = [compute_aperture(model, device, V, t, n=50) for V in voltages_fine]
        ax4.plot(voltages_fine, apertures, 'o-', label=f't={t*1000:.0f}ms', linewidth=2, markersize=4)
    
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Aperture Ratio η')
    ax4.set_title('Aperture vs Voltage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.0)
    
    # 第二行第四列：接触角 vs 开口率关系
    ax5 = axes[1, 3]
    
    for V in [10, 20, 30]:
        thetas = [predictor.predict(voltage=V, time=t) for t in times]
        apertures = [compute_aperture(model, device, V, t, n=50) for t in times]
        ax5.plot(thetas, apertures, 'o-', label=f'V={V}V', markersize=3)
    
    ax5.set_xlabel('Contact Angle θ (°)')
    ax5.set_ylabel('Aperture Ratio η')
    ax5.set_title('θ vs η Relationship')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax5.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/contact_line_analysis.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/contact_line_analysis.png")


def plot_three_mode_evolution(model, device, output_dir):
    """
    绘制三种模式的演化图
    
    展示物理模型的三种模式：
    1. 中心开口模式 (η<50%): 低电压，油墨环绕中心透明区
    2. 四角液滴模式 (η>50%, 早期): 高电压早期，毛细管切断
    3. 单角液滴模式 (η>50%, 稳态): 高电压稳态，油墨汇聚到一角
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    h_ink = PHYSICS["h_ink"]
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    
    # 模式1：中心开口 (V=15V, η<50%)
    V_mode1 = 15
    times_mode1 = [0.002, 0.005, 0.008, 0.012, 0.020]
    
    for j, t in enumerate(times_mode1):
        ax = axes[0, j]
        X, Y, phi = predict_phi_field(model, device, V_mode1, t, z=h_ink/2, n=100)
        
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
        ax.set_aspect('equal')
        
        eta = np.mean(phi < 0.5)
        ax.set_title(f't={t*1000:.0f}ms, η={eta:.2f}', fontsize=10)
        
        if j == 0:
            ax.set_ylabel(f'Mode 1: Center Opening\nV={V_mode1}V\ny (μm)')
    
    # 模式2：四角液滴 (V=30V, 早期 t<12ms)
    V_mode2 = 30
    times_mode2 = [0.002, 0.004, 0.006, 0.008, 0.010]
    
    for j, t in enumerate(times_mode2):
        ax = axes[1, j]
        X, Y, phi = predict_phi_field(model, device, V_mode2, t, z=h_ink/2, n=100)
        
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
        ax.set_aspect('equal')
        
        eta = np.mean(phi < 0.5)
        ax.set_title(f't={t*1000:.0f}ms, η={eta:.2f}', fontsize=10)
        
        if j == 0:
            ax.set_ylabel(f'Mode 2: Four Corners\nV={V_mode2}V (early)\ny (μm)')
    
    # 模式3：单角液滴 (V=30V, 稳态 t>12ms)
    V_mode3 = 30
    times_mode3 = [0.012, 0.014, 0.016, 0.018, 0.020]
    
    for j, t in enumerate(times_mode3):
        ax = axes[2, j]
        X, Y, phi = predict_phi_field(model, device, V_mode3, t, z=h_ink/2, n=100)
        
        im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
        ax.set_aspect('equal')
        
        eta = np.mean(phi < 0.5)
        ax.set_title(f't={t*1000:.0f}ms, η={eta:.2f}', fontsize=10)
        ax.set_xlabel('x (μm)')
        
        if j == 0:
            ax.set_ylabel(f'Mode 3: Single Corner\nV={V_mode3}V (steady)\ny (μm)')
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('φ (1=ink, 0=transparent)')
    
    plt.suptitle('Three-Mode Physical Model Evolution\n(sampled at z=h_ink/2)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"{output_dir}/three_mode_evolution.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/three_mode_evolution.png")


def plot_target_vs_predicted(model, device, output_dir):
    """
    对比目标 φ 值和预测 φ 值
    
    展示 PINN 学习效果
    """
    from src.models.pinn_two_phase import DataGenerator, DEFAULT_CONFIG
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    h_ink = PHYSICS["h_ink"]
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    
    # 创建数据生成器获取目标值
    data_gen = DataGenerator(DEFAULT_CONFIG, torch.device('cpu'))
    
    test_cases = [
        (0, 0.010, "V=0V, t=10ms"),
        (15, 0.010, "V=15V, t=10ms"),
        (30, 0.005, "V=30V, t=5ms"),
        (30, 0.020, "V=30V, t=20ms"),
    ]
    
    n = 50
    x = np.linspace(0, Lx, n)
    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    z = h_ink / 2
    
    for col, (V, t, title) in enumerate(test_cases):
        # 目标值
        ax_target = axes[0, col]
        phi_target = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                phi_target[i, j] = data_gen.target_phi_3d(x[j], y[i], z, t, V)
        
        im = ax_target.contourf(X*1e6, Y*1e6, phi_target, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax_target.contour(X*1e6, Y*1e6, phi_target, levels=[0.5], colors='black', linewidths=2)
        ax_target.set_aspect('equal')
        ax_target.set_title(f'Target: {title}')
        if col == 0:
            ax_target.set_ylabel('Target φ\ny (μm)')
        
        # 预测值
        ax_pred = axes[1, col]
        _, _, phi_pred = predict_phi_field(model, device, V, t, z=z, n=n)
        
        im = ax_pred.contourf(X*1e6, Y*1e6, phi_pred, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax_pred.contour(X*1e6, Y*1e6, phi_pred, levels=[0.5], colors='black', linewidths=2)
        ax_pred.set_aspect('equal')
        ax_pred.set_xlabel('x (μm)')
        
        # 计算误差
        mse = np.mean((phi_pred - phi_target)**2)
        ax_pred.set_title(f'Predicted (MSE={mse:.4f})')
        if col == 0:
            ax_pred.set_ylabel('Predicted φ\ny (μm)')
    
    plt.suptitle('Target vs Predicted φ Field Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_vs_predicted.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/target_vs_predicted.png")


def plot_cv_curves(model, device, output_dir):
    """
    绘制 CV 曲线（Contact angle - Voltage）
    
    包含：
    1. 接触角 vs 电压（理论 Young-Lippmann）
    2. 开口率 vs 电压（PINN 预测）
    3. 接触角动态响应
    4. 电压扫描（升压/降压）
    """
    predictor = HybridPredictor()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ===== 1. 接触角 vs 电压（稳态 Young-Lippmann）=====
    ax1 = axes[0, 0]
    voltages = np.linspace(0, 35, 71)
    
    # 理论 Young-Lippmann 曲线
    theta_yl = []
    for V in voltages:
        theta = predictor.predict(voltage=V, time=0.1)  # 稳态
        theta_yl.append(theta)
    
    ax1.plot(voltages, theta_yl, 'b-', linewidth=2, label='Young-Lippmann (steady)')
    
    # 不同时间点的接触角
    for t in [0.005, 0.010, 0.020]:
        thetas = [predictor.predict(voltage=V, time=t) for V in voltages]
        ax1.plot(voltages, thetas, '--', label=f't={t*1000:.0f}ms', alpha=0.7)
    
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Contact Angle θ (°)')
    ax1.set_title('CV Curve: Contact Angle vs Voltage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='θ=90°')
    ax1.set_xlim(0, 35)
    ax1.set_ylim(40, 130)
    
    # ===== 2. 开口率 vs 电压（PINN 预测）=====
    ax2 = axes[0, 1]
    voltages_pinn = np.linspace(0, 30, 16)
    
    for t in [0.005, 0.010, 0.015, 0.020]:
        apertures = [compute_aperture(model, device, V, t, n=50) for V in voltages_pinn]
        ax2.plot(voltages_pinn, apertures, 'o-', label=f't={t*1000:.0f}ms', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Aperture Ratio η')
    ax2.set_title('CV Curve: Aperture Ratio vs Voltage (PINN)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1.0)
    
    # ===== 3. 接触角动态响应（阶跃电压）=====
    ax3 = axes[1, 0]
    times = np.linspace(0, 0.025, 100)
    
    for V in [10, 15, 20, 25, 30]:
        thetas = [predictor.predict(voltage=V, time=t) for t in times]
        ax3.plot(times*1000, thetas, label=f'V={V}V', linewidth=2)
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Contact Angle θ (°)')
    ax3.set_title('Dynamic Response: θ(t) for Step Voltage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=90, color='red', linestyle=':', alpha=0.5)
    ax3.set_xlim(0, 25)
    
    # ===== 4. 电压扫描（升压/降压迟滞）=====
    ax4 = axes[1, 1]
    
    # 升压扫描
    voltages_up = np.linspace(0, 30, 31)
    apertures_up = [compute_aperture(model, device, V, 0.015, n=50) for V in voltages_up]
    
    # 降压扫描（假设相同，实际可能有迟滞）
    voltages_down = np.linspace(30, 0, 31)
    apertures_down = [compute_aperture(model, device, V, 0.015, n=50) for V in voltages_down]
    
    ax4.plot(voltages_up, apertures_up, 'b-o', label='Voltage Up', linewidth=2, markersize=4)
    ax4.plot(voltages_down, apertures_down, 'r--s', label='Voltage Down', linewidth=2, markersize=4, alpha=0.7)
    
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Aperture Ratio η')
    ax4.set_title('Voltage Sweep (t=15ms)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(0, 1.0)
    
    plt.suptitle('CV Curves: Contact Angle and Aperture vs Voltage', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cv_curves.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/cv_curves.png")


def main():
    # 查找最新的训练输出
    output_dirs = sorted(glob.glob("outputs_pinn_*"))
    if not output_dirs:
        print("❌ 未找到训练输出目录")
        return
    
    latest_dir = output_dirs[-1]
    checkpoint_path = f"{latest_dir}/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 未找到模型文件: {checkpoint_path}")
        return
    
    print(f"使用模型: {checkpoint_path}")
    print(f"输出目录: {latest_dir}")
    print("=" * 60)
    
    # 加载模型
    model, device = load_model(checkpoint_path)
    
    # 生成可视化
    print("\n生成可视化...")
    plot_phi_bottom_view(model, device, latest_dir)
    plot_phi_side_view(model, device, latest_dir)
    plot_aperture_dynamics(model, device, latest_dir)
    plot_contact_angle_comparison(model, device, latest_dir)
    plot_contact_line_analysis(model, device, latest_dir)
    plot_three_mode_evolution(model, device, latest_dir)
    plot_target_vs_predicted(model, device, latest_dir)
    plot_training_curve(latest_dir)
    
    # 列出所有生成的文件
    print("\n" + "=" * 60)
    print("生成的文件:")
    for f in sorted(os.listdir(latest_dir)):
        if f.endswith('.png') or f.endswith('.pth') or f.endswith('.json'):
            size = os.path.getsize(f"{latest_dir}/{f}") / 1024
            print(f"  {f}: {size:.1f} KB")


if __name__ == "__main__":
    main()
