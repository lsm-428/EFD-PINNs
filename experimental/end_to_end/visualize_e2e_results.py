#!/usr/bin/env python3
"""
端到端 PINN 结果可视化
=====================

用于可视化 train_end_to_end.py 训练的模型结果

使用方法:
    python visualize_e2e_results.py [output_dir]
    
    如果不指定 output_dir，会自动查找最新的 outputs_e2e_* 目录

作者: EFD-PINNs Team
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_end_to_end import EndToEndPINN, PHYSICS, load_config


def load_model(checkpoint_path: str, config_path: str = None):
    """加载端到端 PINN 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        load_config(config_path)
    
    config = checkpoint.get("config", {})
    model_cfg = config.get('model', {})
    hidden_dims = model_cfg.get('hidden_dims', [256, 256, 256, 128])
    
    model = EndToEndPINN(hidden_dims).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✅ 模型加载成功: {checkpoint_path}")
    print(f"   最佳损失: {checkpoint.get('best_loss', 'N/A')}")
    
    return model, device


def predict_phi_field(model, device, V, t, z=None, n=100):
    """预测 φ 场"""
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    h_ink = PHYSICS["h_ink"]
    
    if z is None:
        z = h_ink / 2
    
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


def compute_aperture(model, device, V, t, n=50):
    """计算开口率"""
    h_ink = PHYSICS["h_ink"]
    _, _, phi = predict_phi_field(model, device, V, t, z=h_ink/2, n=n)
    return float(np.mean(phi < 0.5))


def plot_phi_bottom_view(model, device, output_dir):
    """绘制油墨层 φ 场俯视图"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    voltages = [0, 10, 20, 30]
    times = [0, 0.005, 0.010, 0.015, 0.020]
    h_ink = PHYSICS["h_ink"]
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(times):
            ax = axes[i, j]
            X, Y, phi = predict_phi_field(model, device, V, t, z=h_ink/2, n=100)
            
            im = ax.contourf(X*1e6, Y*1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
            ax.set_aspect('equal')
            
            eta = np.mean(phi < 0.5)
            ax.set_title(f'V={V}V, t={t*1000:.0f}ms\nη={eta:.2f}', fontsize=10)
            
            if j == 0:
                ax.set_ylabel(f'y (μm)')
            if i == len(voltages) - 1:
                ax.set_xlabel('x (μm)')
    
    plt.suptitle(f'E2E PINN: phi Field (z={h_ink*1e6:.1f}μm)\n(phi=1: ink, phi=0: transparent)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/e2e_phi_bottom_view.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/e2e_phi_bottom_view.png")


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
            ax.axhline(y=PHYSICS["h_ink"]*1e6, color='green', linestyle='--', alpha=0.5)
            
            ax.set_title(f'V={V}V, t={t*1000:.0f}ms', fontsize=10)
            
            if j == 0:
                ax.set_ylabel('z (μm)')
            if i == len(voltages) - 1:
                ax.set_xlabel('x (μm)')
    
    plt.suptitle('E2E PINN: Side View (y=Ly/2)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/e2e_phi_side_view.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/e2e_phi_side_view.png")


def plot_aperture_dynamics(model, device, output_dir):
    """绘制开口率动态响应"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 开口率 vs 时间
    ax1 = axes[0]
    times = np.linspace(0, 0.02, 50)
    voltages = [0, 10, 20, 30]
    
    for V in voltages:
        apertures = [compute_aperture(model, device, V, t, n=30) for t in times]
        ax1.plot(times*1000, apertures, label=f'V={V}V', linewidth=2)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Aperture Ratio')
    ax1.set_title('Aperture Ratio vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 1.0)
    
    # 开口率 vs 电压
    ax2 = axes[1]
    voltages_fine = np.linspace(0, 30, 31)
    
    for t in [0.005, 0.010, 0.015, 0.020]:
        apertures = [compute_aperture(model, device, V, t, n=30) for V in voltages_fine]
        ax2.plot(voltages_fine, apertures, label=f't={t*1000:.0f}ms', linewidth=2)
    
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Aperture Ratio')
    ax2.set_title('Aperture Ratio vs Voltage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/e2e_aperture_dynamics.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/e2e_aperture_dynamics.png")


def plot_stage1_comparison(model, device, output_dir):
    """与 Stage 1 预测对比"""
    try:
        from src.models.aperture_model import EnhancedApertureModel
        from src.predictors.hybrid_predictor import HybridPredictor
    except ImportError:
        print("⚠️ Stage 1 模块不可用，跳过对比")
        return
    
    predictor = HybridPredictor(config_path='config/stage6_wall_effect.json')
    aperture_model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    voltages = np.linspace(0, 30, 31)
    t = 0.015
    
    # Stage 1 预测
    stage1_apertures = []
    for V in voltages:
        theta = predictor.young_lippmann(V)
        eta = aperture_model.contact_angle_to_aperture_ratio(theta)
        stage1_apertures.append(eta)
    
    # E2E PINN 预测
    e2e_apertures = [compute_aperture(model, device, V, t, n=30) for V in voltages]
    
    # 开口率对比
    ax1 = axes[0]
    ax1.plot(voltages, np.array(stage1_apertures) * 100, 'b-o', label='Stage 1 (Analytical)', markersize=4)
    ax1.plot(voltages, np.array(e2e_apertures) * 100, 'r-s', label='E2E PINN', markersize=4)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Aperture Ratio (%)')
    ax1.set_title(f'Stage 1 vs E2E PINN (t={t*1000:.0f}ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 100)
    
    # 误差分析
    ax2 = axes[1]
    errors = np.array(e2e_apertures) - np.array(stage1_apertures)
    ax2.bar(voltages, errors * 100, width=0.8, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Error (E2E - Stage1) (%)')
    ax2.set_title('Prediction Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/e2e_vs_stage1.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/e2e_vs_stage1.png")


def plot_training_history(output_dir):
    """绘制训练历史"""
    history_path = f"{output_dir}/history.json"
    if not os.path.exists(history_path):
        print("⚠️ 未找到训练历史文件")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1 = axes[0]
    if 'loss' in history:
        ax1.semilogy(history['loss'], label='Total Loss')
    if 'phi_loss' in history:
        ax1.semilogy(history['phi_loss'], label='Phi Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 开口率曲线
    ax2 = axes[1]
    epochs = np.arange(len(history.get('aperture_0V', []))) * 100
    if 'aperture_0V' in history:
        ax2.plot(epochs, np.array(history['aperture_0V']) * 100, label='V=0V')
    if 'aperture_20V' in history:
        ax2.plot(epochs, np.array(history['aperture_20V']) * 100, label='V=20V')
    if 'aperture_30V' in history:
        ax2.plot(epochs, np.array(history['aperture_30V']) * 100, label='V=30V')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Aperture Ratio (%)')
    ax2.set_title('Aperture During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/e2e_training_history.png", dpi=150)
    plt.close()
    print(f"✅ 已保存: {output_dir}/e2e_training_history.png")


def main():
    # 查找输出目录
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dirs = sorted(glob.glob("outputs_e2e_*"))
        if not output_dirs:
            print("❌ 未找到 outputs_e2e_* 目录")
            return
        output_dir = output_dirs[-1]
    
    checkpoint_path = f"{output_dir}/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 未找到模型文件: {checkpoint_path}")
        return
    
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 加载模型
    model, device = load_model(checkpoint_path, 'config/stage6_wall_effect.json')
    
    # 生成可视化
    print("\n生成可视化...")
    plot_phi_bottom_view(model, device, output_dir)
    plot_phi_side_view(model, device, output_dir)
    plot_aperture_dynamics(model, device, output_dir)
    plot_stage1_comparison(model, device, output_dir)
    plot_training_history(output_dir)
    
    print("\n" + "=" * 60)
    print("完成！生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            size = os.path.getsize(f"{output_dir}/{f}") / 1024
            print(f"  {f}: {size:.1f} KB")


if __name__ == "__main__":
    main()
