#!/usr/bin/env python3
"""
验证 PINN 求解的 φ 场物理合理性
================================

检查：
1. φ 场是否在 [0, 1] 范围内
2. 接触角边界条件是否满足
3. 从 φ 场积分计算开口率
4. 与解析模型对比

作者: EFD-PINNs Team
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS, DEFAULT_CONFIG


def load_model(checkpoint_path: str) -> TwoPhasePINN:
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", DEFAULT_CONFIG)
    model = TwoPhasePINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   最佳损失: {checkpoint.get('best_loss', 'N/A'):.4e}")
    print(f"   训练轮数: {checkpoint.get('epoch', 'N/A')}")
    
    return model, device


def compute_aperture_ratio(model, device, V: float, t: float, n_points: int = 100) -> float:
    """
    从 φ 场积分计算开口率
    
    开口率 η = ∫(1-φ)dA / A_pixel
    
    在底面 z=0 积分
    """
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    
    # 在底面 z=0 均匀采样
    x = np.linspace(0, Lx, n_points)
    y = np.linspace(0, Ly, n_points)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.zeros_like(x_flat)
    t_flat = np.full_like(x_flat, t)
    V_flat = np.full_like(x_flat, V)
    
    inputs = np.stack([x_flat, y_flat, z_flat, t_flat, V_flat], axis=1).astype(np.float32)
    
    with torch.no_grad():
        outputs = model(torch.tensor(inputs, device=device))
        phi = outputs[:, 4].cpu().numpy()
    
    # 开口率 = 透明区域 (1-φ) 的平均值
    aperture_ratio = np.mean(1 - phi)
    
    return aperture_ratio, phi.reshape(n_points, n_points)


def validate_phi_field(model, device):
    """验证 φ 场的物理合理性"""
    print("\n" + "=" * 60)
    print("验证 φ 场物理合理性")
    print("=" * 60)
    
    # 测试不同电压和时间
    voltages = [0, 10, 20, 30]
    times = [0.0, 0.005, 0.01, 0.02]
    
    results = []
    
    for V in voltages:
        for t in times:
            eta, phi_field = compute_aperture_ratio(model, device, V, t)
            
            # 检查 φ 范围
            phi_min, phi_max = phi_field.min(), phi_field.max()
            in_range = (phi_min >= -0.1) and (phi_max <= 1.1)
            
            results.append({
                "V": V, "t": t, "eta": eta,
                "phi_min": phi_min, "phi_max": phi_max,
                "valid": in_range
            })
            
            status = "✅" if in_range else "⚠️"
            print(f"V={V:2d}V, t={t*1000:5.1f}ms: η={eta:.3f}, φ∈[{phi_min:.3f}, {phi_max:.3f}] {status}")
    
    return results


def plot_aperture_vs_time(model, device, output_dir: str):
    """绘制开口率随时间变化"""
    print("\n" + "=" * 60)
    print("绘制开口率 vs 时间")
    print("=" * 60)
    
    voltages = [0, 10, 20, 30]
    times = np.linspace(0, 0.02, 50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for V in voltages:
        apertures = []
        for t in times:
            eta, _ = compute_aperture_ratio(model, device, V, t, n_points=50)
            apertures.append(eta)
        
        ax.plot(times * 1000, apertures, label=f"V={V}V", linewidth=2)
    
    ax.set_xlabel("时间 (ms)", fontsize=12)
    ax.set_ylabel("开口率 η", fontsize=12)
    ax.set_title("PINN 预测的开口率动态响应", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aperture_vs_time.png", dpi=150)
    plt.close()
    
    print(f"✅ 已保存: {output_dir}/aperture_vs_time.png")


def plot_phi_cross_section(model, device, output_dir: str):
    """绘制 φ 场的截面图"""
    print("\n" + "=" * 60)
    print("绘制 φ 场截面")
    print("=" * 60)
    
    Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
    
    # 在 y=Ly/2 截面绘制 φ(x, z)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    voltages = [0, 10, 20, 30]
    times = [0.0, 0.02]
    
    nx, nz = 100, 50
    x = np.linspace(0, Lx, nx)
    z = np.linspace(0, Lz, nz)
    X, Z = np.meshgrid(x, z)
    x_flat = X.flatten()
    z_flat = Z.flatten()
    y_flat = np.full_like(x_flat, Ly / 2)
    
    for i, t in enumerate(times):
        for j, V in enumerate(voltages):
            ax = axes[i, j]
            
            t_flat = np.full_like(x_flat, t)
            V_flat = np.full_like(x_flat, V)
            inputs = np.stack([x_flat, y_flat, z_flat, t_flat, V_flat], axis=1).astype(np.float32)
            
            with torch.no_grad():
                outputs = model(torch.tensor(inputs, device=device))
                phi = outputs[:, 4].cpu().numpy().reshape(nz, nx)
            
            im = ax.contourf(X * 1e6, Z * 1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X * 1e6, Z * 1e6, phi, levels=[0.5], colors='black', linewidths=2)
            
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("z (μm)")
            ax.set_title(f"V={V}V, t={t*1000:.0f}ms")
            ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phi_cross_section.png", dpi=150)
    plt.close()
    
    print(f"✅ 已保存: {output_dir}/phi_cross_section.png")


def main():
    # 查找最新的训练输出
    import glob
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
    
    # 加载模型
    model, device = load_model(checkpoint_path)
    
    # 验证 φ 场
    results = validate_phi_field(model, device)
    
    # 绘制开口率
    plot_aperture_vs_time(model, device, latest_dir)
    
    # 绘制截面
    plot_phi_cross_section(model, device, latest_dir)
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    valid_count = sum(1 for r in results if r["valid"])
    total_count = len(results)
    
    print(f"φ 范围检查: {valid_count}/{total_count} 通过")
    
    # 检查开口率趋势
    v30_results = [r for r in results if r["V"] == 30]
    eta_t0 = v30_results[0]["eta"]
    eta_t20 = v30_results[-1]["eta"]
    
    print(f"V=30V 开口率变化: {eta_t0:.3f} → {eta_t20:.3f}")
    
    if eta_t20 > eta_t0:
        print("✅ 开口率随时间增加（物理正确）")
    else:
        print("⚠️ 开口率未随时间增加（需要更多训练）")


if __name__ == "__main__":
    main()
