#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两相流 PINN 模型验证脚本
========================

验证训练好的 PINN 模型：
1. 加载最佳模型
2. 在不同电压和时间下预测
3. 与解析解对比
4. 生成验证报告

作者: EFD-PINNs Team
日期: 2025-12-05
"""

import json
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入项目模块
from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS, DEFAULT_CONFIG

# 尝试导入开口率模型
try:
    from src.models.aperture_model import EnhancedApertureModel
    HAS_APERTURE = True
except ImportError:
    HAS_APERTURE = False
    print("警告: EnhancedApertureModel 不可用")


def load_model(model_path: str):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", DEFAULT_CONFIG)
    
    model = TwoPhasePINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, device, config


def predict_phi_distribution(model, device, V: float, t: float, nx: int = 50, ny: int = 50):
    """预测 phi 分布"""
    Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.zeros_like(x_flat)  # z=0 底面
    t_flat = np.full_like(x_flat, t)
    V_flat = np.full_like(x_flat, V)
    
    inputs = torch.tensor(
        np.stack([x_flat, y_flat, z_flat, t_flat, V_flat], axis=1),
        dtype=torch.float32,
        device=device
    )
    
    with torch.no_grad():
        outputs = model(inputs)
        phi = outputs[:, 4].cpu().numpy()
    
    return X, Y, phi.reshape(nx, ny)


def compute_aperture_ratio(phi: np.ndarray, threshold: float = 0.5) -> float:
    """从 phi 分布计算开口率"""
    return np.mean(phi < threshold)


def analytical_aperture_ratio(V: float, t: float) -> float:
    """解析公式计算开口率"""
    theta0 = PHYSICS["theta0"]
    epsilon_r = PHYSICS["epsilon_r"]
    d = PHYSICS["d_dielectric"]
    gamma = PHYSICS["sigma"]
    epsilon_0 = 8.854e-12
    tau = PHYSICS["tau"]
    zeta = PHYSICS["zeta"]
    
    # Young-Lippmann
    cos_theta0 = np.cos(np.radians(theta0))
    ew_term = (epsilon_0 * epsilon_r * V**2) / (2 * gamma * d)
    cos_theta_eq = np.clip(cos_theta0 + ew_term, -1, 1)
    theta_eq = np.degrees(np.arccos(cos_theta_eq))
    
    # 动态响应
    omega_0 = 1.0 / tau
    omega_d = omega_0 * np.sqrt(max(0, 1 - zeta**2))
    exp_term = np.exp(-zeta * omega_0 * t)
    damping = zeta / np.sqrt(1 - zeta**2) if zeta < 1 else 1.0
    theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
        np.cos(omega_d * t) + damping * np.sin(omega_d * t)
    )
    
    # 开口率
    theta_min = np.degrees(np.arccos(np.clip(
        cos_theta0 + (epsilon_0 * epsilon_r * 900) / (2 * gamma * d), -1, 1
    )))
    aperture_ratio = np.clip((theta0 - theta_t) / (theta0 - theta_min), 0, 0.5)
    
    return aperture_ratio


def validate_model(model_path: str, output_dir: str = None):
    """验证模型"""
    print("=" * 60)
    print("两相流 PINN 模型验证")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model, device, config = load_model(model_path)
    print(f"设备: {device}")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试条件
    voltages = [0, 10, 20, 30]
    times = [0.0, 0.005, 0.010, 0.015, 0.020]
    
    results = {
        "model_path": model_path,
        "validation_date": datetime.now().isoformat(),
        "voltages": voltages,
        "times": times,
        "predictions": [],
        "metrics": {}
    }
    
    # 预测和对比
    print("\n预测结果:")
    print("-" * 70)
    print(f"{'电压(V)':<10} {'时间(ms)':<10} {'PINN开口率':<15} {'解析开口率':<15} {'误差':<10}")
    print("-" * 70)
    
    errors = []
    
    for V in voltages:
        for t in times:
            # PINN 预测
            X, Y, phi = predict_phi_distribution(model, device, V, t)
            pinn_aperture = compute_aperture_ratio(phi)
            
            # 解析解
            analytical_aperture = analytical_aperture_ratio(V, t)
            
            # 误差
            error = abs(pinn_aperture - analytical_aperture)
            errors.append(error)
            
            print(f"{V:<10} {t*1000:<10.1f} {pinn_aperture:<15.4f} {analytical_aperture:<15.4f} {error:<10.4f}")
            
            results["predictions"].append({
                "voltage": V,
                "time": t,
                "pinn_aperture": float(pinn_aperture),
                "analytical_aperture": float(analytical_aperture),
                "error": float(error)
            })
    
    print("-" * 70)
    
    # 计算指标
    mae = np.mean(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    
    results["metrics"] = {
        "mae": float(mae),
        "max_error": float(max_error),
        "rmse": float(rmse)
    }
    
    print(f"\n验证指标:")
    print(f"  MAE (平均绝对误差): {mae:.4f}")
    print(f"  Max Error (最大误差): {max_error:.4f}")
    print(f"  RMSE (均方根误差): {rmse:.4f}")
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 1. 开口率对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, V in enumerate(voltages):
        ax = axes[idx // 2, idx % 2]
        
        t_range = np.linspace(0, 0.02, 50)
        pinn_apertures = []
        analytical_apertures = []
        
        for t in t_range:
            X, Y, phi = predict_phi_distribution(model, device, V, t, nx=30, ny=30)
            pinn_apertures.append(compute_aperture_ratio(phi))
            analytical_apertures.append(analytical_aperture_ratio(V, t))
        
        ax.plot(t_range * 1000, pinn_apertures, 'b-', linewidth=2, label='PINN')
        ax.plot(t_range * 1000, analytical_apertures, 'r--', linewidth=2, label='解析解')
        ax.set_xlabel('时间 (ms)')
        ax.set_ylabel('开口率')
        ax.set_title(f'V = {V}V')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aperture_comparison.png", dpi=150)
    plt.close()
    print(f"  保存: {output_dir}/aperture_comparison.png")
    
    # 2. phi 分布对比图
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    test_times = [0.0, 0.01, 0.02]
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(test_times):
            ax = axes[i, j]
            X, Y, phi = predict_phi_distribution(model, device, V, t)
            
            im = ax.contourf(X * 1e6, Y * 1e6, phi, levels=20, cmap='RdYlBu_r')
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            ax.set_title(f'V={V}V, t={t*1000:.0f}ms')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, label='φ')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phi_validation.png", dpi=150)
    plt.close()
    print(f"  保存: {output_dir}/phi_validation.png")
    
    # 保存结果
    with open(f"{output_dir}/validation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  保存: {output_dir}/validation_report.json")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证两相流 PINN 模型")
    parser.add_argument("--model", type=str, default="outputs_pinn_20251205_105428/best_model.pth",
                        help="模型路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    
    args = parser.parse_args()
    
    validate_model(args.model, args.output)
