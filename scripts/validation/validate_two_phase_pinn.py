#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两相流 PINN 模型验证脚本
========================

验证 PINN 模型预测的 phi 分布与解析模型的一致性。

作者: EFD-PINNs Team
日期: 2024-12
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入训练模块
from src.models.pinn_two_phase import (
    TwoPhasePINN as TwoPhaseFlowPINN, 
    DataGenerator as TwoPhaseDataGenerator, 
    DEFAULT_CONFIG, PHYSICS as PHYSICS_CONSTANTS
)

logger = logging.getLogger(__name__)

try:
    from src.predictors.hybrid_predictor import HybridPredictor
    from src.models.aperture_model import EnhancedApertureModel
    HAS_PREDICTOR = True
except ImportError:
    HAS_PREDICTOR = False


def load_model(checkpoint_path: str, config: Dict[str, Any]) -> TwoPhaseFlowPINN:
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoPhaseFlowPINN(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"加载模型: {checkpoint_path}, epoch={checkpoint['epoch']}")
    return model, device


def compute_analytical_phi(x: np.ndarray, y: np.ndarray, V: float, t: float,
                          config: Dict[str, Any]) -> np.ndarray:
    """计算解析模型的 phi 分布"""
    domain = config.get("domain", {})
    Lx = domain.get("Lx", PHYSICS_CONSTANTS["Lx"])
    Ly = domain.get("Ly", PHYSICS_CONSTANTS["Ly"])
    h_ink = domain.get("h_ink", PHYSICS_CONSTANTS["h_ink"])
    
    cx, cy = Lx / 2, Ly / 2
    
    # 使用 EnhancedApertureModel 计算开口率
    if HAS_PREDICTOR:
        aperture_model = EnhancedApertureModel()
        result = aperture_model.predict_enhanced(voltage=V, time=t)
        aperture_ratio = result['aperture_ratio']
    else:
        # 简化解析公式
        theta0 = PHYSICS_CONSTANTS["theta0"]
        epsilon_r = PHYSICS_CONSTANTS["epsilon_r"]
        d = PHYSICS_CONSTANTS["d_dielectric"]
        gamma = PHYSICS_CONSTANTS["sigma"]
        epsilon_0 = 8.854e-12
        tau = PHYSICS_CONSTANTS["tau"]
        zeta = PHYSICS_CONSTANTS["zeta"]
        
        # Young-Lippmann
        cos_theta0 = np.cos(np.radians(theta0))
        ew_term = (epsilon_0 * epsilon_r * V**2) / (2 * gamma * d)
        cos_theta_eq = np.clip(cos_theta0 + ew_term, -1, 1)
        theta_eq = np.degrees(np.arccos(cos_theta_eq))
        
        # 动态响应
        omega_0 = 1.0 / tau
        omega_d = omega_0 * np.sqrt(max(0, 1 - zeta**2))
        exp_term = np.exp(-zeta * omega_0 * t)
        damping_factor = zeta / np.sqrt(1 - zeta**2) if zeta < 1 else 1.0
        theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
            np.cos(omega_d * t) + damping_factor * np.sin(omega_d * t)
        )
        
        # 开口率
        theta_min = np.degrees(np.arccos(np.clip(cos_theta0 + (epsilon_0 * epsilon_r * 30**2) / (2 * gamma * d), -1, 1)))
        aperture_ratio = np.clip((theta0 - theta_t) / (theta0 - theta_min), 0, 0.35)
    
    # 开口半径
    pixel_area = Lx * Ly
    r_open = np.sqrt(aperture_ratio * pixel_area / np.pi) if aperture_ratio > 0 else 0
    
    # 计算 phi
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    interface_width = 2e-6
    
    phi = np.zeros_like(r)
    if r_open <= 0:
        phi[:] = 1.0
    else:
        # 平滑过渡
        r_norm = (r - r_open) / interface_width
        phi = 0.5 * (1 + np.tanh(r_norm * 2))
    
    return phi, aperture_ratio, r_open


def predict_pinn_phi(model: TwoPhaseFlowPINN, x: np.ndarray, y: np.ndarray, 
                     z: float, t: float, V: float, device: torch.device) -> np.ndarray:
    """使用 PINN 模型预测 phi"""
    n_points = len(x)
    z_arr = np.full(n_points, z)
    t_arr = np.full(n_points, t)
    V_arr = np.full(n_points, V)
    
    inputs = np.stack([x, y, z_arr, t_arr, V_arr], axis=1).astype(np.float32)
    inputs_tensor = torch.tensor(inputs, device=device)
    
    with torch.no_grad():
        out = model(inputs_tensor).cpu().numpy()
    
    return out[:, 4]  # phi


def validate_model(checkpoint_path: str, config: Dict[str, Any], output_dir: str = "validation_results"):
    """验证模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model, device = load_model(checkpoint_path, config)
    
    domain = config.get("domain", {})
    Lx = domain.get("Lx", PHYSICS_CONSTANTS["Lx"])
    Ly = domain.get("Ly", PHYSICS_CONSTANTS["Ly"])
    
    # 测试条件
    voltages = [0, 10, 20, 30]
    times = [0.0, 0.005, 0.01, 0.015, 0.02]
    
    # 生成网格
    nx, ny = 100, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(len(voltages), len(times) + 1, figure=fig, width_ratios=[1]*len(times) + [0.05])
    
    errors = []
    
    for i, V in enumerate(voltages):
        for j, t in enumerate(times):
            ax = fig.add_subplot(gs[i, j])
            
            # PINN 预测
            phi_pinn = predict_pinn_phi(model, x_flat, y_flat, 0.0, t, V, device)
            phi_pinn = phi_pinn.reshape(ny, nx)
            
            # 解析模型
            phi_analytical, aperture_ratio, r_open = compute_analytical_phi(x_flat, y_flat, V, t, config)
            phi_analytical = phi_analytical.reshape(ny, nx)
            
            # 计算误差
            mse = np.mean((phi_pinn - phi_analytical)**2)
            mae = np.mean(np.abs(phi_pinn - phi_analytical))
            errors.append({
                "V": V, "t": t, "mse": mse, "mae": mae,
                "aperture_ratio": aperture_ratio, "r_open": r_open * 1e6
            })
            
            # 绘制 PINN 预测
            im = ax.contourf(X * 1e6, Y * 1e6, phi_pinn, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.contour(X * 1e6, Y * 1e6, phi_pinn, levels=[0.5], colors='black', linewidths=1.5)
            
            # 绘制解析模型的界面（虚线）
            ax.contour(X * 1e6, Y * 1e6, phi_analytical, levels=[0.5], colors='lime', 
                      linewidths=2, linestyles='--')
            
            ax.set_aspect('equal')
            
            if i == 0:
                ax.set_title(f't={t*1000:.0f}ms', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'V={V}V\ny (μm)', fontsize=10)
            else:
                ax.set_ylabel('')
            if i == len(voltages) - 1:
                ax.set_xlabel('x (μm)', fontsize=10)
            
            # 添加误差信息
            ax.text(0.02, 0.98, f'MAE={mae:.3f}\nAR={aperture_ratio:.1%}', 
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加颜色条
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im, cax=cax, label='φ (Volume Fraction)')
    
    plt.suptitle('Two-Phase Flow PINN Validation\nSolid: PINN, Dashed: Analytical', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "pinn_vs_analytical.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"验证图像已保存: {fig_path}")
    
    # 保存误差统计
    import pandas as pd
    df = pd.DataFrame(errors)
    csv_path = os.path.join(output_dir, "validation_errors.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"误差统计已保存: {csv_path}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    print(f"平均 MAE: {df['mae'].mean():.4f}")
    print(f"平均 MSE: {df['mse'].mean():.6f}")
    print(f"最大 MAE: {df['mae'].max():.4f} (V={df.loc[df['mae'].idxmax(), 'V']}V, t={df.loc[df['mae'].idxmax(), 't']*1000:.0f}ms)")
    print("=" * 60)
    
    # 绘制误差随电压和时间的变化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 按电压分组
    ax = axes[0]
    for V in voltages:
        subset = df[df['V'] == V]
        ax.plot(subset['t'] * 1000, subset['mae'], 'o-', label=f'V={V}V')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('MAE')
    ax.set_title('MAE vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 按时间分组
    ax = axes[1]
    for t in times:
        subset = df[df['t'] == t]
        ax.plot(subset['V'], subset['mae'], 'o-', label=f't={t*1000:.0f}ms')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('MAE')
    ax.set_title('MAE vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "error_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"误差分析图已保存: {fig_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="两相流 PINN 模型验证")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--config", type=str, default="config_two_phase_flow.json", help="配置文件")
    parser.add_argument("--output", type=str, default="validation_two_phase", help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # 验证
    validate_model(args.checkpoint, config, args.output)


if __name__ == "__main__":
    main()
