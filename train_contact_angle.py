#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一阶段：接触角与开口率预测训练
================================

学习映射: (V, t) → θ(t) → η(t)

物理模型:
  - Young-Lippmann 方程: cos(θ) = cos(θ₀) + ε₀εᵣ(V-V_T)²/(2γd)
  - 二阶欠阻尼动态响应
  - 电容正反馈效应

用法:
    python train_contact_angle.py --epochs 3000
    python train_contact_angle.py --quick-run
    python train_contact_angle.py --aperture-demo --plot

作者: EFD-PINNs Team
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 日志配置
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ContactAngle")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# 物理参数（与 aperture_model.py 和 hybrid_predictor.py 一致）
# 实验参数：SU-8(400nm) + Teflon(400nm)，乙二醇/丙三醇混合液
# 实验结果：6V开始有开口，20V时开口率67%，20V以上可能翻墙
# ============================================================================
PHYSICS_PARAMS = {
    # Young-Lippmann 参数
    'theta0': 120.0,           # 初始接触角 (度)
    'epsilon_0': 8.854e-12,    # 真空介电常数 (F/m)
    'epsilon_r': 3.0,          # SU-8 介电常数
    'gamma': 0.050,            # 表面张力 (N/m) - 乙二醇混合液
    'd_dielectric': 4e-7,      # SU-8 厚度 (m) = 400nm
    'd_hydrophobic': 4e-7,     # Teflon 厚度 (m) = 400nm
    'epsilon_hydrophobic': 1.9,# Teflon 介电常数
    'V_threshold': 3.0,        # 阈值电压 (V) - 实验中6V开始有开口
    
    # 动力学参数
    'tau': 0.005,              # 时间常数 (s)
    'zeta': 0.8,               # 阻尼比
    
    # 电容参数（极性液体导电，不参与电容）
    'epsilon_ink': 3.0,        # 油墨介电常数
    'epsilon_polar': 80.0,     # 极性液体介电常数（导电，实际不参与串联）
    'd_fluid': 20e-6,          # 流体层/围堰高度 (m) = 20μm
    
    # 像素几何
    'pixel_size': 174e-6,      # 像素内沿尺寸 (m)
    'ink_thickness': 3e-6,     # 油墨厚度 (m) = 3-3.5μm
}


# ============================================================================
# 解析物理模型（用于生成训练数据）
# ============================================================================
class AnalyticalModel:
    """解析物理模型：Young-Lippmann + 二阶欠阻尼 + 电容反馈"""
    
    def __init__(self, params: Dict = None):
        self.p = params or PHYSICS_PARAMS
        
        # 派生参数
        self.omega_0 = 1.0 / self.p['tau']
        self.omega_d = self.omega_0 * np.sqrt(max(0, 1 - self.p['zeta']**2))
        
        # 开口率参数
        pixel_area = self.p['pixel_size'] ** 2
        ink_volume = self.p['ink_thickness'] * pixel_area
        L_weir = 4 * self.p['pixel_size']
        r_cross = np.sqrt(2 * ink_volume / (np.pi * L_weir))
        A_strip_min = 2 * r_cross * L_weir
        self.aperture_max = 1 - A_strip_min / pixel_area
    
    def young_lippmann(self, V: float) -> float:
        """Young-Lippmann 方程计算平衡接触角（SU-8 + Teflon 串联电容）"""
        V_eff = max(0, V - self.p['V_threshold'])
        
        # 串联电容的等效厚度
        # d_eff/ε_eff = d_SU8/ε_SU8 + d_Teflon/ε_Teflon
        d_eff = (self.p['d_dielectric'] / self.p['epsilon_r'] + 
                 self.p['d_hydrophobic'] / self.p['epsilon_hydrophobic'])
        
        cos_theta0 = np.cos(np.radians(self.p['theta0']))
        # 注意：d_eff 已归一化到 ε₀=1，所以这里用 ε₀ 而不是 ε₀ε_r
        ew_term = (self.p['epsilon_0'] * V_eff**2) / \
                  (2 * self.p['gamma'] * d_eff)
        cos_theta = np.clip(cos_theta0 + ew_term, -1, 1)
        return np.degrees(np.arccos(cos_theta))

    def dynamic_response(self, t: float, theta_start: float, theta_eq: float) -> float:
        """二阶欠阻尼动态响应"""
        zeta = self.p['zeta']
        
        if t <= 0:
            return theta_start
        
        if zeta >= 1:
            # 临界阻尼或过阻尼
            return theta_eq + (theta_start - theta_eq) * np.exp(-t / self.p['tau'])
        else:
            # 欠阻尼
            exp_term = np.exp(-zeta * self.omega_0 * t)
            damping_factor = zeta / np.sqrt(1 - zeta**2)
            return theta_eq + (theta_start - theta_eq) * exp_term * (
                np.cos(self.omega_d * t) + damping_factor * np.sin(self.omega_d * t)
            )
    
    def calculate_capacitance_ratio(self, eta: float) -> float:
        """
        计算电容比 C(η)/C(0)
        
        关键：极性液体是导电的！
        - 未开口区域：油墨 + SU-8 + Teflon 串联
        - 开口区域：SU-8 + Teflon 串联（极性液体导电，不参与电容）
        """
        eps0 = self.p['epsilon_0']
        
        # 各层电容密度
        C_d = eps0 * self.p['epsilon_r'] / self.p['d_dielectric']
        C_h = eps0 * self.p['epsilon_hydrophobic'] / self.p['d_hydrophobic']
        C_ink = eps0 * self.p['epsilon_ink'] / self.p['d_fluid']
        
        # 未开口区域：三层串联
        C_ink_region = 1.0 / (1.0/C_d + 1.0/C_h + 1.0/C_ink)
        
        # 开口区域：两层串联（极性液体导电）
        C_open_region = 1.0 / (1.0/C_d + 1.0/C_h)
        
        # 并联
        C_0 = C_ink_region  # η=0 时全是油墨
        C_eta = (1 - eta) * C_ink_region + eta * C_open_region
        
        return C_eta / C_0
    
    def theta_to_aperture(self, theta: float) -> float:
        """接触角 → 开口率（含电容正反馈）"""
        theta_change = max(0, self.p['theta0'] - theta)
        
        k = 1.2
        theta_scale = 28.0
        alpha = 0.15
        
        # 迭代求解
        eta = 0.0
        for _ in range(10):
            C_ratio = self.calculate_capacitance_ratio(eta)
            enhancement = 1.0 + alpha * (C_ratio - 1.0)
            x = k * theta_change * enhancement / theta_scale
            eta_new = self.aperture_max * np.tanh(x)
            if abs(eta_new - eta) < 1e-6:
                break
            eta = eta_new
        
        return eta
    
    def predict(self, V: float, t: float, V_initial: float = 0.0, t_step: float = 0.0):
        """完整预测：(V, t) → (θ, η)"""
        theta_eq = self.young_lippmann(V)
        theta_start = self.young_lippmann(V_initial)
        
        if t < t_step:
            theta = theta_start
        else:
            theta = self.dynamic_response(t - t_step, theta_start, theta_eq)
        
        eta = self.theta_to_aperture(theta)
        
        return theta, eta


# ============================================================================
# 神经网络模型
# ============================================================================
class ContactAngleNet(nn.Module):
    """
    接触角预测网络
    
    输入: [V, t, V_initial, t_step] (4维)
    输出: [θ, η] (2维)
    """
    
    def __init__(self, hidden_dims: list = [64, 64, 32]):
        super().__init__()
        
        layers = []
        in_dim = 4  # V, t, V_initial, t_step
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 2))  # θ, η
        
        self.net = nn.Sequential(*layers)
        
        # 输出缩放参数
        self.theta_scale = 60.0   # 接触角范围 60-120°
        self.theta_offset = 90.0  # 中心值
        
    def forward(self, x):
        """
        x: [batch, 4] - [V/30, t/0.02, V_initial/30, t_step/0.02]
        输出: [batch, 2] - [θ, η]
        """
        out = self.net(x)
        
        # θ: 使用 sigmoid 映射到 [60, 120]
        theta = torch.sigmoid(out[:, 0:1]) * self.theta_scale + (120 - self.theta_scale)
        
        # η: 使用 sigmoid 映射到 [0, 0.6]
        eta = torch.sigmoid(out[:, 1:2]) * 0.6
        
        return torch.cat([theta, eta], dim=1)


# ============================================================================
# 数据生成
# ============================================================================
def generate_training_data(
    num_samples: int = 10000,
    t_max: float = 0.02,
    V_max: float = 30.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用解析模型生成训练数据
    
    Returns:
        X: [num_samples, 4] - [V, t, V_initial, t_step] (归一化)
        y: [num_samples, 2] - [θ, η]
    """
    model = AnalyticalModel()
    
    X_list = []
    y_list = []
    
    # 采样策略：覆盖不同电压和时间组合
    voltages = np.linspace(0, V_max, 7)  # 0, 5, 10, 15, 20, 25, 30
    
    samples_per_voltage = num_samples // len(voltages)
    
    for V in voltages:
        for _ in range(samples_per_voltage):
            # 随机时间
            t = np.random.rand() * t_max
            
            # 随机初始电压和阶跃时间
            V_initial = np.random.choice([0, V])  # 从 0 或当前电压开始
            t_step = np.random.rand() * t_max * 0.2  # 阶跃在前 20% 时间内
            
            # 计算目标值
            theta, eta = model.predict(V, t, V_initial, t_step)
            
            # 归一化输入
            X_list.append([V / V_max, t / t_max, V_initial / V_max, t_step / t_max])
            y_list.append([theta, eta])
    
    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    
    if device:
        X = X.to(device)
        y = y.to(device)
    
    return X, y


# ============================================================================
# 物理损失（PINN）
# ============================================================================
def physics_loss(model: nn.Module, X: torch.Tensor, params: Dict) -> torch.Tensor:
    """
    物理约束损失：Young-Lippmann 方程
    
    在稳态时（t >> tau），θ 应该满足 Young-Lippmann 方程
    """
    X.requires_grad_(True)
    pred = model(X)
    theta_pred = pred[:, 0]
    
    # 提取输入
    V = X[:, 0] * 30.0  # 反归一化
    t = X[:, 1] * 0.02
    
    # 只对稳态点（t > 5*tau）施加 Young-Lippmann 约束
    tau = params['tau']
    steady_mask = t > 5 * tau
    
    if steady_mask.sum() == 0:
        return torch.tensor(0.0, device=X.device)
    
    V_steady = V[steady_mask]
    theta_steady = theta_pred[steady_mask]
    
    # Young-Lippmann 目标
    V_eff = torch.clamp(V_steady - params['V_threshold'], min=0)
    
    d_eff = (params['d_dielectric'] / params['epsilon_r'] + 
             params['d_hydrophobic'] / params['epsilon_hydrophobic'])
    d_eff *= params['epsilon_r']
    
    cos_theta0 = np.cos(np.radians(params['theta0']))
    ew_term = (params['epsilon_0'] * params['epsilon_r'] * V_eff**2) / \
              (2 * params['gamma'] * d_eff)
    cos_theta_target = torch.clamp(cos_theta0 + ew_term, -1, 1)
    theta_target = torch.rad2deg(torch.acos(cos_theta_target))
    
    # MSE 损失
    loss = torch.mean((theta_steady - theta_target) ** 2)
    
    return loss


# ============================================================================
# 训练函数
# ============================================================================
def train(
    epochs: int = 3000,
    num_samples: int = 10000,
    batch_size: int = 256,
    lr: float = 1e-3,
    physics_weight: float = 0.1,
    device: torch.device = None,
    output_dir: str = None,
    quick_run: bool = False
):
    """训练接触角预测模型"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if quick_run:
        epochs = 500
        num_samples = 2000
        logger.info("🚀 快速模式: epochs=500, samples=2000")
    
    logger.info(f"🔧 设备: {device}")
    logger.info(f"📊 样本数: {num_samples}, 批次: {batch_size}")
    
    # 生成数据
    logger.info("生成训练数据（基于解析物理模型）...")
    X, y = generate_training_data(num_samples, device=device)
    
    # 划分数据集
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    logger.info(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")
    
    # 创建模型
    model = ContactAngleNet(hidden_dims=[64, 64, 32]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"🏗️  模型参数量: {param_count:,}")
    
    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'physics_loss': [], 'lr': []}
    best_val_loss = float('inf')
    
    logger.info("=" * 60)
    logger.info("开始训练：(V, t) → (θ, η)")
    logger.info("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_physics = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 数据损失
            pred = model(X_batch)
            data_loss = nn.functional.mse_loss(pred, y_batch)
            
            # 物理损失
            phys_loss = physics_loss(model, X_batch, PHYSICS_PARAMS)
            
            # 总损失
            loss = data_loss + physics_weight * phys_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += data_loss.item() * len(X_batch)
            total_physics += phys_loss.item() * len(X_batch)
        
        scheduler.step()
        
        avg_loss = total_loss / len(X_train)
        avg_physics = total_physics / len(X_train)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred, y_val).item()
        
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['physics_loss'].append(avg_physics)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if output_dir:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'params': PHYSICS_PARAMS,
                }, os.path.join(output_dir, 'best_model.pth'))
        
        # 日志
        if epoch % 100 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch:5d}/{epochs} | "
                       f"train={avg_loss:.6f} | val={val_loss:.6f} | "
                       f"physics={avg_physics:.6f} | lr={history['lr'][-1]:.2e}")
    
    # 保存最终模型
    if output_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'params': PHYSICS_PARAMS,
        }, os.path.join(output_dir, 'final_model.pth'))
        
        # 保存历史
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    if output_dir:
        logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    return model, history


# ============================================================================
# 评估和可视化
# ============================================================================
def evaluate_model(model: nn.Module, device: torch.device):
    """评估训练好的模型"""
    model.eval()
    analytical = AnalyticalModel()
    
    print("\n" + "=" * 60)
    print("模型评估：预测 vs 解析")
    print("=" * 60)
    print(f"{'V(V)':<8} {'t(ms)':<8} {'θ_pred':<10} {'θ_anal':<10} {'η_pred':<10} {'η_anal':<10}")
    print("-" * 60)
    
    test_cases = [
        (0, 10), (10, 10), (20, 10), (30, 10),  # 不同电压，稳态
        (30, 1), (30, 3), (30, 5), (30, 10),    # 30V，不同时间
    ]
    
    for V, t_ms in test_cases:
        t = t_ms / 1000.0
        
        # 模型预测
        X = torch.tensor([[V/30, t/0.02, 0, 0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X)
        theta_pred = pred[0, 0].item()
        eta_pred = pred[0, 1].item()
        
        # 解析计算
        theta_anal, eta_anal = analytical.predict(V, t, V_initial=0, t_step=0)
        
        print(f"{V:<8} {t_ms:<8} {theta_pred:<10.2f} {theta_anal:<10.2f} "
              f"{eta_pred:<10.4f} {eta_anal:<10.4f}")
    
    print("=" * 60)


def plot_results(model: nn.Module, device: torch.device, save_path: str = None):
    """绘制预测结果"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")
        return
    
    model.eval()
    analytical = AnalyticalModel()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 稳态：电压 vs 接触角
    ax1 = axes[0, 0]
    voltages = np.linspace(0, 30, 50)
    theta_pred_list = []
    theta_anal_list = []
    
    for V in voltages:
        X = torch.tensor([[V/30, 0.5, 0, 0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X)
        theta_pred_list.append(pred[0, 0].item())
        theta_anal, _ = analytical.predict(V, 0.01, V_initial=0, t_step=0)
        theta_anal_list.append(theta_anal)
    
    ax1.plot(voltages, theta_anal_list, 'b-', linewidth=2, label='解析模型')
    ax1.plot(voltages, theta_pred_list, 'r--', linewidth=2, label='PINN 预测')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Contact Angle (°)')
    ax1.set_title('稳态接触角 vs 电压')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 稳态：电压 vs 开口率
    ax2 = axes[0, 1]
    eta_pred_list = []
    eta_anal_list = []
    
    for V in voltages:
        X = torch.tensor([[V/30, 0.5, 0, 0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X)
        eta_pred_list.append(pred[0, 1].item())
        _, eta_anal = analytical.predict(V, 0.01, V_initial=0, t_step=0)
        eta_anal_list.append(eta_anal)
    
    ax2.plot(voltages, np.array(eta_anal_list)*100, 'b-', linewidth=2, label='解析模型')
    ax2.plot(voltages, np.array(eta_pred_list)*100, 'r--', linewidth=2, label='PINN 预测')
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Aperture Ratio (%)')
    ax2.set_title('稳态开口率 vs 电压')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 动态响应：接触角
    ax3 = axes[1, 0]
    times = np.linspace(0, 0.02, 100)
    V = 30.0
    theta_pred_dyn = []
    theta_anal_dyn = []
    
    for t in times:
        X = torch.tensor([[V/30, t/0.02, 0, 0.001/0.02]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X)
        theta_pred_dyn.append(pred[0, 0].item())
        theta_anal, _ = analytical.predict(V, t, V_initial=0, t_step=0.001)
        theta_anal_dyn.append(theta_anal)
    
    ax3.plot(times*1000, theta_anal_dyn, 'b-', linewidth=2, label='解析模型')
    ax3.plot(times*1000, theta_pred_dyn, 'r--', linewidth=2, label='PINN 预测')
    ax3.axvline(x=1, color='gray', linestyle=':', alpha=0.5, label='阶跃时刻')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Contact Angle (°)')
    ax3.set_title('动态响应：接触角 (0→30V)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 动态响应：开口率
    ax4 = axes[1, 1]
    eta_pred_dyn = []
    eta_anal_dyn = []
    
    for t in times:
        X = torch.tensor([[V/30, t/0.02, 0, 0.001/0.02]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X)
        eta_pred_dyn.append(pred[0, 1].item())
        _, eta_anal = analytical.predict(V, t, V_initial=0, t_step=0.001)
        eta_anal_dyn.append(eta_anal)
    
    ax4.plot(times*1000, np.array(eta_anal_dyn)*100, 'b-', linewidth=2, label='解析模型')
    ax4.plot(times*1000, np.array(eta_pred_dyn)*100, 'r--', linewidth=2, label='PINN 预测')
    ax4.axvline(x=1, color='gray', linestyle=':', alpha=0.5, label='阶跃时刻')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Aperture Ratio (%)')
    ax4.set_title('动态响应：开口率 (0→30V)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"📊 图表已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================================
# 开口率演示（解析模型）
# ============================================================================
def aperture_demo(plot: bool = False):
    """开口率演示模式（使用解析模型）"""
    from src.models.aperture_model import EnhancedApertureModel
    
    model = EnhancedApertureModel()
    
    print("=" * 60)
    print("Stage 1: 开口率模型演示（解析 + 电容反馈）")
    print("=" * 60)
    print("\n不同电压下的稳态开口率:")
    print("-" * 40)
    
    for V in [0, 5, 10, 15, 20, 25, 30]:
        result = model.predict_enhanced(V)
        print(f"  V={V:2d}V: θ={result['theta']:.1f}°, "
              f"η={result['aperture_ratio']:.3f} ({result['aperture_ratio']*100:.1f}%)")
    
    print("\n动态响应 (0→30V):")
    print("-" * 40)
    t, eta = model.aperture_step_response(V_start=0, V_end=30, duration=0.02)
    
    eta_final = eta[-1]
    t_90_idx = np.argmax(eta >= 0.9 * eta_final) if eta_final > 0 else 0
    t_90 = t[t_90_idx] * 1000
    
    print(f"  最终开口率: {eta_final:.3f}")
    print(f"  t_90 响应时间: {t_90:.1f} ms")
    print(f"  超调: {(np.max(eta) - eta_final) / max(eta_final, 1e-6) * 100:.1f}%")
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            voltages = np.linspace(0, 30, 100)
            apertures = [model.predict_enhanced(V)['aperture_ratio'] for V in voltages]
            ax1.plot(voltages, np.array(apertures)*100, 'b-', linewidth=2)
            ax1.set_xlabel('Voltage (V)')
            ax1.set_ylabel('Aperture Ratio (%)')
            ax1.set_title('Steady-State Aperture vs Voltage')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t * 1000, eta * 100, 'b-', linewidth=2)
            ax2.axhline(y=eta_final*100, color='r', linestyle='--', alpha=0.5, 
                       label=f'Final: {eta_final*100:.1f}%')
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Aperture Ratio (%)')
            ax2.set_title('Dynamic Response (0→30V)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('aperture_demo.png', dpi=150)
            print(f"\n图像已保存: aperture_demo.png")
        except ImportError:
            print("\n⚠️ matplotlib 未安装，跳过绘图")


# ============================================================================
# 主函数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="第一阶段：接触角与开口率预测训练")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3000, help="训练轮数")
    parser.add_argument("--num-samples", type=int, default=10000, help="样本数量")
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--physics-weight", type=float, default=0.1, help="物理损失权重")
    
    # 模式
    parser.add_argument("--quick-run", action="store_true", help="快速测试模式")
    parser.add_argument("--aperture-demo", action="store_true", help="开口率演示（解析模型）")
    parser.add_argument("--plot", action="store_true", help="绘制结果图")
    parser.add_argument("--eval-only", action="store_true", help="仅评估已有模型")
    
    # 输出
    parser.add_argument("--output-dir", type=str, default="outputs_contact_angle", help="输出目录")
    parser.add_argument("--model-path", type=str, default=None, help="加载模型路径")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 开口率演示模式
    if args.aperture_demo:
        aperture_demo(plot=args.plot)
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 仅评估模式
    if args.eval_only:
        if args.model_path is None:
            logger.error("请指定 --model-path")
            return
        
        model = ContactAngleNet().to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        evaluate_model(model, device)
        if args.plot:
            plot_results(model, device, save_path=os.path.join(output_dir, 'evaluation.png'))
        return
    
    # 训练
    model, history = train(
        epochs=args.epochs,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        physics_weight=args.physics_weight,
        device=device,
        output_dir=output_dir,
        quick_run=args.quick_run
    )
    
    # 评估
    evaluate_model(model, device)
    
    # 绘图
    if args.plot:
        plot_results(model, device, save_path=os.path.join(output_dir, 'results.png'))
    
    logger.info("✨ 完成!")


if __name__ == "__main__":
    main()
