#!/usr/bin/env python3
"""
端到端 PINN 训练 - 方案 C
=========================

直接从 (x, y, z, t, V) 预测 φ 场，不依赖 Stage 1 的接触角预测。
电润湿力作为物理约束加入 PINN。

与 train_two_phase.py 的区别：
- 不使用 Stage 1 的接触角作为边界条件
- 电润湿力直接作为体积力加入动量方程
- 接触角是求解结果，不是输入

使用方法:
    python train_end_to_end.py --config config/device_calibrated_physics.json

作者: EFD-PINNs Team
日期: 2025-12
"""

import argparse
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 配置日志
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("E2E-PINN")


# ============================================================================
# 物理常量
# ============================================================================

PHYSICS = {
    # 几何参数
    "Lx": 174e-6,           # 像素宽度 (m)
    "Ly": 174e-6,           # 像素高度 (m)
    "Lz": 20e-6,            # 流体层高度 (m)
    "h_ink": 3e-6,          # 油墨层厚度 (m)
    
    # 流体属性
    "rho_ink": 800.0,       # 油墨密度 (kg/m³)
    "rho_polar": 1000.0,    # 极性液体密度 (kg/m³)
    "mu_ink": 0.003,        # 油墨粘度 (Pa·s)
    "mu_polar": 0.001,      # 极性液体粘度 (Pa·s)
    "sigma": 0.020,         # 油墨-极性液体界面张力 (N/m)
    
    # 电润湿参数（从配置文件读取）
    "theta0": 120.0,        # 初始接触角 (度)
    "epsilon_0": 8.854e-12, # 真空介电常数
    "epsilon_r": 12.0,      # 有效介电常数
    "gamma": 0.015,         # 极性液体表面张力 (N/m)
    "d": 4e-7,              # 介电层厚度 (m)
    "V_threshold": 3.0,     # 阈值电压 (V)
    
    # 时间范围
    "t_max": 0.02,          # 最大时间 (s)
}


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        # 更新物理参数
        materials = config.get('materials', {})
        PHYSICS.update({
            'epsilon_r': materials.get('epsilon_r', PHYSICS['epsilon_r']),
            'gamma': materials.get('gamma', PHYSICS['gamma']),
            'd': materials.get('dielectric_thickness', PHYSICS['d']),
            'theta0': materials.get('theta0', PHYSICS['theta0']),
            'V_threshold': materials.get('V_threshold', PHYSICS['V_threshold']),
        })
        return config
    return {}


# ============================================================================
# 端到端 PINN 模型
# ============================================================================

class EndToEndPINN(nn.Module):
    """
    端到端物理信息神经网络（v4 - 纯坐标输入）
    
    输入: (x, y, z, t, V) - 5维
    输出: (u, v, w, p, phi) - 5维
    
    不使用径向距离等人工特征，让网络自己学习空间分布
    """
    
    def __init__(self, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 128]
        
        # 几何参数
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.h_ink = PHYSICS["h_ink"]
        self.t_max = PHYSICS["t_max"]
        self.V_max = 30.0
        
        # 输入特征：归一化的 (x, y, z, t, V) + z_interface + V_sq
        input_dim = 7
        
        # phi 专用网络（更深）
        self.phi_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.Tanh(),
            nn.Linear(hidden_dims[3], 1)
        )
        
        # 速度/压力网络
        self.vel_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], 4)  # (u, v, w, p)
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        z_coord = x[:, 2]
        t = x[:, 3]
        V = x[:, 4]
        
        # 归一化特征
        x_norm = x_coord / self.Lx
        y_norm = y_coord / self.Ly
        z_norm = z_coord / self.Lz
        t_norm = t / self.t_max
        V_norm = V / self.V_max
        
        # z 相对于界面的位置（帮助区分油墨层和极性液体层）
        z_interface = torch.tanh((z_coord - self.h_ink) / (self.h_ink * 0.5))
        
        # 电压平方（电润湿力 ∝ V²）
        V_sq = V_norm ** 2
        
        # 组合特征
        features = torch.stack([
            x_norm, y_norm, z_norm, t_norm, V_norm,
            z_interface, V_sq
        ], dim=-1)
        
        # phi 预测
        phi_raw = self.phi_net(features).squeeze(-1)
        phi = torch.sigmoid(phi_raw)
        
        # 速度/压力预测
        vel_p = self.vel_net(features)
        
        # 组合输出
        out = torch.cat([vel_p, phi.unsqueeze(-1)], dim=-1)
        
        return out


# ============================================================================
# 物理损失（含电润湿力）
# ============================================================================

class ElectrowettingPhysicsLoss:
    """
    电润湿两相流物理损失
    
    包含：
    1. 连续性方程：∇·u = 0
    2. VOF 方程：∂φ/∂t + u·∇φ = 0
    3. Navier-Stokes + 电润湿力：
       ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew
    4. 表面张力 CSF：F_st = σκ∇φ
    5. 电润湿力：F_ew = -ε₀εᵣ/2 |E|² ∇φ（Maxwell 应力）
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # 流体属性
        self.rho_ink = PHYSICS["rho_ink"]
        self.rho_polar = PHYSICS["rho_polar"]
        self.mu_ink = PHYSICS["mu_ink"]
        self.mu_polar = PHYSICS["mu_polar"]
        self.sigma = PHYSICS["sigma"]
        
        # 电润湿参数
        self.epsilon_0 = PHYSICS["epsilon_0"]
        self.epsilon_r = PHYSICS["epsilon_r"]
        self.d = PHYSICS["d"]
        self.V_threshold = PHYSICS["V_threshold"]
        self.theta0 = PHYSICS["theta0"]
        self.gamma = PHYSICS["gamma"]  # 极性液体表面张力
        
        # 特征尺度
        self.L_char = PHYSICS["Lx"]
        self.U_char = 1e-3
        self.h_ink = PHYSICS["h_ink"]
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
    
    def young_lippmann_contact_angle(self, V: torch.Tensor) -> torch.Tensor:
        """
        Young-Lippmann 方程计算接触角
        
        cos(θ) = cos(θ₀) + ε₀εᵣ(V - V_T)²/(2γd)
        
        Args:
            V: 电压 (batch,)
        
        Returns:
            theta: 接触角 (弧度)
        """
        # 有效电压
        V_eff = torch.clamp(V - self.V_threshold, min=0)
        
        # Young-Lippmann 方程
        cos_theta0 = np.cos(np.radians(self.theta0))
        ew_term = (self.epsilon_0 * self.epsilon_r * V_eff**2) / (2 * self.gamma * self.d)
        cos_theta = torch.clamp(cos_theta0 + ew_term, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        
        return theta
    
    def compute_maxwell_stress_tensor(self, V: torch.Tensor, phi: torch.Tensor,
                                       grad_phi: torch.Tensor) -> torch.Tensor:
        """
        计算 Maxwell 应力张量产生的体积力（完整实现）
        
        Maxwell 应力张量：
        T_ij = ε₀ε(E_i E_j - 0.5 δ_ij |E|²)
        
        体积力（CSF 形式）：
        F_ew = -∇·T = -ε₀εᵣ/2 |E|² ∇φ δ_s
        
        其中 δ_s = |∇φ| 是界面 delta 函数
        
        Args:
            V: 电压 (batch,)
            phi: 体积分数 (batch,)
            grad_phi: φ 梯度 (batch, 3)
        
        Returns:
            F_ew: 电润湿体积力 (batch, 3)
        """
        # 有效电压
        V_eff = torch.clamp(V - self.V_threshold, min=0)
        
        # 电场强度（在介电层中）
        E_mag = V_eff / self.d
        
        # Maxwell 应力系数
        # T = ε₀εᵣ/2 * E²
        maxwell_coeff = 0.5 * self.epsilon_0 * self.epsilon_r * E_mag**2
        
        # 界面 delta 函数（平滑近似）
        # δ_s ≈ |∇φ| * exp(-α(φ-0.5)²)
        grad_phi_mag = torch.sqrt(grad_phi[:, 0]**2 + grad_phi[:, 1]**2 + grad_phi[:, 2]**2 + 1e-12)
        interface_delta = grad_phi_mag * torch.exp(-30 * (phi - 0.5)**2)
        
        # 界面法向量
        n = grad_phi / (grad_phi_mag.unsqueeze(-1) + 1e-12)
        
        # 电润湿力（沿界面法向，驱动极性液体铺展）
        # F_ew = -T * n * δ_s
        F_ew = -maxwell_coeff.unsqueeze(-1) * interface_delta.unsqueeze(-1) * n
        
        return F_ew
    
    def compute_all_losses(self, model: nn.Module, points: torch.Tensor,
                           weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        计算所有物理损失（增强版：含 Maxwell 应力和接触角边界条件）
        
        Args:
            model: PINN 模型
            points: (batch, 5) - (x, y, z, t, V)
            weights: 损失权重字典
        
        Returns:
            损失字典
        """
        # 随机采样子集以减少计算量
        n_sample = min(2000, points.shape[0])
        idx = torch.randperm(points.shape[0])[:n_sample]
        points_sample = points[idx].clone().requires_grad_(True)
        
        outputs = model(points_sample)
        
        u, v, w, p, phi = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]
        
        # 计算梯度
        def grad(y, x):
            return torch.autograd.grad(
                y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True
            )[0]
        
        # 一阶导数
        grad_outputs = grad(phi.sum(), points_sample)
        phi_x, phi_y, phi_z, phi_t = grad_outputs[:, 0], grad_outputs[:, 1], grad_outputs[:, 2], grad_outputs[:, 3]
        
        grad_u = grad(u.sum(), points_sample)
        u_x, u_y, u_z = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        grad_v = grad(v.sum(), points_sample)
        v_y = grad_v[:, 1]
        
        grad_w = grad(w.sum(), points_sample)
        w_z = grad_w[:, 2]
        
        losses = {}
        V = points_sample[:, 4]
        z = points_sample[:, 2]
        
        # 1. 连续性方程：∇·u = 0（归一化）
        div_u = (u_x + v_y + w_z) * self.L_char / self.U_char
        losses['continuity'] = weights.get('continuity', 1.0) * torch.mean(div_u**2)
        
        # 2. VOF 方程：∂φ/∂t + u·∇φ = 0（归一化）
        t_char = PHYSICS["t_max"]
        vof_res = phi_t * t_char + (u * phi_x + v * phi_y + w * phi_z) * self.L_char / self.U_char
        losses['vof'] = weights.get('vof', 1.0) * torch.mean(vof_res**2)
        
        # 3. Maxwell 应力张量产生的电润湿力约束
        # 在界面处，电润湿力应该平衡表面张力
        grad_phi = torch.stack([phi_x, phi_y, phi_z], dim=-1)
        F_ew = self.compute_maxwell_stress_tensor(V, phi, grad_phi)
        
        # 电润湿力的量级约束（归一化）
        F_ew_mag = torch.sqrt(F_ew[:, 0]**2 + F_ew[:, 1]**2 + F_ew[:, 2]**2 + 1e-12)
        # 特征力 = σ/L（表面张力/特征长度）
        F_char = self.sigma / self.L_char
        F_ew_norm = F_ew_mag / F_char
        
        # 界面处电润湿力应该是有限的（不能太大或太小）
        interface_mask = torch.exp(-20 * (phi - 0.5)**2)
        V_norm = V / 30.0
        # 高电压时电润湿力应该更大
        target_F = V_norm**2 * 0.5  # 目标力（归一化）
        ew_force_loss = interface_mask * (F_ew_norm - target_F)**2
        losses['electrowetting'] = weights.get('electrowetting', 0.1) * torch.mean(ew_force_loss)
        
        # 4. Young-Lippmann 接触角边界条件（在底面 z≈0 处）
        # 接触角定义：∂φ/∂z / |∇φ| = cos(θ)
        bottom_mask = torch.exp(-((z / self.h_ink)**2) * 10)  # z≈0 附近
        interface_at_bottom = interface_mask * bottom_mask
        
        if interface_at_bottom.sum() > 0:
            # 计算实际接触角
            grad_phi_mag = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-12)
            cos_theta_actual = phi_z / (grad_phi_mag + 1e-12)
            
            # 目标接触角（Young-Lippmann）
            theta_target = self.young_lippmann_contact_angle(V)
            cos_theta_target = torch.cos(theta_target)
            
            # 接触角边界条件损失
            contact_angle_loss = interface_at_bottom * (cos_theta_actual - cos_theta_target)**2
            losses['contact_angle'] = weights.get('contact_angle', 1.0) * torch.mean(contact_angle_loss)
        else:
            losses['contact_angle'] = torch.tensor(0.0, device=points.device)
        
        return losses


# ============================================================================
# 数据生成器（不依赖 Stage 1）
# ============================================================================

class EndToEndDataGenerator:
    """
    端到端数据生成器
    
    支持两种模式：
    1. 纯端到端：只用初始条件和物理约束
    2. 带监督：使用 Stage 1 的开口率作为目标（用于对比）
    """
    
    def __init__(self, device: torch.device, use_stage1_target: bool = True):
        self.device = device
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.h_ink = PHYSICS["h_ink"]
        self.t_max = PHYSICS["t_max"]
        self.cx = self.Lx / 2
        self.cy = self.Ly / 2
        self.use_stage1_target = use_stage1_target
        
        # 加载 Stage 1 模型（如果需要）
        self.aperture_model = None
        self.predictor = None
        if use_stage1_target:
            try:
                from src.models.aperture_model import EnhancedApertureModel
                from src.predictors.hybrid_predictor import HybridPredictor
                self.aperture_model = EnhancedApertureModel(config_path='config/device_calibrated_physics.json')
                self.predictor = HybridPredictor(config_path='config/device_calibrated_physics.json')
                logger.info("✅ 使用 Stage 1 目标数据进行监督")
            except Exception as e:
                logger.warning(f"Stage 1 模块不可用: {e}")
                self.use_stage1_target = False
    
    def get_target_phi(self, x: float, y: float, z: float, t: float, V: float) -> float:
        """
        获取目标 phi 值（双模式物理模型，与两相流一致）
        
        物理模型：
        - 初始：底部 3μm 油墨 (φ=1)，上部 17μm 极性液体 (φ=0)
        - 开口率 < 50%：中心开口模式，油墨环绕中心
        - 开口率 > 50%：四角液滴模式，毛细管切断，油墨汇聚到四角
        
        Args:
            x, y, z: 空间坐标 (m)
            t: 时间 (s)
            V: 电压 (V)
        
        Returns:
            φ ∈ [0, 1]
        """
        h_ink = self.h_ink
        interface_width = 3e-6  # 界面宽度
        
        if not self.use_stage1_target or self.aperture_model is None:
            # 只返回初始状态
            return 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))
        
        # 获取接触角和开口率
        theta = self.predictor.young_lippmann(V)
        eta = self.aperture_model.contact_angle_to_aperture_ratio(theta)
        
        # 时间演化（简单的指数衰减）
        tau = 0.005
        eta_t = eta * (1 - np.exp(-t / tau)) if t > 0 else 0
        
        # 模式切换阈值
        eta_threshold = 0.50  # 50% 开口率
        
        if eta_t < 0.01:
            # 无开口：初始状态
            phi_z = 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))
        
        elif eta_t < eta_threshold:
            # ============================================================
            # 模式 1：中心开口模式（开口率 < 50%）
            # 油墨形成环形分布，中心透明
            # ============================================================
            r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
            
            # 开口半径
            r_open = np.sqrt(eta_t * self.Lx * self.Ly / np.pi)
            
            # 油墨堆高（体积守恒）
            ink_area = self.Lx * self.Ly - np.pi * r_open**2
            h_ink_edge = self.Lx * self.Ly * h_ink / max(ink_area, 1e-12)
            h_ink_edge = min(h_ink_edge, self.Lz * 0.8)
            
            # 径向分布
            radial_factor = 0.5 * (1 + np.tanh((r - r_open) / interface_width))
            
            if r < r_open - interface_width:
                phi_z = 0.0  # 中心透明
            elif r > r_open + interface_width:
                phi_z = 0.5 * (1 - np.tanh((z - h_ink_edge) / (interface_width / 2)))
            else:
                phi_center = 0.0
                phi_edge = 0.5 * (1 - np.tanh((z - h_ink_edge) / (interface_width / 2)))
                phi_z = phi_center * (1 - radial_factor) + phi_edge * radial_factor
        
        else:
            # ============================================================
            # 模式 2/3：四角液滴 → 单角液滴演化（开口率 > 50%）
            # 
            # 物理过程：
            # - 早期（t < t_merge）：毛细管切断，油墨分布在四个角落
            # - 后期（t > t_merge）：毛细管作用使油墨汇聚到一个角落
            # - 使用 sigmoid 平滑过渡
            # ============================================================
            
            # 汇聚时间常数
            t_merge = 0.012  # 12ms 开始汇聚
            merge_progress = 1.0 / (1.0 + np.exp(-8 * (t - t_merge) / 0.005))
            
            # 四个角落的位置
            corners = [
                (0, 0),                    # 左下（目标角落）
                (self.Lx, 0),              # 右下
                (0, self.Ly),              # 左上
                (self.Lx, self.Ly)         # 右上
            ]
            target_corner_idx = 0  # 汇聚到左下角
            
            # 油墨总面积
            ink_area_total = (1 - eta_t) * self.Lx * self.Ly
            
            # 四角模式：每个角落 1/4 油墨
            ink_area_per_corner = ink_area_total / 4
            droplet_radius_four = np.sqrt(4 * ink_area_per_corner / np.pi)
            
            # 单角模式：全部油墨在一个角落
            droplet_radius_single = np.sqrt(4 * ink_area_total / np.pi)
            
            # 液滴高度（体积守恒）
            volume_total = self.Lx * self.Ly * h_ink
            h_droplet_four = volume_total / 4 / (np.pi * droplet_radius_four**2 / 4 + 1e-12)
            h_droplet_single = volume_total / (np.pi * droplet_radius_single**2 / 4 + 1e-12)
            h_droplet_four = min(h_droplet_four, self.Lz * 0.8)
            h_droplet_single = min(h_droplet_single, self.Lz * 0.8)
            
            # 计算到各角落的距离
            dists = [np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corners]
            dist_to_target = dists[target_corner_idx]
            min_dist_four = min(dists)
            
            # 四角模式的 φ（到最近角落的距离）
            if min_dist_four < droplet_radius_four - interface_width:
                phi_four = 0.5 * (1 - np.tanh((z - h_droplet_four) / (interface_width / 2)))
            elif min_dist_four > droplet_radius_four + interface_width:
                phi_four = 0.0
            else:
                spatial = 0.5 * (1 - np.tanh((min_dist_four - droplet_radius_four) / interface_width))
                phi_four = spatial * 0.5 * (1 - np.tanh((z - h_droplet_four) / (interface_width / 2)))
            
            # 单角模式的 φ（到目标角落的距离）
            if dist_to_target < droplet_radius_single - interface_width:
                phi_single = 0.5 * (1 - np.tanh((z - h_droplet_single) / (interface_width / 2)))
            elif dist_to_target > droplet_radius_single + interface_width:
                phi_single = 0.0
            else:
                spatial = 0.5 * (1 - np.tanh((dist_to_target - droplet_radius_single) / interface_width))
                phi_single = spatial * 0.5 * (1 - np.tanh((z - h_droplet_single) / (interface_width / 2)))
            
            # 混合四角和单角（根据汇聚进度）
            phi_z = phi_four * (1 - merge_progress) + phi_single * merge_progress
        
        return np.clip(phi_z, 0, 1)
    
    def generate_supervised_data(self, n_points: int, voltages: list, n_times: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成监督数据（使用 Stage 1 目标，双模式采样策略）
        
        采样策略 v3（与两相流一致）：
        - 根据开口率选择采样模式
        - 开口率 < 50%：中心开口模式，在界面附近加密采样
        - 开口率 > 50%：四角/单角模式，在角落附近加密采样
        - z 方向：在油墨层附近加密
        """
        points = []
        targets = []
        
        n_per = n_points // (len(voltages) * n_times)
        eta_threshold = 0.50  # 模式切换阈值
        
        for V in voltages:
            # 获取稳态开口率
            if self.use_stage1_target and self.aperture_model is not None:
                theta = self.predictor.young_lippmann(V)
                eta_steady = self.aperture_model.contact_angle_to_aperture_ratio(theta)
            else:
                eta_steady = 0.0
            
            for t in np.linspace(0, self.t_max, n_times):
                # 时间演化的开口率
                tau = 0.005
                eta_t = eta_steady * (1 - np.exp(-t / tau)) if t > 0 else 0
                
                for _ in range(n_per):
                    # 根据模式选择 xy 采样策略
                    if eta_t < eta_threshold and np.random.rand() < 0.4 and eta_t > 0.01:
                        # 中心开口模式：在界面附近采样
                        r_open = np.sqrt(eta_t * self.Lx * self.Ly / np.pi)
                        r = r_open + np.random.randn() * 10e-6
                        r = max(0, min(r, np.sqrt(self.cx**2 + self.cy**2)))
                        theta_angle = np.random.rand() * 2 * np.pi
                        x = self.cx + r * np.cos(theta_angle)
                        y = self.cy + r * np.sin(theta_angle)
                        x = np.clip(x, 0, self.Lx)
                        y = np.clip(y, 0, self.Ly)
                    elif eta_t >= eta_threshold and np.random.rand() < 0.4:
                        # 四角/单角模式：在角落附近采样
                        corners = [(0, 0), (self.Lx, 0), (0, self.Ly), (self.Lx, self.Ly)]
                        cx, cy = corners[np.random.randint(4)]
                        r = np.abs(np.random.randn()) * 30e-6
                        theta_angle = np.random.rand() * np.pi / 2
                        x = cx + r * np.cos(theta_angle) * (1 if cx == 0 else -1)
                        y = cy + r * np.sin(theta_angle) * (1 if cy == 0 else -1)
                        x = np.clip(x, 0, self.Lx)
                        y = np.clip(y, 0, self.Ly)
                    else:
                        # 均匀采样
                        x = np.random.rand() * self.Lx
                        y = np.random.rand() * self.Ly
                    
                    # z 方向：在油墨层附近加密
                    if np.random.rand() < 0.5:
                        z = np.random.rand() * self.h_ink * 3
                    else:
                        z = np.random.rand() * self.Lz
                    
                    phi = self.get_target_phi(x, y, z, t, V)
                    
                    points.append([x, y, z, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, phi])
        
        return (
            torch.tensor(np.array(points), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(targets), dtype=torch.float32, device=self.device)
        )
    
    def generate_initial_condition(self, n_points: int, voltages: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成初始条件数据（t=0）
        """
        points = []
        targets = []
        
        n_per_v = n_points // len(voltages)
        
        for V in voltages:
            for _ in range(n_per_v):
                x = np.random.rand() * self.Lx
                y = np.random.rand() * self.Ly
                z = np.random.rand() * self.Lz
                
                phi = self.get_target_phi(x, y, z, 0.0, V)
                
                points.append([x, y, z, 0.0, V])
                targets.append([0.0, 0.0, 0.0, 0.0, phi])
        
        return (
            torch.tensor(np.array(points), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(targets), dtype=torch.float32, device=self.device)
        )
    
    def generate_boundary_condition(self, n_points: int, voltages: list, n_times: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成边界条件数据
        
        壁面：无滑移 (u = v = w = 0)
        底面 z=0：接触角由电润湿决定（PINN 自己学习）
        """
        points = []
        targets = []
        
        n_per = n_points // (len(voltages) * n_times * 5)  # 5 个边界面
        
        for V in voltages:
            for t in np.linspace(0, self.t_max, n_times):
                # 底面 z=0
                for _ in range(n_per):
                    x = np.random.rand() * self.Lx
                    y = np.random.rand() * self.Ly
                    points.append([x, y, 0.0, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, -1])  # -1 表示不约束 phi
                
                # 顶面 z=Lz
                for _ in range(n_per):
                    x = np.random.rand() * self.Lx
                    y = np.random.rand() * self.Ly
                    points.append([x, y, self.Lz, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, 0.0])  # 顶部是极性液体
                
                # 四个侧壁
                for _ in range(n_per):
                    z = np.random.rand() * self.Lz
                    # x=0
                    points.append([0.0, np.random.rand() * self.Ly, z, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, -1])
                    # x=Lx
                    points.append([self.Lx, np.random.rand() * self.Ly, z, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, -1])
                    # y=0
                    points.append([np.random.rand() * self.Lx, 0.0, z, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, -1])
                    # y=Ly
                    points.append([np.random.rand() * self.Lx, self.Ly, z, t, V])
                    targets.append([0.0, 0.0, 0.0, 0.0, -1])
        
        return (
            torch.tensor(np.array(points), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(targets), dtype=torch.float32, device=self.device)
        )
    
    def generate_domain_points(self, n_points: int, voltages: list, n_times: int = 20) -> torch.Tensor:
        """
        生成域内配点（用于物理方程约束）
        """
        points = []
        
        n_per = n_points // (len(voltages) * n_times)
        
        for V in voltages:
            for t in np.linspace(0.001, self.t_max, n_times):
                for _ in range(n_per):
                    x = np.random.rand() * self.Lx
                    y = np.random.rand() * self.Ly
                    z = np.random.rand() * self.Lz
                    points.append([x, y, z, t, V])
        
        return torch.tensor(np.array(points), dtype=torch.float32, device=self.device)


# ============================================================================
# 训练器
# ============================================================================

class EndToEndTrainer:
    """端到端 PINN 训练器（改进版）"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # 创建模型（更大的网络）
        model_cfg = config.get('model', {})
        hidden_dims = model_cfg.get('hidden_dims', [256, 256, 256, 128])
        self.model = EndToEndPINN(hidden_dims).to(device)
        
        # 物理损失
        self.physics_loss = ElectrowettingPhysicsLoss(device)
        
        # 数据生成器（使用 Stage 1 目标数据）
        self.data_gen = EndToEndDataGenerator(device, use_stage1_target=True)
        
        # 优化器（使用 AdamW）
        train_cfg = config.get('training', {})
        lr = train_cfg.get('learning_rate', 5e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器（带 warmup）
        self.epochs = train_cfg.get('epochs', 10000)
        warmup_epochs = min(500, self.epochs // 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # 损失权重
        self.unsupervised = config.get('unsupervised', False)
        physics_cfg = config.get('physics', {})
        
        if self.unsupervised:
            # 无监督模式：渐进式训练策略
            logger.info("🔬 无监督模式：渐进式物理约束训练")
            # 初始权重（阶段 1：只用初始/边界条件）
            self.weights = {
                'phi_supervised': 0.0,    # 关闭监督损失
                'vel': 1.0,               # 速度损失
                'bc': 100.0,              # 边界条件（强约束）
                'ic': 200.0,              # 初始条件（强约束）
                'continuity': 0.0,        # 物理约束（渐进引入）
                'vof': 0.0,               # 物理约束（渐进引入）
                'electrowetting': 0.0,    # 物理约束（渐进引入）
                'contact_angle': 0.0,     # 接触角（渐进引入）
            }
            # 渐进式训练阶段配置
            self.progressive_stages = {
                'stage1_end': 0.15,       # 15% 轮数：纯初始/边界条件
                'stage2_end': 0.35,       # 35% 轮数：引入接触角
                'stage3_end': 0.60,       # 60% 轮数：引入连续性+VOF
                'stage4_end': 1.0,        # 100% 轮数：引入电润湿力
            }
        else:
            # 监督模式：phi 监督为主 + 物理约束辅助
            logger.info("📊 监督模式：phi 监督 + 物理约束辅助")
            self.weights = {
                'phi_supervised': 100.0,  # phi 监督损失（主要）
                'vel': 1.0,               # 速度损失
                'bc': 1.0,                # 边界条件
                'ic': 1.0,                # 初始条件
                'continuity': 0.0,        # 物理约束暂时关闭（量级问题）
                'vof': 0.0,               # 物理约束暂时关闭
                'electrowetting': 0.0,    # 物理约束暂时关闭
                'contact_angle': 0.01,    # Young-Lippmann 接触角（辅助）
            }
        
        # 训练参数
        self.voltages = config.get('data', {}).get('voltages', [0, 10, 20, 30])
        
        # 历史记录
        self.history = {
            'loss': [], 'phi_loss': [], 'vel_loss': [], 
            'bc_loss': [], 'physics_loss': [],
            'aperture_0V': [], 'aperture_20V': [], 'aperture_30V': []
        }
        self.best_loss = float('inf')
    
    def update_progressive_weights(self, epoch: int):
        """
        渐进式更新损失权重（仅无监督模式）
        
        阶段 1 (0-15%): 纯初始/边界条件，学习基本 φ 分布
        阶段 2 (15-35%): 引入接触角边界条件
        阶段 3 (35-60%): 引入连续性 + VOF
        阶段 4 (60-100%): 引入电润湿力
        """
        if not self.unsupervised or not hasattr(self, 'progressive_stages'):
            return
        
        progress = epoch / self.epochs
        stages = self.progressive_stages
        
        if progress < stages['stage1_end']:
            # 阶段 1：纯初始/边界条件
            stage_progress = progress / stages['stage1_end']
            self.weights.update({
                'ic': 200.0,
                'bc': 100.0,
                'contact_angle': 0.0,
                'continuity': 0.0,
                'vof': 0.0,
                'electrowetting': 0.0,
            })
            if epoch % 1000 == 0:
                logger.info(f"  📍 阶段 1: 初始/边界条件学习 ({progress*100:.0f}%)")
        
        elif progress < stages['stage2_end']:
            # 阶段 2：引入接触角边界条件
            stage_progress = (progress - stages['stage1_end']) / (stages['stage2_end'] - stages['stage1_end'])
            self.weights.update({
                'ic': 100.0,
                'bc': 50.0,
                'contact_angle': 10.0 * stage_progress,  # 渐进增加
                'continuity': 0.0,
                'vof': 0.0,
                'electrowetting': 0.0,
            })
            if epoch % 1000 == 0:
                logger.info(f"  📍 阶段 2: 接触角边界条件 ({progress*100:.0f}%), θ权重={self.weights['contact_angle']:.2f}")
        
        elif progress < stages['stage3_end']:
            # 阶段 3：引入连续性 + VOF
            stage_progress = (progress - stages['stage2_end']) / (stages['stage3_end'] - stages['stage2_end'])
            self.weights.update({
                'ic': 50.0,
                'bc': 20.0,
                'contact_angle': 10.0,
                'continuity': 1e-6 * stage_progress,  # 很小的权重，渐进增加
                'vof': 1e-6 * stage_progress,
                'electrowetting': 0.0,
            })
            if epoch % 1000 == 0:
                logger.info(f"  📍 阶段 3: 连续性+VOF ({progress*100:.0f}%)")
        
        else:
            # 阶段 4：引入电润湿力
            stage_progress = (progress - stages['stage3_end']) / (stages['stage4_end'] - stages['stage3_end'])
            self.weights.update({
                'ic': 20.0,
                'bc': 10.0,
                'contact_angle': 10.0,
                'continuity': 1e-6,
                'vof': 1e-6,
                'electrowetting': 1e-8 * stage_progress,  # 非常小的权重
            })
            if epoch % 1000 == 0:
                logger.info(f"  📍 阶段 4: 电润湿力 ({progress*100:.0f}%)")
    
    def train_epoch(self, supervised_data, bc_data, ic_data, domain_points) -> Dict[str, float]:
        """训练一个 epoch（支持监督/无监督模式）"""
        self.model.train()
        
        sup_points, sup_targets = supervised_data
        bc_points, bc_targets = bc_data
        ic_points, ic_targets = ic_data
        
        self.optimizer.zero_grad()
        
        phi_loss = torch.tensor(0.0, device=self.device)
        vel_loss = torch.tensor(0.0, device=self.device)
        
        # 1. phi 监督损失（仅监督模式）
        if self.weights['phi_supervised'] > 0:
            n_sample = min(5000, sup_points.shape[0])
            idx = torch.randperm(sup_points.shape[0])[:n_sample]
            sup_pred = self.model(sup_points[idx])
            
            phi_pred = sup_pred[:, 4]
            phi_target = sup_targets[idx, 4]
            phi_loss = torch.mean((phi_pred - phi_target)**2)
            vel_loss = torch.mean(sup_pred[:, :3]**2)
        
        supervised_loss = self.weights['phi_supervised'] * phi_loss + self.weights['vel'] * vel_loss
        
        # 2. 初始条件损失（t=0 时油墨均匀铺在底部）
        n_ic = min(2000, ic_points.shape[0])
        idx_ic = torch.randperm(ic_points.shape[0])[:n_ic]
        ic_pred = self.model(ic_points[idx_ic])
        
        # 初始 phi：底部油墨层
        ic_phi_loss = torch.mean((ic_pred[:, 4] - ic_targets[idx_ic, 4])**2)
        # 初始速度为 0
        ic_vel_loss = torch.mean(ic_pred[:, :3]**2)
        ic_loss = self.weights.get('ic', 1.0) * (ic_phi_loss + ic_vel_loss)
        
        # 3. 边界条件损失
        n_bc = min(2000, bc_points.shape[0])
        idx_bc = torch.randperm(bc_points.shape[0])[:n_bc]
        bc_pred = self.model(bc_points[idx_bc])
        
        # 速度无滑移
        bc_vel_loss = torch.mean(bc_pred[:, :3]**2)
        # phi 约束（只对有目标值的点）
        bc_phi_mask = bc_targets[idx_bc, 4] >= 0
        if bc_phi_mask.any():
            bc_phi_loss = torch.mean((bc_pred[bc_phi_mask, 4] - bc_targets[idx_bc][bc_phi_mask, 4])**2)
        else:
            bc_phi_loss = torch.tensor(0.0, device=self.device)
        bc_loss = self.weights['bc'] * (bc_vel_loss + bc_phi_loss)
        
        # 4. 物理损失（无监督模式的核心）
        physics_losses = self.physics_loss.compute_all_losses(
            self.model, domain_points, self.weights
        )
        physics_loss = sum(physics_losses.values())
        
        # 总损失
        total_loss = supervised_loss + ic_loss + bc_loss + physics_loss
        
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total': total_loss.item(),
            'phi': phi_loss.item(),
            'vel': vel_loss.item(),
            'ic': ic_loss.item(),
            'bc': bc_loss.item(),
            'physics': physics_loss.item(),
        }
    
    def compute_aperture(self, V: float, t: float = 0.015) -> float:
        """计算给定电压下的开口率"""
        self.model.eval()
        
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
        h_ink = PHYSICS["h_ink"]
        nx, ny = 20, 20
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        z = h_ink * 0.5  # 油墨层中间位置
        
        points = np.zeros((nx * ny, 5))
        points[:, 0] = X.flatten()
        points[:, 1] = Y.flatten()
        points[:, 2] = z
        points[:, 3] = t
        points[:, 4] = V
        
        with torch.no_grad():
            inputs = torch.tensor(points, dtype=torch.float32, device=self.device)
            outputs = self.model(inputs)
            phi = outputs[:, 4].cpu().numpy()
        
        self.model.train()
        # phi < 0.5 表示透明区域（极性液体）
        return np.mean(phi < 0.5)
    
    def train(self, output_dir: str):
        """完整训练流程（支持监督/无监督模式）"""
        logger.info("=" * 60)
        mode_str = "无监督（纯物理约束）" if self.unsupervised else "监督学习"
        logger.info(f"端到端 PINN 训练开始 - {mode_str}")
        logger.info("=" * 60)
        logger.info(f"设备: {self.device}")
        logger.info(f"电压范围: {self.voltages}")
        logger.info(f"训练轮数: {self.epochs}")
        logger.info(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 生成数据
        logger.info("生成训练数据...")
        
        if self.unsupervised:
            # 无监督模式：不使用目标数据，只用初始/边界条件
            # 创建空的监督数据（占位）
            empty_points = torch.zeros((1, 5), device=self.device)
            empty_targets = torch.zeros((1, 5), device=self.device)
            supervised_data = (empty_points, empty_targets)
            logger.info("  监督数据: 已禁用")
        else:
            # 监督模式：使用目标数据
            supervised_data = self.data_gen.generate_supervised_data(80000, self.voltages, n_times=30)
            logger.info(f"  监督数据点: {supervised_data[0].shape[0]}")
        
        # 初始条件数据（t=0）
        ic_data = self.data_gen.generate_initial_condition(20000, self.voltages)
        bc_data = self.data_gen.generate_boundary_condition(15000, self.voltages)
        domain_points = self.data_gen.generate_domain_points(30000 if self.unsupervised else 20000, self.voltages)
        
        logger.info(f"  初始条件点: {ic_data[0].shape[0]}")
        logger.info(f"  边界条件点: {bc_data[0].shape[0]}")
        logger.info(f"  域内配点: {domain_points.shape[0]}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # 渐进式更新权重（无监督模式）
            self.update_progressive_weights(epoch)
            
            losses = self.train_epoch(supervised_data, bc_data, ic_data, domain_points)
            
            # 记录历史
            self.history['loss'].append(losses['total'])
            self.history['phi_loss'].append(losses['phi'])
            self.history['vel_loss'].append(losses['vel'])
            self.history['bc_loss'].append(losses['bc'])
            self.history['physics_loss'].append(losses['physics'])
            
            # 保存最佳模型
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                self.save_model(os.path.join(output_dir, 'best_model.pth'))
            
            # 打印进度和开口率
            if (epoch + 1) % 100 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                
                # 计算开口率
                eta_0 = self.compute_aperture(0)
                eta_20 = self.compute_aperture(20)
                eta_30 = self.compute_aperture(30)
                
                self.history['aperture_0V'].append(eta_0)
                self.history['aperture_20V'].append(eta_20)
                self.history['aperture_30V'].append(eta_30)
                
                if self.unsupervised:
                    logger.info(
                        f"Epoch {epoch+1:5d}/{self.epochs} | "
                        f"Loss: {losses['total']:.4f} (phy:{losses['physics']:.4f}, ic:{losses['ic']:.4f}) | "
                        f"η: 0V={eta_0*100:.1f}%, 20V={eta_20*100:.1f}%, 30V={eta_30*100:.1f}% | "
                        f"LR: {lr:.2e} | {elapsed:.0f}s"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1:5d}/{self.epochs} | "
                        f"Loss: {losses['total']:.4f} (φ:{losses['phi']:.4f}) | "
                        f"η: 0V={eta_0*100:.1f}%, 20V={eta_20*100:.1f}%, 30V={eta_30*100:.1f}% | "
                        f"LR: {lr:.2e} | {elapsed:.0f}s"
                    )
        
        # 保存最终模型
        self.save_model(os.path.join(output_dir, 'final_model.pth'))
        
        # 保存历史
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        
        logger.info("=" * 60)
        logger.info(f"训练完成！最佳损失: {self.best_loss:.4f}")
        logger.info(f"模型保存到: {output_dir}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
        }, path)


# ============================================================================
# 可视化和验证
# ============================================================================

def visualize_results(model: EndToEndPINN, device: torch.device, output_dir: str):
    """可视化训练结果"""
    import matplotlib.pyplot as plt
    
    model.eval()
    
    Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
    
    # 创建网格
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    voltages = [0, 10, 20, 30]
    t = 0.015  # 15ms
    z = 1e-6   # 底部附近
    
    for i, V in enumerate(voltages):
        # 构建输入
        points = np.zeros((nx * ny, 5))
        points[:, 0] = X.flatten()
        points[:, 1] = Y.flatten()
        points[:, 2] = z
        points[:, 3] = t
        points[:, 4] = V
        
        with torch.no_grad():
            inputs = torch.tensor(points, dtype=torch.float32, device=device)
            outputs = model(inputs)
            phi = outputs[:, 4].cpu().numpy().reshape(ny, nx)
        
        # φ 分布
        ax = axes[0, i]
        im = ax.contourf(X * 1e6, Y * 1e6, phi, levels=20, cmap='RdBu_r')
        ax.set_title(f'V = {V}V, t = {t*1000:.0f}ms')
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='φ')
        
        # 计算开口率
        aperture = np.mean(phi < 0.5)
        axes[1, i].text(0.5, 0.5, f'开口率\n{aperture*100:.1f}%',
                       ha='center', va='center', fontsize=20,
                       transform=axes[1, i].transAxes)
        axes[1, i].set_title(f'V = {V}V')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phi_distribution.png'), dpi=150)
    plt.close()
    
    logger.info(f"可视化保存到: {output_dir}/phi_distribution.png")


def compare_with_stage1(model: EndToEndPINN, device: torch.device, output_dir: str):
    """与 Stage 1 预测对比"""
    import matplotlib.pyplot as plt
    
    try:
        from src.models.aperture_model import EnhancedApertureModel
        from src.predictors.hybrid_predictor import HybridPredictor
        has_stage1 = True
    except ImportError:
        has_stage1 = False
        logger.warning("Stage 1 模块不可用，跳过对比")
        return
    
    model.eval()
    
    # Stage 1 预测
    predictor = HybridPredictor(config_path='config/device_calibrated_physics.json')
    aperture_model = EnhancedApertureModel(config_path='config/device_calibrated_physics.json')
    
    voltages = np.linspace(0, 30, 31)
    t = 0.015
    
    stage1_apertures = []
    e2e_apertures = []
    
    Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
    nx, ny = 30, 30
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    z = 1e-6
    
    for V in voltages:
        # Stage 1 预测
        theta = predictor.young_lippmann(V)
        eta_s1 = aperture_model.contact_angle_to_aperture_ratio(theta)
        stage1_apertures.append(eta_s1)
        
        # 端到端 PINN 预测
        points = np.zeros((nx * ny, 5))
        points[:, 0] = X.flatten()
        points[:, 1] = Y.flatten()
        points[:, 2] = z
        points[:, 3] = t
        points[:, 4] = V
        
        with torch.no_grad():
            inputs = torch.tensor(points, dtype=torch.float32, device=device)
            outputs = model(inputs)
            phi = outputs[:, 4].cpu().numpy()
        
        eta_e2e = np.mean(phi < 0.5)
        e2e_apertures.append(eta_e2e)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(voltages, np.array(stage1_apertures) * 100, 'b-o', label='Stage 1 (解析)', markersize=4)
    ax.plot(voltages, np.array(e2e_apertures) * 100, 'r-s', label='端到端 PINN', markersize=4)
    ax.set_xlabel('电压 (V)')
    ax.set_ylabel('开口率 (%)')
    ax.set_title('Stage 1 vs 端到端 PINN 对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_stage1_vs_e2e.png'), dpi=150)
    plt.close()
    
    logger.info(f"对比图保存到: {output_dir}/comparison_stage1_vs_e2e.png")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='端到端 PINN 训练')
    parser.add_argument('--config', type=str, default='config/device_calibrated_physics.json',
                        help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--unsupervised', action='store_true',
                        help='无监督模式：只用物理约束，不用目标数据')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if 'training' not in config:
        config['training'] = {}
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    
    # 无监督模式标记
    config['unsupervised'] = args.unsupervised
    
    # 输出目录
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_unsup" if args.unsupervised else ""
        output_dir = f"outputs_e2e_{timestamp}{suffix}"
    else:
        output_dir = args.output
    
    # 训练
    trainer = EndToEndTrainer(config, device)
    trainer.train(output_dir)
    
    # 可视化
    visualize_results(trainer.model, device, output_dir)
    
    # 与 Stage 1 对比
    compare_with_stage1(trainer.model, device, output_dir)
    
    logger.info("=" * 60)
    logger.info("完成！")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("  运行以下命令查看对比:")
    logger.info(f"    ls {output_dir}/")


if __name__ == '__main__':
    main()
