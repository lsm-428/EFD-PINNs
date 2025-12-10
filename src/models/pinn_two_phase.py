#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EWP 两相流 PINN - 整合优化版
============================

整合所有最佳实践的物理信息神经网络：
1. 径向坐标转换（利用轴对称性）
2. 分离网络架构（phi 和速度场分开）
3. 完整物理损失（连续性 + VOF + Navier-Stokes）
4. 渐进式训练策略
5. 界面加密采样

物理方程：
- 连续性：∇·u = 0
- VOF：∂φ/∂t + u·∇φ = 0
- N-S：ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u

作者: EFD-PINNs Team
日期: 2024-12
"""

import argparse
import datetime
import json
import logging
import os
import random
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    """设置全局随机种子，确保训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 操作确定性（可能略微降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置日志
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("EWP-PINN")

# 导入项目模块
try:
    from src.models.aperture_model import EnhancedApertureModel
    HAS_APERTURE = True
except ImportError:
    HAS_APERTURE = False
    logger.warning("EnhancedApertureModel 不可用，使用解析公式")

try:
    from src.predictors.hybrid_predictor import HybridPredictor
    HAS_HYBRID_PREDICTOR = True
except ImportError:
    HAS_HYBRID_PREDICTOR = False
    logger.warning("HybridPredictor 不可用，使用解析公式计算接触角")


# ============================================================================
# 物理常量
# ============================================================================

PHYSICS = {
    # 几何参数 - 与 Stage 1 配置同步
    "Lx": 174e-6,           # 像素宽度 (m)
    "Ly": 174e-6,           # 像素高度 (m)
    "Lz": 20e-6,            # 围堰高度 (m) = 3μm油墨 + 17μm极性液体
    "h_ink": 3e-6,          # 油墨层厚度 (m)
    "h_polar": 17e-6,       # 极性液体层厚度 (m)
    
    # 流体属性
    "rho_oil": 800.0,       # 油墨密度 (kg/m³)
    "rho_polar": 1000.0,    # 极性液体密度 (kg/m³)
    "mu_oil": 0.003,        # 油墨粘度 (Pa·s)
    "mu_polar": 0.001,      # 极性液体粘度 (Pa·s)
    "sigma": 0.045,         # 油墨-极性液体界面张力 (N/m)
    
    # 电润湿参数 - 与 Stage 1 配置同步 (config/stage6_wall_effect.json)
    "theta0": 120.0,        # 初始接触角 (度)
    "gamma": 0.050,         # 极性液体表面张力 (N/m) - 乙二醇混合液
    "epsilon_r": 3.0,       # SU-8 相对介电常数
    "epsilon_h": 1.9,       # Teflon 相对介电常数
    "d_dielectric": 4e-7,   # SU-8 介电层厚度 (m) = 400nm
    "d_hydrophobic": 4e-7,  # Teflon 疏水层厚度 (m) = 400nm
    "epsilon_0": 8.854e-12, # 真空介电常数
    
    # 动力学参数 - 与 Stage 1 配置同步
    "tau": 0.005,           # 响应时间常数 (s)
    "zeta": 0.8,            # 阻尼比
    
    # 时间范围
    "t_max": 0.02,          # 最大时间 (s)
    
    # 电压阈值 - 与 Stage 1 配置同步
    "V_threshold": 3.0,     # 开始响应的阈值电压 (V)
}

# ============================================================================
# φ 场定义（标准 VOF）
# ============================================================================
# φ = 1: 纯油墨
# φ = 0: 纯极性液体（透明）
# 0 < φ < 1: 界面过渡区
#
# 初始状态 (t=0, V=0):
#   z < h_ink (3μm): φ = 1 (油墨层)
#   z > h_ink: φ = 0 (极性液体层)
#
# 电压响应后:
#   中心区域: φ → 0 (透明，极性液体下沉到基底)
#   边缘区域: φ = 1 且油墨堆高 (体积守恒)
# ============================================================================


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONFIG = {
    "model": {
        "hidden_phi": [64, 64, 64, 32],    # phi 网络
        "hidden_vel": [64, 64, 32],         # 速度网络
    },
    "training": {
        "epochs": 30000,
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "min_lr": 1e-6,
        "gradient_clip": 1.0,
        # 渐进式训练阶段
        "stage1_epochs": 5000,    # 纯数据学习
        "stage2_epochs": 15000,   # 引入连续性+VOF
        "stage3_epochs": 30000,   # 完整物理约束
    },
    "physics": {
        # 数据损失权重
        "interface_weight": 500.0,
        "ic_weight": 100.0,
        "bc_weight": 50.0,
        # 物理损失权重（使用 log1p 缩放后的权重）
        "continuity_weight": 0.5,      # 连续性方程
        "vof_weight": 0.5,             # VOF 方程
        "ns_weight": 0.1,              # Navier-Stokes
        "surface_tension_weight": 0.01, # 表面张力 CSF
    },
    "data": {
        "n_interface": 100000,
        "n_initial": 10000,
        "n_boundary": 10000,
        "n_domain": 20000,
        "voltages": [0, 5, 10, 15, 20, 25, 30],
        "times": 30,
    },
}


# ============================================================================
# 两相流 PINN 模型
# ============================================================================

class TwoPhasePINN(nn.Module):
    """
    两相流物理信息神经网络
    
    特点：
    - 直接使用 (x, y, z, t, V) 作为输入
    - 分离 phi 网络和速度网络
    - 输入: (x, y, z, t, V)
    - 输出: (u, v, w, p, phi)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or DEFAULT_CONFIG
        
        # 几何参数
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.t_max = PHYSICS["t_max"]
        self.cx = self.Lx / 2
        self.cy = self.Ly / 2
        self.r_max = np.sqrt(self.cx**2 + self.cy**2)
        
        # 网络配置
        model_cfg = config.get("model", {})
        hidden_phi = model_cfg.get("hidden_phi", [128, 128, 64, 32])
        hidden_vel = model_cfg.get("hidden_vel", [64, 64, 32])
        
        # Phi 网络：输入 (x_norm, y_norm, z_norm, t_norm, V_norm) - 5维
        self.phi_net = self._build_network(5, 1, hidden_phi)
        
        # 速度网络：输入 (x_norm, y_norm, z_norm, t_norm, V_norm, phi) - 6维
        self.vel_net = self._build_network(6, 4, hidden_vel)
        
        self.apply(self._init_weights)
    
    def _build_network(self, input_dim: int, output_dim: int, hidden_layers: list) -> nn.Sequential:
        """构建全连接网络"""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        """Xavier 初始化"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, 5) - (x, y, z, t, V)
        
        Returns:
            (batch, 5) - (u, v, w, p, phi)
        """
        # 提取坐标
        x_coord, y_coord, z_coord, t, V = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        
        # 归一化到 [0, 1]
        x_norm = x_coord / self.Lx
        y_norm = y_coord / self.Ly
        z_norm = z_coord / self.Lz
        t_norm = t / self.t_max
        V_norm = V / 30.0
        
        # Phi 预测 - 直接使用 (x, y, z, t, V)
        phi_input = torch.stack([x_norm, y_norm, z_norm, t_norm, V_norm], dim=-1)
        phi_raw = self.phi_net(phi_input)
        phi = torch.sigmoid(phi_raw)  # 限制在 [0, 1]
        
        # 速度预测
        vel_input = torch.stack([x_norm, y_norm, z_norm, t_norm, V_norm, phi.squeeze(-1)], dim=-1)
        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]
        
        return torch.cat([u, v, w, p, phi], dim=-1)


# ============================================================================
# 物理损失计算
# ============================================================================

class PhysicsLoss:
    """
    两相流物理损失（优化版）
    
    包含：
    - 连续性方程：∇·u = 0
    - VOF 方程：∂φ/∂t + u·∇φ = 0
    - Navier-Stokes 方程
    - 表面张力 CSF 模型：F_st = σκδ_s n
    
    优化：
    - 使用 log1p 缩放稳定大损失
    - 归一化残差
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.rho_oil = PHYSICS["rho_oil"]
        self.rho_polar = PHYSICS["rho_polar"]
        self.mu_oil = PHYSICS["mu_oil"]
        self.mu_polar = PHYSICS["mu_polar"]
        self.sigma = PHYSICS["sigma"]
        
        # 特征尺度（用于归一化）
        self.L_char = PHYSICS["Lx"]  # 特征长度
        self.U_char = 1e-3           # 特征速度 (m/s)
        self.T_char = self.L_char / self.U_char  # 特征时间
    
    def compute_gradients(self, model: nn.Module, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算所有需要的梯度"""
        points = points.clone().requires_grad_(True)
        outputs = model(points)
        
        u, v, w, p, phi = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]
        
        def grad(y, x):
            return torch.autograd.grad(
                y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True
            )[0]
        
        # 一阶导数
        grads = {"u": u, "v": v, "w": w, "p": p, "phi": phi}
        
        for name, var in [("u", u), ("v", v), ("w", w), ("p", p), ("phi", phi)]:
            g = grad(var.sum(), points)
            grads[f"{name}_x"] = g[:, 0]
            grads[f"{name}_y"] = g[:, 1]
            grads[f"{name}_z"] = g[:, 2]
            grads[f"{name}_t"] = g[:, 3]
        
        # 二阶导数（用于粘性项和曲率）
        for base in ["u", "v", "w", "phi"]:
            for coord, idx in [("x", 0), ("y", 1), ("z", 2)]:
                first = grads[f"{base}_{coord}"]
                second = grad(first.sum(), points)
                grads[f"{base}_{coord}{coord}"] = second[:, idx]
        
        return grads
    
    def continuity_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """连续性方程残差：∇·u = 0（归一化）"""
        div_u = grads["u_x"] + grads["v_y"] + grads["w_z"]
        # 归一化：除以特征速度/长度
        div_u_norm = div_u * self.L_char / self.U_char
        return torch.mean(div_u_norm**2)
    
    def vof_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VOF 方程残差：∂φ/∂t + u·∇φ = 0（归一化）"""
        u, v, w = grads["u"], grads["v"], grads["w"]
        res = grads["phi_t"] + u * grads["phi_x"] + v * grads["phi_y"] + w * grads["phi_z"]
        # 归一化：phi 是无量纲的，时间导数除以 1/T_char
        res_norm = res * self.T_char
        return torch.mean(res_norm**2)
    
    def navier_stokes_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Navier-Stokes 方程残差（归一化）"""
        u, v, w, phi = grads["u"], grads["v"], grads["w"], grads["phi"]
        
        # 混合流体属性
        rho = phi * self.rho_oil + (1 - phi) * self.rho_polar
        mu = phi * self.mu_oil + (1 - phi) * self.mu_polar
        
        # 对流项
        u_conv = u * grads["u_x"] + v * grads["u_y"] + w * grads["u_z"]
        v_conv = u * grads["v_x"] + v * grads["v_y"] + w * grads["v_z"]
        w_conv = u * grads["w_x"] + v * grads["w_y"] + w * grads["w_z"]
        
        # 粘性项
        u_visc = grads["u_xx"] + grads["u_yy"] + grads["u_zz"]
        v_visc = grads["v_xx"] + grads["v_yy"] + grads["v_zz"]
        w_visc = grads["w_xx"] + grads["w_yy"] + grads["w_zz"]
        
        # N-S 残差
        ns_u = rho * (grads["u_t"] + u_conv) + grads["p_x"] - mu * u_visc
        ns_v = rho * (grads["v_t"] + v_conv) + grads["p_y"] - mu * v_visc
        ns_w = rho * (grads["w_t"] + w_conv) + grads["p_z"] - mu * w_visc
        
        # 归一化：除以 ρU²/L
        scale = self.rho_polar * self.U_char**2 / self.L_char
        ns_u_norm = ns_u / (scale + 1e-10)
        ns_v_norm = ns_v / (scale + 1e-10)
        ns_w_norm = ns_w / (scale + 1e-10)
        
        return torch.mean(ns_u_norm**2 + ns_v_norm**2 + ns_w_norm**2)
    
    def surface_tension_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        表面张力 CSF 模型残差
        
        CSF (Continuum Surface Force):
        F_st = σκn δ_s
        
        其中：
        - κ = -∇·n 是曲率
        - n = ∇φ/|∇φ| 是界面法向量
        - δ_s = |∇φ| 是界面 delta 函数
        """
        phi = grads["phi"]
        phi_x, phi_y, phi_z = grads["phi_x"], grads["phi_y"], grads["phi_z"]
        
        # 界面法向量模
        grad_phi_mag = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-10)
        
        # 界面指示函数（在 phi=0.5 附近）
        interface_indicator = torch.exp(-50 * (phi - 0.5)**2)
        
        # 曲率（简化计算）
        phi_xx = grads["phi_xx"]
        phi_yy = grads["phi_yy"]
        phi_zz = grads["phi_zz"]
        laplacian_phi = phi_xx + phi_yy + phi_zz
        
        # 曲率 κ ≈ -∇²φ / |∇φ| (简化)
        kappa = -laplacian_phi / (grad_phi_mag + 1e-10)
        
        # 表面张力应该在界面处平衡
        # 这里我们约束界面处的曲率应该是合理的（不能太大）
        st_residual = interface_indicator * kappa**2
        
        return torch.mean(st_residual)


# ============================================================================
# 数据生成器
# ============================================================================

class DataGenerator:
    """
    训练数据生成器 - 物理正确的边界条件方式
    
    核心思想：
    - 接触角 θ(t) 是边界条件，决定了油墨在基底上的润湿行为
    - PINN 自己学习 φ 场的演化，开口率是求解后的结果
    - 不预设"开口半径"，让物理方程自己决定界面位置
    
    边界条件：
    - 底面 (z=0): 接触角边界条件，∇φ·n = |∇φ|cos(θ)
    - 侧壁: 无滑移 + 固定接触角
    - 初始条件: 油墨均匀铺在底部
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.h_ink = PHYSICS["h_ink"]
        self.t_max = PHYSICS["t_max"]
        self.cx, self.cy = self.Lx / 2, self.Ly / 2
        self.r_max = np.sqrt(self.cx**2 + self.cy**2)
        
        # 初始化 Stage 1 接触角预测器（使用校准后的配置）
        self.contact_angle_predictor = None
        if HAS_HYBRID_PREDICTOR:
            try:
                self.contact_angle_predictor = HybridPredictor(
                    config_path='config/stage6_wall_effect.json',
                    use_model_for_steady_state=False
                )
                logger.info("✅ 已集成 Stage 1 HybridPredictor 作为接触角边界条件")
            except Exception as e:
                logger.warning(f"HybridPredictor 初始化失败: {e}")
        
        # 物理参数
        self.theta0 = PHYSICS["theta0"]  # 初始接触角 120°
    
    def get_contact_angle(self, V: float, t: float) -> float:
        """
        获取动态接触角 - Stage 1 输出作为边界条件
        
        Args:
            V: 电压 (V)
            t: 时间 (s)
        
        Returns:
            接触角 θ(t) (度)
        """
        if self.contact_angle_predictor is not None:
            return self.contact_angle_predictor.predict(
                voltage=V, time=t, V_initial=0.0, t_step=0.0
            )
        else:
            return self._analytical_contact_angle(V, t)
    
    def _analytical_contact_angle(self, V: float, t: float) -> float:
        """
        内置解析公式（备用）- 含阈值电压修正
        
        Young-Lippmann 方程（含阈值电压）：
        cos(θ) = cos(θ₀) + ε₀εᵣ(V - V_T)²/(2γd)
        """
        theta0 = PHYSICS["theta0"]
        epsilon_r = PHYSICS["epsilon_r"]
        d = PHYSICS["d_dielectric"]
        gamma = PHYSICS["sigma"]
        epsilon_0 = 8.854e-12
        tau = PHYSICS["tau"]
        zeta = PHYSICS["zeta"]
        V_threshold = PHYSICS.get("V_threshold", 5.0)  # 阈值电压
        
        # 有效电压 = max(0, V - V_T)
        V_eff = max(0, V - V_threshold)
        
        cos_theta0 = np.cos(np.radians(theta0))
        ew_term = (epsilon_0 * epsilon_r * V_eff**2) / (2 * gamma * d)
        cos_theta_eq = np.clip(cos_theta0 + ew_term, -1, 1)
        theta_eq = np.degrees(np.arccos(cos_theta_eq))
        
        omega_0 = 1.0 / tau
        omega_d = omega_0 * np.sqrt(max(0, 1 - zeta**2))
        exp_term = np.exp(-zeta * omega_0 * t)
        damping = zeta / np.sqrt(1 - zeta**2) if zeta < 1 else 1.0
        theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
            np.cos(omega_d * t) + damping * np.sin(omega_d * t)
        )
        return theta_t
    
    def compute_contact_angle_gradient(self, theta_deg: float) -> Tuple[float, float]:
        """
        计算接触角对应的 φ 梯度方向
        
        接触角定义：液体内部与固体表面的夹角
        θ < 90°: 亲水（润湿）
        θ > 90°: 疏水（不润湿）
        
        在底面 z=0，法向量 n = (0, 0, 1)
        接触角边界条件：∂φ/∂z = |∇φ| * cos(θ)
        
        Returns:
            (cos_theta, sin_theta) 用于边界条件
        """
        theta_rad = np.radians(theta_deg)
        return np.cos(theta_rad), np.sin(theta_rad)
    
    def get_opening_rate(self, V: float, t: float) -> float:
        """
        计算开口率 - 使用 Stage 1 的 EnhancedApertureModel
        
        Args:
            V: 电压 (V)
            t: 时间 (s)
        
        Returns:
            开口率 η ∈ [0, 0.85]
        """
        theta = self.get_contact_angle(V, t)
        
        # 使用 Stage 1 的开口率模型（如果可用）
        if HAS_APERTURE:
            try:
                if not hasattr(self, '_aperture_model'):
                    self._aperture_model = EnhancedApertureModel(
                        config_path='config/stage6_wall_effect.json'
                    )
                return self._aperture_model.contact_angle_to_aperture_ratio(theta)
            except Exception as e:
                logger.warning(f"EnhancedApertureModel 调用失败: {e}")
        
        # 备用：简单线性映射
        theta0 = PHYSICS["theta0"]  # 120°
        theta_min = 60.0  # 最小接触角
        
        if theta >= theta0:
            eta = 0.0
        elif theta <= theta_min:
            eta = 0.85
        else:
            cos_change = np.cos(np.radians(theta)) - np.cos(np.radians(theta0))
            cos_max_change = np.cos(np.radians(theta_min)) - np.cos(np.radians(theta0))
            eta = 0.85 * cos_change / cos_max_change
        
        return np.clip(eta, 0, 0.85)
    
    def target_phi_3d(self, x: float, y: float, z: float, t: float, V: float) -> float:
        """
        计算目标 φ 值（双模式物理模型）
        
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
        eta = self.get_opening_rate(V, t)
        
        # 界面宽度
        interface_width = 3e-6  # 3 μm
        
        # 模式切换阈值
        eta_threshold = 0.50  # 50% 开口率
        
        if eta < 0.01:
            # 无开口：初始状态
            phi_z = 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))
        
        elif eta < eta_threshold:
            # ============================================================
            # 模式 1：中心开口模式（开口率 < 50%）
            # 油墨形成环形分布，中心透明
            # ============================================================
            r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
            
            # 开口半径
            r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
            
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
            ink_area_total = (1 - eta) * self.Lx * self.Ly
            
            # 四角模式：每个角落 1/4 油墨
            ink_area_per_corner = ink_area_total / 4
            droplet_radius_four = np.sqrt(4 * ink_area_per_corner / np.pi)
            
            # 单角模式：全部油墨在一个角落
            droplet_radius_single = np.sqrt(4 * ink_area_total / np.pi)
            
            # 液滴高度（体积守恒）
            # 油墨总体积 = Lx * Ly * h_ink
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
    
    def generate_all_data(self) -> Dict[str, torch.Tensor]:
        """
        生成训练数据 - 中心开口模型 + 标准 VOF 定义
        
        φ = 1: 油墨
        φ = 0: 极性液体（透明）
        """
        data_cfg = self.config.get("data", DEFAULT_CONFIG["data"])
        voltages = data_cfg.get("voltages", [0, 10, 20, 30])
        n_times = data_cfg.get("times", 20)
        
        logger.info("生成训练数据（三模式物理模型，标准 VOF）...")
        logger.info("  φ=1: 油墨, φ=0: 极性液体（透明）")
        logger.info("  模式1: 中心开口（η<50%）")
        logger.info("  模式2: 四角液滴（η>50%, 早期）")
        logger.info("  模式3: 单角液滴（η>50%, 稳态）")
        
        # ============================================================
        # 1. 界面数据（核心训练数据）
        # 在整个时空域采样，提供目标 φ 值
        # ============================================================
        n_interface = data_cfg.get("n_interface", 100000)
        interface_points = []
        interface_targets = []
        
        n_per = n_interface // (len(voltages) * n_times)
        
        for V in voltages:
            for t in np.linspace(0.001, self.t_max, n_times):
                eta = self.get_opening_rate(V, t)
                
                for _ in range(n_per):
                    # 根据模式选择采样策略
                    eta_threshold = 0.50
                    
                    if eta < eta_threshold and np.random.rand() < 0.4 and eta > 0.01:
                        # 中心开口模式：在界面附近采样
                        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
                        r = r_open + np.random.randn() * 10e-6
                        r = max(0, min(r, self.r_max))
                        theta_angle = np.random.rand() * 2 * np.pi
                        x = self.cx + r * np.cos(theta_angle)
                        y = self.cy + r * np.sin(theta_angle)
                        x = np.clip(x, 0, self.Lx)
                        y = np.clip(y, 0, self.Ly)
                    elif eta >= eta_threshold and np.random.rand() < 0.4:
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
                    
                    phi = self.target_phi_3d(x, y, z, t, V)
                    
                    interface_points.append([x, y, z, t, V])
                    interface_targets.append(phi)
        
        logger.info(f"  界面数据点: {len(interface_points)}")
        
        # ============================================================
        # 2. 初始条件：t=0 时油墨均匀铺在底部 3μm
        # φ = 1 (z < h_ink), φ = 0 (z >= h_ink)
        # ============================================================
        n_ic = data_cfg.get("n_initial", 10000)
        ic_points, ic_values = [], []
        
        n_per_v = n_ic // len(voltages)
        
        for V in voltages:
            for _ in range(n_per_v):
                x = np.random.rand() * self.Lx
                y = np.random.rand() * self.Ly
                z = np.random.rand() * self.Lz
                
                # 初始状态：油墨在底部 3μm，使用 tanh 平滑过渡
                interface_width = 1e-6
                phi = 0.5 * (1 - np.tanh((z - self.h_ink) / interface_width))
                phi = np.clip(phi, 0, 1)
                
                ic_points.append([x, y, z, 0.0, V])
                ic_values.append([0.0, 0.0, 0.0, 0.0, phi])
        
        logger.info(f"  初始条件点: {len(ic_points)}")
        
        # ============================================================
        # 3. 壁面边界条件：围堰处油墨堆高
        # ============================================================
        n_bc = data_cfg.get("n_boundary", 10000)
        bc_points, bc_values = [], []
        
        for V in voltages:
            for t in np.linspace(0, self.t_max, 5):
                n_per_bc = n_bc // (len(voltages) * 5 * 4)
                
                for boundary in ['x0', 'xL', 'y0', 'yL']:
                    for _ in range(n_per_bc):
                        if boundary == 'x0':
                            x, y = 0, np.random.rand() * self.Ly
                        elif boundary == 'xL':
                            x, y = self.Lx, np.random.rand() * self.Ly
                        elif boundary == 'y0':
                            x, y = np.random.rand() * self.Lx, 0
                        else:
                            x, y = np.random.rand() * self.Lx, self.Ly
                        
                        z = np.random.rand() * self.Lz
                        
                        # 壁面处使用目标 φ（考虑油墨堆高）
                        phi = self.target_phi_3d(x, y, z, t, V)
                        
                        bc_points.append([x, y, z, t, V])
                        bc_values.append([0.0, 0.0, 0.0, 0.0, phi])
        
        logger.info(f"  壁面边界条件点: {len(bc_points)}")
        
        # ============================================================
        # 4. 域内配点：用于物理方程约束（PDE 残差）
        # ============================================================
        n_domain = data_cfg.get("n_domain", 20000)
        domain_points = []
        for V in voltages:
            for t in np.linspace(0, self.t_max, n_times):
                n_per_domain = n_domain // (len(voltages) * n_times)
                for _ in range(n_per_domain):
                    x = np.random.uniform(0, self.Lx)
                    y = np.random.uniform(0, self.Ly)
                    z = np.random.uniform(0, self.Lz)
                    domain_points.append([x, y, z, t, V])
        
        logger.info(f"  域内配点: {len(domain_points)}")
        
        # 接触角边界条件数据（用于接触角梯度约束）
        n_contact = data_cfg.get("n_interface", 100000) // 2
        contact_points = []
        contact_theta = []
        
        n_per_contact = n_contact // (len(voltages) * n_times)
        for V in voltages:
            for t in np.linspace(0, self.t_max, n_times):
                theta = self.get_contact_angle(V, t)
                for _ in range(n_per_contact):
                    x = np.random.rand() * self.Lx
                    y = np.random.rand() * self.Ly
                    contact_points.append([x, y, 0.0, t, V])  # z=0 底面
                    contact_theta.append(theta)
        
        logger.info(f"  接触角边界条件点: {len(contact_points)}")
        
        return {
            # 界面数据（核心训练数据）
            "interface_points": torch.tensor(np.array(interface_points), dtype=torch.float32, device=self.device),
            "interface_targets": torch.tensor(np.array(interface_targets), dtype=torch.float32, device=self.device),
            # 接触角边界条件
            "contact_points": torch.tensor(np.array(contact_points), dtype=torch.float32, device=self.device),
            "contact_theta": torch.tensor(np.array(contact_theta), dtype=torch.float32, device=self.device),
            # 初始条件
            "ic_points": torch.tensor(np.array(ic_points), dtype=torch.float32, device=self.device),
            "ic_values": torch.tensor(np.array(ic_values), dtype=torch.float32, device=self.device),
            # 壁面边界条件
            "bc_points": torch.tensor(np.array(bc_points), dtype=torch.float32, device=self.device),
            "bc_values": torch.tensor(np.array(bc_values), dtype=torch.float32, device=self.device),
            # 域内配点
            "domain_points": torch.tensor(np.array(domain_points), dtype=torch.float32, device=self.device),
        }


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """两相流 PINN 训练器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型
        self.model = TwoPhasePINN(self.config).to(self.device)
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 数据生成器和物理损失
        self.data_generator = DataGenerator(self.config, self.device)
        self.physics_loss = PhysicsLoss(self.device)
        
        # 训练配置
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 30000)
        self.batch_size = training_cfg.get("batch_size", 4096)
        self.lr = training_cfg.get("learning_rate", 5e-4)
        
        # 渐进式训练阶段
        self.stage1_epochs = training_cfg.get("stage1_epochs", 5000)
        self.stage2_epochs = training_cfg.get("stage2_epochs", 15000)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1500, min_lr=1e-6
        )
        
        # 学习率预热
        self.warmup_epochs = training_cfg.get("warmup_epochs", 500)
        self.warmup_start_lr = self.lr * 0.01  # 从 1% 开始预热
        
        # 训练历史
        self.history = {"epoch": [], "loss": [], "interface": [], "physics": [], "lr": []}
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.early_stop_patience = training_cfg.get("early_stop_patience", 5000)
        
        # 输出目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"outputs_pinn_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_physics_weights(self, epoch: int) -> Dict[str, float]:
        """
        根据训练阶段返回物理损失权重（平滑渐进）
        
        改进：使用平滑过渡而不是阶跃变化
        """
        physics_cfg = self.config.get("physics", {})
        
        if epoch < self.stage1_epochs:
            # 阶段1：纯数据学习
            return {"continuity": 0.0, "vof": 0.0, "ns": 0.0, "surface_tension": 0.0}
        
        elif epoch < self.stage2_epochs:
            # 阶段2：平滑引入连续性和VOF
            # 使用 sigmoid 平滑过渡
            progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
            smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))  # S形曲线
            
            return {
                "continuity": physics_cfg.get("continuity_weight", 0.1) * smooth_factor * 0.1,
                "vof": physics_cfg.get("vof_weight", 0.1) * smooth_factor * 0.1,
                "ns": 0.0,
                "surface_tension": 0.0
            }
        else:
            # 阶段3：完整物理约束（继续平滑增加）
            progress = min(1.0, (epoch - self.stage2_epochs) / 5000.0)
            smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))
            
            return {
                "continuity": physics_cfg.get("continuity_weight", 0.1) * (0.1 + 0.9 * smooth_factor),
                "vof": physics_cfg.get("vof_weight", 0.1) * (0.1 + 0.9 * smooth_factor),
                "ns": physics_cfg.get("ns_weight", 0.01) * smooth_factor,
                "surface_tension": physics_cfg.get("surface_tension_weight", 0.001) * smooth_factor,
            }
    
    def compute_losses(self, data: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """
        计算所有损失 - 中心开口模型 + 标准 VOF
        
        φ = 1: 油墨
        φ = 0: 极性液体（透明）
        
        损失组成：
        1. 界面数据拟合损失（核心）
        2. 接触角边界条件损失
        3. 初始条件损失
        4. 壁面边界条件损失
        5. 物理方程损失
        """
        losses = {}
        physics_cfg = self.config.get("physics", {})
        physics_weights = self.get_physics_weights(epoch)
        
        # ============================================================
        # 1. 界面数据拟合损失（核心！）
        # 使用 target_phi_3d 生成的目标值
        # ============================================================
        idx = torch.randperm(len(data["interface_points"]))[:self.batch_size]
        interface_pts = data["interface_points"][idx]
        interface_tgt = data["interface_targets"][idx]
        
        pred = self.model(interface_pts)
        phi_pred = pred[:, 4]
        
        # MSE 损失
        interface_loss = F.mse_loss(phi_pred, interface_tgt)
        losses["interface"] = interface_loss * physics_cfg.get("interface_weight", 500.0)
        
        # ============================================================
        # 2. 接触角边界条件损失
        # 在底面 z=0，接触角决定了 φ 的梯度方向
        # ============================================================
        idx = torch.randperm(len(data["contact_points"]))[:self.batch_size // 2]
        contact_pts = data["contact_points"][idx].clone().requires_grad_(True)
        pred = self.model(contact_pts)
        phi = pred[:, 4]
        
        # 计算 φ 的梯度
        grad_phi = torch.autograd.grad(
            phi.sum(), contact_pts, create_graph=True, retain_graph=True
        )[0]
        dphi_dz = grad_phi[:, 2]  # z 方向梯度
        grad_phi_mag = torch.sqrt(grad_phi[:, 0]**2 + grad_phi[:, 1]**2 + grad_phi[:, 2]**2 + 1e-10)
        
        # 目标：dphi_dz / |∇φ| = cos(θ)
        theta_rad = data["contact_theta"][idx] * np.pi / 180.0
        target_cos_theta = torch.cos(theta_rad)
        actual_cos_theta = dphi_dz / (grad_phi_mag + 1e-10)
        
        # 界面权重：只在 φ≈0.5 附近施加接触角约束
        interface_weight = torch.exp(-10 * (phi - 0.5)**2)
        
        contact_angle_error = (actual_cos_theta - target_cos_theta)**2 * interface_weight
        losses["contact_angle"] = torch.mean(contact_angle_error) * 100.0
        
        # ============================================================
        # 3. 初始条件损失
        # t=0 时：z < h_ink 处 φ=1，z > h_ink 处 φ=0
        # ============================================================
        idx = torch.randperm(len(data["ic_points"]))[:self.batch_size // 4]
        pred = self.model(data["ic_points"][idx])
        
        ic_phi_loss = F.mse_loss(pred[:, 4:5], data["ic_values"][idx][:, 4:5])
        ic_vel_loss = F.mse_loss(pred[:, :4], data["ic_values"][idx][:, :4])
        
        losses["ic"] = ic_phi_loss * physics_cfg.get("ic_weight", 100.0) + ic_vel_loss * 50.0
        
        # ============================================================
        # 4. 壁面边界条件损失
        # ============================================================
        idx = torch.randperm(len(data["bc_points"]))[:self.batch_size // 4]
        pred = self.model(data["bc_points"][idx])
        
        # 无滑移：速度为零
        losses["bc"] = F.mse_loss(pred[:, :3], data["bc_values"][idx][:, :3]) * physics_cfg.get("bc_weight", 50.0)
        # φ 值约束
        losses["bc"] += F.mse_loss(pred[:, 4:5], data["bc_values"][idx][:, 4:5]) * 30.0
        
        # ============================================================
        # 5. 早期时间约束
        # t < 2ms 时，油墨层内 φ ≈ 1（还没开始响应）
        # ============================================================
        n_early = self.batch_size // 4
        x_early = torch.rand(n_early, device=self.device) * PHYSICS["Lx"]
        y_early = torch.rand(n_early, device=self.device) * PHYSICS["Ly"]
        z_early = torch.rand(n_early, device=self.device) * PHYSICS["h_ink"]  # 只在油墨层内
        t_early = torch.rand(n_early, device=self.device) * 0.002  # t < 2ms
        V_early = torch.rand(n_early, device=self.device) * 30.0
        
        early_points = torch.stack([x_early, y_early, z_early, t_early, V_early], dim=1)
        pred_early = self.model(early_points)
        phi_early = pred_early[:, 4]
        
        # 目标：t < 2ms 时，油墨层内 φ ≈ 1（油墨还没被推开）
        # 标准 VOF：φ=1 表示油墨
        target_phi_early = torch.ones_like(phi_early)
        
        losses["early_time"] = F.mse_loss(phi_early, target_phi_early) * 300.0
        
        # ============================================================
        # 6. 0V 约束：无开口，油墨层内 φ=1
        # ============================================================
        n_0v = self.batch_size // 4
        x_0v = torch.rand(n_0v, device=self.device) * PHYSICS["Lx"]
        y_0v = torch.rand(n_0v, device=self.device) * PHYSICS["Ly"]
        z_0v = torch.rand(n_0v, device=self.device) * PHYSICS["h_ink"]
        t_0v = torch.rand(n_0v, device=self.device) * PHYSICS["t_max"]
        V_0v = torch.zeros(n_0v, device=self.device)
        
        pts_0v = torch.stack([x_0v, y_0v, z_0v, t_0v, V_0v], dim=1)
        phi_0v = self.model(pts_0v)[:, 4]
        
        # 0V 时油墨层内 φ = 1
        losses["zero_voltage"] = F.mse_loss(phi_0v, torch.ones_like(phi_0v)) * 500.0
        
        # ============================================================
        # 7. 单调性约束
        # 中心区域：φ(t2) ≤ φ(t1) 当 t2 > t1（油墨被推开，φ 减小）
        # ============================================================
        n_mono = self.batch_size // 4
        cx, cy = PHYSICS["Lx"] / 2, PHYSICS["Ly"] / 2
        
        # 在中心区域采样
        x_mono = cx + (torch.rand(n_mono, device=self.device) - 0.5) * PHYSICS["Lx"] * 0.4
        y_mono = cy + (torch.rand(n_mono, device=self.device) - 0.5) * PHYSICS["Ly"] * 0.4
        z_mono = torch.rand(n_mono, device=self.device) * PHYSICS["h_ink"]
        V_mono = 10.0 + torch.rand(n_mono, device=self.device) * 20.0  # V > 10V
        
        t1 = torch.rand(n_mono, device=self.device) * 0.01
        t2 = t1 + 0.002 + torch.rand(n_mono, device=self.device) * 0.008
        
        points_t1 = torch.stack([x_mono, y_mono, z_mono, t1, V_mono], dim=1)
        points_t2 = torch.stack([x_mono, y_mono, z_mono, t2, V_mono], dim=1)
        
        phi_t1 = self.model(points_t1)[:, 4]
        phi_t2 = self.model(points_t2)[:, 4]
        
        # 中心区域 φ 应该随时间减小（油墨被推开）
        monotonicity_violation = F.relu(phi_t2 - phi_t1 + 0.02)
        losses["monotonicity"] = torch.mean(monotonicity_violation**2) * 50.0
        
        # ============================================================
        # 8. 电压响应约束
        # 中心区域：V2 > V1 → φ(V2) ≤ φ(V1)（高电压开口更大）
        # ============================================================
        n_volt = self.batch_size // 4
        cx, cy = PHYSICS["Lx"] / 2, PHYSICS["Ly"] / 2
        
        x_volt = cx + (torch.rand(n_volt, device=self.device) - 0.5) * PHYSICS["Lx"] * 0.4
        y_volt = cy + (torch.rand(n_volt, device=self.device) - 0.5) * PHYSICS["Ly"] * 0.4
        z_volt = torch.rand(n_volt, device=self.device) * PHYSICS["h_ink"]
        t_volt = torch.full((n_volt,), 0.015, device=self.device)
        
        V1 = 5.0 + torch.rand(n_volt, device=self.device) * 10.0
        V2 = V1 + 5.0 + torch.rand(n_volt, device=self.device) * 10.0
        V2 = torch.clamp(V2, max=30.0)
        
        pts_V1 = torch.stack([x_volt, y_volt, z_volt, t_volt, V1], dim=1)
        pts_V2 = torch.stack([x_volt, y_volt, z_volt, t_volt, V2], dim=1)
        
        phi_V1 = self.model(pts_V1)[:, 4]
        phi_V2 = self.model(pts_V2)[:, 4]
        
        voltage_violation = F.relu(phi_V2 - phi_V1 + 0.02)
        losses["voltage_response"] = torch.mean(voltage_violation**2) * 100.0
        
        # ============================================================
        # 9. 低电压约束（V < 5V 时油墨层内 φ ≈ 1）
        # ============================================================
        n_low_v = self.batch_size // 4
        x_low = torch.rand(n_low_v, device=self.device) * PHYSICS["Lx"]
        y_low = torch.rand(n_low_v, device=self.device) * PHYSICS["Ly"]
        z_low = torch.rand(n_low_v, device=self.device) * PHYSICS["h_ink"]
        t_low = 0.005 + torch.rand(n_low_v, device=self.device) * 0.015
        V_low = torch.rand(n_low_v, device=self.device) * 5.0
        
        low_v_pts = torch.stack([x_low, y_low, z_low, t_low, V_low], dim=1)
        phi_low_v = self.model(low_v_pts)[:, 4]
        
        # V < 5V 时油墨层内 φ ≈ 1
        losses["low_voltage"] = F.mse_loss(phi_low_v, torch.ones_like(phi_low_v)) * 300.0
        
        # ============================================================
        # 10. 体积守恒约束
        # 油墨总体积 = Lx * Ly * h_ink（标准 VOF：φ=1 时）
        # ============================================================
        n_vol = 15
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
        h_ink = PHYSICS["h_ink"]
        
        # 初始油墨体积
        V_ink_initial = Lx * Ly * h_ink
        
        x_vol = torch.linspace(0, Lx, n_vol, device=self.device)
        y_vol = torch.linspace(0, Ly, n_vol, device=self.device)
        z_vol = torch.linspace(0, Lz, n_vol, device=self.device)
        
        dV = (Lx / n_vol) * (Ly / n_vol) * (Lz / n_vol)
        
        vol_loss = torch.tensor(0.0, device=self.device)
        
        for V_val in [15.0, 25.0]:
            for t_val in [0.010, 0.015]:
                X, Y, Z = torch.meshgrid(x_vol, y_vol, z_vol, indexing='ij')
                pts_3d = torch.stack([
                    X.flatten(), Y.flatten(), Z.flatten(),
                    torch.full((n_vol**3,), t_val, device=self.device),
                    torch.full((n_vol**3,), V_val, device=self.device)
                ], dim=1)
                
                phi_3d = self.model(pts_3d)[:, 4]
                V_ink_current = torch.sum(phi_3d) * dV
                
                vol_error = ((V_ink_current - V_ink_initial) / V_ink_initial)**2
                vol_loss = vol_loss + vol_error
        
        losses["volume_conservation"] = vol_loss * 100.0
        
        # ============================================================
        # 11. 物理方程损失（阶段 2 开始）
        # ============================================================
        if any(w > 0 for w in physics_weights.values()):
            n_physics = min(1000, self.batch_size // 2)
            
            x = torch.rand(n_physics, device=self.device) * PHYSICS["Lx"]
            y = torch.rand(n_physics, device=self.device) * PHYSICS["Ly"]
            z = torch.rand(n_physics, device=self.device) * PHYSICS["Lz"]
            t = torch.rand(n_physics, device=self.device) * PHYSICS["t_max"]
            V = torch.rand(n_physics, device=self.device) * 30.0
            physics_points = torch.stack([x, y, z, t, V], dim=1)
            
            try:
                grads = self.physics_loss.compute_gradients(self.model, physics_points)
                
                # 连续性方程（使用 log1p 缩放）
                if physics_weights["continuity"] > 0:
                    cont_loss = self.physics_loss.continuity_residual(grads)
                    if torch.isfinite(cont_loss):
                        # log1p 缩放：稳定大损失
                        scaled_loss = torch.log1p(cont_loss)
                        losses["continuity"] = scaled_loss * physics_weights["continuity"]
                
                # VOF 方程（使用 log1p 缩放）
                if physics_weights["vof"] > 0:
                    vof_loss = self.physics_loss.vof_residual(grads)
                    if torch.isfinite(vof_loss):
                        scaled_loss = torch.log1p(vof_loss)
                        losses["vof"] = scaled_loss * physics_weights["vof"]
                
                # Navier-Stokes（使用 log1p 缩放）
                if physics_weights["ns"] > 0:
                    ns_loss = self.physics_loss.navier_stokes_residual(grads)
                    if torch.isfinite(ns_loss) and ns_loss < 1e10:
                        scaled_loss = torch.log1p(ns_loss)
                        losses["ns"] = scaled_loss * physics_weights["ns"]
                
                # 表面张力（使用 log1p 缩放）
                if physics_weights.get("surface_tension", 0) > 0:
                    st_loss = self.physics_loss.surface_tension_residual(grads)
                    if torch.isfinite(st_loss):
                        scaled_loss = torch.log1p(st_loss)
                        losses["surface_tension"] = scaled_loss * physics_weights["surface_tension"]
                        
            except Exception as e:
                logger.warning(f"物理损失计算失败: {e}")
        
        losses["total"] = sum(losses.values())
        return losses
    
    def train(self):
        """训练主循环"""
        logger.info("=" * 60)
        logger.info("开始两相流 PINN 训练")
        logger.info("=" * 60)
        
        data = self.data_generator.generate_all_data()
        
        # 保存配置
        with open(f"{self.output_dir}/config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        
        start_time = time.time()
        current_stage = 0
        
        for epoch in range(self.epochs):
            # 学习率预热
            if epoch < self.warmup_epochs:
                warmup_lr = self.warmup_start_lr + (self.lr - self.warmup_start_lr) * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # 检查训练阶段
            if epoch == self.stage1_epochs:
                current_stage = 1
                logger.info(f"\n进入阶段 2：引入连续性和VOF约束")
            elif epoch == self.stage2_epochs:
                current_stage = 2
                logger.info(f"\n进入阶段 3：完整物理约束")
            
            self.model.train()
            self.optimizer.zero_grad()
            
            losses = self.compute_losses(data, epoch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录
            physics_loss = sum(losses.get(k, torch.tensor(0)).item() for k in ["continuity", "vof", "ns"])
            contact_loss = losses.get("contact_angle", torch.tensor(0)).item()
            self.history["epoch"].append(epoch)
            self.history["loss"].append(losses["total"].item())
            self.history["interface"].append(contact_loss)  # 现在是接触角损失
            self.history["physics"].append(physics_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            if epoch % 100 == 0:
                # 预热期后才使用 scheduler
                if epoch >= self.warmup_epochs:
                    self.scheduler.step(losses["total"])
                
                current_loss = losses["total"].item()
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "best_loss": self.best_loss,
                        "config": self.config,
                    }, f"{self.output_dir}/best_model.pth")
                else:
                    self.patience_counter += 100
                
                elapsed = time.time() - start_time
                physics_str = ""
                if "low_voltage" in losses:
                    physics_str += f" | LV: {losses['low_voltage'].item():.2e}"
                if "volume_conservation" in losses:
                    physics_str += f" | Vol: {losses['volume_conservation'].item():.2e}"
                if "ink_distribution_3d" in losses:
                    physics_str += f" | 3D: {losses['ink_distribution_3d'].item():.2e}"
                if "contact_angle" in losses:
                    physics_str += f" | θ: {losses['contact_angle'].item():.2e}"
                if "continuity" in losses:
                    physics_str += f" | C: {losses['continuity'].item():.2e}"
                
                logger.info(
                    f"Epoch {epoch:5d} | Loss: {losses['total'].item():.4e}{physics_str} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s"
                )
        
        # 保存最终模型
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "config": self.config,
        }, f"{self.output_dir}/final_model.pth")
        
        self.visualize()
        
        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳损失: {self.best_loss:.6e}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)
    
    def visualize(self):
        """
        可视化结果
        
        在油墨层中间 (z=h_ink/2) 采样，展示三种模式：
        1. 中心开口模式 (η<50%)
        2. 四角液滴模式 (η>50%, 早期)
        3. 单角液滴模式 (η>50%, 稳态)
        """
        import matplotlib.pyplot as plt
        
        self.model.eval()
        h_ink = PHYSICS["h_ink"]
        
        # phi 分布图 - 在油墨层中间采样
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        voltages = [0, 10, 20, 30]
        times = [0.0, 0.01, 0.02]
        
        nx, ny = 50, 50
        x = np.linspace(0, PHYSICS["Lx"], nx)
        y = np.linspace(0, PHYSICS["Ly"], ny)
        X, Y = np.meshgrid(x, y)
        x_flat, y_flat = X.flatten(), Y.flatten()
        # 在油墨层中间采样 (z=h_ink/2 ≈ 1.5μm)
        z_flat = np.full_like(x_flat, h_ink / 2)
        
        for i, V in enumerate(voltages):
            for j, t in enumerate(times):
                ax = axes[i, j]
                
                t_arr = np.full_like(x_flat, t)
                V_arr = np.full_like(x_flat, V)
                inputs = np.stack([x_flat, y_flat, z_flat, t_arr, V_arr], axis=1).astype(np.float32)
                
                with torch.no_grad():
                    out = self.model(torch.tensor(inputs, device=self.device)).cpu().numpy()
                phi = out[:, 4].reshape(ny, nx)
                
                im = ax.contourf(X * 1e6, Y * 1e6, phi, levels=20, cmap='RdYlBu_r', vmin=0, vmax=1)
                ax.contour(X * 1e6, Y * 1e6, phi, levels=[0.5], colors='black', linewidths=2)
                ax.set_aspect('equal')
                
                # 计算开口率
                eta = np.mean(phi < 0.5)
                
                if i == 0:
                    ax.set_title(f't={t*1000:.0f}ms')
                if j == 0:
                    ax.set_ylabel(f'V={V}V (η={eta:.2f})\ny (μm)')
                if i == len(voltages) - 1:
                    ax.set_xlabel('x (μm)')
        
        plt.suptitle(f'φ Distribution at z={h_ink*1e6:.1f}μm (ink layer)\nφ=1: ink (red), φ=0: transparent (blue)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phi_distribution.png", dpi=150)
        plt.close()
        
        # 训练曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(self.history["epoch"], self.history["loss"], label='Total Loss')
        ax.semilogy(self.history["epoch"], self.history["interface"], label='Interface Loss', alpha=0.7)
        ax.semilogy(self.history["epoch"], self.history["physics"], label='Physics Loss', alpha=0.7)
        ax.axvline(x=self.stage1_epochs, color='r', linestyle='--', alpha=0.5, label='Stage 2')
        ax.axvline(x=self.stage2_epochs, color='g', linestyle='--', alpha=0.5, label='Stage 3')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/training_curve.png", dpi=150)
        plt.close()
        
        logger.info(f"可视化已保存到 {self.output_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EWP 两相流 PINN 训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子（确保可复现）
    set_seed(args.seed)
    logger.info(f"🌱 随机种子: {args.seed}")
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # 命令行参数覆盖
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr
    
    # 训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
