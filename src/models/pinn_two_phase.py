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
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.physics.constraints import PhysicsConstraints, PINNConstraintLayer


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
# 物理常量 - 从统一配置模块导入
# ============================================================================
try:
    from src.config import PHYSICS
except ImportError:
    # 回退：如果配置模块不可用，使用本地定义
    PHYSICS = {
        "Lx": 174e-6,
        "Ly": 174e-6,
        "Lz": 20e-6,
        "h_ink": 3e-6,
        "h_polar": 17e-6,
        "rho_oil": 800.0,
        "rho_polar": 1000.0,
        "mu_oil": 0.003,
        "mu_polar": 0.001,
        "sigma": 0.045,
        "theta0": 120.0,
        "gamma": 0.015,
        "epsilon_r": 12.0,
        "epsilon_h": 1.9,
        "d_dielectric": 4e-7,
        "d_hydrophobic": 4e-7,
        "epsilon_0": 8.854e-12,
        "tau": 0.005,
        "tau_recovery": 0.0075,
        "zeta": 0.8,
        "t_max": 0.05,
        "V_threshold": 3.0,
        "eta_max": 0.85,
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
        "sharpening": 0.1,             # [Added] VOF Sharpening Loss
    },
    "data": {
        "n_interface": 100000,
        "n_initial": 10000,
        "n_boundary": 10000,
        "n_domain": 20000,
        "voltages": [0.5 * i for i in range(0, 61)],
        "times": 50,
    },
}


# ============================================================================
# 两相流 PINN 模型
# ============================================================================

class TwoPhasePINN(nn.Module):
    """
    两相流物理信息神经网络
    
    特点：
    - 支持升压和降压过程
    - 分离 phi 网络和速度网络
    - 输入: (x, y, z, V_from, V_to, t_since) - 6维（三元组格式）
      * V_from: 跳变前电压
      * V_to: 跳变后电压（当前电压）
      * t_since: 跳变后经过的时间
    - 输出: (u, v, w, p, phi)
    
    电压历史的物理意义：
    - V_from = V_to: 恒定电压状态
    - V_from < V_to: 升压过程（电润湿驱动）
    - V_from > V_to: 降压过程（表面张力恢复）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or DEFAULT_CONFIG
        
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
        
        # Phi 网络：输入 (x, y, z, V_from, V_to, t_since) - 6维
        self.phi_net = self._build_network(6, 1, hidden_phi)
        
        # 速度网络：输入 (x, y, z, V_from, V_to, t_since, phi) - 7维
        self.vel_net = self._build_network(7, 4, hidden_vel)
        
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
            x: (batch, 6) - (x, y, z, V_from, V_to, t_since)
        
        Returns:
            (batch, 5) - (u, v, w, p, phi)
        """
        # 确保输入是 (batch, 6)
        if x.shape[1] != 6:
            raise ValueError("TwoPhasePINN expects input of shape (batch, 6).")
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        V_from = x[:, 3:4]
        V_to = x[:, 4:5]
        t_since = x[:, 5:6]
        
        # 归一化到 [0, 1]
        x_norm = x_coord / self.Lx
        y_norm = y_coord / self.Ly
        z_norm = z_coord / self.Lz
        t_norm = t_since / self.t_max
        V_from_norm = V_from / 30.0
        V_to_norm = V_to / 30.0
        
        # Phi 预测 - 使用 (x, y, z, V_from, V_to, t_since)
        phi_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1)
        phi_raw = self.phi_net(phi_input)
        phi = torch.sigmoid(phi_raw)  # 限制在 [0, 1]
        
        # 速度预测 - 输入增加 phi (batch, 7)
        vel_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm, phi], dim=-1)
        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]
        
        return torch.cat([u, v, w, p, phi], dim=-1)
    
    def forward_triplet(
        self,
        spatial_coords: torch.Tensor,
        voltage_triplet: torch.Tensor
    ) -> torch.Tensor:
        """
        三元组格式的前向传播（与 LSTM-Hybrid-PINN 接口一致）
        
        Args:
            spatial_coords: (batch, 3) - (x, y, z)
            voltage_triplet: (batch, 3) - (V_from, V_to, t_since)，已归一化
        
        Returns:
            (batch, 5) - (u, v, w, p, phi)
        """
        # 归一化空间坐标
        x_norm = spatial_coords[:, 0:1] / self.Lx
        y_norm = spatial_coords[:, 1:2] / self.Ly
        z_norm = spatial_coords[:, 2:3] / self.Lz
        
        # 电压三元组已经归一化
        V_from_norm = voltage_triplet[:, 0:1]
        V_to_norm = voltage_triplet[:, 1:2]
        t_norm = voltage_triplet[:, 2:3]
        
        # Phi 网络输入
        phi_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1)
        phi_raw = self.phi_net(phi_input)
        phi = torch.sigmoid(phi_raw)
        
        # 速度网络输入
        vel_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm, phi], dim=-1)
        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]
        
        return torch.cat([u, v, w, p, phi], dim=-1)


# ============================================================================
# 物理损失计算
# ============================================================================

class PhysicsLoss:
    """
    两相流物理损失（重构版）
    
    设计原则：
    - PhysicsConstraints 作为"唯一物理真相"，负责物理方程计算
    - PhysicsLoss 作为"适配层 + 数值保护壳"，负责：
      1. 调用 PhysicsConstraints.compute_core_residuals() 获取原始残差
      2. 对残差进行 NaN/Inf 清理、裁剪
      3. 应用 log1p 缩放和归一化
      4. 界面加权和损失聚合
    
    包含：
    - 连续性方程：∇·u = 0
    - VOF 方程：∂φ/∂t + u·∇φ = 0
    - Navier-Stokes 方程
    - 表面张力 CSF 模型：F_st = σκδ_s n
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
        
        # 实例化 PhysicsConstraints 作为物理方程计算核心
        try:
            from src.config import get_materials_params
            self.physics_constraints = PhysicsConstraints(materials_params=get_materials_params())
        except ImportError:
            self.physics_constraints = PhysicsConstraints()
    
    def _sanitize_tensor(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """
        清理张量中的 NaN/Inf 值
        
        Args:
            tensor: 输入张量
            name: 张量名称（用于日志）
        
        Returns:
            清理后的张量
        """
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"张量 {name} 包含 {nan_count} 个 NaN, {inf_count} 个 Inf，已清零")
            tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        return tensor
    
    def compute_all_residuals(
        self, 
        model: nn.Module, 
        points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有物理残差（通过 PhysicsConstraints）
        
        Args:
            model: PINN 模型
            points: 物理点坐标 (batch, 6)
        
        Returns:
            残差字典
        """
        points = points.clone().requires_grad_(True)
        
        predictions = model(points)
        residuals = self.physics_constraints.compute_core_residuals(
            x_phys=points,
            predictions=predictions,
            model=model
        )
        
        # 清理所有残差中的 NaN/Inf
        for key in residuals:
            residuals[key] = self._sanitize_tensor(residuals[key], key)
        
        return residuals
    
    def explicit_volume_conservation_loss(
        self,
        phi_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        显式体积守恒损失

        直接计算体积积分误差: |∫φ dV - V₀| / V₀
        这是对VOF方程的补充，提供直接的体积守恒约束

        Args:
            phi_pred: 预测的φ场 (batch,)

        Returns:
            体积守恒误差（相对误差）
        """
        # 数值积分（使用黎曼和）
        dV = (PHYSICS["Lx"] * PHYSICS["Ly"] * PHYSICS["Lz"]) / phi_pred.numel()
        volume_integral = torch.sum(phi_pred) * dV

        # 初始油墨体积
        V_initial = PHYSICS["Lx"] * PHYSICS["Ly"] * PHYSICS["h_ink"]

        # 相对误差
        volume_error = torch.abs(volume_integral - V_initial) / (V_initial + 1e-10)

        return volume_error

    def compute_total_loss(
        self,
        model: nn.Module,
        points: torch.Tensor,
        weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总物理损失（推荐入口）

        通过 PhysicsConstraints.compute_core_residuals() 获取残差，
        然后应用归一化、log1p 缩放和加权求和。

        Args:
            model: PINN 模型
            points: 物理点坐标 (batch, 6)
            weights: 各残差项的权重，默认使用均匀权重

        Returns:
            包含各项损失和总损失的字典
        """
        default_weights = {
            'continuity': 0.5,
            'momentum_u': 0.1,
            'momentum_v': 0.1,
            'momentum_w': 0.1,
            'vof': 2.0,  # 方案B: 提升VOF权重
            'surface_tension': 0.05,  # 方案B: 提升表面张力权重
            'volume_conservation': 0.3,
            'explicit_volume': 1.0,  # 方案B: 新增显式体积守恒权重
            'sharpening': 0.1,  # 改进方案: 界面锐化正则化
        }
        weights = weights or default_weights

        # 获取所有残差
        residuals = self.compute_all_residuals(model, points)

        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 预先计算模型输出，供后续特殊Loss使用
        outputs = model(points)
        phi_pred = outputs[:, 4]  # 注意：这里可能未经过 sigmoid，具体取决于模型输出
        # 根据 TwoPhasePINN.forward，输出已经是 sigmoid 后的 [0,1] 值
        
        # 处理标准物理残差
        for key, residual in residuals.items():
            if residual is None:
                continue

            # 计算 MSE
            mse = torch.mean(residual**2)

            # 应用 log1p 缩放稳定大损失
            scaled_loss = torch.log1p(mse)

            # 获取权重
            w = weights.get(key, 0.1)

            # 加权损失
            weighted_loss = w * scaled_loss
            losses[key] = weighted_loss
            total_loss = total_loss + weighted_loss

        # 方案B: 添加显式体积守恒损失
        if 'explicit_volume' in weights and weights['explicit_volume'] > 0:
            phi_clamped = torch.clamp(phi_pred, 0.0, 1.0)
            vol_loss = self.explicit_volume_conservation_loss(phi_clamped)
            losses['explicit_volume'] = weights['explicit_volume'] * vol_loss
            total_loss = total_loss + losses['explicit_volume']
            
        # 改进方案: 添加界面锐化正则化 Loss
        if 'sharpening' in weights and weights['sharpening'] > 0:
            # Loss = phi * (1 - phi)
            # 这会惩罚 0.5 附近的值，迫使 phi 趋向 0 或 1
            sharpening_val = torch.mean(phi_pred * (1.0 - phi_pred))
            losses['sharpening'] = weights['sharpening'] * sharpening_val
            total_loss = total_loss + losses['sharpening']

        losses['total'] = total_loss
        return losses
    
    def compute_gradients(self, model: nn.Module, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算所有需要的梯度"""
        points = points.clone().requires_grad_(True)
        outputs = model(points)
        
        # 提取输出并检查 NaN
        u, v, w, p, phi = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]
        
        if not torch.isfinite(outputs).all():
            logger.warning("模型输出包含 NaN/Inf，跳过本次梯度计算")
            return None

        def grad(y, x):
            g = torch.autograd.grad(
                y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            if g is None:
                return torch.zeros_like(x)
            return g
        
        # 一阶导数
        grads = {"u": u, "v": v, "w": w, "p": p, "phi": phi}
        
        for name, var in [("u", u), ("v", v), ("w", w), ("p", p), ("phi", phi)]:
            g = grad(var.sum(), points)
            grads[f"{name}_x"] = g[:, 0]
            grads[f"{name}_y"] = g[:, 1]
            grads[f"{name}_z"] = g[:, 2]
            # 三元组格式: (x, y, z, V_from, V_to, t_since)
            # t_since 在索引 5
            grads[f"{name}_t"] = g[:, 5]
        
        # 二阶导数（用于粘性项和曲率）
        for base in ["u", "v", "w", "phi"]:
            for coord, idx in [("x", 0), ("y", 1), ("z", 2)]:
                first = grads[f"{base}_{coord}"]
                second = grad(first.sum(), points)
                grads[f"{base}_{coord}{coord}"] = second[:, idx]
        
        # 增加 phi 的混合偏导数用于精确曲率计算
        for c1, i1, c2, i2 in [("x", 0, "y", 1), ("x", 0, "z", 2), ("y", 1, "z", 2)]:
            first = grads[f"phi_{c1}"]
            second = grad(first.sum(), points)
            grads[f"phi_{c1}{c2}"] = second[:, i2]
        
        # 最终有限性检查
        for k, v in grads.items():
            if not torch.isfinite(v).all():
                logger.warning(f"梯度项 {k} 包含 NaN/Inf，尝试清零")
                grads[k] = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                
        return grads
    
    def continuity_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        连续性方程残差：∇·u = 0（归一化）
        
        .. deprecated:: 2025-12-31
            此方法已弃用，请使用 compute_all_residuals() 或 compute_total_loss()。
            物理方程计算已统一到 PhysicsConstraints.compute_core_residuals()。
        """
        div_u = grads["u_x"] + grads["v_y"] + grads["w_z"]
        # 归一化：除以特征速度/长度
        div_u_norm = div_u * self.L_char / self.U_char
        return torch.mean(div_u_norm**2)
    
    def vof_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        VOF 方程残差：∂φ/∂t + u·∇φ = 0（归一化）
        
        .. deprecated:: 2025-12-31
            此方法已弃用，请使用 compute_all_residuals() 或 compute_total_loss()。
            物理方程计算已统一到 PhysicsConstraints.compute_core_residuals()。
        """
        u, v, w = grads["u"], grads["v"], grads["w"]
        res = grads["phi_t"] + u * grads["phi_x"] + v * grads["phi_y"] + w * grads["phi_z"]
        # 归一化：phi 是无量纲的，时间导数除以 1/T_char
        res_norm = res * self.T_char
        return torch.mean(res_norm**2)
    
    def navier_stokes_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Navier-Stokes 方程残差（归一化） - 增强科学严谨性
        
        .. deprecated:: 2025-12-31
            此方法已弃用，请使用 compute_all_residuals() 或 compute_total_loss()。
            物理方程计算已统一到 PhysicsConstraints.compute_core_residuals()。
        """
        u, v, w, phi = grads["u"], grads["v"], grads["w"], grads["phi"]
        phi_x, phi_y, phi_z = grads["phi_x"], grads["phi_y"], grads["phi_z"]
        
        # 混合流体属性
        rho = phi * self.rho_oil + (1 - phi) * self.rho_polar
        mu = phi * self.mu_oil + (1 - phi) * self.mu_polar
        
        # 粘度梯度 (因为 mu 是 phi 的线性函数: mu = phi*mu_oil + (1-phi)*mu_polar)
        # dmu/dx = (mu_oil - mu_polar) * phi_x
        dmu_dphi = self.mu_oil - self.mu_polar
        mu_x = dmu_dphi * phi_x
        mu_y = dmu_dphi * phi_y
        mu_z = dmu_dphi * phi_z
        
        # 对流项
        u_conv = u * grads["u_x"] + v * grads["u_y"] + w * grads["u_z"]
        v_conv = u * grads["v_x"] + v * grads["v_y"] + w * grads["v_z"]
        w_conv = u * grads["w_x"] + v * grads["w_y"] + w * grads["w_z"]
        
        # 增强的粘性项：div(mu * (grad(u) + grad(u)^T))
        # 对于不可压缩流 (div u = 0)，该项简化为: mu*laplacian(u) + grad(mu)*(grad(u) + (grad u)^T)
        u_visc = mu * (grads["u_xx"] + grads["u_yy"] + grads["u_zz"]) + \
                 mu_x * (2 * grads["u_x"]) + \
                 mu_y * (grads["u_y"] + grads["v_x"]) + \
                 mu_z * (grads["u_z"] + grads["w_x"])
        
        v_visc = mu * (grads["v_xx"] + grads["v_yy"] + grads["v_zz"]) + \
                 mu_x * (grads["v_x"] + grads["u_y"]) + \
                 mu_y * (2 * grads["v_y"]) + \
                 mu_z * (grads["v_z"] + grads["w_y"])
                 
        w_visc = mu * (grads["w_xx"] + grads["w_yy"] + grads["w_zz"]) + \
                 mu_x * (grads["w_x"] + grads["u_z"]) + \
                 mu_y * (grads["w_y"] + grads["v_z"]) + \
                 mu_z * (2 * grads["w_z"])
        
        # 表面张力项 (CSF 模型)
        kappa = self._compute_curvature(grads)
        f_st_x = self.sigma * kappa * phi_x
        f_st_y = self.sigma * kappa * phi_y
        f_st_z = self.sigma * kappa * phi_z
        
        # N-S 残差 (包含表面张力)
        ns_u = rho * (grads["u_t"] + u_conv) + grads["p_x"] - u_visc - f_st_x
        ns_v = rho * (grads["v_t"] + v_conv) + grads["p_y"] - v_visc - f_st_y
        ns_w = rho * (grads["w_t"] + w_conv) + grads["p_z"] - w_visc - f_st_z
        
        # 归一化：除以 ρU²/L
        scale = self.rho_polar * self.U_char**2 / self.L_char
        
        # 界面加权：在界面附近施加更强的物理约束，提高数值稳定性
        interface_indicator = torch.exp(-100 * (phi - 0.5)**2)
        weight = 0.1 + 0.9 * interface_indicator
        
        ns_u_norm = (ns_u / (scale + 1e-10)) * weight
        ns_v_norm = (ns_v / (scale + 1e-10)) * weight
        ns_w_norm = (ns_w / (scale + 1e-10)) * weight
        
        return torch.mean(ns_u_norm**2 + ns_v_norm**2 + ns_w_norm**2)
    
    def _compute_curvature(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算界面曲率 kappa = -div(n), n = grad(phi)/|grad(phi)|
        
        精确公式:
        kappa = -(phi_xx*(phi_y^2 + phi_z^2) + phi_yy*(phi_x^2 + phi_z^2) + phi_zz*(phi_x^2 + phi_y^2)
                 - 2*(phi_x*phi_y*phi_xy + phi_x*phi_z*phi_xz + phi_y*phi_z*phi_yz)) / |grad(phi)|^3
        """
        phi_x, phi_y, phi_z = grads["phi_x"], grads["phi_y"], grads["phi_z"]
        phi_xx, phi_yy, phi_zz = grads["phi_xx"], grads["phi_yy"], grads["phi_zz"]
        phi_xy, phi_xz, phi_yz = grads["phi_xy"], grads["phi_xz"], grads["phi_yz"]
        
        grad_mag_sq = phi_x**2 + phi_y**2 + phi_z**2 + 1e-10
        grad_mag = torch.sqrt(grad_mag_sq)
        
        # 精确曲率公式 (3D)
        numerator = (
            phi_xx * (phi_y**2 + phi_z**2) + 
            phi_yy * (phi_x**2 + phi_z**2) + 
            phi_zz * (phi_x**2 + phi_y**2) - 
            2 * (phi_x * phi_y * phi_xy + phi_x * phi_z * phi_xz + phi_y * phi_z * phi_yz)
        )
        
        kappa = -numerator / (grad_mag_sq * grad_mag + 1e-10)
        return kappa

    def surface_tension_residual(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        表面张力平衡残差（可选）
        约束界面处的曲率不要发散
        
        .. deprecated:: 2025-12-31
            此方法已弃用，请使用 compute_all_residuals() 或 compute_total_loss()。
            物理方程计算已统一到 PhysicsConstraints.compute_core_residuals()。
        """
        phi = grads["phi"]
        kappa = self._compute_curvature(grads)
        
        # 界面指示函数（在 phi=0.5 附近）
        interface_indicator = torch.exp(-100 * (phi - 0.5)**2)
        
        # 约束界面处的曲率平滑
        return torch.mean(interface_indicator * kappa**2)


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
                from src.config import CONFIG_PATH
                self.contact_angle_predictor = HybridPredictor(
                    config_path=str(CONFIG_PATH),
                    use_model_for_steady_state=False
                )
                logger.info("✅ 已集成 Stage 1 HybridPredictor 作为接触角边界条件")
            except Exception as e:
                logger.warning(f"HybridPredictor 初始化失败: {e}")
        
        self.theta0 = PHYSICS["theta0"]
        self.use_stage1_eta = self.config.get("stage1_eta_from_model", False)
    
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
        theta = self.get_contact_angle(V, t)
        if HAS_APERTURE and self.use_stage1_eta:
            try:
                if not hasattr(self, '_aperture_model'):
                    from src.config import CONFIG_PATH
                    self._aperture_model = EnhancedApertureModel(
                        config_path=str(CONFIG_PATH)
                    )
                return self._aperture_model.contact_angle_to_aperture_ratio(theta)
            except Exception as e:
                logger.warning(f"EnhancedApertureModel 调用失败: {e}")
        
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
    
    def target_phi_3d(self, x: float, y: float, z: float, t: float, V: float, 
                       V_prev: float = None, t_step: float = 0.0) -> float:
        """
        计算目标 φ 值（支持升压和降压）
        
        物理模型：
        - 初始：底部 3μm 油墨 (φ=1)，上部 17μm 极性液体 (φ=0)
        - 升压（V > V_prev）：电润湿驱动，油墨被推到边缘/角落
        - 降压（V < V_prev）：表面张力恢复，油墨从边缘/角落铺展回中心
        
        升压模式：
        - 开口率 < 50%：中心开口模式（油墨环形分布）
        - 开口率 > 50%：四角液滴模式（油墨在角落）
        
        降压模式：
        - 油墨从当前位置向中心铺展
        - 开口率指数衰减：η(t) = η_at_step × exp(-(t - t_step) / τ_recovery)
        
        Args:
            x, y, z: 空间坐标 (m)
            t: 时间 (s)
            V: 当前电压 (V)
            V_prev: 之前的电压 (V)，默认等于 V（恒定电压）
            t_step: 电压变化时刻 (s)
        
        Returns:
            φ ∈ [0, 1]
        """
        h_ink = self.h_ink
        if V_prev is None:
            V_prev = V
        if t_step is None:
            t_step = 0.0
        t_since = max(0.0, t - t_step)
        eta = None
        if HAS_APERTURE:
            try:
                if not hasattr(self, "_aperture_model"):
                    from src.config import CONFIG_PATH
                    self._aperture_model = EnhancedApertureModel(
                        config_path=str(CONFIG_PATH)
                    )
                _, eta_stage1 = self._aperture_model.theta_eta_from_triad(
                    V_prev, V, t_since
                )
                eta = float(eta_stage1)
            except Exception as e:
                logger.warning(f"EnhancedApertureModel 在 target_phi_3d 中失败: {e}")
        if eta is None:
            is_voltage_down = V < V_prev
            if is_voltage_down:
                tau_recovery = PHYSICS.get("tau_recovery", 0.0075)
                eta_max = self.get_opening_rate(V_prev, 0.020)
                if t_step is not None and t_step >= 0:
                    t_since_local = max(0, t - t_step)
                else:
                    t_since_local = max(0, t - 0.015)
                eta = eta_max * np.exp(-t_since_local / tau_recovery)
                return self._phi_center_opening_mode(x, y, z, eta, h_ink)
            else:
                eta = self.get_opening_rate(V, t)
        
        # 界面宽度
        interface_width = 1.5e-6  # 1.5 μm (方案B: 减小界面宽度提高精度)
        
        # 简化模型：只使用中心开口模式（不使用 corner 模式）
        # 这样可以保证中心始终是透明的，边缘是油墨
        
        if eta < 0.01:
            # 无开口：初始状态
            phi_z = 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))
        
        else:
            # ============================================================
            # 模式 1：中心开口模式（开口率 < 50%）
            # 油墨形成环形分布，中心透明
            # ============================================================
            r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
            
            # 开口半径
            r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
            
            # 油墨堆高（体积守恒）
            # 修复: 不再强制限制高度，而是限制最大开口率以保持体积守恒
            max_eta = 0.85  # 物理上合理的最大开口率
            eta = min(eta, max_eta)  # 限制开口率
            r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)  # 重新计算开口半径
            ink_area = self.Lx * self.Ly - np.pi * r_open**2
            h_ink_edge = self.Lx * self.Ly * h_ink / max(ink_area, 1e-12)
            # 移除了: min(h_ink_edge, self.Lz * 0.8) - 这行代码破坏体积守恒
            
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
        
        return np.clip(phi_z, 0, 1)
    
    def _phi_center_opening_mode(self, x: float, y: float, z: float, eta: float, h_ink: float) -> float:
        """
        中心开口模式的 φ 分布（用于升压和降压）
        
        物理模型：
        - 中心区域透明（φ=0），油墨在边缘形成环形分布
        - 开口半径 r_open = sqrt(η × Lx × Ly / π)
        - 油墨堆高满足体积守恒
        
        Args:
            x, y, z: 空间坐标 (m)
            eta: 开口率 ∈ [0, 1]
            h_ink: 初始油墨厚度 (m)
        
        Returns:
            φ ∈ [0, 1]
        """
        interface_width = 1.5e-6  # 1.5 μm (方案B: 减小界面宽度提高精度)
        
        if eta < 0.01:
            # 无开口：初始状态，油墨均匀铺在底部
            return 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))
        
        if eta > 0.99:
            # 完全开口：几乎没有油墨
            return 0.0
        
        r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        
        # 开口半径
        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
        
        # 油墨堆高（体积守恒）
        # 修复: 限制最大开口率以保持体积守恒
        max_eta = 0.85
        eta = min(eta, max_eta)
        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
        ink_area = self.Lx * self.Ly - np.pi * r_open**2
        h_ink_edge = self.Lx * self.Ly * h_ink / max(ink_area, 1e-12)
        # 移除了: min(h_ink_edge, self.Lz * 0.8) - 破坏体积守恒
        
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
        
        return np.clip(phi_z, 0, 1)
    
    def _sample_point_by_eta(self, eta: float) -> tuple:
        """根据开口率采样空间点"""
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
        
        return x, y, z
    
    def generate_all_data(self) -> Dict[str, torch.Tensor]:
        """
        生成训练数据 - 支持升压和降压
        
        数据格式: (x, y, z, V_from, V_to, t_since) - 三元组格式
        φ = 1: 油墨
        φ = 0: 极性液体（透明）
        
        三元组语义：
        - V_from: 跳变前电压
        - V_to: 跳变后电压（当前电压）
        - t_since: 跳变后经过的时间
        """
        data_cfg = self.config.get("data", DEFAULT_CONFIG["data"])
        
        logger.info("生成训练数据（连续采样模式）...")
        logger.info("  数据格式: (x, y, z, V_from, V_to, t_since)")
        
        # 辅助函数：连续时间采样（偏重早期动态）
        def sample_continuous_times(n_samples):
            """
            连续时间采样 (Beta 分布)
            
            使用 Beta(0.5, 1.0) 分布，自然地在早期 (t=0) 附近采样更多点，
            同时平滑覆盖整个时间域，避免人为的分段点 (如 15ms) 造成的密度突变。
            """
            # Beta(0.5, 1.0) 类似于 1/sqrt(x)，在 0 处概率密度高，向右递减
            # 这非常适合模拟暂态过程（变化集中在早期）
            t_samples = np.random.beta(0.5, 1.0, n_samples) * self.t_max
            return t_samples

        # ============================================================
        # 1. 界面数据（核心训练数据）
        # ============================================================
        n_interface = data_cfg.get("n_interface", 100000)
        interface_points = []
        interface_targets = []
        
        # 1.1 稳态数据 (40%) - V_from = V_to
        # 连续电压采样 [0, 30]
        n_steady = int(n_interface * 0.4)
        V_steady = np.random.uniform(0, 30.0, n_steady)
        t_steady = sample_continuous_times(n_steady)
        
        for V, t in zip(V_steady, t_steady):
            eta = self.get_opening_rate(V, t)
            x, y, z = self._sample_point_by_eta(eta)
            phi = self.target_phi_3d(x, y, z, t, V, V_prev=V)
            interface_points.append([x, y, z, V, V, t])
            interface_targets.append(phi)
            
        # 1.2 升压响应 (30%) - 0 -> V
        n_up = int(n_interface * 0.3)
        V_up = np.random.uniform(1.0, 30.0, n_up)
        t_up = sample_continuous_times(n_up)
        
        for V, t in zip(V_up, t_up):
            eta = self.get_opening_rate(V, t)
            x, y, z = self._sample_point_by_eta(eta)
            phi = self.target_phi_3d(x, y, z, t, V, V_prev=0.0)
            interface_points.append([x, y, z, 0.0, V, t])
            interface_targets.append(phi)
            
        # 1.3 降压响应 (30%) - V -> 0
        n_down = n_interface - n_steady - n_up
        V_down = np.random.uniform(1.0, 30.0, n_down)
        t_down = sample_continuous_times(n_down)
        tau_recovery = PHYSICS.get("tau_recovery", 0.0075)
        
        for V, t in zip(V_down, t_down):
            # 降压逻辑：从 V 的稳态开始指数衰减
            eta_at_fall = self.get_opening_rate(V, 0.020)
            eta = eta_at_fall * np.exp(-t / tau_recovery)
            x, y, z = self._sample_point_by_eta(eta)
            phi = self.target_phi_3d(x, y, z, t, 0.0, V_prev=V)
            interface_points.append([x, y, z, V, 0.0, t])
            interface_targets.append(phi)
        
        logger.info(f"  界面数据点: {len(interface_points)}")
        
        # ============================================================
        # 2. 初始条件：t=0 时油墨均匀铺在底部 3μm
        # ============================================================
        n_ic = data_cfg.get("n_initial", 10000)
        ic_points, ic_values = [], []
        
        # 连续电压采样
        V_ic = np.random.uniform(0, 30.0, n_ic)
        
        for V in V_ic:
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly
            z = np.random.rand() * self.Lz
            
            interface_width = 1e-6
            phi = 0.5 * (1 - np.tanh((z - self.h_ink) / interface_width))
            phi = np.clip(phi, 0, 1)
            
            # 三元组格式: (x, y, z, V_from, V_to, t_since=0)
            ic_points.append([x, y, z, V, V, 0.0])
            ic_values.append([0.0, 0.0, 0.0, 0.0, phi])
        
        logger.info(f"  初始条件点: {len(ic_points)}")
        
        # ============================================================
        # 3. 壁面边界条件
        # ============================================================
        n_bc = data_cfg.get("n_boundary", 10000)
        bc_points, bc_values = [], []
        
        # 混合场景采样：40% 稳态, 30% 升压, 30% 降压
        n_bc_steady = int(n_bc * 0.4)
        n_bc_up = int(n_bc * 0.3)
        n_bc_down = n_bc - n_bc_steady - n_bc_up
        
        # 3.1 稳态 BC
        V_steady = np.random.uniform(0, 30.0, n_bc_steady)
        t_steady = sample_continuous_times(n_bc_steady)
        
        # 3.2 升压 BC
        V_up = np.random.uniform(1.0, 30.0, n_bc_up)
        t_up = sample_continuous_times(n_bc_up)
        
        # 3.3 降压 BC
        V_down = np.random.uniform(1.0, 30.0, n_bc_down)
        t_down = sample_continuous_times(n_bc_down)
        
        # 合并所有场景
        scenarios = []
        for i in range(n_bc_steady):
            scenarios.append((V_steady[i], V_steady[i], t_steady[i]))
        for i in range(n_bc_up):
            scenarios.append((0.0, V_up[i], t_up[i]))
        for i in range(n_bc_down):
            scenarios.append((V_down[i], 0.0, t_down[i]))
            
        for V_from, V_to, t in scenarios:
            # 随机选择一个壁面
            boundary_type = np.random.randint(0, 4) # 0:x0, 1:xL, 2:y0, 3:yL
            
            if boundary_type == 0:   # x=0
                x, y = 0, np.random.rand() * self.Ly
            elif boundary_type == 1: # x=Lx
                x, y = self.Lx, np.random.rand() * self.Ly
            elif boundary_type == 2: # y=0
                x, y = np.random.rand() * self.Lx, 0
            else:                    # y=Ly
                x, y = np.random.rand() * self.Lx, self.Ly
            
            z = np.random.rand() * self.Lz
            
            # 计算目标 phi
            phi = self.target_phi_3d(x, y, z, t, V_to, V_prev=V_from)
            
            # 三元组格式: (x, y, z, V_from, V_to, t_since)
            bc_points.append([x, y, z, V_from, V_to, t])
            bc_values.append([0.0, 0.0, 0.0, 0.0, phi])
        
        logger.info(f"  壁面边界条件点: {len(bc_points)}")
        
        # ============================================================
        # 4. 域内配点
        # ============================================================
        n_domain = data_cfg.get("n_domain", 20000)
        domain_points = []
        
        # 混合场景采样
        n_dom_steady = int(n_domain * 0.4)
        n_dom_up = int(n_domain * 0.3)
        n_dom_down = n_domain - n_dom_steady - n_dom_up
        
        # 生成域内点场景
        dom_scenarios = []
        
        # 稳态
        V_s = np.random.uniform(0, 30.0, n_dom_steady)
        t_s = sample_continuous_times(n_dom_steady)
        for i in range(n_dom_steady):
            dom_scenarios.append((V_s[i], V_s[i], t_s[i]))
            
        # 升压
        V_u = np.random.uniform(1.0, 30.0, n_dom_up)
        t_u = sample_continuous_times(n_dom_up)
        for i in range(n_dom_up):
            dom_scenarios.append((0.0, V_u[i], t_u[i]))
            
        # 降压
        V_d = np.random.uniform(1.0, 30.0, n_dom_down)
        t_d = sample_continuous_times(n_dom_down)
        for i in range(n_dom_down):
            dom_scenarios.append((V_d[i], 0.0, t_d[i]))
            
        for V_from, V_to, t in dom_scenarios:
            x = np.random.uniform(0, self.Lx)
            y = np.random.uniform(0, self.Ly)
            z = np.random.uniform(0, self.Lz)
            domain_points.append([x, y, z, V_from, V_to, t])
        
        logger.info(f"  域内配点: {len(domain_points)}")
        
        # ============================================================
        # 5. 接触角边界条件
        # ============================================================
        n_contact = data_cfg.get("n_interface", 100000) // 2
        contact_points = []
        contact_theta = []
        
        # 混合场景采样
        n_con_steady = int(n_contact * 0.4)
        n_con_up = int(n_contact * 0.3)
        n_con_down = n_contact - n_con_steady - n_con_up
        
        con_scenarios = []
        
        # 稳态
        V_cs = np.random.uniform(0, 30.0, n_con_steady)
        t_cs = sample_continuous_times(n_con_steady)
        for i in range(n_con_steady):
            con_scenarios.append((V_cs[i], V_cs[i], t_cs[i]))
            
        # 升压
        V_cu = np.random.uniform(1.0, 30.0, n_con_up)
        t_cu = sample_continuous_times(n_con_up)
        for i in range(n_con_up):
            con_scenarios.append((0.0, V_cu[i], t_cu[i]))
            
        # 降压
        V_cd = np.random.uniform(1.0, 30.0, n_con_down)
        t_cd = sample_continuous_times(n_con_down)
        for i in range(n_con_down):
            con_scenarios.append((V_cd[i], 0.0, t_cd[i]))
            
        for V_from, V_to, t in con_scenarios:
            # 接触角计算逻辑
            if V_from > V_to:  # 降压过程 (Step Down)
                # 模拟接触角弛豫：从 theta(V_from) 恢复到 theta0
                # 1. 计算初始高压下的稳态接触角
                # 使用 t=1.0 (足够长的时间) 获取稳态值
                theta_high = self.get_contact_angle(V_from, 1.0)
                theta_low = self.theta0
                
                # 2. 指数弛豫
                tau_recovery = PHYSICS.get("tau_recovery", 0.0075)
                decay = np.exp(-t / tau_recovery)
                theta = theta_low + (theta_high - theta_low) * decay
                
            else:  # 升压或稳态
                theta = self.get_contact_angle(V_to, t)
            
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly
            contact_points.append([x, y, 0.0, V_from, V_to, t])
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
    
    def __init__(self, config: Dict[str, Any] = None, resume_path: Optional[str] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型
        self.model = TwoPhasePINN(self.config).to(self.device)
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        self.data_generator = DataGenerator(self.config, self.device)
        self.physics_loss = PhysicsLoss(self.device)
        self.physics_constraints = PhysicsConstraints()
        
        # Stage 1 模型 - 使用统一配置路径
        if HAS_APERTURE:
            from src.config import CONFIG_PATH
            self.stage1_model = EnhancedApertureModel(config_path=str(CONFIG_PATH))
        else:
            self.stage1_model = None
        self.enable_stage1_tutor = self.config.get("stage1_eta_tutor", True)  # 默认启用 Stage1 tutor
        dynamic_weight_cfg = self.config.get("dynamic_weight", {})
        enable_dynamic = dynamic_weight_cfg.get("enable", True)
        self.constraint_layer = PINNConstraintLayer(
            physics_constraints=self.physics_constraints,
            enable_dynamic_weight=enable_dynamic,
            dynamic_weight_config=dynamic_weight_cfg,
            config=self.config,
        )
        
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
        self.history = {
            "epoch": [],
            "loss": [],
            "interface": [],
            "physics": [],
            "lr": [],
            "pinn_continuity": [],
            "pinn_vof": [],
            "pinn_momentum_u": [],
            "pinn_momentum_v": [],
            "pinn_momentum_w": [],
            "pinn_surface_tension": [],
            "pinn_temporal_smoothness": [],
        }
        self.best_loss = float("inf")
        self.best_physics_loss = float("inf")  # 新增：物理损失最佳值
        self.patience_counter = 0
        self.early_stop_patience = training_cfg.get("early_stop_patience", 5000)
        
        # 阶段3才开始记录 best_loss（物理约束完整后）
        self.best_loss_recording_start = self.stage2_epochs  # 从阶段3开始记录
        self.start_epoch = 0
        
        # 输出目录与续训
        if resume_path is not None and os.path.exists(resume_path):
            logger.info(f"正在从 {resume_path} 恢复训练...")
            ckpt = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.best_loss = ckpt.get("best_loss", self.best_loss)
            self.best_physics_loss = ckpt.get("best_physics_loss", self.best_physics_loss)
            self.history = ckpt.get("history", self.history)
            self.start_epoch = ckpt.get("epoch", -1) + 1
            # 续训时，总 epochs = 起始 epoch + 新增 epochs
            self.epochs = self.start_epoch + self.epochs
            self.output_dir = os.path.dirname(resume_path)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"恢复成功，从 Epoch {self.start_epoch} 继续，将训练到 Epoch {self.epochs}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # [2026-01-16] 使用统一路径管理，输出到 outputs/train/
            from src.config.paths import get_output_dir
            self.output_dir = str(get_output_dir(f"train/pinn_{timestamp}"))
        
        # 添加文件日志处理器，将日志同时输出到文件
        log_file = os.path.join(self.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")
            
        # 预先保存一份配置
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """
        保存训练检查点
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_physics_loss": self.best_physics_loss,
            "history": self.history,
            "config": self.config,
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.output_dir, "latest_model.pth")
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，则额外保存
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            # 同时也保留一个带 epoch 的副本，防止覆盖
            if epoch % 1000 == 0:
                torch.save(checkpoint, os.path.join(self.output_dir, f"best_model_epoch_{epoch}.pth"))
        
        # 如果是最终模型
        if is_final:
            final_path = os.path.join(self.output_dir, "final_model.pth")
            torch.save(checkpoint, final_path)
            
    def _plot_curves(self, epoch: int = None):
        """
        绘制并保存训练曲线
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(self.history["epoch"], self.history["loss"], label='Total Loss')
            ax.semilogy(self.history["epoch"], self.history["interface"], label='Interface Loss', alpha=0.7)
            ax.semilogy(self.history["epoch"], self.history["physics"], label='Physics Loss', alpha=0.7)
            ax.semilogy(self.history["epoch"], self.history["pinn_continuity"], label='PINN C', alpha=0.35)
            ax.semilogy(self.history["epoch"], self.history["pinn_vof"], label='PINN VOF', alpha=0.35)
            ax.semilogy(self.history["epoch"], self.history["pinn_momentum_u"], label='PINN Mu', alpha=0.25)
            ax.semilogy(self.history["epoch"], self.history["pinn_momentum_v"], label='PINN Mv', alpha=0.25)
            ax.semilogy(self.history["epoch"], self.history["pinn_momentum_w"], label='PINN Mw', alpha=0.25)
            ax.semilogy(self.history["epoch"], self.history["pinn_surface_tension"], label='PINN ST', alpha=0.25)
            ax.semilogy(self.history["epoch"], self.history["pinn_temporal_smoothness"], label='PINN T', alpha=0.25)
            
            # 标记阶段
            if len(self.history["epoch"]) > 0:
                if self.stage1_epochs < self.history["epoch"][-1]:
                    ax.axvline(x=self.stage1_epochs, color='r', linestyle='--', alpha=0.5, label='Stage 2')
                if self.stage2_epochs < self.history["epoch"][-1]:
                    ax.axvline(x=self.stage2_epochs, color='g', linestyle='--', alpha=0.5, label='Stage 3')
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            title = "Training Loss"
            if epoch is not None:
                title += f" (Epoch {epoch})"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filename = "training_curve.png" if epoch is None else f"training_curve_epoch_{epoch}.png"
            # 始终更新最新的曲线图
            plt.savefig(os.path.join(self.output_dir, "training_curve.png"), dpi=150)
            # 阶段性保存带 epoch 的曲线图
            if epoch is not None and epoch % 5000 == 0:
                plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
            
            plt.close()
        except Exception as e:
            logger.warning(f"保存训练曲线失败: {e}")

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
            # 阶段2：平滑引入连续性和VOF，并以很小权重预热 NS
            progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
            smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))  # S形曲线
            
            ns_base = physics_cfg.get("ns_weight", 0.01)
            ns_scale = 0.05  # 阶段2仅使用 5% 的 NS 权重，避免过早主导
            
            return {
                "continuity": physics_cfg.get("continuity_weight", 0.1) * smooth_factor * 0.1,
                "vof": physics_cfg.get("vof_weight", 0.1) * smooth_factor * 0.1,
                "ns": ns_base * smooth_factor * ns_scale,
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
    
    def get_stage1_weight_factor(self, epoch: int) -> float:
        """
        获取 Stage1 约束的退火权重
        """
        training_cfg = self.config.get("training", {}) if isinstance(self.config.get("training", {}), dict) else {}
        anneal_start = self.stage2_epochs
        anneal_span = int(training_cfg.get("stage1_tutor_anneal_span", 10000))
        anneal_span = max(1000, anneal_span)
        anneal_end = anneal_start + anneal_span

        min_factor = float(training_cfg.get("stage1_tutor_min_factor", 0.20))
        min_factor = float(np.clip(min_factor, 0.0, 1.0))
        
        if epoch < anneal_start:
            return 1.0
        elif epoch > anneal_end:
            return min_factor
        else:
            # 使用余弦退火，后期下降更平滑
            progress = (epoch - anneal_start) / (anneal_end - anneal_start)
            return min_factor + 0.5 * (1.0 - min_factor) * (1 + np.cos(np.pi * progress))

    def compute_losses(self, data: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """
        计算所有损失 - 模块化重构
        
        φ = 1: 油墨, φ = 0: 极性液体
        """
        losses = {}
        physics_cfg = self.config.get("physics", {})
        physics_weights = self.get_physics_weights(epoch)
        stage1_factor = self.get_stage1_weight_factor(epoch)  # Stage1 约束退火
        
        current_stage = 1 if epoch < self.stage1_epochs else (2 if epoch < self.stage2_epochs else 3)
        
        # 1. 界面数据拟合损失 (核心)
        losses["interface"], data_fit_loss = self._compute_data_loss(data, physics_cfg, stage1_factor)
        
        # 2. 接触角边界条件损失
        losses["contact_angle"] = self._compute_contact_angle_loss(data)
        
        # 3. 初始条件 & 壁面边界条件损失
        ib_losses = self._compute_initial_boundary_loss(data, physics_cfg)
        losses.update(ib_losses)
        
        # 4. 早期时间 & 零电压约束
        ez_losses = self._compute_early_zero_voltage_loss()
        losses.update(ez_losses)
        
        # 5. 单调性 & 电压响应约束
        mr_losses = self._compute_monotonicity_response_loss()
        losses.update(mr_losses)
        
        # 6. 开口率相关约束 (eta_ceiling, eta_stage1_early, eta_stage1_late, eta_monotonic)
        eta_losses = self._compute_eta_constraints_loss(stage1_factor)
        losses.update(eta_losses)
        
        # 6.5 φ 场空间分布约束 (防止模式坍塌)
        losses["phi_spatial"] = self._compute_phi_spatial_loss(stage1_factor)

        # 6.6 φ 场几何一致性约束（开口半径 + 边缘堆高）
        losses["phi_geometry"] = self._compute_phi_geometry_loss(stage1_factor)
        
        # 7. 体积守恒约束
        losses["volume_conservation"] = self._compute_volume_conservation_loss(stage1_factor)
        
        # 8. 物理方程损失
        if any(w > 0 for w in physics_weights.values()):
            phys_losses = self._compute_physics_equation_loss(physics_weights, data_fit_loss, epoch, current_stage, data=data)
            losses.update(phys_losses)
        
        # 9. 降压过程：开口率恢复约束（阶段2及以后）
        if epoch >= self.stage1_epochs:
            losses["eta_recovery"] = self.eta_recovery_constraint_loss()

        # 10. 连续性约束：升压→降压转换时刻必须连续（方案B新增）
        if epoch >= self.stage1_epochs:
            losses["continuity_transition"] = self._compute_continuity_transition_loss()

        # 计算总损失
        total_loss = torch.tensor(0.0, device=self.device)
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                # 检查损失是否有效
                if torch.isfinite(val):
                    total_loss = total_loss + val
                else:
                    logger.warning(f"检测到无效损失 {name}: {val.item()}, 将其忽略")
        
        losses["total"] = total_loss
        return losses

    def _compute_data_loss(self, data: Dict[str, torch.Tensor], physics_cfg: Dict, stage1_factor: float):
        """1. 界面数据拟合损失"""
        idx = torch.randperm(len(data["interface_points"]))[:self.batch_size]
        interface_pts = data["interface_points"][idx]
        interface_tgt = data["interface_targets"][idx]
        
        phi_pred = self.model(interface_pts)[:, 4]
        interface_loss = F.mse_loss(phi_pred, interface_tgt)
        
        base_weight = physics_cfg.get("interface_weight", 500.0)
        return interface_loss * base_weight * stage1_factor, interface_loss

    def _compute_contact_angle_loss(self, data: Dict[str, torch.Tensor]):
        """2. 接触角边界条件损失"""
        idx = torch.randperm(len(data["contact_points"]))[:self.batch_size // 2]
        contact_pts = data["contact_points"][idx].clone().requires_grad_(True)
        phi = self.model(contact_pts)[:, 4]
        
        grad_phi = torch.autograd.grad(phi.sum(), contact_pts, create_graph=True, retain_graph=True)[0]
        dphi_dz = grad_phi[:, 2]
        grad_mag = torch.sqrt(torch.sum(grad_phi[:, :3]**2, dim=1) + 1e-10)
        
        theta_rad = data["contact_theta"][idx] * np.pi / 180.0
        target_cos = torch.cos(theta_rad)
        actual_cos = dphi_dz / grad_mag
        
        interface_weight = torch.exp(-100 * (phi - 0.5)**2)
        return torch.mean((actual_cos - target_cos)**2 * interface_weight) * 100.0

    def _compute_initial_boundary_loss(self, data: Dict[str, torch.Tensor], physics_cfg: Dict):
        """3. 初始条件 & 壁面边界条件损失"""
        res = {}
        # IC
        idx_ic = torch.randperm(len(data["ic_points"]))[:self.batch_size // 4]
        pred_ic = self.model(data["ic_points"][idx_ic])
        ic_phi_loss = F.mse_loss(pred_ic[:, 4:5], data["ic_values"][idx_ic][:, 4:5])
        ic_vel_loss = F.mse_loss(pred_ic[:, :4], data["ic_values"][idx_ic][:, :4])
        res["ic"] = ic_phi_loss * physics_cfg.get("ic_weight", 100.0) + ic_vel_loss * 50.0
        
        # BC
        idx_bc = torch.randperm(len(data["bc_points"]))[:self.batch_size // 4]
        pred_bc = self.model(data["bc_points"][idx_bc])
        res["bc"] = F.mse_loss(pred_bc[:, :3], data["bc_values"][idx_bc][:, :3]) * physics_cfg.get("bc_weight", 50.0)
        res["bc"] += F.mse_loss(pred_bc[:, 4:5], data["bc_values"][idx_bc][:, 4:5]) * 30.0
        return res

    def _compute_early_zero_voltage_loss(self):
        """4. 早期时间 & 零电压约束"""
        res = {}
        n = self.batch_size // 4
        # Early Time
        x = torch.rand(n, device=self.device) * PHYSICS["Lx"]
        y = torch.rand(n, device=self.device) * PHYSICS["Ly"]
        z = torch.rand(n, device=self.device) * PHYSICS["h_ink"]
        t = torch.rand(n, device=self.device) * 0.002
        pts_early = torch.stack([x, y, z, torch.zeros(n, device=self.device), torch.rand(n, device=self.device)*30.0, t], dim=1)
        phi_early = self.model(pts_early)[:, 4]
        res["early_time"] = F.mse_loss(phi_early, torch.ones_like(phi_early)) * 300.0
        
        # Zero Voltage
        t_0v = torch.rand(n, device=self.device) * PHYSICS["t_max"]
        pts_0v = torch.stack([x, y, z, torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), t_0v], dim=1)
        phi_0v = self.model(pts_0v)[:, 4]
        res["zero_voltage"] = F.mse_loss(phi_0v, torch.ones_like(phi_0v)) * 500.0
        
        # Low Voltage
        v_low = torch.rand(n, device=self.device) * 5.0
        t_low = 0.005 + torch.rand(n, device=self.device) * 0.015
        pts_low = torch.stack([x, y, z, torch.zeros(n, device=self.device), v_low, t_low], dim=1)
        phi_low = self.model(pts_low)[:, 4]
        res["low_voltage"] = F.mse_loss(phi_low, torch.ones_like(phi_low)) * 300.0
        return res

    def _compute_monotonicity_response_loss(self):
        """5. 单调性 & 电压响应约束"""
        res = {}
        n = self.batch_size // 4
        cx, cy = PHYSICS["Lx"]/2, PHYSICS["Ly"]/2
        x = cx + (torch.rand(n, device=self.device)-0.5)*PHYSICS["Lx"]*0.4
        y = cy + (torch.rand(n, device=self.device)-0.5)*PHYSICS["Ly"]*0.4
        z = torch.rand(n, device=self.device) * PHYSICS["h_ink"]
        
        # Monotonicity
        v_mono = 10.0 + torch.rand(n, device=self.device)*20.0
        t1, t2 = torch.rand(n, device=self.device)*0.01, torch.rand(n, device=self.device)*0.01+0.002
        p1 = torch.stack([x, y, z, torch.zeros(n, device=self.device), v_mono, t1], dim=1)
        p2 = torch.stack([x, y, z, torch.zeros(n, device=self.device), v_mono, t2], dim=1)
        phi1, phi2 = self.model(p1)[:, 4], self.model(p2)[:, 4]
        res["monotonicity"] = torch.mean(F.relu(phi2 - phi1 + 0.02)**2) * 50.0
        
        # Voltage Response
        v1 = 5.0 + torch.rand(n, device=self.device)*10.0
        v2 = torch.clamp(v1 + 5.0 + torch.rand(n, device=self.device)*10.0, max=30.0)
        t_v = torch.full((n,), 0.015, device=self.device)
        pv1 = torch.stack([x, y, z, torch.zeros(n, device=self.device), v1, t_v], dim=1)
        pv2 = torch.stack([x, y, z, torch.zeros(n, device=self.device), v2, t_v], dim=1)
        phiv1, phiv2 = self.model(pv1)[:, 4], self.model(pv2)[:, 4]
        res["voltage_response"] = torch.mean(F.relu(phiv2 - phiv1 + 0.02)**2) * 100.0
        return res

    def _compute_eta_constraints_loss(self, stage1_factor: float):
        """6. 开口率相关约束 - 增强早期时间动态响应"""
        res = {}
        tau = PHYSICS.get("tau", 0.005)  # 5ms 响应时间
        
        # Ceiling
        v_cap = [10.0, 15.0, 20.0, 25.0, 30.0]
        t_cap = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
        eta_max = PHYSICS.get("eta_max", 0.85)
        loss_cap = 0
        count_cap = 0
        for v in v_cap:
            for t in t_cap:
                eta = self.compute_aperture_ratio(0.0, v, t, n_grid=25)
                loss_cap = loss_cap + F.relu(eta - eta_max) ** 2
                count_cap += 1
        if count_cap > 0:
            res["eta_ceiling"] = loss_cap * 200.0 / count_cap
        else:
            res["eta_ceiling"] = torch.tensor(0.0, device=self.device)
        
        # Stage1 Match - 覆盖完整时间范围，随机采样以消除 artifact
        if self.stage1_model and self.enable_stage1_tutor:
            # 随机采样 (0-40ms)，覆盖暂态和稳态
            t_samples = np.random.uniform(0.001, 0.040, 15).tolist()
            v_match = [10.0, 15.0, 20.0, 25.0, 30.0]
            
            loss_s1 = 0
            count = 0
            
            for v in v_match:
                for t in t_samples:
                    _, eta_s1 = self.stage1_model.theta_eta_from_triad(0.0, v, t)
                    eta_pred = self.compute_aperture_ratio(0.0, v, t, n_grid=25)
                    loss_s1 += (eta_pred - eta_s1)**2
                    count += 1
            
            # 使用适中的权重
            res["eta_stage1"] = loss_s1 * 400.0 * stage1_factor / count
            
        # Monotonic
        v_smooth = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        # 随机采样单调性检查点
        t_on = sorted(np.random.uniform(0.001, 0.020, 5).tolist())
        t_off = sorted(np.random.uniform(0.001, 0.040, 5).tolist())
        loss_m, count_m = 0, 0
        for v in v_smooth:
            e_on = [self.compute_aperture_ratio(0.0, v, t, n_grid=20) for t in t_on]
            for i in range(len(e_on)-1):
                loss_m += F.relu(e_on[i] - e_on[i+1] + 0.005)**2
                count_m += 1
            e_off = [self.compute_aperture_ratio(v, 0.0, t, n_grid=20) for t in t_off]
            for i in range(len(e_off)-1):
                loss_m += F.relu(e_off[i+1] - e_off[i] + 0.005)**2
                count_m += 1
        res["eta_monotonic"] = loss_m * 50.0 / count_m
        
        return res

    def _compute_phi_spatial_loss(self, stage1_factor: float):
        """6.5 φ 场空间分布约束 - 防止模式坍塌
        
        问题：模型容易学到 φ 整体偏低（0.4-0.6），导致开口率 100%
        
        物理约束：
        - 中心区域（r < r_open）：φ → 0（透明，极性液体）
        - 边缘区域（r > r_open）：φ → 1（油墨）
        - 界面过渡区：φ 在 0-1 之间平滑过渡
        
        实现：
        1. 对于给定的 (V, t)，计算 Stage1 预测的开口率 η
        2. 根据 η 计算开口半径 r_open = sqrt(η * Lx * Ly / π)
        3. 强制中心点 φ < 0.3，边缘点 φ > 0.7
        """
        if self.stage1_model is None:
            return torch.tensor(0.0, device=self.device)
        
        Lx, Ly, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["h_ink"]
        cx, cy = Lx / 2, Ly / 2
        
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        # 测试不同电压和时间（随机采样）
        test_cases = [
            (0.0, 10.0, float(np.random.uniform(0.010, 0.020))),  # 低电压
            (0.0, 20.0, float(np.random.uniform(0.010, 0.020))),  # 中电压
            (0.0, 30.0, float(np.random.uniform(0.005, 0.012))),  # 高电压早期
            (0.0, 30.0, float(np.random.uniform(0.018, 0.030))),  # 高电压稳态
        ]
        
        n_pts = 20  # 每个区域的采样点数
        
        for V_from, V_to, t_since in test_cases:
            # 获取 Stage1 预测的开口率
            _, eta_s1 = self.stage1_model.theta_eta_from_triad(V_from, V_to, t_since)
            
            if eta_s1 < 0.05:  # 几乎没有开口，跳过
                continue
            
            # 计算开口半径
            r_open = np.sqrt(eta_s1 * Lx * Ly / np.pi)
            
            # 中心区域采样（r < r_open * 0.5）
            r_center = r_open * 0.3
            if r_center > 5e-6:  # 至少 5μm
                theta_angles = torch.rand(n_pts, device=self.device) * 2 * np.pi
                r_samples = torch.rand(n_pts, device=self.device) * r_center
                x_center = cx + r_samples * torch.cos(theta_angles)
                y_center = cy + r_samples * torch.sin(theta_angles)
                z_center = torch.zeros((n_pts,), device=self.device)
                
                pts_center = torch.stack([
                    x_center, y_center, z_center,
                    torch.full((n_pts,), V_from, device=self.device),
                    torch.full((n_pts,), V_to, device=self.device),
                    torch.full((n_pts,), t_since, device=self.device)
                ], dim=1)
                
                phi_center = self.model(pts_center)[:, 4]
                # 中心应该是透明的：φ < 0.3
                loss = loss + torch.mean(F.relu(phi_center - 0.3)**2)
                count += 1
            
            # 边缘区域采样（r > r_open * 1.5，但在像素内）
            r_edge_min = min(r_open * 1.5, min(Lx, Ly) / 2 * 0.8)
            r_edge_max = min(Lx, Ly) / 2 * 0.95
            
            if r_edge_max > r_edge_min:
                theta_angles = torch.rand(n_pts, device=self.device) * 2 * np.pi
                r_samples = r_edge_min + torch.rand(n_pts, device=self.device) * (r_edge_max - r_edge_min)
                x_edge = cx + r_samples * torch.cos(theta_angles)
                y_edge = cy + r_samples * torch.sin(theta_angles)
                # 确保在像素范围内
                x_edge = torch.clamp(x_edge, 1e-6, Lx - 1e-6)
                y_edge = torch.clamp(y_edge, 1e-6, Ly - 1e-6)
                z_edge = torch.zeros((n_pts,), device=self.device)
                
                pts_edge = torch.stack([
                    x_edge, y_edge, z_edge,
                    torch.full((n_pts,), V_from, device=self.device),
                    torch.full((n_pts,), V_to, device=self.device),
                    torch.full((n_pts,), t_since, device=self.device)
                ], dim=1)
                
                phi_edge = self.model(pts_edge)[:, 4]
                # 边缘应该是油墨：φ > 0.7
                loss = loss + torch.mean(F.relu(0.7 - phi_edge)**2)
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 权重：随着训练进行逐渐增加
        base_weight = 500.0  # 较高的权重，防止模式坍塌
        return loss * base_weight * stage1_factor / count

    def _compute_phi_geometry_loss(self, stage1_factor: float):
        if self.stage1_model is None:
            return torch.tensor(0.0, device=self.device)

        Lx, Ly, Lz, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"], PHYSICS["h_ink"]
        v0 = Lx * Ly * h_ink
        cx, cy = Lx / 2, Ly / 2

        test_cases = [
            (0.0, 20.0, 0.015),
            (0.0, 20.0, 0.020),
            (0.0, 30.0, 0.010),
            (0.0, 30.0, 0.020),
        ]

        n_pts = 256
        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for V_from, V_to, t_since in test_cases:
            _, eta_s1 = self.stage1_model.theta_eta_from_triad(V_from, V_to, t_since)
            eta_s1 = float(np.clip(eta_s1, 0.0, 0.95))
            if eta_s1 < 0.02:
                continue

            r_open = float(np.sqrt(eta_s1 * Lx * Ly / np.pi))
            r_open = min(r_open, 0.95 * min(cx, cy))

            ink_area = max(Lx * Ly - np.pi * r_open**2, 1e-12)
            h_edge = float(min(v0 / ink_area, Lz))

            margin = 3e-6
            r_center = max(0.0, r_open - margin)

            if r_center > 5e-6:
                theta = torch.rand(n_pts, device=self.device) * (2 * np.pi)
                rr = torch.sqrt(torch.rand(n_pts, device=self.device)) * r_center
                x = cx + rr * torch.cos(theta)
                y = cy + rr * torch.sin(theta)
                z = torch.rand(n_pts, device=self.device) * min(h_ink, Lz)
                pts = torch.stack(
                    [
                        x,
                        y,
                        z,
                        torch.full((n_pts,), float(V_from), device=self.device),
                        torch.full((n_pts,), float(V_to), device=self.device),
                        torch.full((n_pts,), float(t_since), device=self.device),
                    ],
                    dim=1,
                )
                phi = self.model(pts)[:, 4]
                loss = loss + torch.mean(torch.relu(phi - 0.2) ** 2)
                count += 1

            r_edge_min = min(r_open + margin, 0.9 * min(cx, cy))
            r_edge_max = 0.95 * min(cx, cy)
            if r_edge_max > r_edge_min and h_edge > 1e-7:
                theta = torch.rand(n_pts, device=self.device) * (2 * np.pi)
                rr = r_edge_min + torch.rand(n_pts, device=self.device) * (r_edge_max - r_edge_min)
                x = cx + rr * torch.cos(theta)
                y = cy + rr * torch.sin(theta)
                x = torch.clamp(x, 1e-9, Lx - 1e-9)
                y = torch.clamp(y, 1e-9, Ly - 1e-9)

                z_ink = torch.rand(n_pts, device=self.device) * min(h_edge * 0.8, Lz)
                pts_ink = torch.stack(
                    [
                        x,
                        y,
                        z_ink,
                        torch.full((n_pts,), float(V_from), device=self.device),
                        torch.full((n_pts,), float(V_to), device=self.device),
                        torch.full((n_pts,), float(t_since), device=self.device),
                    ],
                    dim=1,
                )
                phi_ink = self.model(pts_ink)[:, 4]
                loss = loss + torch.mean(torch.relu(0.8 - phi_ink) ** 2)
                count += 1

                z_top_min = min(h_edge * 1.1, Lz * 0.99)
                if z_top_min < Lz:
                    z_polar = z_top_min + torch.rand(n_pts, device=self.device) * (Lz - z_top_min)
                    pts_polar = torch.stack(
                        [
                            x,
                            y,
                            z_polar,
                            torch.full((n_pts,), float(V_from), device=self.device),
                            torch.full((n_pts,), float(V_to), device=self.device),
                            torch.full((n_pts,), float(t_since), device=self.device),
                        ],
                        dim=1,
                    )
                    phi_polar = self.model(pts_polar)[:, 4]
                    loss = loss + torch.mean(torch.relu(phi_polar - 0.2) ** 2)
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)

        base_weight = 200.0
        keep_factor = 0.2 + 0.8 * stage1_factor
        return loss * base_weight * keep_factor / count

    def _compute_volume_conservation_loss(self, stage1_factor: float):
        training_cfg = self.config.get("training", {}) if isinstance(self.config.get("training", {}), dict) else {}
        n_vol = int(training_cfg.get("volume_n_vol", 30000))
        n_vol = max(2000, n_vol)
        Lx, Ly, Lz, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"], PHYSICS["h_ink"]
        v0 = Lx * Ly * h_ink
        v_domain = Lx * Ly * Lz

        loss_vol = torch.tensor(0.0, device=self.device)
        tests = []
        # 随机采样，避免固定时间点 (如 14ms) 的过拟合和 artifact
        # 稳态时间：足够长的时间后
        t_steady = np.random.uniform(0.020, 0.050, 3).tolist()
        
        # 升压时间：覆盖早期和中期，均匀分布
        t_on = np.random.uniform(0.002, 0.025, 5).tolist()
        
        # 降压时间：覆盖早期和中期，均匀分布
        t_off = np.random.uniform(0.002, 0.025, 4).tolist()

        tests = []
        for v in [0.0, 10.0, 20.0, 25.0, 30.0]:
            for t in t_steady:
                tests.append((float(v), float(v), float(t)))

        for t in t_on:
            tests.append((0.0, 30.0, float(t)))
            tests.append((0.0, 25.0, float(t)))

        for t in t_off:
            tests.append((30.0, 0.0, float(t)))
            tests.append((25.0, 0.0, float(t)))

        x = torch.rand(n_vol, device=self.device) * Lx
        y = torch.rand(n_vol, device=self.device) * Ly
        z = torch.rand(n_vol, device=self.device) * Lz
        xyz = torch.stack([x, y, z], dim=1)

        for v_from, v_to, t_since in tests:
            n = xyz.shape[0]
            pts = torch.cat(
                [
                    xyz,
                    torch.full((n, 1), float(v_from), device=self.device),
                    torch.full((n, 1), float(v_to), device=self.device),
                    torch.full((n, 1), float(t_since), device=self.device),
                ],
                dim=1,
            )
            phi = self.model(pts)[:, 4]
            phi = torch.clamp(phi, 0.0, 1.0)
            v_curr = v_domain * phi.mean()
            rel_error = (v_curr - v0) / (v0 + 1e-12)
            loss_vol = loss_vol + rel_error**2

        base_weight = float(training_cfg.get("volume_base_weight", 2000.0))
        stage_weight = 0.2 + (1.0 - float(stage1_factor))
        return loss_vol * base_weight * stage_weight / max(1, len(tests))

    def _compute_continuity_transition_loss(self):
        """
        10. 连续性约束：升压→降压转换时刻必须连续

        物理要求：
        升压结束 (0→V, t=T) 的 φ场 应该等于 降压开始 (V→0, t=0) 的 φ场

        检查点：
        - V = [10, 20, 30] V
        - t = [10, 20, 30] ms
        """
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]

        # 采样空间点 (增加采样点数以减少波动)
        n = min(4000, self.batch_size)
        x = torch.rand(n, device=self.device) * Lx
        y = torch.rand(n, device=self.device) * Ly

        # 在油墨层附近加密采样（60%）+ 均匀采样（40%）
        n_ink = int(n * 0.6)
        z_ink = torch.rand(n_ink, device=self.device) * PHYSICS["h_ink"] * 2
        z_uniform = torch.rand(n - n_ink, device=self.device) * Lz
        z = torch.cat([z_ink, z_uniform])

        # 使用固定时间点采样，减少蒙特卡洛噪声
        t_rise_samples = np.linspace(0.005, 0.030, 6).tolist()
        V_samples = [10.0, 20.0, 30.0]
        
        test_cases = []
        for V in V_samples:
            for t in t_rise_samples:
                test_cases.append((float(V), float(t)))

        loss_continuity = torch.tensor(0.0, device=self.device)
        count = 0

        for V_high, t_rise_end in test_cases:
            # 升压结束状态: (0 → V_high, t = t_rise_end)
            pts_rise = torch.stack([
                x, y, z,
                torch.zeros(n, device=self.device),  # V_from = 0
                torch.full((n,), V_high, device=self.device),  # V_to = V_high
                torch.full((n,), t_rise_end, device=self.device)  # t_since
            ], dim=1)

            pred_rise = self.model(pts_rise)
            phi_rise = pred_rise[:, 4]

            # 降压开始状态: (V_high → 0, t = 0)
            pts_fall = torch.stack([
                x, y, z,
                torch.full((n,), V_high, device=self.device),  # V_from = V_high
                torch.zeros(n, device=self.device),  # V_to = 0
                torch.zeros(n, device=self.device)  # t_since = 0
            ], dim=1)

            pred_fall = self.model(pts_fall)
            phi_fall = pred_fall[:, 4]

            loss_phi = torch.mean((phi_rise - phi_fall) ** 2)
            loss_vel = torch.mean((pred_rise[:, 0:3] - pred_fall[:, 0:3]) ** 2)
            loss_continuity = loss_continuity + loss_phi + 0.1 * loss_vel
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)

        training_cfg = self.config.get("training", {}) if isinstance(self.config.get("training", {}), dict) else {}
        base_weight = float(training_cfg.get("continuity_transition_weight", 2000.0))
        return loss_continuity * base_weight / count

    def _compute_physics_equation_loss(self, weights: Dict, data_fit_loss: torch.Tensor, epoch: int, stage: int, data: Dict[str, torch.Tensor] = None):
        """
        8. 物理方程损失 - 统一使用 PINNConstraintLayer.compute_physics_loss
        
        [2025-12-31] 重构说明：
        - 移除对 PhysicsLoss 旧方法的调用（已标记 deprecated）
        - 统一通过 PINNConstraintLayer.compute_physics_loss 计算物理损失
        - 避免同一物理方程被两套实现重复计入损失
        
        [2026-02-04] 采样优化：
        - 优先使用 DataGenerator 生成的高质量配点 (domain_points, interface_points)
        - 混合随机点和数据点，确保 VOF 和 NS 方程在界面附近得到有效约束
        """
        res = {}
        
        # 1. 准备物理点
        # 策略：混合 50% 域内点 (包含界面附近) + 50% 随机点
        n_batch = min(2000, self.batch_size)  # 增加物理点采样数
        pts_list = []
        
        # (A) 从 DataGenerator 获取的点 (高质量，包含界面加密)
        if data is not None and "domain_points" in data:
            n_domain = len(data["domain_points"])
            idx_dom = torch.randperm(n_domain)[:n_batch // 2]
            pts_list.append(data["domain_points"][idx_dom])
            
        # (B) 从界面点获取 (VOF 关键区域)
        if data is not None and "interface_points" in data:
            n_interface = len(data["interface_points"])
            idx_int = torch.randperm(n_interface)[:n_batch // 4]
            pts_list.append(data["interface_points"][idx_int])
            
        # (C) 补充随机点 (确保全域覆盖)
        n_needed = n_batch - sum(len(p) for p in pts_list)
        if n_needed > 0:
            pts_random = torch.stack([
                torch.rand(n_needed, device=self.device) * PHYSICS["Lx"],
                torch.rand(n_needed, device=self.device) * PHYSICS["Ly"],
                torch.rand(n_needed, device=self.device) * PHYSICS["Lz"],
                torch.rand(n_needed, device=self.device) * 30.0, # V_from (随机)
                torch.rand(n_needed, device=self.device) * 30.0, # V_to
                torch.rand(n_needed, device=self.device) * PHYSICS["t_max"] # t_since
            ], dim=1)
            pts_list.append(pts_random)
            
        pts = torch.cat(pts_list, dim=0)
        
        try:
            phys_weights = {
                "continuity": float(weights.get("continuity", 0.0)),
                "vof": float(weights.get("vof", 0.0)),
                "momentum_u": float(weights.get("ns", 0.0)),
                "momentum_v": float(weights.get("ns", 0.0)),
                "momentum_w": float(weights.get("ns", 0.0)),
                "surface_tension": float(weights.get("surface_tension", 0.0)),
                "volume_conservation": 0.0,
                "explicit_volume": float(weights.get("explicit_volume_weight", 0.0)),
                "sharpening": float(weights.get("sharpening_weight", 0.0)),
            }
            
            # 使用 PhysicsLoss 计算损失
            losses = self.physics_loss.compute_total_loss(self.model, pts, weights=phys_weights)
            
            pinn_loss = losses.get("total", None)
            if isinstance(pinn_loss, torch.Tensor) and torch.isfinite(pinn_loss) and pinn_loss > 0:
                res["pinn_physics"] = pinn_loss
                for k in ["continuity", "vof", "momentum_u", "momentum_v", "momentum_w", "surface_tension", "temporal_smoothness"]:
                    if k in losses and isinstance(losses[k], torch.Tensor) and torch.isfinite(losses[k]):
                        res[f"pinn_{k}"] = losses[k]
            
        except Exception as e:
            logger.warning(f"Physics loss failed: {e}")
        return res
    
    def compute_aperture_ratio(self, V_from: float, V_to: float, t_since: float, n_grid: int = 30) -> torch.Tensor:
        """
        计算开口率：z=0 底面上“极性液体覆盖面积”的比例（软统计）
        
        Args:
            V_from: 跳变前电压 (V)
            V_to: 跳变后电压 (V)
            t_since: 跳变后经过的时间 (s)
            n_grid: 网格分辨率
        """
        if V_from is None:
            V_from = V_to
        
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
        
        x = torch.linspace(0, Lx, n_grid, device=self.device)
        y = torch.linspace(0, Ly, n_grid, device=self.device)
        z = torch.zeros((n_grid * n_grid,), device=self.device)
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack(
            [
                X.flatten(),
                Y.flatten(),
                z,
                torch.full((n_grid * n_grid,), V_from, device=self.device),
                torch.full((n_grid * n_grid,), V_to, device=self.device),
                torch.full((n_grid * n_grid,), t_since, device=self.device),
            ],
            dim=1,
        )
        
        phi = torch.clamp(self.model(points)[:, 4], 0.0, 1.0)

        eval_cfg = self.config.get("eval", {}) if isinstance(self.config.get("eval", {}), dict) else {}
        phi0 = float(eval_cfg.get("aperture_phi0", 0.3))
        eps = float(eval_cfg.get("aperture_eps", 0.05))
        eps = max(1e-6, eps)
        mask = torch.sigmoid((phi0 - phi) / eps)
        return mask.mean()
    
    def eta_recovery_constraint_loss(self, t_fall: float = 0.015, tau_recovery: float = 0.0075, weight: float = 100.0) -> torch.Tensor:
        """
        开口率恢复的软约束（降压后）
        
        物理：η(t) = η_at_fall × exp(-(t - t_fall) / τ_recovery)
        
        Args:
            t_fall: 降压时刻 (s)，默认 12ms
            tau_recovery: 恢复时间常数 (s)，默认 7.5ms
            weight: 损失权重
        """
        if self.stage1_model is not None and self.enable_stage1_tutor:
            t_max = PHYSICS.get("t_max", 0.10)
            max_t_since = max(0.0, min(t_max - t_fall, 0.05))
            t_since_samples = np.linspace(0.0, max_t_since, 8)
            loss = torch.tensor(0.0, device=self.device)
            for t_since in t_since_samples:
                t_since_val = float(t_since)
                _, eta_stage1 = self.stage1_model.theta_eta_from_triad(30.0, 0.0, t_since_val)
                eta_target = torch.tensor(float(eta_stage1), device=self.device)
                eta_pred = self.compute_aperture_ratio(V_from=30.0, V_to=0.0, t_since=t_since_val)
                loss = loss + (eta_pred - eta_target) ** 2
            return loss * weight / len(t_since_samples)
        
        eta_at_fall = self.compute_aperture_ratio(V_from=0.0, V_to=30.0, t_since=t_fall)
        t_samples = [t_fall + dt for dt in [0.002, 0.004, 0.006, 0.008]]
        loss = torch.tensor(0.0, device=self.device)
        for t in t_samples:
            t_since = t - t_fall
            eta_target = eta_at_fall * torch.exp(torch.tensor(-t_since / tau_recovery, device=self.device))
            eta_pred = self.compute_aperture_ratio(V_from=30.0, V_to=0.0, t_since=t_since)
            loss = loss + (eta_pred - eta_target) ** 2
        return loss * weight / len(t_samples)
    
    def fine_tune_lbfgs(self, max_iter=5000):
        """L-BFGS 二阶优化微调"""
        if not self.config.get("training", {}).get("use_lbfgs", False):
            return

        logger.info("=" * 60)
        logger.info(f"开始 L-BFGS 微调 (max_iter={max_iter})...")
        logger.info("=" * 60)
        
        # 重新生成一批高质量数据
        data = self.data_generator.generate_all_data()
        
        # L-BFGS 优化器
        optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(), 
            lr=1.0, 
            max_iter=max_iter,
            max_eval=int(max_iter * 1.25),
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )
        
        pbar = tqdm(total=max_iter, desc="L-BFGS Tuning")
        
        def closure():
            optimizer_lbfgs.zero_grad()
            # 使用当前 epoch (训练结束后的状态)
            losses = self.compute_losses(data, self.epochs)
            loss = losses["total"]
            
            if torch.isfinite(loss):
                loss.backward()
                
                # 记录
                current_loss = loss.item()
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    # 实时保存最佳模型
                    torch.save({
                        'epoch': self.epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer_lbfgs.state_dict(),
                        'loss': self.best_loss,
                        'config': self.config
                    }, os.path.join(self.output_dir, "best_model_lbfgs.pth"))
                
                pbar.set_postfix({"Loss": f"{current_loss:.4e}"})
                pbar.update(1)
                
            return loss
            
        try:
            optimizer_lbfgs.step(closure)
        except Exception as e:
            logger.error(f"L-BFGS 优化过程出错: {e}")
            
        pbar.close()
        logger.info(f"L-BFGS 微调完成! 最终最佳 Loss: {self.best_loss:.6e}")

    def train(self):
        """训练主循环"""
        logger.info("=" * 60)
        logger.info("开始两相流 PINN 训练")
        logger.info("=" * 60)
        
        data = self.data_generator.generate_all_data()
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            # 学习率预热
            if epoch < self.warmup_epochs:
                warmup_lr = self.warmup_start_lr + (self.lr - self.warmup_start_lr) * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # 检查训练阶段
            if epoch == self.stage1_epochs:
                logger.info(f"\n进入阶段 2：引入连续性和VOF约束")
            elif epoch == self.stage2_epochs:
                logger.info(f"\n进入阶段 3：完整物理约束")
            
            self.model.train()
            self.optimizer.zero_grad()
            
            losses = self.compute_losses(data, epoch)
            total_loss = losses["total"]
            
            # 增加稳定性检查：只有在损失有效且大于 0 时才进行反向传播
            if torch.isfinite(total_loss) and total_loss > 0:
                total_loss.backward()
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            else:
                logger.warning(f"Epoch {epoch}: 检测到无效总损失 {total_loss.item()}, 跳过本次优化步")
            
            # 记录历史 (使用更健壮的获取方式)
            def get_val(d, k): return d.get(k, torch.tensor(0.0)).item()
            
            # [2025-12-31] 物理损失统一从 pinn_physics 获取
            # 旧的 continuity, vof, ns, surface_tension 已不再单独计算
            physics_loss = get_val(losses, "pinn_physics")
            interface_loss = get_val(losses, "interface")
            
            self.history["epoch"].append(epoch)
            self.history["loss"].append(total_loss.item())
            self.history["interface"].append(interface_loss)
            self.history["physics"].append(physics_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["pinn_continuity"].append(get_val(losses, "pinn_continuity"))
            self.history["pinn_vof"].append(get_val(losses, "pinn_vof"))
            self.history["pinn_momentum_u"].append(get_val(losses, "pinn_momentum_u"))
            self.history["pinn_momentum_v"].append(get_val(losses, "pinn_momentum_v"))
            self.history["pinn_momentum_w"].append(get_val(losses, "pinn_momentum_w"))
            self.history["pinn_surface_tension"].append(get_val(losses, "pinn_surface_tension"))
            self.history["pinn_temporal_smoothness"].append(get_val(losses, "pinn_temporal_smoothness"))
            
            # 定期评估与保存
            if epoch % 100 == 0:
                # 预热期后才使用 scheduler
                if epoch >= self.warmup_epochs and torch.isfinite(total_loss):
                    self.scheduler.step(total_loss)
                
                current_loss = total_loss.item()
                
                # 计算物理损失（用于评估物理合理性）
                # 优先使用统一后的 pinn_physics
                physics_loss_val = losses.get("pinn_physics", torch.tensor(0.0)).item()
                if physics_loss_val == 0.0:
                    # 兼容旧逻辑（以防万一）
                    physics_loss_val = sum(
                        losses.get(k, torch.tensor(0.0)).item() 
                        for k in ["volume_conservation", "continuity", "vof", "ns"]
                    )
                
                # 只有在阶段3（完整物理约束）之后才更新 best_loss
                # 这样确保保存的模型是物理上合理的
                if epoch >= self.best_loss_recording_start:
                    is_best = current_loss < self.best_loss
                    if is_best:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                        logger.info(f"  ✓ 新最佳损失: {current_loss:.4e} (epoch {epoch})")
                    else:
                        self.patience_counter += 100
                    
                    # 同时跟踪物理损失最佳值
                    if physics_loss_val < self.best_physics_loss:
                        self.best_physics_loss = physics_loss_val
                else:
                    # 阶段1/2：不更新 best_loss，但仍保存检查点
                    is_best = False
                    self.patience_counter = 0  # 阶段1/2不计入早停
                
                # 保存检查点 (包括最新和可能的最佳)
                self._save_checkpoint(epoch, is_best=is_best)
                
                elapsed = time.time() - start_time
                
                # 构建详细的损失字符串
                physics_str = ""
                if "low_voltage" in losses:
                    physics_str += f" | LV: {losses['low_voltage'].item():.2e}"
                if "volume_conservation" in losses:
                    vol_val = losses['volume_conservation'].item()
                    # 体积守恒：显示趋势箭头
                    vol_trend = "↑" if vol_val > 2000 else ("→" if vol_val > 1000 else "↓")
                    physics_str += f" | Vol: {vol_val:.2e}{vol_trend}"
                if "contact_angle" in losses:
                    physics_str += f" | θ: {losses['contact_angle'].item():.2e}"
                if "eta_stage1_early" in losses:
                    physics_str += f" | ηE: {losses['eta_stage1_early'].item():.2e}"
                if "phi_spatial" in losses:
                    physics_str += f" | φS: {losses['phi_spatial'].item():.2e}"
                
                # 物理场损失 (优先检查 pinn_ 前缀)
                continuity_val = losses.get("pinn_continuity", losses.get("continuity"))
                if continuity_val is not None:
                    physics_str += f" | C: {continuity_val.item():.2e}"
                    
                interface_val = losses.get("interface")
                if interface_val is not None:
                    physics_str += f" | IF: {interface_val.item():.2e}"
                
                # 显示当前阶段
                stage_str = f"S{1 if epoch < self.stage1_epochs else (2 if epoch < self.stage2_epochs else 3)}"
                
                logger.info(
                    f"Epoch {epoch:5d} [{stage_str}] | Loss: {current_loss:.4e}{physics_str} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s"
                )

                # 每 1000 轮保存一次中间训练曲线，防止丢失
                if epoch % 1000 == 0 and epoch > 0:
                    self._plot_curves(epoch=epoch)
            
            # 早停检查
            if self.patience_counter >= self.early_stop_patience:
                logger.info(f"\n早停触发：连续 {self.early_stop_patience} 轮损失未下降")
                break
        
        # 训练结束，保存最终结果
        self._save_checkpoint(epoch, is_final=True)
        self._plot_curves()
        
        # [2026-01-30] L-BFGS 微调
        if self.config.get("training", {}).get("use_lbfgs", False):
            lbfgs_iter = self.config.get("training", {}).get("lbfgs_iter", 5000)
            self.fine_tune_lbfgs(max_iter=lbfgs_iter)
        
        # 调用专业评估套件生成全套图表
        try:
            from evaluate import PINNEvaluator
            evaluator = PINNEvaluator()
            best_ckpt = os.path.join(self.output_dir, "best_model.pth")
            if os.path.exists(best_ckpt):
                model, _ = evaluator.load_model(best_ckpt)
                if model:
                    evaluator.plot_dashboard(model, os.path.join(self.output_dir, "pro_dashboard.png"), model_name=self.output_dir)
                    evaluator.plot_phi_grid(model, os.path.join(self.output_dir, "phi_grid_evolution.png"))
                    evaluator.plot_interface_3d(model, os.path.join(self.output_dir, "interface_3d_steady.png"), 30.0, 0.02, 30.0)
                    logger.info(f"✅ 专业评估套件已生成: {self.output_dir}/")
        except Exception as e:
            logger.error(f"生成专业评估仪表盘失败: {e}")
        
        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳损失: {self.best_loss:.6e}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)
    
    def visualize(self):
        """
        可视化结果 - 已弃用，由 evaluate.py 替代
        """
        pass


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EWP 两相流 PINN 训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint 路径，用于续训")
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
    trainer = Trainer(config, resume_path=args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
