"""
接触角边界条件模块

基于 Young-Lippmann 方程计算理论接触角
"""

import logging
from typing import Dict, Optional
import numpy as np
import torch

logger = logging.getLogger("EWP-ContactAngle")

# 默认物理参数
DEFAULT_PHYSICS = {
    "theta0": 120.0,        # 初始接触角 (度)
    "epsilon_r": 4.0,       # 相对介电常数
    "d_dielectric": 4e-7,   # 介电层厚度 (m)
    "sigma": 0.030,         # 界面张力 (N/m)
}


class ContactAngleLoss:
    """
    接触角边界条件损失
    
    基于 Young-Lippmann 方程计算理论接触角，
    并与从 phi 梯度预测的接触角进行比较。
    
    Young-Lippmann 方程:
        cos(θ) = cos(θ₀) + (ε₀εᵣV²)/(2γd)
    
    其中:
        θ₀: 初始接触角
        ε₀: 真空介电常数
        εᵣ: 相对介电常数
        V: 施加电压
        γ: 界面张力
        d: 介电层厚度
    """
    
    def __init__(self, physics_params: Optional[Dict] = None, device: torch.device = None):
        physics = physics_params or DEFAULT_PHYSICS
        self.device = device or torch.device("cpu")
        
        self.theta0 = physics.get("theta0", 120.0)
        self.epsilon_r = physics.get("epsilon_r", 4.0)
        self.d_dielectric = physics.get("d_dielectric", 4e-7)
        self.gamma = physics.get("sigma", 0.030)
        self.epsilon_0 = 8.854e-12  # 真空介电常数
    
    def young_lippmann_angle(self, V: torch.Tensor) -> torch.Tensor:
        """
        计算 Young-Lippmann 平衡接触角
        
        Args:
            V: 电压张量 (batch,)
            
        Returns:
            平衡接触角 (度)
        """
        # cos(θ₀)
        theta0_rad = torch.deg2rad(torch.tensor(self.theta0, device=V.device))
        cos_theta0 = torch.cos(theta0_rad)
        
        # 电润湿项: (ε₀εᵣV²)/(2γd)
        ew_term = (self.epsilon_0 * self.epsilon_r * V**2) / (
            2 * self.gamma * self.d_dielectric
        )
        
        # cos(θ_eq) = cos(θ₀) + ew_term
        cos_theta_eq = torch.clamp(cos_theta0 + ew_term, -1.0, 1.0)
        
        # θ_eq = arccos(cos(θ_eq))
        theta_eq = torch.rad2deg(torch.acos(cos_theta_eq))
        
        return theta_eq
    
    def compute_predicted_angle(self, phi_grad: torch.Tensor) -> torch.Tensor:
        """
        从 phi 梯度计算预测接触角
        
        接触角定义为界面法向量与垂直方向的夹角。
        
        Args:
            phi_grad: phi 梯度张量 (batch, 3) - (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
            
        Returns:
            预测接触角 (度)
        """
        # 水平方向梯度分量
        grad_xy = torch.sqrt(phi_grad[:, 0]**2 + phi_grad[:, 1]**2 + 1e-10)
        
        # 垂直方向梯度分量
        grad_z = phi_grad[:, 2]
        
        # 接触角 = arctan(grad_xy / |grad_z|)
        theta = torch.atan2(grad_xy, torch.abs(grad_z) + 1e-10)
        
        return torch.rad2deg(theta)
    
    def compute_loss(
        self,
        phi: torch.Tensor,
        phi_grad: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        计算接触角损失
        
        只在界面点 (0.1 < φ < 0.9) 计算损失。
        
        Args:
            phi: phi 值张量 (batch,)
            phi_grad: phi 梯度张量 (batch, 3)
            V: 电压张量 (batch,)
            
        Returns:
            接触角损失
        """
        # 界面掩码：只在 0.1 < φ < 0.9 的点计算
        interface_mask = (phi > 0.1) & (phi < 0.9)
        
        if interface_mask.sum() == 0:
            return torch.tensor(0.0, device=phi.device)
        
        # 提取界面点
        phi_grad_interface = phi_grad[interface_mask]
        V_interface = V[interface_mask]
        
        # 计算预测和理论接触角
        theta_pred = self.compute_predicted_angle(phi_grad_interface)
        theta_theory = self.young_lippmann_angle(V_interface)
        
        # MSE 损失
        loss = torch.mean((theta_pred - theta_theory)**2)
        
        return loss
    
    @staticmethod
    def get_weight_schedule(
        epoch: int,
        stage2_epochs: int,
        base_weight: float = 10.0
    ) -> float:
        """
        获取接触角损失权重（平滑过渡）
        
        在 stage 2 之后逐渐增加权重。
        
        Args:
            epoch: 当前 epoch
            stage2_epochs: stage 2 结束的 epoch
            base_weight: 基础权重
            
        Returns:
            当前权重
        """
        if epoch < stage2_epochs:
            return 0.0
        
        # 平滑过渡：使用 tanh 函数
        progress = min(1.0, (epoch - stage2_epochs) / 5000.0)
        smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))
        
        return base_weight * smooth_factor
