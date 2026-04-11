"""
LSTM-PINN 物理损失函数

包含：
- 连续性方程：∇·u = 0
- VOF 方程：∂φ/∂t + u·∇φ = 0
- 体积守恒：∫φdV = const
- 界面约束：φ ∈ [0, 1]，过渡平滑

复用现有 pinn_two_phase.py 中的物理损失计算逻辑
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# 物理常量 - 从统一配置模块导入
try:
    from src.config import PHYSICS
except ImportError:
    # 回退：如果配置模块不可用，使用本地定义
    PHYSICS = {
        # 几何参数
        "Lx": 174e-6,           # 像素宽度 (m)
        "Ly": 174e-6,           # 像素高度 (m)
        "Lz": 20e-6,            # 围堰高度 (m)
        "h_ink": 3e-6,          # 油墨层厚度 (m)
        
        # 流体属性
        "rho_oil": 800.0,       # 油墨密度 (kg/m³)
        "rho_polar": 1000.0,    # 极性液体密度 (kg/m³)
        "mu_oil": 0.003,        # 油墨粘度 (Pa·s)
        "mu_polar": 0.001,      # 极性液体粘度 (Pa·s)
        "sigma": 0.045,         # 油墨-极性液体界面张力 (N/m)
    }


class LSTMPINNPhysicsLoss(nn.Module):
    """
    LSTM-PINN 物理损失函数
    
    与 LSTMPINNModel 配合使用，计算物理约束损失
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化物理损失
        
        Args:
            config: 配置字典，包含物理参数和损失权重
        """
        super().__init__()
        
        config = config or {}
        physics_config = config.get("physics", {})
        
        # 物理参数
        self.Lx = physics_config.get("Lx", PHYSICS["Lx"])
        self.Ly = physics_config.get("Ly", PHYSICS["Ly"])
        self.Lz = physics_config.get("Lz", PHYSICS["Lz"])
        self.h_ink = physics_config.get("h_ink", PHYSICS["h_ink"])
        
        self.rho_oil = physics_config.get("rho_oil", PHYSICS["rho_oil"])
        self.rho_polar = physics_config.get("rho_polar", PHYSICS["rho_polar"])
        self.mu_oil = physics_config.get("mu_oil", PHYSICS["mu_oil"])
        self.mu_polar = physics_config.get("mu_polar", PHYSICS["mu_polar"])
        self.sigma = physics_config.get("sigma", PHYSICS["sigma"])
        
        # 特征尺度（用于归一化）
        self.L_char = self.Lx
        self.U_char = 1e-3  # 特征速度 (m/s)
        self.T_char = self.L_char / self.U_char
        
        # 损失权重
        weights = config.get("loss_weights", {})
        self.continuity_weight = weights.get("continuity", 0.5)
        self.vof_weight = weights.get("vof", 0.5)
        self.ns_weight = weights.get("ns", 0.1)  # NS 方程权重
        self.surface_tension_weight = weights.get("surface_tension", 0.1)  # 表面张力权重
        self.volume_weight = weights.get("volume", 1.0)
        self.interface_weight = weights.get("interface", 0.1)
        
        # 初始油墨体积（用于体积守恒）
        self.initial_ink_volume = self.Lx * self.Ly * self.h_ink
    
    def compute_continuity_loss(
        self,
        velocity: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        计算连续性方程损失：∇·u = 0
        
        Args:
            velocity: (batch, 3) 速度场 (u, v, w)
            spatial_coords: (batch, 3) 空间坐标 (x, y, z)，需要 requires_grad=True
        
        Returns:
            连续性损失标量
        """
        if not spatial_coords.requires_grad:
            spatial_coords = spatial_coords.clone().requires_grad_(True)
        
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        # 计算速度梯度
        def grad_component(y, x, idx):
            """计算 y 对 x 的第 idx 个分量的梯度"""
            grad_y = torch.autograd.grad(
                y.sum(), x,
                create_graph=True,
                retain_graph=True
            )[0]
            return grad_y[:, idx]
        
        # ∂u/∂x + ∂v/∂y + ∂w/∂z
        du_dx = grad_component(u, spatial_coords, 0)
        dv_dy = grad_component(v, spatial_coords, 1)
        dw_dz = grad_component(w, spatial_coords, 2)
        
        div_u = du_dx + dv_dy + dw_dz
        
        # 归一化
        div_u_norm = div_u * self.L_char / self.U_char
        
        return torch.mean(div_u_norm ** 2)
    
    def compute_vof_loss(
        self,
        phi: torch.Tensor,
        velocity: torch.Tensor,
        spatial_coords: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 VOF 方程损失：∂φ/∂t + u·∇φ = 0
        
        Args:
            phi: (batch, 1) φ 值
            velocity: (batch, 3) 速度场 (u, v, w)
            spatial_coords: (batch, 3) 空间坐标，需要 requires_grad=True
            time: (batch, 1) 时间，需要 requires_grad=True
        
        Returns:
            VOF 损失标量
        """
        if not spatial_coords.requires_grad:
            spatial_coords = spatial_coords.clone().requires_grad_(True)
        if not time.requires_grad:
            time = time.clone().requires_grad_(True)
        
        phi_flat = phi.squeeze(-1)
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        # ∂φ/∂t
        dphi_dt = torch.autograd.grad(
            phi_flat.sum(), time,
            create_graph=True,
            retain_graph=True
        )[0].squeeze(-1)
        
        # ∇φ
        grad_phi = torch.autograd.grad(
            phi_flat.sum(), spatial_coords,
            create_graph=True,
            retain_graph=True
        )[0]
        dphi_dx, dphi_dy, dphi_dz = grad_phi[:, 0], grad_phi[:, 1], grad_phi[:, 2]
        
        # u·∇φ
        advection = u * dphi_dx + v * dphi_dy + w * dphi_dz
        
        # VOF 残差
        vof_residual = dphi_dt + advection
        
        # 归一化
        vof_residual_norm = vof_residual * self.T_char
        
        return torch.mean(vof_residual_norm ** 2)
    
    def compute_volume_conservation_loss(
        self,
        phi: torch.Tensor,
        spatial_coords: torch.Tensor,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        计算体积守恒损失：∫φdV = const
        
        使用蒙特卡洛积分估计油墨体积
        
        Args:
            phi: (batch, 1) φ 值
            spatial_coords: (batch, 3) 空间坐标
            n_samples: 用于积分的采样点数
        
        Returns:
            体积守恒损失标量
        """
        # 计算当前油墨体积（蒙特卡洛积分）
        # V_ink = ∫φdV ≈ (Lx × Ly × Lz / N) × Σφ
        domain_volume = self.Lx * self.Ly * self.Lz
        
        # 使用所有点估计体积
        estimated_volume = domain_volume * phi.mean()
        
        # 相对误差
        relative_error = (estimated_volume - self.initial_ink_volume) / self.initial_ink_volume
        
        return relative_error ** 2
    
    def compute_interface_smoothness_loss(
        self,
        phi: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        计算界面平滑性损失
        
        约束 φ 在界面处平滑过渡，避免阶跃
        
        Args:
            phi: (batch, 1) φ 值
            spatial_coords: (batch, 3) 空间坐标，需要 requires_grad=True
        
        Returns:
            界面平滑性损失标量
        """
        if not spatial_coords.requires_grad:
            spatial_coords = spatial_coords.clone().requires_grad_(True)
        
        phi_flat = phi.squeeze(-1)
        
        # ∇φ
        grad_phi = torch.autograd.grad(
            phi_flat.sum(), spatial_coords,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # |∇φ|
        grad_phi_mag = torch.sqrt((grad_phi ** 2).sum(dim=-1) + 1e-10)
        
        # 界面指示函数（在 φ ≈ 0.5 处）
        interface_indicator = torch.exp(-50 * (phi_flat - 0.5) ** 2)
        
        # 界面处的梯度应该是有限的（不能太大）
        # 典型界面宽度 ~3μm，所以 |∇φ| ~ 1 / 3e-6 ~ 3e5
        max_gradient = 1.0 / 3e-6
        gradient_penalty = torch.relu(grad_phi_mag - max_gradient) / max_gradient
        
        return torch.mean(interface_indicator * gradient_penalty ** 2)
    
    def compute_ns_loss(
        self,
        phi: torch.Tensor,
        velocity: torch.Tensor,
        pressure: torch.Tensor,
        spatial_coords: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Navier-Stokes 方程残差（完整版，含压力梯度）
        
        ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
        
        Args:
            phi: (batch, 1) φ 值
            velocity: (batch, 3) 速度场 (u, v, w)
            pressure: (batch, 1) 压力场 p
            spatial_coords: (batch, 3) 空间坐标，需要 requires_grad=True
            time: (batch, 1) 时间，需要 requires_grad=True
        
        Returns:
            NS 损失标量
        """
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        p = pressure.squeeze(-1)
        phi_flat = phi.squeeze(-1)
        
        # 混合流体属性
        rho = phi_flat * self.rho_oil + (1 - phi_flat) * self.rho_polar
        mu = phi_flat * self.mu_oil + (1 - phi_flat) * self.mu_polar
        
        # 一阶导数
        def grad_scalar(y, x):
            return torch.autograd.grad(
                y.sum(), x, create_graph=True, retain_graph=True
            )[0]
        
        # 速度对空间的梯度
        grad_u = grad_scalar(u, spatial_coords)
        grad_v = grad_scalar(v, spatial_coords)
        grad_w = grad_scalar(w, spatial_coords)
        
        u_x, u_y, u_z = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        v_x, v_y, v_z = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]
        w_x, w_y, w_z = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2]
        
        # 压力梯度
        grad_p = grad_scalar(p, spatial_coords)
        p_x, p_y, p_z = grad_p[:, 0], grad_p[:, 1], grad_p[:, 2]
        
        # 速度对时间的梯度
        u_t = grad_scalar(u, time).squeeze(-1)
        v_t = grad_scalar(v, time).squeeze(-1)
        w_t = grad_scalar(w, time).squeeze(-1)
        
        # 对流项
        u_conv = u * u_x + v * u_y + w * u_z
        v_conv = u * v_x + v * v_y + w * v_z
        w_conv = u * w_x + v * w_y + w * w_z
        
        # 二阶导数（粘性项）
        u_xx = grad_scalar(u_x, spatial_coords)[:, 0]
        u_yy = grad_scalar(u_y, spatial_coords)[:, 1]
        u_zz = grad_scalar(u_z, spatial_coords)[:, 2]
        
        v_xx = grad_scalar(v_x, spatial_coords)[:, 0]
        v_yy = grad_scalar(v_y, spatial_coords)[:, 1]
        v_zz = grad_scalar(v_z, spatial_coords)[:, 2]
        
        w_xx = grad_scalar(w_x, spatial_coords)[:, 0]
        w_yy = grad_scalar(w_y, spatial_coords)[:, 1]
        w_zz = grad_scalar(w_z, spatial_coords)[:, 2]
        
        u_laplacian = u_xx + u_yy + u_zz
        v_laplacian = v_xx + v_yy + v_zz
        w_laplacian = w_xx + w_yy + w_zz
        
        # NS 残差（完整版，含压力梯度）
        # ρ(∂u/∂t + u·∇u) + ∇p - μ∇²u = 0
        ns_u = rho * (u_t + u_conv) + p_x - mu * u_laplacian
        ns_v = rho * (v_t + v_conv) + p_y - mu * v_laplacian
        ns_w = rho * (w_t + w_conv) + p_z - mu * w_laplacian
        
        # 归一化：ρU²/L
        scale = self.rho_polar * self.U_char**2 / self.L_char
        ns_u_norm = ns_u / (scale + 1e-10)
        ns_v_norm = ns_v / (scale + 1e-10)
        ns_w_norm = ns_w / (scale + 1e-10)
        
        # 使用 log1p 稳定大损失
        ns_loss = torch.mean(ns_u_norm**2 + ns_v_norm**2 + ns_w_norm**2)
        return torch.log1p(ns_loss)
    
    def compute_surface_tension_loss(
        self,
        phi: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        计算表面张力 CSF 模型残差
        
        CSF (Continuum Surface Force):
        F_st = σκn δ_s
        
        其中：
        - κ = -∇·n 是曲率
        - n = ∇φ/|∇φ| 是界面法向量
        - δ_s = |∇φ| 是界面 delta 函数
        
        约束：界面处的曲率应该是合理的（不能太大）
        """
        phi_flat = phi.squeeze(-1)
        
        def grad_scalar(y, x):
            return torch.autograd.grad(
                y.sum(), x, create_graph=True, retain_graph=True
            )[0]
        
        # ∇φ
        grad_phi = grad_scalar(phi_flat, spatial_coords)
        phi_x, phi_y, phi_z = grad_phi[:, 0], grad_phi[:, 1], grad_phi[:, 2]
        
        # |∇φ|
        grad_phi_mag = torch.sqrt(phi_x**2 + phi_y**2 + phi_z**2 + 1e-10)
        
        # 界面指示函数（在 φ ≈ 0.5 处）
        interface_indicator = torch.exp(-50 * (phi_flat - 0.5)**2)
        
        # 二阶导数
        phi_xx = grad_scalar(phi_x, spatial_coords)[:, 0]
        phi_yy = grad_scalar(phi_y, spatial_coords)[:, 1]
        phi_zz = grad_scalar(phi_z, spatial_coords)[:, 2]
        laplacian_phi = phi_xx + phi_yy + phi_zz
        
        # 曲率 κ ≈ -∇²φ / |∇φ|
        kappa = -laplacian_phi / (grad_phi_mag + 1e-10)
        
        # 表面张力约束：界面处的曲率应该是合理的
        # 典型曲率 ~ 1/R，R ~ 10μm，所以 κ ~ 1e5
        kappa_normalized = kappa * self.L_char  # 无量纲化
        st_residual = interface_indicator * kappa_normalized**2
        
        return torch.mean(st_residual)
    
    def forward(
        self,
        phi: torch.Tensor,
        velocity: Optional[torch.Tensor],
        spatial_coords: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        pressure: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有物理损失
        
        Args:
            phi: (batch, 1) φ 值
            velocity: (batch, 3) 速度场，可选
            spatial_coords: (batch, 3) 空间坐标
            time: (batch, 1) 时间，可选
            pressure: (batch, 1) 压力场，可选
        
        Returns:
            损失字典，包含各项损失和总损失
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=phi.device)
        
        # 体积守恒损失（总是计算）
        volume_loss = self.compute_volume_conservation_loss(phi, spatial_coords)
        losses["volume"] = volume_loss
        total_loss = total_loss + self.volume_weight * volume_loss
        
        # 界面平滑性损失
        if spatial_coords.requires_grad:
            interface_loss = self.compute_interface_smoothness_loss(phi, spatial_coords)
            losses["interface"] = interface_loss
            total_loss = total_loss + self.interface_weight * interface_loss
            
            # 表面张力损失
            try:
                st_loss = self.compute_surface_tension_loss(phi, spatial_coords)
                losses["surface_tension"] = st_loss
                total_loss = total_loss + self.surface_tension_weight * st_loss
            except RuntimeError:
                losses["surface_tension"] = torch.tensor(0.0, device=phi.device)
        
        # 连续性损失（需要速度场）
        if velocity is not None and spatial_coords.requires_grad:
            continuity_loss = self.compute_continuity_loss(velocity, spatial_coords)
            losses["continuity"] = continuity_loss
            total_loss = total_loss + self.continuity_weight * continuity_loss
        
        # VOF 损失（需要速度场和时间）
        if velocity is not None and time is not None:
            if spatial_coords.requires_grad and time.requires_grad:
                vof_loss = self.compute_vof_loss(phi, velocity, spatial_coords, time)
                losses["vof"] = vof_loss
                total_loss = total_loss + self.vof_weight * vof_loss
                
                # NS 方程损失（需要压力场）
                if pressure is not None:
                    try:
                        ns_loss = self.compute_ns_loss(phi, velocity, pressure, spatial_coords, time)
                        losses["ns"] = ns_loss
                        total_loss = total_loss + self.ns_weight * ns_loss
                    except RuntimeError:
                        losses["ns"] = torch.tensor(0.0, device=phi.device)
        
        losses["total"] = total_loss
        
        return losses


class SimplifiedPhysicsLoss(nn.Module):
    """
    简化版物理损失（用于初期训练）
    
    只包含：
    - φ 范围约束：φ ∈ [0, 1]
    - 体积守恒：∫φdV = const
    
    不需要速度场，适合 LSTM-PINN 初期训练
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        config = config or {}
        physics_config = config.get("physics", {})
        
        self.Lx = physics_config.get("Lx", PHYSICS["Lx"])
        self.Ly = physics_config.get("Ly", PHYSICS["Ly"])
        self.Lz = physics_config.get("Lz", PHYSICS["Lz"])
        self.h_ink = physics_config.get("h_ink", PHYSICS["h_ink"])
        
        self.initial_ink_volume = self.Lx * self.Ly * self.h_ink
        
        weights = config.get("loss_weights", {})
        self.volume_weight = weights.get("volume", 1.0)
        self.range_weight = weights.get("range", 0.1)
    
    def forward(
        self,
        phi: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算简化物理损失
        
        Args:
            phi: (batch, 1) φ 值
            spatial_coords: (batch, 3) 空间坐标
        
        Returns:
            损失字典
        """
        losses = {}
        
        # φ 范围约束（sigmoid 已经保证，这里是额外惩罚）
        range_loss = torch.mean(torch.relu(-phi) + torch.relu(phi - 1))
        losses["range"] = range_loss
        
        # 体积守恒
        domain_volume = self.Lx * self.Ly * self.Lz
        estimated_volume = domain_volume * phi.mean()
        relative_error = (estimated_volume - self.initial_ink_volume) / self.initial_ink_volume
        volume_loss = relative_error ** 2
        losses["volume"] = volume_loss
        
        # 总损失
        total_loss = self.range_weight * range_loss + self.volume_weight * volume_loss
        losses["total"] = total_loss
        
        return losses
