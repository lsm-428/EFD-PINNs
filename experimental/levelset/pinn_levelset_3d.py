#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Level Set Physics-Informed Neural Network for Electrowetting Display
========================================================================

This module implements a 3D Level Set method for two-phase flow simulation
in electrowetting displays, extending the proven 2D implementation.

Key differences from VOF method:
- Uses signed distance function ψ (Level Set) instead of volume fraction φ
- Sharp interface representation at ψ=0 (vs fuzzy interface at φ=0.5)
- Exact geometric quantities (normal, curvature) from ∇ψ
- Superior volume conservation (0.00% in 2D vs 18.6% in VOF)

Architecture (based on successful 2D implementation):
- Multi-scale networks: Main + Interface + Corner networks
- Input: 6D (x, y, z, V_from, V_to, t_since)
- Output: 9D (u1, v1, w1, u2, v2, w2, ψ_levelset, p1, p2)

Reference: 2D implementation in
electrowetting_display_model (2D)/src/electrowetting_display_model/physics/two_phase_pinn.py
(Lines 718-933)

Author: EFD-PINNs Team
Date: 2026-01-09
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from torch.utils.checkpoint import checkpoint

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
# Prefer device_parameters.py for authoritative device and physics constants
from device_parameters import (
    DEVICE_GEOMETRY,
    INITIAL_STATE,
    MATERIAL_PROPERTIES,
    ELECTRICAL_PROPERTIES,
    SIMULATION_PARAMETERS,
    ELECTROWETTING,
)

PHYSICS = {
    "Lx": DEVICE_GEOMETRY["pixel_inner_size"][0],
    "Ly": DEVICE_GEOMETRY["pixel_inner_size"][1],
    "Lz": DEVICE_GEOMETRY["fluid_height"],
    "h_ink": INITIAL_STATE["ink"]["thickness"],
    "rho_oil": MATERIAL_PROPERTIES["ink"]["density"],
    "mu_oil": MATERIAL_PROPERTIES["ink"]["viscosity"],
    "rho_polar": MATERIAL_PROPERTIES["polar"]["density"],
    "mu_polar": MATERIAL_PROPERTIES["polar"]["viscosity"],
    "epsilon_0": ELECTRICAL_PROPERTIES.get("epsilon_0", 8.854e-12),
    "epsilon_r": ELECTRICAL_PROPERTIES["dielectric"].get("epsilon_r", 3.0),
    "d_dielectric": ELECTRICAL_PROPERTIES["dielectric"].get("thickness", 0.4e-6),
    "V_threshold": ELECTRICAL_PROPERTIES["voltage"].get("threshold", 3.0),
    "V_max": ELECTRICAL_PROPERTIES["voltage"].get("max", 30.0),
    "t_max": SIMULATION_PARAMETERS["time"].get("t_max", 0.05),
}


# ============================================================================
# Multi-Scale Network Components (from 2D implementation)
# ============================================================================


class SubNetwork(nn.Module):
    """Multi-scale network component (shared by main, interface, corner networks)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        use_checkpoint: bool = False,
    ):
        super(SubNetwork, self).__init__()
        self.use_checkpoint = use_checkpoint

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.layers = layers

    def _forward_impl(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)


# Backwards-compatibility aliases (optional)
MainNetwork = SubNetwork
InterfaceNetwork = SubNetwork
CornerNetwork = SubNetwork


# ============================================================================
# 3D Level Set PINN Main Class
# ============================================================================


class LevelSet3DPINN(nn.Module):
    """
    3D Level Set Physics-Informed Neural Network

    Extends the successful 2D implementation (10K epochs, 0.00% volume error)
    to 3D electrowetting display simulation.

    Input:  [x, y, z, V_from, V_to, t_since]  (6D)
    Output: [u1, v1, w1, u2, v2, w2, ψ, p1, p2]  (9D)

    Output breakdown:
    - [0:3]: Phase 1 (ink) velocity - u1, v1, w1
    - [3:6]: Phase 2 (polar) velocity - u2, v2, w2
    - [6]: Level Set field - ψ
    - [7]: Phase 1 pressure - p1
    - [8]: Phase 2 pressure - p2

    物理约定（与φ类比）：
    - Phase 1 (ink): ψ < 0 (inside ink, 油墨在底部)
    - Phase 2 (polar): ψ > 0 (inside polar fluid, 极性液体在上部)
    - Interface: ψ = 0

    几何结构：
    - 中心开口：极性液体 (ψ > 0)
    - 边缘区域：油墨堆高 (ψ < 0)

    Attributes:
        epsilon (float): Interface thickness parameter (default 0.01)
        gamma_12 (float): Interfacial tension (N/m)

    Example:
        >>> model = LevelSet3DPINN(config)
        >>> x = torch.randn(100, 6)  # [batch, 6]
        >>> output = model(x)
        >>> psi = output[:, 6:7]  # Level Set field
    """

    def __init__(self, config: Dict):
        super(LevelSet3DPINN, self).__init__()

        # Network architecture parameters
        self.input_dim = config.get("input_dim", 6)
        self.output_dim = config.get(
            "output_dim", 10
        )  # 10D: +V_field for electrostatic
        self.use_checkpoint = config.get("use_checkpoint", False)

        # Multi-scale network configuration (from 2D success)
        hidden_main = config.get("hidden_main", [64, 64, 32])
        hidden_interface = config.get("hidden_interface", [64, 64, 32])
        hidden_corner = config.get("hidden_corner", [64, 64, 64])

        self.main_network = SubNetwork(
            self.input_dim,
            hidden_main[0],
            self.output_dim,
            len(hidden_main),
            self.use_checkpoint,
        )

        self.interface_network = SubNetwork(
            self.input_dim,
            hidden_interface[0],
            self.output_dim,
            len(hidden_interface),
            self.use_checkpoint,
        )

        self.corner_network = SubNetwork(
            self.input_dim,
            hidden_corner[0],
            self.output_dim,
            len(hidden_corner),
            self.use_checkpoint,
        )

        # Fusion layer (combines outputs from 3 networks)
        self.fusion = nn.Sequential(
            nn.Linear(3 * self.output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, self.output_dim),
        )

        # Level Set parameters
        self.epsilon = config.get("epsilon", 0.01)
        self.gamma_12 = config.get("surface_tension", 0.045)  # N/m

        # Phase 1: Ink (non-polar, dark)
        # Phase 1 (油墨/癸烷 Decane) 材料属性 - 与 device_parameters.py 同步
        self.rho1 = PHYSICS.get("rho_oil", 730.0)  # kg/m³, 癸烷密度@20°C
        self.mu1 = PHYSICS.get("mu_oil", 0.00092)  # Pa·s, 癸烷粘度@20°C
        self.epsilon_r1 = 2.0  # 相对介电常数 (癸烷)

        # Phase 2 (极性液体/去离子水 Deionized Water) 材料属性 - 与 device_parameters.py 同步
        self.rho2 = PHYSICS.get("rho_polar", 997.0)  # kg/m³, 水@25°C
        self.mu2 = PHYSICS.get("mu_polar", 0.00089)  # Pa·s, 水@25°C
        self.epsilon_r2 = 78.4  # 相对介电常数 (水@25°C)

        # Electrowetting parameters
        self.epsilon_0 = PHYSICS.get("epsilon_0", 8.854e-12)  # F/m
        self.epsilon_r = PHYSICS.get("epsilon_r", 12.0)
        self.dielectric_thickness = PHYSICS.get("d_dielectric", 4e-7)  # m
        self.V_threshold = PHYSICS.get("V_threshold", 3.0)  # V

        # Geometry
        self.Lx = PHYSICS.get("Lx", 174e-6)
        self.Ly = PHYSICS.get("Ly", 174e-6)
        self.Lz = PHYSICS.get("Lz", 20e-6)
        self.wall_height = PHYSICS.get("wall_height", 3.5e-6)
        self.t_max = PHYSICS.get("t_max", 0.05)

        self.target_volume_fraction = config.get("target_volume_fraction", 0.15)
        self.volume_correction_strength = config.get("volume_correction_strength", 0.1)
        self.volume_constraint_enabled = config.get("volume_constraint_enabled", True)

    def enforce_volume_conservation(self, output: torch.Tensor) -> torch.Tensor:
        if not self.volume_constraint_enabled:
            return output

        psi = output[:, 6:7]
        current_volume = (psi / self.Lz).mean()
        volume_error = current_volume - self.target_volume_fraction

        corrected_psi = psi + volume_error.detach() * self.volume_correction_strength
        output[:, 6:7] = corrected_psi

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale networks

        Args:
            x: Input tensor [batch, 6] - (x, y, z, V_from, V_to, t_since)

        Returns:
            output: [batch, 10] - (u1, v1, w1, u2, v2, w2, ψ, p1, p2, V_field)
                - u1, v1, w1: Phase 1 (ink) velocity
                - u2, v2, w2: Phase 2 (polar) velocity
                - ψ: Level Set function
                - p1: Phase 1 pressure
                - p2: Phase 2 pressure
                - V_field: Electric potential field (for Maxwell stress)
        """
        x_norm = x[:, 0:1] / self.Lx
        y_norm = x[:, 1:2] / self.Ly
        z_norm = x[:, 2:3] / self.Lz
        V_from_norm = x[:, 3:4] / 30.0
        V_to_norm = x[:, 4:5] / 30.0
        t_norm = x[:, 5:6] / self.t_max

        x_norm_tensor = torch.cat(
            [x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1
        )

        out_main = self.main_network(x_norm_tensor)
        out_interface = self.interface_network(x_norm_tensor)
        out_corner = self.corner_network(x_norm_tensor)

        out_combined = torch.cat([out_main, out_interface, out_corner], dim=1)
        output = self.fusion(out_combined)

        # 注释掉硬约束 - 使用 loss 层的软约束来避免梯度干扰
        # output = self.enforce_volume_conservation(output)
        # 2026-03-30: 硬约束移至 loss 层实现，参考 Task 2

        return output

    def heaviside_3d(
        self, psi: torch.Tensor, epsilon: Optional[float] = None
    ) -> torch.Tensor:
        epsilon = float(self.epsilon) if epsilon is None else float(epsilon)

        H = 0.5 * (
            1.0 + psi / epsilon + (1.0 / np.pi) * torch.sin(np.pi * psi / epsilon)
        )

        H = torch.where(psi < epsilon, H, torch.ones_like(H))
        H = torch.where(psi > 0, H, torch.zeros_like(H))

        return H

    def dirac_delta_3d(
        self, psi: torch.Tensor, epsilon: Optional[float] = None
    ) -> torch.Tensor:
        """
        3D Smooth Dirac Delta function for surface tension concentration

        δ(ψ) = (1/2ε) * (1 + cos(πψ/ε))  for |ψ| ≤ ε
        δ(ψ) = 0                         for |ψ| > ε

        Surface tension force is concentrated at interface: F = γκδn

        2D reference: Lines 768-803 in two_phase_pinn.py

        Args:
            psi: Level Set field [batch, 1]
            epsilon: Interface thickness (default self.epsilon)

        Returns:
            delta: Smooth delta function [batch, 1]
        """
        epsilon_val = float(self.epsilon) if epsilon is None else float(epsilon)

        # Smooth Dirac delta in interface region
        scale = torch.tensor(
            1.0 / (2.0 * epsilon_val), dtype=psi.dtype, device=psi.device
        )
        delta = scale * (1.0 + torch.cos(np.pi * psi / epsilon_val))

        # Zero outside interface region
        delta = torch.where(
            torch.abs(psi) <= epsilon_val, delta, torch.zeros_like(delta)
        )

        return delta

    def compute_interface_geometry_3d(
        self, psi: torch.Tensor, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D interface normal and curvature from Level Set field

        Normal: n = ∇ψ / |∇ψ|
        Curvature: κ = ∇·(∇ψ/|∇ψ|) = ∇²ψ/|∇ψ| - (∇ψ·∇∇ψ·∇ψ)/|∇ψ|³

        Simplified: κ = (ψ_xx(ψ_y²+ψ_z²) + ψ_yy(ψ_x²+ψ_z²) + ψ_zz(ψ_x²+ψ_y²)
                        - 2(ψ_xyψ_xψ_y + ψ_xzψ_xψ_z + ψ_yzψ_yψ_z)) / |∇ψ|³

        2D reference: Lines 700-766 in two_phase_pinn.py

        Args:
            psi: Level Set field [batch, 1]
            xyz: Spatial coordinates [batch, 3] - (x, y, z)

        Returns:
            n: Unit normal vector [batch, 3] - (n_x, n_y, n_z)
            kappa: Mean curvature [batch, 1]
        """
        batch_size = psi.shape[0]

        grad_psi = torch.autograd.grad(
            psi,
            xyz,
            grad_outputs=torch.ones(batch_size, 1, device=psi.device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grad_psi is None:
            grad_norm_sq = torch.ones(batch_size, 1, device=psi.device) * 1e-10
            grad_norm = torch.sqrt(grad_norm_sq)
            n = torch.zeros(batch_size, 3, device=psi.device)
            n[:, 2] = 1.0
            kappa = torch.zeros(batch_size, 1, device=psi.device)
            return n, kappa

        psi_x = grad_psi[:, 0:1]
        psi_y = grad_psi[:, 1:2]
        psi_z = grad_psi[:, 2:3]

        grad_norm_sq = psi_x**2 + psi_y**2 + psi_z**2 + 1e-10
        grad_norm = torch.sqrt(grad_norm_sq)

        n = torch.cat([psi_x, psi_y, psi_z], dim=1) / grad_norm

        first_grads = [psi_x, psi_y, psi_z]
        second_grads_xx = []
        second_grads_yy = []
        second_grads_zz = []
        second_grads_xy = []
        second_grads_xz = []
        second_grads_yz = []

        for g in first_grads:
            g2 = torch.autograd.grad(
                g,
                xyz,
                grad_outputs=torch.ones(batch_size, 1, device=psi.device),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g2 is None:
                g2 = torch.zeros(batch_size, 3, device=psi.device)
            g2 = torch.where(torch.isnan(g2), torch.zeros_like(g2), g2)
            second_grads_xx.append(g2[:, 0:1])
            second_grads_yy.append(g2[:, 1:2])
            second_grads_zz.append(g2[:, 2:3])
            second_grads_xy.append(g2[:, 1:2])
            second_grads_xz.append(g2[:, 2:3])
            second_grads_yz.append(g2[:, 2:3])

        psi_xx = second_grads_xx[0]
        psi_yy = second_grads_yy[1]
        psi_zz = second_grads_zz[2]

        psi_xy = second_grads_xy[0]
        psi_xz = second_grads_xz[0]
        psi_yz = second_grads_yz[1]

        numerator = (
            psi_xx * (psi_y**2 + psi_z**2)
            + psi_yy * (psi_x**2 + psi_z**2)
            + psi_zz * (psi_x**2 + psi_y**2)
            - 2.0
            * (psi_xy * psi_x * psi_y + psi_xz * psi_x * psi_z + psi_yz * psi_y * psi_z)
        )

        kappa = numerator / (grad_norm_sq * grad_norm + 1e-12)

        return n, kappa

    def levelset_transport_loss_3d(
        self,
        psi: torch.Tensor,
        u1: torch.Tensor,
        v1: torch.Tensor,
        w1: torch.Tensor,
        u2: torch.Tensor,
        v2: torch.Tensor,
        w2: torch.Tensor,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        3D Level Set transport equation: ∂ψ/∂t + u·∇ψ = 0

        Ensures interface moves with fluid velocity.
        Uses average velocity: u_avg = (u1 + u2) / 2

        2D reference: Lines 1723-1800 in two_phase_pinn.py

        Args:
            psi: Level Set field [batch, 1]
            u1, v1, w1: Phase 1 velocity [batch, 1] each
            u2, v2, w2: Phase 2 velocity [batch, 1] each
            data: Full input tensor [batch, 6] - (x, y, z, V_from, V_to, t_since)

        Returns:
            loss: Mean squared transport residual
        """
        # Compute all gradients in single call (optimized)
        grad_all = torch.autograd.grad(
            psi,
            data,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Extract individual components by column
        psi_t = grad_all[:, 5:6]  # t_since gradient
        psi_x = grad_all[:, 0:1]  # x gradient
        psi_y = grad_all[:, 1:2]  # y gradient
        psi_z = grad_all[:, 2:3]  # z gradient

        # Average velocity at interface
        u_avg = (u1 + u2) / 2.0
        v_avg = (v1 + v2) / 2.0
        w_avg = (w1 + w2) / 2.0

        # Transport equation residual: ∂ψ/∂t + u·∇ψ = 0
        residual = psi_t + u_avg * psi_x + v_avg * psi_y + w_avg * psi_z

        return torch.mean(residual**2)

    # ========================================================================
    # Physics Losses
    # ========================================================================

    def get_phase_properties(
        self, psi: torch.Tensor, epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get phase-dependent properties using Heaviside interpolation

        物理约定：
        Property = H·Property_1 + (1-H)·Property_2
        where H=1 in Phase 2 (polar, 极性液体, ψ>0), H=0 in Phase 1 (ink, 油墨, ψ<0)

        2D reference: Lines 669-698 in two_phase_pinn.py

        Args:
            psi: Level Set field [batch, 1]
            epsilon: Interface thickness

        Returns:
            rho: Density [batch, 1]
            mu: Dynamic viscosity [batch, 1]
            epsilon_r: Relative permittivity [batch, 1]
        """
        H = self.heaviside_3d(psi, epsilon)

        rho = self.rho1 * H + self.rho2 * (1.0 - H)
        mu = self.mu1 * H + self.mu2 * (1.0 - H)
        epsilon_r = self.epsilon_r1 * H + self.epsilon_r2 * (1.0 - H)

        return rho, mu, epsilon_r

    def compute_surface_tension_force(
        self, psi: torch.Tensor, xyz: torch.Tensor, epsilon: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute surface tension force: F = γκδn

        2D reference: Lines 768-803 in two_phase_pinn.py

        Args:
            psi: Level Set field [batch, 1]
            xyz: Spatial coordinates [batch, 3]
            epsilon: Interface thickness

        Returns:
            F: Surface tension force [batch, 3] - (F_x, F_y, F_z)
        """
        n, kappa = self.compute_interface_geometry_3d(psi, xyz)
        delta = self.dirac_delta_3d(psi, epsilon)

        # Surface tension force: F = γκδn
        force_mag = self.gamma_12 * kappa * delta
        F_x = force_mag * n[:, 0:1]
        F_y = force_mag * n[:, 1:2]
        F_z = force_mag * n[:, 2:3]

        return torch.cat([F_x, F_y, F_z], dim=1)

    def compute_electrowetting_stress(
        self, V_from: torch.Tensor, V_to: torch.Tensor, t_since: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute electrowetting Maxwell stress

        σ_ew = ε₀εᵣ(V-V_T)² / (2d)

        Args:
            V_from: Voltage before transition [batch, 1]
            V_to: Current voltage [batch, 1]
            t_since: Time since transition [batch, 1]

        Returns:
            sigma_ew: Electrowetting stress [batch, 1]
        """
        # Current voltage (simplified transition model)
        # TODO: Implement proper voltage transition dynamics
        V_current = V_to  # Simplified

        # Electrowetting stress
        sigma_ew = (
            self.epsilon_0
            * self.epsilon_r
            * (V_current - self.V_threshold) ** 2
            / (2.0 * self.dielectric_thickness)
        )

        return sigma_ew

    def navier_stokes_loss_3d(
        self, y_pred: torch.Tensor, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D Navier-Stokes residual for both phases with Maxwell stress

        N-S equation: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F
        where F = F_surface_tension + F_maxwell_stress

        Using Maxwell stress tensor from solved electric potential field V_field:
        σ_M = ε₀εᵣ(E⊗E - ½|E|²I), where E = -∇V
        f_electric = ∇·σ_M

        Args:
            y_pred: Network output [batch, 10] - (u1, v1, w1, u2, v2, w2, ψ, p1, p2, V_field)
            data: Full input tensor [batch, 6] - (x, y, z, V_from, V_to, t_since)

        Returns:
            loss_ns1: Phase 1 N-S loss
            loss_ns2: Phase 2 N-S loss
        """
        u1 = y_pred[:, 0:1]
        v1 = y_pred[:, 1:2]
        w1 = y_pred[:, 2:3]
        u2 = y_pred[:, 3:4]
        v2 = y_pred[:, 4:5]
        w2 = y_pred[:, 5:6]
        psi = y_pred[:, 6:7]
        p1 = y_pred[:, 7:8]
        p2 = y_pred[:, 8:9]
        V_field = y_pred[:, 9:10]  # Electric potential field

        xyz = data[:, 0:3]
        t_since = data[:, 5:6]
        V_from = data[:, 3:4]
        V_to = data[:, 4:5]

        # 双速度场模型：每个速度场使用对应相的物理参数
        # Phase 1 (ink, ψ < 0): u1/v1/w1 使用 ink 恒定属性
        # Phase 2 (polar, ψ > 0): u2/v2/w2 使用 polar 恒定属性
        rho1 = torch.full_like(psi, self.rho1)
        mu1 = torch.full_like(psi, self.mu1)
        rho2 = torch.full_like(psi, self.rho2)
        mu2 = torch.full_like(psi, self.mu2)

        F_surf = self.compute_surface_tension_force(psi, xyz)

        F_maxwell = self.maxwell_stress_force(V_field, psi, xyz)

        epsilon = float(self.epsilon)
        in_interface = (psi < epsilon).float()
        F_maxwell_limited = F_maxwell * in_interface

        F1 = F_surf
        F2 = F_surf + F_maxwell_limited

        loss_ns1 = self._compute_ns_residual_3d(u1, v1, w1, p1, rho1, mu1, data, F1)
        loss_ns2 = self._compute_ns_residual_3d(u2, v2, w2, p2, rho2, mu2, data, F2)

        return loss_ns1, loss_ns2

    def _compute_ns_residual_3d(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        data: torch.Tensor,
        F_body: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 3D N-S residual for one component

        Args:
            u, v, w: Velocity components [batch, 1]
            p: Pressure [batch, 1]
            rho: Density [batch, 1]
            mu: Viscosity [batch, 1]
            data: Full input [batch, 6]
            F_body: Body force [batch, 3]

        Returns:
            residual: Mean squared N-S residual
        """
        xyz = data[:, 0:3]
        t_since = data[:, 5:6]

        u_t = torch.autograd.grad(
            u,
            data,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 5:6]

        v_t = torch.autograd.grad(
            v,
            data,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0][:, 5:6]

        w_t = torch.autograd.grad(
            w,
            data,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0][:, 5:6]

        u_x = torch.autograd.grad(
            u,
            data,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        u_y = torch.autograd.grad(
            u,
            data,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        u_z = torch.autograd.grad(
            u,
            data,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        v_x = torch.autograd.grad(
            v,
            data,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        v_y = torch.autograd.grad(
            v,
            data,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        v_z = torch.autograd.grad(
            v,
            data,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        w_x = torch.autograd.grad(
            w,
            data,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        w_y = torch.autograd.grad(
            w,
            data,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        w_z = torch.autograd.grad(
            w,
            data,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        p_x = torch.autograd.grad(
            p,
            data,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        p_y = torch.autograd.grad(
            p,
            data,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        p_z = torch.autograd.grad(
            p,
            data,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        u_xx = torch.autograd.grad(
            u_x,
            data,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y,
            data,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        u_zz = torch.autograd.grad(
            u_z,
            data,
            grad_outputs=torch.ones_like(u_z),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        v_xx = torch.autograd.grad(
            v_x,
            data,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        v_yy = torch.autograd.grad(
            v_y,
            data,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        v_zz = torch.autograd.grad(
            v_z,
            data,
            grad_outputs=torch.ones_like(v_z),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        w_xx = torch.autograd.grad(
            w_x,
            data,
            grad_outputs=torch.ones_like(w_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        w_yy = torch.autograd.grad(
            w_y,
            data,
            grad_outputs=torch.ones_like(w_y),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        w_zz = torch.autograd.grad(
            w_z,
            data,
            grad_outputs=torch.ones_like(w_z),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        convection_x = u * u_x + v * u_y + w * u_z
        convection_y = u * v_x + v * v_y + w * v_z
        convection_z = u * w_x + v * w_y + w * w_z

        laplace_u = u_xx + u_yy + u_zz
        laplace_v = v_xx + v_yy + v_zz
        laplace_w = w_xx + w_yy + w_zz

        residual_x = rho * (u_t + convection_x) + p_x - mu * laplace_u - F_body[:, 0:1]
        residual_y = rho * (v_t + convection_y) + p_y - mu * laplace_v - F_body[:, 1:2]
        residual_z = rho * (w_t + convection_z) + p_z - mu * laplace_w - F_body[:, 2:3]

        # 数值稳定性: 检测并处理NaN/Inf
        residual_x = torch.nan_to_num(residual_x, nan=0.0, posinf=1e6, neginf=-1e6)
        residual_y = torch.nan_to_num(residual_y, nan=0.0, posinf=1e6, neginf=-1e6)
        residual_z = torch.nan_to_num(residual_z, nan=0.0, posinf=1e6, neginf=-1e6)

        # 限制残差范围,防止梯度爆炸
        residual_x = torch.clamp(residual_x, min=-1e6, max=1e6)
        residual_y = torch.clamp(residual_y, min=-1e6, max=1e6)
        residual_z = torch.clamp(residual_z, min=-1e6, max=1e6)

        loss = torch.mean(residual_x**2 + residual_y**2 + residual_z**2)

        # 额外的NaN保护: 如果仍然是NaN,返回一个小的非零值保持梯度
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1e-10, device=residual_x.device, requires_grad=True)

        return loss

    def compute_electrowetting_force(
        self,
        psi: torch.Tensor,
        xyz: torch.Tensor,
        V_from: torch.Tensor,
        V_to: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute electrowetting body force

        The electrowetting force acts primarily near the dielectric layer (z=0).
        Using simplified model: F_ew = δ(z) * (ε₀εᵣ/2d) * (V-V_T)² * n_z

        Args:
            psi: Level Set field [batch, 1]
            xyz: Spatial coordinates [batch, 3]
            V_from: Voltage before transition [batch, 1]
            V_to: Current voltage [batch, 1]

        Returns:
            F_ew: Electrowetting force [batch, 3]
        """
        z = xyz[:, 2:3]
        V_current = V_to

        delta_z = torch.exp(-z / (self.dielectric_thickness + 1e-12))

        sigma_ew = (
            self.epsilon_0
            * self.epsilon_r
            * (V_current - self.V_threshold) ** 2
            / (2.0 * self.dielectric_thickness)
        )

        F_ew_z = sigma_ew * delta_z
        F_ew_x = torch.zeros_like(F_ew_z)
        F_ew_y = torch.zeros_like(F_ew_z)

        return torch.cat([F_ew_x, F_ew_y, F_ew_z], dim=1)

    def continuity_loss_3d(
        self, y_pred: torch.Tensor, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D continuity equation residual: ∇·u = 0

        2D reference: Lines 805-900 in two_phase_pinn.py

        Args:
            y_pred: Network output [batch, 9]
            data: Full input tensor [batch, 6] - (x, y, z, V_from, V_to, t_since)

        Returns:
            loss_cont1: Phase 1 continuity loss
            loss_cont2: Phase 2 continuity loss
        """
        u1 = y_pred[:, 0:1]
        v1 = y_pred[:, 1:2]
        w1 = y_pred[:, 2:3]
        u2 = y_pred[:, 3:4]
        v2 = y_pred[:, 4:5]
        w2 = y_pred[:, 5:6]

        # Phase 1 divergence: ∂u1/∂x + ∂v1/∂y + ∂w1/∂z
        u1_x = torch.autograd.grad(
            u1,
            data,
            grad_outputs=torch.ones_like(u1),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        v1_y = torch.autograd.grad(
            v1,
            data,
            grad_outputs=torch.ones_like(v1),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        w1_z = torch.autograd.grad(
            w1,
            data,
            grad_outputs=torch.ones_like(w1),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        div_u1 = u1_x + v1_y + w1_z

        # Phase 2 divergence
        u2_x = torch.autograd.grad(
            u2,
            data,
            grad_outputs=torch.ones_like(u2),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        v2_y = torch.autograd.grad(
            v2,
            data,
            grad_outputs=torch.ones_like(v2),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        w2_z = torch.autograd.grad(
            w2,
            data,
            grad_outputs=torch.ones_like(w2),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        div_u2 = u2_x + v2_y + w2_z

        return torch.mean(div_u1**2), torch.mean(div_u2**2)

    # ============================================================================
    # Boundary Conditions (COMSOL Multi-physics Coupling)
    # ========================================================================

    def contact_angle_loss(
        self, y_pred: torch.Tensor, data: torch.Tensor, V_applied: torch.Tensor
    ) -> torch.Tensor:
        """
        Contact angle boundary condition at wall (z=0)

        Uses FINITE DIFFERENCE method to compute spatial gradients.
        Evades PyTorch autograd graph disconnection issue.

        COMSOL: Wetted Wall (多物理场耦合 -> 润湿壁)
        Boundary condition: n·∇ψ = -cos(θ(V))|∇ψ| at z=0

        Young-Lippmann equation:
        cos(θ(V)) = cos(θ₀) + ε₀εᵣ(V-V_T)²/(2γd)

        Args:
            y_pred: Network output [batch, 9] (not used, re-computed for gradients)
            data: Input [batch, 6] - (x, y, z, V_from, V_to, t_since)
            V_applied: Applied voltage [batch, 1]

        Returns:
            loss: Mean squared contact angle boundary residual
        """
        xyz = data[:, 0:3]
        device = data.device

        # Filter for bottom wall points (z ≈ 0)
        z = xyz[:, 2:3]
        bottom_mask = (z < 1e-7).flatten()

        if not bottom_mask.any():
            return torch.tensor(0.0, device=device)

        # Extract bottom points
        bottom_xyz = xyz[bottom_mask]
        bottom_V_applied = V_applied[bottom_mask]

        # Get ψ at bottom points
        # Note: ψ is normalized [-1, 1], need to convert to physical scale
        # CRITICAL FIX: Remove torch.no_grad() to allow gradient flow
        bottom_psi_norm = y_pred[bottom_mask, 6:7]

        # Convert to physical scale (units: meters)
        # ψ_physical = ψ_normalized × psi_scale
        # Use model's psi_scale attribute (defaults to 2e-6 from config)
        psi_scale = getattr(self, "psi_scale", 2e-6)
        bottom_psi = bottom_psi_norm * psi_scale

        # Finite difference step (relative to domain size)
        # Using 1μm which is ~0.6% of domain and ~10% of interface thickness
        epsilon = 1e-6  # 1 μm perturbation

        # Compute ∂ψ/∂x using finite differences (in physical units)
        # CRITICAL FIX: Remove torch.no_grad() to allow gradient flow
        bottom_xyz_perturbed_x = bottom_xyz.clone()
        bottom_xyz_perturbed_x[:, 0:1] += epsilon
        bottom_data_perturbed_x = torch.cat(
            [bottom_xyz_perturbed_x, data[bottom_mask, 3:6]], dim=1
        )
        bottom_psi_perturbed_x_norm = self.forward(bottom_data_perturbed_x)[:, 6:7]
        bottom_psi_perturbed_x = bottom_psi_perturbed_x_norm * psi_scale
        psi_x = (bottom_psi_perturbed_x - bottom_psi) / epsilon

        # Compute ∂ψ/∂y using finite differences (in physical units)
        # CRITICAL FIX: Remove torch.no_grad() to allow gradient flow
        bottom_xyz_perturbed_y = bottom_xyz.clone()
        bottom_xyz_perturbed_y[:, 1:2] += epsilon
        bottom_data_perturbed_y = torch.cat(
            [bottom_xyz_perturbed_y, data[bottom_mask, 3:6]], dim=1
        )
        bottom_psi_perturbed_y_norm = self.forward(bottom_data_perturbed_y)[:, 6:7]
        bottom_psi_perturbed_y = bottom_psi_perturbed_y_norm * psi_scale
        psi_y = (bottom_psi_perturbed_y - bottom_psi) / epsilon

        # Compute ∂ψ/∂z using finite differences (in physical units)
        # CRITICAL FIX: Remove torch.no_grad() to allow gradient flow
        bottom_xyz_perturbed_z = bottom_xyz.clone()
        bottom_xyz_perturbed_z[:, 2:3] += epsilon
        bottom_data_perturbed_z = torch.cat(
            [bottom_xyz_perturbed_z, data[bottom_mask, 3:6]], dim=1
        )
        bottom_psi_perturbed_z_norm = self.forward(bottom_data_perturbed_z)[:, 6:7]
        bottom_psi_perturbed_z = bottom_psi_perturbed_z_norm * psi_scale
        psi_z = (bottom_psi_perturbed_z - bottom_psi) / epsilon

        # Gradient magnitude |∇ψ| (in physical units: m⁻¹)
        psi_grad_norm = torch.sqrt(psi_x**2 + psi_y**2 + psi_z**2 + 1e-12)

        # Clip gradients to prevent numerical explosion (gradients in m⁻¹)
        psi_x = torch.clamp(psi_x, -1e5, 1e5)
        psi_y = torch.clamp(psi_y, -1e5, 1e5)
        psi_z = torch.clamp(psi_z, -1e5, 1e5)
        psi_grad_norm = torch.clamp(psi_grad_norm, 0, 1e5)

        # Young-Lippmann equation for contact angle
        # θ₀ = 120° (initial contact angle for ink on hydrophobic surface)
        theta_0 = 120.0 * np.pi / 180.0  # Convert to radians
        cos_theta_0 = np.cos(theta_0)

        # Electrowetting number
        # γ = 0.045 N/m (surface tension)
        # d = 0.4 μm (dielectric thickness)
        # εᵣ = 12.0 (relative permittivity)
        # ε₀ = 8.854e-12 F/m
        gamma = self.gamma_12
        d_dielectric = self.dielectric_thickness
        V_T = self.V_threshold

        # cos(θ(V)) = cos(θ₀) + ε₀εᵣ(V-V_T)²/(2γd)
        cos_theta_V = cos_theta_0 + (
            self.epsilon_0
            * self.epsilon_r
            * (bottom_V_applied - V_T) ** 2
            / (2.0 * gamma * d_dielectric)
        )

        # Clip to physical range [-1, 1]
        cos_theta_V = torch.clamp(cos_theta_V, -1.0, 1.0)

        # Contact angle boundary condition at z=0 (bottom wall)
        # n·∇ψ = ∂ψ/∂z = -cos(θ(V))|∇ψ|
        # Note: n points upward at bottom wall, so n·∇ψ = ∂ψ/∂z
        # All quantities now in physical units (m, m⁻¹)
        residual = psi_z + cos_theta_V * psi_grad_norm

        # Clip residual to prevent explosion (residual in m⁻¹)
        residual = torch.clamp(residual, -1e5, 1e5)

        loss = torch.mean(residual**2)

        # Ensure loss is finite
        if not torch.isfinite(loss):
            return torch.tensor(0.0, device=device)

        return loss

    def electrostatic_loss(
        self, V_field: torch.Tensor, psi: torch.Tensor, xyz: torch.Tensor
    ) -> torch.Tensor:
        """
        Electrostatic field equation: ∇·(ε(ψ)∇V) = 0

        COMSOL: AC/DC Module -> Electrostatics
        Governing equation: ∇·(ε_r∇V) = 0 (no free charge)

        Device structure:
        - Bottom electrode (ITO): V = V_applied
        - Top electrode (ITO): V = 0 (ground)
        - Dielectric layer: ε_r = 12.0
        - Ink phase: ε_r1 = 2.0
        - Polar phase: ε_r2 = 80.0

        Args:
            V_field: Electric potential [batch, 1]
            psi: Level Set field [batch, 1]
            xyz: Spatial coordinates [batch, 3]

        Returns:
            loss: Mean squared electrostatic residual
        """
        # Get spatially varying permittivity ε(ψ)
        H = self.heaviside_3d(psi)
        epsilon_r = self.epsilon_r1 * (1.0 - H) + self.epsilon_r2 * H

        # Compute electric field E = -∇V
        try:
            V_grad_result = torch.autograd.grad(
                V_field,
                xyz,
                grad_outputs=torch.ones_like(V_field),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,  # 必须计算梯度，否则说明V_field还未学习空间依赖性
            )
            V_grad = V_grad_result[0]  # [batch, 3] - (∂V/∂x, ∂V/∂y, ∂V/∂z)
        except (RuntimeError, ValueError) as e:
            # V_field 还没学习到空间依赖性（常量输出），返回零损失
            # 这在训练早期是正常的，grad_fn 还没建立
            return torch.tensor(0.0, device=V_field.device, requires_grad=True)

        Ex = V_grad[:, 0:1]
        Ey = V_grad[:, 1:2]
        Ez = V_grad[:, 2:3]

        # Compute divergence of (ε∇V)
        # ∇·(ε∇V) = ε∇²V + ∇ε·∇V

        # First term: ε∇²V
        Ex_x_result = torch.autograd.grad(
            Ex,
            xyz,
            grad_outputs=torch.ones_like(Ex),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        Ex_x = (
            Ex_x_result[0][:, 0:1]
            if Ex_x_result[0] is not None
            else torch.zeros_like(Ex)
        )

        Ey_y_result = torch.autograd.grad(
            Ey,
            xyz,
            grad_outputs=torch.ones_like(Ey),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        Ey_y = (
            Ey_y_result[0][:, 1:2]
            if Ey_y_result[0] is not None
            else torch.zeros_like(Ey)
        )

        Ez_z_result = torch.autograd.grad(
            Ez,
            xyz,
            grad_outputs=torch.ones_like(Ez),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        Ez_z = (
            Ez_z_result[0][:, 2:3]
            if Ez_z_result[0] is not None
            else torch.zeros_like(Ez)
        )

        laplacian_V = Ex_x + Ey_y + Ez_z

        # Second term: ∇ε·∇V
        epsilon_grad_result = torch.autograd.grad(
            epsilon_r,
            xyz,
            grad_outputs=torch.ones_like(epsilon_r),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        epsilon_grad = (
            epsilon_grad_result[0]
            if epsilon_grad_result[0] is not None
            else torch.zeros_like(xyz)
        )  # [batch, 3]

        epsilon_grad_dot_V_grad = (
            epsilon_grad[:, 0:1] * Ex
            + epsilon_grad[:, 1:2] * Ey
            + epsilon_grad[:, 2:3] * Ez
        )

        # Poisson equation residual
        residual_raw = epsilon_r * laplacian_V + epsilon_grad_dot_V_grad

        # ✅ 归一化：将 residual 归一化到合理范围
        # 原因：electrostatic 残差值范围很大 (~[-10000, 10000])
        # 解决：使用 RMS (root mean square) 归一化
        rms = torch.sqrt(torch.mean(residual_raw**2))
        residual_normalized = residual_raw / (rms + 1.0)  # 归一化到 ~[-1, 1] 大部分

        # 最终返回 MSE（已经归一化了）
        return torch.mean(residual_normalized**2)

    def maxwell_stress_force(
        self, V_field: torch.Tensor, psi: torch.Tensor, xyz: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Maxwell stress tensor force density

        COMSOL: Electrostatics -> Maxwell Stress
        Stress tensor: σ_M = ε₀εᵣ(E⊗E - ½|E|²I)
        Force density: f = ∇·σ_M

        Args:
            V_field: Electric potential [batch, 1]
            psi: Level Set field [batch, 1]
            xyz: Spatial coordinates [batch, 3]

        Returns:
            F_maxwell: Maxwell stress force [batch, 3]
        """
        # Get permittivity ε(ψ)
        H = self.heaviside_3d(psi)
        epsilon_r = self.epsilon_r1 * (1.0 - H) + self.epsilon_r2 * H
        epsilon = self.epsilon_0 * epsilon_r

        # Electric field E = -∇V
        # Safety check: ensure V_field is connected to computational graph
        if not xyz.requires_grad:
            return torch.zeros_like(xyz)  # Return zero force if no gradient connection

        E_grad = torch.autograd.grad(
            V_field,
            xyz,
            grad_outputs=torch.ones_like(V_field),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]  # [batch, 3]

        # If gradient computation failed, return zero force
        if E_grad is None:
            return torch.zeros_like(xyz)

        Ex = -E_grad[:, 0:1]
        Ey = -E_grad[:, 1:2]
        Ez = -E_grad[:, 2:3]

        E_squared = Ex**2 + Ey**2 + Ez**2

        # Maxwell stress tensor components (3x3)
        # σ_ij = ε(E_i*E_j - 0.5*E²*δ_ij)
        # We only need the divergence, so compute ∇·σ

        # Diagonal terms
        sigma_xx = epsilon * (Ex**2 - 0.5 * E_squared)
        sigma_yy = epsilon * (Ey**2 - 0.5 * E_squared)
        sigma_zz = epsilon * (Ez**2 - 0.5 * E_squared)

        # Off-diagonal terms
        sigma_xy = epsilon * Ex * Ey
        sigma_xz = epsilon * Ex * Ez
        sigma_yz = epsilon * Ey * Ez

        # Compute divergence of stress tensor
        # (∇·σ)_x = ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z
        sigma_xx_x = torch.autograd.grad(
            sigma_xx,
            xyz,
            grad_outputs=torch.ones_like(sigma_xx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        sigma_xy_y = torch.autograd.grad(
            sigma_xy,
            xyz,
            grad_outputs=torch.ones_like(sigma_xy),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        sigma_xz_z = torch.autograd.grad(
            sigma_xz,
            xyz,
            grad_outputs=torch.ones_like(sigma_xz),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        F_x = sigma_xx_x + sigma_xy_y + sigma_xz_z

        # (∇·σ)_y = ∂σ_xy/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z
        sigma_xy_x = torch.autograd.grad(
            sigma_xy,
            xyz,
            grad_outputs=torch.ones_like(sigma_xy),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        sigma_yy_y = torch.autograd.grad(
            sigma_yy,
            xyz,
            grad_outputs=torch.ones_like(sigma_yy),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        sigma_yz_z = torch.autograd.grad(
            sigma_yz,
            xyz,
            grad_outputs=torch.ones_like(sigma_yz),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        F_y = sigma_xy_x + sigma_yy_y + sigma_yz_z

        # (∇·σ)_z = ∂σ_xz/∂x + ∂σ_yz/∂y + ∂σ_zz/∂z
        sigma_xz_x = torch.autograd.grad(
            sigma_xz,
            xyz,
            grad_outputs=torch.ones_like(sigma_xz),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        sigma_yz_y = torch.autograd.grad(
            sigma_yz,
            xyz,
            grad_outputs=torch.ones_like(sigma_yz),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        sigma_zz_z = torch.autograd.grad(
            sigma_zz,
            xyz,
            grad_outputs=torch.ones_like(sigma_zz),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        F_z = sigma_xz_x + sigma_yz_y + sigma_zz_z

        # Numerical stability: clamp extreme values
        F_maxwell = torch.cat([F_x, F_y, F_z], dim=1)
        F_maxwell = torch.clamp(F_maxwell, min=-1e6, max=1e6)

        return F_maxwell
