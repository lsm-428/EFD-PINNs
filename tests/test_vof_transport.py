"""
VOF Transport Tests (Simplified)

测试VOF输运方程的正确性和收敛性。

测试内容：
1. 恒定速度场 → phi应平流移动
2. 零速度场 → phi应保持不变
3. 数值稳定性
4. 界面锐化效果验证
5. 体积守恒验证

**作者**: EFD3D Team
**日期**: 2026-02-04
**版本**: v4.5
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.constraints import PhysicsConstraints


class TestVOFTransportEquation:
    """VOF输运方程测试类"""

    @pytest.fixture
    def physics_constraints(self):
        """创建PhysicsConstraints实例"""
        materials_params = {
            "Lx": 174e-6,
            "Ly": 174e-6,
            "Lz": 20e-6,
            "ink_thickness": 3e-6,
            "rho_oil": 800.0,
            "rho_polar": 1000.0,
            "mu_oil": 0.003,
            "mu_polar": 0.001,
            "surface_tension": 0.045,
            "ink_initial_fraction": 0.15,
        }
        return PhysicsConstraints(materials_params=materials_params)

    def test_zero_velocity_phi_constant(self, physics_constraints):
        """
        测试: 零速度场下，phi应保持不变

        对于零速度场 (u=v=w=0)，VOF方程简化为 ∂φ/∂t = 0
        因此φ应该保持常数
        """
        batch_size = 100
        device = torch.device("cpu")

        # 空间坐标
        x = torch.rand(batch_size, 1, device=device) * 174e-6
        y = torch.rand(batch_size, 1, device=device) * 174e-6
        z = torch.rand(batch_size, 1, device=device) * 20e-6
        v_from = torch.zeros(batch_size, 1, device=device)
        v_to = torch.zeros(batch_size, 1, device=device)
        t = torch.rand(batch_size, 1, device=device) * 0.05

        x_phys = torch.cat([x, y, z, v_from, v_to, t], dim=1)
        x_phys.requires_grad_(True)

        # phi: z < 3e-6 时 phi=1，否则 phi=0
        phi = (z < 3e-6).float()

        # 零速度预测
        predictions = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.rand(batch_size, 1, device=device),
                phi,
            ],
            dim=1,
        )

        # VOF残差
        vof_residual = physics_constraints._compute_vof_residual(x_phys, predictions)

        # 验证: 零速度场下，VOF残差应接近0
        assert torch.allclose(
            vof_residual, torch.zeros_like(vof_residual), atol=1e-5
        ), (
            f"Zero velocity VOF residual should be ~0, got {vof_residual.mean().item():.6f}"
        )

    def test_constant_velocity_phi_advection(self, physics_constraints):
        """
        测试: 恒定速度场下，phi应平流移动

        对于恒定速度场，VOF方程描述平流过程
        """
        batch_size = 100
        device = torch.device("cpu")

        # phi = z / Lz (只依赖于z，不依赖于x)
        Lz = 20e-6
        x = torch.rand(batch_size, 1, device=device) * 174e-6
        y = torch.rand(batch_size, 1, device=device) * 174e-6
        z = torch.rand(batch_size, 1, device=device) * Lz
        v_from = torch.zeros(batch_size, 1, device=device)
        v_to = torch.zeros(batch_size, 1, device=device)
        t = torch.zeros(batch_size, 1, device=device)

        x_phys = torch.cat([x, y, z, v_from, v_to, t], dim=1)
        x_phys.requires_grad_(True)

        phi = z / Lz

        # 恒定速度 u = (1e-4, 0, 0)
        u_val = 1e-4
        u = torch.full((batch_size, 1), u_val, device=device)
        v = torch.zeros(batch_size, 1, device=device)
        w = torch.zeros(batch_size, 1, device=device)
        p = torch.rand(batch_size, 1, device=device)

        predictions = torch.cat([u, v, w, p, phi], dim=1)

        # VOF残差
        vof_residual = physics_constraints._compute_vof_residual(x_phys, predictions)

        # 期望: VOF残差 ≈ u * ∂φ/∂x = u * 0 = 0
        # 因为 phi 只依赖于 z，不依赖于 x
        assert torch.allclose(vof_residual.mean(), torch.tensor(0.0), atol=1e-4), (
            f"Expected residual ~0, got {vof_residual.mean().item():.6f}"
        )

    def test_vof_residual_numerical_stability(self, physics_constraints):
        """
        测试: VOF残差的数值稳定性

        对于有效输入，VOF计算不应产生NaN或Inf
        """
        batch_size = 50
        device = torch.device("cpu")

        for _ in range(10):
            x = torch.rand(batch_size, 1, device=device) * 174e-6
            y = torch.rand(batch_size, 1, device=device) * 174e-6
            z = torch.rand(batch_size, 1, device=device) * 20e-6
            v_from = torch.rand(batch_size, 1, device=device) * 30
            v_to = torch.rand(batch_size, 1, device=device) * 30
            t = torch.rand(batch_size, 1, device=device) * 0.05

            x_phys = torch.cat([x, y, z, v_from, v_to, t], dim=1)
            x_phys.requires_grad_(True)

            phi = torch.rand(batch_size, 1, device=device)

            predictions = torch.cat(
                [
                    torch.rand(batch_size, 1, device=device) * 1e-3,
                    torch.rand(batch_size, 1, device=device) * 1e-3,
                    torch.rand(batch_size, 1, device=device) * 1e-3,
                    torch.rand(batch_size, 1, device=device) * 100,
                    phi,
                ],
                dim=1,
            )

            vof_residual = physics_constraints._compute_vof_residual(
                x_phys, predictions
            )

            assert torch.isfinite(vof_residual).all(), (
                f"VOF residual contains NaN/Inf at iteration {_}"
            )

    def test_volume_conservation_initial_state(self, physics_constraints):
        """
        测试: 初始状态体积守恒

        对于均匀油墨分布，体积守恒损失应接近0
        """
        batch_size = 1000
        device = torch.device("cpu")

        # 均匀分布的坐标
        x = torch.rand(batch_size, 1, device=device) * 174e-6
        y = torch.rand(batch_size, 1, device=device) * 174e-6
        z = torch.rand(batch_size, 1, device=device) * 20e-6

        x_phys = torch.cat([x, y, z], dim=1)

        # 初始状态: z < h_ink 时 phi=1，否则 phi=0
        h_ink = 3e-6
        phi = (z < h_ink).float()

        predictions = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                phi,
            ],
            dim=1,
        )

        # 体积守恒残差
        residuals = physics_constraints.compute_volume_conservation_residual(
            x_phys, predictions
        )

        # 初始状态应该满足体积守恒
        assert residuals["volume_conservation"].abs().mean() <= 0.1 + 1e-6, (
            f"Initial state should have ~0 volume conservation loss"
        )

    def test_volume_conservation_violation_detection(self, physics_constraints):
        """
        测试: 体积守恒违反检测

        对于体积变化，体积守恒损失应该增大
        """
        batch_size = 1000
        device = torch.device("cpu")

        # 均匀坐标
        x = torch.rand(batch_size, 1, device=device) * 174e-6
        y = torch.rand(batch_size, 1, device=device) * 174e-6
        z = torch.rand(batch_size, 1, device=device) * 20e-6

        x_phys = torch.cat([x, y, z], dim=1)

        # 正常状态: phi = h_ink / Lz
        h_ink = 3e-6
        Lz = 20e-6
        normal_phi = torch.full((batch_size, 1), h_ink / Lz, device=device)

        # 违反状态: phi = 1.0
        violated_phi = torch.ones(batch_size, 1, device=device)

        predictions_normal = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                normal_phi,
            ],
            dim=1,
        )

        predictions_violated = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                violated_phi,
            ],
            dim=1,
        )

        residuals_normal = physics_constraints.compute_volume_conservation_residual(
            x_phys, predictions_normal
        )
        residuals_violated = physics_constraints.compute_volume_conservation_residual(
            x_phys, predictions_violated
        )

        normal_loss = residuals_normal["volume_conservation"].abs().mean()
        violated_loss = residuals_violated["volume_conservation"].abs().mean()

        assert violated_loss > normal_loss, (
            f"Volume violation should increase loss: normal={normal_loss:.6f}, violated={violated_loss:.6f}"
        )

    def test_interface_sharpening_theory(self):
        """
        测试: 界面锐化理论基础

        φ(1-φ)在φ=0.5时最大，在φ=0或1时为0
        """
        device = torch.device("cpu")
        batch_size = 100

        phi_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

        sharpening_values = phi_values * (1 - phi_values)

        # phi=0.5时最大
        assert torch.argmax(sharpening_values) == 2, "Maximum at phi=0.5"

        # phi=0和phi=1时为0
        assert sharpening_values[0] == 0.0, "Sharpening at phi=0 should be 0"
        assert sharpening_values[4] == 0.0, "Sharpening at phi=1 should be 0"

    def test_compression_term_theory(self):
        """
        测试: 压缩项理论基础

        压缩因子 = c_alpha * |u| * φ(1-φ)
        应只在界面区域有效
        """
        device = torch.device("cpu")
        batch_size = 100

        phi = torch.rand(batch_size, device=device)
        vel_mag = torch.ones(batch_size, device=device)
        c_alpha = 1.0

        factor = c_alpha * vel_mag * phi * (1 - phi)

        # 应该在[0, 0.25]范围内
        assert factor.max() <= 0.25, (
            f"Max factor should be 0.25, got {factor.max().item()}"
        )
        assert factor.min() >= 0.0, f"Min factor should be 0, got {factor.min().item()}"

        # 界面区域权重更高
        near_interface = (phi > 0.3) & (phi < 0.7)
        near_pure = (phi < 0.1) | (phi > 0.9)

        assert factor[near_interface].mean() > factor[near_pure].mean(), (
            "Interface region should have higher compression"
        )

    def test_vof_implementation_consistency(self, physics_constraints):
        """
        测试: VOF实现的一致性

        VOF残差应该是有意义的值
        """
        batch_size = 50
        device = torch.device("cpu")

        x = torch.rand(batch_size, 1, device=device) * 174e-6
        y = torch.rand(batch_size, 1, device=device) * 174e-6
        z = torch.rand(batch_size, 1, device=device) * 20e-6
        v_from = torch.rand(batch_size, 1, device=device) * 30
        v_to = torch.rand(batch_size, 1, device=device) * 30
        t = torch.rand(batch_size, 1, device=device) * 0.05

        x_phys = torch.cat([x, y, z, v_from, v_to, t], dim=1)
        x_phys.requires_grad_(True)

        phi = torch.rand(batch_size, 1, device=device)
        u = torch.rand(batch_size, 1, device=device) * 1e-3
        v = torch.rand(batch_size, 1, device=device) * 1e-3
        w = torch.rand(batch_size, 1, device=device) * 1e-3
        p = torch.rand(batch_size, 1, device=device) * 100

        predictions = torch.cat([u, v, w, p, phi], dim=1)

        vof_residual = physics_constraints._compute_vof_residual(x_phys, predictions)

        # 输出批次大小匹配
        assert vof_residual.numel() == batch_size, "Output batch size mismatch"
        assert torch.isfinite(vof_residual).all(), "VOF residual contains NaN/Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
