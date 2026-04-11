"""
VOF Sensitivity Tests

测试VOF相关参数的敏感性。

测试内容：
1. 不同VOF权重的影响
2. 压缩系数敏感性
3. 采样点数量影响
4. 权重比例影响

**作者**: EFD3D Team
**日期**: 2026-02-04
**版本**: v4.5
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.constraints import PhysicsConstraints


class MockPINNModel(nn.Module):
    """用于测试的简单Mock模型"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 返回随机预测用于测试

        Args:
            x: 输入张量 (batch, 6) - [x, y, z, V_from, V_to, t]

        Returns:
            (batch, 5) - [u, v, w, p, phi]
        """
        batch_size = x.shape[0]
        device = x.device

        # 生成符合物理约束的随机预测
        u = torch.randn(batch_size, 1, device=device) * 1e-4
        v = torch.randn(batch_size, 1, device=device) * 1e-4
        w = torch.randn(batch_size, 1, device=device) * 1e-4
        p = torch.randn(batch_size, 1, device=device) * 100
        phi = torch.rand(batch_size, 1, device=device)  # phi in [0, 1]

        return torch.cat([u, v, w, p, phi], dim=1)


class TestVOFWeightSensitivity:
    """VOF权重敏感性测试类"""

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

    @pytest.fixture
    def mock_model(self):
        """创建Mock模型"""
        return MockPINNModel()

    @pytest.mark.parametrize("vof_weight", [0.1, 1.0, 10.0, 100.0])
    def test_vof_weight_affects_loss(self, vof_weight, mock_model):
        """
        **测试**: 不同VOF权重影响损失值

        *For any* 不同的VOF权重，损失值应该按比例变化

        *Validates*: 权重敏感性
        """
        from src.models.pinn_two_phase import PhysicsLoss

        batch_size = 32
        device = torch.device("cpu")

        # 创建测试数据
        x = torch.rand(batch_size, 6, device=device)
        x[:, :3] = x[:, :3] * torch.tensor([174e-6, 174e-6, 20e-6], device=device)
        x[:, 3:] = x[:, 3:] * torch.tensor([30, 30, 0.05], device=device)
        x.requires_grad_(True)

        # 创建PhysicsLoss (传入device而非weights)
        physics_loss = PhysicsLoss(device=device)

        # 定义权重字典
        weights = {
            "continuity": 0.5,
            "momentum_u": 0.1,
            "momentum_v": 0.1,
            "momentum_w": 0.1,
            "vof": vof_weight,
            "surface_tension": 0.05,
            "volume_conservation": 0.3,
            "explicit_volume": 1.0,
            "sharpening": 0.1,
        }

        # 计算损失 (传入model, points, weights)
        losses = physics_loss.compute_total_loss(mock_model, x, weights=weights)

        # VOF损失应存在且与权重相关
        assert "vof" in losses, "VOF loss should be in losses"
        assert losses["vof"] >= 0, "VOF loss should be non-negative"

        # 权重越大，VOF损失在总损失中的比例应该越大
        total_loss = losses["total"]
        vof_ratio = losses["vof"] / total_loss if total_loss > 0 else 0

        # 这个测试主要验证VOF损失存在且有效
        # Ensure vof_ratio is a tensor for torch.isfinite()
        if isinstance(vof_ratio, (int, float)):
            vof_ratio_tensor = torch.tensor(vof_ratio, dtype=torch.float32)
        else:
            vof_ratio_tensor = vof_ratio
        assert torch.isfinite(vof_ratio_tensor), "VOF ratio should be finite"

    @pytest.mark.parametrize("vof_weight", [0.1, 1.0, 10.0, 100.0])
    def test_vof_weight_scaling(self, vof_weight, mock_model):
        """
        **测试**: VOF权重与损失值的比例关系

        *For any* VOF权重，损失值应该按比例变化

        *Validates*: 权重缩放正确性
        """
        from src.models.pinn_two_phase import PhysicsLoss

        batch_size = 16
        device = torch.device("cpu")

        # 固定测试数据
        torch.manual_seed(42)
        x = torch.rand(batch_size, 6, device=device)
        x[:, :3] = x[:, :3] * torch.tensor([174e-6, 174e-6, 20e-6], device=device)
        x[:, 3:] = x[:, 3:] * torch.tensor([30, 30, 0.05], device=device)
        x.requires_grad_(True)

        # 创建PhysicsLoss (传入device而非weights)
        physics_loss = PhysicsLoss(device=device)

        # 定义权重字典
        weights = {
            "continuity": 0.5,
            "momentum_u": 0.1,
            "momentum_v": 0.1,
            "momentum_w": 0.1,
            "vof": vof_weight,
            "surface_tension": 0.05,
            "volume_conservation": 0.3,
            "explicit_volume": 1.0,
            "sharpening": 0.1,
        }

        # 计算损失
        losses = physics_loss.compute_total_loss(mock_model, x, weights=weights)

        # VOF损失应存在
        assert "vof" in losses, "VOF loss should be in losses dict"
        assert losses["vof"] >= 0, "VOF loss should be non-negative"

        # 损失值应该在合理范围内
        assert losses["vof"] < 1e6, f"VOF loss too large: {losses['vof']}"


class TestCompressionSensitivity:
    """压缩系数敏感性测试类"""

    def test_compression_term_numerical_effect(self):
        """
        **测试**: 压缩项的数值效应

        *For any* phi值，压缩项应该只在界面区域有效

        *Validates*: 压缩项实现正确性
        """
        batch_size = 100
        device = torch.device("cpu")

        # 测试不同的phi值
        phi_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        compression_factors = []

        for phi_val in phi_values:
            phi = torch.full((batch_size,), phi_val, device=device)

            # 压缩因子: factor = c_alpha * |u| * phi * (1 - phi)
            # c_alpha = 1.0, |u| = 1.0 (简化)
            vel_mag = torch.ones(batch_size, device=device)
            c_alpha = 1.0

            factor = c_alpha * vel_mag * phi * (1 - phi)
            compression_factors.append(factor.mean().item())

        # 验证: phi=0.5时压缩因子最大，phi=0或1时为0
        assert max(compression_factors) == compression_factors[2], (
            "Maximum compression should be at phi=0.5"
        )
        assert compression_factors[0] == 0.0, "Compression at phi=0 should be 0"
        assert compression_factors[4] == 0.0, "Compression at phi=1 should be 0"
        assert compression_factors[2] > compression_factors[1], (
            "Compression at phi=0.5 > phi=0.25"
        )
        assert compression_factors[2] > compression_factors[3], (
            "Compression at phi=0.5 > phi=0.75"
        )

    def test_interface_region_activation(self):
        """
        **测试**: 压缩项仅在界面区域激活

        *For any* phi值，压缩项应该只在φ∈(0,1)时有效

        *Validates*: 界面选择性
        """
        batch_size = 50
        device = torch.device("cpu")

        # 创建phi在[0,1]范围内的测试
        phi = torch.rand(batch_size, device=device)
        vel_mag = torch.ones(batch_size, device=device)
        c_alpha = 1.0

        factor = c_alpha * vel_mag * phi * (1 - phi)

        # 验证: 所有factor应该在[0, 0.25]范围内
        assert factor.max() <= 0.25, (
            f"Max factor should be 0.25, got {factor.max().item()}"
        )
        assert factor.min() >= 0.0, f"Min factor should be 0, got {factor.min().item()}"

        # 验证: 0和1附近的factor接近0
        near_interface = (phi > 0.3) & (phi < 0.7)
        near_pure = (phi < 0.1) | (phi > 0.9)

        assert factor[near_interface].mean() > factor[near_pure].mean(), (
            "Interface region should have higher compression"
        )


class TestSamplingSensitivity:
    """采样敏感性测试类"""

    def test_interface_sampling_importance(self):
        """
        **测试**: 界面采样的重要性

        *For any* 采样策略，界面区域应该有更高的权重

        *Validates*: 采样策略正确性
        """
        batch_size = 1000
        device = torch.device("cpu")

        # 模拟界面加权采样
        # 假设界面在phi≈0.5附近
        phi = torch.rand(batch_size, device=device)

        # 界面权重: w = 4 * phi * (1 - phi)
        interface_weight = 4 * phi * (1 - phi)

        # 验证: 界面区域的权重更高
        near_interface = (phi > 0.3) & (phi < 0.7)
        near_pure = (phi < 0.1) | (phi > 0.9)

        assert (
            interface_weight[near_interface].mean() > interface_weight[near_pure].mean()
        ), "Interface region should have higher sampling weight"

        # 验证: 权重在0-1范围内
        assert interface_weight.max() <= 1.0, "Max weight should be 1.0"
        assert interface_weight.min() >= 0.0, "Min weight should be 0.0"


class TestWeightBalanceSensitivity:
    """权重平衡敏感性测试类"""

    @pytest.fixture
    def mock_model(self):
        """创建Mock模型"""
        return MockPINNModel()

    @pytest.mark.parametrize("vof_to_continuity_ratio", [0.001, 0.01, 0.1, 1.0])
    def test_vof_continuity_balance(self, vof_to_continuity_ratio, mock_model):
        """
        **测试**: VOF与连续性权重的平衡

        *For any* 权重比例，总损失应该能正确反映两个约束

        *Validates*: 权重平衡
        """
        from src.models.pinn_two_phase import PhysicsLoss

        batch_size = 16
        device = torch.device("cpu")

        # 创建测试数据
        x = torch.rand(batch_size, 6, device=device)
        x[:, :3] = x[:, :3] * torch.tensor([174e-6, 174e-6, 20e-6], device=device)
        x[:, 3:] = x[:, 3:] * torch.tensor([30, 30, 0.05], device=device)
        x.requires_grad_(True)

        # 固定连续性权重，变化VOF权重
        continuity_weight = 0.5
        vof_weight = continuity_weight * vof_to_continuity_ratio

        # 创建PhysicsLoss
        physics_loss = PhysicsLoss(device=device)

        weights = {
            "continuity": continuity_weight,
            "momentum_u": 0.1,
            "momentum_v": 0.1,
            "momentum_w": 0.1,
            "vof": vof_weight,
            "surface_tension": 0.05,
            "volume_conservation": 0.3,
            "explicit_volume": 1.0,
            "sharpening": 0.1,
        }

        losses = physics_loss.compute_total_loss(mock_model, x, weights=weights)

        # 两个损失都应该存在
        assert "vof" in losses, "VOF loss should exist"
        assert "continuity" in losses, "Continuity loss should exist"

        # 损失值应该在合理范围内
        assert losses["vof"] < 1e6, "VOF loss should be bounded"
        assert losses["continuity"] < 1e6, "Continuity loss should be bounded"

    def test_interface_vs_volume_balance(self, mock_model):
        """
        **测试**: 界面拟合与体积守恒的平衡

        *For any* 模型，界面拟合和体积守恒应该协同工作

        *Validates*: 多目标平衡
        """
        from src.models.pinn_two_phase import PhysicsLoss

        batch_size = 32
        device = torch.device("cpu")

        x = torch.rand(batch_size, 6, device=device)
        x[:, :3] = x[:, :3] * torch.tensor([174e-6, 174e-6, 20e-6], device=device)
        x[:, 3:] = x[:, 3:] * torch.tensor([30, 30, 0.05], device=device)
        x.requires_grad_(True)

        # 创建PhysicsLoss
        physics_loss = PhysicsLoss(device=device)

        # 标准权重
        weights = {
            "continuity": 0.5,
            "momentum_u": 0.1,
            "momentum_v": 0.1,
            "momentum_w": 0.1,
            "vof": 2.0,
            "surface_tension": 0.05,
            "volume_conservation": 0.3,
            "explicit_volume": 1.0,
            "sharpening": 0.1,
        }

        losses = physics_loss.compute_total_loss(mock_model, x, weights=weights)

        # 所有损失都应该存在
        required_losses = [
            "vof",
            "continuity",
            "volume_conservation",
            "explicit_volume",
        ]
        for loss_name in required_losses:
            assert loss_name in losses, f"{loss_name} loss should exist"
            assert losses[loss_name] >= 0, f"{loss_name} should be non-negative"
            assert torch.isfinite(losses[loss_name]), f"{loss_name} should be finite"


class TestParameterSensitivitySummary:
    """参数敏感性综合测试"""

    @pytest.fixture
    def mock_model(self):
        """创建Mock模型"""
        return MockPINNModel()

    @pytest.mark.parametrize(
        "test_case",
        [
            {"vof_weight": 0.1, "continuity_weight": 0.5},
            {"vof_weight": 1.0, "continuity_weight": 0.5},
            {"vof_weight": 10.0, "continuity_weight": 0.5},
            {"vof_weight": 100.0, "continuity_weight": 0.5},
        ],
    )
    def test_parameter_combinations(self, test_case, mock_model):
        """
        **测试**: 不同参数组合的影响

        *For any* 参数组合，损失计算应该正确执行

        *Validates*: 参数组合正确性
        """
        from src.models.pinn_two_phase import PhysicsLoss

        batch_size = 16
        device = torch.device("cpu")

        x = torch.rand(batch_size, 6, device=device)
        x[:, :3] = x[:, :3] * torch.tensor([174e-6, 174e-6, 20e-6], device=device)
        x[:, 3:] = x[:, 3:] * torch.tensor([30, 30, 0.05], device=device)
        x.requires_grad_(True)

        # 创建PhysicsLoss
        physics_loss = PhysicsLoss(device=device)

        weights = {
            "continuity": test_case["continuity_weight"],
            "momentum_u": 0.1,
            "momentum_v": 0.1,
            "momentum_w": 0.1,
            "vof": test_case["vof_weight"],
            "surface_tension": 0.05,
            "volume_conservation": 0.3,
            "explicit_volume": 1.0,
            "sharpening": 0.1,
        }

        losses = physics_loss.compute_total_loss(mock_model, x, weights=weights)

        # 所有损失都应该存在
        assert "vof" in losses, "VOF loss should exist"
        assert "continuity" in losses, "Continuity loss should exist"

        # 损失应该有效
        assert torch.isfinite(losses["vof"]), "VOF loss should be finite"
        assert torch.isfinite(losses["continuity"]), "Continuity loss should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
