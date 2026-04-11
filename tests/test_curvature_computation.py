"""
曲率计算单元测试

测试 VOF (Volume of Fluid) 界面曲率计算的正确性。
使用球面解析解验证数值计算的精度。

用途：
- 验证 PINN 模型中表面张力物理约束的实现
- 确保 VOF 界面表示的数值精度
- 调试两相流模型中的毛细现象

数学原理：
- Level set 方法计算平均曲率：κ = -∇²φ / |∇φ|³
- 球面解析解：κ = 2/R
"""

import torch
import pytest


def _compute_curvature_logic(phi_grads):
    """
    计算界面曲率（核心逻辑）

    Args:
        phi_grads: VOF 场的梯度字典，包含：
            - 一阶导数: phi_x, phi_y, phi_z
            - 二阶导数: phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz

    Returns:
        kappa: 界面平均曲率
    """
    phi_x = phi_grads["phi_x"]
    phi_y = phi_grads["phi_y"]
    phi_z = phi_grads["phi_z"]
    phi_xx = phi_grads["phi_xx"]
    phi_yy = phi_grads["phi_yy"]
    phi_zz = phi_grads["phi_zz"]
    phi_xy = phi_grads["phi_xy"]
    phi_xz = phi_grads["phi_xz"]
    phi_yz = phi_grads["phi_yz"]

    grad_mag_sq = phi_x**2 + phi_y**2 + phi_z**2 + 1e-10
    grad_mag = torch.sqrt(grad_mag_sq)

    numerator = (
        phi_xx * (phi_y**2 + phi_z**2) +
        phi_yy * (phi_x**2 + phi_z**2) +
        phi_zz * (phi_x**2 + phi_y**2) -
        2 * (phi_x * phi_y * phi_xy + phi_x * phi_z * phi_xz + phi_y * phi_z * phi_yz)
    )
    kappa = -numerator / (grad_mag_sq * grad_mag + 1e-10)
    return kappa


class TestCurvatureComputation:
    """曲率计算测试套件"""

    @pytest.mark.parametrize("R,epsilon,rtol", [
        (0.5, 0.05, 0.15),   # 标准球面
        (1.0, 0.05, 0.15),   # 大半径
        (0.25, 0.03, 0.20),  # 小半径
    ])
    def test_sphere_curvature(self, R, epsilon, rtol):
        """
        测试球面曲率计算

        球面解析解：κ = 2/R
        验证数值计算与解析解的相对误差在允许范围内

        Args:
            R: 球面半径
            epsilon: VOF 界面厚度参数
            rtol: 允许的相对误差
        """
        # 在球面上取点 (R, 0, 0)
        x = torch.tensor([[R, 0.0, 0.0]], requires_grad=True)

        # VOF 场：phi(r) = 0.5 * (1 - tanh((r - R) / epsilon))
        # phi=1 inside (r < R), phi=0 outside (r > R)
        r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
        phi = 0.5 * (1 - torch.tanh((r - R) / epsilon))

        # 一阶导数
        grad_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        phi_x, phi_y, phi_z = grad_phi[:, 0], grad_phi[:, 1], grad_phi[:, 2]

        # 二阶导数
        phi_xx = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0][:, 0]
        phi_yy = torch.autograd.grad(phi_y.sum(), x, create_graph=True)[0][:, 1]
        phi_zz = torch.autograd.grad(phi_z.sum(), x, create_graph=True)[0][:, 2]
        phi_xy = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0][:, 1]
        phi_xz = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0][:, 2]
        phi_yz = torch.autograd.grad(phi_y.sum(), x, create_graph=True)[0][:, 2]

        phi_grads = {
            "phi_x": phi_x, "phi_y": phi_y, "phi_z": phi_z,
            "phi_xx": phi_xx, "phi_yy": phi_yy, "phi_zz": phi_zz,
            "phi_xy": phi_xy, "phi_xz": phi_xz, "phi_yz": phi_yz
        }

        kappa = _compute_curvature_logic(phi_grads)

        # 解析解：3D 球面曲率 κ = 2/R
        expected_kappa = 2.0 / R

        # 验证相对误差
        relative_error = abs(kappa.item() - expected_kappa) / expected_kappa

        assert relative_error < rtol, (
            f"Sphere curvature error too large: "
            f"calculated={kappa.item():.4f}, "
            f"expected={expected_kappa:.4f}, "
            f"relative_error={relative_error:.2%}, "
            f"tolerance={rtol:.2%}"
        )

    def test_curvature_magnitude(self):
        """测试曲率大小的物理合理性"""
        R = 0.5
        epsilon = 0.05

        x = torch.tensor([[R, 0.0, 0.0]], requires_grad=True)
        r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
        phi = 0.5 * (1 - torch.tanh((r - R) / epsilon))

        grad_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        phi_x, phi_y, phi_z = grad_phi[:, 0], grad_phi[:, 1], grad_phi[:, 2]

        phi_xx = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0][:, 0]
        phi_yy = torch.autograd.grad(phi_y.sum(), x, create_graph=True)[0][:, 1]
        phi_zz = torch.autograd.grad(phi_z.sum(), x, create_graph=True)[0][:, 2]
        phi_xy = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0][:, 1]
        phi_xz = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0][:, 2]
        phi_yz = torch.autograd.grad(phi_y.sum(), x, create_graph=True)[0][:, 2]

        phi_grads = {
            "phi_x": phi_x, "phi_y": phi_y, "phi_z": phi_z,
            "phi_xx": phi_xx, "phi_yy": phi_yy, "phi_zz": phi_zz,
            "phi_xy": phi_xy, "phi_xz": phi_xz, "phi_yz": phi_yz
        }

        kappa = _compute_curvature_logic(phi_grads)

        # 曲率大小应在合理范围内（不能为零或无穷大）
        assert 0.1 < abs(kappa.item()) < 100.0, (
            f"Curvature magnitude {abs(kappa.item()):.4f} is not physically reasonable"
        )


# 保留独立运行功能（用于手动测试）
if __name__ == "__main__":
    print("Running curvature computation tests...")
    pytest.main([__file__, "-v"])
